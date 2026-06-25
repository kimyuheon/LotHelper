use std::net::TcpStream;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::sync::Mutex;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::Manager;
use tauri_plugin_dialog::DialogExt;

const LLAMA_HOST: &str = "127.0.0.1";
const LLAMA_PORT: u16 = 8080;

const BASE_SYSTEM: &str = "You are CppAI, a helpful coding assistant powered by \
    Qwen2.5-Coder. Answer concisely and use Markdown code blocks for code.";

const EDIT_SYSTEM: &str = "When you want to create or modify a file, output its FULL \
    new content in a fenced code block whose info string is `file:<relative/path>` \
    (for example ```file:src/main.py). Use one block per file and keep explanations brief.";

const AGENT_SYSTEM: &str = "You are CppAI, an autonomous coding agent working inside the \
    user's project folder. Use the provided tools to inspect and modify files \
    (list_files, read_file, write_file, make_dir) and to run shell commands \
    (run_command) such as build, compile, or test commands. Always use workspace-relative \
    paths (never absolute, never '..'). Read before you overwrite.\n\
    IMPORTANT WORKFLOW: after writing or changing code, BUILD or COMPILE the project with \
    run_command (e.g. `cargo build`, `npm run build`, `python -m py_compile <file>`, \
    `tsc --noEmit`). If the build fails, read the error output, FIX the code, and build \
    again. Repeat until the build succeeds (exit code 0). Only when the build passes, \
    stop calling tools and give a short summary of what you did. Use non-interactive \
    commands only — never start long-running servers or watchers.";

fn chat_url() -> String {
    format!("http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions")
}

/// Auto-started llama-server child process (None if reused/failed).
struct LlamaServer(Mutex<Option<Child>>);

/// Currently selected project folder.
struct Workspace(Mutex<Option<PathBuf>>);

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct ChatRequest {
    messages: Vec<ChatMessage>,
    stream: bool,
    temperature: f32,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
}

#[derive(Deserialize)]
struct Choice {
    message: ChatMessage,
}

#[derive(Serialize)]
struct AgentResult {
    reply: String,
    actions: Vec<String>,
    changed: Vec<String>,
}

// ---- llama-server lifecycle ------------------------------------------------

fn resource_base(app: &tauri::App) -> PathBuf {
    if let Ok(dir) = app.path().resource_dir() {
        if dir.join("llama").exists() || dir.join("models").exists() {
            return dir;
        }
    }
    if let Ok(exe) = std::env::current_exe() {
        let mut dir = exe.parent().map(Path::to_path_buf);
        while let Some(d) = dir {
            if d.join("llama").exists() || d.join("models").exists() {
                return d;
            }
            dir = d.parent().map(Path::to_path_buf);
        }
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn llama_binary(base: &Path) -> PathBuf {
    let (sub, exe) = if cfg!(target_os = "windows") {
        ("windows", "llama-server.exe")
    } else if cfg!(target_os = "macos") {
        ("macos", "llama-server")
    } else {
        ("linux", "llama-server")
    };
    base.join("llama").join(sub).join(exe)
}

fn find_model(base: &Path) -> Option<PathBuf> {
    std::fs::read_dir(base.join("models"))
        .ok()?
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("gguf")))
}

fn server_running() -> bool {
    let addr = format!("{LLAMA_HOST}:{LLAMA_PORT}");
    match addr.parse() {
        Ok(sa) => TcpStream::connect_timeout(&sa, Duration::from_millis(300)).is_ok(),
        Err(_) => false,
    }
}

fn spawn_llama(base: &Path) -> std::io::Result<Child> {
    let bin = llama_binary(base);
    if !bin.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("llama-server를 찾을 수 없습니다: {}", bin.display()),
        ));
    }
    let model = find_model(base).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "models/ 폴더에 .gguf 모델이 없습니다.",
        )
    })?;

    let mut cmd = Command::new(&bin);
    cmd.args([
        "-m",
        &model.to_string_lossy(),
        "--host",
        LLAMA_HOST,
        "--port",
        &LLAMA_PORT.to_string(),
        "-ngl",
        "99",
        "-c",
        "8192",
    ]);

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    cmd.spawn()
}

// ---- workspace + file tools ------------------------------------------------

fn ws_root(state: &tauri::State<'_, Workspace>) -> Result<PathBuf, String> {
    state
        .0
        .lock()
        .unwrap()
        .clone()
        .ok_or_else(|| "작업 폴더가 선택되지 않았습니다.".to_string())
}

/// Resolve a workspace-relative path, rejecting anything that escapes the root.
fn resolve(root: &Path, rel: &str) -> Result<PathBuf, String> {
    let mut out = root.to_path_buf();
    for comp in Path::new(rel).components() {
        match comp {
            Component::Normal(c) => out.push(c),
            Component::CurDir => {}
            Component::ParentDir => return Err("'..' 경로는 허용되지 않습니다.".to_string()),
            _ => return Err("절대 경로는 허용되지 않습니다.".to_string()),
        }
    }
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) {
    const SKIP: [&str; 6] = ["node_modules", "target", ".git", "dist", "build", ".venv"];
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries.flatten() {
        if out.len() >= 2000 {
            return;
        }
        let name = e.file_name().to_string_lossy().to_string();
        if SKIP.contains(&name.as_str()) {
            continue;
        }
        let p = e.path();
        if p.is_dir() {
            walk(root, &p, out);
        } else if let Ok(rel) = p.strip_prefix(root) {
            out.push(rel.to_string_lossy().replace('\\', "/"));
        }
    }
}

#[tauri::command]
async fn select_workspace(
    app: tauri::AppHandle,
    state: tauri::State<'_, Workspace>,
) -> Result<Option<String>, String> {
    match app.dialog().file().blocking_pick_folder() {
        Some(fp) => {
            let path = fp.into_path().map_err(|e| e.to_string())?;
            *state.0.lock().unwrap() = Some(path.clone());
            Ok(Some(path.to_string_lossy().to_string()))
        }
        None => Ok(None),
    }
}

#[tauri::command]
fn current_workspace(state: tauri::State<'_, Workspace>) -> Option<String> {
    state
        .0
        .lock()
        .unwrap()
        .as_ref()
        .map(|p| p.to_string_lossy().to_string())
}

#[tauri::command]
fn workspace_files(state: tauri::State<'_, Workspace>) -> Result<Vec<String>, String> {
    let root = ws_root(&state)?;
    let mut files = Vec::new();
    walk(&root, &root, &mut files);
    files.sort();
    Ok(files)
}

#[tauri::command]
fn read_file(path: String, state: tauri::State<'_, Workspace>) -> Result<String, String> {
    let root = ws_root(&state)?;
    let full = resolve(&root, &path)?;
    std::fs::read_to_string(&full).map_err(|e| e.to_string())
}

#[tauri::command]
fn write_file(
    path: String,
    content: String,
    state: tauri::State<'_, Workspace>,
) -> Result<(), String> {
    let root = ws_root(&state)?;
    let full = resolve(&root, &path)?;
    if let Some(parent) = full.parent() {
        std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    std::fs::write(&full, content).map_err(|e| e.to_string())
}

#[tauri::command]
fn open_in_vscode(
    path: Option<String>,
    state: tauri::State<'_, Workspace>,
) -> Result<(), String> {
    let root = ws_root(&state)?;
    let target = match path {
        Some(p) => resolve(&root, &p)?,
        None => root,
    };

    let spawned = if cfg!(windows) {
        Command::new("cmd")
            .args(["/C", "code"])
            .arg(&target)
            .spawn()
    } else {
        Command::new("code").arg(&target).spawn()
    };

    if spawned.is_err() {
        // Fall back to the OS file manager if VS Code's `code` CLI isn't on PATH.
        #[cfg(windows)]
        let _ = Command::new("explorer").arg(&target).spawn();
        #[cfg(target_os = "macos")]
        let _ = Command::new("open").arg(&target).spawn();
        #[cfg(all(unix, not(target_os = "macos")))]
        let _ = Command::new("xdg-open").arg(&target).spawn();
    }
    Ok(())
}

// ---- chat + agent ----------------------------------------------------------

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[tauri::command]
async fn llama_ready() -> bool {
    reqwest::Client::new()
        .get(format!("http://{LLAMA_HOST}:{LLAMA_PORT}/health"))
        .timeout(Duration::from_millis(800))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

#[tauri::command]
async fn chat(messages: Vec<ChatMessage>, edit_blocks: bool) -> Result<String, String> {
    let system = if edit_blocks {
        format!("{BASE_SYSTEM}\n{EDIT_SYSTEM}")
    } else {
        BASE_SYSTEM.to_string()
    };
    let mut full = vec![ChatMessage {
        role: "system".to_string(),
        content: system,
    }];
    full.extend(messages);

    let body = ChatRequest {
        messages: full,
        stream: false,
        temperature: 0.7,
    };

    let url = chat_url();
    let resp = reqwest::Client::new()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("llama-server에 연결할 수 없습니다 ({url}): {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("llama-server 오류 {status}: {text}"));
    }

    let parsed: ChatResponse = resp.json().await.map_err(|e| format!("응답 파싱 실패: {e}"))?;
    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "응답에 choices가 없습니다.".to_string())
}

/// Run a shell command in the workspace, capturing combined output, with a hard
/// timeout (the child is killed if it overruns).
fn run_shell(root: &Path, command: &str, timeout: Duration) -> String {
    let mut cmd = if cfg!(windows) {
        let mut c = Command::new("cmd");
        c.args(["/C", command]);
        c
    } else {
        let mut c = Command::new("sh");
        c.args(["-c", command]);
        c
    };
    cmd.current_dir(root)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }

    let child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return format!("실행 실패: {e}"),
    };
    let pid = child.id();

    // Read the process to completion on a worker thread so we can time out.
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(child.wait_with_output());
    });

    let out = match rx.recv_timeout(timeout) {
        Ok(Ok(output)) => {
            let mut s = String::from_utf8_lossy(&output.stdout).into_owned();
            s.push_str(&String::from_utf8_lossy(&output.stderr));
            let code = output.status.code().unwrap_or(-1);
            format!("exit code: {code}\n{s}")
        }
        Ok(Err(e)) => format!("실행 오류: {e}"),
        Err(_) => {
            // Timed out — kill the process tree.
            #[cfg(windows)]
            let _ = Command::new("taskkill")
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .output();
            #[cfg(unix)]
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).output();
            format!("TIMEOUT: {}초 내에 끝나지 않아 종료했습니다.", timeout.as_secs())
        }
    };

    // Keep context small: prefer the tail, where errors usually appear.
    const MAX: usize = 4000;
    if out.len() > MAX {
        let mut start = out.len() - MAX;
        while start < out.len() && !out.is_char_boundary(start) {
            start += 1;
        }
        format!("...(앞부분 생략)...\n{}", &out[start..])
    } else {
        out
    }
}

fn agent_tools() -> Value {
    json!([
        { "type": "function", "function": {
            "name": "list_files",
            "description": "List all files in the workspace (relative paths).",
            "parameters": { "type": "object", "properties": {} }
        }},
        { "type": "function", "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace.",
            "parameters": { "type": "object",
                "properties": { "path": { "type": "string", "description": "workspace-relative path" } },
                "required": ["path"] }
        }},
        { "type": "function", "function": {
            "name": "write_file",
            "description": "Create or overwrite a file with the given full content.",
            "parameters": { "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "workspace-relative path" },
                    "content": { "type": "string", "description": "the full file content" }
                },
                "required": ["path", "content"] }
        }},
        { "type": "function", "function": {
            "name": "make_dir",
            "description": "Create a directory (and parents) in the workspace.",
            "parameters": { "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"] }
        }},
        { "type": "function", "function": {
            "name": "run_command",
            "description": "Run a non-interactive shell command in the workspace (e.g. build, \
                compile, or test). Returns the exit code and combined stdout/stderr. \
                Use this to build the project and to verify fixes.",
            "parameters": { "type": "object",
                "properties": { "command": { "type": "string", "description": "the shell command" } },
                "required": ["command"] }
        }}
    ])
}

fn run_tool(
    root: &Path,
    name: &str,
    args: &Value,
    actions: &mut Vec<String>,
    changed: &mut Vec<String>,
) -> String {
    let arg = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("");
    match name {
        "list_files" => {
            let mut files = Vec::new();
            walk(root, root, &mut files);
            files.sort();
            actions.push("📂 파일 목록 조회".to_string());
            if files.is_empty() {
                "(빈 폴더)".to_string()
            } else {
                files.join("\n")
            }
        }
        "read_file" => {
            let path = arg("path");
            actions.push(format!("👀 읽기: {path}"));
            match resolve(root, path).and_then(|p| std::fs::read_to_string(p).map_err(|e| e.to_string())) {
                Ok(c) => c,
                Err(e) => format!("ERROR: {e}"),
            }
        }
        "write_file" => {
            let path = arg("path");
            match resolve(root, path) {
                Ok(full) => {
                    if let Some(parent) = full.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match std::fs::write(&full, arg("content")) {
                        Ok(()) => {
                            actions.push(format!("✏️ 쓰기: {path}"));
                            changed.push(path.to_string());
                            format!("OK: wrote {path}")
                        }
                        Err(e) => format!("ERROR: {e}"),
                    }
                }
                Err(e) => format!("ERROR: {e}"),
            }
        }
        "make_dir" => {
            let path = arg("path");
            match resolve(root, path).and_then(|p| std::fs::create_dir_all(p).map_err(|e| e.to_string())) {
                Ok(()) => {
                    actions.push(format!("📁 폴더 생성: {path}"));
                    format!("OK: created {path}")
                }
                Err(e) => format!("ERROR: {e}"),
            }
        }
        "run_command" => {
            let command = arg("command");
            actions.push(format!("⚙️ 실행: {command}"));
            run_shell(root, command, Duration::from_secs(240))
        }
        other => format!("ERROR: unknown tool {other}"),
    }
}

#[tauri::command]
async fn agent_chat(
    messages: Vec<ChatMessage>,
    state: tauri::State<'_, Workspace>,
) -> Result<AgentResult, String> {
    let root = ws_root(&state)?;
    let tools = agent_tools();
    let client = reqwest::Client::new();

    let mut convo: Vec<Value> = vec![json!({ "role": "system", "content": AGENT_SYSTEM })];
    for m in &messages {
        convo.push(json!({ "role": m.role, "content": m.content }));
    }

    let mut actions: Vec<String> = Vec::new();
    let mut changed: Vec<String> = Vec::new();

    for _ in 0..20 {
        let body = json!({
            "messages": convo,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.3,
            "stream": false,
        });

        let url = chat_url();
        let resp = client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("llama-server에 연결할 수 없습니다 ({url}): {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(format!("llama-server 오류 {status}: {text}"));
        }

        let v: Value = resp.json().await.map_err(|e| format!("응답 파싱 실패: {e}"))?;
        let msg = v["choices"][0]["message"].clone();
        convo.push(msg.clone());

        let tool_calls = msg["tool_calls"].as_array().cloned().unwrap_or_default();
        if tool_calls.is_empty() {
            let reply = msg["content"].as_str().unwrap_or_default().to_string();
            changed.dedup();
            return Ok(AgentResult { reply, actions, changed });
        }

        for tc in tool_calls {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let raw = &tc["function"]["arguments"];
            let args: Value = if raw.is_string() {
                serde_json::from_str(raw.as_str().unwrap_or("{}")).unwrap_or_else(|_| json!({}))
            } else {
                raw.clone()
            };
            let result = run_tool(&root, &name, &args, &mut actions, &mut changed);
            convo.push(json!({ "role": "tool", "tool_call_id": id, "content": result }));
        }
    }

    changed.dedup();
    Ok(AgentResult {
        reply: "(반복 한도에 도달해 중단했습니다. 다시 시도해 주세요.)".to_string(),
        actions,
        changed,
    })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(tauri_plugin_dialog::init())
        .manage(Workspace(Mutex::new(None)))
        .setup(|app| {
            let base = resource_base(app);
            let child = if server_running() {
                println!("llama-server가 이미 {LLAMA_PORT} 포트에서 실행 중 — 재사용합니다.");
                None
            } else {
                match spawn_llama(&base) {
                    Ok(c) => {
                        println!("llama-server 시작됨 (pid {})", c.id());
                        Some(c)
                    }
                    Err(e) => {
                        eprintln!("llama-server 자동 시작 실패: {e}");
                        None
                    }
                }
            };
            app.manage(LlamaServer(Mutex::new(child)));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            chat,
            agent_chat,
            llama_ready,
            select_workspace,
            current_workspace,
            workspace_files,
            read_file,
            write_file,
            open_in_vscode
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|app_handle, event| {
        if let tauri::RunEvent::Exit = event {
            if let Some(state) = app_handle.try_state::<LlamaServer>() {
                if let Some(mut child) = state.0.lock().unwrap().take() {
                    let _ = child.kill();
                }
            }
        }
    });
}
