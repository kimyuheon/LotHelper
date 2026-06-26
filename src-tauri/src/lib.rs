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
    user's project folder. You act by emitting fenced code blocks that the runtime \
    executes for you:\n\
    - To CREATE or OVERWRITE a file, output its FULL content in a block whose info \
    string is `file:<relative/path>` — e.g. ```file:src/main.cpp ... ```\n\
    - To RUN a shell command (build/compile/test/run), output a ```run block with one \
    shell command per line.\n\
    Use workspace-relative paths only (never absolute, never '..'). After you emit \
    blocks, the runtime applies the files, runs the commands, and sends you their output. \
    Read the output: if a build fails, FIX the files and run again. Repeat until it builds \
    and runs cleanly. Keep prose short.\n\
    CRITICAL: text alone does NOTHING. To create/edit a file you MUST emit a ```file: \
    block; to run/build you MUST emit a ```run block. Never claim you did something \
    (\"it works\", \"build succeeded\") unless you actually emitted the block that does it. \
    EXAMPLE — write a file then build it:\n\
    ```file:main.cpp\n\
    #include <cstdio>\n\
    int main(){ printf(\"hi\"); return 0; }\n\
    ```\n\
    ```run\n\
    cl /EHsc main.cpp\n\
    ```\n\
    When the task is fully done and the build passed, reply with a one-line summary and \
    NO code blocks (that ends the task).\n\
    For C/C++ on Windows the MSVC compiler `cl` is available (e.g. `cl /EHsc main.cpp`); \
    on Linux/macOS use `g++`/`clang++`. Prefer a single compile command over CMake \
    unless the task needs it. For Win32 GUI/dialog code you MUST link the needed \
    libraries explicitly, e.g. `cl /EHsc main.cpp user32.lib gdi32.lib` — a bare \
    `cl /EHsc main.cpp` fails to link WinAPI functions like MessageBox. For non-ASCII \
    text (e.g. Korean) in C/C++ source, add the `/utf-8` flag and use wide strings \
    (L\"...\", MessageBoxW).";

fn chat_url() -> String {
    format!("http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions")
}

/// Auto-started llama-server child process (None if reused/failed).
struct LlamaServer(Mutex<Option<Child>>);

/// Currently selected project folder.
struct Workspace(Mutex<Option<PathBuf>>);

/// Base directory containing `llama/` and `models/` (resolved at startup).
struct AppBase(PathBuf);

#[derive(Serialize)]
struct LlamaStatus {
    /// "starting" (running or loading) | "choose" (pick a model) | "no_model".
    state: String,
    models: Vec<String>,
}

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
    let exe = if cfg!(windows) {
        "llama-server.exe"
    } else {
        "llama-server"
    };
    base.join("llama").join(exe)
}

/// All `*.gguf` files in `models/`, sorted by name.
fn find_models(base: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(base.join("models"))
        .map(|rd| {
            rd.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().map_or(false, |x| x.eq_ignore_ascii_case("gguf")))
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

fn server_running() -> bool {
    let addr = format!("{LLAMA_HOST}:{LLAMA_PORT}");
    match addr.parse() {
        Ok(sa) => TcpStream::connect_timeout(&sa, Duration::from_millis(300)).is_ok(),
        Err(_) => false,
    }
}

fn spawn_llama(base: &Path, model: &Path) -> std::io::Result<Child> {
    let bin = llama_binary(base);
    if !bin.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("llama-server를 찾을 수 없습니다: {}", bin.display()),
        ));
    }

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
        // Required so the server parses the model's function calls into the
        // OpenAI `tool_calls` field (needed for agent mode).
        "--jinja",
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

// ---- llama startup commands ------------------------------------------------

/// Tell the frontend whether the server is up/loading, or which model to pick.
#[tauri::command]
fn llama_status(
    base: tauri::State<'_, AppBase>,
    llama: tauri::State<'_, LlamaServer>,
) -> LlamaStatus {
    let has_child = llama.0.lock().unwrap().is_some();
    if has_child || server_running() {
        return LlamaStatus {
            state: "starting".to_string(),
            models: vec![],
        };
    }
    let models: Vec<String> = find_models(&base.0)
        .iter()
        .filter_map(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .collect();
    if models.is_empty() {
        LlamaStatus {
            state: "no_model".to_string(),
            models,
        }
    } else {
        LlamaStatus {
            state: "choose".to_string(),
            models,
        }
    }
}

/// Start llama-server with the chosen model (no-op if one is already running).
#[tauri::command]
fn start_model(
    name: String,
    base: tauri::State<'_, AppBase>,
    llama: tauri::State<'_, LlamaServer>,
) -> Result<(), String> {
    if server_running() || llama.0.lock().unwrap().is_some() {
        return Ok(());
    }
    let model = base.0.join("models").join(&name);
    if !model.exists() {
        return Err(format!("모델을 찾을 수 없습니다: {name}"));
    }
    let child = spawn_llama(&base.0, &model).map_err(|e| e.to_string())?;
    *llama.0.lock().unwrap() = Some(child);
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

/// Capture the MSVC build environment (so `cl`, `nmake`, etc. become available)
/// by locating vcvars64.bat via vswhere and dumping its environment once. Cached.
/// Returns None if Visual Studio / Build Tools isn't installed.
#[cfg(windows)]
fn msvc_env() -> &'static Option<Vec<(String, String)>> {
    use std::os::windows::process::CommandExt;
    const NO_WINDOW: u32 = 0x0800_0000;
    static ENV: std::sync::OnceLock<Option<Vec<(String, String)>>> = std::sync::OnceLock::new();

    ENV.get_or_init(|| {
        let pf86 = std::env::var("ProgramFiles(x86)").ok()?;
        let vswhere =
            Path::new(&pf86).join("Microsoft Visual Studio\\Installer\\vswhere.exe");
        if !vswhere.exists() {
            return None;
        }
        let out = Command::new(&vswhere)
            .args(["-latest", "-products", "*", "-property", "installationPath"])
            .creation_flags(NO_WINDOW)
            .output()
            .ok()?;
        let inst = String::from_utf8_lossy(&out.stdout).lines().next()?.trim().to_string();
        if inst.is_empty() {
            return None;
        }
        let vcvars = Path::new(&inst).join("VC\\Auxiliary\\Build\\vcvars64.bat");
        if !vcvars.exists() {
            return None;
        }

        // Dump the environment that vcvars sets up.
        let bat = std::env::temp_dir().join("cppai_vcvars_dump.bat");
        std::fs::write(
            &bat,
            format!("@echo off\r\ncall \"{}\" >nul 2>&1\r\nset\r\n", vcvars.display()),
        )
        .ok()?;
        let dump = Command::new("cmd")
            .args(["/C"])
            .arg(&bat)
            .creation_flags(NO_WINDOW)
            .output()
            .ok()?;

        let text = String::from_utf8_lossy(&dump.stdout);
        let vars: Vec<(String, String)> = text
            .lines()
            .filter_map(|l| l.split_once('=').map(|(k, v)| (k.to_string(), v.to_string())))
            .collect();
        if vars.is_empty() {
            None
        } else {
            Some(vars)
        }
    })
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
        // Make the MSVC toolchain (cl, etc.) available for C/C++ builds.
        if let Some(env) = msvc_env() {
            cmd.envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())));
        }
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

/// Parse the model's reply into file blocks (```file:PATH ... ```) and run
/// blocks (```run / ```sh / ```bash / ```cmd / ```powershell ...).
fn parse_blocks(text: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut files = Vec::new();
    let mut runs = Vec::new();
    let mut lines = text.lines();

    while let Some(line) = lines.next() {
        let Some(rest) = line.trim_start().strip_prefix("```") else {
            continue;
        };
        let info = rest.trim().to_string();

        // Collect the block body up to the closing fence.
        let mut body = String::new();
        for l in lines.by_ref() {
            if l.trim_start().starts_with("```") {
                break;
            }
            body.push_str(l);
            body.push('\n');
        }
        if body.ends_with('\n') {
            body.pop();
        }

        if let Some(path) = info
            .strip_prefix("file:")
            .or_else(|| info.strip_prefix("File:"))
        {
            files.push((path.trim().to_string(), body));
        } else if matches!(
            info.to_lowercase().as_str(),
            "run" | "sh" | "bash" | "cmd" | "shell" | "console" | "powershell" | "ps" | "bat"
        ) {
            for cmd in body.lines() {
                let cmd = cmd.trim();
                if !cmd.is_empty() {
                    runs.push(cmd.to_string());
                }
            }
        }
    }

    (files, runs)
}

fn truncate_str(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// A snapshot of the workspace's text files (names + contents) so the agent can
/// see and edit existing code. Capped to stay within the model's context.
fn workspace_snapshot(root: &Path) -> String {
    let mut files = Vec::new();
    walk(root, root, &mut files);
    files.sort();

    let mut out = String::new();
    let mut total = 0usize;
    for rel in &files {
        if total > 6000 {
            out.push_str("...(나머지 파일 생략)...\n");
            break;
        }
        if let Ok(content) = std::fs::read_to_string(root.join(rel)) {
            let snippet = truncate_str(&content, 2500);
            out.push_str(&format!("--- {rel} ---\n{snippet}\n\n"));
            total += snippet.len();
        }
    }
    out
}

#[tauri::command]
async fn agent_chat(
    messages: Vec<ChatMessage>,
    state: tauri::State<'_, Workspace>,
) -> Result<AgentResult, String> {
    let root = ws_root(&state)?;
    let client = reqwest::Client::new();

    let mut convo: Vec<Value> = vec![json!({ "role": "system", "content": AGENT_SYSTEM })];
    let snapshot = workspace_snapshot(&root);
    if !snapshot.trim().is_empty() {
        convo.push(json!({
            "role": "system",
            "content": format!(
                "Current files in the workspace. To edit one, re-emit a ```file:<path> block \
                 with its FULL new content:\n\n{snapshot}"
            ),
        }));
    }
    for m in &messages {
        convo.push(json!({ "role": m.role, "content": m.content }));
    }

    let mut actions: Vec<String> = Vec::new();
    let mut changed: Vec<String> = Vec::new();
    let mut last_runs: Vec<String> = Vec::new();
    let mut last_run_sig = String::new();
    let mut nudged = false;

    for _ in 0..12 {
        let body = json!({
            "messages": convo,
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
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or_default()
            .to_string();
        convo.push(json!({ "role": "assistant", "content": content.clone() }));

        let (file_blocks, runs) = parse_blocks(&content);

        // No blocks emitted. If nothing has been done yet, the model is likely
        // replying conversationally instead of acting — nudge it once. Otherwise
        // treat this as the final answer.
        if file_blocks.is_empty() && runs.is_empty() {
            if actions.is_empty() && !nudged {
                nudged = true;
                convo.push(json!({
                    "role": "user",
                    "content": "당신은 ```file: 또는 ```run 블록을 출력하지 않아 아무 작업도 \
                        실행되지 않았습니다. 텍스트만으로는 아무 일도 일어나지 않습니다. \
                        요청을 수행하려면 지금 해당 블록을 출력하세요."
                }));
                continue;
            }
            changed.sort();
            changed.dedup();
            return Ok(AgentResult { reply: content, actions, changed });
        }

        // Stuck: repeating the same command(s) without writing/changing any file.
        if file_blocks.is_empty() && !runs.is_empty() && runs == last_runs {
            changed.sort();
            changed.dedup();
            return Ok(AgentResult {
                reply: "같은 명령을 반복하기만 해서 중단했습니다. 모델이 빌드 오류를 스스로 \
                        고치지 못하는 것 같습니다. '먼저 main.cpp에 ~코드를 써줘'처럼 \
                        파일 작성을 명시해 더 작게 나눠 지시해 주세요."
                    .to_string(),
                actions,
                changed,
            });
        }
        last_runs = runs.clone();

        let mut feedback = String::new();

        for (path, contents) in &file_blocks {
            match resolve(&root, path) {
                Ok(full) => {
                    if let Some(parent) = full.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    match std::fs::write(&full, contents) {
                        Ok(()) => {
                            actions.push(format!("✏️ 쓰기: {path}"));
                            changed.push(path.clone());
                            feedback.push_str(&format!("wrote {path}\n"));
                        }
                        Err(e) => feedback.push_str(&format!("write {path} ERROR: {e}\n")),
                    }
                }
                Err(e) => feedback.push_str(&format!("write {path} ERROR: {e}\n")),
            }
        }

        let mut any_fail = false;
        let mut run_sig = String::new();
        let mut last_output = String::new();
        for cmd in &runs {
            actions.push(format!("⚙️ 실행: {cmd}"));
            let out = run_shell(&root, cmd, Duration::from_secs(240));
            if !out.contains("exit code: 0") {
                any_fail = true;
            }
            run_sig.push_str(cmd);
            run_sig.push('\n');
            run_sig.push_str(&out);
            last_output = out.clone();
            feedback.push_str(&format!("$ {cmd}\n{out}\n"));
        }

        // Stuck: a command keeps failing with the exact same output as last round
        // (the model's edits aren't changing the result) → stop and show the error.
        if any_fail && !run_sig.is_empty() && run_sig == last_run_sig {
            changed.sort();
            changed.dedup();
            return Ok(AgentResult {
                reply: format!(
                    "빌드가 같은 오류로 계속 실패해서 중단했습니다. 모델이 이 오류를 \
                     스스로 못 고치는 것 같습니다.\n\n```\n{last_output}\n```\n\n\
                     오류를 직접 보고 고치거나, 더 작게 나눠 지시해 주세요."
                ),
                actions,
                changed,
            });
        }
        last_run_sig = run_sig;

        if any_fail {
            feedback.push_str(
                "\nA command FAILED. If a needed source file does not exist yet, output its \
                 FULL content in a ```file:<path> block FIRST, then build again. If there are \
                 compile errors, edit the file to fix them. Do NOT repeat the same command \
                 unchanged.",
            );
        } else if runs.is_empty() {
            feedback.push_str(
                "Files applied. Build/run to verify, or reply with a one-line summary and \
                 no code blocks if the task is complete.",
            );
        }

        convo.push(json!({ "role": "user", "content": feedback }));
    }

    changed.sort();
    changed.dedup();
    Ok(AgentResult {
        reply: "(반복 한도에 도달해 중단했습니다. 더 작게 나눠 다시 시도해 주세요.)".to_string(),
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
                let models = find_models(&base);
                match models.len() {
                    // Exactly one model → start it automatically.
                    1 => match spawn_llama(&base, &models[0]) {
                        Ok(c) => {
                            println!("llama-server 시작됨 (pid {})", c.id());
                            Some(c)
                        }
                        Err(e) => {
                            eprintln!("llama-server 자동 시작 실패: {e}");
                            None
                        }
                    },
                    // Multiple models → let the user choose in the UI.
                    0 => {
                        eprintln!("models/ 폴더에 .gguf 모델이 없습니다.");
                        None
                    }
                    n => {
                        println!("모델 {n}개 발견 — 사용자 선택을 기다립니다.");
                        None
                    }
                }
            };
            app.manage(AppBase(base));
            app.manage(LlamaServer(Mutex::new(child)));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            chat,
            agent_chat,
            llama_ready,
            llama_status,
            start_model,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_file_and_run_blocks() {
        let t = "Sure:\n```file:src/main.cpp\nint main(){return 0;}\n```\nNow build:\n```run\ncl /EHsc src/main.cpp\n```";
        let (files, runs) = parse_blocks(t);
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].0, "src/main.cpp");
        assert!(files[0].1.contains("int main"));
        assert_eq!(runs, vec!["cl /EHsc src/main.cpp".to_string()]);
    }

    #[test]
    fn no_blocks_means_done() {
        let (files, runs) = parse_blocks("All done. The build succeeded.");
        assert!(files.is_empty() && runs.is_empty());
    }

    #[test]
    fn agent_system_example_uses_real_newlines() {
        // Guard against the `\\n`-in-the-example bug (model parroted literal \n).
        assert!(!AGENT_SYSTEM.contains("\\n"), "system prompt has literal backslash-n");
        assert!(AGENT_SYSTEM.contains("```run\ncl /EHsc main.cpp\n```"));
        // The example must itself parse as a valid run block.
        let (_files, runs) = parse_blocks(AGENT_SYSTEM);
        assert!(runs.iter().any(|c| c == "cl /EHsc main.cpp"));
    }
}
