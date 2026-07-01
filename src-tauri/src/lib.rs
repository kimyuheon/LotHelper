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
    - To READ a file's full content, output ```read:<relative/path> (empty body). \
    The runtime returns the file so you can see code that the snapshot truncated.\n\
    - To SEARCH the project for text, output ```search:<text> (empty body). \
    The runtime returns matching `path:line: text` results.\n\
    - To DELETE a file or folder, output ```delete:<relative/path> (empty body).\n\
    Use workspace-relative paths only (never absolute, never '..'). A list of the \
    project's files and a snapshot of small files are provided to you each turn; use \
    read/search to inspect anything not fully shown BEFORE editing it.\n\
    PATHS — be consistent: REUSE the exact paths shown in the project file list. If a \
    file already exists (e.g. main.cpp), edit THAT path — never create a duplicate in a \
    different folder (do NOT make src/main.cpp when main.cpp exists). Do NOT invent \
    subfolders for a simple program — keep its files together in the workspace root. \
    Files that reference each other (a .cpp, its .rc resource script, and resource.h) \
    MUST be in the SAME folder. After you emit \
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
    (L\"...\", MessageBoxW). A Win32 .rc resource script must be compiled with `rc` \
    FIRST (`rc main.rc` produces main.res), then linked: `cl /EHsc main.cpp main.res \
    user32.lib` — `cl main.cpp main.rc` does NOT work. For a simple dialog/message, \
    prefer a single file using MessageBoxW with NO .rc and NO resource.h.";

fn chat_url() -> String {
    format!("http://{LLAMA_HOST}:{LLAMA_PORT}/v1/chat/completions")
}

/// Auto-started llama-server child process (None if reused/failed).
struct LlamaServer(Mutex<Option<Child>>);

/// Currently selected project folder.
struct Workspace(Mutex<Option<PathBuf>>);

/// Base directory containing `llama/` and `models/` (resolved at startup).
struct AppBase(PathBuf);

/// GPU layer count (`-ngl`) for llama-server. Persisted to `<base>/ngl.txt`.
struct NglSetting(Mutex<i32>);

/// Name of the model we last started, so we can restart it (e.g. after -ngl change).
struct LastModel(Mutex<Option<String>>);

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

/// GPU layer count, persisted to `<base>/ngl.txt` (default 99 = offload all).
fn load_ngl(base: &Path) -> i32 {
    std::fs::read_to_string(base.join("ngl.txt"))
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(99)
}

fn save_ngl(base: &Path, ngl: i32) {
    let _ = std::fs::write(base.join("ngl.txt"), ngl.to_string());
}

fn spawn_llama(base: &Path, model: &Path, ngl: i32) -> std::io::Result<Child> {
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
        &ngl.to_string(),
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
    ngl: tauri::State<'_, NglSetting>,
    last: tauri::State<'_, LastModel>,
) -> Result<(), String> {
    if server_running() || llama.0.lock().unwrap().is_some() {
        return Ok(());
    }
    let model = base.0.join("models").join(&name);
    if !model.exists() {
        return Err(format!("모델을 찾을 수 없습니다: {name}"));
    }
    let n = *ngl.0.lock().unwrap();
    let child = spawn_llama(&base.0, &model, n).map_err(|e| e.to_string())?;
    *llama.0.lock().unwrap() = Some(child);
    *last.0.lock().unwrap() = Some(name);
    Ok(())
}

#[tauri::command]
fn get_ngl(ngl: tauri::State<'_, NglSetting>) -> i32 {
    *ngl.0.lock().unwrap()
}

/// Update the -ngl setting and persist it (applies on the next server start).
#[tauri::command]
fn set_ngl(value: i32, base: tauri::State<'_, AppBase>, ngl: tauri::State<'_, NglSetting>) {
    let v = value.clamp(0, 999);
    *ngl.0.lock().unwrap() = v;
    save_ngl(&base.0, v);
}

/// Kill the app-started llama-server (if any) and restart the last model with
/// the current -ngl. Used to apply a new -ngl to a running server.
#[tauri::command]
fn restart_llama(
    base: tauri::State<'_, AppBase>,
    llama: tauri::State<'_, LlamaServer>,
    ngl: tauri::State<'_, NglSetting>,
    last: tauri::State<'_, LastModel>,
) -> Result<(), String> {
    // Pick the model to (re)start: the last one we started, else the single model.
    let name = last.0.lock().unwrap().clone().or_else(|| {
        let models = find_models(&base.0);
        if models.len() == 1 {
            models[0]
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
        } else {
            None
        }
    });
    let Some(name) = name else {
        return Err("재시작할 모델을 알 수 없습니다. 모델을 먼저 선택하세요.".to_string());
    };

    // Stop the server we started (can't restart a manually-run one).
    if let Some(mut child) = llama.0.lock().unwrap().take() {
        let _ = child.kill();
        let _ = child.wait();
    } else if server_running() {
        return Err("외부에서 실행한 서버는 앱이 재시작할 수 없습니다. 수동으로 다시 띄워주세요.".to_string());
    }

    let model = base.0.join("models").join(&name);
    let n = *ngl.0.lock().unwrap();
    let child = spawn_llama(&base.0, &model, n).map_err(|e| e.to_string())?;
    *llama.0.lock().unwrap() = Some(child);
    *last.0.lock().unwrap() = Some(name);
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

/// Actions the model requested via fenced blocks.
#[derive(Default)]
struct Blocks {
    /// `file:PATH` blocks — (path, full content) to write.
    files: Vec<(String, String)>,
    /// `run`/`sh`/`bash`/... blocks — shell commands to execute.
    runs: Vec<String>,
    /// `read:PATH` blocks — files whose full content to return to the model.
    reads: Vec<String>,
    /// `search:TEXT` blocks — project searches to run and return matches for.
    searches: Vec<String>,
    /// `delete:PATH` blocks — files or folders to remove.
    deletes: Vec<String>,
}

impl Blocks {
    fn is_empty(&self) -> bool {
        self.files.is_empty()
            && self.runs.is_empty()
            && self.reads.is_empty()
            && self.searches.is_empty()
            && self.deletes.is_empty()
    }
}

/// Parse the model's reply into the action blocks it emitted.
fn parse_blocks(text: &str) -> Blocks {
    let mut b = Blocks::default();
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

        let lower = info.to_lowercase();
        if let Some(path) = info
            .strip_prefix("file:")
            .or_else(|| info.strip_prefix("File:"))
        {
            b.files.push((path.trim().to_string(), body));
        } else if let Some(path) = info
            .strip_prefix("read:")
            .or_else(|| info.strip_prefix("Read:"))
        {
            b.reads.push(path.trim().to_string());
        } else if let Some(q) = info
            .strip_prefix("search:")
            .or_else(|| info.strip_prefix("Search:"))
            .or_else(|| info.strip_prefix("grep:"))
        {
            b.searches.push(q.trim().to_string());
        } else if let Some(path) = info
            .strip_prefix("delete:")
            .or_else(|| info.strip_prefix("Delete:"))
            .or_else(|| info.strip_prefix("remove:"))
            .or_else(|| info.strip_prefix("rm:"))
        {
            b.deletes.push(path.trim().to_string());
        } else if matches!(
            lower.as_str(),
            "run" | "sh" | "bash" | "cmd" | "shell" | "console" | "powershell" | "ps" | "bat"
        ) {
            for cmd in body.lines() {
                let cmd = cmd.trim();
                if !cmd.is_empty() {
                    b.runs.push(cmd.to_string());
                }
            }
        }
    }

    b
}

/// A flat, sorted list of the workspace's files (for project structure).
fn workspace_tree(root: &Path) -> String {
    let mut files = Vec::new();
    walk(root, root, &mut files);
    files.sort();
    files.truncate(400);
    files.join("\n")
}

/// Search workspace text files for `pattern` (case-insensitive), returning
/// `path:line: text` matches, capped to keep the context small.
fn search_workspace(root: &Path, pattern: &str) -> String {
    if pattern.is_empty() {
        return "(빈 검색어)".to_string();
    }
    let needle = pattern.to_lowercase();
    let mut files = Vec::new();
    walk(root, root, &mut files);
    files.sort();

    let mut out = String::new();
    let mut hits = 0;
    for rel in &files {
        if hits >= 60 {
            out.push_str("...(검색 결과 더 있음)...\n");
            break;
        }
        let Ok(content) = std::fs::read_to_string(root.join(rel)) else {
            continue;
        };
        for (i, line) in content.lines().enumerate() {
            if line.to_lowercase().contains(&needle) {
                let snippet = truncate_str(line.trim(), 200);
                out.push_str(&format!("{rel}:{}: {snippet}\n", i + 1));
                hits += 1;
                if hits >= 60 {
                    break;
                }
            }
        }
    }
    if out.is_empty() {
        format!("'{pattern}' 검색 결과 없음")
    } else {
        out
    }
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

    let tree = workspace_tree(&root);
    if !tree.trim().is_empty() {
        convo.push(json!({
            "role": "system",
            "content": format!("Project files (workspace-relative paths):\n{tree}"),
        }));
    }
    let snapshot = workspace_snapshot(&root);
    if !snapshot.trim().is_empty() {
        convo.push(json!({
            "role": "system",
            "content": format!(
                "Snapshot of small files (larger ones are truncated — use ```read:<path> \
                 for full content). To edit a file, re-emit a ```file:<path> block with its \
                 FULL new content:\n\n{snapshot}"
            ),
        }));
    }
    for m in &messages {
        convo.push(json!({ "role": m.role, "content": m.content }));
    }

    let mut actions: Vec<String> = Vec::new();
    let mut changed: Vec<String> = Vec::new();
    let mut last_run_sig = String::new();
    let mut retries = 0;
    let mut temperature = 0.3f32;
    let mut identical_fails = 0;

    for _ in 0..25 {
        let body = json!({
            "messages": convo,
            "temperature": temperature,
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

        let blocks = parse_blocks(&content);

        // No blocks emitted. The model is either done, or it replied
        // conversationally / gave up ("can't", "안 됩니다"). If it hasn't acted
        // yet or is refusing, push back hard and retry (with more randomness) a
        // few times instead of accepting the give-up. Otherwise it's the final answer.
        if blocks.is_empty() {
            let gave_up = {
                let low = content.to_lowercase();
                ["cannot", "can't", "unable", "impossible", "not possible", "i'm sorry", "i am sorry"]
                    .iter()
                    .any(|m| low.contains(m))
                    || [
                        "안 됩니다", "안됩니다", "할 수 없", "불가능", "죄송", "못 하", "못합니다",
                        "안돼", "안 돼", "어렵습니다",
                    ]
                    .iter()
                    .any(|m| content.contains(m))
            };

            if retries < 3 && (actions.is_empty() || gave_up) {
                retries += 1;
                temperature = 0.8; // add variation so the retry isn't identical
                convo.push(json!({
                    "role": "user",
                    "content": "포기하지 마세요. '안 된다/불가능/죄송' 같은 말 대신, 지금 바로 \
                        ```file: 또는 ```run 블록으로 실제로 시도하세요. 한 방법이 막히면 \
                        완전히 다른 접근을 쓰고, 문제를 더 작게 쪼개서라도 진행하세요. \
                        텍스트만으로는 아무 일도 일어나지 않습니다."
                }));
                continue;
            }
            changed.sort();
            changed.dedup();
            return Ok(AgentResult { reply: content, actions, changed });
        }

        let mut feedback = String::new();

        // Deletions: remove files or folders (workspace-confined).
        for path in &blocks.deletes {
            match resolve(&root, path) {
                Ok(full) => {
                    let res = if full.is_dir() {
                        std::fs::remove_dir_all(&full)
                    } else {
                        std::fs::remove_file(&full)
                    };
                    match res {
                        Ok(()) => {
                            actions.push(format!("🗑️ 삭제: {path}"));
                            changed.push(path.clone());
                            feedback.push_str(&format!("deleted {path}\n"));
                        }
                        Err(e) => feedback.push_str(&format!("delete {path} ERROR: {e}\n")),
                    }
                }
                Err(e) => feedback.push_str(&format!("delete {path} ERROR: {e}\n")),
            }
        }

        // Read-only inspection: return file contents and search matches.
        for path in &blocks.reads {
            actions.push(format!("👀 읽기: {path}"));
            match resolve(&root, path).and_then(|p| std::fs::read_to_string(p).map_err(|e| e.to_string())) {
                Ok(c) => feedback.push_str(&format!("--- {path} ---\n{}\n\n", truncate_str(&c, 6000))),
                Err(e) => feedback.push_str(&format!("read {path} ERROR: {e}\n")),
            }
        }
        for q in &blocks.searches {
            actions.push(format!("🔍 검색: {q}"));
            let r = search_workspace(&root, q);
            feedback.push_str(&format!("search '{q}':\n{r}\n\n"));
        }

        for (path, contents) in &blocks.files {
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
        for cmd in &blocks.runs {
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

        // Track repeated identical failures (the model's edits aren't changing
        // the result). Don't give up immediately — push it to try a DIFFERENT
        // approach, and only stop after several identical rounds in a row.
        if any_fail && !run_sig.is_empty() && run_sig == last_run_sig {
            identical_fails += 1;
        } else {
            identical_fails = 0;
        }
        last_run_sig = run_sig;

        if identical_fails >= 3 {
            changed.sort();
            changed.dedup();
            return Ok(AgentResult {
                reply: format!(
                    "다른 방법을 시도했는데도 같은 오류가 계속 나서 중단했습니다.\n\n\
                     ```\n{last_output}\n```\n\n오류를 직접 보고 고치거나, 더 작게 나눠 \
                     지시해 주세요."
                ),
                actions,
                changed,
            });
        }

        if any_fail && identical_fails >= 1 {
            // Same error as before → tell it to change strategy, don't repeat.
            feedback.push_str(
                "\n[중요] 방금과 똑같은 오류입니다. 같은 방법은 통하지 않습니다. \
                 완전히 다른 접근을 시도하세요 — 다른 컴파일 옵션/라이브러리, 다른 \
                 코드 구조나 함수, 또는 문제를 더 작게 쪼개기. 같은 명령·코드를 반복하지 마세요.",
            );
        } else if any_fail {
            feedback.push_str(
                "\nA command FAILED. If a needed source file does not exist yet, output its \
                 FULL content in a ```file:<path> block FIRST, then build again. If there are \
                 compile errors, edit the file to fix them. Do NOT repeat the same command \
                 unchanged.",
            );
        } else if blocks.runs.is_empty() {
            feedback.push_str(
                "Done. Build/run to verify if you wrote code, or reply with a one-line summary \
                 and no code blocks if the task is complete.",
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
            let ngl = load_ngl(&base);
            let mut last_model: Option<String> = None;
            let child = if server_running() {
                println!("llama-server가 이미 {LLAMA_PORT} 포트에서 실행 중 — 재사용합니다.");
                None
            } else {
                let models = find_models(&base);
                match models.len() {
                    // Exactly one model → start it automatically.
                    1 => match spawn_llama(&base, &models[0], ngl) {
                        Ok(c) => {
                            println!("llama-server 시작됨 (pid {})", c.id());
                            last_model =
                                models[0].file_name().map(|n| n.to_string_lossy().to_string());
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
            app.manage(NglSetting(Mutex::new(ngl)));
            app.manage(LastModel(Mutex::new(last_model)));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            chat,
            agent_chat,
            llama_ready,
            llama_status,
            start_model,
            get_ngl,
            set_ngl,
            restart_llama,
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
        let b = parse_blocks(t);
        assert_eq!(b.files.len(), 1);
        assert_eq!(b.files[0].0, "src/main.cpp");
        assert!(b.files[0].1.contains("int main"));
        assert_eq!(b.runs, vec!["cl /EHsc src/main.cpp".to_string()]);
    }

    #[test]
    fn parses_read_search_delete_blocks() {
        let b = parse_blocks("Look:\n```read:src/lib.rs\n```\n```search:TODO\n```\n```delete:src\n```");
        assert_eq!(b.reads, vec!["src/lib.rs".to_string()]);
        assert_eq!(b.searches, vec!["TODO".to_string()]);
        assert_eq!(b.deletes, vec!["src".to_string()]);
        assert!(b.files.is_empty() && b.runs.is_empty());
        assert!(!b.is_empty());
    }

    #[test]
    fn no_blocks_means_done() {
        let b = parse_blocks("All done. The build succeeded.");
        assert!(b.is_empty());
    }

    #[test]
    fn agent_system_example_uses_real_newlines() {
        // Guard against the `\\n`-in-the-example bug (model parroted literal \n).
        assert!(!AGENT_SYSTEM.contains("\\n"), "system prompt has literal backslash-n");
        assert!(AGENT_SYSTEM.contains("```run\ncl /EHsc main.cpp\n```"));
        // The example must itself parse as a valid run block.
        let b = parse_blocks(AGENT_SYSTEM);
        assert!(b.runs.iter().any(|c| c == "cl /EHsc main.cpp"));
    }
}
