use serde::{Deserialize, Serialize};

/// llama-server's OpenAI-compatible chat endpoint.
const LLAMA_URL: &str = "http://127.0.0.1:8080/v1/chat/completions";

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

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

/// Send the conversation to the local llama-server and return the reply.
#[tauri::command]
async fn chat(messages: Vec<ChatMessage>) -> Result<String, String> {
    let mut full = vec![ChatMessage {
        role: "system".to_string(),
        content: "You are CppAI, a helpful coding assistant powered by Qwen2.5-Coder. \
                  Answer concisely and use Markdown code blocks for code."
            .to_string(),
    }];
    full.extend(messages);

    let body = ChatRequest {
        messages: full,
        stream: false,
        temperature: 0.7,
    };

    let client = reqwest::Client::new();
    let resp = client
        .post(LLAMA_URL)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("llama-server에 연결할 수 없습니다 ({LLAMA_URL}): {e}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("llama-server 오류 {status}: {text}"));
    }

    let parsed: ChatResponse = resp
        .json()
        .await
        .map_err(|e| format!("응답 파싱 실패: {e}"))?;

    parsed
        .choices
        .into_iter()
        .next()
        .map(|c| c.message.content)
        .ok_or_else(|| "응답에 choices가 없습니다.".to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![greet, chat])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
