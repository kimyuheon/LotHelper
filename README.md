# LotHelper (CppAI)

로컬 **llama.cpp** 서버에 붙는 데스크톱 채팅 앱입니다.
**Tauri 2 + React 19 + TypeScript** 프런트엔드가 로컬에서 돌아가는 LLM(예: Qwen2.5-Coder)과 대화합니다.

- 💬 대화형 채팅 UI (대화 맥락 유지)
- 🧑‍💻 Markdown + 코드 블록 문법 강조 + 복사 버튼
- 🔌 백엔드는 llama-server의 OpenAI 호환 API(`/v1/chat/completions`)에 연결

---

## 필요 환경 (다른 PC에서 처음 셋업할 때)

1. **Node.js** 18+ (`npm` 포함)
2. **Rust** 툴체인 — https://rustup.rs (`cargo`, `rustc`)
3. **Tauri 사전 요구사항** — Windows의 경우 *Microsoft C++ Build Tools* 와 *WebView2*
   (Windows 11에는 WebView2 기본 포함). 참고: https://tauri.app/start/prerequisites/
4. **llama.cpp** 의 `llama-server` 실행 파일과 **GGUF 모델 파일** 1개
   - llama.cpp: https://github.com/ggml-org/llama.cpp
   - 예시 모델: `qwen2.5-coder-7b-instruct-q4_k_m.gguf`

---

## 셋업

```bash
git clone https://github.com/kimyuheon/LotHelper.git
cd LotHelper
npm install
```

> `node_modules/`, `src-tauri/target/`, `dist/` 는 git에 포함되지 않으므로
> 클론 후 위 `npm install` 과 첫 빌드(아래)가 필요합니다.

---

## 실행

### 1) llama-server 먼저 띄우기

모델 경로는 본인 환경에 맞게 바꾸세요. (Vulkan 빌드 + GPU 오프로드 예시)

```powershell
& "경로\llama-server.exe" `
  -m "경로\models\qwen2.5-coder-7b-instruct-q4_k_m.gguf" `
  --host 127.0.0.1 --port 8080 `
  -ngl 99 -c 8192
```

- `--port 8080` 은 앱이 기대하는 기본 포트입니다.
- `-ngl 99` GPU 레이어 오프로드, `-c 8192` 컨텍스트 길이.

### 2) 앱 실행 (개발 모드)

```bash
npm run tauri dev
```

llama-server가 떠 있지 않으면 채팅창에
`llama-server에 연결할 수 없습니다` 오류가 표시됩니다.

---

## 코딩 모드 (프로젝트 만들고 코드 편집)

상단 탭에서 모드를 고릅니다. 채팅 외 모드는 먼저 **📁 폴더 선택**으로 작업 폴더를 지정해야 합니다.

| 모드 | 동작 |
| --- | --- |
| **채팅** | 일반 대화. 파일을 건드리지 않음 |
| **제안→승인** | 모델이 파일 변경을 제안 → 각 파일의 **적용** 버튼으로 저장 |
| **자동 적용** | 모델의 파일 제안을 자동 저장 |
| **에이전트** | 모델이 도구(list/read/write/make_dir/run_command)로 스스로 파일을 읽고 쓰며, 프로젝트를 **빌드하고 오류를 고쳐 성공할 때까지 반복**함 |

> 에이전트는 `run_command`로 작업 폴더 안에서 빌드/테스트 명령을 직접 실행합니다
> (예: `cargo build`, `npm run build`, `tsc --noEmit`). 명령당 240초 타임아웃,
> 한 요청당 최대 20번 반복합니다. 비대화형(서버/워처 제외) 명령만 사용하도록 지시됩니다.

- 파일은 선택한 작업 폴더 안에서만 생성/수정됩니다(폴더 밖 경로·`..` 차단).
- **VSCode로 열기** 버튼은 `code` CLI가 PATH에 있어야 합니다
  (VSCode 명령 팔레트 → "Shell Command: Install 'code' command in PATH").
  없으면 파일 탐색기로 대체로 열립니다.

## 빌드 (배포용 실행 파일)

```bash
npm run tauri build
```

결과물은 `src-tauri/target/release/` 및 `bundle/` 아래에 생성됩니다.

---

## 설정 바꾸기

| 항목 | 위치 |
| --- | --- |
| llama-server 주소/포트 | `src-tauri/src/lib.rs` 의 `LLAMA_URL` 상수 |
| 시스템 프롬프트 | `src-tauri/src/lib.rs` 의 `chat()` 안 system 메시지 |
| 온도(temperature) 등 | `src-tauri/src/lib.rs` 의 `ChatRequest` |
| 채팅 UI | `src/App.tsx`, `src/App.css` |
| Markdown/코드 렌더링 | `src/Markdown.tsx`, `src/Markdown.css` |

---

## 구조

```
src/                프런트엔드 (React)
  App.tsx           채팅 화면 + invoke("chat")
  Markdown.tsx      마크다운/코드블록 렌더링
src-tauri/src/
  lib.rs            chat 커맨드 → llama-server 호출
```
