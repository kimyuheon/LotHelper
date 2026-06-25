# macOS용 llama-server

여기에 macOS용 `llama-server` 실행파일을 넣으세요.

- 권장 백엔드: **Metal** (Apple Silicon은 기본 내장 가속)
- 받는 곳:
  - `brew install llama.cpp` 후 `$(brew --prefix)/bin/llama-server` 복사, 또는
  - https://github.com/ggml-org/llama.cpp/releases 의 `macos-*` 빌드, 또는
  - 직접 빌드:
    ```bash
    cmake -B build -DGGML_METAL=ON && cmake --build build --config Release -j
    ```
- Apple Silicon(M계열)과 Intel Mac은 아키텍처가 다릅니다. 둘 다 배포하려면
  `arm64`/`x86_64` 바이너리를 각각 준비하거나 universal 바이너리(`lipo`)로 합치세요.

> 실행 권한 필요: `chmod +x llama-server`
> 실제 바이너리는 용량이 커서 git에 커밋되지 않습니다(.gitignore 처리).
