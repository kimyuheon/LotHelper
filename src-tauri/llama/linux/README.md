# Linux(Ubuntu)용 llama-server

여기에 Linux용 `llama-server` 실행파일과 **동반 .so 라이브러리**를 넣으세요.

- 권장 백엔드: **Vulkan** / **CUDA**(NVIDIA) / **ROCm**(AMD) / CPU
- 받는 곳: https://github.com/ggml-org/llama.cpp/releases 의 `ubuntu-*` 빌드
  또는 직접 빌드:
  ```bash
  cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release -j
  # build/bin/llama-server 와 build/bin/*.so 를 이 폴더로 복사
  ```

> 실행 권한 필요: `chmod +x llama-server`
> 실제 바이너리/.so 는 용량이 커서 git에 커밋되지 않습니다(.gitignore 처리).
