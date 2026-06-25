# Windows용 llama-server

여기에 Windows용 `llama-server.exe` 와 **동반 DLL 전부**를 넣으세요.

- 권장 백엔드: **Vulkan**(범용 GPU) 또는 **CUDA**(NVIDIA)
- 받는 곳: https://github.com/ggml-org/llama.cpp/releases
  - 예: `llama-*-bin-win-vulkan-x64.zip` 압축을 풀어 그 안의
    `llama-server.exe`, `ggml*.dll`, `llama.dll` 등을 **이 폴더에 통째로** 복사

> 실행파일만 넣고 DLL을 빼면 실행되지 않습니다. zip 안의 파일을 모두 넣으세요.
> 실제 바이너리/DLL은 용량이 커서 git에 커밋되지 않습니다(.gitignore 처리).
