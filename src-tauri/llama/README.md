# llama-server 바이너리

이 폴더에 **현재 OS용 llama-server 실행파일**과 동반 라이브러리를 넣으세요.
머신마다 OS가 하나뿐이므로 OS별 하위 폴더는 두지 않습니다.

- **Windows**: `llama-server.exe` + 동반 DLL 전부 (예: `ggml*.dll`, `llama.dll`)
  - 권장 백엔드: Vulkan(범용 GPU) 또는 CUDA(NVIDIA)
- **Linux**: `llama-server` + 동반 `.so`  (`chmod +x llama-server`)
- **macOS**: `llama-server` (Metal)  (`chmod +x`, 필요 시 `xattr -dr com.apple.quarantine`)
  - 칩에 맞는 아키텍처(arm64/x86_64) 바이너리

받는 곳: https://github.com/ggml-org/llama.cpp/releases (또는 직접 빌드)

> 앱은 시작 시 `llama/llama-server(.exe)` 를 자동 실행합니다(이미 :8080에 서버가
> 떠 있으면 재사용). 모델은 `../models/` 의 `.gguf` 를 사용합니다.
> 실제 바이너리/DLL은 용량이 커서 git에 커밋되지 않습니다(README만 추적).
