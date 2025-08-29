# GUFF Related Commands

Collection of commands relate to GUFF the model format used with Llama.cpp.

## Merge GUFF

For running inference using vLLM, we need to merge multi-file GUFF into a single file.

### Build

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON -DLLAMA_CURL=ON
cmake --build build --config Release
```

### Merge

```bash
cd build/bin
./llama-gguf-split --merge model.gguf.part1of2 merged_model.gguf
```
