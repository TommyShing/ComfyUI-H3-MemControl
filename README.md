# ComfyUI H3 MemControl

Test version of H3 memory control nodes. This package does not modify ComfyUI official code.

## Nodes

| Node | Input | Output | Purpose |
|---|---|---|---|
| `H3MemControlSetup` | `model: MODEL`, `clip: CLIP?`, `vae: VAE?`, `audio_vae: VAE?` | same objects | Inspect and register H3/Qwen block containers, VAE cache, memory state |
| `H3MemControlCleanup` | `passthrough: ANY`, `stage: COMBO` | `passthrough: ANY` | Release managed block state, buffers, and LoRA references; does not clear user IO |
| `H3MemControlVAECache` | `vae: VAE`, `audio_vae: VAE?` | same objects | Keep one VAE instance per file path for the Comfy process lifetime |
| `H3MemControlDebug` | `model: MODEL?` | `status: STRING` | Return resident block, budget, cache, VRAM/RAM log state |

This is an experimental test build. It monitors block access and memory state, and also attempts experimental block scheduling by loading accessed blocks onto the configured compute device and evicting under a byte budget. If blocks are still in meta state or no CUDA device is available, it falls back to monitoring only. It logs lifecycle events to the ComfyUI console; `H3MemControlDebug` can be used instead when console output is not needed.

## Workflow

```text
CLIPLoader -> H3MemControlSetup -> MiniMaxH3ImageToVideo
UNETLoader -> Turbo/LoRA -> H3MemControlSetup -> Sampler
VAELoader -> H3MemControlVAECache -> H3MemControlSetup -> H3 node / VAE decode
Sampler -> H3MemControlCleanup -> VAEDecode
```

## Development

```powershell
E:\Stable Diffusion\ComfyUI-aki-v3.2\python\python.exe -m unittest discover -s tests -v
```
