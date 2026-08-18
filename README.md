# ComfyUI H3 MemControl

Test version of H3 memory control nodes. This package does not modify ComfyUI official code.

## Nodes

| Node | Input | Output | Purpose |
|---|---|---|---|
| `H3MemControlSetup` | `model: MODEL`, `clip: CLIP?`, `vae: VAE?`, `audio_vae: VAE?`, `manage_qwen: BOOLEAN` | same objects | Register H3 and qwen block scheduling and VAE cache; qwen scheduling is on by default |
| `H3MemControlCleanup` | `passthrough: ANY`, `stage: COMBO` | `passthrough: ANY` | Release managed block state, buffers, and LoRA references; does not clear user IO |
| `H3MemControlVAECache` | `vae: VAE`, `audio_vae: VAE?` | same objects | Keep one VAE instance per file path for the Comfy process lifetime |
| `H3MemControlDebug` | `model: MODEL?` | `status: STRING` | Return resident block, budget, cache, VRAM/RAM log state |

This is an experimental test build. It monitors block access and memory state, and schedules qwen and H3 by loading accessed layers/blocks onto the configured compute device and evicting under a byte budget. MemControl disables Comfy's prefetch scanning for both managed qwen layers and H3 blocks so they are only loaded on real access. Auto cleanup releases qwen after text encoding and H3 after sampling. `manage_qwen=false` remains only as an explicit Comfy-native fallback.

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
