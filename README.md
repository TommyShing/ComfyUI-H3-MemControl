# ComfyUI H3 MemControl

Test version of H3 memory control nodes. This package does not modify ComfyUI official code.

## Nodes

| Node | Input | Output | Purpose |
|---|---|---|---|
| `H3MemControlSetup` | `model: MODEL`, `clip: CLIP?`, `vae: VAE?`, `audio_vae: VAE?`, `manage_qwen: BOOLEAN` | same objects | Register H3 block scheduling and VAE cache; qwen scheduling is off by default |
| `H3MemControlCleanup` | `passthrough: ANY`, `stage: COMBO` | `passthrough: ANY` | Release managed block state, buffers, and LoRA references; does not clear user IO |
| `H3MemControlVAECache` | `vae: VAE`, `audio_vae: VAE?` | same objects | Keep one VAE instance per file path for the Comfy process lifetime |
| `H3MemControlDebug` | `model: MODEL?` | `status: STRING` | Return resident block, budget, cache, VRAM/RAM log state |

This is an experimental test build. It monitors block access and memory state, and attempts experimental H3 block scheduling by loading accessed blocks onto the configured compute device and evicting under a byte budget. Qwen block scheduling is disabled by default (`manage_qwen=false`) because naive qwen block swapping conflicts with Comfy prefetch/dequant on 8GB; H3 remains managed by MemControl. For H3, MemControl disables Comfy's dynamic-vbar prefetch queue so the managed block loop is not scanned twice, and auto cleanup unloads native qwen/VAE models before H3 sampling starts.

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
