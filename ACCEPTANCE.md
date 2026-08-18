# H3 MemControl Acceptance

Acceptance and current plan for the MemControl test build. ComfyUI files are never modified.

## Lifecycle Under Test

```text
qwen: SSD/内存 -> 显存一层 -> 文字编码完释放
H3: SSD/内存 -> 显存一层 -> 生成完释放
VAE: 常驻内存 -> 解码时进显存 -> 解码完回内存
```

## Test Table

| # | Check | Pass criteria | Current status |
|---|---|---|---|
| 1 | Setup registers qwen and H3 | Default `manage_qwen=true`; logs show qwen layers and H3 blocks | Unit verified, runtime pending |
| 2 | qwen no prefetch scan | `list(self.layers)` no longer schedules layers; only real forward access logs appear | Unit verified, runtime pending |
| 3 | qwen released after text encoding | Cleanup node releases qwen resident blocks and VRAM drops | Code path present, runtime pending |
| 4 | H3 no prefetch scan | No first pass over `blocks`; each real block access starts at count=1 | Unit verified, runtime pending |
| 5 | H3 released after sampling | Cleanup node releases H3 resident blocks and VRAM drops | Code path present, runtime pending |
| 6 | VAE cache | One instance per file path for process lifetime; VAE only in VRAM during decode | Cache code present, runtime pending |
| 7 | No ComfyUI changes | Git diff and repo contain no edits under ComfyUI path | Pass |
| 8 | 8GB workflow completes | qwen completes, H3 completes, VAE decode completes without OOM | Pending real workflow test |

## Current Plan

1. Keep qwen and H3 owned by MemControl, with Comfy excluded from loading/prefetching their layers.
2. Verify on the real workflow that qwen is released by the cleanup node after text encoding and before H3 sampling.
3. Verify on the real workflow that H3 no longer pre-scans blocks and only loads one managed block for real forward access.
4. If the real run still OOMs, use the `evict_after_*` memory logs to separate "weights were not released" from "activations/other models still occupy VRAM".
5. Only after those checks pass, treat the architecture as accepted and run the full VAE decode path.

## Current Evidence

- Unit tests: 10 passed.
- Compile check: passed.
- H3 `_forward` patch compiles against current Comfy source.
- Qwen `Llama2_.forward` patch compiles against current Comfy source.
- CUDA microtest: qwen-like layer container with a one-layer limit keeps only the active layer resident and returns weights to CPU after cleanup.
- CUDA microtest: MemControl evict/restore returns managed weights to CPU and clears resident state after cleanup.
