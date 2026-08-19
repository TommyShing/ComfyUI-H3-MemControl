# MemControl INT8 Compute and Cache Plan

Goal: analyze whether qwen/H3 can run with int8 compute instead of the current
"int8 storage -> bf16 dequant -> bf16 matmul" path, without modifying ComfyUI
files if possible. Quantized weights stay in RAM; runtime caches/activations
use VRAM; qwen/H3 run one layer/block at a time with chunked compute.

## Current Compute Flow

```text
qwen:
- embedding: int8 stored, dequantized/partial
- RMSNorm: high precision
- attention QKV/linear: current path dequantizes to bf16
- attention softmax: high precision
- MLP gate/up/down: current path dequantizes to bf16
- residual/add: high precision

H3:
- token_refiner blocks: same Linear dequant path as qwen
- main blocks:
  - attention can use comfy-kitchen int8 attention when ModelAttentionBackend selects it
  - MLP/linear still use dequant/bf16 path in the current stack
- final layers and norms: high precision
```

## Which Parts Can Be INT8

| Part | INT8 feasible | Precision risk | Notes |
|---|---|---|---|
| Linear projections (QKV, O, MLP) | Yes | Medium | Use dynamic per-row activation quant + int8 GEMM + dequant output |
| Attention Q/K/V | Yes | Low-Medium | comfy-kitchen int8 attention exists; H3 has an alignment bug in an unmerged PR |
| RMSNorm/LayerNorm | No | High | Keep high precision |
| Residual/add | No | High | Keep high precision |
| RoPE/position embedding | No | High | Keep high precision |
| Final conditioning output | No | High | Directly consumed by H3, keep high precision |
| Embedding lookup | Partial | Low | Can gather/dequant only used rows |

Precision-sensitive steps should remain bf16/fp32. INT8 should be limited to
GEMM/attention paths that naturally scale back to high precision.

## Native Comfy Feasibility

Comfy already has:

- `ModelAttentionBackend` node with `comfy kitchen attention` for int8 attention.
- `comfy_kitchen.int8_linear` kernel with ConvRot support.
- A narrow native path (`linear_input_act`) that uses `int8_linear` for some
  merged SwiGLU TensorWise INT8 layers.

The current qwen/H3 MLP does not go through that native path. It uses
`cast_bias_weight` which dequantizes ConvRot INT8 weights to bf16.

Conclusion:

- Attention: possible with existing Comfy node, plus a contiguous-input fix
  needed for H3. The fix can be supplied by MemControl without editing Comfy.
- Linear/MLP: not currently forced by native Comfy. A custom node or custom ops
  patch is needed to route managed layers through `int8_linear`.
- No-Comfy-edit constraint is respected; runtime patching from custom nodes is
  still on the table.

## Implementation Direction

1. Keep high precision for norms/residual/RoPE/final output.
2. Route qwen and H3 Linear layers through `comfy_kitchen.int8_linear` when:
   - weight is `QuantizedTensor`
   - shape/backend constraints are satisfied
   - bias/LoRA/patches can be handled
3. Route H3 attention through comfy-kitchen int8 attention with contiguous q/k/v.
4. Add A/B verification: same seed, compare conditioning and latent output with
   a tolerance; do not accept silent quality regression.
5. If native Comfy cannot do this safely, implement it in a separate custom node
   owned by MemControl.

## VRAM Peak Estimate

Assumptions:

- Quantized weights live in RAM, not VRAM.
- Only one qwen layer or one H3 block is active at a time.
- Runtime cast buffers/caches live in VRAM and are cleared by MemControl at phase boundaries.
- Attention/MLP use chunking where supported.
- 8GB card, about 7.6-7.7GB usable after driver/other overhead.

| Component | Estimated peak |
|---|---|
| Comfy/VAE/base runtime | 1.0-1.6 GB |
| qwen current int8 weight in VRAM | 0.9 GB (if int8 linear) |
| qwen dequant bf16 fallback weight | 1.9 GB (if fallback needed) |
| qwen text-only activations | 0.5-1.5 GB |
| qwen with reference/video tokens | 1-4 GB, depends on token count |
| H3 one current block | 0.7 GB |
| H3 chunked attention/MLP activations | 1-4 GB |
| VAE decode | 0.5-2 GB |

Realistic peak for the intended plan:

```text
base 1.6 + qwen int8 weight 0.9 + qwen activations 1.5 + bounded cast buffers 1-2
= about 5-6 GB
```

That fits an 8GB card if caches are cleared and only one model phase is active.
The previous OOM came from whole-layer `block.to(cuda)` plus unmanaged cast
buffers, which are outside this plan. If reference/video token count is large,
the estimate should be verified with actual logs before accepting the plan.

## Cache Management List

MemControl can safely clear at lifecycle boundaries:

- STREAM_CAST_BUFFERS
- STREAM_AIMDO_CAST_BUFFERS
- PREFETCH_QUEUES
- GRAPH_MODULES / GRAPH_WARMED_MODULES / GRAPH_CAPTURE_STREAMS
- CROSS_STEP_STATE
- module attrs: _prefetch / _v_weight / _v_bias / _v_signature

Must not be blindly cleared:

- PINNED_MEMORY / host buffers / dynamic_pins
- dynamic_vbars
- current_loaded_models
- mmap / DIRTY_MMAPS
- active streams and graph state
