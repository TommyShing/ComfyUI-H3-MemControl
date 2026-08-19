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

## Stage Occupancy Model

Each stage is sequential and uses its own weights. Stage weights and compute
temporaries should be modeled together, not as separate global items.

```text
H       = persistent hidden state, size S * D * bytes
W_i     = weights needed by stage i, split into sequentially used pieces
T_i     = temporary compute memory for stage i, split by chunk size C
Base    = Comfy/other base runtime

Stage peak = Base + H + max over pieces( piece_weight + piece_temp )
```

Because pieces are sequential, the peak is not the sum of all stage weights and
all stage temporaries. Each stage has different weight values, but the shape and
size pattern is similar. Chunking replaces `S` with `C` for temporaries; weight
pieces depend on the chosen granularity.

## qwen H and MLP Detail

H is the persistent hidden state and must stay in VRAM during a layer because
it is used by norm and residual paths. The 6-step table below lists step
intermediates separately from H. At 55k tokens H is about 0.55GB.

| Step | dequant VRAM weight | step intermediate (excluding H) |
|---|---|---|
| norm | ~0 | 0.55GB |
| attention | 0.2-0.3GB | 1.5-2.5GB |
| residual | 0 | 0.55GB |
| norm | ~0 | 0.55GB |
| MLP | 1.3-1.5GB | 2.8-5.6GB |
| residual | 0 | 0.55GB |

MLP sub-steps:

| MLP sub-step | dequant VRAM weight | intermediate |
|---|---|---|
| gate | 0.4-0.5GB | 2.8GB |
| up | 0.4-0.5GB | 2.8GB |
| product | 0 | 2.8GB |
| down | 0.4-0.5GB | input 2.8GB + output 0.55GB |

If gate/up are evaluated with both alive, peak is about 5.6GB for the MLP
intermediate. Chunking or fusing reduces that to the chunk size.

## Dynamic Async Policy

Async buffer size is not fixed at one full layer. It should be sized to the next
weight/compute piece that would actually be prefetched.

```text
Before prefetching next piece:
estimate = Base + H + current_piece_peak + next_piece_buffer

if estimate <= 8GB budget:
    use the second async stream to prefetch
else:
    do not prefetch
    load the next piece synchronously when needed
```

This lets small consecutive pieces overlap while avoiding OOM. MemControl owns
the decision and tracks actual per-stage/per-piece sizes instead of assuming a
constant buffer.

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
- Async weight loading can keep up to two full-weight-sized cast buffers in VRAM.
- 8GB card, about 7.6-7.7GB usable after driver/other overhead.

Async loading is included: one cast buffer is roughly the size of the weight being
transferred. qwen weight is about 0.9GB, so two async buffers are about 1.8GB.
H3 block is about 0.7GB, so two async buffers are about 1.4GB. MemControl will
keep the two streams for performance but clear the buffers between phases, so
the two-buffer worst case is bounded and not allowed to persist.

In the encoding node, qwen encoding runs first; VAE encode runs only when
keyframes/references exist, after qwen has finished. They are sequential, not
simultaneous. MemControl's VAE wrapper also releases VAE VRAM after each use.

Phases are sequential, not simultaneous:

```text
Phase 1: qwen text encode
Phase 2: H3 sampling
Phase 3: VAE decode
```

Do not add all three phases together.

| Phase | Estimated peak |
|---|---|
| qwen text-only | base 1.6 + current int8 weight 0.9 + activations 0.5-1.5 + cast buffers 0.9-1.8 = 3.9-5.8 GB |
| qwen with reference/video | base 1.6 + current int8 weight 0.9 + activations 1-4 + cast buffers 0.9-1.8 = 4.4-8.3 GB |
| H3 chunked | base 1.6 + current block 0.7 + activations 1-4 + cast buffers 0.7-1.4 = 4.0-7.7 GB |
| VAE decode | base 1.6 + VAE 0.5-1 + decode activations 0.5-1.5 = 2.6-4.1 GB |

The intended plan fits 8GB for text-only qwen and H3 if MemControl clears cast
buffers between phases. Reference/video token count and H3 activation peaks are
the main risks; they must be confirmed with real logs before accepting the plan.

Reference token estimate for 1152x640:

- One image/2-frame vision block: about 2880 tokens.
- 9 reference images: about 26k tokens.
- One 5s reference video is sampled at 2fps, about 5 vision blocks: about 14.4k tokens.
- Two 5s reference videos: about 28.8k tokens.
- Combined max: roughly 55k+ tokens.

At 5120 hidden and 25600 MLP width, 55k tokens alone can require:

- hidden/activation tensors: 0.5-1 GB
- MLP intermediates: 2-4 GB
- attention Q/K/V and related state: 2-4+ GB

That exceeds 8GB before counting base runtime and cast buffers. MemControl
cannot fix activation explosion caused by too many reference tokens. If max
references are required, qwen attention/MLP chunking or lower reference token
counts are needed.

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
