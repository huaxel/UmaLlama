# UmaLLama

A performance-focused fork of [llama.cpp](https://github.com/ggml-org/llama.cpp) for
running local LLM inference on AMD APU hardware — integrated GPUs, handhelds,
and any system with shared memory. UmaLLama tracks upstream `llama.cpp`
closely but diverges freely where performance is on the line: we cherry-pick
from upstream PRs, borrow from other forks, and ship our own optimizations.
If a change makes inference faster on lower-spec hardware, it has a home here
regardless of where it originated.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Upstream: ggml-org/llama.cpp](https://img.shields.io/badge/upstream-ggml--org%2Fllama.cpp-orange.svg)](https://github.com/ggml-org/llama.cpp)
[![Parent project: fewtarius/llama-ai](https://img.shields.io/badge/parent-fewtarius%2Fllama--ai-green.svg)](https://github.com/fewtarius/llama-ai)

[llama-ai parent project](https://github.com/fewtarius/llama-ai) / [CLIO agentic client](https://github.com/SyntheticAutonomicMind/CLIO) / [Upstream llama.cpp](https://github.com/ggml-org/llama.cpp) / [ggml](https://github.com/ggml-org/ggml)

---

## Overview

UmaLLama is the C++ inference engine of the [llama-ai ecosystem](https://github.com/fewtarius/llama-ai). While upstream `llama.cpp` focuses on raw token execution, UmaLLama optimizes for **end-to-end agentic throughput** on shared-memory AMD APU hardware (18–30K prompt contexts on 5–20 t/s hardware).

- **This repository:** Core C++ inference engine with persistent KV caching, MoE residency paging, fused Vulkan shaders, and APU tuning.
- **Turnkey installation:** The [parent project](https://github.com/fewtarius/llama-ai) provides automated hardware detection, ISA tuning, runner scripts, and benchmarks.

### Persistent SSD-backed KV cache

Conversation state survives server restarts and power cycles. Three-tier
architecture keeps active conversations in RAM, demotes idle ones to disk, and
restores from disk on cold start:

- **Hot** — current session, in RAM. Instant restore within the same conversation.
- **Warm** — previous sessions, same server run. In RAM until memory pressure
  pushes them to cold.
- **Cold** — on disk. Survives server restarts.

Tiers promote and demote automatically based on turn activity. Per-conversation
ring buffers prevent unbounded disk growth. Kernel readahead
(`posix_fadvise` on Linux, `readahead` on macOS) overlaps SSD I/O with CPU work.
Conversation-hash and model-compatibility-hash checks prevent mismatched
checkpoint restoration.

**System prompt cache.** Global cross-conversation cache keyed on the prompt prefix tokens. The first evaluation writes the state; subsequent requests skip prompt eval. Recurrent states are preserved for hybrid MoE/SSM models with `n_past` capped to boundary (default: 8 entries per model, 30-day expiry).

**Hybrid MoE checkpoint restore.** Separates recurrent states from attention cells in hybrid models (Qwen3.5/3.6, Gemma 4, GLM-4.7, DeepSeek MLA). Exposes `llama_memory_seq_rm_attn_only` to clear attention cells without disturbing full-sequence recurrent memory.

### Host-memory prompt cache (`--cache-ram`)

An in-memory ring of serialized KV state blobs for **agentic interleaving** (where an agent makes divergent sub-queries before resuming main context):

- **Multi-slot enabled (`--parallel > 1`):** Accumulates divergent state across idle slots so longest-common-prefix (LCP) matching can hot-swap tasks without full reprocessing.
- **Single-slot efficiency:** In single-slot setups, UmaLLama automatically skips RAM serialization round-trips; the zero-cost VRAM checkpoint ring (`slot.prompt.checkpoints`) handles prefix restores directly.

### MoE expert management

Four subsystems work together to execute MoE models larger than physical RAM:

- **Activation tracking:** Endpoints (`/expert-stats`, `/expert-tracking`) and C API (`llama_expert_stats_get`) provide real-time per-layer activation counts.
- **SSD residency (`--moe-expert-residency`):** Tracks recency and frequency, using `madvise(MADV_WILLNEED)` to keep hot experts paged in RAM while cold weights remain mmap'd on SSD (validated hit rates: 95.5% on Qwen3.6-35B-A3B, 89.5% on GLM-4.7-Flash).
- **Host-RAM offload (`--cpu-moe`):** Directs MoE FFN weights to host RAM via `tensor_buft_overrides` while keeping attention on GPU, allowing 30–90 GB MoE quants to run on 32 GB unified memory.
- **Co-activation tracking:** Persists a per-layer joint-activation matrix (`~/.umallama/coactivation/{model}.json`) to prewarm correlated expert pairs together.

### User isolation

First-class multi-user tenancy: routes checkpoints to a `u/` namespace, enforces per-user concurrency limits with HTTP 429 errors, and implements slot affinity for cache locality (compatible with OpenAI SDK via `extra_body={"llama_user_id": "..."}`).

### Vulkan performance stack (AMD/RADV)

- **DeepSeek-V4 Lightning Indexer:** Custom fused op for compressed-KV attention; dequantizes K into shared memory once per block and iterates Q across heads (verified 108/108 on `test-backend-ops`).
- **DeepSeek-V4 hyper-connection fusion (`DSV4_HC_*`):** Fuses softmax-scale-iterate passes for HC=4 (prefill +16.4%, decode +41.1% on Strix Halo).
- **Quantized-KV FA dequant-once:** Integrated upstream (`dc72703fc`); dequantizes and transposes into contiguous f16 scratch once per layer to avoid redundant dequantizations on UMA memory.
- **APU submit batching:** Defaults to 8 `nodes_per_submit` on UMA devices to avoid the 2s `amdgpu` kernel lockup timeout (100 on discrete GPUs).
- **Subgroup pinning:** Pins 32-wide subgroups for coopmat1 FA where narrowing is free on RDNA3 wave64 hardware.
- **Grouped-GEMM & concat transpose:** Fused-scale epilogue and probe controls for `mul_mat_id`; opt-in tiled dim-0 concat transpose (`GGML_VK_CONCAT_TRANSPOSE=1`).

### CPU ISA detection & custom architectures

- **Automated ISA detection:** The parent project probes hardware features to build with matching AVX-512 flags (`avx512_bf16` on Zen 4, `avx512_vnni` on Sapphire Rapids).
- **DFlash & Laguna-S-2.1:** Generic decoder contract framework (`src/models/dflash.cpp`) and native support for sigmoid-routed MoE Laguna architectures (`src/models/laguna.cpp`).
- **`common::host_available_ram()`:** Cross-platform available-RAM query used for safety gating and automatic memory budget sizing.

---

## Benchmarks

### Strix Halo (Nimo Axis N161)

Radeon 8060S, Vulkan backend. Speedup is prompt eval speedup: cold `prompt_ms`
divided by warm `prompt_ms` (with warm path restoring the prefix from SSD rather
than re-evaluating it). Full data in the
[parent project benchmarks](https://github.com/fewtarius/llama-ai#benchmarking).

| Model | Cold TTFT | Warm TTFT | Speedup | Gen t/s |
|-------|----------:|----------:|--------:|--------:|
| DeepSeek-V4-Flash-0731 UD-IQ3_XXS (256x8.4B MoE, Lightning Indexer) | 97.0s | 0.34s | **282x** | 6.3-17.5 |
| MiniMax-M2.7 Q2_K_XL (256x4.9B MoE) | 60.3s | 3.19s | **19x** | 12.0-17.2 |
| Qwen3-235B-A22B Thinking-2507 IQ2_M (235B-A22B MoE) | 100.5s | 1.04s | **96x** | 10.6-13.6 |
| Qwen3.5-122B-A10B Q5_K_M (122B-A10B MoE, Mamba hybrid) | 52.9s | 0.31s | **171x** | 9.5-10.6 |
| Qwen3.5-122B-A10B UD-Q4_K_XL (122B-A10B MoE, Mamba hybrid) | 53.5s | 0.28s | **189x** | 23.5-32.7 |
| gpt-oss-120b Q8_K_XL (120B MoE) | 25.0s | 0.24s | **104x** | 18.2-19.9 |
| Qwen3-Coder-Next Q8_K_XL (512x2.5B MoE, Mamba hybrid) | 48.5s | 0.22s | **221x** | 18.9-19.9 |
| Qwen3.6-35B-A3B Q8_K_XL (35B-A3B MoE, Mamba hybrid) | 20.5s | 0.14s | **149x** | 14.2-17.5 |
| GLM-4.7-Flash Q8_K_XL (64x2.6B MoE, MLA) | 85.5s | 0.18s | **478x** | 13.9-20.3 |
| gemma-4-26B-A4B Q5_K_M (26B-A4B MoE, sliding window) | 16.7s | 0.14s | **123x** | 14.0-14.7 |
| Qwen3.6-27B Q8_K_XL (27B dense, Mamba hybrid) | 85.3s | 0.36s | **238x** | 4.7-4.8 |
| Laguna-S-2.1 Q4_K_XL (256x4.5B MoE, DFlash target) | 50.0s | 0.48s | **104x** | 11.2-16.9 |
| gpt-oss-20b Q6_K_XL (20B MoE) | 11.1s | 0.10s | **112x** | 22.3-23.8 |

### Ayaneo Flip KB

Radeon 780M, 32GB unified memory, Vulkan backend.

| Model | Cold TTFT | Warm TTFT | Speedup | Gen t/s |
|-------|----------:|----------:|--------:|--------:|
| gemma-4-26B Q5_K_M (26B, dense) | 63.8s | 0.46s | **138.4x** | 14-17 |
| gpt-oss-20b Q6_K_XL (20B, 3.6B active MoE) | 45.9s | 0.24s | **193.5x** | 25-31 |
| Qwen3.6-35B-A3B Q4_K_XL (35B, 3B active MoE) | 70.9s | 0.38s | **187.9x** | 19-24 |

### Real-world CLIO performance (Strix Halo)

Qwen3.6-35B Q8_K_XL, 196K context, 32 threads, Vulkan backend. CLIO evaluates a
project: read files, check git history, run commands, write a final analysis.

| Turn | Prompt tokens | TTFT | Gen t/s |
|------|---------------|------|---------|
| T0 | 21,336 | 26.5s | 36.5 |
| T1 | 1,623 | 2.8s | 37.9 |
| T2 | 5,152 | 8.0s | 37.7 |
| T3 | 4,428 | 7.5s | 38.2 |

**Total: 88 seconds** (32,539 prompt tokens, 1,656 generated). First turn
evaluates the full system prompt cold. Every subsequent turn uses in-memory
checkpoints to restore the cached prefix, evaluating only the new content.
Generation holds steady at 36–38 t/s across all turns.

---

## UmaLLama CLI flags

### SSD cache

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-ssd PATH` | (off) | Enable SSD-backed KV cache |
| `--cache-ssd-checkpoints N` | 64 | Max checkpoints per slot |
| `--cache-ssd-max-cold N` | 0 | Max cold tier checkpoints (0 = unlimited) |
| `--cache-ssd-max-conversations N` | 16 | Max conversation directories |
| `--cache-ssd-hot-ram N` | auto | Hot tier RAM budget in MiB (0 = auto) |
| `--cache-ssd-warm-ram N` | auto | Warm tier RAM budget in MiB (0 = auto) |
| `--cache-ssd-cold-maxsize N` | 0 | Global cap on total cold tier bytes across all conversations in MiB (0 = unlimited). When exceeded, oldest conversations are evicted as whole directories. |
| `--cache-ssd-no-fsync` | disabled | Skip fsync on checkpoint writes for reduced write latency (benchmarking only; env equivalent `LLAMA_ARG_NO_FSYNC=1`) |

### System prompt cache

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-ssd-system-prompts N` | 8 | Max global system prompt entries cached for reuse across conversations |
| `--cache-ssd-system-max-days N` | 30 | Expire system prompt entries unused for N days |

### Host-memory prompt cache

| Flag | Default | Description |
|------|---------|-------------|
| `--cache-ram N` | 8192 | Host-memory prompt cache size in MiB. Only useful with `--parallel > 1`; with a single slot the save+load round-trip is a no-op (skipped automatically). Set to 0 to disable. `-1` for no limit. |
| `--cache-idle-slots` | enabled | Save idle slots to the prompt cache on new task. Requires `--cache-ram > 0`. With `--kv-unified`, idle slots are also cleared to make room for new tasks. |
| `--parallel N` (`-np`) | 1 | Number of server slots. Each slot holds its own KV cache; multiple slots let several conversations run concurrently without blocking each other. UmaLLama's parent launcher (`llama-run.sh`) defaults to 1; raise to 2+ to host multiple agent sessions. Per-slot KV memory cost is `n_ctx * layer_count * head_dim * 2 (K+V) * dtype_bytes`. |

### User isolation

| Flag | Default | Description |
|------|---------|-------------|
| `--max-concurrent-per-user N` | 0 | Per-user slot cap (0 = unlimited) |

When the cap is hit, the server returns HTTP 429:

```json
{
  "error": {
    "code": 429,
    "message": "User 'tenant-42-user-7' has reached the concurrent request limit (2)",
    "type": "rate_limit_error"
  }
}
```

To identify a request, pass `llama_user_id` in the request body. OpenAI SDK
callers use `extra_body={"llama_user_id": "..."}`. Validated to
`^[a-zA-Z0-9\-_]+$` with a 512-char ceiling.

### Vulkan APU/iGPU tuning

| Flag | Default | Description |
|------|---------|-------------|
| `GGML_VK_NODES_PER_SUBMIT` | auto | Override automatic `nodes_per_submit` (APU defaults to 8 to avoid 2s amdgpu timeout; dGPU defaults to 100) |
| `GGML_VK_DISABLE_LIGHTNING_INDEXER` | (unset) | Set to `1` to disable the DeepSeek-V4 Lightning Indexer fused op |
| `GGML_VK_DISABLE_DSV4_HC[_COMB\|_PRE\|_POST]` | (unset) | Set to `1` to disable individual DSV4 hyper-connection fused ops |
| `GGML_VK_CONCAT_TRANSPOSE` | (unset) | Set to `1` to enable the tiled concat_transpose shader for transposed dim-0 concat |
| `GGML_VK_MMID_SCALE_EPILOGUE` | (unset) | Set to `1` to enable fused-scale epilogue in mul_mat_id tile shaders |
| `GGML_VK_MMID_PROBE` | (unset) | Set to `1` to log diagnostic probe information for mul_mat_id tile execution and thread counts |

> **Note on Quantized-KV Flash Attention:**
> The quantized-KV FA dequant-once optimization is merged upstream (commit `dc72703fc` from [ggml-org/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494)) and enabled automatically for q8_0 KV prefill. Legacy manual scratch tuning flags (`GGML_VK_NO_FA_SCRATCH_TRANSPOSE`, `GGML_VK_FA_SCRATCH_SAFETY_MB`, `GGML_VK_FA_SCRATCH_FORCE`) are retired; see [docs/patch-set-status.md](docs/patch-set-status.md) for carry history.

### MoE expert residency / offload

| Flag | Default | Description |
|------|---------|-------------|
| `--moe-expert-residency` / `--no-moe-expert-residency` | disabled | Master switch. Tracks MoE expert activations and uses `madvise` to keep hot experts paged into RAM and cold ones released back to the mmap'd file. Requires `--load-mode mmap` (default). |
| `--moe-resident-per-layer N` | 32 | Max experts kept hot per MoE layer (per-layer LRU size). Must be > 0. |
| `--moe-prewarm-top-k N` | 16 | Experts to prewarm per layer at startup. Set to 0 to disable prewarm. |
| `--moe-residency-debug` `[on\|off]` | off | Linux only. Periodically call `mincore()` on each tracked expert and log the physical residency ratio alongside the software policy state. For development and correctness verification, not production. |
| `--moe-residency-debug-interval N` | 64 | Linux only. Decodes between `mincore()` samples. The `mincore()` call costs O(experts) per sample; tune this to balance observability against overhead. |
| `--cpu-moe` | disabled | Route all MoE FFN expert tensors to host RAM via `tensor_buft_overrides` (LLM_FFN_EXPS_REGEX). Combine with `--moe-expert-residency` on memory-constrained hardware to mmap the full weights from SSD while keeping only the LRU-resident expert subset in RAM. |
| `--n-cpu-moe N` | 0 | Route only the first N MoE layers' expert weights to host RAM (0 = use `--cpu-moe` value). |

Environment-variable equivalents: `LLAMA_ARG_MOE_EXPERT_RESIDENCY`,
`LLAMA_ARG_MOE_RESIDENT_PER_LAYER`, `LLAMA_ARG_MOE_PREWARM_TOP_K`,
`LLAMA_ARG_MOE_RESIDENCY_DEBUG`, `LLAMA_ARG_MOE_RESIDENCY_DEBUG_INTERVAL`,
`LLAMA_ARG_CPU_MOE`, `LLAMA_ARG_N_CPU_MOE`.

Hit rate and latency data per model is in
[`docs/moe-expert-residency.md`](docs/moe-expert-residency.md).

---

## MoE expert tracking API

### `GET /expert-stats`

Per-layer expert activation counts, frequencies, and token counts:

```json
{
  "n_expert": 256,
  "n_expert_used": 8,
  "total_tokens": 1500,
  "tracking_enabled": true,
  "layers": [
    {
      "layer": 0,
      "activations": [
        {"expert": 42, "count": 150, "frequency": 0.0125},
        {"expert": 7,  "count": 148, "frequency": 0.0123}
      ]
    }
  ]
}
```

### `POST /expert-tracking`

Enable/disable tracking and optionally reset counters:

```json
{"enabled": true, "reset": true}
```

### C API

```c
llama_expert_tracking_enable(ctx, true);

// Per-layer stats (returns 0 on success, -1 if tracking disabled)
struct llama_expert_stats stats;
llama_expert_stats_get(ctx, /*layer=*/0, &stats);

// Reset all counters
llama_expert_stats_reset(ctx);

// Model-level constants
int32_t n_expert      = llama_model_n_expert(model);
int32_t n_expert_used = llama_model_n_expert_used(model);
```

The `llama_expert_stats` struct exposes `activations[]` (per-expert count and
frequency) and `token_count` for the layer.

### MoE expert residency C API

```c
// Configure and enable residency
struct llama_moe_residency_config cfg = llama_moe_residency_config_default();
cfg.max_resident_per_layer = 32;  // max experts kept hot per layer
cfg.prewarm_top_k   = 16;      // experts to prewarm at startup
llama_moe_residency_enable(ctx, &cfg);

// Read stats
struct llama_moe_residency_stats rs;
llama_moe_residency_stats_get(ctx, &rs);

// Disable (releases all MADV_WILLNEED pages)
llama_moe_residency_disable(ctx);
```

---

## Building UmaLLama

For end-to-end install (runner scripts, GPU detection, benchmark harness, GTT
configuration), use the [parent project](https://github.com/fewtarius/llama-ai):

```bash
git clone --recurse-submodules https://github.com/fewtarius/llama-ai.git
cd llama-ai
./scripts/rebuild.sh
```

To build just the inference engine from this repo:

```bash
# Vulkan backend on Linux AMD APU / GPU
cmake -B build -DGGML_VULKAN=ON
cmake --build build --config Release -j$(nproc)
```

All standard `llama.cpp` build options are supported. UmaLLama adds no new
build flags — everything is runtime config via the CLI flags above.

---

## Documentation

- [Parent project (fewtarius/llama-ai)](https://github.com/fewtarius/llama-ai) — runner
  scripts, GPU detection, benchmarks, end-to-end install
- [CLIO](https://github.com/SyntheticAutonomicMind/CLIO) — agentic AI client
  optimized for UmaLLama's persistent cache
- [User isolation design](docs/development/user-isolation-design.md) —
  architecture for the `user_id` / `u/` namespace / `--max-concurrent-per-user`
  features
- [MoE expert residency](docs/moe-expert-residency.md) — `--moe-expert-residency`
  / `--cpu-moe` mechanism, hit rate measurements, C API
- [Upstream llama.cpp](https://github.com/ggml-org/llama.cpp) — the base project
  we fork from

---

## Relationship with upstream llama.cpp

UmaLLama tracks upstream `llama.cpp` master through periodic merges while carrying optimizations tuned for shared-memory hardware:

- **Upstream sync:** Merged periodically; features landing upstream are rebased onto stock implementations.
- **Carried optimizations:** External performance PRs and fork carries (Nathanw1014 flash-attention shaders, RDNA3.5 tuning) are maintained until upstreamed.
- **UmaLLama originals:** Persistent KV cache, MoE expert residency, Lightning Indexer, DFlash/Laguna, user isolation, and Vulkan APU tuning.

See [AGENTS.md](AGENTS.md) and [docs/patch-set-status.md](docs/patch-set-status.md) for the active carry table and conventions.

---

## License

**UmaLLama source code:** MIT (same as upstream `llama.cpp`, see
[LICENSE](LICENSE), Copyright (c) 2023-2026 The ggml authors). Core UmaLLama
additions — the persistent KV cache, user isolation, Lightning Indexer, Vulkan
shader work, DFlash/Laguna model support, and the `common::host_available_ram()`
utility — are released under MIT terms. The MoE expert residency and
co-activation tracking subsystems (`src/llama-moe-residency.*`,
`src/llama-moe-coact.*`) are licensed under GPL-3.0-or-later
(Copyright (c) 2026 fewtarius).

**llama-ai parent project:** GPL-3.0-or-later for source, CC-BY-NC-SA-4.0 for
documentation (see the [parent project LICENSE](https://github.com/fewtarius/llama-ai/blob/main/LICENSE)).

**ROCm components:** Carry AMD's license (see `licenses/roc*` in the ROCm SDK
bundle downloaded by `scripts/rebuild.sh`).
