# Ecosystem Landscape & Upstream PR Alignment Strategy

**Date:** 2026-10-12  
**Purpose:** Technical survey of the Strix Halo / local inference ecosystem and UmaLLama's core architectural principle: **staying as close to upstream `llama.cpp` as possible, maintaining all improvements in a modular, PR-ready state.**

---

## 1. Guiding Philosophy: Upstream Proximity & PR Viability

UmaLLama is not intended to be a permanent, hard divergence from [llama.cpp](https://github.com/ggml-org/llama.cpp). The primary architectural contract is:

1. **Upstream Alignment First:** Any feature, fix, or optimization should be developed and maintained as if it were a candidate pull request for upstream `llama.cpp`.
2. **Minimal Invasiveness:** Core tensor computation graphs, memory pools, and API boundaries in `ggml` and `src/llama.cpp` must remain untouched or modified only via clean, minimal hooks.
3. **Decoupled Modular Subsystems:** Features specific to advanced agentic workflows (such as persistent SSD KV caching and MoE expert paging) must reside in separate modules (`common/` or dedicated source files) so that upstream `git merge upstream/master` operations remain clean and conflict-free.
4. **Zero Lock-In:** Strictly adhere to the universal GGUF model format. Never adopt proprietary weight formats or closed binary distributions.

---

## 2. Comparative Survey: The Strix Halo / Local Inference Landscape

The emergence of unified memory APUs (such as AMD Strix Halo, Ryzen AI Max+ 395 with Radeon 8060S / `gfx1151`, up to 128 GiB unified memory) has generated multiple competing approaches to local inference:

```
[Universal & Upstream-Aligned] ──────────────────────────► [Hard Divergence / Proprietary]

     strix-llama            UmaLLama               Strata                 Gufo                Halogen
 (Community patchset)   (This Project)       (Niko1221/Strata)      (gufo-org/gufo)     (peonist-ai/hgn)
          │                    │                     │                     │                    │
   • llama.cpp patch    • Upstream fork       • ggml/llama fork      • Clean-sheet MIT    • Closed binary
   • Universal GGUF     • Universal GGUF      • Curated GGUF         • Curated zoo        • Proprietary .hgn
   • Vulkan/ROCm FA     • Vulkan (RADV)       • VRAM MoE cache       • PagedAttention     • Hardcoded ASM
   • Stateless          • 3-Tier SSD Cache    • CPU ∥ GPU execution  • All-in-one media   • 1800+ t/s prefill
   • Zero divergence    • MoE kernel paging   • Discrete PCIe focus  • Multi-daemon sub   • Locked hardware
```

### Detailed Engine Profiles

#### A. `strix-llama` / `strix-halo-llamacpp`
- **What it is:** A community-maintained patchset and containerized toolbox for running `llama.cpp` on `gfx1151` (Strix Halo).
- **Architecture:** Direct patch on `llama.cpp` master. Applies targeted Vulkan subgroup controls (`VK_EXT_subgroup_size_control`), 32-wide wave pinning, cooperative matrix (`VK_KHR_cooperative_matrix`) alignment, and quantized KV cache (`-ctk q8_0 -ctv q8_0`) tuning.
- **Upstream Viability:** High. Most of its changes are direct candidates for upstream `ggml-vulkan.cpp` or ROCm pull requests.
- **Limitations:** Treats the engine as a stateless server. Caches are volatile and wiped on process restart; multi-turn prompt evaluation must be redone from scratch.
- **UmaLLama Relationship:** UmaLLama incorporates these low-level Vulkan and hardware tuning fixes, but layers on an agentic memory hierarchy.

#### B. Gufo (`gufo-org/gufo`)
- **What it is:** A clean-sheet, vertically integrated local inference engine developed specifically for Strix Halo workstations.
- **Architecture:** Completely independent codebase (MIT). Combines PagedAttention and continuous batching inspired by vLLM with an all-in-one multimodal server daemon supporting text (Qwen 3.8, DeepSeek-V4), audio (TTS/ASR), and image generation under a single runtime.
- **Upstream Viability:** Non-existent. Gufo is not a fork of `llama.cpp` and cannot contribute directly to or pull from the upstream repository.
- **Limitations:** Narrow model support (curated models only), lacks the massive 130+ architecture catalog and community quantization ecosystem of GGUF.
- **UmaLLama Relationship:** Gufo demonstrates the maximum throughput of in-memory continuous batching, but sacrifices the ecosystem breadth, portability, and simplicity of `llama.cpp`.

#### C. Strata (`Niko1221/Strata` / `stratallm.org`)
- **What it is:** A specialized local inference engine designed to run massive 125B+ MoE models (specifically Qwen3.8-Flash-Next with 24,000+ experts) on consumer GPUs with only 12–16 GiB VRAM.
- **Architecture:** Forked from `ggml`/`llama.cpp` primitives. Redesigns the memory model by treating GPU VRAM as a dynamic LRU/predictive cache for hot experts while holding the wider model in system RAM and memory-mapping auxiliary lookup tables from SSD. Executes non-resident experts on CPU concurrently with GPU attention.
- **Upstream Viability:** Low to moderate. The core scheduling and memory allocation logic heavily diverges from upstream `llama.cpp` execution graphs.
- **Limitations:** Built primarily to overcome the discrete PCIe transfer barrier (16–32 GB/s) between system RAM and discrete VRAM.
- **UmaLLama Relationship:** On Strix Halo's unified memory architecture (UMA), CPU and GPU share the same ~270 GB/s physical memory pool, eliminating the need for PCIe streaming. UmaLLama achieves MoE residency using the Linux kernel's zero-copy `madvise` virtual memory subsystem (`src/llama-moe-residency.cpp`) instead of complex PCIe data movement.

#### D. Halogen (`peonist-ai/halogen-flash-server`)
- **What it is:** A proprietary, closed-source binary inference server distributed in Docker containers specifically for Qwen-Flash on Strix Halo.
- **Architecture:** Bare-metal single-silicon optimization. Discards all hardware abstraction and the standard GGUF format, using custom `.hgn` binary weight containers aligned to RDNA 3.5 memory controller burst sizes.
- **Upstream Viability:** Zero. Entirely closed and incompatible with open standards.
- **Limitations:** Total platform and model lock-in. Cannot run general open-source models, arbitrary GGUFs, or custom architectures.
- **UmaLLama Relationship:** Halogen demonstrates the theoretical ceiling of hardware-specific memory saturation (1,500–1,800+ t/s prefill), but UmaLLama rejects its proprietary, locked-down architecture in favor of upstream-compatible open source.

---

## 3. UmaLLama's Upstream PR Taxonomy

To realize the goal of being "equivalent to something that is PR'able at any given time," UmaLLama categorizes all additions into two distinct tiers:

### Tier 1: Direct Upstream PR Candidates
These are self-contained bugfixes and optimizations written strictly within upstream conventions, intended to be submitted to `ggml-org/llama.cpp`:

| Component | Files | Description | PR Status / Target |
| :--- | :--- | :--- | :--- |
| **Vulkan Subgroup Init-Order Fix** | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Moves feature probe assignments (`lightning_indexer`, `dsv4_hc`) after `computeFullSubgroups` verification so pipelines compile on RADV. | Ready for upstream PR |
| **Coopmat1 P-Fragment Hoist** | `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` | Hoists P-fragment loads out of the `hsv_tile` loop for FlashAttention (+5% speedup on APUs). | Candidate PR |
| **Coopmat1 Query-Major `Psh`** | `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp` | Stores `Psh` in query-major order so the GEMM2 A load vectorizes cleanly. | Candidate PR |
| **32-Wide Subgroup Pinning** | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Pins `required_subgroup_size=32` where narrowing is free on RDNA wave64 devices. | Candidate PR |
| **APU `nodes_per_submit` Tuning** | `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Lowers default command-buffer batching to 8 on UMA devices to prevent command processor pipeline stalls. | Candidate PR |
| **Host Available RAM Query** | `common/host-ram.h`, `common/host-ram.cpp` | Clean, cross-platform physical available RAM query utility (`host_available_ram()`). | Clean `common/` PR |
| **Model Architectures** | `src/models/laguna.cpp`, `src/models/dflash.cpp` | Clean implementations of Laguna-S-2.1 and DFlash speculative drafting contracts. | Upstream model PR |

### Tier 2: Modular Out-of-Tree Extensions
These subsystems provide UmaLLama's unique agentic throughput features. They are designed to be strictly modular so they do not interfere with upstream rebases:

| Subsystem | Location | Upstream Decoupling Strategy |
| :--- | :--- | :--- |
| **Persistent 3-Tier SSD KV Cache** | `common/kv-ssd-cache.cpp`<br>`common/kv-ssd-system-cache.cpp`<br>`tools/server/server-context-ssd-manager.cpp` | Resides entirely in `common/` and `tools/server/`. Uses public C API (`llama_state_get_data`, `llama_state_set_data`) without modifying internal llama KV cache structs. |
| **MoE `madvise` Expert Residency** | `src/llama-moe-residency.cpp`<br>`src/llama-moe-coact.cpp` | Operates on the mmap virtual address space of the model file. Hooks into `llama-context.cpp` via a single non-invasive callback (`llama_moe_residency_touch`) that is trivially gated or stubbed out. |
| **User & Slot Isolation** | `include/llama.h`, `tools/server/` | Exposes optional `user_id` and slot affinity tokens at the server request layer. |

---

## 4. Upstream Synchronization & Maintenance Rules

1. **Regular Sync Cadence:** Rebase or merge from `upstream/master` periodically (weekly or after significant upstream releases).
2. **Immediate Carry Retirement:** When an upstream PR merges that supersedes an UmaLLama carry (as tracked in [`docs/patch-set-status.md`](patch-set-status.md)), drop the local carry immediately in favor of upstream's implementation.
3. **Format & Code Style Integrity:**
   - Follow upstream `llama.cpp` formatting: 4 spaces, no tabs, LF endings, C++17 for llama engine, C11 for `ggml`.
   - Avoid modern C++ template magic or heavy STL dependencies in core inference paths.
   - Run clang-format check before committing:
     ```bash
     git diff --name-only upstream/master | grep -E '\.(c|cpp|h|hpp)$' | xargs clang-format --dry-run -Werror
     ```
4. **Isolate Feature Flags:** Gating controls for experimental or platform-specific optimizations must use environment variables (`GGML_VK_*` or `LLAMA_ARG_*`) so baseline behavior matches upstream defaults.
