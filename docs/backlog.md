# UmaLLama Feature Backlog & Upstream PR Roadmap

**Date:** 2026-10-12  
**Purpose:** Prioritized backlog of optimizations and features adapted from the Strix Halo and local inference ecosystem (`strix-llama`, `Gufo`, `Strata`), designed to maintain **strict upstream `llama.cpp` PR readiness and modular decoupling**.

---

## Guiding Principles for Backlog Items

1. **Candidate-PR Ready:** Every item must be implemented with clean, upstream-compatible code style, minimal diff footprint, and no invasive changes to core GGML tensor graphs.
2. **Modular Decoupling:** Enhancements specific to agentic workflows or storage tiering must live in `common/` or behind non-invasive hooks.
3. **Hardware-Focused Leverage:** Target AMD APU unified memory architecture (Strix Halo / RDNA 3.5 / `gfx1151`) where high bandwidth and shared CPU/GPU address space yield disproportionate gains.

---

## Active Backlog Items

### Item 1: Native BF16 KV-Cache FlashAttention on RDNA 3.5
* **Inspiration:** `strix-llama` / Upstream PR #26856
* **Category:** Tier 1 (Direct Upstream PR Candidate)
* **Target Files:**
  * `ggml/src/ggml-vulkan/ggml-vulkan.cpp`
  * `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn.comp`
* **Problem:** In stock `llama.cpp`, requesting a BF16 KV cache (`-ctk bf16 -ctv bf16`) on AMD hardware often trips a fallback conversion path (`BF16 -> F16`), incurring a ~15–20% decode penalty.
* **Proposed Solution:** Enable native BF16 arithmetic in the Vulkan FlashAttention compute shader for devices advertising `VK_KHR_shader_float16_int8` and native BF16 capabilities, eliminating unnecessary dequantization passes.
* **Expected Benefit:** +10% to +15% prompt prefill throughput for native BF16 models (Qwen 2.5/3.5, DeepSeek) on Strix Halo.
* **Upstream PR Viability:** **10/10** — Minimal shader diff, pure performance bugfix, zero architecture divergence.

---

### Item 2: Atomic Agentic Cancellation & Slot State Rollback
* **Inspiration:** `Gufo` (`gufo-org/gufo`)
* **Category:** Tier 1 / Server Layer (Candidate Upstream PR)
* **Target Files:**
  * `tools/server/server.cpp`
  * `tools/server/server-context-ssd-manager.cpp`
* **Problem:** In multi-turn coding agent loops (e.g. Pi, CLIO), generations are frequently canceled mid-stream (user interrupt or agent tool call). An aborted slot often leaves partial tokens in the KV ring or forces an expensive prompt re-evaluation on the next turn.
* **Proposed Solution:** Implement atomic slot cancellation that cleanly rewinds the KV cache state to the last complete turn boundary. For UmaLLama, wire this into `server-context-ssd-manager.cpp` to automatically roll back to the last valid on-disk SSD checkpoint.
* **Expected Benefit:** Zero wasted prompt evaluations ($0\text{ ms}$ TTFT resumption after interrupt) and immunity to KV slot corruption during interactive agent steering.
* **Upstream PR Viability:** **9/10** — Self-contained server enhancement that significantly improves agent harness reliability.

---

### Item 3: Dynamic Context-Thresholded KV Cache Quantization
* **Inspiration:** `Strata` (`Niko1221/Strata`)
* **Category:** Tier 1 / Memory Management (Candidate Upstream PR)
* **Target Files:**
  * `src/llama-kv-cache.cpp`
  * `include/llama.h`
  * `common/arg.cpp`
* **Problem:** Stock `llama.cpp` requires static KV quantization upfront for the entire lifetime of the process (`-ctk q8_0` or `-ctk f16`). Short prompts suffer unnecessary quantization noise, while deep prompts (>32k tokens) exhaust memory bandwidth if unquantized.
* **Proposed Solution:** Introduce an optional threshold flag (e.g., `--kv-quant-threshold <tokens>`) where the KV cache remains uncompressed (FP16/BF16) for short contexts ($\le 4\text{k}$ tokens), automatically compressing older KV blocks to Q8_0 or Q4_0 once context depth exceeds the threshold.
* **Expected Benefit:** Preserves full floating-point precision on short system instructions and reasoning prefixes, while protecting memory bandwidth on long agent transcripts.
* **Upstream PR Viability:** **8/10** — Clean addition to the KV cache manager; highly requested feature in upstream discussions.

---

### Item 4: Asynchronous MoE Expert Overlap (CPU $\parallel$ GPU)
* **Inspiration:** `Strata` (`Niko1221/Strata`)
* **Category:** Tier 2 (Modular Out-of-Tree Extension)
* **Target Files:**
  * `src/llama-moe-residency.cpp`
  * `src/llama-context.cpp`
* **Problem:** When an MoE model exceeds physical RAM and a token activates a non-resident expert, GPU execution bubbles while the kernel pages the expert into physical memory.
* **Proposed Solution:** Overlap the execution of non-resident experts on CPU using AVX-512 while the GPU processes dense attention layers and resident experts, accumulating outputs at the layer barrier.
* **Expected Benefit:** Hides paging latency during cold expert routing without stalling the GPU compute pipeline.
* **Upstream PR Viability:** **7/10** — Maintained in UmaLLama's modular `llama-moe-residency.cpp` subsystem with minimal non-invasive hooks in `llama-context.cpp`.
