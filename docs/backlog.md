# UmaLLama Feature Backlog & Upstream PR Roadmap

**Date:** 2026-10-12  
**Purpose:** Prioritized backlog of optimizations and features adapted from the Strix Halo and local inference ecosystem (`strix-llama`, `Gufo`, `Strata`), designed to maintain **strict upstream `llama.cpp` PR readiness and modular decoupling**.

---

## Guiding Principles for Backlog Items

1. **Candidate-PR Ready:** Every item must be implemented with clean, upstream-compatible code style, minimal diff footprint, and no invasive changes to core GGML tensor graphs.
2. **Modular Decoupling:** Enhancements specific to agentic workflows or storage tiering must live in `common/` or behind non-invasive hooks.
3. **Hardware-Focused Leverage:** Target AMD APU unified memory architecture (Strix Halo / RDNA 3.5 / `gfx1151`) where high bandwidth and shared CPU/GPU address space yield disproportionate gains.

---

## Completed Backlog Items

### Item 1: Native BF16 KV-Cache FlashAttention on RDNA 3/3.5/4
* **Inspiration:** `strix-llama` / Upstream PR #26856
* **Category:** Tier 1 (Direct Upstream PR Candidate)
* **Status:** **Completed** (commit `517ebcaf5`)
* **Implemented Changes:**
  * Enabled native BF16 K/V reading in the FlashAttention tile kernel with FP32 accumulation and packed BF16 PV paths on RDNA3/RDNA3.5/RDNA4 architectures (`ggml/src/ggml-cuda/common.cuh`, `fattn.cu`, `fattn-tile.cu`, `fattn-tile.cuh`).
  * Utilizes `v_dot2_f32_bf16` and `v_perm_b32` on `__gfx1100__` through `__gfx1201__` (including Strix Halo `__gfx1150__`/`__gfx1151__`).
  * Added head-112 template instantiations and expanded operator test coverage in `tests/test-backend-ops.cpp`.
* **Verification:** Clean build and 100% pass on FlashAttention operator test suite (`7732/7732 tests passed`).

---

### Item 2: Atomic Agentic Cancellation & Slot State Rollback
* **Inspiration:** `Gufo` (`gufo-org/gufo`)
* **Category:** Tier 1 / Server Layer (Candidate Upstream PR)
* **Status:** **Completed** (commit `5e18a39e2`)
* **Implemented Changes:**
  * Added prompt-boundary snapshot checkpointing and `rollback_slot_on_cancel()` in `tools/server/server-context.cpp`.
  * Integrated with `tools/server/server-queue.cpp` to abort streaming promptly upon client disconnect/cancellation (`SERVER_TASK_TYPE_CANCEL`).
  * Rolls back in-memory/SSD checkpoint indices or trims partial tokens via `mem.seq_rm` when supported (`ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART`).
  * Added automated unit test in `tools/server/tests/unit/test_slot_save.py` (`test_slot_cancel_atomic_rollback`).
* **Verification:** Built cleanly and passed server unit test suite with 100% prompt cache hit on post-abort resumption ($0\text{ ms}$ TTFT).

---

## Active Backlog Items

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
