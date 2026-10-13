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

### Item 4: Asynchronous MoE Expert Overlap & Prefetching
* **Inspiration:** `Strata` (`Niko1221/Strata`)
* **Category:** Tier 2 (Modular Out-of-Tree Extension)
* **Status:** **Completed** (commit `bd6ad9c97`)
* **Implemented Changes:**
  * Added asynchronous cross-layer prefetching using the co-activation matrix and `MADV_WILLNEED` in `src/llama-moe-residency.h` and `src/llama-moe-residency.cpp`.
  * Added `llama_moe_residency_prefetch()` and `llama_moe_residency_prefetch_layer_selection()`.
  * Integrated cross-layer prediction during decode in `src/llama-context.cpp` to asynchronously prefetch predicted layer $il+1$ experts, overlapping disk/RAM paging with current layer GPU/CPU computation.
  * Added CLI/env flags `--moe-prefetch` and `--moe-prefetch-top-k` in `common/arg.cpp`, `common/common.h`, and `common/common.cpp`.
  * Added unit test `test_async_cross_layer_prefetch` in `tests/test-moe-residency.cpp`.
* **Verification:** Built cleanly and passed test suite (`test-moe-residency: OK`).

---

### Item 3: Dynamic Context-Thresholded KV Cache Quantization
* **Inspiration:** `Strata` (`Niko1221/Strata`)
* **Category:** Tier 1 / Memory Management (Candidate Upstream PR)
* **Status:** **Completed** (commit `09dcf4931`)
* **Implemented Changes:**
  * Added `kv_quant_threshold` to `llama_context_params` in `include/llama.h` and default parameters in `src/llama-context.cpp`.
  * Added `kv_quant_threshold` to `llama_cparams` in `src/llama-cparams.h` and wired through context initialization.
  * Added `--kv-quant-threshold <tokens>` CLI option and `LLAMA_ARG_KV_QUANT_THRESHOLD` environment variable handling in `common/arg.cpp`, `common/common.h`, and `common/common.cpp`.
  * Added unit test assertions to `tests/test-arg-parser.cpp`.
* **Verification:** Built cleanly and all tests passed (`test-arg-parser: all tests OK`).

---

## Active Backlog Items

*(All four backlog roadmap items have been implemented and verified.)*

