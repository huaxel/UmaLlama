# Patch-set status

UmaLLama diverges from `upstream/master` by carrying third-party work. Re-evaluate this table when upstream merges or upstream PRs close — row status changes weekly.

> **2026-10-13 audit pass:** a code-level audit of the table rows revealed drift in
> several "Dormant / not wired" entries. The q4/q5 transpose pipelines ARE registered
> (the upstream q8_0-only claim was wrong); `concat_transpose.comp` IS wired default-ON
> (the carry is now indistinguishable from upstream's); the f16 contiguize shader IS
> registered and reachable via `GGML_VK_FA_KV_CONTIG=1`; the `lightning_indexer_cm*`
> shaders ARE registered, just not dispatched. Rows 2, 3, 8, 9, 17 below were
> corrected. The scratch-transpose env vars previously listed in row 1
> (`GGML_VK_NO_FA_SCRATCH_TRANSPOSE`, `GGML_VK_FA_SCRATCH_SAFETY_MB`,
> `GGML_VK_FA_SCRATCH_FORCE`) are not in the canonical build.

## 2026-10-13 sync (`upstream/master` 77c4540ce)

29 commits merged; the only fork-specific files that needed conflict resolution were
`tests/CMakeLists.txt` (replaced the 3 deleted test files with the new
`tests/llama-context/` + `test-llama-context.cpp` consolidation; kept our
`test-seq-rm-attn-only.cpp` for the issue #8 regression, dropped the two
`test-recurrent-state-rollback` invocations against `deepseek4-moe` and
`kimi-k3-moe` as they are subsumed by the new `test-llama-context --models ...`).

- **`b9f3599fb` llama: do not map foreign ranges** (#30405) — orthogonal to the MoE
  residency subsystem. The fork's `src/llama-moe-residency.cpp` operates on the model's
  `mmap` ranges, not on the backend buffer layer. The new per-context, per-range buffer
  split is complementary: it stops the GPU buffer from pinning the full file
  (27 GiB on Qwen3.8-Flash-Next IQ4_NL cold cache, 475 → 788 t/s on pp4096), while
  our residency layer decides which pages to keep resident via `madvise`. Both layers
  apply; verify on the production Qwen3.6-35B-A3B-MTP before claiming a combined
  speedup.
- **`f441d2d9d` model: load YaRN beta metadata for laguna** (#30331) — touches our
  `src/models/laguna.cpp` (a 3-line addition at the top of `load_arch_hparams`).
  Trivial rebase.
- **`a17e94d22` tests: consolidate context state tests** (#30023) — collapses
  `test-save-load-state`, `test-recurrent-state-rollback`, and
  `test-state-restore-fragmented` into a single `test-llama-context` matrix. Pre-existing
  test bug surfaced: `case_rollback` asserts `nmse == 0` on architectures that do not
  satisfy the in-file TODO precondition (rollback is only bit-exact after a ubatch
  with more than `n_rs_seq` tokens). The `deepseek4` dummy fixture hits this and fails
  with `nmse 0.0` (an exact token mismatch, not numerical noise). Fix-forward: added
  `arch_rollback_is_inexact()` per-arch skip helper, mirroring the existing
  `arch_reserves_single_seq()` pattern. Upstream PR #30473 opened (auto-assigned to
  ggerganov).
- **`cb55ffe6e` graph: skip the LM head for pooled embeddings** (#30445),
  **`988015788` graph: avoid reallocations in the "no cache" path** (#30439),
  **`ed019cd8b` llama: skip logits buffer allocation** (#30255),
  **`ace20f679` graph: pad the KQ bias** (#30446) — graph-side refactor. Auto-merged
  into `src/llama-context.{cpp,h}`, `src/llama-graph.{cpp,h}`, `include/llama.h`. The
  fork's expert-activation tracking hook in `llama-context.cpp` survives unchanged.

Carry rows unchanged. `Vulkan APU nodes_per_submit auto-lower` row's "Defaults to 8 on
UMA" claim was re-verified by inspection — the canonical build still hardcodes 100 in
`ggml-vulkan.cpp` and the APU-default-8 logic is not present. Mark for retirement on
the next sync if the table is not refreshed.

## 2026-10-12 sync (`upstream/master` 8bb3cf0d2)

- Vulkan `mul_mat_id` tiled prepass (#29998): upstream extended hoisting and tiling into
  `count_experts.comp`, emitting `data_row_map` directly on GPU. The legacy row-list/hoisting
  prepass (`pipeline_mmid_row_lists`) was dropped as superseded. UmaLLama's `fused_scale`
  binding (binding 5), `fusion_flags`, and `GGML_VK_MMID_PROBE` were rebased onto the new tile mapping.
- Slot context checkpoints appendix: upstream natively saves and restores slot context
  checkpoints within the slot file appendix. UmaLLama retains its sidecar `.ckpt` and tip
  synthesis fallback when the appendix is empty.
- Speculative decoding position & embeddings: adapted to upstream's `batch.set_embd_state()`.
- Chat PEG parser & streaming: preserved UmaLLama's malformed DSML sanitization and salvage
  fallback across upstream's PEG parser refactoring.

## 2026-10-08 sync (`upstream/master` 9c2e0e491)

Dropped as superseded (no table rows — none existed for them):

- qwen4exp MTP loader/graph carry (bool-ctor base, hand-rolled `block_count`/`ratios`/
  converter MTP hack): upstream now ships MTP support end to end (`no_build` base,
  `_MTP_EXTRA` converter mapping, `load_block` trunk/mtp flags). Fork keeps only its
  MTP-only-file fallback (`trunk_flags = TENSOR_NOT_REQUIRED` when the trunk is absent).
- QSA decode-gather carry (`build_attn_qsa_gather`, `qsa_inps`): dead declarations
  referencing the removed `llm_graph_input_qsa` type; upstream's k-pool indexer covers
  the path. Removed.
- Recurrent snapshot shift-register (`build_rs` caller): uncompilable against upstream's
  redesigned snapshot inputs (`s_copy_tail` + callee-side extra-state copy). Dropped;
  single-sequence rollback re-validated bit-exact post-merge (Qwen3.5-0.8B warm append).
  Note: `test-recurrent-state-rollback` multi-seq split-replay subtest fails on pristine
  upstream 9c2e0e491 too (earlier, at full-logits position 6) — upstream breakage, not a
  fork regression.
- Doc drift found during the sync: `GGML_VK_NO_FA_SCRATCH_TRANSPOSE`,
  `GGML_VK_FA_SCRATCH_SAFETY_MB`, `GGML_VK_FA_SCRATCH_FORCE` are named below but no
  code reads them; concat-transpose dispatch is wired default-ON (commit 5d5b67444),
  not unwired as tabled. Rows need a refresh pass.

**Convention:**
- "Merged upstream" — drop our copy on the next upstream merge.
- "Not upstreamed" — keep carrying; re-check upstream status each merge.
- "Upstream added" — upstream has the same feature; ours differs in tuning/gating. Keep both until the differences can be rebased onto the upstream version.

| Carry | Source | Upstream status | UmaLLama-specific additions |
|-------|--------|-----------------|------------------------------|
| Quantized-KV FA prefill dequant (Vulkan) | [Nathanw1014/llama.cpp#25494](https://github.com/ggml-org/llama.cpp/pull/25494) | **Merged upstream** as `dc72703fc` | Retains the UmaLLama host-RAM gate via `common::host_available_ram()` and printf-style warning. The scratch env vars (`GGML_VK_NO_FA_SCRATCH_TRANSPOSE`, `GGML_VK_FA_SCRATCH_SAFETY_MB`, `GGML_VK_FA_SCRATCH_FORCE`) listed in earlier versions of this row are not in the canonical build. The f16 contiguize and concat-transpose variants below are live |
| FA dequant-once to q4_0/q4_1/q5_0/q5_1 KV | Nathanw1014 carry | **Merged upstream** as `dc72703fc` | All five `dequant_*_transpose` pipelines (q4_0/q4_1/q5_0/q5_1/q8_0) are registered and reachable via the quant-KV FA path. The f16 variant is wired as well (see next row) |
| FA contiguize strided f16 KV | Nathanw1014 carry | **Merged upstream** as `dc72703fc` | `dequant_f16_transpose.comp` is registered and reachable when `GGML_VK_FA_KV_CONTIG=1` and the K/V rows are strided (prefill only, `neq1 >= 64`). Default off: source-indexing bug fixed 2026-09-29 but perf measurement shows a net regression (+12..+41% on Strix Halo) since the upstream FA dispatch refactor removed the strided-read penalty it was built to dodge. See `RDNA3_NOTES.md` |
| Coopmat1 FA P-fragment hoist | Nathanw1014 carry | Not upstreamed | Hoists the P-fragment load out of the `hsv_tile` loop. Measured +5% on Qwen3.6-35B-A3B prefill, Strix Halo |
| Coopmat1 FA Psh query-major | Nathanw1014 carry | Not upstreamed | Stores `Psh` query-major so the GEMM2 A load vectorizes |
| 32-wide subgroup pinning (coopmat1 FA) | Nathanw1014 carry | Not upstreamed | Pins `required_subgroup_size=32` where narrowing is free on RDNA3 wave64 |
| Bound command buffers by memory traffic | Nathanw1014 carry | **Dormant / not wired** | The canonical build still uses a fixed 100-node default; no memory-traffic bound is active |
| Concat transpose shader | Nathanw1014 carry | **Merged upstream** as `dc72703fc` | `concat_transpose.comp` is generated, registered (`pipeline_concat_transpose_i32`), dispatched, and **default ON** (gated by `GGML_VK_CONCAT_TRANSPOSE`; `=0` disables). Carried since `404732f8c`; rewired default-ON in `5d5b67444` |
| MMID row-list prepass | Nathanw1014 carry | **Dormant / partial** | `mmid_row_lists.comp` is in the tree and in the shader gen, but no `pipeline_mmid_row_lists` is created and no runtime dispatch site is registered. The per-pipeline `supports_row_lists` capability gate is wired (see `ggml_vk_create_pipeline(... bool supports_row_lists ...)`) and threads through to the per-row mmid shader readers. The legacy prepass remains as a future toggle: a fully-wired re-introduction would need the dispatch + a per-row-list shape gate |
| MMID f16-B probe | Nathanw1014 carry | **Dormant / not wired** | No `GGML_VK_MMID_F16B` control or active probe remains in the canonical build |
| MMID wave32 probe | Nathanw1014 carry | **Dormant / not wired** | No `GGML_VK_MMID_WAVE32` control or active probe remains in the canonical build. (`GGML_VK_FA_WAVE32` exists but is for FA, not MMID) |
| MMID scale cache (q5_K, q4_K, superblock-amortized) | Nathanw1014 carry | **Dormant / not wired** | No active shared-memory scale-cache implementation remains in the canonical MMID path |
| FA MMQ dot product fp32 scaling | Nathanw1014 carry | Not upstreamed | Scales the MMQ dot product in fp32 before narrowing for numerical stability |
| FA split-K reduce shader | Nathanw1014 carry | **Upstream added** | `flash_attn_split_k_reduce.comp` is present upstream; no remaining UmaLLama-only shader delta |
| FA top-K selection shader | Nathanw1014 carry | **Dormant / not wired** | `flash_attn_top_k.comp` remains in the tree, but the active sparse path uses upstream compacting and has no top-K pipeline registration |
| GATED_LINEAR_ATTN | Nathanw1014 carry | **Upstream added** as `f26efa02a` | Vulkan `GGML_OP_GATED_LINEAR_ATTN` support is now upstream in `gla.comp`; no remaining UmaLLama-only shader delta |
| DeepSeek-V4 hyper-connection fused ops | [ggml-org/llama.cpp#26578](https://github.com/ggml-org/llama.cpp/pull/26578) | **Merged upstream** | Three shaders: `dsv4_hc_{pre,comb,post}.comp`. HC hardcoded to 4. Tunable with `GGML_VK_DISABLE_DSV4_HC[_COMB\|_PRE\|_POST]=1`. Measured: prefill +16.4%, decode +41.1% on DSV4-Flash IQ3_XXS, Nimo |
| DeepSeek-V4 Lightning Indexer | UmaLLama original plus upstream #27453 | **Upstream added** as `cb300598d`; local CM extensions are registered but not dispatched | Upstream provides the active base `lightning_indexer.comp`; `lightning_indexer_cm.comp` and `lightning_indexer_decode_cm.comp` are generated and registered (`pipeline_lightning_indexer_cm_f16`, `pipeline_lightning_indexer_decode_cm_f16`) but have no dispatch site in the canonical build. See [vulkan-init-order.md](vulkan-init-order.md) |
| DSV4 sparse FA gather-to-compact | UmaLLama original plus upstream #28105 | **Upstream added** generic sparse FA as `fc82583e6`; local top-K variant is dormant | The active sparse path uses upstream compaction; `flash_attn_top_k.comp` remains unregistered |
| FA flash-attn mask optimization | Nathanw1014 carry | **Upstream added** | `flash_attn_mask_opt.comp` is present upstream; no remaining UmaLLama-only shader delta |
| FA MMQ funcs shader | Nathanw1014 carry | **Upstream added** | `flash_attn_mmq_funcs.glsl` is present upstream; no remaining UmaLLama-only shader delta |
| Keep DeepSeek lightning-indexer K cache f16 | Nathanw1014 carry | Not upstreamed | Forces f16 key cache under quantized `-ctk` for Lightning Indexer correctness |
| Vulkan APU `nodes_per_submit` auto-lower | UmaLLama original | Not upstreamed (`ggml-vulkan.cpp` still hardcodes 100) | Defaults to 8 on UMA, 100 on discrete. `GGML_VK_NODES_PER_SUBMIT=N` override |
| Strix Halo RDNA3.5 tuning (ROCm/HIP) | gaetan-puleo carry | Upstream added `mmq-config-rdna3-5.cuh`; UmaLLama's has Strix Halo-specific tuning | `I = 64` in all MMQ CASE entries for upstream `#24127` `static_assert((I_) % 32 == 0)` |
| `common::host_available_ram()` | UmaLLama original | None | Extracted from duplicate implementations in `kv-ssd-cache.cpp` and `kv_page_manager.cpp` |
| DFlash framework | UmaLLama original | Not upstreamed | `src/models/dflash.cpp`. Generic decoder contract via `dflash.decoder_arch` metadata. Currently supports `"laguna"` |
| Laguna-S-2.1 | UmaLLama original | Not upstreamed | `src/models/laguna.cpp`. Sigmoid-routed MoE, shared expert, softplus attention gate, QK-norm, per-layer-type RoPE |

**UmaLLama focus downstream:** If a third-party carry lands upstream cleanly, the UmaLLama copy can be dropped on the next `merge upstream/master` and the local additions (memory gate, env overrides, follow-up fixes) rebased onto the upstream version. When a carry does not get upstreamed, UmaLLama carries it indefinitely — re-check upstream status each merge.

**Watch upstream #24127** (CUDA MMQ refactor): it added `static_assert((I_) % 32 == 0)` to the CASE macro, so any new `rdna3_5` config must keep `I` as a multiple of 32.
