# STRATA-KV: Research Brief

**Basis:** `STRATA_KV.py` and the evidence from our runs so far:
- **HEDGE_KV v2:** SmolLM3-3B, 81 qasper items.
- **QUASAR_KV:** Llama-3.2-1B-Instruct, 670 paired LongBench items.

**Venues** were verified in October 2026. Items marked "arXiv" are not peer-reviewed. Re-check the 2026 preprints in full text before submission; only their abstracts were available.

---

## 1. Ultimate objective and rationale

**Objective.** Cut the memory a long-context LLM needs for its KV cache by 4–16×, make decoding faster than the uncompressed cache, and lose as little task accuracy as possible. All of this without training or calibration, in the setting that matters for serving: the context is compressed **once, before the question is known** (query-agnostic reuse, as in NVIDIA kvpress).

**Why it matters.**
- The KV cache grows linearly with context length and batch size, and in long-context decoding it dominates both memory and memory bandwidth.
- Decoding is memory-bound, so a step's latency is set by the bytes it reads.
- A compressor therefore helps speed only if the decoder reads the compressed bytes directly. Saving memory alone is not enough.

**Rationale for the code.** Each version was a response to measured failures:

| Version | What our runs showed | Consequence |
|---|---|---|
| HedgeKV v2 (SmolLM3) | Beat the official presses by 14–19 points at matched memory. Its minimax over query groups did nothing (−0.35, p = 0.57), and compression cost about 100 s per sample. | Keep coverage pricing and per-token precision; drop minimax. |
| QUASAR (Llama-3.2-1B) | Beat every baseline at every budget. Its spectral rank axis did nothing (metric basis −0.2, p = 0.80; spectral basis +0.9, p = 0.21). Decode attention was 30–300× slower than SDPA. | Keep coverage and mixed precision; drop the rank axis; design the decode kernel first. |
| **STRATA (this script)** | Not yet run on real models. | Must show accuracy at least QUASAR's, plus real speed. |

---

## 2. Research gaps (top-tier venues first)

**G1. Token importance under an unknown future query is estimated with a biased proxy.**
- **Window-based scoring.** SnapKV (NeurIPS 2024), PyramidKV (COLM 2025), Ada-KV (NeurIPS 2025) and TOVA (EMNLP 2024) score tokens with queries at the end of the context. In query-agnostic reuse those queries are one arbitrary sample of the questions to come.
- **Cumulative attention.** H2O (NeurIPS 2023) accumulates attention causally, which over-weights early tokens.
- **Reconstruction.** KVzip (NeurIPS 2025) needs an extra reconstruction forward pass.
- **Parametric models.** Expected Attention (arXiv 2025) assumes Gaussian future queries.
- **Our evidence.** Replacing whole-context sampled queries with the end window costs:
  - HedgeKV on SmolLM3: −9.1 points at 1/8 and −18.6 at 1/16 (p ≤ 0.001);
  - QUASAR on Llama: −4.2 points at 1/8 (p < 0.001).

**G2. Eviction and quantisation are optimised separately, although at equal bytes the best answer mixes them.**
- **Eviction-only:** H2O, SnapKV, StreamingLLM (ICLR 2024).
- **Quantisation-only:** KIVI (ICML 2024), KVQuant (NeurIPS 2024), QuaRot (NeurIPS 2024), SpinQuant (ICLR 2025).
- **Evidence the mix wins:**
  - "More Tokens, Lower Precision" (Findings of EMNLP 2025) shows that keeping more tokens at lower precision beats pruning at equal memory.
  - Importance-driven mixed precision exists: ZipCache (NeurIPS 2024); MiKV (arXiv; venue unverified).
  - RDKV (arXiv 2026) allocates bit widths, with 0 bits meaning eviction, by a knapsack.
- **What is missing:** none of these allocates eviction and precision jointly under query-agnostic importance with a measured, exact byte budget.

**G3. Mixed precision is not fast in practice.**
- Fast low-bit decode kernels exist for **uniform** precision: KIVI's kernels, QServe (MLSys 2025), BitDecoding (arXiv).
- Per-token mixed precision fragments the cache into irregular blocks. QUASAR is our own counter-example: accurate, but 30–300× slower than SDPA.
- **Closest prior art:**
  - Minima-KV (arXiv 2026): two page formats with an online-softmax merge.
  - KVmix (AAAI 2026): separate kernels per width.
  - "Don't Waste Bits!" (arXiv 2026): learned per-token precision with a latency penalty, on models of 1.7B or fewer.
- **Not found:** one-pass split-K decoding over per-token 2/4/8-bit plus exact strata, with a layout designed so every stratum is a dense tensor.

**G4. Evaluation gap.** Many compression papers report nominal rather than measured memory, run question-aware protocols, or do not compare against official implementations at matched bytes.

---

## 3. Contributions, hypotheses and research questions

**RQ1 (pricing).** In query-agnostic reuse, does pricing tokens with the context's own queries, sampled uniformly over its length and normalised per segment, beat end-window pricing at the same memory?
- **H1:** `strata` > `strata_window` at 1/8 memory on ≥ 7/9 models.

**RQ2 (allocation).** Does one Lagrangian allocation over {drop, 2, 4, 8 bits, exact} beat:
- importance-free single precision (H2a: `strata` > `strata_uniform`), and
- the strongest simple eviction-plus-quantisation hybrid (H2b: `strata` > `snapkv_q4` and `snapkv_q2`)?

**RQ3 (accuracy vs SOTA).** Does STRATA beat every official kvpress press at 1/4, 1/8 and 1/16 memory, and match KIVI-4 / KIVI-2 at less memory?
- **H3:** pre-registered contrasts, Holm-corrected per benchmark, then ≥ 7/9 models and a Wilcoxon test over models.

**RQ4 (speed).** Can a mixed-precision cache be decoded faster than the full cache, with cost tracking bytes stored?
- **H4a:** the fused stratum kernel beats full SDPA for one decode step at long contexts (≥ 16k) at 1/8 memory.
- **H4b:** its latency grows with stored bytes, not with token count.
- **H4c:** compression time (`compress_s`) is a small fraction of prefill.

**Contributions (if the hypotheses hold).**
- **C1.** Coverage pricing for query-agnostic compression (G1).
- **C2.** Head-synchronous precision strata. Each head ranks its own tokens, rank position j gets one option shared by all heads, and so every stratum is a dense tensor with no padding and no index. The options come from one exact-budget Lagrangian allocation (G2).
- **C3.** A split-K flash-decoding kernel over planar-packed affine codes. Logits come from tensor-core dots against raw codes plus one per-token affine correction, so the bytes read equal the bytes stored (G3).
- **C4.** A measured-memory, official-baseline, query-agnostic evaluation protocol with paired statistics (G4).

---

## 4. How each hypothesis is tested

| Hypothesis | Mechanism in the code | Test |
|---|---|---|
| H1 | `coverage_stats`: per-segment query groups, exact Prop. 1 drop cost, first-order key/value costs in the residual-stream metric | `strata` vs `strata_window` (pre-registered contrast) |
| H2a/H2b | `StrataMethod.compress_layer`: exact per-token code errors in the metrics; `head_sync_allocate`: batched Lagrangian | `strata` vs `strata_uniform`, `snapkv_q4`, `snapkv_q2` |
| H3 | Same | `strata` vs all `kvpress:*`, `snapkv`, `streamingllm`, `evict`, `am`, `kivi2`, `kivi4` |
| H4a/b | `StrataState` and `_strata_tier_kernel`; `attention_microbench` (CUDA graphs; full SDPA vs eviction SDPA vs STRATA at equal bytes) | `microbench.csv` at 4k–128k tokens |
| H4c | `native_strata_generate` | `compress_s`, `prefill_s`, `ttft_s` |

**Correctness guards (asserted at start-up or per sample):**
- Codec optimality against Max (1960).
- Prop. 1 is exact.
- Rotation identity.
- The allocator is feasible, monotone, and all-exact when the budget allows.
- Decoder output equals the harness reconstruction.
- Packed bytes equal the allocator's bits, also in bf16.
- The kernel is used only after it matches the reference on the sample's own code.
- Harness vs SDPA, full cache vs HF `generate()`, and our pipeline vs kvpress's own pipeline.

**Statistics.**
- Paired per-item differences, macro-averaged over tasks.
- Task-stratified bootstrap CI and sign-flip p-value.
- Only budget-compliant units enter a contrast.
- Cross-path contrasts are path-adjusted (difference in differences against full − native_full).
- Holm correction per benchmark; across models, ≥ 7/9 models plus a Wilcoxon test.

---

## 5. What is novel, and what kind of contribution it is

**Kind.** A training-free **compression technique plus a systems co-design** (allocation, layout and kernel). It is **not a new model architecture**. Each part has precedent; the contribution is how they are combined and engineered.

| Part | Status |
|---|---|
| Hadamard rotation, uniform codes, flash-decoding | **Credited** (QuaRot, KIVI/QServe, FlashDecoding) |
| Prop. 1 single-entry softmax gain | Elementary algebra, used as an exact pricing tool (our use) |
| Coverage pricing (C1) | **Not found** in the literature as stated; adjacent to H2O, Compactor and LORE-KV |
| Head-synchronous strata (C2) | **Not found**; mixed precision itself is known (ZipCache, MiKV) |
| One-pass multi-stratum split-K kernel over affine planar codes (C3) | **Not found** for 4 or more per-token tiers; Minima-KV has 2 formats |
| Exact-budget Lagrangian over drop/bits/exact (C2) | Related to RDKV (arXiv); ours adds coverage pricing, head synchrony and the kernel-aligned layout |

**Honest assessment.** C1–C3 together form a defensible, novel technique contribution if the experiments confirm H1–H4. On its own, any single component would be judged incremental.

---

## 6. Are the gaps addressed, and can we guarantee the result?

**No result can be assured "100%" before the experiments are run.** Here is what the evidence supports and what remains open.

| Gap | Addressed in code | Evidence so far | Still open |
|---|---|---|---|
| G1 | Yes | Coverage beat the end window on both models we ran (+4.2 to +18.6, p ≤ 0.001) | STRATA itself on ≥ 7 models |
| G2 | Yes | QUASAR (same coverage + allocation idea) beat SnapKV + KIVI-4 (+1.9 / +2.9 / +5.0) and all presses (+4 to +15) | Whether STRATA's affine codes match QUASAR's accuracy (expected similar: QUASAR's spectral parts were not significant) |
| G3 | Yes (kernel + layout) | Kernel matches the reference to ≤ 2.5e-3 in Triton's interpreter. **No GPU timing yet.** | H4: speed on GPU |
| G4 | Yes | Measured bytes equal analytic bytes; official presses; paired tests | — |

**Why the expected results are plausible:**
- **Accuracy.** STRATA keeps the two components that our ablations showed carry the gains: coverage pricing and per-token mixed precision with eviction. It drops the component that did not (spectral rank).
- **Speed.** Decode attention is bandwidth-bound. The kernel reads exactly the stored bytes, with one launch per stratum (≤ 3) plus a small merge. At long contexts and 1/8 memory, the bytes moved are 8× fewer than the full cache's, so speed-ups are physically expected. At short contexts (≤ 4k), fixed launch costs may make it no faster than SDPA.

**What we cannot claim, and must state in the paper:**
1. **Losslessness.** QUASAR lost 2.1 / 6.3 / 9.2 points to the full cache at 1/4 / 1/8 / 1/16. Training-free compression at 8–16× is lossy for every method we tested. The claim is a better accuracy–memory–latency trade-off, not "no loss".
2. **Beating eviction on latency at equal bytes.** Eviction attention with fused SDPA reads fewer tokens and launches one kernel. STRATA's advantage is accuracy at equal bytes; its latency is expected between eviction and the full cache.
3. **Generality.** Each model so far shows parts of the picture. The decision rule needs ≥ 7 of 9 models.

**Results that would make this the backbone of a paper:**
- H1–H3 supported by the decision rule.
- The microbenchmark showing STRATA faster than full SDPA from 16k tokens, with latency tracking bytes.
- `compress_s` well below prefill time.

If H4 fails, the paper is an accuracy-at-memory paper with an honest systems section. If H1 fails on most models, coverage is not a contribution and the method reduces to mixed-precision allocation, which is incremental over ZipCache and RDKV.
