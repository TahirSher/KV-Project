# COVE-KV: Research Brief

**Basis:** `COVE_KV.py` and every measurement we have so far:
- **HEDGE_KV v2:** SmolLM3-3B, 81 qasper items.
- **QUASAR_KV:** Llama-3.2-1B-Instruct, 670 paired LongBench items.
- **STRATA_KV:** Llama-3.2-1B-Instruct, decode microbenchmark on GPU plus a 7-item partial run.

**Venues** were verified in October 2026. Items marked "arXiv" are not peer-reviewed. This brief replaces `STRATA_RESEARCH_BRIEF.md`.

---

## 1. Ultimate objective and rationale

**Objective.**
- **Memory:** cut the KV cache of long-context LLMs by 4–16×.
- **Speed:** make decoding measurably faster than with the full cache.
- **Accuracy:** lose as little task accuracy as possible.
- **Constraints:** no training or calibration, and the context is compressed once, before the question is known.

**What the last runs taught us, and why COVE looks the way it does.**

| Finding | Evidence |
|---|---|
| At equal bytes, attention over the kept tokens with the model's own SDPA is the fastest decode | GPU, one layer, 64k tokens: full 89 µs; SDPA over 1/8 of the tokens 23 µs. Our mixed-precision kernels: QUASAR ≥ 4 ms; STRATA 344 µs, with no fall as bytes shrank. |
| Pricing tokens by the context's own queries over its whole length is what carried accuracy | Coverage vs end window: +9.1 / +18.6 (HedgeKV, SmolLM3, 1/8 / 1/16); +4.2 (QUASAR, Llama, 1/8); all p ≤ 0.001 |
| Low-bit mixed precision gains little on small models and costs speed | KIVI-2 −8 points vs full on Llama qasper; QUASAR beat int8 eviction by only +1.8 to +3.1 |

COVE keeps the pricing, prices eviction with its exact error model, and stores exact tokens that the model's own fused kernel reads. Speed is therefore guaranteed by construction: it is attention over `n_keep` tokens. The research question becomes which tokens to keep.

---

## 2. Research gaps (top-tier venues first)

**G1. Query-agnostic importance is estimated from a biased or expensive proxy.**
- **End-window scores:** SnapKV (NeurIPS 2024), PyramidKV (COLM 2025), Ada-KV (NeurIPS 2025) and TOVA (EMNLP 2024) score tokens with the queries at the end of the context. In query-agnostic reuse that window is one arbitrary sample of the future questions.
- **Cumulative attention:** H2O (NeurIPS 2023) accumulates attention causally, so early tokens collect more mass.
- **Reconstruction:** KVzip (NeurIPS 2025 oral) gets query-agnostic scores by reconstructing the context, at 2–3× the prefill cost.
- **Parametric:** Expected Attention (arXiv 2025) assumes Gaussian future queries.
- **Leverage scores:** Compactor (arXiv 2025) ignores the values and the output projection.

**G2. Eviction scores use the wrong error model.**
- Scores rank tokens by attention mass (SnapKV, H2O, TOVA) or by squared per-token errors (our own earlier drop price).
- The exact eviction error (Prop. 1 below) shows eviction errors add **coherently**. The right additive price is the first-order term p_j ‖v_j − o‖ in the output-projection metric.

**G3. "Fast" is often claimed but not measured on the deployment kernel.**
- Masking implementations, e.g. kvpress's AdaKV and KVzip, keep the full cache in memory: their compression is nominal.
- Mixed-precision KV caches need custom kernels that are rarely faster than eviction at equal bytes. Our own two attempts were 4–15× slower.

---

## 3. Hypotheses and research questions

**RQ1 (pricing).** Does coverage pricing beat end-window pricing at equal memory and identical decode cost?
- **H1:** `cove` > `cove_window` at 1/8, on ≥ 7/9 models.

**RQ2 (error model).** Does the bound-based price beat the squared independent-error price?
- **H2:** `cove` > `cove_l2`.

**RQ3 (SOTA).** Does COVE beat the official kvpress presses at 1/4, 1/8 and 1/16 memory?
- **H3:** this includes the strongest query-agnostic ones: KVzip, Compactor and Expected Attention.

**RQ4 (systems).**
- **H4a:** COVE's decode step costs what attention over `n_keep` tokens costs: 1.5–4.9× faster than full at 4k–64k on your GPU, as measured.
- **H4b:** its compression time (`compress_s`) is a small fraction of the prefill, and far below KVzip's extra 2–3× prefill.

**Contributions (if the hypotheses hold).**
- **C1.** A query-agnostic, coverage-based eviction price: an exact error bound in the residual-stream metric, under the context's own queries sampled over its whole length, with per-segment normalisation.
- **C2.** The theory: an exact eviction identity, the bound, and why squared prices are optimistic for eviction.
- **C3.** A deployment-path evaluation at real (not nominal) memory against official implementations, including the query-agnostic SOTA.

---

## 4. How each hypothesis is tested

| H | Mechanism | Test |
|---|---|---|
| H1 | `coverage_scores` (segment groups + window group) | `cove` vs `cove_window` |
| H2 | `score="l1"` (bound) vs `"l2"` (squared) | `cove` vs `cove_l2` |
| H3 | `cove_select`: sinks + window + top-scored, equal count per head | `cove` vs every `kvpress:*` (incl. `kvzip`, `compactor`), `snapkv_q2/q4`, `evict`, `am`, `kivi2/4` |
| H4 | `native_cove_generate` (HF SDPA + compressed DynamicCache); `attention_microbench` | `tokens_per_s`, `ttft_s`, `compress_s`, `cache_bytes_measured`; `microbench.csv` |

**Correctness guards.**
- Prop. 1 identity (to 1e-13) and the bound, on random sets.
- Selection sizes, with sinks and window always kept.
- Keep-all equals the full cache.
- A token read only by an early segment is kept, though the end window ranks it low.
- Harness vs SDPA, and full vs HF `generate()`.
- The deployment path vs its harness twin (`cove_ref`).
- Our pipeline vs kvpress's own pipeline.

**Statistics.** Paired per-item differences, macro over tasks, task-stratified bootstrap CI, sign-flip p, Holm per benchmark, and ≥ 7/9 models plus Wilcoxon across models. `--report` computes the same from a run's log or CSV while it is in progress.

---

## 5. What is novel, and what kind of contribution it is

**Kind.** A training-free **token-selection method with theory**, evaluated on the deployment path. It is **not** a new architecture and **not** a new kernel; decoding uses the model's own SDPA on purpose.

**Novelty.**
- **C1 / C2:** not found as stated in the literature.
- **Neighbours:** H2O (whole-context but causal mass), KVzip (query-agnostic, by reconstruction), Compactor (query-agnostic leverage scores), VATP-type value-aware scores (attention × value norm, not in the output metric, no coverage).
- **Honest size of the claim:** a scoring rule is a modest contribution. It becomes a strong paper only if H1–H3 hold, especially against KVzip at equal real memory and at a fraction of its compression cost.

---

## 6. Are the gaps addressed, and can the outcome be guaranteed?

**No outcome can be guaranteed before the runs.** This is the current state.

| Gap | In code | Evidence so far | Open |
|---|---|---|---|
| G1 | Yes | The coverage ingredient won in all three earlier methods (+4.2 to +18.6, p ≤ 0.001) | COVE itself; KVzip and Compactor were never run against it |
| G2 | Yes | Identity and bound verified numerically | H2 on real data |
| G3 | Yes | SDPA over the kept tokens: 1.5–4.9× faster than full on your GPU (measured) | Accuracy at that speed (H3) |

**Why the expected results are plausible:**
- **Speed** is not a hypothesis about a kernel any more. It is the measured cost of attention over fewer tokens, which every eviction method shares.
- **Accuracy** rests on the one ingredient that won repeatedly in our data.

**What must be stated in the paper:**
1. **Not lossless.** Every compressor we measured loses accuracy at 1/8–1/16 memory (QUASAR lost 6.3 points at 1/8 on Llama-3.2-1B). The claim is the best accuracy at a given memory and speed.
2. **Same speed as other eviction methods.** At equal token count every eviction method has the same decode cost. COVE's speed advantage exists only against the full cache, quantised caches and masked "compression". Its accuracy must beat them at the same speed.
3. **Quantised hybrids may win.** They can still be more accurate at equal bytes (`snapkv_q4`, `kivi4`), but they are slower without kernels. This is a trade-off to report, not hide.

**The result that makes this a paper:** COVE beats KVzip, Compactor, Expected Attention and SnapKV at equal real memory on ≥ 7/9 models, with `compress_s` much smaller than KVzip's overhead.

**If that fails:** coverage pricing is not a contribution on its own, and the honest outcome is a negative-result or analysis paper.
