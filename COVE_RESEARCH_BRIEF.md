# COVE-KV v2: Research Brief

This brief answers the six questions of `Code_Assessments.md` for `COVE_KV.py` v2. It replaces the v1 brief and `STRATA_RESEARCH_BRIEF.md`.

v2 adds three things:
- **COVE-Q**, a quantised arm: the same token selection, with more tokens kept at lower precision for the same bytes.
- **The closest published baselines** that kvpress 0.5.5 ships: CriticalKV, KeyDiff and KVgrad.
- **A formal non-inferiority test**, so that "without losing results" becomes a statement that can be falsified.

**Evidence base:**
- HEDGE_KV v2: SmolLM3-3B, 81 qasper items.
- QUASAR_KV: Llama-3.2-1B, 670 paired LongBench items.
- STRATA_KV: Llama-3.2-1B, GPU decode microbenchmark and two partial runs. The latest has 4 qasper items, because out-of-memory errors from other processes on the shared GPU stopped the rest.

**Literature:** searched in October 2026. Venues marked * were not verified against the proceedings, and full texts could not be fetched. Read the papers in §5 in full before claiming novelty.

---

## 1. Objective and rationale

**Objective.** Compress the KV cache of long-context LLMs by 4–16× and decode faster than with the full cache, losing no more than a pre-registered margin of task accuracy. There is no training or calibration. The context is compressed once, before the question is known (the kvpress protocol).

**What our own measurements force on the design.** These are measured on your GPU, not assumed.

| Finding | Evidence |
|---|---|
| Reading fewer tokens through the model's own SDPA is the only route that was faster in practice | One layer, 64k tokens: full SDPA 90.8 µs; SDPA over 1/8 of the tokens 20.3 µs (4.5× faster); over 1/16, 15.0 µs (6.0×) |
| Our mixed-precision kernels were slower than the full cache | STRATA kernel: 259–342 µs at 64k, 3–4× slower than full; QUASAR ≥ 4 ms |
| Mixed precision cost time end to end | STRATA compression took 1.63 s after a 0.15 s prefill. Its time to first token (TTFT) was 1.45 s against 0.24 s for the full cache. |
| Pricing tokens by the context's own queries over its whole length carried accuracy every time | +9.1 / +18.6 points (HedgeKV, 1/8 and 1/16); +4.2 (QUASAR, 1/8); all p ≤ 0.001 |

**Rationale.**
- **Speed:** COVE evicts. Its kept tokens form a dense bf16 cache read by SDPA, so its decode cost is attention over `n_keep` tokens.
- **Accuracy:** COVE-Q spends the same bytes on 3–5× more tokens as KIVI codes. This is where quantisation helps: memory, not speed.
- **What you asked for and what the data allows:** you want quantisation to deliver speed, memory and accuracy together. No method in our data did all three at once. At equal bytes, quantisation buys accuracy (more context kept) and costs decode time. Eviction buys decode time. COVE gives you both points on the same selection, measured, so the trade-off is visible rather than hidden.

---

## 2. Research gaps

**G1. Query-agnostic scores are either biased or expensive.**
- **End-window scores bias towards the last queries:** SnapKV (NeurIPS 2024), PyramidKV (COLM 2025), Ada-KV (NeurIPS 2025).
- **Cumulative mass is causal:** H2O (NeurIPS 2023).
- **Context reconstruction costs 2–3× the prefill:** KVzip (NeurIPS 2025), and KVgrad (ICML 2026), which adds a gradient.
- **Sampled continuations are needed:** LORE-KV (arXiv 2610.07643*).
- **Parametric query models:** Expected Attention*, GVote (ICLR 2026*).
- **Key geometry only:** Compactor*, KeyDiff*.
- **Open question:** is there a score that uses the queries the prefill has already computed, over the whole context, at no extra forward pass, that is as good as reconstruction?

**G2. Value- and output-aware scores rest on single-token or loose bounds.**
- **CAOTE** (ICLR 2026) has the exact single-token change.
- **CriticalKV** (ICML 2026) bounds the output change with ‖vW_O‖₁ weighting.
- **What was not found:** using the exact set-level eviction identity (Prop. 1) to justify an additive score.

**G3. Precision versus token count at equal memory has been studied only with end-window selection.**
- **MTLP** (Findings of EMNLP 2025) used SnapKV.
- **Not found:** whether coverage selection changes the optimum, measured on the deployment path with real bytes and real decode time.

**G4. "Lossless" is claimed loosely.**
- KV-compression papers report mean differences.
- **Not found:** a pre-registered non-inferiority margin with one-sided tests and multiplicity control.
- Masking implementations (AdaKV, KVzip and KVgrad in kvpress) report nominal, not real, memory.

---

## 3. Contributions, hypotheses and research questions

| RQ | Hypothesis (pre-registered contrast) | Decision rule |
|---|---|---|
| RQ1. Does coverage pricing beat published query-agnostic and output-aware scores at equal real memory? | **H1:** `cove@r` > each `kvpress:*@r`, including KVzip, KVgrad, CriticalKV, Compactor and Expected Attention, and > `snapkv@r` | One-sided Holm per benchmark, α = 0.05; supported on ≥ 7/9 models plus a Wilcoxon test across models |
| RQ2. Is the loss relative to the full cache within the margin? | **H2:** NI, `mean(A − native_full) > −δ`, with δ = 1 point (0.01 nat/token for long PPL), for `cove`, `cove_q2` and `cove_q4` at every budget | One-sided α = 0.025 (bootstrap-inversion p), Holm within the non-inferiority family, the 95% CI lower bound > −δ, and ≥ 30 paired units |
| RQ3. At equal bytes, do more coded tokens beat fewer exact tokens? | **H3:** `cove_q{b}@r` > `cove@r` | As H1 |
| RQ4. Does coverage selection improve the coded arm? | **H4:** `cove_q{b}@r` > `snapkv_q{b}@r` (MTLP) and > `kvpress:kvzip@r` | As H1 |
| RQ5. What do speed and memory actually cost? | **H5** is descriptive: `tokens_per_s`, `ttft_s`, `compress_s`, `cache_bytes_measured`, and the one-layer microbenchmark (graph and eager timing) | Reported, not tested |

The ablations at 1/8 are `cove_window`, `cove_l2` and `cove_pool5`. Each isolates one design choice.

**Contributions, if the hypotheses hold:**
- **C1:** the coverage score, with the set-level identity and bound behind it (Props. 1–2).
- **C2:** COVE-Q, the precision–count trade-off on coverage selection, with its first-order error split (Prop. 4) and real-byte, real-time measurements.
- **C3:** a deployment-path evaluation at real memory against official implementations, with a pre-registered non-inferiority test.

---

## 4. How each gap is addressed in the code

| Gap | Mechanism (function) | Test |
|---|---|---|
| G1 | `coverage_groups` and `coverage_scores`: query rows captured during the model's own cached prefill (`CacheCapture`), with no extra pass | H1; `compress_s` against KVzip's and KVgrad's prefill |
| G2 | `score="l1"`: p_j‖v_j − o‖ in the W_OᵀW_O metric (Prop. 1); the squared single-token price is the `cove_l2` ablation | H1 vs CriticalKV; `cove` vs `cove_l2` |
| G3 | `cove_q_plan` (bytes-matched counts), `kivi_pack` / `kivi_unpack`, `CodedLayer` (one layer dequantised at a time, then SDPA), optional `torch.compile` fusion | H3, H4; microbenchmark rows `cove_q*` |
| G4 | `NI:` contrasts; `ni_p_value` inverts the task-stratified bootstrap. A sign-flip test of d + δ would read the items both methods fail (d = 0 exactly) as evidence. Holm within the family; cross-model Wilcoxon test on estimate + δ; masked presses flagged as nominal memory | H2 |

**Correctness guards** run at start-up (`cove_tests`) and in `--smoke`:
- Prop. 1 holds to 8e-14, and its bound holds.
- Prop. 4's bound holds on random cases.
- COVE-Q codes dequantise identically to the KIVI baseline (difference 0).
- The payload equals its analytic size byte for byte.
- `cove_q_plan` is within budget and maximal.
- `CodedLayer` returns `[codes ; exact ; new]`.
- Selection invariants hold.
- The deployment path agrees with its harness twin for `cove` and `cove_q4`.
- The full cache agrees with HF `generate()` and with kvpress's own pipeline.

---

## 5. Kind of novelty

**Kind:** a training-free selection-and-storage method with theory and a pre-registered evaluation. It is an incremental **combination**, not a new paradigm, architecture or kernel.

Every element is published or close to it:

| Element | Closest prior work |
|---|---|
| Output-change score | CAOTE (single token, value space) |
| Attention × output-projected value | CriticalKV, KVgrad, LaProx* |
| Whole-context or pseudo-query scoring | H2O, KVzip, EchoPress*, LORE-KV* |
| More tokens at lower precision | MTLP |
| Quantiser | KIVI |

**Not found as such:**
- the set-level identity and triangle bound used to justify an additive p_j‖v_j − o‖ score in the W_OᵀW_O metric;
- the context's own prefill queries, sampled per segment with per-segment normalisation, at no extra forward pass;
- equal counts per head, giving a dense SDPA cache;
- the coded arm on the same selection.

The size of the claim depends entirely on H1 and H4 against KVzip, KVgrad, CriticalKV and MTLP. **LORE-KV, EchoPress and KVgrad are the most likely to pre-empt C1.** Read them in full first.

---

## 6. Are all gaps addressed, and can the result be guaranteed as "100% achieved"?

**No.** No method or script can guarantee an empirical result before the experiment, and claiming one would be scientifically wrong.

What can be guaranteed, and is checked, is that:
- every gap has a mechanism in the code and a pre-registered test;
- memory is measured in real bytes;
- speed is measured on the deployment path;
- the statistics control for multiplicity.

| Gap | In code | Evidence so far | Still open |
|---|---|---|---|
| G1 | Yes | The coverage ingredient won in 3 of 3 earlier methods (+4.2 to +18.6, p ≤ 0.001) | COVE against KVzip, KVgrad and CriticalKV was never run |
| G2 | Yes | Props. 1 and 4 verified numerically | H1 against CriticalKV on real data |
| G3 | Yes | Toy random-init model (smoke), KL to the full cache at 1/4 memory: `cove_q4` 0.139, `cove` 0.299, `snapkv_q4` 1.086. That is the direction H3 and H4 predict, but it is not evidence about LLMs. On CPU, COVE-Q decoding was 9–12× slower than full attention. | H3 and H4 on real models; COVE-Q decode time on GPU |
| G4 | Yes | — | H2 needs a large sample (see the note below) |

**Sample size for H2.** Under the normal approximation, with true difference 0, one-sided α = 0.025 and 80% power, n ≈ 7.84 σ_d² / δ².
- σ_d is the standard deviation of the paired per-item differences. It is an assumption here; measure it with `--report`.
- If σ_d ≈ 15 points, n ≈ 1,800 paired items.
- The full LongBench macro average (16 tasks × ~200 items) is enough.
- A single task (≈ 200 items) is not: H2 is decided on the macro average.

**What the latest STRATA run does and does not show.**
- Four qasper items are about 1–2% of what a decision needs. The full cache scored 0 on most of them, so no accuracy conclusion can be drawn from it.
- What it does show is systems behaviour: compression ≈ 10× prefill, TTFT 6× the full cache, and a decode kernel slower than the full cache. Those are the reasons STRATA is dropped.

**Honest expectations.**
1. **COVE** will be faster than the full cache, with a measured decode step 1.3–6× faster at 1/4–1/16. Whether it passes the NI test at 1/8 is uncertain: QUASAR lost 6.3 points at 1/8 on Llama-3.2-1B.
2. **COVE-Q** is the arm most likely to pass NI. At 1/8 memory it keeps about 40% of the tokens at 4 bits. Its decode is not expected to be faster than the full cache at 1/8, and may be close to or slower than it. That trade-off has to be reported.
3. **If H1 fails against KVzip, KVgrad or CriticalKV,** the honest outcome is an analysis paper: coverage pricing is not enough on its own.

**Run plan, kept small so that a shared GPU does not out-of-memory the run:**
- Start with `--methods cove kvpress:kvzip kvpress:criticalkv kvpress:kvgrad kvpress:snapkv snapkv_q` on qasper, hotpotqa and trec, with at least 50 items each.
- Read the result with `--report`.
- Then go to the full suite across models.
