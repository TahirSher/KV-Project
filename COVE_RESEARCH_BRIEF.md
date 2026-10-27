# COVE-KV v3: Research Brief

This brief covers `COVE_KV.py` v3. It answers the six questions of `Code_Assessments.md` and replaces the v2 brief.

**New evidence:**
- **Source:** the v2 run on Qwen2.5-3B-Instruct, LongBench qasper, 167 paired items, 72 methods.
- **Comparison basis:** all differences are paired per item and compared against the full cache on the same path (`native_full`, bf16 SDPA). Intervals are 95% task-stratified bootstrap CIs; p-values come from a sign-flip test with Holm correction per benchmark.
- **Scope:** one model and one task, so no general claim follows from it.

---

## 1. Objective, and what the v2 run says about it

**Objective.** Smaller KV memory and faster inference, with no loss beyond a pre-registered margin. No training is involved, and the context is compressed before the question is known.

### 1.1 Accuracy (score points)

| | 1/4 memory | 1/8 | 1/16 |
|---|---|---|---|
| `cove` (exact tokens) vs full | **−5.8** [−10.5, −1.3] | **−12.7** | **−19.2** |
| `cove_q4` vs full | **+0.1** [−1.7, +2.0] | −4.4 | −8.9 |
| `cove_q2` vs full | −2.9 | −3.8 | −8.0 |
| `cove_q4` vs `cove`, same bytes | +6.0 | **+8.3** | **+10.2** |
| `cove_q4` vs SnapKV + KIVI-4 (MTLP) | +0.9 | **+10.4** | **+9.1** |
| `cove` vs kvpress SnapKV / Compactor (real memory) | +8.7 / +1.9 | +7.5 / +2.1 | +4.0 / −1.4 |
| `cove` vs KVgrad / KVzip (masked keys, nominal memory) | −5.5 / −2.7 | −5.8 / −0.1 | +0.4 / +8.9 |

Bold values have Holm p < 0.05 or a CI that excludes 0. In addition, coverage scoring beat end-window scoring by **+7.7** [+3.4, +12.0].

**What this shows:**
1. **Eviction alone is not lossless.** Every eviction method loses at least 5 points at 1/4 and more below that.
2. **Spending the same bytes on more tokens at 2–4 bits is far better** than keeping fewer exact tokens: +8 to +11 points at 1/8 and 1/16.
3. **Coverage selection is what makes the coded arm work.** It beat SnapKV selection at the same code width by about 10 points.
4. **The best width depends on the budget.** 4 bits won at 1/4; 2 bits won at 1/8 and 1/16. This is the motivation for COVE-QA's per-layer choice.
5. **`cove_q4` at 1/4 is indistinguishable from the full cache** (+0.1). Non-inferiority is *not* shown, though: the lower CI bound, −1.7, is below the −1 margin. A single task is underpowered, as the v2 brief predicted.

### 1.2 Speed and memory

| Finding | Cause | Fix in v3 |
|---|---|---|
| COVE time to first token was 1.7–2.0 s, against 0.41 s for the full cache | Compression took 1.3–1.6 s: the scoring loop synchronised with the host about 720 times per sample, each costing ~1 ms on the shared GPU | Batched scoring with no host synchronisation; it gives the same scores as v2 (relative error 2e-7) |
| COVE-Q attention was 4–90× slower than full SDPA | The codes were dequantised into memory before SDPA | A fused Triton kernel reads the 2/4-bit codes and dequantises in registers |
| `native_full` and every press decoded at 16–17 tokens/s, against 47–50 for `cove` | **Measurement artefact.** A FLOP-counting forward pass ran inside their decode timer, with answers of only 3–4 tokens. | FLOP measurement moved outside the timer, plus a fixed-length decode probe (`decode_probe_ms`) |
| The eager microbenchmark column read ~0.9–2 ms for everything | It measured host-synchronisation latency, not compute | Back-to-back timing |

### 1.3 Where your expectation is wrong for this workload

On qasper (about 4.7k tokens) with a 3B model, the full KV cache is **168 MiB against a 6.4 GiB peak**, and the weights dominate. No KV compressor can make this workload much faster or smaller.

Speed and memory gains appear only at long contexts (32k–128k tokens) or in batched serving. The speed claim therefore has to be tested on RULER at 32k+ and on LongBench-v2, not on qasper.

### 1.4 Baselines that failed in v2, and the fixes

| Baseline | What failed | Fix |
|---|---|---|
| CriticalKV | Failed on every item: Qwen2's config has no `head_dim` | Set from the model |
| PyramidKV | Failed on 75 of 167 items at 1/4: Hugging Face builds one attention mask from layer 0's length | `LayerCausal` builds a mask for each layer |
| KVgrad, KVzip | Out-of-memory errors from other processes on the shared GPU | Retried, or left for `--resume` |

---

## 2. Research gaps (what the data and the literature leave open)

**G1. Query-agnostic selection without extra cost.**
- KVzip and KVgrad pay 10–16× the prefill time.
- KVgrad was the only method that beat COVE at 1/4 and 1/8, using nominal (masked) memory.

**G2. Precision versus token count at equal bytes, chosen per sample and per layer.**
- MTLP (Findings of EMNLP 2025) fixes one width per budget and uses SnapKV selection.
- KVTuner* searches layer widths offline.
- ZipCache (NeurIPS 2024) mixes precisions by token importance.
- **Not found:** an online, per-layer choice between exact eviction and 4- or 2-bit coding at fixed bytes, made by measuring output distortion on held-out queries of the same context.

**G3. Quantised caches that are also fast on the deployment path.**
- Our dequantise-then-SDPA path was 4–90× slower than full attention.
- Fused low-bit kernels exist (KIVI, QServe*, FlashInfer*). The open engineering question is whether a coded cache from a coverage selection can reach the decode speed of exact eviction at the same bytes.

**G4. "Lossless" claims without a pre-registered non-inferiority test.** Unchanged from v2.

(* venue not verified against the proceedings.)

---

## 3. Contributions, hypotheses and research questions

| H | Claim (pre-registered contrast) | Status after v2 |
|---|---|---|
| H1 | `cove` > each baseline at the same budget | Supported against real-memory presses except Compactor; **refuted** against KVgrad at 1/4 and 1/8 (nominal memory) |
| H2 | Non-inferiority to the full cache: margin 1 point, one-sided 0.025, Holm, ≥ 30 units | Not shown for any method; `cove_q4@1/4` is the only candidate (+0.1 [−1.7, +2.0]) |
| H3 | `cove_qa` (and `cove_q2`, `cove_q4`) > `cove` at the same bytes | Supported for `cove_q2` and `cove_q4` at 1/8 and 1/16 |
| H4 | `cove_qa` > `cove_q2` and > `cove_q4` (per-layer choice) | **New in v3, untested** |
| H5 | `cove_qa` > SnapKV + KIVI (MTLP) and > KVzip, KVgrad, CriticalKV, Compactor | `cove_q4` > MTLP at 1/8 and 1/16; the rest untested |
| H6 | Speed: TTFT, `compress_s`, `decode_probe_ms`, fused kernel vs SDPA (descriptive) | v2 numbers invalid (§1.2); re-measure |

**Contributions if the hypotheses hold:**
- **C1:** coverage selection (Props. 1–2).
- **C2:** COVE-QA, a per-layer width choice by held-out measured distortion (Props. 4–5).
- **C3:** a deployment-path evaluation at real memory with a non-inferiority test. The fused kernel is engineering, not a contribution.

---

## 4. How each gap is addressed (functions in `COVE_KV.py`)

| Gap | Mechanism | Test |
|---|---|---|
| G1 | `CoveragePlan` and batched `coverage_scores`: one pass per layer from query rows captured during the model's own prefill | `compress_s` and `ttft_s` vs KVzip and KVgrad; H1 |
| G2 | `validation_positions` (held-out queries), `choose_width` (distortion D_b of the cache exactly as it will be read), `cove_q_plan` (bytes-matched counts) | H3, H4, H5; the `layer_widths` column |
| G3 | `_coded_decode_kernel` and `coded_attention` (split-K, dequantisation in registers, exact online-softmax merge); `fused_decode_check` turns the kernel off if it errs; `LayerCausal` | Microbenchmark rows `/fused` vs `/dequant` vs `full_sdpa` vs `cove/sdpa`; `decode_probe_ms` |
| G4 | `NI:` contrasts (bootstrap-inversion p, Holm, minimum units) | H2 |

**Correctness checks, all passed on CPU:**
- Batched scores equal the v2 per-group scores (2.3e-7 relative).
- Validation queries are disjoint from scoring queries.
- At a whole-context budget, COVE-QA keeps width 16 with distortion 0.
- The fused kernel matches dequantise + SDPA to 4–7e-7 relative error (Triton interpreter).
- **In generation:**
  - The fused path gives exactly the same tokens as the dequantise path.
  - With layers of different widths and lengths, the deployment path gives exactly the same tokens as the fp32 harness.
- **Smoke run:**
  - Each deployment path matches its harness twin, for `cove` and `cove_qa` (1.000).
  - PyramidKV and CriticalKV now run.

---

## 5. Kind of novelty

**Kind:** a combination and an empirical method, not a new paradigm.

**Published elements:**
- Coverage scoring from prefill queries is close to H2O, KVzip, EchoPress* and LORE-KV*.
- The output-aware term is close to CAOTE and CriticalKV.
- More tokens at lower precision is MTLP.
- The quantiser is KIVI.
- Fused decoding is standard.

**Not found as such:** an online, per-layer choice of precision versus count at fixed bytes, by held-out measured distortion, on a coverage selection.

**Size of the claim:** in v2, the value-aware score form did not matter (`cove` vs `cove_l2`: +0.5, not significant). What mattered was the coverage and the precision–count trade-off, so v3 builds on those two.

---

## 6. Are the gaps addressed, and is the objective "100% achieved"?

**No, and no script can guarantee it before the experiment.**

**What is guaranteed:**
- Every gap has a mechanism and a pre-registered test.
- Memory is measured in real bytes.
- Timings are now valid (§1.2 fixes).
- The fused kernel and the mixed-length decode path are checked for exactness.

**What is still open:**
1. **The fused kernel's GPU speed is unknown.** It was verified for correctness only, in Triton's interpreter. If it is not faster than full SDPA at 32k+ tokens, the coded arm stays a memory-and-accuracy arm, not a speed arm.
2. **Whether COVE-QA's per-layer choice beats fixed widths (H4)** is untested.
3. **Non-inferiority needs about 1,800 paired items** (the LongBench macro average). 167 qasper items cannot show it.

**Expectation, stated honestly:**
- At 1/4 memory, `cove_qa` or `cove_q4` is the arm most likely to pass non-inferiority.
- At 1/8, the v2 data show a loss of about 4 points. Removing that loss would need a better selection or more bits, not just engineering.

**Run plan (sequential, on the shared GPU):**
1. `python COVE_KV.py --microbench 4096 16384 65536 131072 --micro-shape 128 2 8`. This checks whether the kernel delivers.
2. `python COVE_KV.py --suite longbench --models Qwen2.5-3B-Instruct --methods cove cove_qa cove_q kvpress:kvzip kvpress:kvgrad kvpress:criticalkv kvpress:compactor kvpress:snapkv snapkv_q` (all tasks).
3. `python COVE_KV.py --suite ruler --models Qwen2.5-3B-Instruct --methods cove cove_qa kvpress:snapkv kvpress:compactor`. This is the long-context speed and memory test.
4. Read results with `--report` at any time.
