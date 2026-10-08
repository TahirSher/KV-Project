# LCTC-KV: Research Assessment

**Subject:** `LCTC_KV.py` (Loss-aware, Closed-loop Transform Coding of the KV cache, training-free)
**Predecessor:** `KV_LDT_v12_2.py` (lexical fidelity of training-free cross-layer KV sharing)
**Method of review:** I read all 1,681 lines of the code and checked it against the literature. I could not execute it: the CPU PyTorch wheel host is blocked in this environment, so the `--smoke` run was **not** performed. Every number below is either taken from the code or computed analytically from the model configurations.

---

## 0. Verdict first (no generosity)

| Dimension | Rating | Why |
|---|---|---|
| Mathematical soundness of the codec | **Strong** | Generalised KLT in a Fisher metric, Lagrangian (equal-slope) bit allocation over operational RD tables, closed-loop DPCM-style prediction. Each step is derived correctly (Sec. 4). |
| Evaluation rigour (harness, statistics) | **Strong** | Exact-identity self-tests, streaming-equivalence proof plus an empirical decode test, matched-rate rule (`rate_ok`), Holm correction, pre-registered cross-model rule with leave-one-family-out. This is better than most KV-compression papers. |
| Novelty of individual components | **Moderate / incremental** | Every building block has close prior art: pre-RoPE coding and Fisher sensitivity (KVQuant), inter-layer linear prediction of K/V (AQUA-KV; the code itself labels it "AQUA-KV-style"), PCA plus adaptive bit allocation for the KV cache (KVTC), sequential closed-loop calibration (GPTQ/BRECQ for weights). |
| Novelty of the combination | **Moderate–good** | A *single loss-calibrated Lagrange multiplier* that allocates bits jointly over layers × {K,V} × eigen-components of a Fisher-whitened, inter-layer-predicted, closed-loop residual is, as far as I know, not published as one method. It is a defensible *technique* contribution, not a new model. |
| Closing the "high memory cost" gap **as claimed by the current experiments** | **Weak** | Memory is computed analytically, never measured at long context. Codec side-parameters (up to ~1.4 GB fp32 for 8B models) are not charged. Decode latency is not measured and is about two to three orders of magnitude above attention cost per cached row. Evaluation context is 1,024 tokens, which is not where KV memory is the bottleneck. |
| Fairness against baselines | **Weak–moderate** | Calibration uses WikiText-2 *train* and the test uses WikiText-2 *test*: same domain, so this favours LCTC over calibration-free KIVI. The strongest direct competitors (KVQuant, AQUA-KV, KVTC) are absent. |

**Bottom line:** the codec is scientifically sound and the methodology is publication-grade. In its current form, though, the paper would be attacked on (i) novelty vs KVQuant + AQUA-KV + KVTC, (ii) the missing systems/memory evidence, and (iii) in-domain calibration. All three are fixable (Sec. 6.4). Nothing can be "100 % assured" before the runs (Sec. 6.5).

---

## 1. Ultimate objective and rationale

### 1.1 Objective
Reduce the memory of the KV cache, which grows as `2 · L · H_kv · d · T · bytes` and dominates inference memory at long context. Do it **without training**, at an average rate of 1.5–5 bits per cached element, while keeping next-token loss close to the fp16 cache. Then prove, under a pre-registered statistical rule, that this beats the matched-rate state of the art (KIVI) and training-free cross-layer sharing (CLA).

### 1.2 Rationale (why this code exists after `KV_LDT_v12_2`)
1. **The predecessor found a negative result.** Training-free cross-layer substitution (CLA-style reuse of layer *l−1*'s K/V at layer *l*) is catastrophic. The reproduced anchors in `PRIOR_PPL` show WikiText-2 PPL going from ≈ 7–11 (full) to 85–3,133 (CLA), e.g. Llama-3.2-1B 9.38 → 3,133.
2. **Diagnosis encoded in `coordinate_mismatch`.** Adjacent layers are *representationally similar* (high CKA) but *not coordinate-aligned*: `fvu_identity ≥ 1`, while an out-of-sample linear predictor achieves `fvu_linear ≪ 1`. Substitution therefore fails, but **prediction plus residual coding** can still exploit cross-layer redundancy.
3. **Treat the cache as a source-coding problem.** Pick the transform, the bit allocation and the error metric that minimise *expected loss increase* (not MSE) at a given rate. This is classical transform coding (KLT + Lagrangian allocation + DPCM) moved into the loss geometry of the network.

This narrative links the two studies into one argument ("why sharing fails → how to exploit the same redundancy correctly"). That is a strength; use it.

---

## 2. Research gaps (top-tier literature only)

| # | Gap | Closest top-tier evidence | What is actually missing |
|---|---|---|---|
| **G1** | **Wrong distortion metric.** KV quantisers minimise MSE (or per-group range error) in activation space, which is not loss. | KIVI (Liu et al., ICML 2024): per-channel K, per-token V, min/max, MSE-agnostic. KVQuant (Hooper et al., NeurIPS 2024): uses a **diagonal** Fisher to place non-uniform levels. FWSVD (Hsu et al., ICLR 2022): Fisher-weighted SVD for *weights*. OBS/OBD (Hassibi & Stork, NeurIPS 1993; LeCun et al., 1990): second-order loss model. | A **full per-head Fisher block** used as the *metric of the transform itself* (generalised KLT), so decorrelation and allocation are both done in loss units. KVQuant's diagonal Fisher cannot rotate the basis. |
| **G2** | **K and V (and layers) get budgets by heuristic.** | KIVI: same bits for K and V. KVTuner (ICML 2025): searches layer-wise K/V precision pairs. ZipCache (NeurIPS 2024): saliency-based mixed precision over tokens. | An **analytically optimal** (equal-slope, Shoham & Gersho 1988; Everett 1963) joint allocation across K, V, layers and components, in **commensurable loss units**. MSE units for K and V are not commensurable, which is why `Variant` forbids `euclid + joint_kv`. That check is correct. |
| **G3** | **Open-loop calibration.** Each layer's coder is fitted on clean activations, but at inference it sees activations already perturbed by earlier compressed layers (error propagation). | GPTQ (Frantar et al., ICLR 2023) and BRECQ (Li et al., ICLR 2021) do sequential, propagation-aware calibration **for weights**. KV methods (KIVI, KVQuant, Palu ICLR 2025, AQUA-KV ICML 2025) calibrate on clean or per-layer statistics. | Closed-loop, propagation-aware **KV-codec** calibration, where layer *l* is fitted on the activations it will actually see. *Caveat:* check AQUA-KV's exact training protocol before claiming full novelty. If its predictors are trained on quantised previous-layer inputs, G3 is partially covered. |
| **G4** | **Simulated compression ≠ deployed compression.** Many papers "fake-quantise" a full forward pass, which can leak future information or ignore the streaming order. | Common practice in KIVI, KVQuant and others; rarely proven equivalent. | A **proof** that the masked single-pass simulation equals streaming decoding (docstring of `masked_attention`) plus an **empirical** check with a real decoder that stores `uint8` codes (`StreamingLCTC`, `decode_equivalence`). This is a methodological contribution, not a model one. |
| **E1–E2** | **Why does training-free cross-layer sharing fail?** | MiniCache (NeurIPS 2024), CLA (Brandon et al., NeurIPS 2024, *trained*), KVSharer (arXiv 2024), xKV (arXiv 2025). Similarity is measured by cosine/CKA (Kornblith et al., ICML 2019). | Showing that similarity (CKA) does not imply substitutability (FVU of identity), while linear predictability does. This explains the failure, and the lowest-rate LCTC is contrasted against CLA. |

**Directly competing work you must cite and differentiate from (otherwise a reviewer will):**
- **KVQuant** (NeurIPS 2024): pre-RoPE key quantisation, Fisher-weighted non-uniform levels, dense-and-sparse outliers. This is ≈ 3 of your ingredients.
- **AQUA-KV** (Shutova et al., ICML 2025): linear predictors of K/V from the previous layer, residual quantisation. This is your `xlayer`.
- **KVTC** (Staniszewski & Łańcucki, 2025; arXiv, reported for ICLR 2026, so verify the venue): PCA decorrelation + dynamic-programming bit allocation + entropy coding for the KV cache. This is your transform + allocation, but in MSE units and without prediction or closed-loop fitting.
- **TurboQuant / QJL / PolarQuant** (Zandieh et al.; QJL at AAAI 2025): rotation + scalar quantisation with near-optimal distortion guarantees. This is a theoretical competitor to the KLT.
- **QuaRot** (NeurIPS 2024): Hadamard rotation for 4-bit KV.

---

## 3. Hypotheses / research questions the code operationalises

The code defines them precisely through `contrasts()` and `coordinate_mismatch()`:

| ID | Hypothesis (directional, A has lower WikiText-2 NLL than B at rate_A ≤ rate_B) | Contrast in code |
|---|---|---|
| **H0 (primary)** | LCTC at 3.0 bits beats KIVI-2 (effective 2 + 32/32 = 3.0 bits). LCTC at 5.0 bits beats KIVI-4 (5.0 bits). | `lctc_vs_kivi2`, `lctc_vs_kivi4` |
| **H1 / G1** | Fisher-metric coding beats Euclidean coding at matched rate (both with separate K/V budgets, so only the metric changes). | `G1_loss_metric`: `lctc_sep_kv` vs `lctc_euclid_sep` |
| **H2 / G2** | A joint K/V Lagrangian budget beats separate equal budgets. | `G2_joint_kv` |
| **H3 / G3** | Propagation-aware sequential calibration beats open-loop calibration. | `G3_propagation` |
| **H4** | Inter-layer prediction reduces loss at matched rate. | `xlayer_prediction` |
| **E1** | Adjacent layers: high CKA, `fvu_identity ≥ 1`, `fvu_linear ≪ 1`. | `coordinate_mismatch` |
| **E2** | LCTC at the lowest rate (1.5 bits) beats CLA (≈ 16 · (1 − reused/L) bits, i.e. ~9–10 bits) at a far lower rate. | `lowest_rate_vs_cla` |
| **P5** | The second-order model ½ E[eᵀFe] predicts observed ΔNLL (rank correlation). | `P5_spearman` in `run_model` |
| **G4** | Single-pass simulation equals the streaming decoder. | `decode_equivalence` |

The **decision rule** (`cross_model`) is pre-specified: Holm p < 0.05 **and** rate_A ≤ rate_B in ≥ 7/9 models **and** a one-sided Wilcoxon over models with p < 0.05, plus leave-one-family-out robustness. This is good practice. With n = 9 the minimum one-sided Wilcoxon p is 1/512 ≈ 0.002, so the test has power.

---

## 4. How each hypothesis is addressed scientifically (and whether the maths is right)

### 4.1 Signal model
For each layer *l*, rows are tokens: `x_K ∈ ℝⁿ` (**pre-RoPE**, via the exact `rope_invert`) and `x_V ∈ ℝⁿ`, with `n = H_kv · d`. Coding pre-RoPE keys removes the position-dependent rotation that destroys per-channel statistics. KVQuant showed this matters; your inverse is exact (`x = y·c̄/|c|²`) and is self-tested against the model's own RoPE.

### 4.2 Loss metric (G1). Correct
Second-order Taylor expansion of the summed NLL in a perturbation `e` of the cache, with first-order terms vanishing in expectation and the Hessian replaced by the Fisher (Gauss–Newton):

ΔNLL ≈ ½ · E_t[ e_tᵀ F e_t ],  F_{l,h} = (1/N) Σ_t g_t g_tᵀ,  g_t = ∂L/∂x_{l,h,t}

`compute_fisher` obtains `g_t` exactly through zero-valued leaves (`FisherProbe`) injected *pre-RoPE*, so the gradient is with respect to the quantity actually coded. Shrinkage `F ← (1−ε)F + ε·tr(F)/d·I` with ε = 0.05 keeps F well-conditioned.

*Mathematical check:* with `S = F^{1/2}` (symmetric), `y = x·S` gives `‖e_y‖² = e_x S Sᵀ e_xᵀ = e_xᵀ F e_x`. ✔ The synthesis `Fd = Uᵀ S⁻¹` exactly inverts the analysis `E = S U`. ✔

**Weaknesses you must state:**
1. This is the **empirical** Fisher (true labels), not the true Fisher (labels sampled from the model). Kunstner et al. (NeurIPS 2019) show the empirical Fisher can misrepresent curvature. Add a true-Fisher variant, or justify the choice.
2. Cross-token, cross-head and cross-layer curvature terms are dropped (block-diagonal per head). P5 tests this, but only on ≈ 8 points per model (Sec. 6.3).
3. ε = 0.05 is fixed, not the Ledoit–Wolf optimal shrinkage, so "Ledoit-Wolf-style" is the correct wording. Keep it.

### 4.3 Transform. Correct and optimal under its assumptions
`fit_part` diagonalises the covariance of the whitened residual, `C = E[yᵀy]`, which is equivalent to the generalised eigenproblem `Σ v = λ F⁻¹ v`. Under high-resolution, Gaussian assumptions the KLT is the optimal orthogonal transform for the weighted MSE (Huang & Schultheiss, 1963; Gersho & Gray, 1992). The coding gain over coding `x` directly is the AM/GM ratio of the eigenvalues; 6.02 dB of gain saves 1 bit per component. **You do not report this gain yet.** It is the cheapest strong *a priori* justification you can add (Sec. 6.4, item 6).

### 4.4 Bit allocation (G2). Correct
For each component *k* and bit-depth *b ∈ {0,1,2,3,4,5,6,8}*, `operational_rd` measures the actual distortion `D[k,b]` of a clipped uniform quantiser, with the clip searched over a 24-point geometric grid. Choosing `argmin_b D[k,b] + θ·b` with **one θ for every component, every layer and both K and V** (`joint_kv`) is the Lagrangian solution of `min Σ D s.t. Σ b ≤ R`. Its optimality condition is equal RD slopes (Shoham & Gersho, 1988). Because D is in *loss units*, one θ across K, V and layers is meaningful. It is the operational analogue of reverse water-filling. θ is found by bisection on log θ, then corrected by secant/Newton steps on the closed-loop rate. ✔

*Caveat:* the Lagrangian solves only points on the lower convex hull of the RD table (a known property). With 8 bit levels this costs little.

### 4.5 Inter-layer prediction (H4). Correct, closed-loop
K_l is predicted from `[K̂_{l−1}, V̂_{l−1}]` and V_l from `[K̂_{l−1}, V̂_{l−1}, K̂_l]` using **reconstructions**, not clean values. This is closed-loop DPCM (Jayant & Noll, 1984), so encoder and decoder never drift. The LMMSE predictor is metric-independent, because its error covariance is Loewner-minimal; the docstring's claim is true. The ridge α is chosen on a held-out set of whole sequences. ✔

### 4.6 Propagation-aware calibration (G3). Correct
`Calibrator.sequential` refits layer *l* on activations produced when layers *< l* are already compressed. The `open_loop` ablation computes reconstructions but attends exactly (`shadow=True`), so only the calibration distribution changes. This ablation is clean.

### 4.7 Outliers
Rows whose whitened residual energy exceeds the 0.999 quantile are stored exactly, and **their cost is charged** (`extra_bits`: 16 bits/element + a 32-bit index). The 1-bit-per-row flag is not charged; that is negligible (2/n bits/element) but should be stated.

### 4.8 Streaming equivalence (G4). Correct
Token *n*'s K/V at every layer depend only on tokens ≤ *n*. In the masked pass, query *m* reads exact K/V for sinks and the last `WINDOW` tokens and decoded K/V for everything else, which is exactly the streaming rule. `decode_equivalence` checks logits, argmax agreement and row-level code agreement against a real `uint8` decoder.

### 4.9 Statistics. Mostly correct
Paired sign-flip randomisation test plus a paired bootstrap over windows, Holm across contrasts, and the cross-model rule. **One issue:** WikiText windows overlap in *context* (window 1024, stride 512), although their scored tokens are disjoint. Units are therefore mildly dependent, and the bootstrap/sign-flip is slightly anti-conservative. Use a block bootstrap over consecutive windows, or report that the p-values survive it.

---

## 5. Novelty: what is genuinely new, what is incremental

**Type of contribution:** a **novel technique / method composition** with a theoretical justification, plus a **methodological contribution** (proven-equivalent evaluation harness and a pre-registered decision rule). It is **not a new model** and **not new theory**.

| Component | Novelty | Honest status |
|---|---|---|
| Pre-RoPE key coding | None | KVQuant (NeurIPS 2024) |
| Outlier rows kept exact | None | KVQuant dense-and-sparse; SKVQ |
| Inter-layer linear prediction | Low | AQUA-KV (ICML 2025) |
| PCA/KLT + adaptive bit allocation for the KV cache | Low–moderate | KVTC (2025) does this in MSE units |
| **Generalised KLT in a full per-head Fisher metric** | **Moderate** | Extends FWSVD (weights) and KVQuant (diagonal Fisher) to a full-block metric that *rotates* the basis |
| **One loss-unit Lagrange multiplier jointly over layers × K/V × components** | **Moderate–good** | Principled replacement for KVTuner-style search and KIVI's equal K/V bits. This is the strongest single selling point. |
| **Closed-loop + propagation-aware fitting of a KV codec** | **Moderate** | GPTQ/BRECQ ideas moved to KV coding; verify against AQUA-KV |
| CKA vs substitutability vs predictability (E1) | **Moderate (analysis)** | A new explanation of why training-free sharing fails |
| Proven single-pass ≡ streaming equivalence | **Methodological** | Rare in the KV literature |

**How to phrase the claim so it survives review:** *"We cast training-free KV-cache compression as rate–distortion-optimal transform coding in the loss metric of the network. A single Lagrange multiplier, expressed in units of expected NLL increase, allocates bits jointly across layers, keys/values and Fisher-whitened eigen-components of a closed-loop inter-layer prediction residual."* Do **not** claim that pre-RoPE coding, outlier handling or inter-layer prediction are new.

---

## 6. Have the gaps been closed? How to assess, and what can and cannot be assured

### 6.1 Gap-by-gap status

| Gap | Implemented? | Isolated by an ablation? | Closed? |
|---|---|---|---|
| G1 loss metric | ✔ | ✔ `lctc_sep_kv` vs `lctc_euclid_sep` | Only if H1 passes; the empirical-Fisher caveat remains |
| G2 joint allocation | ✔ | ✔ | Only if H2 passes |
| G3 propagation | ✔ | ✔ (clean `shadow` ablation) | Only if H3 passes |
| G4 streaming equivalence | ✔ proof + test | n/a | **Yes** for correctness; test length is only T = 96 |
| E1/E2 sharing failure | ✔ | ✔ | Yes as analysis, if fvu_identity ≥ 1 and fvu_linear ≪ 1 hold |
| **"High memory cost" (the stated objective)** | **Partly** | — | **No, not yet demonstrated** (Sec. 6.2) |

### 6.2 Critical issues for the memory claim (ordered by severity)

1. **Codec side-information is not charged.** Per layer, the stored parameters are about `S_K, S_V (2n²) + E, Fd (≤ 4n²) + P_K (2n²) + P_V (3n²) ≈ 11n²` floats. Computed from the model configs (r = n upper bound, fp32 as stored now):

   | Model | n | Codec params | fp32 | KV per token (fp16) | Break-even context @ 3 bits (fp32 / fp16 params) |
   |---|---|---|---|---|---|
   | Qwen2.5-1.5B | 256 | 20 M | 77 MiB | 28 KiB | ≈ 3.5k / 1.7k tokens |
   | Llama-3.2-1B | 512 | 46 M | 176 MiB | 32 KiB | ≈ 6.9k / 3.5k |
   | Llama-3.2-3B | 1024 | 323 M | 1.23 GiB | 112 KiB | ≈ 13.9k / 6.9k |
   | Llama-3.1-8B | 1024 | 369 M | 1.38 GiB | 128 KiB | ≈ 13.9k / 6.9k |

   The cost is shared across a batch and across requests, so it amortises, but it must be reported. Below the break-even point, **LCTC uses more memory than the fp16 cache.** Fixes: store the block-diagonal `S` as blocks (H·d² instead of n²), keep only the `r` active rows of `Fd`/columns of `E`, store everything in fp16, and make `P` low-rank.

2. **Decode compute and latency.** Every decode step reconstructs every compressed row of every layer: synthesis `r·n` plus the predictors `2n²` (K) and `3n²` (V). For n = 1024 that is ≈ 12.6 MFLOP per cached row per layer. Attention reading the same row costs ≈ 4·H_q·d ≈ 16 kFLOP (Llama-3.1-8B), so reconstruction is ≈ **800× the attention cost**. In addition, the inter-layer chain means layer *l* cannot be decoded without layer *l−1*'s reconstruction, which blocks fusing decode into attention the way KIVI and KVQuant do. Expect reviewers to ask for tokens/s. At minimum, report throughput and propose the cheaper deployable variant (`lctc_no_xlayer`, which has no chain).

3. **Peak memory during decode.** `StreamingLCTC` materialises the full-precision cache of a layer (and of the previous layer) at each step. The *storage* fraction is low, but the *transient peak* per layer is the uncompressed size. Report both.

4. **Context length.** PPL uses 1,024-token windows and passkey goes up to 4,096. The memory problem lives at 32k–128k. `mem_fraction_T32768` is analytic: ≈ 0.188 at 3 bits (5.3×) and ≈ 0.095 at 1.5 bits (10.5×), with codec parameters excluded. Add RULER (COLM 2024) or LongBench (ACL 2024) at ≥ 32k on Llama-3.1-8B, and a measured `torch.cuda.max_memory_allocated`.

5. **Packing is theoretical.** Codes are stored as `uint8`; `codes_bytes_packed` is computed, not implemented. Say so, or implement variable-width bit packing.

### 6.3 Other validity threats

- **In-domain calibration (serious).** Calibration on WikiText-2 train and testing on WikiText-2 test gives LCTC a domain advantage over KIVI, which needs no calibration. Calibrate on C4/FineWeb and test on WikiText-2, PG-19 and LongBench. Report both settings.
- **KIVI causality depends on `WINDOW ≥ KIVI_GROUP`.** In the single pass, a key group's min/max includes up to 31 later tokens. This is causal only because all of them fall inside the exact window. Default 32/32 is fine. **The smoke config (`WINDOW=8`) leaks future tokens into KIVI**, so smoke contrasts vs KIVI are invalid. Add `assert WINDOW >= KIVI_GROUP` in `Config.__post_init__`.
- **P5 is under-powered.** The Spearman correlation runs over ≈ 8 heterogeneous variants per model. A stronger test is to inject the codec at a single layer at several θ and regress observed on predicted ΔNLL, per layer.
- **Passkey is teacher-forced, not generated.** That is acceptable (it is equivalent for greedy decoding of the answer), but state it.
- **The CLA baseline is a strawman** for a "beats SOTA" claim. It is appropriate only for the E2 explanatory claim.

### 6.4 What to add so the results can carry the paper (priority order)

1. **Baselines:** KVQuant-3bit (pre-RoPE per-channel NUQ + 1 % outliers), AQUA-KV, and a **KVTC-like** variant. You can get the latter almost for free: `Variant(metric="euclid", joint_kv=False, xlayer=False, propagation=False)`, plus a DP allocation. If LCTC does not beat that variant, the paper's novelty collapses. Also add QuaRot-style Hadamard + per-token quantisation.
2. **Out-of-domain calibration** (C4 → WikiText/PG-19/LongBench).
3. **Charge codec parameters** and report the break-even context; compress the parameters (block `S`, active-only `E`/`Fd`, fp16, low-rank `P`).
4. **Long-context runs** (≥ 32k): RULER/LongBench, measured peak memory, tokens/s.
5. **Entropy coding of codes.** Report the empirical code entropy per component. KVTC gains substantially from it, and it is a free extra rate reduction that also strengthens your RD story. Optimising θ against entropy rather than fixed-length bits (ECSQ) would be a genuine extension.
6. **Report the KLT coding gain** `10·log10(AM/GM of λ)` per layer, in Euclidean vs Fisher metric, and the prediction gain `1/fvu_linear`. These are *a priori* quantitative justifications for H1 and H4 that hold before any PPL is measured.
7. **True Fisher vs empirical Fisher** ablation (Kunstner et al., 2019).
8. **Block bootstrap** over consecutive windows.
9. Run `decode_equivalence` at T ≥ 1,024, not 96, so the stored fraction reflects the asymptotic regime rather than sinks + window.

### 6.5 Can we "100 % assure" the expected results? **No.** Here is why, and what *is* justified

Nobody can honestly guarantee empirical outcomes before running the experiments. A paper that claims certainty is weaker, not stronger. What you **can** assert:

**Theoretically guaranteed, independent of data:**
- For the *calibration distribution* and the *quadratic loss model*, the Lagrangian allocation is optimal among allocations on the RD hull (Everett, 1963), and the generalised KLT is the optimal orthogonal transform under Gaussian high-resolution assumptions.
- Closed-loop prediction cannot drift: the encoder and decoder see identical inputs.
- The single-pass simulation equals streaming decoding (proof), up to floating-point reassociation.

**Strongly expected from prior evidence, but not guaranteed:**
- **H0 (LCTC > KIVI-2 at 3 bits):** likely. KVQuant at about 3 bits already beats KIVI-like uniform quantisers in PPL, and KVTC reports large gains from decorrelation plus allocation. KV activations are strongly anisotropic (Palu, Eigen Attention), so the KLT coding gain should be large.
- **H2 (joint K/V):** likely. Keys and values have very different sensitivities (KVTuner, KIVI's asymmetric design), so a common θ should move bits towards the more sensitive tensor.
- **H3 (propagation):** likely at low rates (1.5–2 bits), where error accumulation is large. **It may be null at 3–5 bits.** Pre-register that expectation.

**Genuinely uncertain:**
- **H1 (Fisher vs Euclid):** depends on how well 32 × 512 tokens estimate d×d = 16k-parameter blocks per head, and on empirical-vs-true Fisher. It could be small or null on some families.
- **H4 (inter-layer prediction):** depends on `fvu_linear`. If pre-RoPE keys are weakly predictable across layers, the gain will be small. E1 measures this directly, so read it before writing the claim.
- **P5:** the second-order model is often off by a constant factor at low rates. Expect good *rank* agreement and a poor *absolute* ratio.

**What becomes the backbone of the paper, depending on outcome:**
- If H0 + H2 + H3 pass under the 7/9 + Wilcoxon rule **with out-of-domain calibration and the KVQuant/KVTC baselines included**: the backbone is "loss-unit RD-optimal KV transform coding beats matched-rate SOTA", with G2 and G3 as the mechanisms.
- If H1 or H4 fail: report them honestly as null ablations. The paper still stands on the RD framework and on E1 (the explanation of why sharing fails).
- If H0 fails against KVTC-like or KVQuant: the contribution reduces to the analysis (E1/E2) and the evaluation methodology (G4). That is still publishable at a workshop or Findings level, not main-track.

---

## 7. References (core)

- Liu et al. **KIVI**: tuning-free asymmetric 2-bit quantisation for KV cache. *ICML 2024.*
- Hooper et al. **KVQuant**: towards 10 million context length LLM inference with KV cache quantisation. *NeurIPS 2024.*
- Shutova et al. **Cache Me If You Must** (AQUA-KV): adaptive key-value quantisation for LLMs. *ICML 2025.*
- Staniszewski & Łańcucki. **KV Cache Transform Coding** (KVTC). *arXiv 2025 / ICLR 2026 (verify).*
- Chang et al. **Palu**: KV-cache compression with low-rank projection. *ICLR 2025.*
- Ashkboos et al. **QuaRot**. *NeurIPS 2024.*
- He et al. **ZipCache**. *NeurIPS 2024.*
- Li et al. **KVTuner**: sensitivity-aware layer-wise mixed-precision KV cache quantisation. *ICML 2025.*
- Zandieh et al. **QJL**. *AAAI 2025*; **TurboQuant** *(2025, verify venue).*
- Liu et al. **MiniCache**. *NeurIPS 2024.* Brandon et al. **Cross-Layer Attention**. *NeurIPS 2024.*
- Xiao et al. **StreamingLLM** (attention sinks). *ICLR 2024.*
- Frantar et al. **GPTQ**. *ICLR 2023.* Li et al. **BRECQ**. *ICLR 2021.*
- Hassibi & Stork. **Optimal Brain Surgeon**. *NeurIPS 1993.* Hsu et al. **FWSVD**. *ICLR 2022.*
- Kunstner, Hennig & Balles. **Limitations of the empirical Fisher approximation**. *NeurIPS 2019.*
- Kornblith et al. **Similarity of neural network representations revisited** (CKA). *ICML 2019.*
- Huang & Schultheiss. Block quantisation of correlated Gaussian variables. *IEEE Trans. Comm. Sys., 1963.*
- Shoham & Gersho. Efficient bit allocation for an arbitrary set of quantisers. *IEEE Trans. ASSP, 1988.* Everett. Generalized Lagrange multiplier method. *Operations Research, 1963.*
- Gersho & Gray. *Vector Quantization and Signal Compression*, 1992. Jayant & Noll. *Digital Coding of Waveforms*, 1984.
- Mohtashami & Jaggi. Landmark attention (passkey). *NeurIPS 2023.* Hsieh et al. **RULER**. *COLM 2024.* Bai et al. **LongBench**. *ACL 2024.*

> Venue/authorship for the 2025 arXiv items (KVTC, TurboQuant, xKV, KVSharer) should be re-verified before submission.
