"""
QUASAR_KV.py — QUASAR: QUery-metric Adaptive Spectral Allocation of Rank and precision for the
KV cache, with an embedded code that is read directly (no reconstruction) at decode time.

A training-free, calibration-free KV-cache compressor for the compress-the-context protocol
(the context is compressed once, before the question is seen; NVIDIA kvpress' setting).  One
rule decides, for every cached token of every KV head, between DROP, EXACT (fp16) and one of the
layer's EMBEDDED LEVELS: a prefix of an order of (keys | values, spectral band, refinement stage)
steps in the head's own per-context spectral basis.  Token eviction, low rank and low precision
are points of one choice set under an exact per-sample memory budget, and the stored format is
the one the decoder reads: queries are mapped into the coefficient space once, every step of the
order is one table gather + one batched matmul over a PREFIX of the tokens, values are
aggregated in coefficient space and lifted once.  The evaluation engine (LongBench, LongBench-v2,
RULER, long PPL, official kvpress baselines, paired tests, decision rules) is the reviewed
engine of HEDGE_KV.py (v2), unchanged except where stated in section 6.

════════════════════════════════════════════════════════════════════════════════════════
1. WHY A NEW METHOD (evidence from HEDGE_KV v2 on SmolLM3-3B, LongBench qasper, 81 items)
════════════════════════════════════════════════════════════════════════════════════════
  * HedgeKV beat every official kvpress press by 14-19 points at 1/4-1/16 memory, and matched
    KIVI-4 at 2.5x less memory.  But its own ablations showed WHERE the gain comes from:
      - pricing tokens with the context's queries over its WHOLE length instead of the end
        window: +9.1 (1/8, p = 0.001) and +18.6 (1/16, p < 0.001) points;
      - importance-driven mixed precision instead of one precision: +13.6 (1/8);
      - its claimed novelty, the minimax over query groups, did NOT matter: hedge vs the
        average over groups -0.35 [-1.4, 0.5], p = 0.57 (72 of 81 items identical).
  * Its compression cost 101.7 s per sample (60x the prefill), almost constant in the context
    length: ~150 000 host synchronisations of a scalar bisection inside 50 Hedge rounds.
  * Quantising every coordinate of a token at one width wastes bits on directions that no
    query reads: a key's error matters only through q . e, a value's only through W_O e.
  * Our SnapKV replica scored 2.6-3.5 points below kvpress' SnapKV on SmolLM3: kvpress
    recomputes the window queries and applies RoPE in EVERY layer, but SmolLM3 has no RoPE in
    every 4th layer (its press scores those layers with rotated queries against unrotated keys).
    The replica uses the model's own queries.  Official presses remain the reported baselines.
  QUASAR keeps what worked (context-wide coverage, Prop. 1 pricing, per-token adaptivity),
  drops what did not (minimax), adds the RANK axis in a basis chosen for the metric that
  matters, makes the allocation a handful of batched passes, and stores a code whose decoding
  cost falls with its size.

════════════════════════════════════════════════════════════════════════════════════════
2. METHOD (one-shot, per layer, after the prefill of the context; all on the device)
════════════════════════════════════════════════════════════════════════════════════════
  Regions.  Sinks [0, N_SINK) and the last WINDOW context tokens stay exact; every other token
  of every KV head is coded.
  (a) Coverage statistics.  Groups of QUASAR_QUERIES query positions sampled in every
      QUASAR_SEGMENT-token segment of the context, plus the end window (reads at distance <
      QUASAR_LOCAL count only for the window group).  From each group's exact attention
      (probabilities p, output o, u_j = ||v_j - o||^2 in the residual-stream metric
      M_a = W_O,a^T W_O,a of query head a), every coded token gets
          drop     D_j = E[(p_j / (1 - p_j))^2 u_j]                         (Prop. 1, exact)
          key      D_j(e) ~ A_j e^T S e,   A_j = E[p_j^2 u_j]               (Prop. 1, 1st order)
          value    D_j(e) ~ W_j e^T Mbar e, W_j = E[sum_a p_{a,j}^2]
      with S the sensitivity-weighted query second moment and Mbar the read-weighted mean of
      the M_a ("decoupling": the token factor and the direction factor are estimated
      separately).  Groups are normalised by their drop-everything distortion and averaged.
  (b) Spectral bases.  Per head and side: c = (x - mu) L U with L = (metric + eps)^{1/2}
      (S for keys, Mbar for values) and U the eigenvectors of the importance-weighted second
      moment of the whitened vectors; coefficients are normalised by their RMS sigma_i and by a
      per-token fp16 scale rho.  The stored read-out matrix T = L^{-1} U diag(sigma) (fp16,
      only the columns used) turns stored coefficients back: x_hat = mu + (rho u_hat) T^T.
  (c) Embedded levels.  Bands of the basis (QUASAR_BANDS: 1/8, 1/8, 1/4, 1/2 of head_dim,
      leading first) x refinement stages (QUASAR_STAGES: 2, 2, 4 bits; stage t codes the
      residual of the previous stages with Lloyd-Max levels trained on it).  The metric error of
      every token after every (band, stage) is EXACT (Prop. 3); the layer's order of the
      (side, band, stage) steps is the greedy merge of the per-band chains by error removed per
      bit, summed over heads and tokens (level 1 = first stage of the leading band of keys AND
      values).  Level l of a token = the first l entries of that order.
  (d) Head-synchronous allocation.  In each head, tokens are ranked by importance (drop
      distortion + level-1 distortion); rank position j of every head receives the SAME option
      (drop | level 1..L | exact).  Every head therefore stores the same number of tokens per
      level (different tokens): all stored tensors are dense (Hkv, n_level, bytes) with no
      padding and no per-token index.  The options of the positions, and the deepest level used
      (which fixes how many columns of T are stored), minimise the summed distortion under the
      layer's budget (keep x the layer's fp16 context cache): Lagrangian relaxation with the
      multiplier bracketed by QUASAR_ROUNDS batched passes of 32 multipliers (Prop. 4); levels
      are then assigned in importance order (most important positions deepest).
  (e) Decoding (QuasarState.attend, the deployment path): q_tilde = q T_K once per KV head;
      for each key step, scores[:, :n_s] += q_tilde[band] . LUT[codes] (one 256-entry byte table
      maps a packed byte to its 8/b levels); scores = rho_K scores + q . mu_K; one softmax over
      exact rows and coded tokens; for each value step, acc[band] += (p rho_V)[:, :n_s] LUT[codes];
      out += acc T_V^T + (sum p) mu_V.  No key or value is reconstructed.  The coded part has two
      implementations: the reference (PyTorch: a gather + a batched matmul per step) and fused
      Triton kernels (one launch for the key scores and one for the value sums of a layer; they
      unpack the bytes in registers and read nothing but the packed code).  The kernels are used
      on CUDA only after reproducing the reference on the sample's own code (check_kernel).

════════════════════════════════════════════════════════════════════════════════════════
3. THEORY (every identity is checked numerically in quasar_tests at start-up)
════════════════════════════════════════════════════════════════════════════════════════
  Prop 1 (one entry, exact).  Changing ONLY logit j of a softmax by delta changes the output
  by G_j(delta)(v_j - o), G_j = p_j (e^delta - 1) / (1 + p_j (e^delta - 1)); delta -> -inf
  (eviction) gives -p_j / (1 - p_j).  A value error eps_j adds p_j eps_j exactly.
  Prop 2 (metric-optimal rank).  For keys k with second moment Sigma_k and queries with
  second moment S, the rank-r linear code minimising E[(q^T (k - k_hat))^2] keeps the leading
  r coordinates of k^T S^{1/2} U, U the eigenvectors of S^{1/2} Sigma_k S^{1/2} (Eckart-Young
  in whitened coordinates).  Credited: KQ-SVD (AISTATS 2026, calibrated, offline) and
  NOVA-KV (2608.04074, offline transforms) derive the same query metric; AttSVD (2610.06927)
  reports a per-prompt basis for attention logits.  QUASAR's basis is per context and head,
  weighted by token sensitivity, and also for values under the output-projection metric.
  Prop 3 (exact separability).  In an orthonormal basis of the metric space the metric error
  of any code is the sum of its coefficient errors, so the table of every token's error after
  every (band, stage) prefix is exact (not a model); only the decoupling in (a) approximates.
  Prop 4 (allocation).  For fixed levels, choosing one option per position under a bit budget
  is a multiple-choice knapsack; for every multiplier mu the per-position argmin of
  D + mu bits (with the best cap) is optimal for its own bit total (Everett, 1963), bits are
  non-increasing in mu, and the bracketing returns the smallest feasible mu found, so the
  returned allocation is feasible and optimal for the convexified problem up to the
  multiplier resolution.  The level order is optimal for the aggregate when every chain is
  convex (greedy merge; the idea of post-compression rate-distortion optimisation in EBCOT /
  JPEG 2000, Taubman 2000); the shared level histogram per position is a restriction (all
  heads store the same number of tokens per level) that buys dense decoding.
  Credited prior work on the other axes: per-token rank (VarRate 2607.15498; MixedDimKV
  2603.20616, plain key PCA, no bits), rank and bits per group (JoLT 2607.12550), bits and
  eviction by a multiple-choice knapsack (RDKV 2605.08317), progressive bit-prefix codes
  (ReadKV 2610.11245), nested rank per head (MatryoshkaKV 2410.14731), quantised pruning
  ("More Tokens, Lower Precision", Findings of EMNLP 2025), mixed precision by importance
  (MiKV, ZipCache).  Not found in the literature (searched Oct 2026; re-check before
  submission, and read AttSVD and VarRate in full): ONE per-token choice among eviction,
  embedded (rank x refinement) levels and exact storage, in a per-context metric-whitened
  basis for keys AND values, stored head-synchronously as rate-sorted prefixes so that decoding
  is dense matmuls over token prefixes in the coefficient domain.

════════════════════════════════════════════════════════════════════════════════════════
4. METHODS COMPARED AT MATCHED MEMORY (per sample: stored bits <= KEEP x fp16 context cache)
════════════════════════════════════════════════════════════════════════════════════════
  quasar          QUASAR (proposed), every budget
  quasar_native   QUASAR on the deployment path at PRIMARY_KEEP: SDPA prefill in the model
                  dtype, compression, the full cache freed, decoding from the packed code
  quasar_euclid   same, plain PCA basis and Euclidean error (the basis of the prior art)
  quasar_rot      same, a random rotation basis (no spectral compaction; HedgeKV-like codec)
  quasar_window   same, sensitivities from the end window only (SnapKV / RDKV-type prior)
  snapkv_q2/_q4   SnapKV selection + KIVI-2 / KIVI-4 codes at matched bytes ("More Tokens,
                  Lower Precision": the strongest simple eviction + quantisation hybrid)
  evict, am       chunk-local int8 eviction / Attention-Matching-lite (as in v2)
  snapkv, streamingllm   harness replicas of kvpress' presses
  kvpress:*       OFFICIAL NVIDIA kvpress 0.5.5 presses (SnapKV, PyramidKV, AdaKV-SnapKV,
                  Expected Attention, TOVA, Knorm, StreamingLLM) on the native HF path
  kivi2, kivi4    KIVI (Liu et al., ICML 2024), at their own rates
  Pre-registered contrasts: quasar vs every baseline at every budget; vs quasar_euclid,
  quasar_rot, quasar_window (the three ingredients) and KIVI at PRIMARY_KEEP; two-sided path
  checks quasar_native vs quasar, full vs native_full, snapkv vs kvpress:snapkv.  Decision
  rule: Holm per benchmark, then >= 7/9 models and a Wilcoxon test over models.

════════════════════════════════════════════════════════════════════════════════════════
5. SYSTEMS MEASUREMENTS (what each number is, and how faithful it is)
════════════════════════════════════════════════════════════════════════════════════════
  payload_bytes / cache_bytes_measured   QUASAR: bytes of the packed code (codes of every step,
                      fp16 scales, read-out matrices, means, level histogram) + exact rows;
                      equal to the allocator's bits / 8 (asserted in quasar_tests).
  compress_s          quasar_native: wall clock of the compression of all layers (GPU-synchronised).
  kernel, kernel_rel_err   quasar_native: whether the fused kernels were used for the sample and
                      their max relative deviation from the reference path.
  prefill_s, ttft_s, decode_s, tokens_per_s, peak_*_bytes   NATIVE rows only (kvpress presses,
                      native_full, quasar_native).  quasar_native runs the model eagerly through the
                      router (as the harness does), its coded attention through the kernels when
                      they passed the check; native_full uses HF's cache and fused SDPA.
  microbench.csv      ONE layer's decode-step attention at contexts MICRO_LENGTHS: full SDPA vs
                      SDPA over an evicting method's tokens (same bytes) vs QUASAR's compressed-
                      domain attention (quasar_torch: reference path; quasar_kernel: fused Triton
                      kernels, CUDA only, after the equivalence check), each captured in a CUDA
                      graph on GPU (launch overhead removed, as in a serving engine).
  attn_flops_first_step   analytic matmul FLOPs of the first decoded token (QUASAR: in the
                      coefficient domain, incl. the query map and the lift).
  harness_*           the fp32 reference harness (accuracy runs): never deployment speed.
  quasar_*            per-sample means over layers: kept (not dropped), exact, level (mean level
                      of coded tokens), coded_bits (per coded token and head), rank_k / rank_v
                      (stored basis columns), levels, fixed_frac (basis + histogram share of the
                      code), distortion (summed distortion / drop-everything).

════════════════════════════════════════════════════════════════════════════════════════
6. ENGINE CHANGES SINCE HEDGE_KV v2
════════════════════════════════════════════════════════════════════════════════════════
  * Resume reads bench_samples.csv with typed columns (identifiers and notes as strings,
    numbers as numbers).  A reader with keep_default_na=False turns every column with an empty
    cell into text and the budget check of the analysis then fails (keep_budget '0.125').
  * HedgeKV and its minimax solver are removed (quasar_rot is the closest in codec).
  * New: SnapKV + KIVI-b baseline, QUASAR deployment path, decode-attention microbenchmark
    (also standalone: --microbench), compress_s in the systems table.

Usage
    python QUASAR_KV.py --smoke                                   # self-tests + benchmark engine, CPU
    python QUASAR_KV.py --microbench 4096 16384 65536 131072 --micro-shape 128 8 4   # kernel timings
    python QUASAR_KV.py --check-data --suite longbench longbench-v2 ruler longppl
    python QUASAR_KV.py --suite longbench ruler --models Llama-3.1-8B-Instruct \\
           --tasks qasper hotpotqa trec repobench-p --ruler-lengths 4096 8192 16384
    python QUASAR_KV.py --suite longbench --models SmolLM3-3B --tasks qasper \\
           --methods quasar snapkv_q kvpress:snapkv kvpress:tova evict
    Useful: --keep 0.25 0.125 0.0625 --max-samples 50 --query-aware --allow-missing-kvpress
Requires torch >= 2.5 (SDPA enable_gqa), triton (optional: fused decode kernels), transformers >= 4.56 (< 5.3 for kvpress 0.5.5; tested on 5.2.0), scipy,
pandas, datasets, huggingface_hub, fuzzywuzzy or rapidfuzz, rouge, kvpress == 0.5.5.

VALIDATION (be explicit in any write-up; nothing below is a real-LLM result)
  Run offline on CPU only (Python 3.13, torch 2.14.1, transformers 5.2.0, kvpress 0.5.5,
  triton 3.8 in interpreter mode).  NOT run here: real LLMs, real Hugging Face datasets (no Hub
  access in the development environment), any GPU timing.  Every claim about LongBench / RULER
  accuracy and about speed is a hypothesis for the pre-registered decision rules.
  * quasar_tests: Lloyd-Max MSE 0.3634 / 0.1175 / 0.03455 / 0.009501 (Max 1960: 0.3634 / 0.1175 /
    0.03454 / 0.009497); embedded stages (2, 2, 4 bits) MSE 0.1182 / 0.0200 / 0.00060; packed bytes
    decode to their levels exactly through one byte table; Prop. 1 exact (3e-16, eviction
    included); Prop. 2 on queries and keys in different subspaces: logit MSE at rank 4 / 8 / 16 is
    0.166 / 0.049 / 0.0027 with the query-metric basis vs 0.506 / 0.286 / 0.071 with key PCA;
    Prop. 3: band error tables == direct metric error (relative gap 7e-8); allocator feasible,
    monotone in the budget, all-exact when affordable; compressed-domain decoding == harness
    reconstruction (max |diff| 7e-7); packed bytes == allocator bits / 8 exactly, and the decoder's
    measured bytes in bf16 == analytic exactly.
  * Fused kernels (Triton interpreter, CPU): scores and value sums == reference path within 4.7e-7
    relative (18-21 embedded steps, 1-5 query tokens, GQA ratio 2 and 4).  Their GPU speed is NOT
    measured here; on CUDA they are used per sample only after check_kernel passes.
  * --smoke (tiny character-level Llama, synthetic stand-ins of every benchmark; official kvpress
    presses through the pipes shim): harness == SDPA (1.3e-5); evict keep-all and QUASAR
    all-exact == full cache (0 for QUASAR); decoding from the compressed state == single pass for
    full, KIVI, SnapKV, StreamingLLM, SnapKV + KIVI-4, evict, AM and every QUASAR variant (1.000);
    full == HF generate(); native path == kvpress' official pipeline (1.000).  quasar_native agrees
    with its harness twin on 7 / 8 tokens: SDPA and the fp32 harness give keys that differ by
    ~1e-6, which flips near-tied allocation choices (up to 31% of the per-head importance ranks in
    the deepest layer, histograms within a few tokens); with the same allocation the two decoders
    agree to 7e-7.  QUASAR stores 0.2497 / 0.4992 of the cache at budgets 0.25 / 0.5.
  * Fidelity, prefix protocol (smoke model, budget 0.25; KL(full || method), lower is better):
    QUASAR 0.053 | quasar_euclid 0.067 | quasar_window 0.080 | quasar_rot 0.159 | evict 0.189 |
    AM-lite 0.698 | SnapKV + KIVI-2 0.719 | SnapKV + KIVI-4 1.086 | SnapKV 1.203 | StreamingLLM
    1.138; KIVI-2 0.086 at 0.395 memory, KIVI-4 0.0017 at 0.516; QUASAR at 0.5: 0.0069.  HedgeKV v2
    on the same model and protocol: 0.074.  A toy model: no accuracy claim is drawn from it.
  * Decode-attention microbenchmark, CPU, eager (no CUDA graphs, no kernels): full SDPA 15-34 us,
    QUASAR reference path 1.0-2.3 ms per step at 512-2048 tokens.  The reference path is
    launch-bound (one gather + one matmul per embedded step, ~20 steps per side): it is the
    correctness reference, NOT the speed path.  The speed claim rests on the fused kernels and
    is open until --microbench is run on a GPU.
  * Compression on CPU (8192 tokens, 8 KV heads, d = 128, one layer, 4 contended threads): ~25 s,
    dominated by the 4 batched 128 x 128 eigendecompositions; on GPU compress_s records it.
  * Resume: the typed reader of bench_samples.csv keeps numeric columns numeric; bench_analysis
    runs on the smoke CSV and on the SmolLM3 CSV of the HedgeKV v2 run.
"""

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import logging
import math
import re
import shlex
import string
import sys
import time
import types
import zlib
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import lru_cache
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from packaging.version import Version
from scipy import stats
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

if Version(transformers.__version__) < Version("4.56"):
    raise RuntimeError(f"transformers>=4.56 is required; found {transformers.__version__}")
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

# ════════════════════════════════════════════════════════════════════════════
# GLOBALS
# ════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(os.environ.get("KV_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
SEED = 42
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("quasar_kv")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = torch.device("cpu")
    COMPUTE_DTYPE = torch.float32
logger.info(f"device={DEVICE} compute_dtype={COMPUTE_DTYPE} torch={torch.__version__} "
            f"transformers={transformers.__version__}")

BIG = 1 << 40                      # "never" for completion / hiding times


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in keys])


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelConfig:
    name: str
    model_id: str
    family: str
    eval_batch: int = 4


@dataclass(frozen=True)
class ChunkMethod:
    """
    A chunk-local token rule applied to every completed chunk of every (layer, KV head): the
    `tokens` entries with the largest within-chunk attention share under the W_OBS queries
    that read the chunk locally are kept (evict: as int8 vectors; am: Attention-Matching
    keys, fitted biases and least-squares values, int8), the rest are dropped.
    """
    name: str
    family: str                    # "evict" | "am"
    target_bits: float             # bits per K/V element of the compacted region (<=)

    def __post_init__(self):
        if self.family not in ("evict", "am"):
            raise ValueError(f"unknown family {self.family!r}")


def chunk_method(name: str, family: str, bits: float) -> ChunkMethod:
    return ChunkMethod(name, family, bits)


# settings that do not change any per-unit result (where results go, which models / method
# subsets are run in one invocation, resuming); everything else enters the run hash
RESULT_NEUTRAL_FIELDS = ("OUTPUT_DIR", "BENCH_MODELS", "BENCH_METHODS", "RESUME", "ALLOW_MISSING_KVPRESS")


@dataclass
class Config:
    OUTPUT_DIR: str = str(Path(os.environ.get("KV_QUASAR_OUTPUT_DIR", PROJECT_ROOT / "quasar_results")))

    # ── Cache layout (all methods) ────────────────────────────────────
    N_SINK: int = 4
    WINDOW: int = 32               # exact recent window (incl. current token)
    CHUNK: int = 128               # tokens per chunk (chunk-local baselines)
    W_OBS: int = 32                # observed queries per chunk (chunk-local baselines); QUASAR's window group
    KEEP_BITS: int = 8             # precision of the chunk-local baselines' stored tokens

    # ── QUASAR ────────────────────────────────────────────────────────
    QUASAR_BANDS: Tuple[float, ...] = (0.125, 0.125, 0.25, 0.5)   # spectral bands (fractions of head_dim), leading first
    QUASAR_STAGES: Tuple[int, ...] = (2, 2, 4)   # embedded stage bits per coefficient (~2, ~4, ~8 bit after 1, 2, 3 stages)
    QUASAR_SEGMENT: int = 512      # context positions per coverage group
    QUASAR_QUERIES: int = 16       # sampled query positions per coverage group
    QUASAR_LOCAL: int = 64         # segment groups count reads at distance >= QUASAR_LOCAL (long-range attention)
    QUASAR_EPS: float = 0.05       # ridge of the metrics, as a fraction of their mean eigenvalue
    QUASAR_FLOOR: float = 1e-3     # sensitivity floor: tokens no query reads still prefer fidelity when bits are spare
    QUASAR_ROUNDS: int = 4         # batched bracketing rounds (32 multipliers each) of the Lagrange multiplier
    NATIVE_QUASAR: bool = True     # also run QUASAR on the deployment path (packed code, model dtype) at PRIMARY_KEEP
    QUASAR_KERNEL: bool = True     # fused Triton decode kernels on CUDA (enabled per sample only after matching the reference)
    SHARED_PREFILL: bool = True    # one exact context pass per sample for all one-shot methods
    MICRO_LENGTHS: Tuple[int, ...] = (4096, 16384, 65536)   # decode-attention microbenchmark contexts
    MICRO_REPEATS: int = 50

    # ── Baselines ─────────────────────────────────────────────────────
    KIVI_BITS: Tuple[int, ...] = (2, 4)
    MTLP_BITS: Tuple[int, ...] = (2, 4)   # SnapKV + KIVI-b hybrid (More Tokens, Lower Precision)
    KIVI_GROUP: int = 32
    CLA_REUSE: int = 2
    CLA_EXEMPT: float = 0.20
    AM_ITERS: int = 200
    AM_RIDGE: float = 1e-3
    ATTN_CHUNK: int = 256

    # ── Long-context benchmarks (Hugging Face) ────────────────────────
    # All default models use full attention in every layer (Mistral-7B-Instruct-v0.3:
    # sliding_window = null; Qwen3 / Qwen2.5 / SmolLM3: use_sliding_window = false); Runner
    # refuses any model whose config activates sliding-window layers before it is evaluated.
    BENCH_MODELS: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig("Llama-3.1-8B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Llama", 1),
        ModelConfig("Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-7B-Instruct", "Qwen", 1),
        ModelConfig("Mistral-7B-Instruct-v0.3", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral", 1),
        ModelConfig("Qwen3-8B", "Qwen/Qwen3-8B", "Qwen", 1),
        ModelConfig("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "Llama", 1),
        ModelConfig("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", "Qwen", 1),
        ModelConfig("SmolLM3-3B", "HuggingFaceTB/SmolLM3-3B", "SmolLM", 1),
        ModelConfig("Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", "Llama", 1),
        ModelConfig("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", "Qwen", 1),
    ])
    LONGBENCH_HF: str = "Xnhyacinth/LongBench"          # kvpress' parquet conversion
    LONGBENCH_ZIP_REPOS: Tuple[str, ...] = ("zai-org/LongBench", "THUDM/LongBench")   # official data.zip
    LONGBENCH_TASKS: Tuple[str, ...] = ("qasper", "hotpotqa", "trec", "repobench-p")
    LONGBENCH_E: bool = False                            # LongBench-E (length-balanced) variants
    LB_MAX_PROMPT_TOKENS: int = 31500                    # official LongBench max_length (32k-class models)
    LONGBENCH_V2_HF: str = "THUDM/LongBench-v2"          # official pred.py: split "train"
    LBV2_MAX_PROMPT_TOKENS: int = 120000                 # official LongBench-v2 max_len (128k models)
    LBV2_MAX_NEW_TOKENS: int = 128                       # official, without chain of thought
    RULER_HF: str = "simonjegou/ruler"                   # kvpress' RULER; config = context length
    RULER_LENGTHS: Tuple[int, ...] = (4096, 8192, 16384)
    RULER_MAX_PER_TASK: Optional[int] = 100
    LONG_PPL_DATASETS: Tuple[str, ...] = ("c4", "pg19")
    LONG_PPL_LEN: int = 16384
    LONG_PPL_EVAL: int = 1024                            # scored continuation after the compressed prefix
    LONG_PPL_N: int = 20
    C4_HF: str = "allenai/c4"
    C4_FILE: str = "en/c4-validation.00000-of-00008.json.gz"
    PG19_SOURCES: Tuple[Tuple[str, str], ...] = (("emozilla/pg19", "test"), ("emozilla/pg19-test", "test"))
    KEEP_FRACTIONS: Tuple[float, ...] = (0.25, 0.125, 0.0625)   # 4x, 8x, 16x smaller context cache
    PRIMARY_KEEP: float = 0.125
    SNAPKV_WINDOW: int = 64                              # kvpress SnapKVPress defaults (v0.5.5)
    SNAPKV_KERNEL: int = 5
    # official NVIDIA kvpress presses run as SOTA baselines on the native HF path; a real
    # benchmark run stops if kvpress cannot be imported (ALLOW_MISSING_KVPRESS overrides)
    KVPRESS_PRESSES: Tuple[str, ...] = ("snapkv", "pyramidkv", "adakv_snapkv", "expected_attention",
                                        "tova", "knorm", "streaming_llm")
    ALLOW_MISSING_KVPRESS: bool = False
    MEASURE_FLOPS: bool = True                           # FlopCounterMode over the first decode step
    QUERY_AWARE: bool = False                            # True: question is compressed with the context
    USE_CHAT_TEMPLATE: bool = True
    MAX_SAMPLES: Optional[int] = None                    # per task (None = all)
    BENCH_INCLUDE_CLA: bool = False
    BENCH_METHODS: Tuple[str, ...] = ()                  # name prefixes to run (empty = all); full always runs

    # ── Statistics ────────────────────────────────────────────────────
    N_BOOT: int = 2000
    N_PERM: int = 10000
    ALPHA: float = 0.05
    BUDGET_TOL: float = 0.02                             # benchmarks: measured memory_A <= memory_B (1 + tol)
    DECISION_MIN_MODEL_FRACTION: float = 7.0 / 9.0
    MIN_MODELS_WILCOXON: int = 6
    RESUME: bool = True
    SMOKE: bool = False

    def __post_init__(self):
        if self.WINDOW < 1 or self.W_OBS < 1 or self.W_OBS > self.WINDOW + self.CHUNK:
            raise ValueError("need WINDOW >= 1 and 1 <= W_OBS <= WINDOW + CHUNK")
        if self.WINDOW < self.KIVI_GROUP:
            # KIVI groups keys over tokens; a group is causal in the single pass only if all
            # of its tokens are inside the exact window until the group is complete.
            raise ValueError("WINDOW >= KIVI_GROUP is required for a causal KIVI simulation")
        if self.PRIMARY_KEEP not in self.KEEP_FRACTIONS:
            raise ValueError("PRIMARY_KEEP must be one of KEEP_FRACTIONS")
        if not set(self.QUASAR_STAGES) <= {1, 2, 4, 8} or abs(sum(self.QUASAR_BANDS) - 1.0) > 1e-9:
            raise ValueError("QUASAR_STAGES must be drawn from {1, 2, 4, 8} (byte-aligned packing) and "
                             "QUASAR_BANDS must sum to 1")
        if self.LONG_PPL_EVAL >= self.LONG_PPL_LEN:
            raise ValueError("LONG_PPL_EVAL must be shorter than LONG_PPL_LEN")
        # Results live under a hash of every result-relevant setting: resuming can never mix
        # outputs of different configurations (changing a setting starts a fresh directory).
        self.RUN_ID = self.config_hash()
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, "runs", self.RUN_ID)
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

    def config_hash(self) -> str:
        relevant = {k: v for k, v in asdict(self).items() if k not in RESULT_NEUTRAL_FIELDS}
        return hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()[:12]

    def kivi_effective_bits(self, bits: int) -> float:
        return bits + 32.0 / self.KIVI_GROUP


# ════════════════════════════════════════════════════════════════════════════
# MEMORY ACCOUNTING (exact, analytic; bits per K/V element of a compacted chunk)
# ════════════════════════════════════════════════════════════════════════════

def token_bits(d: int, keep_bits: int) -> int:
    """One stored token-like (k, v) pair: 2 d values, plus an fp16 absmax scale per vector."""
    return 2 * d * keep_bits + (32 if keep_bits < 16 else 0)


def entry_token_bits(method: ChunkMethod, d: int, keep_bits: int) -> int:
    """One stored token-like entry of a chunk: a kept token, or an AM compact token (+ fp16 bias)."""
    return token_bits(d, keep_bits) + (16 if method.family == "am" else 0)


def chunk_budget(method: ChunkMethod, d: int, C: int, keep_bits: int) -> Dict[str, float]:
    """Number of stored token entries per chunk that fits the target rate."""
    per_tok = entry_token_bits(method, d, keep_bits)
    m = max(-1, min(int(math.floor(method.target_bits * C * 2 * d / per_tok)), C))
    return {"tokens": m, "bits_per_element": max(m, 0) * per_tok / (C * 2 * d), "feasible": m >= 1}

# ════════════════════════════════════════════════════════════════════════════
# ATTENTION ROUTING
# ════════════════════════════════════════════════════════════════════════════

ROUTER_ATTN = "kv_router"
_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]


class Router:
    active = None


def router_attention(module, query, key, value, attention_mask, *args, **kwargs):
    ctl = Router.active
    if ctl is None:
        return _SDPA(module, query, key, value, attention_mask, *args, **kwargs)
    # every controller computes plain causal softmax attention: a sliding window (passed as a
    # kwarg by Mistral, as a module attribute by Qwen2/3 and SmolLM3) or a logit soft-cap
    # would be silently ignored, so they are refused here (Runner also checks the config)
    if kwargs.get("sliding_window") or getattr(module, "sliding_window", None) or kwargs.get("softcap"):
        raise RuntimeError(f"layer {module.layer_idx}: sliding-window / soft-capped attention is not supported")
    return ctl.attend(module, query, key, value, attention_mask, kwargs)


AttentionInterface.register(ROUTER_ATTN, router_attention)
AttentionMaskInterface.register(ROUTER_ATTN, sdpa_mask)


@contextmanager
def routed(controller):
    previous = Router.active
    Router.active = controller
    try:
        yield controller
    finally:
        Router.active = previous


def rep_heads(x: Optional[torch.Tensor], rep: int) -> Optional[torch.Tensor]:
    """KV head h serves query heads h*rep .. h*rep+rep-1 (HF repeat_kv order)."""
    return None if x is None else x.repeat_interleave(rep, dim=1)


# ════════════════════════════════════════════════════════════════════════════
# STORAGE FORMAT
# ════════════════════════════════════════════════════════════════════════════

FP16_MAX = 65504.0


def as_stored(x: torch.Tensor) -> torch.Tensor:
    """Round to the fp16 storage format (saturating), so every simulation reads exactly the
    values whose bytes are charged (scales, zero points, AM biases)."""
    return x.clamp(-FP16_MAX, FP16_MAX).to(torch.float16).to(x.dtype)


# ════════════════════════════════════════════════════════════════════════════
# CHUNK FINALISATION (shared by the simulation, the streaming decoder and diagnostics)
# ════════════════════════════════════════════════════════════════════════════

def absmax_quant(x: torch.Tensor, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-vector symmetric absmax quantisation: integer codes in [-qmax, qmax] and the fp16
    scale (charged in token_bits) that the dequantised vector is computed with."""
    qmax = 2 ** (bits - 1) - 1
    s = as_stored(x.abs().amax(-1, keepdim=True) / qmax).clamp_min(2.0 ** -24)   # >= smallest fp16
    return (x / s).round().clamp(-qmax, qmax), s


def fake_quant_tokens(x: torch.Tensor, bits: int) -> torch.Tensor:
    """What a reader of the stored token sees: codes x fp16 scale.  bits >= 16 keeps the vector
    as the model produced it, like the exact rows (bf16 / fp16 activations are representable
    in the fp16 storage format)."""
    if bits >= 16:
        return x
    codes, s = absmax_quant(x, bits)
    return codes * s


def group_queries(q: torch.Tensor, Hkv: int) -> torch.Tensor:
    """(B, Hq, t, d) -> (B, Hkv, rep * t, d): pool the query heads served by each KV head."""
    B, Hq, t, d = q.shape
    return q.view(B, Hkv, Hq // Hkv, t, d).reshape(B, Hkv, (Hq // Hkv) * t, d)


@torch.no_grad()
def finalize_chunk(k: torch.Tensor, v: torch.Tensor, qref: torch.Tensor, method: ChunkMethod,
                   plan: Dict, keep_bits: int, cfg: "Config") -> Dict:
    """
    k, v (B, Hkv, C, d) fp32 post-RoPE chunk; qref (B, Hkv, nq, d) scaled observed queries.
    Returns the chunk's token entries {k, v, b, idx} (kept tokens, or AM tokens).
    """
    B, H, C, d = k.shape
    m = plan["tokens"]
    scores = torch.einsum("bhqd,bhcd->bhqc", qref, k)
    share = scores.softmax(-1).mean(2)                                  # within-chunk attention share
    idx = torch.topk(share, m, dim=-1).indices.sort(-1).values
    if method.family == "am":
        am = attention_matching(k, v, qref, scores, idx, cfg)
        return {"k": fake_quant_tokens(am["k"], keep_bits), "v": fake_quant_tokens(am["v"], keep_bits),
                "b": as_stored(am["b"]), "idx": idx}
    gk = torch.gather(k, 2, idx[..., None].expand(B, H, m, d))
    gv = torch.gather(v, 2, idx[..., None].expand(B, H, m, d))
    return {"k": fake_quant_tokens(gk, keep_bits), "v": fake_quant_tokens(gv, keep_bits),
            "b": torch.zeros((B, H, m), device=k.device), "idx": idx}

@torch.no_grad()
def attention_matching(k, v, qref, scores, idx, cfg: "Config") -> Dict:
    """
    Attention-Matching-lite (after Zweiger et al., ICML 2026), per chunk and KV head:
    keys = top-t original keys by observed attention; multiplicative weights w >= 0 by
    NNLS on the chunk's attention mass Z(q) (projected gradient); values by ridge least
    squares on the chunk's attention outputs.  Reference queries = observed window queries
    (the paper's stronger repeat-prefill / self-study references are not used here).
    """
    B, H, C, d = k.shape
    t = idx.shape[-1]
    shift = scores.amax(-1, keepdim=True)
    Eall = (scores - shift).exp()                                      # (B,H,nq,C)
    Z = Eall.sum(-1)                                                   # (B,H,nq)
    E = torch.gather(Eall, 3, idx[:, :, None, :].expand(B, H, Eall.shape[2], t))
    G = E.transpose(-1, -2) @ E                                        # (B,H,t,t)
    b = (E.transpose(-1, -2) @ Z[..., None])[..., 0]
    L = torch.linalg.matrix_norm(G, ord=2).clamp_min(1e-30)[..., None]
    w = torch.full((B, H, t), 1.0, device=k.device) * (Z.mean(-1, keepdim=True) / E.sum(-1).mean(-1, keepdim=True).clamp_min(1e-30))
    for _ in range(cfg.AM_ITERS):
        w = (w - ((G @ w[..., None])[..., 0] - b) / L).clamp_min(0.0)
    alpha = E * w[:, :, None, :]
    alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-30)
    O = (Eall / Z[..., None]) @ v                                     # chunk-local outputs (B,H,nq,d)
    Ga = alpha.transpose(-1, -2) @ alpha
    Ga = Ga + cfg.AM_RIDGE * Ga.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * \
        torch.eye(t, device=k.device) + 1e-12 * torch.eye(t, device=k.device)
    Vn = torch.linalg.solve(Ga, alpha.transpose(-1, -2) @ O)
    kk = torch.gather(k, 2, idx[..., None].expand(B, H, t, d))
    return {"k": kk, "v": Vn, "b": w.clamp_min(1e-30).log()}


def chunk_plan(T: int, cfg: "Config", horizon: Optional[int] = None) -> List[Tuple[int, int, int]]:
    """(start, end, tau) of every chunk compacted before position min(T, horizon) (the
    horizon freezes compression at the end of a context; later tokens stay exact)."""
    H = T if horizon is None else min(T, horizon)
    out, s = [], cfg.N_SINK
    while s + cfg.CHUNK - 1 < H:
        e = s + cfg.CHUNK - 1
        tau = e + cfg.WINDOW
        if tau > H - 1:
            break
        out.append((s, e, tau))
        s += cfg.CHUNK
    return out


def chunked_attention(q, scaling, q_pos, K, V, kv_pos, hide_at, tok, chunk) -> torch.Tensor:
    """
    Attention over two kinds of entries, one softmax:
      exact tokens  (K, V at kv_pos), readable by query m iff kv_pos <= m < hide_at
      token entries logit q.k + b, value v, visible iff tau <= m (b = -inf: padding)
    hide_at is (Tk,) or per KV head (B, Hkv, Tk) (head-wise eviction such as SnapKV).
    q (B, Hq, Tq, d) -> out (B, Tq, Hq, d).
    """
    B, Hq, Tq, d = q.shape
    rep = Hq // K.shape[1]
    Kr, Vr = rep_heads(K.float(), rep), rep_heads(V.float(), rep)
    dev = q.device
    kv_pos, hide_at = kv_pos.to(dev), hide_at.to(dev)
    if hide_at.dim() == 3:
        hide_at = rep_heads(hide_at, rep)[:, :, None, :]                # (B, Hq, 1, Tk)
    if tok is not None:
        tk, tv, tb = rep_heads(tok["k"], rep), rep_heads(tok["v"], rep), rep_heads(tok["b"], rep)
        ttau = tok["tau"].to(dev)
    out = torch.empty((B, Hq, Tq, d), dtype=torch.float32, device=dev)
    for s in range(0, Tq, chunk):
        qs = q[:, :, s:s + chunk].float() * scaling
        qp = q_pos[s:s + chunk].to(dev)
        readable = (kv_pos[None, :] <= qp[:, None]) & (qp[:, None] < (hide_at if hide_at.dim() == 4
                                                                       else hide_at[None, :]))
        blocks = [(qs @ Kr.transpose(-1, -2)).masked_fill(~readable, float("-inf"))]
        if tok is not None:
            blocks.append((qs @ tk.transpose(-1, -2) + tb[:, :, None]).masked_fill(
                ~(ttau[None, :] <= qp[:, None]), float("-inf")))
        p = torch.cat(blocks, -1).softmax(-1)
        T0 = Kr.shape[2]
        o = p[..., :T0] @ Vr
        if tok is not None:
            o = o + p[..., T0:] @ tv
        out[:, :, s:s + chunk] = o
    return out.transpose(1, 2).to(q.dtype).contiguous()


# ════════════════════════════════════════════════════════════════════════════
# LAYER METHODS (one object per evaluated configuration)
# ════════════════════════════════════════════════════════════════════════════

class LayerState:
    """
    Compressed cache of one layer after a frozen prefill, ready for decoding: exact rows in
    a pre-allocated buffer with per-KV-head liveness (hide), plus the visible token entries.
    New tokens are appended exact (all methods), as in the compress-the-context-once
    protocol of KV-compression benchmarks.
    """

    def __init__(self, K, V, pos, hide, tok, extra: int, shared: Optional[int] = None):
        # keep only rows readable by at least one KV head: decode memory and FLOPs then scale
        # with the compressed cache (head-wise evicted rows remain masked within kept columns)
        cols = (hide.to(K.device) >= BIG).any(0).any(0)
        K, V, pos, hide = K[:, :, cols], V[:, :, cols], pos.to(K.device)[cols], hide.to(K.device)[:, :, cols]
        B, H, n, d = K.shape
        cap = n + extra
        self.K = K.new_zeros((B, H, cap, d))
        self.V = V.new_zeros((B, H, cap, d))
        self.K[:, :, :n], self.V[:, :, :n] = K, V
        self.pos = torch.full((cap,), BIG, dtype=torch.long, device=K.device)
        self.pos[:n] = pos.to(K.device)
        self.hide = torch.zeros((B, H, cap), dtype=torch.long, device=K.device)
        self.hide[:, :, :n] = hide.to(K.device)
        self.n, self.shared = n, shared
        self.tok = None if tok is None else {**tok, "tau": torch.full_like(tok["tau"], -1)}

    def append(self, k, v, pos):
        t = k.shape[2]
        if self.n + t > self.K.shape[2]:
            raise RuntimeError("decode buffer exhausted")
        self.K[:, :, self.n:self.n + t], self.V[:, :, self.n:self.n + t] = k, v
        self.pos[self.n:self.n + t] = pos
        self.hide[:, :, self.n:self.n + t] = BIG
        self.n += t

    def attend(self, q, scaling, qpos, chunk):
        n = self.n
        return chunked_attention(q, scaling, qpos, self.K[:, :, :n], self.V[:, :, :n], self.pos[:n],
                                 self.hide[:, :, :n], self.tok, chunk)


class LayerMethod:
    """
    A KV-cache policy.  entries() returns the cache a layer reads during a single forward:
    exact K/V with hide_at (when a row stops being readable) plus token entries.  freeze_at
    = H freezes compression at position H (benchmarks: the end of the context); with
    capture_extra set, the state at the end of the forward is kept for decoding.
    """
    name = "full"
    family = "global"
    one_shot = True                 # compresses only at the horizon: can reuse a shared prefill

    def __init__(self, cfg: "Config"):
        self.cfg = cfg
        self.freeze_at: Optional[int] = None
        self.capture_extra: Optional[int] = None
        self.states: Dict[int, LayerState] = {}

    def begin(self):
        self.states = {}

    def horizon(self, T: int) -> int:
        return T if self.freeze_at is None else min(T, self.freeze_at)

    def entries(self, li, q, k, v, scaling, pos):
        return k, v, torch.full((k.shape[2],), BIG, dtype=torch.long), None

    def entry_bits(self, kind: str, d: int) -> float:
        """Storage of one entry of a kind ('exact', 'tok') per KV head, in bits."""
        return 2 * d * 16.0 if kind == "exact" else 0.0

    def layer(self, li, q, k, v, scaling, pos) -> torch.Tensor:
        K, V, hide, tok = self.entries(li, q, k, v, scaling, pos)
        if self.capture_extra is not None:
            self.capture(li, K, V, pos, hide, tok)
        return chunked_attention(q, scaling, pos, K, V, pos, hide, tok, self.cfg.ATTN_CHUNK)

    def capture(self, li, K, V, pos, hide, tok, shared=None):
        T = K.shape[2]
        B, H = K.shape[0], K.shape[1]
        hide = hide.to(K.device)
        hide = hide.expand(B, H, T) if hide.dim() == 1 else hide
        live = hide >= BIG                                          # readable from now on
        alive_hide = torch.where(live, torch.full_like(hide, BIG), torch.zeros_like(hide))
        if tok is not None:                     # entries visible to the next position T
            vis = (tok["tau"].to(K.device) <= T)
            tok = {"k": tok["k"][:, :, vis], "v": tok["v"][:, :, vis], "b": tok["b"][:, :, vis],
                   "tau": tok["tau"][vis.cpu()]}
            if tok["k"].shape[2] == 0:
                tok = None
        self.states[li] = LayerState(K, V, pos, alive_hide, tok, self.capture_extra, shared)

    def state_bits(self, d: int, context_len: int) -> Dict[str, float]:
        """Stored bits for the CONTEXT part of the captured state (all layers, KV heads)."""
        exact = tok = 0.0
        for li, st in self.states.items():
            if st.shared is not None:
                continue
            ctx = (st.pos[:st.n] < context_len)[None, None, :] & (st.hide[:, :, :st.n] >= BIG)
            exact += float(ctx.sum()) * self.entry_bits("exact", d)
            if st.tok is not None:                              # captured entries are all visible
                tok += float(torch.isfinite(st.tok["b"]).sum()) * self.entry_bits("tok", d)
        return {"exact": exact, "tok": tok, "total": exact + tok}

class KIVIMethod(LayerMethod):
    family = "kivi"
    one_shot = False                # quantises while prefilling (rows leave the window)

    def __init__(self, cfg, bits: int):
        super().__init__(cfg)
        self.bits, self.name = bits, f"kivi{bits}"
        self.stored: Dict[int, Dict] = {}

    def begin(self):
        super().begin()
        self.stored = {}

    def entry_bits(self, kind, d):
        return 2 * d * (16.0 if kind == "exact" else self.cfg.kivi_effective_bits(self.bits))

    def entries(self, li, q, k, v, scaling, pos):
        c = self.cfg
        H = self.horizon(k.shape[2])
        kq = kivi_quant(k, self.bits, c.KIVI_GROUP, True, c.N_SINK)
        vq = kivi_quant(v, self.bits, c.KIVI_GROUP, False, c.N_SINK)
        # a token is read exact until it leaves the window (n + W), then from its quantised
        # copy; with a horizon H, tokens still inside the window at H stay exact for good.
        # Every group read quantised is complete when its first token is (WINDOW >= KIVI_GROUP).
        pc = pos.cpu()
        q_at = pc + c.WINDOW
        quant = (pc >= c.N_SINK) & (q_at <= H - 1)
        if self.capture_extra is not None:             # the stored codes of the quantised rows
            n_q = int(quant.sum())
            n_g = -(-n_q // c.KIVI_GROUP)
            self.stored[li] = {"n_q": n_q,
                               "k": {"codes": kq["codes"][:, :, :n_q], "lo": kq["lo"][:, :, :n_g],
                                     "scale": kq["scale"][:, :, :n_g]},
                               "v": {x: vq[x][:, :, :n_q] for x in ("codes", "lo", "scale")}}
        hide = torch.where(quant, q_at, torch.full_like(pc, BIG))
        b = torch.zeros(k.shape[:3], device=k.device)
        b[:, :, ~quant.to(k.device)] = float("-inf")
        tok = {"k": kq["xhat"], "v": vq["xhat"], "b": b, "tau": torch.where(quant, q_at, torch.full_like(pc, BIG))}
        return k, v, hide, tok


def kivi_quant(x: torch.Tensor, bits: int, group: int, along_tokens: bool, n_sink: int) -> Dict:
    """
    KIVI (Liu et al., ICML 2024): asymmetric min / max quantisation of keys per channel over
    groups of `group` consecutive tokens, and of values per token over groups of `group`
    channels; fp16 zero point (lo) and scale per group; rows before n_sink stay exact.
    Returns the dequantised tensor "xhat" (what the model reads) and, for the rows from
    n_sink on, the uint8 "codes" and the fp16 "lo" / "scale" of every group (what is stored),
    so the measured payload is computed from exactly the codes the simulation reads.
    """
    qmax = 2 ** bits - 1
    y = x.float()
    body = y[:, :, n_sink:]
    B, H, n, d = body.shape
    if along_tokens:
        n_g = -(-n // group)
        pad = n_g * group - n
        if pad:                    # replicating the last row leaves a partial group's min / max unchanged
            body = torch.cat([body, body[:, :, -1:].expand(B, H, pad, d)], 2)
        g = body.reshape(B, H, n_g, group, d)
        red = 3
    else:
        width = group if d % group == 0 else d
        g = body.reshape(B, H, n, d // width, width)
        red = 4
    lo = as_stored(g.amin(red, keepdim=True))
    sc = as_stored((g.amax(red, keepdim=True) - lo) / qmax).clamp_min(2.0 ** -24)
    codes = ((g - lo) / sc).round().clamp(0, qmax)
    deq = (codes * sc + lo).reshape(B, H, -1, d)[:, :, :n]
    return {"xhat": torch.cat([y[:, :, :n_sink], deq], 2),
            "codes": codes.reshape(B, H, -1, d)[:, :, :n].to(torch.uint8),
            "lo": lo, "scale": sc}


class CLAMethod(LayerMethod):
    """Training-free cross-layer sharing, exactly KV_LDT_v12_2's `full` (all positions)."""
    family = "cla"
    name = "cla"
    one_shot = False

    def __init__(self, cfg, L: int):
        super().__init__(cfg)
        ec = min(int(math.ceil(L * cfg.CLA_EXEMPT)), L - 1)
        self.map = {t: ec + ((t - ec) // cfg.CLA_REUSE) * cfg.CLA_REUSE
                    for t in range(ec, L) if (t - ec) % cfg.CLA_REUSE}
        self.L = L
        self.cache = {}

    def begin(self):
        super().begin()
        self.cache = {}

    def layer(self, li, q, k, v, scaling, pos):
        if li in set(self.map.values()):
            self.cache[li] = (k, v)
        if li in self.map:
            k, v = self.cache[self.map[li]]
        K, V, hide, tok = LayerMethod.entries(self, li, q, k, v, scaling, pos)
        if self.capture_extra is not None:
            self.capture(li, K, V, pos, hide, tok, shared=self.map.get(li))
        return chunked_attention(q, scaling, pos, K, V, pos, hide, tok, self.cfg.ATTN_CHUNK)


class ChunkedMethod(LayerMethod):
    """Chunk-local baselines (evict = quantised pruning, am = Attention-Matching-lite) at matched memory."""
    family = "chunked"
    one_shot = False

    def __init__(self, cfg, method: ChunkMethod, d: int, keep_bits: Optional[int] = None):
        super().__init__(cfg)
        self.m, self.name, self.family = method, method.name, method.family
        self.keep_bits = cfg.KEEP_BITS if keep_bits is None else keep_bits
        self.plan = chunk_budget(method, d, cfg.CHUNK, self.keep_bits)

    def entry_bits(self, kind, d):
        return 2 * d * 16.0 if kind == "exact" else entry_token_bits(self.m, d, self.keep_bits)

    def entries(self, li, q, k, v, scaling, pos):
        """
        Finalise every chunk compacted before the horizon in ONE batched call (chunks are
        stacked on the batch axis; finalize_chunk is batch-generic), then unfold the entries
        in chunk order.  A chunk compacted at tau is read exact by queries < tau.
        """
        c = self.cfg
        plan_T = chunk_plan(k.shape[2], c, self.freeze_at)
        B, Hkv, n, d = k.shape
        hide = torch.full((n,), BIG, dtype=torch.long)
        if not plan_T:
            return k, v, hide, None
        nch = len(plan_T)
        kc = torch.cat([k[:, :, s:e + 1] for s, e, _ in plan_T]).float()          # (nch*B, H, C, d)
        vc = torch.cat([v[:, :, s:e + 1] for s, e, _ in plan_T]).float()
        qref = torch.cat([group_queries(q[:, :, tau - c.W_OBS + 1: tau + 1].float() * scaling, Hkv)
                          for _, _, tau in plan_T])
        t = finalize_chunk(kc, vc, qref, self.m, self.plan, self.keep_bits, c)
        taus = torch.tensor([tau for _, _, tau in plan_T], dtype=torch.long)
        for s_, e_, tau in plan_T:
            hide[s_:e_ + 1] = tau

        def unfold(x):                       # (nch*B, H, a, ...) -> (B, H, nch*a, ...)
            y = x.reshape(nch, B, *x.shape[1:])
            y = y.permute(1, 2, 0, *range(3, y.dim()))
            return y.reshape(B, Hkv, nch * x.shape[2], *x.shape[3:])
        tok = {x: unfold(t[x]) for x in ("k", "v", "b")}
        tok["tau"] = taus.repeat_interleave(t["k"].shape[2])
        return k, v, hide, tok

def snapkv_scores(q, k, H: int, scaling, cfg: "Config") -> torch.Tensor:
    """
    SnapKV (Li et al., NeurIPS 2024) scores, replicated from NVIDIA kvpress' SnapKVPress (v0.5.5):
    the last SNAPKV_WINDOW context queries attend to the context (causal, fp32 softmax); the
    window columns are dropped; scores = mean over the window queries, avg_pool1d (kernel
    SNAPKV_KERNEL, padding k//2, count_include_pad as torch's default), averaged over the query
    heads of each KV group; the window gets max + 1 so it is kept.  Returns (B, Hkv, H).
    The queries are the model's own (post-RoPE where the layer applies RoPE): kvpress instead
    recomputes them from the hidden states and applies RoPE in EVERY layer, which differs on
    models with RoPE-free layers (SmolLM3 uses no RoPE in every 4th layer).
    """
    B, Hkv = k.shape[0], k.shape[1]
    w = cfg.SNAPKV_WINDOW
    rep = q.shape[1] // Hkv
    qo = q_at(q, torch.arange(H - w, H)).float()
    kk = rep_heads(k[:, :, :H].float(), rep)
    att = (qo @ kk.transpose(-1, -2)) * scaling                         # (B,Hq,w,H)
    att = att + torch.triu(torch.full_like(att, float("-inf")), diagonal=H - w + 1)
    att = att.softmax(-1)[..., :-w]                                    # (B,Hq,w,H-w)
    sc = att.mean(-2)
    sc = F.avg_pool1d(sc, kernel_size=cfg.SNAPKV_KERNEL, padding=cfg.SNAPKV_KERNEL // 2, stride=1)
    sc = sc.view(B, Hkv, rep, H - w).mean(2)
    return F.pad(sc, (0, w), value=float(sc.max()) + 1)


class SnapKVMethod(LayerMethod):
    """
    SnapKV, one-shot at the horizon (compression after prefill): the top n_kept = max(1, int(P *
    keep)) positions per KV head by snapkv_scores survive (kvpress' compute_n_kept).  Queries
    before the horizon use full attention.
    """
    family = "snapkv"

    def __init__(self, cfg, keep: int, name: str):
        super().__init__(cfg)
        self.keep, self.name = keep, name

    def entries(self, li, q, k, v, scaling, pos):
        T = k.shape[2]
        H = self.horizon(T)
        B, Hkv = k.shape[0], k.shape[1]
        if self.freeze_at is None or self.keep >= H or H <= self.cfg.SNAPKV_WINDOW:
            return LayerMethod.entries(self, li, q, k, v, scaling, pos)
        top = snapkv_scores(q, k, H, scaling, self.cfg).topk(self.keep, dim=-1).indices
        kept = torch.zeros((B, Hkv, T), dtype=torch.bool, device=k.device)
        kept.scatter_(-1, top, True)
        kept[..., H:] = True                                               # post-horizon tokens
        hide = torch.where(kept, torch.full((B, Hkv, T), BIG, dtype=torch.long, device=k.device),
                           torch.full((B, Hkv, T), H, dtype=torch.long, device=k.device))
        return k, v, hide, None


class SnapKVQuantMethod(LayerMethod):
    """
    "More Tokens, Lower Precision" (Zhang et al., Findings of EMNLP 2025; quantised pruning) with
    SnapKV selection, the strongest simple hybrid of eviction and quantisation: the last
    SNAPKV_WINDOW context tokens stay exact (fp16); of the others, the top tokens by
    snapkv_scores are stored with KIVI-b codes (keys per channel over groups of KIVI_GROUP kept
    tokens, values per token over groups of KIVI_GROUP channels; fp16 zero point and scale per
    group), as many as keep x the fp16 context cache allows.
    """
    family = "mtlp"

    def __init__(self, cfg, keep: float, bits: int, name: str, P: int, d: int):
        super().__init__(cfg)
        self.bits, self.name = bits, name
        w = cfg.SNAPKV_WINDOW
        per_tok = 2 * d * cfg.kivi_effective_bits(bits)
        self.n_q = int(math.floor((keep * P * 2 * d * 16.0 - min(w, P) * 2 * d * 16.0) / per_tok))
        self.stored: Dict[int, Dict] = {}

    def begin(self):
        super().begin()
        self.stored = {}

    def entry_bits(self, kind, d):
        return 2 * d * (16.0 if kind == "exact" else self.cfg.kivi_effective_bits(self.bits))

    def entries(self, li, q, k, v, scaling, pos):
        c = self.cfg
        T = k.shape[2]
        H = self.horizon(T)
        B, Hkv, _, d = k.shape
        if self.freeze_at is None or H <= c.SNAPKV_WINDOW:
            return LayerMethod.entries(self, li, q, k, v, scaling, pos)
        n_q = max(0, min(self.n_q, H - c.SNAPKV_WINDOW))
        hide = torch.full((B, Hkv, T), BIG, dtype=torch.long, device=k.device)
        hide[:, :, :H - c.SNAPKV_WINDOW] = H
        if n_q == 0:
            return k, v, hide, None
        sc = snapkv_scores(q, k, H, scaling, c)[..., :H - c.SNAPKV_WINDOW]
        idx = sc.topk(n_q, dim=-1).indices.sort(-1).values                 # (B, Hkv, n_q), position order
        gk = k.float().gather(2, idx[..., None].expand(B, Hkv, n_q, d))
        gv = v.float().gather(2, idx[..., None].expand(B, Hkv, n_q, d))
        kq = kivi_quant(gk, self.bits, c.KIVI_GROUP, True, 0)
        vq = kivi_quant(gv, self.bits, c.KIVI_GROUP, False, 0)
        if self.capture_extra is not None:
            self.stored[li] = {"k": {x: kq[x] for x in ("codes", "lo", "scale")},
                               "v": {x: vq[x] for x in ("codes", "lo", "scale")}}
        tok = {"k": kq["xhat"], "v": vq["xhat"], "b": torch.zeros((B, Hkv, n_q), device=k.device),
               "tau": torch.full((n_q,), H, dtype=torch.long)}
        return k, v, hide, tok


class StreamingLLMMethod(LayerMethod):
    """
    StreamingLLM (Xiao et al., ICLR 2024): sinks + the most recent tokens.  With a horizon it
    is one-shot as kvpress' StreamingLLMPress (n_sink = N_SINK = 4, n_kept = int(P * keep),
    no key re-rotation); without one it is the streaming policy (every query reads the sinks
    + its last keep - N_SINK tokens), used for long-document perplexity.
    """
    family = "streamingllm"

    def __init__(self, cfg, keep: int, name: str):
        super().__init__(cfg)
        self.keep, self.name = keep, name

    def entries(self, li, q, k, v, scaling, pos):
        c = self.cfg
        T = k.shape[2]
        pc = pos.cpu()
        recent = max(1, self.keep - c.N_SINK)
        if self.freeze_at is None:
            hide = torch.where(pc < c.N_SINK, torch.full_like(pc, BIG), pc + recent)
        else:
            H = self.horizon(T)
            drop = (pc >= c.N_SINK) & (pc < H - recent)
            hide = torch.where(drop, torch.full_like(pc, H), torch.full_like(pc, BIG))
        return k, v, hide, None


class Sim:
    """Routes every attention call of a forward to a LayerMethod (fp32 attention for all)."""

    def __init__(self, method: LayerMethod):
        self.method = method

    def begin(self, positions):
        self.pos = positions
        self.method.begin()

    def attend(self, module, q, k, v, mask, kwargs):
        scaling = kwargs.get("scaling") or module.scaling
        return self.method.layer(module.layer_idx, q, k, v, scaling, self.pos), None


# ════════════════════════════════════════════════════════════════════════════
# QUASAR 1/5: CODEBOOKS (Lloyd-Max scalar levels; embedded stages; byte-aligned packing)
# ════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=None)
def lloyd_max_gaussian(bits: int) -> torch.Tensor:
    """
    Lloyd-Max reconstruction levels of N(0, 1) with 2^bits cells (Max, 1960), computed by
    Lloyd iterations on a fine grid.  MSE 0.3634 / 0.1175 / 0.03454 / 0.009497 for 1-4 bits
    (quasar_tests checks these against Max's table).
    """
    n = 2 ** bits
    x = np.linspace(-9.0, 9.0, 360001)
    w = np.exp(-0.5 * x * x)
    w /= w.sum()
    c = stats.norm.ppf((np.arange(n) + 0.5) / n)
    for _ in range(500):
        cell = np.searchsorted((c[1:] + c[:-1]) / 2, x)
        mass = np.bincount(cell, w, n)
        c = np.bincount(cell, w * x, n) / np.maximum(mass, 1e-300)
    return torch.tensor(c, dtype=torch.float32)


def lloyd_max_mse(bits: int) -> float:
    c = lloyd_max_gaussian(bits).double().numpy()
    x = np.linspace(-9.0, 9.0, 360001)
    w = np.exp(-0.5 * x * x)
    w /= w.sum()
    cell = np.searchsorted((c[1:] + c[:-1]) / 2, x)
    return float((w * (x - c[cell]) ** 2).sum())


@lru_cache(maxsize=None)
def rotation_matrix(d: int, key: str) -> torch.Tensor:
    """
    Seeded random orthogonal matrix R (d x d): a randomised Hadamard transform H diag(s) / sqrt(d)
    when d is a power of two (QuaRot / TurboQuant style), otherwise the Q factor of a seeded
    Gaussian matrix.  Used only by the quasar_rot ablation (a basis without energy compaction).
    """
    g = _rng("rotation", key)
    if d & (d - 1) == 0:
        Hm = np.array([[1.0]])
        while Hm.shape[0] < d:
            Hm = np.block([[Hm, Hm], [Hm, -Hm]])
        R = Hm / math.sqrt(d) * g.choice([-1.0, 1.0], d)[None, :]
    else:
        Q, Rr = np.linalg.qr(g.standard_normal((d, d)))
        R = Q * np.sign(np.diag(Rr))[None, :]
    return torch.tensor(R, dtype=torch.float32)


@lru_cache(maxsize=None)
def stage_codebooks(stage_bits: Tuple[int, ...]) -> List[torch.Tensor]:
    """
    Embedded (residual) scalar code of a normalised coefficient u ~ N(0, 1): stage 0 is the
    Lloyd-Max code at stage_bits[0] bits; stage t > 0 is the Lloyd-Max code OF THE RESIDUAL
    u - u_hat_{t-1} (its distribution is not Gaussian: the levels are trained by Lloyd iterations
    on a deterministic quantile sample of N(0,1) passed through the previous stages).
    u_hat_t = sum_{t' <= t} level_t'[code_t'] is linear in the codes, so a reader of the first t
    stages adds t table look-ups; every prefix of the stages is a valid (coarser) code.  MSE at
    (2, 2, 4) bits: 0.1175 / 0.0196 / 0.00056 after 2 / 4 / 8 bits (direct Lloyd-Max: 0.1175 /
    0.0095 / 0.00013): the price of embedding is ~0.5 bit at 4 bits and ~1 bit at 8 bits.
    """
    x = stats.norm.ppf((np.arange(200000) + 0.5) / 200000)
    cur, out = np.zeros_like(x), []
    for t, b in enumerate(stage_bits):
        r = x - cur
        if t == 0:
            c = lloyd_max_gaussian(b).double().numpy()
        else:
            n = 2 ** b
            c = np.quantile(r, (np.arange(n) + 0.5) / n)
            for _ in range(150):
                cell = np.searchsorted((c[1:] + c[:-1]) / 2, r)
                c = np.bincount(cell, r, n) / np.maximum(np.bincount(cell, None, n), 1)
        cur = cur + c[np.searchsorted((c[1:] + c[:-1]) / 2, r)]
        out.append(torch.tensor(c, dtype=torch.float32))
    return out


def embedded_quantize(u: torch.Tensor, stage_bits: Tuple[int, ...]):
    """Codes (uint8) of every stage and the cumulative reconstructions u_hat_t (fp32)."""
    codes, cum = [], []
    cur = torch.zeros_like(u)
    for cb in stage_codebooks(tuple(stage_bits)):
        cb = cb.to(u.device)
        c = torch.bucketize(u - cur, (cb[1:] + cb[:-1]) / 2)
        cur = cur + cb[c]
        codes.append(c.to(torch.uint8))
        cum.append(cur)
    return codes, cum


def pack_rows(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """(..., w) codes < 2^bits -> (..., w * bits / 8) uint8, first code in the most significant bits."""
    per = 8 // bits
    c = codes.to(torch.int32).reshape(*codes.shape[:-1], codes.shape[-1] // per, per)
    shifts = torch.arange(per - 1, -1, -1, device=codes.device, dtype=torch.int32) * bits
    return (c << shifts).sum(-1).to(torch.uint8)


@lru_cache(maxsize=None)
def byte_lut(stage_bits: Tuple[int, ...], t: int) -> torch.Tensor:
    """(256, 8 / b) fp32 table: one stored byte -> the stage-t levels of its 8 / b codes (decode
    of a packed row is ONE gather, the dequantiser is never materialised element by element)."""
    b = stage_bits[t]
    per = 8 // b
    levels = stage_codebooks(tuple(stage_bits))[t]
    byte = torch.arange(256)[:, None]
    shifts = torch.arange(per - 1, -1, -1)[None, :] * b
    return levels[(byte >> shifts) & (2 ** b - 1)]


def band_edges(d: int, fractions: Tuple[float, ...]) -> List[Tuple[int, int]]:
    """Dimension bands of the spectral basis, leading (largest-eigenvalue) directions first."""
    widths = [max(4, int(round(d * f / 4)) * 4) for f in fractions[:-1]]
    widths.append(d - sum(widths))
    if widths[-1] < 4 or any(w % 4 for w in widths):
        raise ValueError(f"bands {fractions} do not split head_dim {d} into multiples of 4")
    out, a = [], 0
    for w in widths:
        out.append((a, a + w))
        a += w
    return out


# ════════════════════════════════════════════════════════════════════════════
# QUASAR 2/5: COVERAGE STATISTICS (the context's own queries over its whole length)
# ════════════════════════════════════════════════════════════════════════════

def single_entry_gain(p: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """
    Prop. 1: shifting ONLY logit j of a softmax by delta moves the output by
    G_j(delta) (v_j - o) with G_j = p_j (e^delta - 1) / (1 + p_j (e^delta - 1)).  delta = -inf
    (eviction) gives -p_j / (1 - p_j).  Exact for one entry (quasar_tests checks it).
    """
    t = p * torch.expm1(delta.clamp(max=30.0))
    return t / (1.0 + t).clamp_min(1e-12)


def coverage_groups(H: int, cfg: "Config") -> List[np.ndarray]:
    """
    Observation positions (all < H: no question, answer or future token is ever used).  One
    group per context segment of QUASAR_SEGMENT positions (QUASAR_QUERIES positions sampled
    without replacement, seeded by (H, segment)), plus the last W_OBS context positions, the
    'window' group (always last), which is the only group SnapKV-style scoring uses.
    Positions start at N_SINK + QUASAR_LOCAL, the first that can see a compressible key at
    long range.
    """
    lo, win = cfg.N_SINK + cfg.QUASAR_LOCAL, max(cfg.N_SINK + cfg.QUASAR_LOCAL, H - cfg.W_OBS)
    groups = []
    for s in range(lo, win, cfg.QUASAR_SEGMENT):
        seg = np.arange(s, min(s + cfg.QUASAR_SEGMENT, win))
        n = min(len(seg), cfg.QUASAR_QUERIES)
        groups.append(np.sort(_rng("cover_group", H, s).choice(seg, n, replace=False)))
    groups.append(np.arange(win, H))
    return [g_ for g_ in groups if len(g_)]


@torch.no_grad()
def coverage_stats(q, k, v, scaling, H: int, rs: int, re: int, M: torch.Tensor, cfg: "Config",
                   window_only: bool = False) -> Dict:
    """
    First-order sensitivities of one layer's attention output (in the residual stream, metric
    M_a = W_O,a^T W_O,a per query head) to the compression of every coded token j in [rs, re),
    under the context's own queries.  For a query (head a, position m) with exact probabilities
    p and output o, and u_j = ||v_j - o||^2_{M_a}:
        drop            D = (p_j / (1 - p_j))^2 u_j                      (Prop. 1, exact)
        key error e     D ~ p_j^2 u_j (q . e)^2                          (Prop. 1, first order)
        value error e   D = p_j^2 e^T M_a e                              (exact for one entry)
    Each group's terms are averaged over its queries and normalised by the group's
    drop-everything distortion N_g (every context segment gets the same say: coverage), then
    averaged over groups.  Decoupling the token-dependent and direction-dependent parts gives
        D_key(j, e) ~ A_j e^T S e,     D_value(j, e) ~ W_j e^T Mbar e,
    with S the sensitivity-weighted second moment of the queries (sum_j A_j-normalised) and Mbar
    the read-weighted mean of the M_a.  Reads at distance < QUASAR_LOCAL count only for the
    window group (it stands for the question, which reads everything).
    Returns A, W, DD (Hkv, n), S (Hkv, d, d) and Mbar (Hkv, d, d); no host synchronisation.
    """
    Hkv, d, n = k.shape[1], k.shape[-1], re - rs
    rep = q.shape[1] // Hkv
    groups = coverage_groups(H, cfg)
    if window_only:
        groups = groups[-1:]
    dev = k.device
    A = torch.zeros((Hkv, n), device=dev)
    DD = torch.zeros((Hkv, n), device=dev)
    WVa = torch.zeros((Hkv, rep, n), device=dev)
    S = torch.zeros((Hkv, d, d), device=dev)
    live = torch.zeros((), device=dev)
    VR = v[0, :, rs:re].float()
    Mg = M.view(Hkv, rep, d, d)                      # query heads grouped by KV head (HF repeat_kv order)
    vMv = torch.stack([((VR @ Mg[:, r]) * VR).sum(-1) for r in range(rep)], 1)      # (Hkv, rep, n)
    for gi, P in enumerate(groups):
        pos = torch.as_tensor(P, device=dev)
        nq, mmax = len(P), int(P.max()) + 1
        te = min(re, mmax)
        if te <= rs:
            continue
        nt = te - rs
        qg = (q_at(q, pos)[0].float() * scaling).view(Hkv, rep * nq, d)
        Sc = (qg @ k[0, :, :mmax].float().transpose(-1, -2)).view(Hkv, rep, nq, mmax)
        Sc = Sc.masked_fill(torch.arange(mmax, device=dev) > pos[:, None], float("-inf"))
        p = (Sc - Sc.logsumexp(-1, keepdim=True)).exp()
        o = (p.view(Hkv, rep * nq, mmax) @ v[0, :, :mmax].float()).view(Hkv, rep, nq, d)
        local = 1 if gi == len(groups) - 1 else cfg.QUASAR_LOCAL
        far = torch.arange(rs, te, device=dev) <= (pos[:, None] - local)
        pR = p[..., rs:te] * far
        oM = torch.einsum("hrqd,hrde->hrqe", o, Mg)
        u2 = (vMv[:, :, None, :nt] - 2 * (oM.reshape(Hkv, rep * nq, d) @ VR[:, :nt].transpose(-1, -2))
              .view(Hkv, rep, nq, nt) + (oM * o).sum(-1)[..., None]).clamp_min(0.0)
        dd = ((pR / (1 - pR).clamp_min(1e-6)).pow(2) * u2).sum((1, 2)) / nq           # (Hkv, nt)
        a = pR.pow(2) * u2                                                               # (Hkv, rep, nq, nt)
        N = dd.sum()
        wgt = torch.where(N > 0, 1.0 / N.clamp_min(1e-30), torch.zeros_like(N))
        A[:, :nt] += a.sum((1, 2)) / nq * wgt
        DD[:, :nt] += dd * wgt
        WVa[:, :, :nt] += pR.pow(2).sum(2) / nq * wgt
        alpha = a.sum(-1) / nq                                                           # (Hkv, rep, nq)
        qa = qg.view(Hkv, rep, nq, d)
        S += torch.einsum("hrq,hrqd,hrqe->hde", alpha, qa, qa) * wgt
        live = live + (N > 0).float()
    live = live.clamp_min(1.0)
    A, DD, WVa, S = A / live, DD / live, WVa / live, S / live
    S = S / A.sum(-1).clamp_min(1e-30)[:, None, None]             # normalised: D_key = A_j e^T S e
    omega = WVa.sum(-1)                                            # (Hkv, rep) read mass per query head
    Mbar = torch.einsum("hr,hrde->hde", omega, Mg) / omega.sum(-1).clamp_min(1e-30)[:, None, None]
    # heads nothing reads at long range: fall back to the plain mean metric
    Mbar = torch.where(omega.sum(-1)[:, None, None] > 0, Mbar, Mg.mean(1))
    eta = cfg.QUASAR_FLOOR                    # tokens no query reads still prefer fidelity when bits are spare
    W = WVa.sum(1)
    return {"A": A + eta * A.mean(), "W": W + eta * W.mean(), "DD": DD + eta * DD.mean(), "S": S,
            "Mbar": Mbar, "groups": groups}


# ════════════════════════════════════════════════════════════════════════════
# QUASAR 3/5: SPECTRAL BASES (metric whitening + importance-weighted KLT, per context and head)
# ════════════════════════════════════════════════════════════════════════════

def psd_sqrt(S: torch.Tensor, eps: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """L = (S + eps tr(S)/d I)^{1/2} and its inverse, batched over heads (symmetric eigh).
    The ridge keeps directions the context's queries never used readable by a question that
    does, and bounds the condition number of L by sqrt(1 + 1 / eps) for a rank-one S."""
    S = 0.5 * (S + S.transpose(-1, -2)).float()
    d = S.shape[-1]
    tr = S.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-30)
    S = S + (eps * tr)[..., None, None] * torch.eye(d, dtype=S.dtype, device=S.device)
    w, U = torch.linalg.eigh(S)
    w = w.clamp_min(w.amax(-1, keepdim=True) * 1e-6)
    L = (U * w.sqrt()[..., None, :]) @ U.transpose(-1, -2)
    Li = (U * w.rsqrt()[..., None, :]) @ U.transpose(-1, -2)
    return L, Li


def klt_basis(X: torch.Tensor, wts: torch.Tensor) -> torch.Tensor:
    """Eigenvectors (columns, decreasing eigenvalue) of the weighted second moment of X (Hkv, n, d)."""
    C = (X * wts[..., None]).transpose(-1, -2) @ X / wts.sum(-1).clamp_min(1e-30)[:, None, None]
    _, U = torch.linalg.eigh(0.5 * (C + C.transpose(-1, -2)))
    return U.flip(-1)


@torch.no_grad()
def spectral_side(X: torch.Tensor, sens: torch.Tensor, metric: Optional[torch.Tensor], basis: str,
                  eps: float, rot_key: str) -> Dict:
    """
    One side (keys or values) of one layer.  X (Hkv, n, d) are the centred vectors; metric the
    d x d matrix the distortion is measured with (S for keys, Mbar for values; None = Euclidean,
    the scale-free metric of the prior-art ablations).  The coefficient map is
        c = (x - mu) L U,   x - mu = c U^T L^{-1}
    with L = metric^{1/2} and U the KLT of the metric-whitened vectors (weighted by the tokens'
    sensitivities): an ORTHONORMAL basis of the metric space, so the metric error of a code is the
    sum of its coefficient errors (Prop. 3) and truncating to the leading r coefficients is the
    best rank-r linear code of the weighted vectors in that metric (Prop. 2, Eckart-Young).
    basis 'rot' replaces U by a seeded random rotation (no energy compaction).
    Returns coefficients C (Hkv, n, d), the per-coefficient scale sigma (Hkv, d) and the read-out
    matrix T = L^{-1} U diag(sigma) (Hkv, d, d): a stored coefficient row z reads back as z T^T.
    """
    Hkv, n, d = X.shape
    if metric is None:
        L = Li = torch.eye(d, device=X.device).expand(Hkv, d, d)
    else:
        L, Li = psd_sqrt(metric, eps)
    Xm = X @ L
    if basis == "rot":
        U = rotation_matrix(d, rot_key).to(X.device).T.expand(Hkv, d, d)
    else:
        w = sens / sens.mean(-1, keepdim=True).clamp_min(1e-30)
        U = klt_basis(Xm, 0.5 * w + 0.5)                       # half importance, half uniform
    C = Xm @ U
    sigma = C.pow(2).mean(1).sqrt().clamp_min(1e-8)
    return {"C": C, "sigma": sigma, "T": (Li @ U) * sigma[:, None, :]}


# ════════════════════════════════════════════════════════════════════════════
# QUASAR 4/5: EMBEDDED LEVELS AND THEIR EXACT DISTORTIONS; HEAD-SYNCHRONOUS ALLOCATION
# ════════════════════════════════════════════════════════════════════════════

def band_errors(C: torch.Tensor, sigma: torch.Tensor, rho: torch.Tensor, cum: List[torch.Tensor],
                bands: List[Tuple[int, int]]) -> torch.Tensor:
    """
    E (Hkv, n, nb, T+1): squared coefficient error of every band after 0 (nothing stored: the
    band's energy) .. T embedded stages.  The coefficient read back is rho sigma u_hat.
    """
    errs = [C.pow(2)] + [(C - rho[..., None] * sigma[:, None, :] * u_hat).pow(2) for u_hat in cum]
    cs = torch.stack(errs, -1).cumsum(2)                                     # (Hkv, n, d, T+1)
    cs = torch.cat([torch.zeros_like(cs[:, :, :1]), cs], 2)
    return torch.stack([cs[:, :, b] - cs[:, :, a] for a, b in bands], 2)


def build_path(gainK: torch.Tensor, gainV: torch.Tensor, bits_kt: List[List[int]]) -> List[List[Tuple[int, int, int]]]:
    """
    The layer's embedded order.  gainK / gainV (nb, T): sensitivity-weighted metric error removed
    by stage t of band k, summed over heads and tokens; bits_kt[k][t]: its bits per token.
    Level 1 stores the first stage of the leading band of keys AND values (a token needs both);
    every later level adds the (side, band, stage) with the largest error removed per bit whose
    previous stage is stored (greedy merge of the per-band chains: optimal for the aggregate when
    every chain is convex, as in post-compression rate-distortion optimisation, EBCOT).
    Returns levels as lists of (side, band, stage), side 0 = keys, 1 = values.
    """
    nb, T = gainK.shape
    gain = [gainK.tolist(), gainV.tolist()]
    nxt = [[0] * nb, [0] * nb]
    levels = [[(0, 0, 0), (1, 0, 0)]]
    nxt[0][0] = nxt[1][0] = 1
    while True:
        best, arg = -1.0, None
        for s in (0, 1):
            for k in range(nb):
                t = nxt[s][k]
                if t < T:
                    r = gain[s][k][t] / bits_kt[k][t]
                    if r > best:
                        best, arg = r, (s, k, t)
        if arg is None:
            return levels
        levels.append([arg])
        nxt[arg[0]][arg[1]] += 1


@torch.no_grad()
def head_sync_allocate(Dsort: torch.Tensor, cost: torch.Tensor, fixed: torch.Tensor, budget: float,
                       rounds: int = 4, grid: int = 32) -> Dict:
    """
    Dsort (n, nlev): distortion, summed over the KV heads, of giving rank position j (the j-th
    most important token OF EACH HEAD) option l, columns [drop, level 1 .. level L, exact];
    cost (nlev,) bits of option l summed over heads; fixed (L + 1,) the layer's fixed bits when the
    deepest level used is c (0 = nothing coded: no basis).  Minimises sum_j Dsort[j, l_j] subject
    to sum_j cost[l_j] + fixed[c] <= budget by Lagrangian relaxation (Everett, 1963): for a
    multiplier mu every position takes argmin_l D + mu cost among the options allowed by the
    cap c, and c minimises the total.  The bits are non-increasing in mu, so the smallest
    feasible mu is bracketed by `rounds` rounds of `grid` multipliers evaluated in ONE batched
    pass each (log scale, [e^-80, e^25] -> resolution 105 / grid^rounds), all on the device.
    The returned allocation is feasible (bits <= budget) whenever dropping everything is.
    """
    n, nlev = Dsort.shape
    L = nlev - 2
    dev = Dsort.device

    def evaluate(log_mu):                                       # (M,) -> bits (M,), choice (M, n), cap (M,)
        mu = log_mu.exp()
        Q = Dsort[None] + mu[:, None, None] * cost
        base, bidx = torch.stack([Q[..., 0], Q[..., -1]], -1).min(-1)
        bidx = torch.where(bidx == 0, torch.zeros_like(bidx), torch.full_like(bidx, nlev - 1))
        if L > 0:
            cm, cidx = Q[..., 1:-1].cummin(-1)                   # best level <= c, per position
            tot = torch.cat([base.sum(1)[:, None], torch.minimum(base[..., None], cm).sum(1)], 1) + mu[:, None] * fixed
            c = tot.argmin(1)
            cc = (c - 1).clamp_min(0)[:, None, None].expand(-1, n, 1)
            use = (c > 0)[:, None] & (cm.gather(2, cc)[..., 0] < base)
            choice = torch.where(use, cidx.gather(2, cc)[..., 0] + 1, bidx)
        else:
            c = torch.zeros(len(log_mu), dtype=torch.long, device=dev)
            choice = bidx
        return cost[choice].sum(1) + fixed[c], choice, c

    lo, hi = torch.tensor(-80.0, device=dev), torch.tensor(25.0, device=dev)
    steps = torch.linspace(0.0, 1.0, grid, device=dev)
    for _ in range(rounds):
        g_ = lo + (hi - lo) * steps
        ok = evaluate(g_)[0] <= budget
        i = torch.where(ok.any(), ok.float().argmax(), torch.tensor(grid - 1, device=dev))
        lo, hi = g_[(i - 1).clamp_min(0)], g_[i]
    bits, choice, c = evaluate(hi[None])
    return {"choice": choice[0], "cap": c[0], "bits": bits[0]}


# ════════════════════════════════════════════════════════════════════════════
# QUASAR DECODE KERNELS (Triton): read only the packed bytes of every embedded step
# ════════════════════════════════════════════════════════════════════════════
# Step table (int32, one row per step of one side, in level order):
#   [byte offset of the step in the side's flat code buffer, n_s (tokens owning it), bytes per
#    row, first basis column a, width w, bits b, offset of its levels in the level table]
# The flat buffer holds, per step, a (Hkv, n_s, bytes) block.  Codes are unpacked in registers
# (shift / mask), mapped through the stage's Lloyd-Max levels and reduced against the query
# coefficients (scores) or the attention weights (values).  The PyTorch path (QuasarState.
# attend with use_kernel False) is the reference; the kernel path is enabled only after an
# equivalence check against it (QuasarState.check_kernel).

try:
    import triton
    import triton.language as tl
    TRITON_OK = True
except Exception:                                                  # noqa: BLE001 (no Triton: reference path only)
    TRITON_OK = False

if TRITON_OK:
    @triton.jit
    def _quasar_scores_kernel(qt_ptr, codes_ptr, steps_ptr, levels_ptr, out_ptr, n_rows, n_c, n_steps,
                              R: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, W: tl.constexpr):
        """out[h, r, j] = sum over the steps owned by token j of q_tilde[h, r, band] . level[code[h, j, band]]."""
        h = tl.program_id(0)
        j = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        r = tl.program_id(2) * BLOCK_R + tl.arange(0, BLOCK_R)
        i = tl.arange(0, W)
        acc = tl.zeros((BLOCK_R, BLOCK_N), dtype=tl.float32)
        for s in range(n_steps):
            off = tl.load(steps_ptr + s * 7 + 0)
            ns = tl.load(steps_ptr + s * 7 + 1)
            nb = tl.load(steps_ptr + s * 7 + 2)
            a = tl.load(steps_ptr + s * 7 + 3)
            w = tl.load(steps_ptr + s * 7 + 4)
            b = tl.load(steps_ptr + s * 7 + 5)
            lo = tl.load(steps_ptr + s * 7 + 6)
            per = 8 // b
            m2 = (j < ns)[:, None] & (i < w)[None, :]
            byt = tl.load(codes_ptr + off + (h * ns + j)[:, None] * nb + (i // per)[None, :], mask=m2, other=0)
            code = (byt.to(tl.int32) >> ((per - 1 - i % per) * b)[None, :]) & ((1 << b) - 1)
            val = tl.where(m2, tl.load(levels_ptr + lo + code, mask=m2, other=0.0), 0.0)
            q = tl.load(qt_ptr + (h * n_rows + r)[:, None] * R + (a + i)[None, :],
                        mask=(r < n_rows)[:, None] & (i < w)[None, :], other=0.0)
            acc += tl.dot(q, tl.trans(val), input_precision="ieee")
        tl.store(out_ptr + (h * n_rows + r)[:, None] * n_c + j[None, :], acc,
                 mask=(r < n_rows)[:, None] & (j < n_c)[None, :])

    @triton.jit
    def _quasar_values_kernel(p_ptr, codes_ptr, steps_ptr, levels_ptr, out_ptr, n_rows, n_c, n_steps,
                              R: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr, W: tl.constexpr):
        """out[h, r, band] += sum_j p[h, r, j] level[code[h, j, band]] (split over token blocks, atomics)."""
        h = tl.program_id(0)
        j = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        r = tl.program_id(2) * BLOCK_R + tl.arange(0, BLOCK_R)
        i = tl.arange(0, W)
        p = tl.load(p_ptr + (h * n_rows + r)[:, None] * n_c + j[None, :],
                    mask=(r < n_rows)[:, None] & (j < n_c)[None, :], other=0.0)
        for s in range(n_steps):
            off = tl.load(steps_ptr + s * 7 + 0)
            ns = tl.load(steps_ptr + s * 7 + 1)
            nb = tl.load(steps_ptr + s * 7 + 2)
            a = tl.load(steps_ptr + s * 7 + 3)
            w = tl.load(steps_ptr + s * 7 + 4)
            b = tl.load(steps_ptr + s * 7 + 5)
            lo = tl.load(steps_ptr + s * 7 + 6)
            per = 8 // b
            m2 = (j < ns)[:, None] & (i < w)[None, :]
            byt = tl.load(codes_ptr + off + (h * ns + j)[:, None] * nb + (i // per)[None, :], mask=m2, other=0)
            code = (byt.to(tl.int32) >> ((per - 1 - i % per) * b)[None, :]) & ((1 << b) - 1)
            val = tl.where(m2, tl.load(levels_ptr + lo + code, mask=m2, other=0.0), 0.0)
            part = tl.dot(p, val, input_precision="ieee")
            tl.atomic_add(out_ptr + (h * n_rows + r)[:, None] * R + (a + i)[None, :], part,
                          mask=(r < n_rows)[:, None] & (i < w)[None, :])


def kernel_block(n: int) -> int:
    return 64 if n >= 64 else max(16, 1 << max(0, (n - 1).bit_length()))


def quasar_scores(qt: torch.Tensor, side: Dict, n_c: int) -> torch.Tensor:
    """qt (Hkv, rows, R) fp32 -> scores (Hkv, rows, n_c) fp32 with the fused kernel."""
    Hkv, rows, R = qt.shape
    out = torch.empty((Hkv, rows, n_c), dtype=torch.float32, device=qt.device)
    BR, BN = 16, kernel_block(n_c)
    grid = (Hkv, triton.cdiv(n_c, BN), triton.cdiv(rows, BR))
    _quasar_scores_kernel[grid](qt.contiguous(), side["flat"], side["table"], side["levels"], out, rows, n_c,
                                side["table"].shape[0], R=R, BLOCK_R=BR, BLOCK_N=BN, W=side["W"])
    return out


def quasar_values(pv: torch.Tensor, side: Dict, R: int) -> torch.Tensor:
    """pv (Hkv, rows, n_c) fp32 -> coefficient sums (Hkv, rows, R) fp32 with the fused kernel."""
    Hkv, rows, n_c = pv.shape
    out = torch.zeros((Hkv, rows, R), dtype=torch.float32, device=pv.device)
    BR, BN = 16, kernel_block(n_c)
    grid = (Hkv, triton.cdiv(n_c, BN), triton.cdiv(rows, BR))
    _quasar_values_kernel[grid](pv.contiguous(), side["flat"], side["table"], side["levels"], out, rows, n_c,
                                side["table"].shape[0], R=R, BLOCK_R=BR, BLOCK_N=BN, W=side["W"])
    return out




# ════════════════════════════════════════════════════════════════════════════
# QUASAR 5/5: THE METHOD (harness entries, packed storage, compressed-domain decoding)
# ════════════════════════════════════════════════════════════════════════════

class QuasarMethod(LayerMethod):
    """
    QUASAR, applied once when the context ends (horizon H): sinks and the last WINDOW context
    tokens stay exact; every other token of every KV head is dropped, kept exact, or stored at
    one of the layer's embedded levels (a prefix of the layer's (side, band, stage) order) in
    the head's spectral basis.  Variants: metric 'query' (keys: the context's sensitivity-
    weighted query metric; values: the output-projection metric) or 'euclid' (plain PCA of the
    vectors, the prior-art basis); basis 'klt' or 'rot' (seeded random rotation: no spectral
    compaction); window_only (sensitivities from the end window alone, the SnapKV-type prior).
    """
    family = "quasar"
    one_shot = True

    def __init__(self, cfg, keep: float, name: str, runner: "Runner", metric: str = "query",
                 basis: str = "klt", window_only: bool = False):
        super().__init__(cfg)
        self.keep, self.name, self.runner = keep, name, runner
        self.metric_kind, self.basis, self.window_only = metric, basis, window_only
        self.stages = tuple(cfg.QUASAR_STAGES)
        self.bands = band_edges(runner.d, tuple(cfg.QUASAR_BANDS))
        self.metrics: Dict[int, torch.Tensor] = {}
        self.cbits: Dict[int, float] = {}
        self.alloc: Dict[int, Dict] = {}
        self.diag: List[Dict] = []

    def begin(self):
        super().begin()
        self.cbits, self.alloc, self.diag = {}, {}, []

    def entry_bits(self, kind, d):
        return 2 * d * 16.0 if kind == "exact" else 0.0           # coded tokens: self.cbits

    def out_metric(self, li: int, dev) -> torch.Tensor:
        """M_a = W_O,a^T W_O,a (d x d) for every query head a of layer li (from the weights)."""
        if li not in self.metrics:
            W = self.runner.layers[li].self_attn.o_proj.weight.float()            # (hidden, Hq d)
            d, Hq = self.runner.d, self.runner.Hq
            Wa = W.view(W.shape[0], Hq, d).permute(1, 0, 2)                       # (Hq, hidden, d)
            self.metrics[li] = (Wa.transpose(-1, -2) @ Wa).to(dev)
        return self.metrics[li]

    @torch.no_grad()
    def compress_layer(self, li: int, q, k, v, scaling, H: int) -> Optional[Dict]:
        """The allocation of one layer (everything a reader of the stored code needs), or None
        when nothing is compressible (the layer stays exact)."""
        c = self.cfg
        rs, re = c.N_SINK, H - c.WINDOW
        if re - rs <= 0 or k.shape[0] != 1:
            return None
        Hkv, d, n = k.shape[1], k.shape[-1], re - rs
        dev = k.device
        st = coverage_stats(q, k, v, scaling, H, rs, re, self.out_metric(li, dev), c, self.window_only)
        KR, VR = k[0, :, rs:re].float(), v[0, :, rs:re].float()
        mu = [as_stored(KR.mean(1, keepdim=True)), as_stored(VR.mean(1, keepdim=True))]
        sens = [st["A"], st["W"]]
        if self.metric_kind == "query":
            metrics, kappa = [st["S"], st["Mbar"]], [torch.ones(Hkv, device=dev)] * 2
        else:   # Euclidean basis and error; isotropic surrogate of the metric for the allocation
            metrics = [None, None]
            kappa = [st["S"].diagonal(dim1=-2, dim2=-1).mean(-1), st["Mbar"].diagonal(dim1=-2, dim2=-1).mean(-1)]
        sides = []
        for s, X in enumerate((KR, VR)):
            sp = spectral_side(X - mu[s], sens[s], metrics[s], self.basis, c.QUASAR_EPS, f"quasar{li}{'kv'[s]}")
            z = sp["C"] / sp["sigma"][:, None, :]
            rho = as_stored(z.pow(2).mean(-1).sqrt()).clamp_min(2.0 ** -24)
            codes, cum = embedded_quantize(z / rho[..., None], self.stages)
            E = band_errors(sp["C"], sp["sigma"], rho, cum, self.bands) * kappa[s][:, None, None, None]
            sides.append({**sp, "rho": rho, "codes": codes, "cum": cum, "E": E})
        nb, T = len(self.bands), len(self.stages)
        widths = [b - a for a, b in self.bands]
        bits_kt = [[widths[kk] * self.stages[t] for t in range(T)] for kk in range(nb)]
        gains = [(sens[s][..., None, None] * (sides[s]["E"][..., :-1] - sides[s]["E"][..., 1:])).sum((0, 1))
                 for s in (0, 1)]
        levels = build_path(gains[0].cpu(), gains[1].cpu(), bits_kt)
        Lv = len(levels)
        tab = np.full((Lv + 1, 2, nb), -1, dtype=np.int64)          # stage stored per (level, side, band)
        cost_lv = np.zeros(Lv + 1)
        for lv, steps in enumerate(levels, start=1):
            tab[lv] = tab[lv - 1]
            cost_lv[lv] = cost_lv[lv - 1]
            for s, kk, t in steps:
                tab[lv, s, kk] = t
                cost_lv[lv] += bits_kt[kk][t]
        cost_lv[1:] += 2 * 16                                       # fp16 scale rho of key and value
        idx = torch.as_tensor(tab + 1, device=dev)                  # +1: column 0 of E is "nothing stored"
        err = [sides[s]["E"][:, :, torch.arange(nb, device=dev), idx[:, s]].sum(-1) for s in (0, 1)]  # (Hkv, n, Lv+1)
        for sd in sides:                                            # analysis-only tensors
            del sd["E"], sd["C"]
        D = sens[0][..., None] * err[0] + sens[1][..., None] * err[1]
        Dopt = torch.cat([st["DD"][..., None], D[..., 1:], torch.zeros_like(D[..., :1])], -1)   # drop | 1..Lv | exact
        order = (st["DD"] + D[..., 1]).argsort(-1, descending=True)                              # (Hkv, n)
        Dsort = Dopt.gather(1, order[..., None].expand(Hkv, n, Lv + 2)).sum(0)                  # (n, Lv+2)
        cost_tok = np.concatenate([[0.0], cost_lv[1:], [2 * d * 16.0]])
        cost = torch.tensor(cost_tok * Hkv, device=dev, dtype=torch.float32)
        rk = np.zeros(Lv + 1, dtype=np.int64)
        rv = np.zeros(Lv + 1, dtype=np.int64)
        for lv in range(1, Lv + 1):
            used = [[kk for kk in range(nb) if tab[lv, s, kk] >= 0] for s in (0, 1)]
            rk[lv], rv[lv] = self.bands[max(used[0])][1], self.bands[max(used[1])][1]
        hist_bits = (Lv + 2) * 32.0
        fixed_np = np.where(np.arange(Lv + 1) > 0, Hkv * ((rk + 1) * d + (rv + 1) * d) * 16.0, 0.0) + hist_bits
        fixed = torch.tensor(fixed_np, device=dev, dtype=torch.float32)
        budget = self.keep * H * Hkv * 2 * d * 16.0 - (rs + (H - re)) * Hkv * 2 * d * 16.0
        if budget < hist_bits:                                      # sinks + window already exceed the budget
            choice = torch.zeros(n, dtype=torch.long, device=dev)
            cap = torch.zeros((), dtype=torch.long, device=dev)
        else:
            al = head_sync_allocate(Dsort, cost, fixed, budget, c.QUASAR_ROUNDS)
            choice, cap = al["choice"], al["cap"]
        lev = choice.sort(descending=True).values                  # monotone: most important positions deepest
        counts = torch.bincount(lev, minlength=Lv + 2)
        cnt = counts.tolist()                                       # the one host synchronisation of the layer
        cap_i = int(cap)
        n_ex, n_c = cnt[Lv + 1], sum(cnt[1:Lv + 1])
        lev_cpu = lev.cpu().numpy()
        R = [int(rk[cap_i]), int(rv[cap_i])]
        # the stored code's bits (checked against payload()); exact-level tokens are exact rows,
        # charged with the sinks and the window by LayerMethod.state_bits
        coded_np = (lev_cpu >= 1) & (lev_cpu <= Lv)
        bits_f = float((cost_tok[lev_cpu[coded_np]] * Hkv).sum() + fixed_np[cap_i])
        distortion = float((Dsort.gather(1, lev[:, None])[:, 0]).sum() / Dsort[:, 0].sum().clamp_min(1e-30))
        return {"H": H, "rs": rs, "re": re, "order": order, "lev": lev, "lev_cpu": lev_cpu, "counts": cnt,
                "levels": levels, "tab": tab, "n_ex": n_ex, "n_c": n_c, "cap": cap_i, "R": R, "mu": mu,
                "sides": sides, "bits": bits_f, "fixed": float(fixed_np[cap_i]), "distortion": distortion,
                "cost_tok": cost_tok}

    def reconstruct(self, al: Dict, s: int) -> torch.Tensor:
        """What a reader of the code sees for the coded tokens of side s (Hkv, n_c, d), in the
        coded tokens' storage order (most important first): mu + (rho u_hat) T^T."""
        side, R, n_ex, n_c = al["sides"][s], al["R"][s], al["n_ex"], al["n_c"]
        idx = al["order"][:, n_ex:n_ex + n_c]                                    # (Hkv, n_c) token ids
        lev = al["lev"][n_ex:n_ex + n_c]                                         # (n_c,)
        Hkv = idx.shape[0]
        tab = torch.as_tensor(al["tab"][:, s], device=lev.device)               # (Lv+1, nb)
        u = torch.zeros((Hkv, n_c, R), device=lev.device)
        for kk, (a, b) in enumerate(self.bands):
            if a >= R:
                break
            st_k = tab[lev, kk]                                                  # stage per coded token
            for t, cum in enumerate(side["cum"]):
                sel = (st_k == t)[None, :, None]
                g = cum.gather(1, idx[..., None].expand(Hkv, n_c, cum.shape[-1]))[..., a:b]
                u[..., a:b] = torch.where(sel, g, u[..., a:b])
        rho = side["rho"].gather(1, idx)
        Ts = as_stored(side["T"][:, :, :R])                                      # (Hkv, d, R) as stored
        return al["mu"][s] + (rho[..., None] * u) @ Ts.transpose(-1, -2)

    def record(self, al: Dict):
        """Per-layer diagnostics of an allocation (means over layers are reported per sample)."""
        lv = al["lev_cpu"]
        Lv = len(al["levels"])
        coded = (lv >= 1) & (lv <= Lv)
        self.diag.append({"kept": float((lv > 0).mean()), "exact": float((lv == Lv + 1).mean()),
                          "level": float(lv[coded].mean()) if coded.any() else 0.0,
                          "coded_bits": float(al["cost_tok"][lv[coded]].mean()) if coded.any() else 0.0,
                          "rank_k": al["R"][0], "rank_v": al["R"][1], "levels": Lv,
                          "fixed_frac": al["fixed"] / max(al["bits"], 1.0), "distortion": al["distortion"]})

    def entries(self, li, q, k, v, scaling, pos):
        T = k.shape[2]
        if self.freeze_at is None:
            raise RuntimeError("QUASAR compresses once at a horizon (freeze_at)")
        H = self.horizon(T)
        full_hide = torch.full((T,), BIG, dtype=torch.long)
        al = self.compress_layer(li, q, k, v, scaling, H)
        if al is None:
            return k, v, full_hide, None
        Hkv, dev = k.shape[1], k.device
        rs, re, n_ex, n_c = al["rs"], al["re"], al["n_ex"], al["n_c"]
        hide = torch.full((1, Hkv, T), BIG, dtype=torch.long, device=dev)
        hide[:, :, rs:re] = H
        exact_ids = al["order"][:, :n_ex] + rs                                   # exact-level tokens per head
        hide[0].scatter_(1, exact_ids, BIG)
        tok = None
        if n_c > 0:
            tok = {"k": self.reconstruct(al, 0)[None], "v": self.reconstruct(al, 1)[None],
                   "b": torch.zeros((1, Hkv, n_c), device=dev), "tau": torch.full((n_c,), H, dtype=torch.long)}
        self.record(al)
        if self.capture_extra is not None:
            self.cbits[li] = al["bits"]
            al["packed"] = self.pack(al)
        al.pop("sides")                       # keep only the stored code (budget-sized), never the analysis tensors
        if self.capture_extra is not None:
            self.alloc[li] = al
        return k, v, hide, tok

    def state_bits(self, d: int, context_len: int) -> Dict[str, float]:
        out = super().state_bits(d, context_len)
        out["tok"] = float(sum(self.cbits.values()))
        out["total"] = out["exact"] + out["tok"]
        return out

    def pack(self, al: Dict) -> Dict:
        """The stored code of one layer (all tensors that a decoder keeps, nothing else): per step of
        the embedded order, the packed codes of the tokens deep enough to own it; fp16 scales,
        read-out matrices and means; the level histogram.  Exact rows are not included."""
        n_ex, n_c, cnt = al["n_ex"], al["n_c"], al["counts"]
        Lv = len(al["levels"])
        idx = al["order"][:, n_ex:n_ex + n_c]
        Hkv = idx.shape[0]
        steps = []
        for lv in range(1, al["cap"] + 1):
            n_s = sum(cnt[lv:Lv + 1])                       # coded tokens at level >= lv (a prefix)
            for s, kk, t in al["levels"][lv - 1]:
                a, b = self.bands[kk]
                codes = al["sides"][s]["codes"][t].gather(1, idx[:, :n_s, None].expand(Hkv, n_s, al["sides"][s]["codes"][t].shape[-1]))
                steps.append({"side": s, "a": a, "b": b, "stage": t, "n": n_s,
                              "codes": pack_rows(codes[..., a:b], self.stages[t])})
        out = {"steps": steps, "counts": torch.tensor(cnt, dtype=torch.int32), "n_c": n_c}
        if n_c:
            for s, nm in ((0, "k"), (1, "v")):
                out[f"rho_{nm}"] = al["sides"][s]["rho"].gather(1, idx).to(torch.float16)
                out[f"T_{nm}"] = as_stored(al["sides"][s]["T"][:, :, :al["R"][s]]).to(torch.float16)
                out[f"mu_{nm}"] = al["mu"][s][:, 0].to(torch.float16)
        return out

    def packed(self, li: int, dtype=torch.float16) -> Dict:
        """The stored code of layer li with its matrices in the decoder's dtype."""
        p = dict(self.alloc[li]["packed"])
        for x in ("T_k", "T_v", "mu_k", "mu_v"):
            if x in p:
                p[x] = p[x].to(dtype)
        return p

    def payload(self, li: int) -> int:
        """MEASURED bytes of the stored code of layer li (exact rows excluded)."""
        p = self.packed(li)
        return tensor_bytes([[st["codes"] for st in p["steps"]], p["counts"]] +
                            [p[x] for x in ("rho_k", "rho_v", "T_k", "T_v", "mu_k", "mu_v") if x in p])

    def native_state(self, li: int, K: torch.Tensor, V: torch.Tensor, extra: int, dtype) -> "QuasarState":
        """Decoder state of layer li: exact rows (sinks, exact-level tokens, window) in the model dtype +
        the packed code (read without reconstruction).  K, V: the layer's context cache (1, Hkv, H, d)."""
        H = K.shape[2]
        al = self.alloc.get(li)
        if al is None:
            return QuasarState(K.to(dtype), V.to(dtype), None, self.stages, extra, H)
        rs, re, n_ex = al["rs"], al["re"], al["n_ex"]
        Hkv, d = K.shape[1], K.shape[-1]
        ex_ids = torch.cat([torch.arange(rs, device=K.device).expand(Hkv, rs), al["order"][:, :n_ex] + rs,
                            torch.arange(re, H, device=K.device).expand(Hkv, H - re)], 1)
        Ke = K[0].gather(1, ex_ids[..., None].expand(Hkv, ex_ids.shape[1], d))[None].to(dtype)
        Ve = V[0].gather(1, ex_ids[..., None].expand(Hkv, ex_ids.shape[1], d))[None].to(dtype)
        return QuasarState(Ke, Ve, self.packed(li, dtype), self.stages, extra, H)

    def attn_flops_first_step(self, Hq: int, d: int) -> float:
        """Matmul-class FLOPs of the first decoded token in the compressed domain (all layers)."""
        total = 0.0
        for li, st in self.states.items():
            Hkv = st.K.shape[1]
            rep = Hq // Hkv
            rows = (st.n0 if isinstance(st, QuasarState) else
                    float((st.hide[:, :, :st.n] >= BIG).sum()) / Hkv) + 1           # exact rows per head (+ new token)
            f = 4 * d * rows
            p = st.p if isinstance(st, QuasarState) else (self.packed(li) if li in self.alloc else None)
            if p is not None and p["n_c"]:
                f += 2 * d * (p["T_k"].shape[-1] + p["T_v"].shape[-1] + 2)        # q -> coefficients, lift, means
                f += sum(2 * (x["b"] - x["a"]) * x["n"] for x in p["steps"])
            total += rep * Hkv * f
        return total

    def diagnostics(self) -> Dict[str, float]:
        if not self.diag:
            return {}
        df = pd.DataFrame(self.diag)
        return {f"quasar_{c_}": float(df[c_].mean()) for c_ in df.columns}


class QuasarState:
    """
    Compressed cache of one layer as a decoder holds it (model dtype): exact rows in a buffer
    with room for the question and the answer, and the embedded code (per side, ONE flat uint8
    buffer holding every step's (Hkv, n_s, bytes) block, plus a step table).  attend() never
    reconstructs a key or value: the query is mapped into each head's coefficient space once
    (q_tilde = q T_K), every step of the embedded order adds q_tilde[band] . level[codes] for the
    prefix of tokens that own it, the softmax runs over exact rows and coded tokens together, and
    the coded tokens' weighted coefficients are lifted back with one matmul
    (o += (p rho_V) C T_V^T + (sum p) mu_V).  Two implementations of the coded part: the
    reference (one byte-table gather + one batched matmul per step, PyTorch) and the fused
    Triton kernels (one launch per side, packed bytes only; used when use_kernel is set, which
    QuasarState.check_kernel does only after matching the reference).  Same interface as
    LayerState (append / attend / n / shared), so DecodeCtl drives it.
    """

    def __init__(self, Ke, Ve, packed: Optional[Dict], stages, extra: int, H: int):
        B, Hkv, n0, d = Ke.shape
        self.K = Ke.new_zeros((B, Hkv, n0 + extra, d))
        self.V = Ve.new_zeros((B, Hkv, n0 + extra, d))
        self.K[:, :, :n0], self.V[:, :, :n0] = Ke, Ve
        self.n0 = self.n = n0
        self.H, self.shared, self.p = H, None, packed
        self.dtype = Ke.dtype
        self.use_kernel = False
        dev = Ke.device
        self.luts = [byte_lut(tuple(stages), t).to(dev, self.dtype) for t in range(len(stages))]
        self.sides = [None, None]
        if packed is not None and packed["n_c"] > 0:
            books = stage_codebooks(tuple(stages))
            lvl_off = np.cumsum([0] + [len(b_) for b_ in books])[:-1]
            levels = torch.cat(books).to(dev)
            for s in (0, 1):
                steps = [st for st in packed["steps"] if st["side"] == s]
                flat = torch.cat([st["codes"].reshape(-1) for st in steps]).to(dev)
                rows, off = [], 0
                for st in steps:                        # the step's codes become a view of the flat buffer
                    numel = st["codes"].numel()
                    st["codes"] = flat[off:off + numel].view(st["codes"].shape)
                    rows.append([off, st["n"], st["codes"].shape[-1], st["a"], st["b"] - st["a"],
                                 stages[st["stage"]], int(lvl_off[st["stage"]])])
                    off += numel
                W = max(16, 1 << max(0, (max(r_[4] for r_ in rows) - 1).bit_length()))
                self.sides[s] = {"flat": flat, "levels": levels, "W": W,
                                 "table": torch.tensor(rows, dtype=torch.int32, device=dev)}

    def context_bytes(self) -> int:
        """Bytes of the stored context: exact rows + the packed code (the decoder's real memory)."""
        p = self.p
        extra = 0 if p is None else tensor_bytes([p["counts"]] + [s_["flat"] for s_ in self.sides if s_ is not None] +
                                                [p[x] for x in ("rho_k", "rho_v", "T_k", "T_v", "mu_k", "mu_v") if x in p])
        return tensor_bytes([self.K[:, :, :self.n0], self.V[:, :, :self.n0]]) + extra

    def append(self, k, v, pos):
        t = k.shape[2]
        if self.n + t > self.K.shape[2]:
            raise RuntimeError("decode buffer exhausted")
        self.K[:, :, self.n:self.n + t], self.V[:, :, self.n:self.n + t] = k, v
        self.n += t

    def _deq(self, st) -> torch.Tensor:
        lut = self.luts[st["stage"]]
        x = lut[st["codes"].int()]                                   # (Hkv, n_s, bytes, 8 / b)
        return x.reshape(x.shape[0], x.shape[1], st["b"] - st["a"])

    def coded_scores(self, qt: torch.Tensor) -> torch.Tensor:
        """qt (B, Hkv, rows, R_k) -> sum over owned steps of qt[band] . levels (B, Hkv, rows, n_c)."""
        p = self.p
        if self.use_kernel:
            return quasar_scores(qt[0].float(), self.sides[0], p["n_c"])[None].to(qt.dtype)
        sc = torch.zeros((*qt.shape[:3], p["n_c"]), dtype=qt.dtype, device=qt.device)
        for st in p["steps"]:
            if st["side"] == 0:
                sc[..., :st["n"]] += qt[..., st["a"]:st["b"]] @ self._deq(st).transpose(-1, -2)
        return sc

    def coded_values(self, pv: torch.Tensor) -> torch.Tensor:
        """pv (B, Hkv, rows, n_c) -> sum_j pv_j levels_j per band (B, Hkv, rows, R_v)."""
        p = self.p
        R = p["T_v"].shape[-1]
        if self.use_kernel:
            return quasar_values(pv[0].float(), self.sides[1], R)[None].to(pv.dtype)
        acc = torch.zeros((*pv.shape[:3], R), dtype=pv.dtype, device=pv.device)
        for st in p["steps"]:
            if st["side"] == 1:
                acc[..., st["a"]:st["b"]] += pv[..., :st["n"]] @ self._deq(st)
        return acc

    def attend(self, q, scaling, qpos, chunk=None):
        B, Hq, Tq, d = q.shape
        Hkv = self.K.shape[1]
        rep = Hq // Hkv
        dt = self.dtype
        qg = q.reshape(B, Hkv, rep * Tq, d).to(dt)          # (h, r, t) rows: HF repeat_kv head order
        n = self.n
        s_ex = (qg @ self.K[:, :, :n].transpose(-1, -2)).float() * scaling
        if n > self.n0:                                     # causal among the appended (question / answer) rows
            row_pos = self.H + torch.arange(n - self.n0, device=q.device)
            bad = row_pos[None, :] > qpos.to(q.device)[:, None]                  # (Tq, n - n0)
            s_ex[..., self.n0:] = s_ex[..., self.n0:].masked_fill(bad.repeat(rep, 1), float("-inf"))
        p = self.p
        coded = p is not None and p["n_c"] > 0
        if coded:
            sc = self.coded_scores(qg @ p["T_k"])
            sc = (sc.float() * p["rho_k"].float()[:, None, :] + (qg @ p["mu_k"][..., None]).float()) * scaling
            att = torch.cat([s_ex, sc], -1).softmax(-1)
        else:
            att = s_ex.softmax(-1)
        o = att[..., :n].to(dt) @ self.V[:, :, :n]
        if coded:
            pc = att[..., n:]
            acc = self.coded_values((pc * p["rho_v"].float()[:, None, :]).to(dt))
            o = o + acc @ p["T_v"].transpose(-1, -2) + pc.sum(-1, keepdim=True).to(dt) * p["mu_v"][:, None, :]
        return o.reshape(B, Hq, Tq, d).transpose(1, 2).to(q.dtype).contiguous()

    @torch.no_grad()
    def check_kernel(self, tol: float = 2e-2) -> Dict:
        """Enable the fused kernels iff they reproduce the reference on random queries (relative
        max error of the attention output <= tol, which covers bf16 rounding of the reference)."""
        p = self.p
        if not TRITON_OK or p is None or p["n_c"] == 0:
            return {"kernel": False, "kernel_rel_err": float("nan")}
        g = torch.Generator(device="cpu").manual_seed(SEED)
        Hq = self.K.shape[1] * 4
        q = torch.randn(1, Hq, 2, self.K.shape[-1], generator=g).to(self.K.device, self.dtype)
        qpos = torch.tensor([self.H, self.H + 1])
        self.use_kernel = False
        ref = self.attend(q, self.K.shape[-1] ** -0.5, qpos).float()
        try:
            self.use_kernel = True
            out = self.attend(q, self.K.shape[-1] ** -0.5, qpos).float()
            err = float((out - ref).abs().max() / ref.abs().max().clamp_min(1e-30))
        except Exception as e:                                       # noqa: BLE001 (reference path stays in use)
            logger.warning(f"  QUASAR kernel unavailable ({e!r}); reference path used")
            err = float("inf")
        self.use_kernel = err <= tol
        return {"kernel": self.use_kernel, "kernel_rel_err": err}


# ════════════════════════════════════════════════════════════════════════════
# MODEL RUNNER
# ════════════════════════════════════════════════════════════════════════════

def bos_prefix(tokenizer) -> List[int]:
    bos = tokenizer.bos_token_id if tokenizer is not None else None
    return [bos] if bos is not None and tokenizer("a").input_ids[:1] == [bos] else []


def check_full_attention(conf, name: str):
    """
    Refuse, before any weight is loaded, a model whose config activates sliding-window
    layers (Mistral passes config.sliding_window to every layer; Qwen2/3 and SmolLM3 only
    through use_sliding_window + layer_types) or soft-capped attention logits: the harness
    computes plain causal softmax attention over all cached tokens.
    """
    sliding = []
    if getattr(conf, "sliding_window", None) is not None and getattr(conf, "use_sliding_window", True) is not False:
        lt = getattr(conf, "layer_types", None)
        sliding = [i for i, t in enumerate(lt) if t == "sliding_attention"] if lt is not None \
            else list(range(conf.num_hidden_layers))
    if sliding or getattr(conf, "attn_logit_softcapping", None):
        raise RuntimeError(f"{name}: sliding-window layers {sliding} / logit soft-capping are not supported")


class Runner:
    def __init__(self, mc: ModelConfig, cfg: Config, model=None, tokenizer=None):
        self.mc, self.cfg = mc, cfg
        if model is None:
            logger.info(f"Loading {mc.name} ({mc.model_id})")
            check_full_attention(AutoConfig.from_pretrained(mc.model_id), mc.name)
            tokenizer = AutoTokenizer.from_pretrained(mc.model_id)
            model = AutoModelForCausalLM.from_pretrained(mc.model_id, attn_implementation=ROUTER_ATTN,
                                                         dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True)
        check_full_attention(model.config, mc.name)
        self.tokenizer, self.model = tokenizer, model.to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.get_decoder().layers
        self.L = len(self.layers)
        conf = self.model.config
        self.Hq, self.Hkv = conf.num_attention_heads, conf.num_key_value_heads
        self.d = self.layers[0].self_attn.head_dim
        self.max_pos = int(getattr(conf, "max_position_embeddings", 4096))
        for i, layer in enumerate(self.layers):
            if getattr(layer.self_attn, "layer_idx", None) != i:
                raise RuntimeError(f"{mc.name}: layer {i} has no matching self_attn.layer_idx")
        self.bos = bos_prefix(tokenizer)
        logger.info(f"  layers={self.L} Hq={self.Hq} Hkv={self.Hkv} d={self.d}")

    def forward(self, ids: torch.Tensor, method: Optional[LayerMethod], **kw):
        pos = torch.arange(ids.shape[1], device=DEVICE)
        if method is None:
            return self.model(input_ids=ids, use_cache=False, **kw)
        sim = Sim(method)
        sim.begin(pos)
        with routed(sim):
            return self.model(input_ids=ids, position_ids=pos[None].expand(ids.shape[0], -1),
                              use_cache=False, **kw)

    def release(self):
        del self.model
        free_memory()


# ════════════════════════════════════════════════════════════════════════════
# SELF-TESTS: codebooks, packing, Props 1-3, distortion model, allocator, decode identity
# ════════════════════════════════════════════════════════════════════════════

MAX_1960_MSE = {1: 0.3634, 2: 0.1175, 3: 0.03454, 4: 0.009497}   # Lloyd-Max, N(0, 1)


def _fake_runner(d: int, Hkv: int, rep: int, W_o: Optional[torch.Tensor] = None, dev=None):
    """The attributes QuasarMethod reads from a Runner (head_dim, heads, o_proj weight)."""
    dev = dev or DEVICE
    W = W_o if W_o is not None else torch.eye(d).repeat(1, Hkv * rep) / math.sqrt(Hkv * rep)
    attn = types.SimpleNamespace(o_proj=types.SimpleNamespace(weight=W.to(dev)))
    return types.SimpleNamespace(d=d, Hq=Hkv * rep, Hkv=Hkv, layers=[types.SimpleNamespace(self_attn=attn)])


@torch.no_grad()
def quasar_tests(cfg: "Config") -> Dict:
    """Synthetic checks of every identity QUASAR relies on.  Raises on any failed identity."""
    out = {}
    g = torch.Generator().manual_seed(SEED)
    for b, ref in MAX_1960_MSE.items():
        mse = lloyd_max_mse(b)
        out[f"lloyd_max_mse_{b}bit"] = mse
        if abs(mse - ref) / ref > 2e-3:
            raise RuntimeError(f"Lloyd-Max {b}-bit MSE {mse:.5f} != Max (1960) {ref}")
    # embedded stages: every prefix is a valid code and the error falls with every stage
    st_bits = tuple(cfg.QUASAR_STAGES)
    u = torch.randn(200000, generator=g)
    codes, cum = embedded_quantize(u, st_bits)
    errs = [float((u - c_) .pow(2).mean()) for c_ in cum]
    for t, e in enumerate(errs):
        out[f"embedded_mse_after_stage{t}"] = e
    if any(errs[t + 1] >= errs[t] for t in range(len(errs) - 1)):
        raise RuntimeError(f"embedded code error must fall with every stage: {errs}")
    if abs(errs[0] - lloyd_max_mse(st_bits[0])) > 2e-3:
        raise RuntimeError("the first embedded stage must be the Lloyd-Max code")
    # packing: one byte-table gather reproduces the levels of every packed code exactly
    for t, b in enumerate(st_bits):
        cc = codes[t][:4096].reshape(64, 64)
        lut = byte_lut(st_bits, t)
        deq = lut[pack_rows(cc, b).int()].reshape(64, 64)
        if not torch.equal(deq, stage_codebooks(st_bits)[t][cc.long()]):
            raise RuntimeError(f"packed stage {t} ({b} bit) does not decode to its levels")
    # Prop. 1: exact single-entry perturbation, including eviction (delta = -inf)
    z = torch.randn(64, generator=g, dtype=torch.float64) * 2
    Vv = torch.randn(64, 8, generator=g, dtype=torch.float64)
    p = z.softmax(0)
    o = p @ Vv
    err = 0.0
    for j, delta in ((3, -2.5), (10, 0.7), (20, 6.0), (int(p.argmax()), -float("inf")), (5, 1e-3)):
        z2 = z.clone()
        z2[j] = z2[j] + delta
        gain = single_entry_gain(p[j], torch.tensor(delta, dtype=torch.float64))
        err = max(err, float((z2.softmax(0) @ Vv - o - gain * (Vv[j] - o)).abs().max()))
    out["prop1_max_abs_error"] = err
    if err > 1e-9:
        raise RuntimeError(f"Prop. 1 is exact, error {err}")
    # Prop. 2: under a query metric S, the metric-whitened KLT beats key PCA at every rank
    d, n = 32, 4000
    Aq = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    Ak = torch.linalg.qr(torch.randn(d, d, generator=g))[0]
    qs = torch.randn(3000, d, generator=g) * torch.logspace(0, -2, d) @ Aq.T      # queries: own subspace
    ks = torch.randn(n, d, generator=g) * torch.logspace(0, -1, d) @ Ak.T         # keys: another one
    S = qs.T @ qs / len(qs)
    one = torch.ones(1, n)
    sq = spectral_side(ks[None], one, S[None], "klt", 0.0, "t")
    se = spectral_side(ks[None], one, None, "klt", 0.0, "t")
    for r in (4, 8, 16):
        lq = float(((qs @ (ks - sq["C"][0, :, :r] @ (sq["T"][0, :, :r] / sq["sigma"][0, :r]).T).T) ** 2).mean())
        le = float(((qs @ (ks - se["C"][0, :, :r] @ (se["T"][0, :, :r] / se["sigma"][0, :r]).T).T) ** 2).mean())
        out[f"prop2_logit_mse_rank{r}_query_metric"], out[f"prop2_logit_mse_rank{r}_key_pca"] = lq, le
        if not lq < le:
            raise RuntimeError(f"Prop. 2: query-metric basis not better than key PCA at rank {r} ({lq} vs {le})")
    # Prop. 3: per-band errors in the orthonormal metric basis add up to the metric error exactly
    rho = sq["C"].div(sq["sigma"][:, None, :]).pow(2).mean(-1).sqrt()
    _, cum = embedded_quantize(sq["C"] / sq["sigma"][:, None, :] / rho[..., None], st_bits)
    bands = band_edges(d, tuple(cfg.QUASAR_BANDS))
    E = band_errors(sq["C"], sq["sigma"], rho, cum, bands)
    khat = (rho[..., None] * cum[0]) @ sq["T"].transpose(-1, -2)                  # stage 0 of every band
    L, _ = psd_sqrt(S[None], 0.0)
    direct = float((((ks[None] - khat) @ L) ** 2).sum())
    table = float(E[..., 1].sum())
    out["prop3_relative_gap"] = abs(direct - table) / direct
    if out["prop3_relative_gap"] > 1e-3:
        raise RuntimeError(f"Prop. 3: band errors {table} != metric error {direct}")
    # allocator: feasible, monotone in the budget, and everything exact when the budget allows
    nlev, npos = 6, 300
    Dl = torch.rand(npos, 1, generator=g) * torch.tensor([1.0, 0.5, 0.25, 0.12, 0.06, 0.0])
    cost = torch.tensor([0.0, 100.0, 200.0, 400.0, 800.0, 4096.0])
    fixed = torch.tensor([64.0, 3000.0, 3000.0, 6000.0, 6000.0])
    prev = -1.0
    for budget in (5e3, 3e4, 1e5, 3e5, 2e6):
        al = head_sync_allocate(Dl, cost, fixed, budget)
        b_ = float(al["bits"])
        if b_ > budget + 1e-3:
            raise RuntimeError(f"allocator exceeded its budget ({b_} > {budget})")
        dist = float(Dl.gather(1, al["choice"][:, None]).sum())
        if prev >= 0 and dist > prev + 1e-6:
            raise RuntimeError("allocator distortion must not rise with the budget")
        prev = dist
    if not bool((al["choice"] == nlev - 1).all()):
        raise RuntimeError("with a budget for every token in full precision, all must be exact")
    # decoupled distortion model vs the actual output error of a real allocation (reported), and
    # the compressed-domain decoder vs the harness reconstruction (an identity: asserted)
    Hkv, rep, dd, H = 2, 2, 32, 600
    fr = _fake_runner(dd, Hkv, rep, dev=torch.device("cpu"))
    tc = replace(cfg, QUASAR_SEGMENT=64, QUASAR_QUERIES=8, QUASAR_LOCAL=16)
    spec = torch.exp(-torch.arange(dd) / 6.0)
    Kx = torch.randn(1, Hkv, H, dd, generator=g) * spec.sqrt() * 3 + 1.0
    Vx = torch.randn(1, Hkv, H, dd, generator=g)
    Qx = torch.randn(1, Hkv * rep, H, dd, generator=g) * spec.sqrt() * 3
    m = QuasarMethod(tc, 0.15, "t", fr)
    m.begin()
    m.freeze_at, m.capture_extra = H, 4
    pos = torch.arange(H)
    K2, V2, hide, tok = m.entries(0, Qx, Kx, Vx, dd ** -0.5, pos)
    m.capture(0, K2, V2, pos, hide, tok)
    qn = torch.randn(1, Hkv * rep, 1, dd, generator=g) * spec.sqrt() * 3
    kn, vn = torch.randn(1, Hkv, 1, dd, generator=g), torch.randn(1, Hkv, 1, dd, generator=g)
    st_h = m.states[0]
    st_h.append(kn, vn, torch.tensor([H]))
    o_h = st_h.attend(qn, dd ** -0.5, torch.tensor([H]), 64)
    st_n = m.native_state(0, Kx, Vx, 4, torch.float32)
    st_n.append(kn, vn, torch.tensor([H]))
    o_n = st_n.attend(qn, dd ** -0.5, torch.tensor([H]))
    out["decode_identity_max_abs"] = float((o_h - o_n).abs().max())
    if out["decode_identity_max_abs"] > 1e-4 * max(1.0, float(o_h.abs().max())):
        raise RuntimeError(f"compressed-domain decoding != harness reconstruction ({out['decode_identity_max_abs']})")
    out["payload_vs_analytic_bits"] = (m.payload(0) * 8 + 0.0) / (m.cbits[0]) - 1.0
    if abs(out["payload_vs_analytic_bits"]) > 1e-9:
        raise RuntimeError(f"packed bytes {m.payload(0)} != allocator bits / 8 {m.cbits[0] / 8}")
    st_b = m.native_state(0, Kx, Vx, 4, torch.bfloat16)             # the deployment format on GPU
    exact_bits = st_b.n0 * Hkv * 2 * dd * 16
    out["native_bf16_bytes_vs_analytic"] = st_b.context_bytes() * 8 / (m.cbits[0] + exact_bits) - 1.0
    if abs(out["native_bf16_bytes_vs_analytic"]) > 1e-9:
        raise RuntimeError("the decoder's measured bytes (bf16) must equal the allocator's bits")
    o_full = chunked_attention(qn, dd ** -0.5, torch.tensor([H]), torch.cat([Kx, kn], 2), torch.cat([Vx, vn], 2),
                               torch.arange(H + 1), torch.full((H + 1,), BIG, dtype=torch.long), None, 64)
    out["toy_layer_rel_output_error"] = float((o_n - o_full).pow(2).sum() / o_full.pow(2).sum())
    out["toy_layer_memory_fraction"] = (m.cbits[0] + float((hide[0, :, :H] >= BIG).sum()) * 2 * dd * 16) / (H * Hkv * 2 * dd * 16)
    logger.info("quasar tests: " + ", ".join(f"{k_}={v_:.4g}" for k_, v_ in out.items()))
    return out


def flop_formula_test():
    """attn_flops_first_step's formula (4 d per readable row per query head) == FlopCounterMode."""
    from torch.utils.flop_counter import FlopCounterMode
    g = torch.Generator().manual_seed(SEED)
    B, Hkv, rep, d, n, nt = 1, 2, 2, 16, 40, 12
    K, V = torch.randn(B, Hkv, n, d, generator=g), torch.randn(B, Hkv, n, d, generator=g)
    tok = {"k": torch.randn(B, Hkv, nt, d, generator=g), "v": torch.randn(B, Hkv, nt, d, generator=g),
           "b": torch.zeros(B, Hkv, nt), "tau": torch.zeros(nt, dtype=torch.long)}
    q = torch.randn(B, Hkv * rep, 1, d, generator=g)
    with FlopCounterMode(display=False) as fc:
        chunked_attention(q, 0.25, torch.tensor([n]), K, V, torch.arange(n), torch.full((n,), BIG), tok, 256)
    measured, analytic = fc.get_total_flops(), Hkv * rep * 4 * d * (n + nt)
    logger.info(f"FLOP formula test: analytic {analytic} vs FlopCounterMode {measured}")
    if measured != analytic:
        raise RuntimeError(f"analytic attention FLOPs {analytic} != measured {measured}")


@torch.no_grad()
def harness_tests(runner: "Runner", ids: List[int]) -> Dict:
    """
    (1) negative control: the fp32 simulation without compression reproduces SDPA;
    (2) identities: chunk-local eviction that keeps every token at 16 bit, and QUASAR with a
        budget that fits every token in full precision, reproduce the full cache.
    In fp32 the logits must agree to 1e-3.  With bf16 / fp16 weights every attention output is
    rounded to the model dtype after an fp32 softmax whose summation order differs between
    the paths, so the check is argmax agreement >= 0.98 (max |dlogit| is reported).
    """
    c = runner.cfg
    x = torch.tensor([ids], device=DEVICE)
    T = x.shape[1]
    base = runner.forward(x, None).logits.float()[0]
    full = runner.forward(x, LayerMethod(c)).logits.float()[0]
    cm = ChunkedMethod(c, chunk_method("keep_all", "evict", 16.0), runner.d, keep_bits=16)
    if cm.plan["tokens"] != c.CHUNK:
        raise RuntimeError("keep-all plan must keep every token")
    qm = QuasarMethod(c, 2.0, "quasar_all", runner)
    qm.freeze_at = T - 4
    out = {}
    for name, ref, other in (("sdpa_vs_fp32_sim", base, full),
                             ("evict_keepall_vs_full", full, runner.forward(x, cm).logits.float()[0]),
                             ("quasar_allexact_vs_full", full, runner.forward(x, qm).logits.float()[0])):
        out[f"{name}_max_abs"] = float((ref - other).abs().max())
        out[f"{name}_argmax"] = float((ref.argmax(-1) == other.argmax(-1)).float().mean())
    logger.info("  harness tests: " + ", ".join(f"{k_}={v_:.3g}" for k_, v_ in out.items()))
    exact = COMPUTE_DTYPE == torch.float32
    for name in ("sdpa_vs_fp32_sim", "evict_keepall_vs_full", "quasar_allexact_vs_full"):
        if (exact and out[f"{name}_max_abs"] > 1e-3) or (not exact and out[f"{name}_argmax"] < 0.98):
            raise RuntimeError(f"harness self-test {name} failed: {out}")
    return out


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════════

def holm(p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.maximum.accumulate((m - np.arange(m)) * p[ok][idx])
    tmp = np.empty(m)
    tmp[idx] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out



# ════════════════════════════════════════════════════════════════════════════
# LONG-CONTEXT BENCHMARKS: LongBench, RULER, long-document perplexity (C4, PG-19)
# ════════════════════════════════════════════════════════════════════════════
#
# Protocol (as in NVIDIA kvpress and most KV-compression papers): the CONTEXT is prefilled
# and compressed; compression is then frozen; the question, the answer prefix and the
# generated tokens are appended exact for every method.  QUERY_AWARE = True instead
# compresses context + question (SnapKV's original, question-aware setting).  Memory is
# matched PER SAMPLE: every compressed method stores at most KEEP x the fp16 cache of the
# context (measured from the captured state, reported per sample).
#
# LongBench prompts: split of the OFFICIAL THUDM/LongBench prompts into context / question /
# answer-prefix parts (identical to NVIDIA kvpress' Xnhyacinth/LongBench conversion; the
# concatenation reproduces the official prompt exactly for 12/16 English tasks and by
# construction for trec, triviaqa, samsum, lcc).  Generation lengths: official
# dataset2maxlen.  Metrics: official LongBench metrics.py / eval.py, verbatim.  Chat
# templates are applied to every task except the official exceptions (trec, triviaqa,
# samsum, lsht, lcc, repobench-p).  Over-long prompts follow the official pred.py rule
# (middle_truncate: token-level, on the whole prompt, before the chat template).

LB_CONTEXT_PREFIX = {
    'narrativeqa': 'You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\n',
    'qasper': 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\n',
    'multifieldqa_en': 'Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\n',
    'hotpotqa': 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n',
    '2wikimqa': 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n',
    'musique': 'Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\n',
    'gov_report': 'You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\n',
    'qmsum': 'You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\n',
    'multi_news': 'You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\n',
    'trec': 'Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n',
    'triviaqa': 'Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n',
    'samsum': 'Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n',
    'passage_count': 'There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\n',
    'passage_retrieval_en': 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n',
    'lcc': 'Please complete the code given below. \n{context}',
    'repobench-p': 'Please complete the code given below. \n{context}',
}
LB_QUESTION_TEMPLATE = {
    'narrativeqa': 'Question: {input}\n\n',
    'qasper': 'Question: {input}\n\n',
    'multifieldqa_en': 'Question: {input}\n',
    'hotpotqa': 'Question: {input}\n',
    '2wikimqa': 'Question: {input}\n',
    'musique': 'Question: {input}\n',
    'gov_report': 'Now, write a one-page summary of the report.\n\n',
    'qmsum': 'Query: {input}\n',
    'multi_news': 'Now, write a one-page summary of all the news.\n\n',
    'trec': '{input}',
    'triviaqa': '{input}',
    'samsum': '{input}',
    'passage_count': 'Please enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\n',
    'passage_retrieval_en': '{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\n',
    'lcc': '{input}',
    'repobench-p': '{input}',
}
LB_ANSWER_PREFIX = {
    'narrativeqa': 'Answer:',
    'qasper': 'Answer:',
    'multifieldqa_en': 'Answer:',
    'hotpotqa': 'Answer:',
    '2wikimqa': 'Answer:',
    'musique': 'Answer:',
    'gov_report': 'Summary:',
    'qmsum': 'Answer:',
    'multi_news': 'Summary:',
    'trec': 'Type:',
    'triviaqa': 'Answer:',
    'samsum': 'Summary:',
    'passage_count': 'The final answer is: ',
    'passage_retrieval_en': 'The answer is: ',
    'lcc': 'Next line of code:\n',
    'repobench-p': 'Next line of code:\n',
}
LB_MAX_NEW_TOKENS = {
    'narrativeqa': 128,
    'qasper': 128,
    'multifieldqa_en': 64,
    'hotpotqa': 32,
    '2wikimqa': 32,
    'musique': 32,
    'gov_report': 512,
    'qmsum': 512,
    'multi_news': 512,
    'trec': 64,
    'triviaqa': 32,
    'samsum': 128,
    'passage_count': 32,
    'passage_retrieval_en': 32,
    'lcc': 64,
    'repobench-p': 64,
}

LB_FIRST_LINE_TASKS = ("trec", "triviaqa", "samsum", "lsht")
LB_NO_CHAT_TASKS = ("trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p")
LB_SUFFIX_STRIP = {"trec": "Type:", "triviaqa": "Answer:", "samsum": "Summary:"}


# ── Official LongBench metrics (THUDM/LongBench/LongBench/metrics.py, English part) ─────
def lb_normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def lb_count_score(prediction, ground_truth, **kwargs):
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return float(0.0 if len(numbers) == 0 else right_num / len(numbers))


def lb_retrieval_score(prediction, ground_truth, **kwargs):
    ground_truth_id = re.findall(r'Paragraph (\d+)', ground_truth)[0]
    numbers = re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth_id))
    return float(0.0 if len(numbers) == 0 else right_num / len(numbers))


def _fuzz_ratio(a: str, b: str) -> float:
    """fuzzywuzzy.fuzz.ratio (official); rapidfuzz is the same Indel ratio; difflib last."""
    try:
        from fuzzywuzzy import fuzz
        return float(fuzz.ratio(a, b))
    except ImportError:
        pass
    try:
        from rapidfuzz import fuzz
        return float(round(fuzz.ratio(a, b)))
    except ImportError:
        import difflib
        return float(round(100 * difflib.SequenceMatcher(None, a, b).ratio()))


def lb_code_sim_score(prediction, ground_truth, **kwargs):
    all_lines = prediction.lstrip('\n').split('\n')
    prediction = ""
    for line in all_lines:
        if ('`' not in line) and ('#' not in line) and ('//' not in line):
            prediction = line
            break
    return _fuzz_ratio(prediction, ground_truth) / 100


def lb_classification_score(prediction, ground_truth, **kwargs):
    em_match_list = []
    all_classes = kwargs["all_classes"]
    for class_name in all_classes:
        if class_name in prediction:
            em_match_list.append(class_name)
    for match_term in em_match_list:            # (official code removes while iterating; kept)
        if match_term in ground_truth and match_term != ground_truth:
            em_match_list.remove(match_term)
    return (1.0 / len(em_match_list)) if ground_truth in em_match_list else 0.0


def lb_rouge_score(prediction, ground_truth, **kwargs):
    try:
        from rouge import Rouge
    except ImportError:
        raise RuntimeError("summarisation tasks need the official `rouge` package (pip install rouge)")
    try:
        scores = Rouge().get_scores([prediction], [ground_truth], avg=True)
    except Exception:
        return 0.0
    return scores["rouge-l"]["f"]


def lb_f1_score(prediction, ground_truth, **kwargs):
    from collections import Counter
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    return (2 * precision * recall) / (precision + recall)


def lb_qa_f1_score(prediction, ground_truth, **kwargs):
    return lb_f1_score(lb_normalize_answer(prediction).split(), lb_normalize_answer(ground_truth).split())


LB_METRIC = {
    "narrativeqa": lb_qa_f1_score, "qasper": lb_qa_f1_score, "multifieldqa_en": lb_qa_f1_score,
    "hotpotqa": lb_qa_f1_score, "2wikimqa": lb_qa_f1_score, "musique": lb_qa_f1_score,
    "gov_report": lb_rouge_score, "qmsum": lb_rouge_score, "multi_news": lb_rouge_score,
    "trec": lb_classification_score, "triviaqa": lb_qa_f1_score, "samsum": lb_rouge_score,
    "passage_retrieval_en": lb_retrieval_score, "passage_count": lb_count_score,
    "lcc": lb_code_sim_score, "repobench-p": lb_code_sim_score,
}


def longbench_sample_score(task: str, prediction: str, answers: Sequence[str], all_classes) -> float:
    """Per-sample score as in the official scorer(); the task score is 100 x the mean."""
    if task in LB_FIRST_LINE_TASKS:
        prediction = prediction.lstrip('\n').split('\n')[0]
    score = 0.0
    for ground_truth in answers:
        score = max(score, LB_METRIC[task](prediction, ground_truth, all_classes=all_classes))
    return 100.0 * score


def ruler_sample_score(task: str, prediction: str, refs: Sequence[str]) -> float:
    """
    RULER (Hsieh et al., COLM 2024), official scripts/eval: control characters are replaced
    by newlines and the prediction stripped (kvpress' copy deletes them instead; the official
    rule is used); case-insensitive substring match, string_match_part for qa_* tasks and
    string_match_all otherwise; per sample x 100.
    """
    pred = re.sub(r"[\x00-\x1f]", "\n", prediction.strip()).strip().lower()
    hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
    return 100.0 * (max(hits) if task.split("_")[0] == "qa" else sum(hits) / len(hits))


# ── LongBench v2 (Bai et al., ACL 2025), official THUDM/LongBench pred.py, 0-shot, no CoT ─────
# prompts/0shot.txt split at the question: CONTEXT + QUESTION is the official prompt exactly
LBV2_CONTEXT = "Please read the following text and answer the question below.\n\n<text>\n{doc}\n</text>\n\n"
LBV2_QUESTION = ("What is the correct answer to this question: {q}\nChoices:\n(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n\n"
                 'Format your response as follows: "The correct answer is (insert answer here)".')


def lbv2_extract_answer(response: str) -> Optional[str]:
    """Official extract_answer: the first 'The correct answer is (X)', else '... is X'."""
    response = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", response) or \
        re.search(r"The correct answer is ([A-D])", response)
    return match.group(1) if match else None


def lbv2_sample_score(prediction: str, answer: str) -> float:
    return 100.0 * float(lbv2_extract_answer(prediction) == answer)


def metric_tests():
    """Hand-computed checks of the metric implementations (run at start-up)."""
    checks = {
        "qa_f1": (lb_qa_f1_score("The answer is Paris.", "paris"), 0.5),
        "qa_f1_exact": (lb_qa_f1_score("an Eiffel tower", "Eiffel Tower"), 1.0),
        "classification": (lb_classification_score("NUM", "NUM", all_classes=["NUM", "LOC"]), 1.0),
        "classification_two": (lb_classification_score("LOC or NUM", "NUM", all_classes=["NUM", "LOC"]), 0.5),
        "retrieval": (lb_retrieval_score("Paragraph 3", "Paragraph 3"), 1.0),
        "count": (lb_count_score("There are 7 of 7", "7"), 1.0),
        "code_sim": (lb_code_sim_score("\nx = foo(1)\nmore", "x = foo(1)"), 1.0),
        "code_sim_skip_comment": (lb_code_sim_score("# c\nx = 1", "x = 1"), 1.0),
        "ruler_all": (ruler_sample_score("niah_multikey_1", "a and b", ["a", "c"]), 50.0),
        "ruler_part": (ruler_sample_score("qa_1", "Paris", ["paris", "rome"]), 100.0),
        "ruler_control_char_is_newline": (ruler_sample_score("niah_single_1", "12\x003", ["123"]), 0.0),
        "first_line": (longbench_sample_score("trec", "\nNUM\nLOC", ["NUM"], ["NUM", "LOC"]), 100.0),
        "lbv2_paren": (lbv2_sample_score("**The correct answer is (B)**", "B"), 100.0),
        "lbv2_bare": (lbv2_sample_score("The correct answer is C.", "C"), 100.0),
        "lbv2_first_match": (lbv2_sample_score("The correct answer is (A). The correct answer is (B)", "B"), 0.0),
    }
    bad = {k: v for k, v in checks.items() if abs(v[0] - v[1]) > 1e-9}
    if bad:
        raise RuntimeError(f"metric self-tests failed: {bad}")
    logger.info(f"metric tests passed ({len(checks)} checks)")


# ── Datasets (Hugging Face) ──────────────────────────────────────────────────────────
# Every loader takes `limit`: with a limit the dataset is STREAMED and only the first rows are
# read, which is how preflight_data() exercises the exact loader code before any model loads.
def _hf_rows(repo: str, *args, limit: Optional[int] = None, **kw) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset(repo, *args, streaming=limit is not None, **kw)
    return [dict(r) for r in (itertools.islice(ds, limit) if limit is not None else ds)]


def load_longbench(task: str, cfg: "Config", limit: Optional[int] = None) -> List[Dict]:
    """
    One LongBench(-E) task as dicts with context / question / answer_prefix / answers /
    all_classes / length.  Xnhyacinth/LongBench (kvpress' parquet conversion; config = task
    or task_e) first, then the official THUDM / zai-org data.zip read directly (the official
    loading script does not run with datasets >= 4).
    """
    name = f"{task}_e" if cfg.LONGBENCH_E else task
    try:
        rows = _hf_rows(cfg.LONGBENCH_HF, name, split="test", limit=limit)
    except Exception as e:
        rows = _load_longbench_zip(name, task, cfg, [f"{cfg.LONGBENCH_HF} {name}: {e!r}"[:300]])[:limit]
    out = []
    for i, r in enumerate(rows):
        ans = r.get("answers")
        ans = list(ans) if ans is not None and not isinstance(ans, str) else [ans]
        cls = r.get("all_classes")
        cls = list(cls) if cls is not None and not isinstance(cls, str) else cls
        out.append({"id": r.get("_id", str(i)), "context": r["context"], "question": r["question"],
                    "answer_prefix": r.get("answer_prefix", LB_ANSWER_PREFIX[task]), "answers": ans,
                    "all_classes": cls, "length": r.get("length"), "task": task})
    return out


def _load_longbench_zip(name, task, cfg, errors) -> List[Dict]:
    import zipfile
    from huggingface_hub import hf_hub_download
    for repo in cfg.LONGBENCH_ZIP_REPOS:
        try:
            path = hf_hub_download(repo_id=repo, filename="data.zip", repo_type="dataset")
            with zipfile.ZipFile(path) as z:
                member = next(m for m in z.namelist() if m.endswith(f"/{name}.jsonl") or m == f"{name}.jsonl")
                raw = [json.loads(line) for line in z.read(member).decode("utf-8").splitlines() if line.strip()]
            break
        except Exception as e:
            errors.append(f"{repo} data.zip: {e!r}"[:300])
    else:
        raise RuntimeError("LongBench could not be loaded:\n  " + "\n  ".join(errors))
    rows = []
    for r in raw:
        inp = r["input"]
        if task in LB_SUFFIX_STRIP:
            inp = inp.removesuffix(LB_SUFFIX_STRIP[task])
        rows.append({**r, "context": LB_CONTEXT_PREFIX[task].format(context=r["context"]),
                     "question": LB_QUESTION_TEMPLATE[task].format(input=inp),
                     "answer_prefix": LB_ANSWER_PREFIX[task]})
    return rows


def load_longbench_v2(cfg: "Config", limit: Optional[int] = None) -> List[Dict]:
    """LongBench v2: THUDM/LongBench-v2, split "train" (official pred.py); official 0-shot
    template with stripped fields; the official aggregate is the accuracy over all items."""
    out = []
    for r in _hf_rows(cfg.LONGBENCH_V2_HF, split="train", limit=limit):
        out.append({"id": r["_id"], "context": LBV2_CONTEXT.format(doc=r["context"].strip()),
                    "question": LBV2_QUESTION.format(q=r["question"].strip(), a=r["choice_A"].strip(),
                                                     b=r["choice_B"].strip(), c=r["choice_C"].strip(),
                                                     d=r["choice_D"].strip()),
                    "answer_prefix": "", "answers": [r["answer"]], "task": "all",
                    "difficulty": r["difficulty"], "length_class": r["length"], "domain": r["domain"]})
    return out


def load_ruler(length: int, cfg: "Config", limit: Optional[int] = None) -> List[Dict]:
    """RULER as hosted by kvpress: simonjegou/ruler, config = context length (4096, ...)."""
    df = pd.DataFrame(_hf_rows(cfg.RULER_HF, str(length), split="test", limit=limit))
    out = []
    for task, g in df.groupby("task", sort=True):
        if cfg.RULER_MAX_PER_TASK and limit is None:
            g = g.sample(n=min(len(g), cfg.RULER_MAX_PER_TASK), random_state=SEED)
        for i, r in g.iterrows():
            out.append({"id": f"{task}-{i}", "context": r["context"], "question": r["question"],
                        "answer_prefix": r["answer_prefix"], "answers": list(r["answer"]),
                        "max_new_tokens": int(r["max_new_tokens"]), "task": task, "length": length})
    return out


def long_ppl_texts(name: str, cfg: "Config"):
    """Raw documents (streamed): C4 (allenai/c4, one validation shard) or PG-19 (the first
    source of PG19_SOURCES that loads; one book per document)."""
    if name == "c4":
        from datasets import load_dataset
        for r in load_dataset(cfg.C4_HF, data_files={"validation": cfg.C4_FILE}, split="validation", streaming=True):
            yield r["text"]
        return
    if name != "pg19":
        raise ValueError(f"unknown long-PPL dataset {name!r}")
    errors = []
    for repo, split in cfg.PG19_SOURCES:
        try:
            from datasets import load_dataset
            ds = iter(load_dataset(repo, split=split, streaming=True))
            first = next(ds)
        except Exception as e:
            errors.append(f"{repo}[{split}]: {e!r}"[:300])
            continue
        yield first["text"]
        for r in ds:
            yield r["text"]
        return
    raise RuntimeError("PG-19 could not be loaded:\n  " + "\n  ".join(errors))


def load_long_ppl(name: str, runner: "Runner", cfg: "Config") -> List[List[int]]:
    """LONG_PPL_N sequences of exactly LONG_PPL_LEN tokens (BOS + text): C4 documents are
    concatenated; PG-19 contributes the first tokens of each book long enough."""
    tok, bos = runner.tokenizer, runner.bos
    L = min(cfg.LONG_PPL_LEN, runner.max_pos) - len(bos)
    seqs, buf = [], []
    sep = tok("\n\n", add_special_tokens=False).input_ids
    for text in long_ppl_texts(name, cfg):
        ids = tok(text, add_special_tokens=False).input_ids
        if name == "c4":
            buf += ids + sep
            while len(buf) >= L and len(seqs) < cfg.LONG_PPL_N:
                seqs.append(bos + buf[:L])
                buf = buf[L:]
        elif len(ids) >= L:
            seqs.append(bos + ids[:L])
        if len(seqs) >= cfg.LONG_PPL_N:
            break
    return seqs


def preflight_data(cfg: "Config", suites: Sequence[str]) -> List[str]:
    """
    Runs every loader the selected suites use on its first row (streaming) and checks the
    fields the evaluation reads, BEFORE any model is loaded: schema drift on the Hub fails
    here in seconds instead of hours into a run.  Returns the problems (empty: all good).
    """
    checks = []
    if "longbench" in suites:
        checks += [(f"LongBench/{t}", lambda t=t: load_longbench(t, cfg, limit=1)) for t in cfg.LONGBENCH_TASKS]
    if "longbench-v2" in suites:
        checks.append(("LongBench-v2", lambda: load_longbench_v2(cfg, limit=1)))
    if "ruler" in suites:
        checks += [(f"RULER/{n}", lambda n=n: load_ruler(n, cfg, limit=1)) for n in cfg.RULER_LENGTHS]
    if "longppl" in suites:
        checks += [(f"long-PPL/{n}", lambda n=n: [{"text": next(iter(long_ppl_texts(n, cfg)))}])
                   for n in cfg.LONG_PPL_DATASETS]
    problems = []
    for label, fn in checks:
        try:
            item = fn()[0]
            empty = [k for k in ("context", "question", "answers", "text") if k in item and not item[k]]
            if empty:
                raise ValueError(f"empty fields {empty}")
            logger.info(f"  preflight {label}: ok ({', '.join(sorted(item))})")
        except Exception as e:
            problems.append(f"{label}: {e!r}"[:400])
            logger.error(f"  preflight {label}: FAILED {e!r}"[:400])
    return problems


# ── Prompt assembly ─────────────────────────────────────────────────────────────────
CHAT_MARGIN = 64                       # tokens reserved for the chat template around the prompt


def middle_truncate(tok, ctx: str, tail: str, max_len: int) -> Tuple[str, bool]:
    """
    Official LongBench / LongBench-v2 truncation (pred.py): the whole prompt is tokenised
    (with special tokens) and, if longer than max_len, only its first and last max_len // 2
    tokens are kept and decoded back to text (skip_special_tokens), BEFORE any chat template.
    The question + answer prefix (tail) lies inside the kept last half, so it is kept
    verbatim and only the context is cut (the context / question token boundary may merge
    differently by at most one token).  Returns the context to use and whether it was cut.
    """
    ids = tok(ctx + tail).input_ids
    if len(ids) <= max_len:
        return ctx, False
    half = max_len // 2
    n_tail = len(tok(tail, add_special_tokens=False).input_ids)
    if n_tail >= half:
        raise ValueError(f"question of {n_tail} tokens does not fit half of the {max_len}-token budget")
    return (tok.decode(ids[:half], skip_special_tokens=True)
            + tok.decode(ids[len(ids) - half:len(ids) - n_tail], skip_special_tokens=True)), True


def encode_item(runner: "Runner", item: Dict, max_new: int, cfg: "Config", chat_ok: bool = True,
                max_prompt: Optional[int] = None) -> Dict:
    """
    Token ids of context and question parts (kvpress style): with a chat template, the user
    turn contains context + question and the generation prompt follows; the answer prefix is
    appended after it.  Without a template the model BOS (if any) starts the context.  The
    prompt is first truncated by the official rule (middle_truncate) to the benchmark's
    max_prompt, and in any case to the model's context minus the generation budget.
    Returns ids, ctx_len (context tokens), context_len (what is compressed: the context, or
    context + question with QUERY_AWARE), and flags.
    """
    tok = runner.tokenizer
    ctx, q, ap = item["context"], item["question"], item["answer_prefix"]
    limit = runner.max_pos - max_new - CHAT_MARGIN
    ctx, truncated = middle_truncate(tok, ctx, q + ap, limit if max_prompt is None else min(max_prompt, limit))
    use_chat = bool(chat_ok and cfg.USE_CHAT_TEMPLATE and getattr(tok, "chat_template", None))
    if use_chat:
        sep = "<<<QUASAR_SEPARATOR_7f3a>>>"
        if sep in ctx:
            sep = "#" * (len(ctx) + 10)                                     # kvpress' separator
        text = tok.apply_chat_template([{"role": "user", "content": ctx + sep}], add_generation_prompt=True,
                                       tokenize=False)
        ctx_text, suffix = text.split(sep)
        ctx_ids = tok.encode(ctx_text, add_special_tokens=False)
        q_text = q + suffix + ap
    else:
        ctx_ids = runner.bos + tok.encode(ctx, add_special_tokens=False)
        q_text = q + ap
    ids = ctx_ids + tok.encode(q_text, add_special_tokens=False)
    return {"ids": ids, "ctx_len": len(ctx_ids), "context_len": len(ids) if cfg.QUERY_AWARE else len(ctx_ids),
            "chat": use_chat, "truncated": truncated}


# ── Matched-memory methods per sample ───────────────────────────────────────────────
@dataclass(frozen=True)
class BenchSpec:
    name: str
    family: str                        # full | kivi | cla | snapkv | streamingllm | mtlp | chunk | quasar | quasar_native | native | kvpress
    keep: Optional[float] = None       # memory budget: fraction of the fp16 context cache
    kivi_bits: Optional[int] = None    # KIVI / MTLP code width
    chunk: Optional[ChunkMethod] = None
    press: Optional[str] = None        # kvpress press name (official implementation)
    variant: Optional[str] = None      # QUASAR variant (QUASAR_VARIANTS)

    @property
    def label(self) -> str:
        """Method family as reported in result rows."""
        if self.family == "chunk":
            return self.chunk.family
        return self.family if self.family in ("quasar_native", "mtlp") else (self.variant or self.family)


@lru_cache(maxsize=1)
def kvpress_module():
    """
    The official NVIDIA kvpress package, or None.  kvpress 0.5.5 imports fire < 0.7, whose
    only use of the module `pipes` (removed in Python 3.13) is pipes.quote == shlex.quote, so
    a one-attribute shim makes it importable on Python 3.13.
    """
    if importlib.util.find_spec("pipes") is None:
        shim = types.ModuleType("pipes")
        shim.quote = shlex.quote
        sys.modules["pipes"] = shim
    try:
        import kvpress
        if Version(transformers.__version__) >= Version("5.3"):
            logger.warning(f"kvpress 0.5.5 needs transformers < 5.3 (its forward hook reads the attention "
                           f"kwarg cache_position, which transformers {transformers.__version__} no longer "
                           f"passes): the official presses are unavailable in this environment")
            return None
        return kvpress
    except Exception as e:
        logger.warning(f"kvpress unavailable ({e!r})")
        return None


def build_press(name: str, keep: float):
    """Official kvpress presses, constructed as in kvpress' evaluate_registry.py (v0.5.5)."""
    K = kvpress_module()
    presses = {"snapkv": lambda: K.SnapKVPress(), "pyramidkv": lambda: K.PyramidKVPress(),
               "adakv_snapkv": lambda: K.AdaKVPress(K.SnapKVPress()),
               "expected_attention": lambda: K.AdaKVPress(K.ExpectedAttentionPress(epsilon=1e-2)),
               "tova": lambda: K.TOVAPress(), "knorm": lambda: K.KnormPress(),
               "streaming_llm": lambda: K.StreamingLLMPress()}
    press = presses[name]()
    press.compression_ratio = 1.0 - keep       # as evaluate.py: set after construction (AdaKV delegates)
    return press


def kvpress_n_kept(k_len: int, keep: float) -> int:
    """kvpress' compute_n_kept(k_len, compression_ratio = 1 - keep), same float arithmetic."""
    ratio = 1.0 - keep
    return 0 if ratio >= 1.0 else max(1, int(k_len * (1 - ratio)))


MASKING_PRESSES = ("adakv_snapkv", "expected_attention")   # AdaKV masks keys: memory is nominal
ALWAYS_RUN = ("full", "native_full")                        # references of every contrast / path control


QUASAR_VARIANTS = {"quasar": {},
                   "quasar_euclid": {"metric": "euclid"},               # plain PCA basis + Euclidean error (prior art)
                   "quasar_rot": {"metric": "euclid", "basis": "rot"},  # no spectral compaction (rotation codec)
                   "quasar_window": {"window_only": True}}              # end-window sensitivities (SnapKV-type prior)


def bench_specs(cfg: "Config", for_ppl: bool = False) -> List[BenchSpec]:
    """Every method at every budget (QUASAR's ablations and its deployment path at PRIMARY_KEEP)."""
    out = [BenchSpec("full", "full")] + [BenchSpec(f"kivi{b}", "kivi", kivi_bits=b) for b in cfg.KIVI_BITS]
    if not for_ppl:
        out.append(BenchSpec("native_full", "native"))       # HF SDPA + DynamicCache reference
    presses = list(cfg.KVPRESS_PRESSES) if (not for_ppl and kvpress_module() is not None) else []
    for r in cfg.KEEP_FRACTIONS:
        t = f"{r:g}"
        out += [BenchSpec(f"kvpress:{pn}@{t}", "kvpress", r, press=pn) for pn in presses]
        out += [BenchSpec(f"snapkv@{t}", "snapkv", r), BenchSpec(f"streamingllm@{t}", "streamingllm", r)]
        out += [BenchSpec(f"snapkv_q{b}@{t}", "mtlp", r, kivi_bits=b) for b in cfg.MTLP_BITS]
        out += [BenchSpec(f"{fam}@{t}", "chunk", r, chunk=chunk_method(f"{fam}@{t}", fam, 1.0)) for fam in ("evict", "am")]
        out.append(BenchSpec(f"quasar@{t}", "quasar", r, variant="quasar"))
    r0 = f"{cfg.PRIMARY_KEEP:g}"
    out += [BenchSpec(f"{v_}@{r0}", "quasar", cfg.PRIMARY_KEEP, variant=v_)
            for v_ in ("quasar_euclid", "quasar_rot", "quasar_window")]
    if cfg.NATIVE_QUASAR and not for_ppl:
        out.append(BenchSpec(f"quasar_native@{r0}", "quasar_native", cfg.PRIMARY_KEEP, variant="quasar"))
    if cfg.BENCH_INCLUDE_CLA:
        out.append(BenchSpec("cla", "cla"))
    if cfg.BENCH_METHODS:                                  # e.g. ("quasar@", "snapkv@")
        out = [s_ for s_ in out if s_.name in ALWAYS_RUN or any(s_.name.startswith(m) for m in cfg.BENCH_METHODS)]
    return out


def chunk_rate_for_budget(P: int, keep: float, method: ChunkMethod, d: int, cfg: "Config"):
    """
    Bits per element of the compacted chunks such that sinks + pending + window (exact) plus
    the chunks fit keep x the fp16 cache of a P-token context.  Falls back to one token per
    chunk if the budget is below it (flagged; the measured memory decides whether the sample
    enters a contrast).
    """
    n_ch = len(chunk_plan(P, cfg, P))
    if n_ch == 0:
        return None, "no chunk completes (context too short)"
    exact = P - n_ch * cfg.CHUNK
    r = 16.0 * (keep * P - exact) / (n_ch * cfg.CHUNK)
    floor = entry_token_bits(method, d, cfg.KEEP_BITS) / (cfg.CHUNK * 2 * d)
    if r < floor:
        return floor, "over budget (cheapest code used)"
    return r, ""


def make_method(spec: BenchSpec, P: int, runner: "Runner", cfg: "Config"):
    if spec.family == "full":
        return LayerMethod(cfg), ""
    if spec.family == "kivi":
        return KIVIMethod(cfg, spec.kivi_bits), ""
    if spec.family == "cla":
        return CLAMethod(cfg, runner.L), ""
    if spec.family in ("native", "kvpress"):
        return None, ""                                     # handled by native_generate
    if spec.family == "snapkv":
        keep = kvpress_n_kept(P, spec.keep)
        return SnapKVMethod(cfg, keep, spec.name), ("" if P > cfg.SNAPKV_WINDOW else "context <= window")
    if spec.family == "streamingllm":
        keep = max(cfg.N_SINK + 1, kvpress_n_kept(P, spec.keep))
        return StreamingLLMMethod(cfg, keep, spec.name), ("" if keep < P else "budget >= context")
    if spec.family == "mtlp":
        m = SnapKVQuantMethod(cfg, spec.keep, spec.kivi_bits, spec.name, P, runner.d)
        return m, ("" if P > cfg.SNAPKV_WINDOW and m.n_q > 0 else "budget below the exact window")
    if spec.family in ("quasar", "quasar_native"):
        note = "" if P > cfg.N_SINK + cfg.WINDOW else "context <= sinks + window"
        return QuasarMethod(cfg, spec.keep, spec.name, runner, **QUASAR_VARIANTS[spec.variant]), note
    r, note = chunk_rate_for_budget(P, spec.keep, spec.chunk, runner.d, cfg)
    if r is None:
        return LayerMethod(cfg), note
    m = ChunkedMethod(cfg, replace(spec.chunk, target_bits=r), runner.d)
    m.name, m.family = spec.name, spec.chunk.family
    if not m.plan["feasible"]:
        return LayerMethod(cfg), "infeasible chunk code"
    return m, note

# ── Systems measurements (definition and fidelity of every field: header, section 6) ──

def _sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def _reset_peak():
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak() -> float:
    return float(torch.cuda.max_memory_allocated()) if DEVICE.type == "cuda" else float("nan")


def tensor_bytes(obj) -> int:
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def pack_codes(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Bit-pack non-negative integer codes (< 2^bits) into a uint8 tensor (MSB first)."""
    c = codes.reshape(-1).to(torch.int64).cpu().numpy()
    bitplanes = ((c[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)
    return torch.from_numpy(np.packbits(bitplanes.reshape(-1)))


@torch.no_grad()
def payload_bytes(method: LayerMethod, context_len: int) -> int:
    """
    MEASURED bytes of the compressed context cache, materialised in its storage format:
    fp16 exact rows; KIVI: its own codes bit-packed + fp16 zero points and scales of every
    group; chunk-local tokens: bit-packed absmax codes + one fp16 scale per vector (AM: + fp16
    bias); QUASAR: QuasarMethod.payload (packed codes of every embedded step, fp16 scales,
    read-out matrices, means and the level histogram); SnapKV + KIVI-b: as KIVI.
    Must agree with the analytic state_bits up to byte padding (checked per sample).
    """
    total = 0
    for li, st in method.states.items():
        if st.shared is not None:
            continue
        n = st.n
        live = (st.hide[:, :, :n] >= BIG) & (st.pos[:n] < context_len)[None, None]
        total += tensor_bytes(st.K[:, :, :n][live].half()) + tensor_bytes(st.V[:, :, :n][live].half())
        if isinstance(method, (KIVIMethod, SnapKVQuantMethod)):
            for part in ((method.stored[li]["k"], method.stored[li]["v"]) if li in method.stored else ()):
                total += tensor_bytes(pack_codes(part["codes"], method.bits)) + \
                    tensor_bytes(part["lo"].half()) + tensor_bytes(part["scale"].half())
        elif isinstance(method, QuasarMethod):
            if li in method.alloc:
                total += method.payload(li)
        elif st.tok is not None:
            kb = method.keep_bits
            for x in (st.tok["k"], st.tok["v"]):
                if kb >= 16:
                    total += tensor_bytes(x.half())
                else:
                    codes, sc = absmax_quant(x, kb)
                    total += tensor_bytes(pack_codes(codes + 2 ** (kb - 1) - 1, kb)) + tensor_bytes(sc.half())
            if method.m.family == "am":
                total += tensor_bytes(st.tok["b"].half())
    return int(total)


def attn_flops_first_step(method: LayerMethod, Hq: int, d: int) -> float:
    """
    ANALYTIC matmul-class attention FLOPs of the FIRST decoded token over the captured state,
    all layers: each query head reads the exact rows live for its KV head plus the new token
    and the token entries, 4 d each (QK and PV; flop_formula_test checks the formula against
    FlopCounterMode).  Every later step adds 4 d Hq L per generated token for every method.
    """
    total = 0.0
    for st in method.states.values():
        src = method.states[st.shared] if st.shared is not None else st
        B, Hkv = src.K.shape[:2]
        rows = float((src.hide[:, :, :src.n] >= BIG).sum()) / B + Hkv          # summed over KV heads
        if src.tok is not None:
            rows += float(torch.isfinite(src.tok["b"]).sum()) / B
        total += (Hq // Hkv) * 4 * d * rows
    return total


class EagerCount:
    """
    Explicit-matmul attention used ONLY inside the FLOP measurement of a native decode step:
    torch's FlopCounterMode does not count every fused SDPA backend (e.g. the CPU flash
    kernel), which would hide the attention cost.  One query token reads all cached keys.
    """

    def attend(self, module, q, k, v, mask, kwargs):
        rep = q.shape[1] // k.shape[1]
        att = (q @ rep_heads(k, rep).transpose(-1, -2)) * (kwargs.get("scaling") or module.scaling)
        if isinstance(mask, torch.Tensor):
            att = att.masked_fill(~mask, float("-inf")) if mask.dtype == torch.bool else att + mask
        return (att.softmax(-1) @ rep_heads(v, rep)).transpose(1, 2).contiguous(), None


def measured_step_flops(fn) -> float:
    """torch.utils.flop_counter over one callable (None if unavailable)."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fc:
            fn()
        return float(fc.get_total_flops())
    except Exception:
        return float("nan")


# ── Generation with a compressed cache ──────────────────────────────────────────────
class DecodeCtl:
    """Single-token (or multi-token) steps against captured LayerStates."""

    def __init__(self, states: Dict[int, LayerState], cfg: "Config"):
        self.states, self.cfg = states, cfg

    def begin(self, positions):
        self.pos = positions

    def attend(self, module, q, k, v, mask, kwargs):
        st = self.states[module.layer_idx]
        scaling = kwargs.get("scaling") or module.scaling
        if st.shared is None:
            st.append(k, v, self.pos)
            src = st
        else:
            src = self.states[st.shared]          # CLA: the source layer already appended this step
        return src.attend(q, scaling, self.pos, self.cfg.ATTN_CHUNK), None


def run_flags() -> Dict:
    """Execution settings that affect timings, recorded with every systems row."""
    return {"cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark), "device": str(DEVICE)}


@torch.no_grad()
def generate(runner: "Runner", method: LayerMethod, ids: List[int], context_len: int, max_new: int,
             stop_ids: set, stop_fn=None, measure: bool = True, prefill: Optional[Dict] = None) -> Tuple[List[int], Dict]:
    """
    Greedy decoding on the reference harness.  One single-pass forward over the prompt with
    compression frozen at context_len (state captured), then token-by-token steps that read
    the compressed state.  Bytes and FLOPs are implementation-independent; the wall-clock and
    allocator fields are prefixed harness_ because they time an fp32 reference without fused
    kernels and must never be read as deployment speed (only path = native rows are).  With a
    shared prefill of the context (one-shot methods), the method compresses it and runs only
    the question tokens; harness_prefill_s is then compression + question time.
    """
    P = len(ids)
    shared = prefill is not None and method.one_shot
    _reset_peak()
    _sync()
    t0 = time.perf_counter()
    if shared:
        if prefill["H"] != context_len:
            raise ValueError("the shared prefill must cover exactly the context")
        compress_prefill(method, prefill, (P - context_len) + max_new + 1)
        logits = run_tokens(runner, method.states, ids[context_len:], context_len) if P > context_len \
            else prefill["last_logits"]
    else:
        method.freeze_at, method.capture_extra = context_len, max_new + 1
        logits = runner.forward(torch.tensor([ids], device=DEVICE), method, logits_to_keep=1).logits[0, -1]
    nxt = int(logits.float().argmax())
    _sync()
    prefill_s = time.perf_counter() - t0
    peak_prefill = _peak()
    method.capture_extra = None
    states = method.states
    mem = method.state_bits(runner.d, context_len)
    full_bits = context_len * runner.L * runner.Hkv * 2 * runner.d * 16.0
    info = {"memory_fraction": mem["total"] / full_bits, "analytic_bytes": mem["total"] / 8.0,
            "full_cache_bytes": full_bits / 8.0, "prompt_len": P, "context_len": context_len,
            "harness_prefill_s": prefill_s, "harness_peak_prefill_bytes": peak_prefill,
            "path": "harness", "model_dtype": str(next(runner.model.parameters()).dtype),
            "attention_dtype": "float32 (reference harness)", "shared_prefill": shared,
            "harness_shared_prefill_s": prefill["seconds"] if shared else float("nan"), **run_flags()}
    if measure:
        info["payload_bytes"] = payload_bytes(method, context_len)
        info["payload_vs_analytic"] = info["payload_bytes"] / info["analytic_bytes"] - 1 if mem["total"] else 0.0
        info["sim_state_bytes"] = tensor_bytes([[st.K[:, :, :st.n], st.V[:, :, :st.n], st.tok]
                                                for st in states.values() if st.shared is None])
        info["attn_flops_first_step"] = (method.attn_flops_first_step(runner.Hq, runner.d)
                                         if isinstance(method, QuasarMethod) else
                                         attn_flops_first_step(method, runner.Hq, runner.d))
        info["attn_flops_first_step_full"] = 4.0 * runner.d * (P + 1) * runner.Hq * runner.L
    if isinstance(method, QuasarMethod):
        info.update(method.diagnostics())
    ctl = DecodeCtl(states, runner.cfg)
    out = []
    _reset_peak()
    _sync()
    t1 = time.perf_counter()
    for i in range(max_new):
        out.append(nxt)
        if nxt in stop_ids or i == max_new - 1 or (stop_fn is not None and stop_fn(out)):
            break
        pos = torch.tensor([P + i], device=DEVICE)
        ctl.begin(pos)

        def step(_n=nxt, _p=pos):
            with routed(ctl):
                return runner.model(input_ids=torch.tensor([[_n]], device=DEVICE), position_ids=_p[None],
                                    use_cache=False).logits[0, -1]
        if measure and i == 0 and runner.cfg.MEASURE_FLOPS:
            snapshot = {li: st.n for li, st in states.items()}
            info["flops_first_step_measured"] = measured_step_flops(step)
            for li, st in states.items():                           # undo the measured step's append
                st.n = snapshot[li]
        lg = step()
        nxt = int(lg.float().argmax())
    _sync()
    decode_s = time.perf_counter() - t1
    info.update(n_generated=len(out), harness_decode_s=decode_s, harness_peak_decode_bytes=_peak(),
                harness_tokens_per_s=(len(out) - 1) / decode_s if len(out) > 1 and decode_s > 0 else float("nan"))
    method.states = {}
    free_memory()
    return out, info


@torch.no_grad()
def native_generate(runner: "Runner", ids: List[int], context_len: int, max_new: int, stop_ids: set,
                    stop_fn=None, press=None, nominal_keep: Optional[float] = None) -> Tuple[List[int], Dict]:
    """
    Hugging Face SDPA + DynamicCache path, optionally with an OFFICIAL kvpress press.  Mirrors
    kvpress' KVPressTextGenerationPipeline._forward / generate_answer (v0.5.5): the context is
    prefilled through model.model under the press, then question tokens at positions
    context_len.. and greedy decoding read the (compressed) cache.  Same token ids as the harness.
    """
    from transformers import DynamicCache
    if Router.active is not None:
        raise RuntimeError("native path must run without an active harness controller")
    model = runner.model
    ctx = torch.tensor([ids[:context_len]], device=DEVICE)
    qst = torch.tensor([ids[context_len:]], device=DEVICE)
    cache = DynamicCache()
    _reset_peak()
    _sync()
    t0 = time.perf_counter()
    with (press(model) if press is not None else nullcontext()):
        h = model.model(input_ids=ctx, past_key_values=cache).last_hidden_state
    _sync()
    prefill_s = time.perf_counter() - t0
    peak_prefill = _peak()
    lens = [cache.get_seq_length(li) for li in range(runner.L)]
    cache_bytes = sum(tensor_bytes([lay.keys, lay.values]) for lay in cache.layers)
    t1 = time.perf_counter()
    if qst.shape[1] > 0:
        pos = torch.arange(context_len, context_len + qst.shape[1], device=DEVICE)[None]
        logits = model(input_ids=qst, past_key_values=cache, position_ids=pos, logits_to_keep=1).logits[0, -1]
        pos = pos[:, -1:] + 1
    else:
        logits = model.get_output_embeddings()(h[0, -1])
        pos = torch.tensor([[context_len]], device=DEVICE)
    nxt = int(logits.float().argmax())
    _sync()
    ttft_s = prefill_s + time.perf_counter() - t1
    full_bytes = context_len * runner.L * runner.Hkv * 2 * runner.d * 2
    measured_fraction = sum(lens) / (runner.L * context_len)
    info = {"prompt_len": len(ids), "context_len": context_len, "prefill_s": prefill_s, "ttft_s": ttft_s,
            "peak_prefill_bytes": peak_prefill, "path": "native", "full_cache_bytes": float(full_bytes),
            "model_dtype": str(next(model.parameters()).dtype), "attention_dtype": "model dtype (SDPA)",
            "cache_bytes_measured": float(cache_bytes),
            "memory_fraction": nominal_keep if nominal_keep is not None else measured_fraction,
            "memory_fraction_measured": measured_fraction, **run_flags(),
            # rows read by the first decoded token: compressed context + question + itself
            "attn_flops_first_step": 4.0 * runner.d * runner.Hq * (sum(lens) + runner.L * (len(ids) - context_len + 1)),
            "attn_flops_first_step_full": 4.0 * runner.d * (len(ids) + 1) * runner.Hq * runner.L}
    out = []
    _reset_peak()
    _sync()
    t2 = time.perf_counter()
    for i in range(max_new):
        out.append(nxt)
        if nxt in stop_ids or i == max_new - 1 or (stop_fn is not None and stop_fn(out)):
            break
        if i == 0 and runner.cfg.MEASURE_FLOPS:
            snap = [(lay.keys, lay.values) for lay in cache.layers]
            with routed(EagerCount()):                    # explicit matmuls: counted by torch
                info["flops_first_step_measured"] = measured_step_flops(
                    lambda: model(input_ids=torch.tensor([[nxt]], device=DEVICE), past_key_values=cache,
                                  position_ids=pos))
            for lay, (kk, vv) in zip(cache.layers, snap):           # undo the measured step's append
                lay.keys, lay.values = kk, vv
        lg = model(input_ids=torch.tensor([[nxt]], device=DEVICE), past_key_values=cache, position_ids=pos).logits[0, -1]
        pos = pos + 1
        nxt = int(lg.float().argmax())
    _sync()
    decode_s = time.perf_counter() - t2
    info.update(n_generated=len(out), decode_s=decode_s, peak_decode_bytes=_peak(),
                tokens_per_s=(len(out) - 1) / decode_s if len(out) > 1 and decode_s > 0 else float("nan"),
                payload_bytes=float(cache_bytes))
    cache = None                                    # release the cache before returning
    free_memory()
    return out, info


def run_spec(runner, spec: BenchSpec, enc: Dict, max_new: int, stop: set, stop_fn, cfg,
             pre: Optional[Dict] = None) -> Tuple[List[int], Dict, str]:
    """One (sample, method) generation on the right path."""
    if spec.family == "native":
        gen, info = native_generate(runner, enc["ids"], enc["context_len"], max_new, stop, stop_fn)
        return gen, info, ""
    if spec.family == "kvpress":
        nominal = kvpress_n_kept(enc["context_len"], spec.keep) / enc["context_len"]
        try:
            press = build_press(spec.press, spec.keep)
            gen, info = native_generate(runner, enc["ids"], enc["context_len"], max_new, stop, stop_fn, press,
                                        nominal_keep=nominal if spec.press in MASKING_PRESSES else None)
        except Exception as e:                        # e.g. SnapKV asserts context > window
            return [], {"memory_fraction": float("nan"), "path": "native"}, f"kvpress error: {e!r}"[:200]
        return gen, info, ("masked keys: nominal memory" if spec.press in MASKING_PRESSES else "")
    method, note = make_method(spec, enc["context_len"], runner, cfg)
    if spec.family == "quasar_native":
        if len(enc["ids"]) <= enc["context_len"]:
            return [], {"memory_fraction": float("nan"), "path": "native"}, "no question tokens (query-aware)"
        gen, info = native_quasar_generate(runner, method, enc["ids"], enc["context_len"], max_new, stop, stop_fn)
        return gen, info, note
    gen, info = generate(runner, method, enc["ids"], enc["context_len"], max_new, stop, stop_fn,
                         prefill=pre if spec.family in SHARED_FAMILIES else None)
    return gen, info, note


def stop_ids_for(runner: "Runner") -> set:
    ids = set()
    for src in (runner.tokenizer.eos_token_id, getattr(runner.model.generation_config, "eos_token_id", None)):
        if src is None:
            continue
        ids |= set(src) if isinstance(src, (list, tuple)) else {int(src)}
    return ids


@torch.no_grad()
def generation_equivalence(runner: "Runner", specs: List[BenchSpec], ids: List[int], context_len: int,
                           n_new: int, cfg: "Config") -> pd.DataFrame:
    """
    The fast path (frozen single pass + compressed-state decoding) must reproduce a single
    pass over prompt + generated tokens with the same horizon (teacher forcing).  For the full
    cache it must also match Hugging Face's own cached greedy generate().  context_len is the
    context / question boundary (never the query-aware one), so the official kvpress
    pipeline check always has question tokens.
    """
    rows = []
    harness_gen = {}
    for spec in specs:
        if spec.family in ("native", "kvpress", "quasar_native"):
            continue
        m, note = make_method(spec, context_len, runner, cfg)
        gen, _ = generate(runner, m, ids, context_len, n_new, set())
        harness_gen[spec.name] = gen
        shared_agree = float("nan")
        if m.one_shot and spec.family in SHARED_FAMILIES:            # the shared-prefill path
            m2, _ = make_method(spec, context_len, runner, cfg)
            gen2, _ = generate(runner, m2, ids, context_len, n_new, set(),
                               prefill=shared_prefill(runner, ids[:context_len]))
            shared_agree = float(np.mean([a == b for a, b in zip(gen, gen2)])) if len(gen) == len(gen2) else 0.0
        ref_m, _ = make_method(spec, context_len, runner, cfg)
        ref_m.freeze_at = context_len
        x = torch.tensor([ids + gen[:-1]], device=DEVICE)
        lg = runner.forward(x, ref_m, logits_to_keep=len(gen)).logits[0].float()
        agree = float((lg.argmax(-1).cpu() == torch.tensor(gen)).float().mean())
        rows.append({"method": spec.name, "argmax_agreement": agree, "shared_path_agreement": shared_agree,
                     "n_tokens": len(gen), "note": note})
    full = [r for r in rows if r["method"] == "full"]
    if full:
        hf = runner.model.generate(torch.tensor([ids], device=DEVICE), max_new_tokens=n_new, do_sample=False,
                                   num_beams=1, eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        m, _ = make_method(BenchSpec("full", "full"), context_len, runner, cfg)
        gen, _ = generate(runner, m, ids, context_len, n_new, set())
        full[0]["hf_generate_agreement"] = float(np.mean([a == b for a, b in zip(gen, hf)]))
    for spec in specs:
        if spec.family == "quasar_native":
            m, _ = make_method(spec, context_len, runner, cfg)
            gen, inf = native_quasar_generate(runner, m, ids, context_len, n_new, set())
            twin = f"quasar@{spec.keep:g}"
            row = {"method": spec.name, "n_tokens": len(gen), "note": "deployment path (packed code)",
                   "harness_twin": twin, "memory_fraction_measured": inf["memory_fraction_measured"],
                   "memory_fraction": inf["memory_fraction"]}
            if twin in harness_gen:
                row["harness_agreement"] = float(np.mean([a == b for a, b in zip(gen, harness_gen[twin])]))
            rows.append(row)
            continue
        if spec.family not in ("native", "kvpress"):
            continue
        try:
            press = build_press(spec.press, spec.keep) if spec.family == "kvpress" else None
            gen, _ = native_generate(runner, ids, context_len, n_new, set(), press=press)
        except Exception as e:                       # recorded, never fatal (e.g. a press failing)
            rows.append({"method": spec.name, "note": f"native path error: {e!r}"[:200]})
            logger.warning(f"  generation equivalence: {spec.name} failed ({e!r})"[:300])
            continue
        row = {"method": spec.name, "n_tokens": len(gen), "note": "native path"}
        # harness counterpart: full (fp32 harness vs model-dtype SDPA) and the SnapKV /
        # StreamingLLM replicas vs the official presses (same kept set => same tokens)
        twin = "full" if spec.family == "native" else \
            {"snapkv": "snapkv", "streaming_llm": "streamingllm"}.get(spec.press, "") + f"@{spec.keep:g}"
        if twin in harness_gen:
            row["harness_agreement"] = float(np.mean([a == b for a, b in zip(gen, harness_gen[twin])]))
            row["harness_twin"] = twin
        try:
            row.update(official_pipeline_agreement(runner, ids, context_len, n_new, spec))
        except Exception as e:
            row["note"] = f"official pipeline error: {e!r}"[:200]
        rows.append(row)
    df = pd.DataFrame(rows)
    logger.info("  generation equivalence: " + ", ".join(
        f"{r['method']}={r.get('argmax_agreement', r.get('harness_agreement', float('nan'))):.3f}"
        for r in rows) +
        (f"; full vs HF generate {full[0]['hf_generate_agreement']:.3f}" if full else ""))
    return df


@torch.no_grad()
def official_pipeline_agreement(runner: "Runner", ids, context_len, n_new, spec: BenchSpec) -> Dict:
    """
    native_generate must reproduce kvpress' own KVPressTextGenerationPipeline._forward (same
    context / question ids, same press, its EOS-only stopping); 1.0 = identical decoded answer.
    """
    K = kvpress_module()
    if K is None:
        return {}
    pipe = K.KVPressTextGenerationPipeline(model=runner.model, tokenizer=runner.tokenizer)
    eos = runner.model.generation_config.eos_token_id
    eos = set(eos) if isinstance(eos, (list, tuple)) else ({int(eos)} if eos is not None else set())
    q = ids[context_len:]
    if not q:
        raise ValueError("the official pipeline check needs question tokens after context_len")
    press = build_press(spec.press, spec.keep) if spec.family == "kvpress" else None
    ans = pipe._forward({"context_ids": torch.tensor([ids[:context_len]]),
                         "questions_ids": [torch.tensor([q])]}, max_new_tokens=n_new, press=press)[0]
    press = build_press(spec.press, spec.keep) if spec.family == "kvpress" else None
    gen, _ = native_generate(runner, ids, context_len, n_new, eos, press=press)
    mine = runner.tokenizer.decode(gen, skip_special_tokens=True)
    return {"official_pipeline_agreement": float(mine == ans)}



# ── Shared prefill (one exact context pass per sample for every one-shot method) ──────
#
# A one-shot method (full cache, QUASAR, SnapKV, SnapKV + KIVI-b, StreamingLLM) compresses only at the horizon
# H: every query before H reads the exact cache.  Its keys, values and queries over the context
# are therefore the SAME for every such method, so the context is prefilled once per sample
# (fp32 attention, exactly as each method's own single pass computes it), and each method only
# compresses the captured cache and runs the question and the answer from its own state.
# generation_equivalence() checks this path against each method's single pass
# (shared_path_agreement).  Methods that compress while prefilling (KIVI, chunk-local evict /
# AM, CLA) keep the single pass.

SHARED_FAMILIES = ("full", "quasar", "snapkv", "streamingllm", "mtlp")


class QRows:
    """The query rows a context pass kept, addressed by absolute position (q_at)."""

    def __init__(self, rows: torch.Tensor, positions: torch.Tensor, H: int):
        self.rows = rows                                                  # (B, Hq, n, d)
        self.shape = (rows.shape[0], rows.shape[1], H, rows.shape[3])
        self.index = torch.full((H,), -1, dtype=torch.long, device=rows.device)
        self.index[positions.to(rows.device)] = torch.arange(len(positions), device=rows.device)

    def at(self, pos: torch.Tensor) -> torch.Tensor:
        idx = self.index[pos.to(self.index.device)]
        if bool((idx < 0).any()):
            raise KeyError("query rows requested that the shared prefill did not keep")
        return self.rows[:, :, idx]


def q_at(q, pos: torch.Tensor) -> torch.Tensor:
    """Rows of q (B, Hq, T, d) at absolute positions pos, for a tensor or a QRows."""
    return q.at(pos) if isinstance(q, QRows) else q[:, :, pos.to(q.device)]


def kept_query_positions(H: int, cfg: "Config") -> torch.Tensor:
    """Every context position whose query a one-shot method reads: QUASAR's coverage groups and the
    last max(SNAPKV_WINDOW, W_OBS) positions (SnapKV's observation window)."""
    pos = set(range(max(0, H - max(cfg.SNAPKV_WINDOW, cfg.W_OBS)), H))
    for g_ in coverage_groups(H, cfg):
        pos.update(int(p) for p in g_)
    return torch.tensor(sorted(pos), dtype=torch.long)


class PrefillCapture:
    """Exact fp32 attention over the context (LayerMethod's computation) that records every
    layer's keys, values, scaling and the query rows at kept_query_positions."""

    def __init__(self, keep_q: torch.Tensor, cfg: "Config"):
        self.keep_q, self.cfg = keep_q, cfg
        self.K, self.V, self.Q, self.scaling = {}, {}, {}, {}

    def begin(self, positions):
        self.pos = positions

    def attend(self, module, q, k, v, mask, kwargs):
        li, scaling = module.layer_idx, kwargs.get("scaling") or module.scaling
        self.K[li], self.V[li], self.scaling[li] = k, v, scaling
        self.Q[li] = q[:, :, self.keep_q.to(q.device)]
        return chunked_attention(q, scaling, self.pos, k, v, self.pos, torch.full((k.shape[2],), BIG, dtype=torch.long),
                                 None, self.cfg.ATTN_CHUNK), None


@torch.no_grad()
def shared_prefill(runner: "Runner", ctx_ids: List[int]) -> Dict:
    H = len(ctx_ids)
    keep = kept_query_positions(H, runner.cfg)
    cap = PrefillCapture(keep, runner.cfg)
    pos = torch.arange(H, device=DEVICE)
    cap.begin(pos)
    _sync()
    t0 = time.perf_counter()
    with routed(cap):
        lg = runner.model(input_ids=torch.tensor([ctx_ids], device=DEVICE), position_ids=pos[None],
                          use_cache=False, logits_to_keep=1).logits[0, -1]
    _sync()
    return {"H": H, "K": cap.K, "V": cap.V, "Q": cap.Q, "keep": keep, "scaling": cap.scaling,
            "last_logits": lg, "seconds": time.perf_counter() - t0}


@torch.no_grad()
def compress_prefill(method: LayerMethod, pre: Dict, extra: int):
    """Compress a shared prefill with a one-shot method; the captured states go to method.states."""
    H = pre["H"]
    method.begin()
    method.freeze_at, method.capture_extra = H, extra
    pos = torch.arange(H, device=DEVICE)
    for li in sorted(pre["K"]):
        q = QRows(pre["Q"][li], pre["keep"], H)
        K, V, hide, tok = method.entries(li, q, pre["K"][li], pre["V"][li], pre["scaling"][li], pos)
        method.capture(li, K, V, pos, hide, tok)
    method.capture_extra = None


@torch.no_grad()
def run_tokens(runner: "Runner", states: Dict[int, LayerState], toks: List[int], start: int, hidden: bool = False):
    """Feed toks at positions start.. through the compressed states (appended exact); returns the
    last logits, or every hidden state."""
    ctl = DecodeCtl(states, runner.cfg)
    pos = torch.arange(start, start + len(toks), device=DEVICE)
    ctl.begin(pos)
    x = torch.tensor([toks], device=DEVICE)
    with routed(ctl):
        if hidden:
            return runner.model.get_decoder()(input_ids=x, position_ids=pos[None], use_cache=False).last_hidden_state[0]
        return runner.model(input_ids=x, position_ids=pos[None], use_cache=False, logits_to_keep=1).logits[0, -1]

# ── QUASAR on the deployment path (model dtype, packed code, compressed-domain decoding) ──
class NativeCapture:
    """Prefill with the model's own SDPA kernel (model dtype) that records every layer's keys,
    values and the query rows at kept positions: what a serving system has after prefill."""

    def __init__(self, keep_q: torch.Tensor):
        self.keep_q = keep_q
        self.K, self.V, self.Q, self.scaling = {}, {}, {}, {}

    def begin(self, positions):
        self.pos = positions

    def attend(self, module, q, k, v, mask, kwargs):
        li = module.layer_idx
        self.K[li], self.V[li] = k, v
        self.scaling[li] = kwargs.get("scaling") or module.scaling
        self.Q[li] = q[:, :, self.keep_q.to(q.device)]
        return _SDPA(module, q, k, v, mask, **kwargs)


@torch.no_grad()
def native_quasar_generate(runner: "Runner", method: "QuasarMethod", ids: List[int], context_len: int,
                           max_new: int, stop_ids: set, stop_fn=None, measure: bool = True) -> Tuple[List[int], Dict]:
    """
    QUASAR as it would be deployed: SDPA prefill of the context in the model dtype (queries kept
    only at the coverage positions), compression of every layer (timed), the stored code packed,
    the full context cache freed, then the question and greedy decoding read the packed code
    through QuasarState.attend (no key or value is ever reconstructed).  ttft_s = prefill +
    compression + question; cache_bytes_measured = bytes the decoder holds for the context.
    """
    if Router.active is not None:
        raise RuntimeError("native path must run without an active harness controller")
    model, H, dt = runner.model, context_len, next(runner.model.parameters()).dtype
    keep = kept_query_positions(H, runner.cfg)
    cap = NativeCapture(keep)
    pos = torch.arange(H, device=DEVICE)
    cap.begin(pos)
    _reset_peak()
    _sync()
    t0 = time.perf_counter()
    with routed(cap):
        model.model(input_ids=torch.tensor([ids[:H]], device=DEVICE), position_ids=pos[None], use_cache=False)
    _sync()
    prefill_s = time.perf_counter() - t0
    extra = (len(ids) - H) + max_new + 1
    method.begin()
    method.freeze_at, method.capture_extra = H, extra
    states = {}
    t1 = time.perf_counter()
    for li in sorted(cap.K):
        al = method.compress_layer(li, QRows(cap.Q[li], keep, H), cap.K[li], cap.V[li], cap.scaling[li], H)
        if al is not None:
            method.cbits[li] = al["bits"]
            method.record(al)
            al["packed"] = method.pack(al)
            al.pop("sides")
            method.alloc[li] = al
        states[li] = method.native_state(li, cap.K[li], cap.V[li], extra, dt)
        cap.K[li] = cap.V[li] = cap.Q[li] = None                   # the full context cache is released
    _sync()
    compress_s = time.perf_counter() - t1
    kern = {"kernel": False, "kernel_rel_err": float("nan")}
    if runner.cfg.QUASAR_KERNEL and TRITON_OK and DEVICE.type == "cuda":
        coded = [st for st in states.values() if st.p is not None and st.p["n_c"] > 0]
        if coded:
            kern = coded[0].check_kernel()
            for st in coded:
                st.use_kernel = kern["kernel"]
    peak_prefill = _peak()
    ctx_bytes = sum(st.context_bytes() for st in states.values())
    full_bytes = H * runner.L * runner.Hkv * 2 * runner.d * 2
    t2 = time.perf_counter()
    if len(ids) > H:
        logits = run_tokens(runner, states, ids[H:], H)
    else:                                                          # (no question: last context logits are not kept)
        raise ValueError("the native QUASAR path needs question tokens after the context")
    nxt = int(logits.float().argmax())
    _sync()
    ttft_s = prefill_s + compress_s + time.perf_counter() - t2
    bits = sum(method.cbits.values()) + sum(
        float(st.n0) * runner.Hkv * 2 * runner.d * 16.0 for st in states.values())
    info = {"prompt_len": len(ids), "context_len": H, "prefill_s": prefill_s, "compress_s": compress_s,
            "ttft_s": ttft_s, "peak_prefill_bytes": peak_prefill, "path": "native", "full_cache_bytes": float(full_bytes),
            "model_dtype": str(dt), "attention_dtype": "model dtype (compressed-domain QUASAR)",
            "cache_bytes_measured": float(ctx_bytes), "memory_fraction": bits / (full_bytes * 8.0),
            "memory_fraction_measured": ctx_bytes / full_bytes, "analytic_bytes": bits / 8.0, **run_flags(), **kern,
            **method.diagnostics()}
    if measure:
        method.states = states
        info["attn_flops_first_step"] = method.attn_flops_first_step(runner.Hq, runner.d)
        info["attn_flops_first_step_full"] = 4.0 * runner.d * (len(ids) + 1) * runner.Hq * runner.L
    ctl = DecodeCtl(states, runner.cfg)
    out = []
    _reset_peak()
    _sync()
    t3 = time.perf_counter()
    for i in range(max_new):
        out.append(nxt)
        if nxt in stop_ids or i == max_new - 1 or (stop_fn is not None and stop_fn(out)):
            break
        p_ = torch.tensor([len(ids) + i], device=DEVICE)
        ctl.begin(p_)
        with routed(ctl):
            lg = model(input_ids=torch.tensor([[nxt]], device=DEVICE), position_ids=p_[None],
                       use_cache=False).logits[0, -1]
        nxt = int(lg.float().argmax())
    _sync()
    decode_s = time.perf_counter() - t3
    info.update(n_generated=len(out), decode_s=decode_s, peak_decode_bytes=_peak(),
                tokens_per_s=(len(out) - 1) / decode_s if len(out) > 1 and decode_s > 0 else float("nan"),
                payload_bytes=float(ctx_bytes))
    method.states, method.alloc = {}, {}
    free_memory()
    return out, info


# ── Decode-attention microbenchmark (one layer, synthetic cache, CUDA graphs on GPU) ─────
@torch.no_grad()
def attention_microbench(cfg: "Config", d: int = 128, Hkv: int = 8, rep: int = 4, dtype=None) -> pd.DataFrame:
    """
    Latency of ONE decode step of attention for one layer at matched memory, isolating the
    attention kernel from the rest of the model: (a) SDPA over the full fp16/bf16 cache, (b) SDPA
    over the tokens an evicting method keeps at the same bytes (keep x context), (c) QUASAR's
    compressed-domain attention over a code produced by QUASAR itself from a synthetic layer
    (keys with a decaying spectrum and a shared offset, random values, queries from the same
    subspace).  On CUDA each callable is captured in a CUDA graph (launch overhead removed, as in
    a serving engine) and replayed; on CPU it is timed eagerly.  Columns: context, keep, method,
    microseconds per step (median of MICRO_REPEATS), cache bytes, speed-up vs full SDPA.
    The reference path materialises each step's dequantised block (one gather + one matmul per
    step); the kernels read only the packed bytes.
    """
    dtype = dtype or (COMPUTE_DTYPE if DEVICE.type == "cuda" else torch.float32)
    rows = []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for H in cfg.MICRO_LENGTHS:
        spec = torch.exp(-torch.arange(d) / (d / 8.0))                     # decaying key spectrum
        Kc = (torch.randn(1, Hkv, H, d, generator=g) * spec.sqrt() + 2.0).to(DEVICE)
        Vc = torch.randn(1, Hkv, H, d, generator=g).to(DEVICE)
        keep_q = kept_query_positions(H, cfg)
        Qc = (torch.randn(1, Hkv * rep, len(keep_q), d, generator=g) * spec.sqrt()).to(DEVICE)
        scaling = d ** -0.5
        fake = types.SimpleNamespace(d=d, Hq=Hkv * rep, layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(
            o_proj=types.SimpleNamespace(weight=torch.eye(d).repeat(1, Hkv * rep).to(DEVICE) / math.sqrt(Hkv * rep))))])
        q1 = torch.randn(1, Hkv * rep, 1, d, generator=g).to(DEVICE, dtype)
        Kf, Vf = Kc.to(dtype).contiguous(), Vc.to(dtype).contiguous()          # Hkv heads: GQA-aware SDPA
        full = lambda: F.scaled_dot_product_attention(q1, Kf, Vf, enable_gqa=True)
        res = {"full_sdpa": (_time_callable(full, cfg), tensor_bytes([Kf, Vf]))}
        for keep in cfg.KEEP_FRACTIONS:
            n_keep = max(1, int(H * keep))
            Ke, Ve = Kf[:, :, :n_keep].contiguous(), Vf[:, :, :n_keep].contiguous()
            ev = lambda Ke=Ke, Ve=Ve: F.scaled_dot_product_attention(q1, Ke, Ve, enable_gqa=True)
            res[f"evict_sdpa@{keep:g}"] = (_time_callable(ev, cfg), tensor_bytes([Ke, Ve]))
            m = QuasarMethod(cfg, keep, f"quasar@{keep:g}", fake)
            m.begin()
            m.freeze_at, m.capture_extra = H, 1
            al = m.compress_layer(0, QRows(Qc, keep_q, H), Kc, Vc, scaling, H)
            al["packed"] = m.pack(al)
            al.pop("sides")
            m.alloc[0] = al
            st = m.native_state(0, Kc, Vc, 1, dtype)
            qa = torch.randn(1, Hkv * rep, 1, d, generator=g).to(DEVICE, dtype)
            qp = torch.tensor([H], device=DEVICE)
            fn = lambda st=st, qa=qa, qp=qp: st.attend(qa, scaling, qp)
            res[f"quasar_torch@{keep:g}"] = (_time_callable(fn, cfg), st.context_bytes())
            if cfg.QUASAR_KERNEL and TRITON_OK and DEVICE.type == "cuda":
                chk = st.check_kernel()
                if chk["kernel"]:
                    res[f"quasar_kernel@{keep:g}"] = (_time_callable(fn, cfg), st.context_bytes())
                else:
                    logger.warning(f"  microbench: QUASAR kernel rejected (rel. err {chk['kernel_rel_err']:.3g})")
        base = res["full_sdpa"][0]
        for name, (us, nbytes) in res.items():
            rows.append({"context": H, "method": name, "us_per_step": us, "cache_bytes": nbytes,
                         "speedup_vs_full": base / us if us > 0 else float("nan"),
                         "device": str(DEVICE), "dtype": str(dtype),
                         "timer": "cuda_graph" if DEVICE.type == "cuda" else "eager_cpu"})
        logger.info("  microbench H=%d: " % H + ", ".join(f"{k_}={v_[0]:.1f}us/{v_[1] / 2**20:.2f}MiB"
                                                          for k_, v_ in res.items()))
    return pd.DataFrame(rows)


def _time_callable(fn, cfg: "Config") -> float:
    """Median microseconds of fn(); CUDA: captured once in a CUDA graph, then replayed (falls back
    to eager timing if capture fails, recorded in the log)."""
    reps = cfg.MICRO_REPEATS
    if DEVICE.type == "cuda":
        try:
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    fn()
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            ts = []
            for _ in range(reps):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                graph.replay()
                b.record()
                torch.cuda.synchronize()
                ts.append(a.elapsed_time(b) * 1000.0)
            return float(np.median(ts))
        except Exception as e:                                   # noqa: BLE001 (recorded, timed eagerly)
            logger.warning(f"  CUDA graph capture failed ({e!r}); timing eagerly")
    for _ in range(3):
        fn()
    ts = []
    for _ in range(reps):
        _sync()
        t = time.perf_counter()
        fn()
        _sync()
        ts.append((time.perf_counter() - t) * 1e6)
    return float(np.median(ts))


# ── Benchmark evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_prefix(runner: "Runner", method: LayerMethod, ids: List[int], cfg: "Config",
                ref_logp: Optional[torch.Tensor] = None, pre: Optional[Dict] = None) -> Tuple[Dict, torch.Tensor]:
    """
    PREFIX protocol, identical for every harness method: the first H = L - LONG_PPL_EVAL
    tokens are the context, compressed once at the horizon H (as before a question); tokens
    H+1 .. L-1 are predicted by queries at positions >= H, which read the compressed prefix
    plus the exact tokens since H.  Returns the NLL, the stored fraction of the prefix cache
    (from the captured state of the same forward) and, given the full cache's log-probs
    ref_logp, the mean KL(full || method) per scored token: FIDELITY to the uncompressed
    model, which a model's own length-extrapolation quirks cannot flatter.  With a shared
    prefill pre of the first H tokens, a one-shot method compresses it and reads the scored
    tokens from its state (same computation, one context pass for all methods).
    """
    L = len(ids)
    H = L - cfg.LONG_PPL_EVAL
    x = torch.tensor([ids], device=DEVICE)
    pos = torch.arange(L, device=DEVICE)
    if pre is not None and method.one_shot:
        compress_prefill(method, pre, cfg.LONG_PPL_EVAL)
        h = run_tokens(runner, method.states, ids[H:L - 1], H, hidden=True)
    else:
        method.freeze_at, method.capture_extra = H, 0
        sim = Sim(method)
        sim.begin(pos)
        with routed(sim):
            h = runner.model.get_decoder()(input_ids=x, position_ids=pos[None], use_cache=False).last_hidden_state[0]
        h = h[H:L - 1]
    logp = runner.model.get_output_embeddings()(h).float().log_softmax(-1)
    nll = -logp.gather(-1, x[0, H + 1:L, None])[:, 0]
    mem = method.state_bits(runner.d, H)["total"] / (H * runner.L * runner.Hkv * 2 * runner.d * 16.0)
    method.states, method.capture_extra = {}, None
    out = {"nll": float(nll.sum()), "n_tok": int(nll.numel()), "memory_fraction": mem,
           **(method.diagnostics() if isinstance(method, QuasarMethod) else {})}
    if ref_logp is not None:
        out["kl"] = float((ref_logp.exp() * (ref_logp - logp)).sum(-1).mean())
    return out, logp

def run_benchmarks(runner: "Runner", cfg: "Config", out_dir: str, suites: Sequence[str], synthetic=None) -> pd.DataFrame:
    """
    Runs the selected long-context suites: one row per (sample, method) with the score, the
    measured memory and the systems fields, flushed after every sample (resume skips
    finished pairs).  Predictions go to bench_preds.jsonl (last record per pair wins) and the
    official-format files pred/<benchmark>/<method>/<task>.jsonl are rewritten from it at
    the end, so an interrupted and resumed run never duplicates a prediction.
    """
    os.makedirs(out_dir, exist_ok=True)
    specs = bench_specs(cfg)
    stop = stop_ids_for(runner)
    rows_path, preds_path = os.path.join(out_dir, "bench_samples.csv"), os.path.join(out_dir, "bench_preds.jsonl")
    rows = []
    if cfg.RESUME and os.path.exists(rows_path):
        prev = pd.read_csv(rows_path, dtype={"benchmark": str, "task": str, "item": str, "method": str,
                                             "family": str, "note": str, "path": str})
        prev["note"] = prev["note"].fillna("")
        rows = prev.to_dict("records")
        logger.info(f"  resume: {len(rows)} rows from {rows_path}")
    done = {(r["benchmark"], r["task"], str(r["item"]), r["method"]) for r in rows}

    def flush():                                         # atomic: an interrupted write never corrupts the file
        tmp = rows_path + ".tmp"
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, rows_path)

    def run_items(bench, task, items, scorer, max_new_of, chat_ok, max_prompt):
        sub = items if not cfg.MAX_SAMPLES else [items[i] for i in sorted(
            _rng("sub", bench, task).choice(len(items), min(len(items), cfg.MAX_SAMPLES), replace=False))]
        for it in sub:
            max_new = max_new_of(it)
            t_tok = time.perf_counter()
            enc = encode_item(runner, it, max_new, cfg, chat_ok, max_prompt)
            tokenize_s = time.perf_counter() - t_tok
            first_line = task in LB_FIRST_LINE_TASKS and bench.startswith("longbench")

            def stop_fn(gen, _tok=runner.tokenizer):
                return first_line and "\n" in _tok.decode(gen, skip_special_tokens=True).lstrip("\n")
            pre = None                       # one exact context pass shared by the one-shot methods
            item_rows = []                   # rows of THIS item (for the log line)
            for spec in specs:
                key = (bench, task, str(it["id"]), spec.name)
                if key in done:
                    continue
                if pre is None and cfg.SHARED_PREFILL and spec.family in SHARED_FAMILIES:
                    pre = shared_prefill(runner, enc["ids"][:enc["context_len"]])
                gen, info, note = run_spec(runner, spec, enc, max_new, stop, stop_fn, cfg, pre)
                pred = runner.tokenizer.decode(gen, skip_special_tokens=True)
                score = float("nan") if note.startswith("kvpress error") else scorer(it, pred)
                _new_row = {"benchmark": bench, "task": task, "item": str(it["id"]), "method": spec.name,
                             "family": spec.label,
                             "keep_budget": spec.keep, "score": score, "note": note, "length": it.get("length"),
                             "difficulty": it.get("difficulty"), "length_class": it.get("length_class"),
                             "truncated": enc["truncated"], "chat": enc["chat"], "tokenize_s": tokenize_s, **info}
                rows.append(_new_row)
                item_rows.append(_new_row)
                with open(preds_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": list(key), "pred": pred, "answers": it["answers"],
                                        "all_classes": it.get("all_classes"), "length": it.get("length")},
                                       ensure_ascii=False) + "\n")
                done.add(key)
            pre = None
            free_memory()
            flush()
            logger.info(f"    {bench}/{task} item {it['id']}: P={len(enc['ids'])} " + ", ".join(
                f"{r['method']}={r['score']:.0f}" for r in item_rows))

    if "longbench" in suites:
        bench = "longbench-e" if cfg.LONGBENCH_E else "longbench"
        for task in cfg.LONGBENCH_TASKS:
            items = synthetic["longbench"][task] if synthetic else load_longbench(task, cfg)
            logger.info(f"  {bench}/{task}: {len(items)} items")
            run_items(bench, task, items,
                      lambda it, pred, _t=task: longbench_sample_score(_t, pred, it["answers"], it.get("all_classes")),
                      lambda it, _t=task: LB_MAX_NEW_TOKENS[_t], task not in LB_NO_CHAT_TASKS, cfg.LB_MAX_PROMPT_TOKENS)
    if "longbench-v2" in suites:
        items = synthetic["longbench-v2"] if synthetic else load_longbench_v2(cfg)
        logger.info(f"  longbench-v2: {len(items)} items")
        run_items("longbench-v2", "all", items, lambda it, pred: lbv2_sample_score(pred, it["answers"][0]),
                  lambda it: cfg.LBV2_MAX_NEW_TOKENS, True, cfg.LBV2_MAX_PROMPT_TOKENS)
    if "ruler" in suites:
        for length in cfg.RULER_LENGTHS:
            if length > runner.max_pos:
                logger.warning(f"  RULER {length} exceeds the model context ({runner.max_pos}); skipped")
                continue
            items = synthetic["ruler"] if synthetic else load_ruler(length, cfg)
            for task in sorted({it["task"] for it in items}):
                run_items(f"ruler-{length}", task, [it for it in items if it["task"] == task],
                          lambda it, pred: ruler_sample_score(it["task"], pred, it["answers"]),
                          lambda it: it["max_new_tokens"], True, None)
    if "longppl" in suites:
        specs_ppl = bench_specs(cfg, for_ppl=True)
        for name in cfg.LONG_PPL_DATASETS:
            bench = f"longppl-{name}"
            seqs = synthetic["longppl"] if synthetic else load_long_ppl(name, runner, cfg)
            for si, ids in enumerate(seqs):
                H = len(ids) - cfg.LONG_PPL_EVAL                   # compressed prefix (the "context")
                todo = [s_ for s_ in specs_ppl if (bench, "all", str(si), s_.name) not in done]
                if not todo:
                    continue
                full_spec = next(s_ for s_ in specs_ppl if s_.name == "full")
                pre = shared_prefill(runner, ids[:H]) if cfg.SHARED_PREFILL else None
                ref, ref_logp = eval_prefix(runner, make_method(full_spec, H, runner, cfg)[0], ids, cfg, pre=pre)
                for spec in todo:
                    t0 = time.time()
                    method, note = make_method(spec, H, runner, cfg)
                    r = ref if spec.name == "full" else eval_prefix(runner, method, ids, cfg, ref_logp, pre)[0]
                    base = {"task": "all", "item": str(si), "method": spec.name, "family": spec.label,
                            "keep_budget": spec.keep, "note": note, "memory_fraction": r["memory_fraction"],
                            "context_len": H, "path": "harness", "harness_seconds": time.time() - t0,
                            **{k_: v_ for k_, v_ in r.items() if k_.startswith("quasar_")}}
                    rows.append({**base, "benchmark": bench, "score": -r["nll"] / r["n_tok"], "nll": r["nll"],
                                 "n_tok": r["n_tok"]})
                    rows.append({**base, "benchmark": f"{bench}-kl", "score": -r.get("kl", 0.0),
                                 "kl": r.get("kl", 0.0)})
                    done.add((bench, "all", str(si), spec.name))
                del ref_logp, pre
                free_memory()
                flush()
            g = pd.DataFrame([r_ for r_ in rows if r_["benchmark"] in (bench, f"{bench}-kl")])
            for m_, gm in (g.groupby("method") if len(g) else []):
                p_, k_ = gm[gm.benchmark == bench], gm[gm.benchmark == f"{bench}-kl"]
                logger.info(f"    {bench} {m_:<20} ppl={math.exp(-p_.score.mean()):.4f} "
                            f"KL(full||m)={-k_.score.mean():.4f} memory={p_.memory_fraction.mean():.3f}")
    flush()
    write_predictions(preds_path, done, out_dir)
    return pd.DataFrame(rows)


def write_predictions(preds_path: str, done: set, out_dir: str):
    """Official LongBench eval.py input format, one file per (benchmark, method, task)."""
    if not os.path.exists(preds_path):
        return
    latest = {}
    with open(preds_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            latest[tuple(rec.pop("key"))] = rec
    files = defaultdict(list)
    for (bench, task, item, method), rec in latest.items():
        if (bench, task, item, method) in done:
            files[(bench, method, task)].append(rec)
    for (bench, method, task), recs in files.items():
        path = os.path.join(out_dir, "pred", bench, method)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, f"{task}.jsonl"), "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)


def bench_contrasts(cfg: "Config") -> List[Tuple[str, str, str]]:
    """Pre-registered contrasts (A better than B is the hypothesis; harness checks are two-sided)."""
    out = []
    for r in cfg.KEEP_FRACTIONS:
        t = f"{r:g}"
        out += [(f"quasar_vs_{b_}@{t}", f"quasar@{t}", f"{b_}@{t}")
                for b_ in [f"snapkv_q{b}" for b in cfg.MTLP_BITS] + ["evict", "am", "snapkv", "streamingllm"]]
        out += [(f"quasar_vs_kvpress:{pn}@{t}", f"quasar@{t}", f"kvpress:{pn}@{t}") for pn in cfg.KVPRESS_PRESSES]
        out.append((f"{HARNESS_CHECK}snapkv@{t}", f"snapkv@{t}", f"kvpress:snapkv@{t}"))
    t0 = f"{cfg.PRIMARY_KEEP:g}"
    out += [(f"quasar_vs_{b_}@{t0}", f"quasar@{t0}", f"{b_}@{t0}") for b_ in ("quasar_euclid", "quasar_rot", "quasar_window")]
    out += [(f"quasar@{t0}_vs_kivi{b}", f"quasar@{t0}", f"kivi{b}") for b in cfg.KIVI_BITS]
    out.append((f"{HARNESS_CHECK}quasar_native@{t0}", f"quasar_native@{t0}", f"quasar@{t0}"))
    out.append((f"{HARNESS_CHECK}full", "full", "native_full"))
    return out

HARNESS_CHECK = "harness_check:"    # two-sided implementation checks; never enter decision rules


def paired_score_test(d: np.ndarray, strata: np.ndarray, n_boot: int, n_perm: int, seed: int) -> Dict:
    """
    Higher is better.  d = per-unit paired differences A - B; strata = task of each unit.
    The estimand is the benchmark's official aggregate: the MACRO mean over tasks of the
    per-task mean difference (LongBench / RULER average task scores; LongBench-v2 and long
    PPL have one stratum, i.e. the plain mean over items, which for long PPL is also the
    token-weighted mean because every sequence has exactly LONG_PPL_LEN tokens).  CI:
    bootstrap that resamples units within each task; p: two-sided paired sign-flip test of
    the same statistic (exact under H0 of exchangeable signs within units).
    """
    nan = float("nan")
    if len(d) == 0:
        return {"estimate": nan, "ci_low": nan, "ci_high": nan, "p_value": nan, "n_units": 0, "n_tasks": 0}
    tasks = np.unique(strata)
    idx = [np.flatnonzero(strata == t) for t in tasks]
    w = np.zeros(len(d))
    for ix in idx:                                       # macro weights: 1 / (n_tasks * n_task)
        w[ix] = 1.0 / (len(tasks) * len(ix))
    est = float((w * d).sum())
    rng = np.random.default_rng(seed)
    boot = np.zeros(n_boot)
    for ix in idx:
        boot += d[ix][rng.integers(0, len(ix), size=(n_boot, len(ix)))].mean(1) / len(tasks)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, len(d))) * (w * d)).sum(1)
    p = (1 + np.sum(np.abs(null) >= abs(est) - 1e-12)) / (n_perm + 1)
    return {"estimate": est, "ci_low": float(np.percentile(boot, 2.5)), "ci_high": float(np.percentile(boot, 97.5)),
            "p_value": float(p), "n_units": len(d), "n_tasks": len(tasks)}


LB_E_BUCKETS = ((0, 4000, "0-4k"), (4000, 8000, "4-8k"), (8000, BIG, "8k+"))   # official eval.py scorer_e
SYSTEMS_COLS = ("tokenize_s", "prefill_s", "compress_s", "ttft_s", "decode_s", "tokens_per_s", "peak_prefill_bytes",
                "peak_decode_bytes", "harness_prefill_s", "harness_decode_s", "harness_tokens_per_s",
                "harness_peak_prefill_bytes", "harness_peak_decode_bytes", "payload_bytes", "analytic_bytes",
                "payload_vs_analytic", "sim_state_bytes", "cache_bytes_measured", "full_cache_bytes",
                "attn_flops_first_step", "attn_flops_first_step_full", "flops_first_step_measured",
                "memory_fraction", "memory_fraction_measured")


def bench_analysis(df: pd.DataFrame, cfg: "Config", model: str, family: str, out_dir: str) -> pd.DataFrame:
    """
    Task scores (official aggregation: per-task mean, macro average over tasks; LongBench-E
    also per length bucket; LongBench-v2 also per difficulty and length class), systems
    measurements, and pre-specified paired contrasts.
    Memory: a contrast uses only BUDGET-COMPLIANT units: at a shared budget both methods store
    <= KEEP (1 + BUDGET_TOL) on that sample; against an unbudgeted code (KIVI) memory_A <=
    memory_B (1 + BUDGET_TOL).  Excluded units are counted; the all-unit estimate is reported.
    Numerical path: harness methods run attention in fp32, native ones (official kvpress
    presses) in the model dtype.  A cross-path contrast is therefore a difference-in-
    differences on each unit, (A - B) - s (full - native_full) with s = +1 when A is the
    harness method, which removes the additive effect of the path itself; without both
    full-cache references it is reported as uncontrolled and never enters a decision.
    """
    if df.empty:
        return df
    df = df[df.score.notna()].copy()
    df["note"] = df.note.fillna("").astype(str) if "note" in df else ""
    agg = df.groupby(["benchmark", "task", "method"], as_index=False).agg(
        score=("score", "mean"), n=("score", "size"), memory_fraction=("memory_fraction", "mean"),
        flagged=("note", lambda s_: float((s_.str.len() > 0).mean())))
    parts = [agg]
    for bench, g in agg[~agg.benchmark.str.startswith("longppl")].groupby("benchmark"):
        n_tasks = g.task.nunique()
        mac = g.groupby("method", as_index=False).agg(score=("score", "mean"), n=("n", "sum"),
                                                      memory_fraction=("memory_fraction", "mean"),
                                                      flagged=("flagged", "mean"), n_t=("task", "nunique"))
        mac = mac[mac.n_t == n_tasks].drop(columns="n_t")      # macro only over complete task sets
        parts.append(mac.assign(benchmark=bench, task="macro_avg"))
    if "length" in df and df.benchmark.str.startswith("longbench-e").any():
        e = df[df.benchmark.str.startswith("longbench-e") & df.length.notna()]
        for lo, hi, lab in LB_E_BUCKETS:
            b = e[(e.length >= lo) & (e.length < hi)]
            if len(b):
                parts.append(b.groupby(["benchmark", "task", "method"], as_index=False).agg(
                    score=("score", "mean"), n=("score", "size"), memory_fraction=("memory_fraction", "mean"))
                    .assign(task=lambda x, _l=lab: x.task + f"[{_l}]"))
    v2 = df[df.benchmark == "longbench-v2"]
    for col in ("difficulty", "length_class"):                  # official result.py breakdowns
        if col in v2 and v2[col].notna().any():
            parts.append(v2.groupby(["benchmark", col, "method"], as_index=False).agg(
                score=("score", "mean"), n=("score", "size"), memory_fraction=("memory_fraction", "mean"))
                .assign(task=lambda x, _c=col: "all[" + x[_c].astype(str) + "]").drop(columns=col))
    agg = pd.concat(parts, ignore_index=True)
    agg.insert(0, "model", model)
    agg.to_csv(os.path.join(out_dir, "bench_task_scores.csv"), index=False)

    sys_cols = [c_ for c_ in SYSTEMS_COLS if c_ in df]
    gen = df[~df.benchmark.str.startswith("longppl")]
    if sys_cols and len(gen):
        sysd = gen.groupby(["benchmark", "method"], as_index=False)[sys_cols].median(numeric_only=True)
        sysd = sysd.merge(gen.groupby(["benchmark", "method"], as_index=False).path.first(), on=["benchmark", "method"])
        sysd["speed_comparable"] = sysd.path == "native"        # harness timings are not deployment speed
        sysd.insert(0, "model", model)
        sysd["statistic"] = "median over samples"
        sysd.to_csv(os.path.join(out_dir, "bench_systems.csv"), index=False)

    rows = []
    for bench, g in df.groupby("benchmark"):
        g = g[g.task == "all"] if bench.startswith("longppl") else g
        g = g.assign(unit=g.task.astype(str) + "/" + g["item"].astype(str))
        path_of = g.groupby("method").path.first().to_dict()
        gap = None                                              # per-unit path effect at the full cache
        if {"full", "native_full"} <= set(g.method):
            gap = g[g.method == "full"][["unit", "score"]].merge(
                g[g.method == "native_full"][["unit", "score"]], on="unit", suffixes=("_h", "_n"))
            gap = gap.assign(path_gap=gap.score_h - gap.score_n)[["unit", "path_gap"]]
        for name, A, B in bench_contrasts(cfg):
            ga, gb = g[g.method == A], g[g.method == B]
            if ga.empty or gb.empty:
                continue
            m = ga[["unit", "task", "score", "memory_fraction", "keep_budget"]].merge(
                gb[["unit", "score", "memory_fraction", "keep_budget"]], on="unit", suffixes=("_a", "_b"))
            check = name.startswith(HARNESS_CHECK)
            cross = path_of[A] != path_of[B] and not check
            role = "check" if check else ("uncontrolled_cross_path" if cross and gap is None else "hypothesis")
            if cross and gap is not None:
                m = m.merge(gap, on="unit")
            tol = 1 + cfg.BUDGET_TOL
            if check:
                ok = np.ones(len(m), bool)
            elif m.keep_budget_a.notna().all() and m.keep_budget_b.notna().all():   # same budget: both within it
                ok = ((m.memory_fraction_a <= m.keep_budget_a * tol + 1e-9) &
                      (m.memory_fraction_b <= m.keep_budget_b * tol + 1e-9)).to_numpy()
            else:                                       # vs an unbudgeted code (KIVI): A no larger than B
                ok = (m.memory_fraction_a <= m.memory_fraction_b * tol + 1e-9).to_numpy()
            d = (m.score_a - m.score_b).to_numpy()
            d_raw = d
            if cross and gap is not None:
                d = d - (1.0 if path_of[A] == "harness" else -1.0) * m.path_gap.to_numpy()
            strata = m.task.astype(str).to_numpy()
            r = paired_score_test(d[ok], strata[ok], cfg.N_BOOT, cfg.N_PERM, SEED)
            rows.append({"model": model, "model_family": family, "benchmark": bench, "contrast": name, "A": A,
                         "B": B, "role": role, "cross_path": cross, "path_adjusted": cross and gap is not None,
                         "memory_A": float(m.memory_fraction_a[ok].mean()) if ok.any() else float("nan"),
                         "memory_B": float(m.memory_fraction_b[ok].mean()) if ok.any() else float("nan"),
                         "score_A": float(m.score_a[ok].mean()) if ok.any() else float("nan"),
                         "score_B": float(m.score_b[ok].mean()) if ok.any() else float("nan"),
                         "estimate_unadjusted": paired_score_test(d_raw[ok], strata[ok], 1, 1, SEED)["estimate"],
                         "n_excluded_over_budget": int((~ok).sum()),
                         "estimate_all_units": paired_score_test(d, strata, 1, 1, SEED)["estimate"], **r})
    ct = pd.DataFrame(rows)
    if len(ct):
        ct["p_holm"] = np.nan
        hyp = ct.role == "hypothesis"
        for bench, idx in ct[hyp].groupby("benchmark").groups.items():
            ct.loc[idx, "p_holm"] = holm(ct.loc[idx, "p_value"])
        ct["supported"] = hyp & (ct.estimate > 0) & (ct.p_holm < cfg.ALPHA) & (ct.n_units > 0)
        ct.to_csv(os.path.join(out_dir, "bench_contrasts.csv"), index=False)
        for r in ct[ct.role != "hypothesis"].itertuples():
            logger.info(f"  {r.benchmark} {r.contrast} [{r.role}]: diff={r.estimate:+.3f} p={r.p_value:.3g}")
    return ct


def bench_cross_model(cfg: "Config"):
    cts = [pd.read_csv(p) for p in Path(cfg.RESULTS_DIR).glob("*/bench/bench_contrasts.csv")]
    if not cts:
        return
    ct = pd.concat(cts, ignore_index=True)
    ct.to_csv(os.path.join(cfg.RESULTS_DIR, "bench_all_contrasts.csv"), index=False)
    if "role" in ct:                                        # harness checks never enter decisions
        ct = ct[ct.role == "hypothesis"]
    sysd = [pd.read_csv(p) for p in Path(cfg.RESULTS_DIR).glob("*/bench/bench_systems.csv")]
    if sysd:
        pd.concat(sysd, ignore_index=True).to_csv(os.path.join(cfg.RESULTS_DIR, "bench_all_systems.csv"), index=False)
    scores = [pd.read_csv(p) for p in Path(cfg.RESULTS_DIR).glob("*/bench/bench_task_scores.csv")]
    if scores:
        pd.concat(scores, ignore_index=True).to_csv(os.path.join(cfg.RESULTS_DIR, "bench_all_task_scores.csv"), index=False)
    rows = []
    for (bench, name), g in ct.groupby(["benchmark", "contrast"], sort=False):
        n = len(g)
        need = int(math.ceil(cfg.DECISION_MIN_MODEL_FRACTION * n))
        pw = float(stats.wilcoxon(g.estimate, alternative="greater").pvalue) if n >= cfg.MIN_MODELS_WILCOXON \
            else float("nan")
        met = int(g.supported.sum())
        verdict = ("insufficient models" if n < cfg.MIN_MODELS_WILCOXON else
                   "supported" if (met >= need and pw < cfg.ALPHA) else "not supported")
        rows.append({"benchmark": bench, "contrast": name, "models_evaluated": n, "models_meeting_rule": met,
                     "models_required": need, "median_estimate": float(g.estimate.median()),
                     "p_wilcoxon_models": pw, "verdict": verdict})
    dr = pd.DataFrame(rows)
    dr.to_csv(os.path.join(cfg.RESULTS_DIR, "bench_decision_rules.csv"), index=False)
    for r in dr.itertuples():
        logger.info(f"BENCH DECISION {r.benchmark:<14} {r.contrast:<28} {r.verdict:<20} "
                    f"{r.models_meeting_rule}/{r.models_evaluated} median={r.median_estimate:+.3f}")
    official = {c_ for c_, _, _ in bench_contrasts(cfg) if ":" in c_ and not c_.startswith(HARNESS_CHECK)}
    missing = sorted(official - set(ct.contrast))
    if missing:                    # never silent: a paper table without these rows must say so
        pd.DataFrame({"contrast": missing}).to_csv(os.path.join(cfg.RESULTS_DIR, "bench_missing_contrasts.csv"),
                                                   index=False)
        logger.warning(f"{len(missing)} official-baseline contrasts have no data (kvpress missing or failed): "
                       f"{missing[:4]}{' ...' if len(missing) > 4 else ''}")


def run_bench_model(mc: "ModelConfig", cfg: "Config", suites: Sequence[str], prebuilt=None):
    out_dir = os.path.join(cfg.RESULTS_DIR, mc.name, "bench")
    if prebuilt is None:
        runner = Runner(mc, cfg)
        synthetic = None
    else:
        model, tokenizer, synthetic = prebuilt
        runner = Runner(mc, cfg, model=model, tokenizer=tokenizer)
    os.makedirs(out_dir, exist_ok=True)
    probe = synthetic["longbench"][cfg.LONGBENCH_TASKS[0]][0] if synthetic else {
        "context": " ".join(["The grass is green. The sky is blue."] * 120) + " The code is 4711.",
        "question": " What is the code?", "answer_prefix": " Answer:", "answers": ["4711"]}
    enc = encode_item(runner, probe, 8, cfg)
    with open(os.path.join(out_dir, "harness_tests.json"), "w") as f:
        json.dump(harness_tests(runner, enc["ids"]), f, indent=2)
    t0 = f"{cfg.PRIMARY_KEEP:g}"
    eq_specs = [s for s in bench_specs(cfg) if s.name in
                ("full", "kivi2", f"snapkv@{t0}", f"streamingllm@{t0}", f"evict@{t0}", f"am@{t0}", f"quasar@{t0}",
                 f"quasar_euclid@{t0}", f"quasar_rot@{t0}", f"quasar_window@{t0}", f"snapkv_q4@{t0}",
                 f"quasar_native@{t0}", "native_full", f"kvpress:snapkv@{t0}", f"kvpress:streaming_llm@{t0}")]
    generation_equivalence(runner, eq_specs, enc["ids"], enc["ctx_len"], 8, cfg).to_csv(
        os.path.join(out_dir, "generation_equivalence.csv"), index=False)
    attention_microbench(cfg, runner.d, runner.Hkv, runner.Hq // runner.Hkv).assign(model=mc.name).to_csv(
        os.path.join(out_dir, "microbench.csv"), index=False)
    if synthetic:                      # positive control: gold answers must score 100 end to end
        for task, items in synthetic["longbench"].items():
            for it in items:
                sc = longbench_sample_score(task, it["answers"][0], it["answers"], it.get("all_classes"))
                if abs(sc - 100.0) > 1e-9:
                    raise RuntimeError(f"gold answer scored {sc} on {task}")
        for it in synthetic["ruler"]:
            if ruler_sample_score(it["task"], " ".join(it["answers"]), it["answers"]) != 100.0:
                raise RuntimeError("gold answer did not score 100 on RULER")
        for it in synthetic["longbench-v2"]:
            if lbv2_sample_score(f"The correct answer is ({it['answers'][0]})", it["answers"][0]) != 100.0:
                raise RuntimeError("gold answer did not score 100 on LongBench-v2")
        logger.info("  positive control passed: gold answers score 100 on every synthetic task")
    df = run_benchmarks(runner, cfg, out_dir, suites, synthetic)
    bench_analysis(df, cfg, mc.name, mc.family, out_dir)
    if prebuilt is None:
        runner.release()


def smoke_bench_setup(cfg: "Config"):
    """Tiny character-level model + synthetic LongBench / RULER / long-PPL stand-ins (offline)."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import LlamaConfig, PreTrainedTokenizerFast
    chars = list(" abcdefghijklmnopqrstuvwxyz0123456789:?.=,;\n#")
    vocab = {"<s>": 0, "</s>": 1, "<pad>": 2, **{ch: i + 3 for i, ch in enumerate(chars)}}
    tk = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<pad>"))
    tk.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tk.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    g = np.random.default_rng(SEED)
    letters = "abcdefghijklmnopqrstuvwxyz"

    def kv_doc(n_pairs):
        keys = ["".join(g.choice(list(letters), 3)) for _ in range(n_pairs)]
        vals = ["".join(g.choice(list("0123456789"), 2)) for _ in range(n_pairs)]
        return keys, vals, " ".join(f"{k}={v};" for k, v in zip(keys, vals))

    def text_sample(n_pairs):
        keys, vals, doc = kv_doc(n_pairs)
        j = int(g.integers(n_pairs))
        return f"{doc} q:{keys[j]}? a:{vals[j]}\n"

    conf = LlamaConfig(vocab_size=len(vocab), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                       num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                       bos_token_id=0, eos_token_id=1, pad_token_id=2, tie_word_embeddings=True)
    torch.manual_seed(SEED)
    model = AutoModelForCausalLM.from_config(conf, attn_implementation=ROUTER_ATTN)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(300):
        batch = [tok(text_sample(20)).input_ids[:256] for _ in range(16)]
        L = min(len(b) for b in batch)
        x = torch.tensor([b[:L] for b in batch])
        loss = model(input_ids=x, labels=x).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    logger.info(f"bench smoke: trained char-level model, final loss {loss.item():.3f}")

    def lb_items(task, n):
        out = []
        for i in range(n):
            keys, vals, doc = kv_doc(40)
            j = int(g.integers(40))
            if task == "trec":
                out.append({"id": str(i), "context": doc, "question": f" q:{keys[j]}?", "answer_prefix": " a:",
                            "answers": [vals[j]], "all_classes": sorted(set(vals)), "length": len(doc), "task": task})
            else:
                out.append({"id": str(i), "context": doc, "question": f" q:{keys[j]}?", "answer_prefix": " a:",
                            "answers": [vals[j]], "all_classes": None, "length": len(doc), "task": task})
        return out
    synthetic = {"longbench": {t: lb_items(t, 3) for t in cfg.LONGBENCH_TASKS},
                 "ruler": [{**it, "task": "niah_single_1", "max_new_tokens": 4, "id": f"r{i}"}
                           for i, it in enumerate(lb_items("qasper", 3))],
                 "longbench-v2": [{**it, "task": "all", "id": f"v{i}", "answers": ["ABCD"[i % 4]],
                                   "difficulty": ("easy", "hard")[i % 2], "length_class": "short"}
                                  for i, it in enumerate(lb_items("qasper", 4))],
                 "longppl": [tok(text_sample(60)).input_ids[:600] for _ in range(3)]}
    return model, tok, synthetic


def main():
    ap = argparse.ArgumentParser(description="QUASAR: embedded spectral rate allocation of the KV cache")
    ap.add_argument("--suite", nargs="+", default=["longbench"], choices=["longbench", "longbench-v2", "ruler", "longppl"],
                    help="long-context benchmarks with generation from the compressed cache (longppl: prefix protocol)")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None, help="LongBench tasks (default: Config.LONGBENCH_TASKS)")
    ap.add_argument("--longbench-e", action="store_true")
    ap.add_argument("--lbv2-max-prompt", type=int, default=None, help="LongBench-v2 prompt budget (official 120000)")
    ap.add_argument("--ruler-lengths", nargs="*", type=int, default=None)
    ap.add_argument("--keep", nargs="*", type=float, default=None, help="memory budgets (fractions of fp16 cache)")
    ap.add_argument("--max-samples", type=int, default=None, help="per task")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="benchmark method name prefixes, e.g. quasar evict snapkv@ (full and native_full always run)")
    ap.add_argument("--query-aware", action="store_true", help="compress the question with the context")
    ap.add_argument("--long-ppl-len", type=int, default=None)
    ap.add_argument("--check-data", action="store_true",
                    help="load the first row of every dataset the selected suites use, check it, and exit")
    ap.add_argument("--allow-missing-kvpress", action="store_true",
                    help="run benchmarks without the official kvpress baselines (recorded as missing)")
    ap.add_argument("--smoke", action="store_true", help="offline CPU run: self-tests + tiny model + synthetic data")
    ap.add_argument("--microbench", nargs="*", type=int, default=None, metavar="LEN",
                    help="only the decode-attention microbenchmark (contexts LEN ..., default Config.MICRO_LENGTHS) "
                         "for --micro-shape head_dim kv_heads query_heads_per_kv, then exit")
    ap.add_argument("--micro-shape", nargs=3, type=int, default=(128, 8, 4))
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    kw = {"MAX_SAMPLES": a.max_samples, "QUERY_AWARE": a.query_aware, "LONGBENCH_E": a.longbench_e,
          "RESUME": not a.no_resume, "ALLOW_MISSING_KVPRESS": a.allow_missing_kvpress}
    if a.output:
        kw["OUTPUT_DIR"] = a.output
    if a.tasks:
        kw["LONGBENCH_TASKS"] = tuple(a.tasks)
    if a.lbv2_max_prompt:
        kw["LBV2_MAX_PROMPT_TOKENS"] = a.lbv2_max_prompt
    if a.ruler_lengths:
        kw["RULER_LENGTHS"] = tuple(a.ruler_lengths)
    if a.keep:
        kw["KEEP_FRACTIONS"] = tuple(a.keep)
        kw["PRIMARY_KEEP"] = sorted(a.keep)[len(a.keep) // 2]
    if a.long_ppl_len:
        kw["LONG_PPL_LEN"] = a.long_ppl_len
    if a.methods:
        kw["BENCH_METHODS"] = tuple(a.methods)
    if a.smoke:
        kw.update(OUTPUT_DIR=a.output or str(PROJECT_ROOT / "quasar_smoke"), SMOKE=True, CHUNK=32, WINDOW=8,
                  W_OBS=8, KIVI_GROUP=8, N_BOOT=500, N_PERM=2000, MIN_MODELS_WILCOXON=2, RESUME=False,
                  AM_ITERS=100, KEEP_FRACTIONS=(0.5, 0.25), PRIMARY_KEEP=0.25, LB_MAX_PROMPT_TOKENS=1500,
                  LBV2_MAX_PROMPT_TOKENS=1500, RULER_LENGTHS=(1024,), LONG_PPL_DATASETS=("synthetic",),
                  LONG_PPL_LEN=600, LONG_PPL_EVAL=120, QUASAR_SEGMENT=64, QUASAR_QUERIES=8, QUASAR_LOCAL=16,
                  MICRO_LENGTHS=(512, 2048), MICRO_REPEATS=5, ALLOW_MISSING_KVPRESS=True)
    if a.microbench is not None and a.microbench:
        kw["MICRO_LENGTHS"] = tuple(a.microbench)
    cfg = Config(**kw)
    if a.check_data:
        problems = preflight_data(cfg, a.suite)
        raise SystemExit(("data check FAILED:\n  " + "\n  ".join(problems)) if problems else 0)
    fh = logging.FileHandler(os.path.join(cfg.RESULTS_DIR, "quasar_kv.log"))
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    logger.info(f"run {cfg.RUN_ID}: results in {cfg.RESULTS_DIR}")
    with open(os.path.join(cfg.RESULTS_DIR, "config.json"), "w") as f:
        json.dump({"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   **{k: ([asdict(m) for m in v] if k == "BENCH_MODELS" else v)
                      for k, v in cfg.__dict__.items()}}, f, indent=2, default=str)
    with open(os.path.join(cfg.RESULTS_DIR, "quasar_tests.json"), "w") as f:
        json.dump(quasar_tests(cfg), f, indent=2)
    flop_formula_test()
    metric_tests()
    if a.microbench is not None:
        d_, hkv_, rep_ = a.micro_shape
        mb = attention_microbench(cfg, d_, hkv_, rep_)
        mb.to_csv(os.path.join(cfg.RESULTS_DIR, "microbench.csv"), index=False)
        logger.info(f"microbenchmark written to {os.path.join(cfg.RESULTS_DIR, 'microbench.csv')}")
        return
    if not cfg.SMOKE:
        problems = preflight_data(cfg, a.suite)             # before any model is loaded
        if problems:
            raise SystemExit("data check FAILED (fix the dataset ids / columns first):\n  " + "\n  ".join(problems))
    if cfg.KVPRESS_PRESSES and kvpress_module() is None and not cfg.ALLOW_MISSING_KVPRESS:
        raise SystemExit("kvpress (official SOTA baselines) cannot be imported: pip install kvpress==0.5.5 "
                         "(needs transformers < 5.3), or pass --allow-missing-kvpress to run without them")
    if cfg.SMOKE:
        run_bench_model(ModelConfig("tiny-char-llama", "random-init", "Llama", 1), cfg, a.suite,
                        smoke_bench_setup(cfg))
    else:
        catalog = {m.name: m for m in cfg.BENCH_MODELS}
        if a.models:
            missing = [m for m in a.models if m not in catalog]
            if missing:
                raise SystemExit(f"unknown model(s) {missing}; add them to Config.BENCH_MODELS")
        for mc in ([catalog[m] for m in a.models] if a.models else cfg.BENCH_MODELS):
            try:
                run_bench_model(mc, cfg, a.suite)
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"{mc.name}: out of memory in benchmarks ({e}); skipped")
                free_memory()
    bench_cross_model(cfg)


if __name__ == "__main__":
    main()
