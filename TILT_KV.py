"""
TILT_KV.py — TiltKV: exponential-family (cumulant) compaction of the KV cache.

A training-free, calibration-free KV-cache compressor whose core object is the
cumulant generating function (CGF) of a block of cached keys.  It replaces the old,
completed part of the cache by a few exact "tail" tokens plus second-order
exponential-family atoms whose attention MASS and VALUE both depend on the query.
Memory and decode-time attention FLOPs both shrink.  The evaluation protocol follows
KV_LDT_v12_2.py: exact negative controls, paired randomisation tests, Holm
correction and fixed cross-model decision rules.

════════════════════════════════════════════════════════════════════════════════════════
1. THE GAP
════════════════════════════════════════════════════════════════════════════════════════
For one KV head and a query q (scaled by 1/sqrt(d)), a set C of n cached tokens enters
attention ONLY through
        Z_C(q) = sum_{j in C} exp(q.k_j)          N_C(q) = sum_{j in C} exp(q.k_j) v_j,
because  o(q) = (Z_rest o_rest + N_C) / (Z_rest + Z_C).  Compressing C is therefore the
problem of representing the two functions q -> log Z_C(q) and q -> N_C(q)/Z_C(q) for
FUTURE queries, not of reconstructing the tokens.

Every published merging / compaction / residual method represents a compacted entry by a
key, a value and a query-INDEPENDENT log-mass:
  mean / weighted merging with a count or size bias (KeepKV, AAAI'26; SemantiCache;
  D2O; CaM; ToMe-style log-size), Attention Matching (Zweiger et al., ICML 2026: fitted
  scalar biases beta), MomentKV (2026: count, means and value-key covariance with a
  FIRST-order centred expansion), ResKV (2026: mean key + count), SelKV (2026: prefill
  bias).  Proposition 2 below shows that for compacted entries built on the block mean,
  this form is a first-order truncation whose error, the "attention sag", is the
  query-dependent CGF kappa_C(q) ~ 1/2 q^T Sigma q.  No constant bias can remove a
  query-dependent error.  Searches (Oct 2026) found no method using the second-order,
  query-dependent log-mass together with exponentially tilted, query-dependent values.

  Related use of the same mathematics, credited: "Quantized Keys Steal Attention"
  (2026) uses the moment generating function of QUANTISATION noise to correct the
  Jensen bias of quantised keys; Expected Attention (2025) uses a second-order term of
  the QUERY distribution to score tokens for eviction.  TiltKV applies the cumulant
  expansion to the empirical distribution of the CACHED keys/values of a block, and
  turns it into a compressed representation with a guarantee (Prop. 4).

════════════════════════════════════════════════════════════════════════════════════════
2. THEORY (proof sketches; Props 2-4 are checked numerically in self_tests)
════════════════════════════════════════════════════════════════════════════════════════
  Notation: P_C = empirical distribution of (k, v) over C; mu = E k, mu_v = E v,
  Sigma = Cov k, Sigma_vk = Cov(v, k) (population moments of the n tokens),
  kappa_C(q) = log E exp(q.(k - mu)), R_C = max_j ||k_j - mu||.

  Prop 1 (exact cumulant representation).
      log Z_C(q) = log n + q.mu + kappa_C(q),      N_C/Z_C = E_{P_C^q}[v],
  where P_C^q ∝ exp(q.k) P_C is the exponential tilt of P_C.  Expanding,
      kappa_C(q) = 1/2 q^T Sigma q + O(E|q.(k-mu)|^3),
      E_{P^q}[v] = mu_v + Sigma_vk q + O(third cumulants).
  Proof: definition of Z_C; the tilted mean of v is the p-gradient at p = 0 of the joint
  CGF log E exp(q.k + p.v), whose second-order Taylor term is q^T Sigma_vk^T p.

  Prop 2 (attention sag is query-dependent).  An entry with key mu and query-independent
  log-mass log n + beta has log-mass error  kappa_C(q) - beta.  kappa_C is convex with
  kappa_C(0) = 0 and kappa_C >= 0 (Jensen).  The error is constant in q only if
  kappa_C is, i.e. only if q.k is constant over C for every q in the query support.
  Hence beta = 0 (mean merge) always UNDER-weights merged mass, and any fitted beta
  trades errors across queries.

  Prop 3 (second-order atom).  The atom
      logit_C(q) = log n + q.mu + 1/2 q^T Sigma q,      value_C(q) = mu_v + Sigma_vk q
  is exact for jointly Gaussian P_C (all higher cumulants vanish).  In general its
  log-mass error is the third and higher cumulants along q.

  Prop 4 (feasible projection; guarantee).  For every q,
      0 <= kappa_C(q) <= max_j q.(k_j - mu) <= ||q|| R_C.
  The projection of the second-order estimate onto [0, ||q|| R_C] contains the truth, so
  (1-D non-expansiveness of projection onto a convex set) its error never exceeds the
  unprojected error, and it never assigns more mass than the block can carry.  On the
  projected branch the tilt is scaled by d kappa / d t (along q) divided by q^T Sigma q,
  as in the Gaussian case.

  Prop 4' (value projection).  The exact tilted mean N_C/Z_C is a convex combination of
  the v_j, so it lies in the ball B(mu_v, R_v), R_v = max_j ||v_j - mu_v||.  The linear
  tilt mu_v + Sigma_vk q is exact for Gaussian blocks but unbounded in ||q||; projecting
  it onto B(mu_v, R_v) can only reduce its error (projection onto a convex set that
  contains the truth is non-expansive).

  Prop 6 (freezing; after Derrida's Random Energy Model, 1981).  For n i.i.d. Gaussian
  logits of spread sigma, (1/n) sum exp(x_j) has log ~ 1/2 sigma^2 for sigma <= sigma_c =
  sqrt(2 ln n) ("high temperature") and ~ sigma sigma_c - ln n beyond ("frozen": the sum is
  carried by its maximum, which is ~ sigma sqrt(2 ln n)).  A FINITE block therefore obeys
  the cumulant expansion only in the hot phase; for sharp (retrieval / induction) queries
  the second-order term over-estimates the block's mass quadratically and steals attention
  from the correct token (observed in the --smoke run: +9.3 nats mean over-estimate,
  43 nats maximum, before this correction).  TiltKV uses the C^1 two-phase estimate with
  the matching tilt scale sigma_c / sigma, then the hard projection of Prop. 4.  The ratio
  sigma / sigma_c is also a per-(chunk, head, query) diagnostic of compressibility: frozen
  blocks need exact tail tokens; hot blocks are summarised accurately by moments.  REM has
  been used to analyse softmax attention; we found no use of it to compress the KV cache.

  Prop 5 (cost).  With Sigma stored as a rank-r query-weighted factor plus a diagonal,
  and Sigma_vk as a rank-r factor, one atom stores (3+3r) d + r + 3 numbers.  Reading it
  costs O((2+3r) d) per query head, versus 2 d per token for every one of its n tokens,
  so both memory and decode FLOPs fall once n > (3+3r)/2.

  Tail selection (heuristic, ablated).  Inside each block the tokens with the largest
  within-block attention share under recently OBSERVED queries are kept individually.
  These carry the largest positive deviations q.(k - mu), which dominate the third and
  higher cumulants of the remainder.  The low-rank directions of Sigma and Sigma_vk are
  chosen in the metric of observed query energy.  Every statistic is computed online
  from the sequence itself: no calibration data, no stored side parameters, no domain
  shift.

════════════════════════════════════════════════════════════════════════════════════════
3. CACHE LAYOUT (identical for all chunked methods; streaming-exact)
════════════════════════════════════════════════════════════════════════════════════════
  positions [0, N_SINK)                        exact (attention sinks), all methods
  chunk c = [N_SINK + cC, N_SINK + (c+1)C)     compacted from query position
                                               tau_c = last + WINDOW onwards, using
                                               queries (tau_c - W_OBS, tau_c]
  everything not yet in a compacted chunk      exact (recent window + pending chunk)
  The single-pass simulation reads, for every query m, exactly what a streaming decoder
  holds at step m.  decode_equivalence() checks this against a real streaming decoder
  that discards the exact tokens of each chunk once it is compacted.

════════════════════════════════════════════════════════════════════════════════════════
4. METHODS COMPARED AT MATCHED MEMORY (bits per element of the compacted region)
════════════════════════════════════════════════════════════════════════════════════════
  tilt        TiltKV: tails + 2nd-order projected atom + tilted value        (proposed)
  evict       same tails, bulk dropped; the atom's bytes go to more tails    (SnapKV-like)
  mean        tails + mean atom with log-size bias                           (KeepKV/SemantiCache class)
  moment1     tails + first-order mass + tilted value                        (MomentKV-like)
  am          Attention-Matching-lite: top keys, NNLS biases, LS values      (Zweiger et al. 2026)
  ablations   tilt_notilt, tilt_noproj, tilt_nophase, tilt_randsel,
              tilt_adaptive (cross-fitted atoms-vs-tails choice on observed queries)
  global      full cache, KIVI-2/4 (Liu et al. 2024), training-free CLA (KV_LDT_v12_2 `full`)
  Re-implementations of others' methods are simplified ("-like"/"-lite"); state this.

════════════════════════════════════════════════════════════════════════════════════════
5. LONG-CONTEXT BENCHMARKS (Hugging Face datasets; --suite longbench ruler longppl)
════════════════════════════════════════════════════════════════════════════════════════
  LongBench   Xnhyacinth/LongBench (parquet; NVIDIA kvpress' conversion of THUDM/LongBench);
              fallback: the official data.zip of zai-org/THUDM LongBench read directly (the
              loading script no longer runs with datasets >= 4).  Default tasks: qasper,
              hotpotqa, trec, repobench-p (any English task, --tasks; LongBench-E with
              --longbench-e).  Prompts = official THUDM templates split into context /
              question / answer prefix (verified identical to the official prompts),
              official generation lengths, official metrics (qa-F1, classification,
              code-sim, ROUGE-L, retrieval, count) and first-line post-processing, official
              no-chat-template tasks, middle truncation at MAX_CONTEXT_TOKENS.
  RULER       simonjegou/ruler (kvpress), data_dir = context length (4096 / 8192 / 16384 ...);
              13 tasks; string_match_all / string_match_part (qa_*) as in RULER and kvpress.
  Long PPL    allenai/c4 (validation shard, documents concatenated) and emozilla/pg19-test
              (PG-19 books); streaming single pass of LONG_PPL_LEN tokens; NLL by position.
  Protocol    the context is prefilled and compressed, compression is FROZEN at the end of
              the context, question + answer prefix + generated tokens are kept exact for
              every method (kvpress' compress-the-context protocol; --query-aware also
              compresses the question).  Greedy decoding runs token by token from the
              captured compressed state (LayerState); generation_equivalence() checks it
              against a single pass and, for the full cache, against HF generate().
  Memory      matched PER SAMPLE: every compressed method stores <= KEEP x the fp16 context
              cache (KEEP_FRACTIONS = 1/4, 1/8, 1/16); the stored fraction is measured from
              the captured state and reported per sample; contrasts use only units where
              both methods are within KEEP (1 + BUDGET_TOL) (vs KIVI: memory_A <= memory_B
              (1 + BUDGET_TOL)); exclusions are counted.
  Baselines   (a) OFFICIAL NVIDIA kvpress presses (v0.5.5, constructed as its
              evaluate_registry): SnapKV, PyramidKV, AdaKV-SnapKV, Expected Attention
              (AdaKV, eps 1e-2), TOVA, Knorm, StreamingLLM; compression_ratio = 1 - KEEP; run on
              the native path (below).  AdaKV-based presses MASK keys rather than removing
              them, so their memory is the nominal budget (flagged per row).
              (b) harness: full cache, KIVI-2/4 (Liu et al., 2024; residual 32, not 128),
              SnapKV replica of kvpress' SnapKVPress (window 64, kernel 5, n_kept =
              int(n (1 - ratio))), StreamingLLM, chunk-local eviction / mean merge /
              Attention-Matching-lite, ablations.  harness_check: contrasts (full vs native_full,
              SnapKV replica vs official) are two-sided implementation checks, never decisions.
  Paths       harness   fp32 reference simulation (TiltKV and everything compared at the
                        attention-statistics level); decoding from LayerState.
              native    Hugging Face SDPA + DynamicCache in the model dtype; kvpress'
                        KVPressTextGenerationPipeline._forward / generate_answer replicated
                        (checked against the official pipeline: official_pipeline_agreement).
  Statistics  per-sample paired differences; estimand = macro mean over tasks (official
              aggregation); task-stratified bootstrap CI and sign-flip p; Holm per
              benchmark; cross-model rule (>= 7/9 models significant and Wilcoxon over models).
  Outputs     results/<model>/bench/: bench_samples.csv (per sample x method, with all
              systems fields), bench_task_scores.csv (per task, macro_avg, LongBench-E length
              buckets 0-4k / 4-8k / 8k+), bench_systems.csv, bench_contrasts.csv,
              pred/<benchmark>/<method>/<task>.jsonl in the official LongBench eval.py format;
              top level: bench_decision_rules.csv, bench_all_systems.csv.

════════════════════════════════════════════════════════════════════════════════════════
6. SYSTEMS MEASUREMENTS (per sample; what each number is, and how faithful it is)
════════════════════════════════════════════════════════════════════════════════════════
  payload_bytes       MEASURED nbytes of the compressed context cache materialised in its
                      storage format: fp16 exact rows; bit-packed int codes + fp16 scales
                      (8-bit tails; KIVI codes + per-group fp16 scale / zero); fp16 TiltKV
                      atoms (Prop. 5 fields).  analytic_bytes = state_bits / 8 must agree.
                      native rows: nbytes of the DynamicCache tensors after the press.
  memory_fraction     analytic stored bits / fp16 context cache (harness), measured cached
                      tokens / context tokens (native), nominal budget (AdaKV masking).
  sim_state_bytes     MEASURED allocator size of the harness decode state (dense, dequantised,
                      compute dtype) -- what this reference implementation holds, not a kernel.
  peak_*_bytes        torch.cuda.max_memory_allocated, reset before prefill and before decode
                      (weights + activations + cache; NaN on CPU).
  tokenize_s          tokenisation + chat template of the sample (data loading excluded).
  prefill_s, ttft_s   synchronised wall clock; TTFT = prefill (+ question pass on native).
  decode_s, tokens/s  synchronised wall clock of the greedy loop; tokens/s = (n_gen - 1) / decode_s.
  attn_flops_*        ANALYTIC attention FLOPs per decoded token over the actual cache state
                      (2 per multiply-add; Prop. 5 atom cost), *_full for the uncompressed cache.
  flops_decode_step_measured   torch.utils.flop_counter.FlopCounterMode over one complete
                      decode step (matmul / SDPA class ops counted by torch; elementwise ops not).
  model_dtype, attention_dtype recorded per row.
  FIDELITY: harness timings and peaks measure an fp32 reference implementation without
  fused kernels; they are NOT deployment speed for TiltKV.  Only native / kvpress rows
  measure an optimised path.  A fused TiltKV kernel does not exist yet; its speed is not
  claimed.  Bytes and FLOPs are implementation-independent and comparable across paths.

Usage
    python TILT_KV.py --smoke                                     # core suite, offline CPU
    python TILT_KV.py --smoke --suite longbench ruler longppl     # benchmark engine, offline
    python TILT_KV.py --models Llama-3.2-1B                       # core suite, real model
    python TILT_KV.py --suite longbench ruler --models Llama-3.1-8B-Instruct \
           --tasks qasper hotpotqa trec repobench-p --ruler-lengths 4096 8192 16384
    python TILT_KV.py --suite longppl --models Llama-3.1-8B-Instruct --long-ppl-len 16384
    Useful: --keep 0.25 0.125 --max-samples 50 --methods tilt@ snapkv@ streamingllm@ --query-aware
Requires torch >= 2.4, transformers >= 4.56 (tested on 5.2.0), scipy, pandas, datasets and
huggingface_hub (benchmarks), fuzzywuzzy or rapidfuzz (code tasks), rouge (summarisation
tasks), matplotlib (optional), kvpress >= 0.5.5 (official SOTA baselines; optional, skipped
with a warning if absent -- on Python 3.13 its dependency `fire` imports the removed module
`pipes`; use Python <= 3.12 or provide a one-line pipes.py: `from shlex import quote`).
Gated models (Llama) need `huggingface-cli login`.

Validation status (be explicit in any write-up)
  Verified with --smoke only: CPU, two tiny models (Llama; SmolLM3 with NoPE layers)
  trained on a synthetic COPY task, a retrieval-heavy worst case for compaction.
  * Theory checks (synthetic): second-order log-mass error 0.14 vs 3.53 first-order on
    Gaussian blocks; REM freezing correction 4.8 vs 29.5 under sharp queries; both
    projections never worse, value projection 6.7 vs 49.
  * H1 on the toy models' real chunks: second order beat first order on 100% of chunks;
    measured attention sag tracks 1/2 q^T Sigma q (Spearman 0.92-0.96).
  * Matched budget, 2-3 bits: TiltKV PPL 11.0 / 30.7 vs eviction 30.1 / 66.3,
    mean merge 45.9 / 117.9, Attention-Matching-lite 28.1 / 67.9.  Freezing correction
    and the variance term were Holm-significant in 2/2 models.
  * Negative findings: at 6 bits eviction beats TiltKV (atoms cost tokens that the copy
    task needs: a rate crossover); the tilted value gave no gain; the observed-query
    adaptive choice HURT (it does not generalise to future queries) and random tail
    selection matched attention-based selection; KIVI at its own 6-bit rate was far better
    on this retrieval task.
  * Harness: fp32 simulation == SDPA (3e-6); streaming decoder == single pass (argmax
    100%, |dlogit| <= 9e-5, identical kept sets), with compacted exact tokens discarded.
  * Benchmark engine (--smoke --suite longbench ruler longppl; tiny char-level model,
    synthetic stand-ins): decoding from the compressed state reproduced the single pass
    token for token for full, KIVI, SnapKV, StreamingLLM, TiltKV, mean, AM (agreement 1.000)
    and matched HF generate() for the full cache; measured memory 0.24-0.25 at a 0.25
    budget; gold answers score 100 through the whole pipeline; metric unit tests pass.
  * Official kvpress 0.5.5 (tiny model, same smoke): native path == kvpress'
    KVPressTextGenerationPipeline._forward (identical answers) for no press, SnapKV and
    StreamingLLM; the harness SnapKV / StreamingLLM replicas generate the same tokens as the
    official presses (agreement 1.000) at the same kept fraction (0.2476 at keep 0.25).
    PyramidKV raised inside kvpress at keep 0.5 on the 329-token toy prompt (recorded as a
    press error, score NaN, excluded); it ran at keep 0.25.
  * Systems: payload_bytes == analytic_bytes exactly for full, SnapKV, StreamingLLM,
    TiltKV, eviction, mean merge and AM; KIVI payload is 0.4-0.5% above analytic (fp16 scale /
    zero of the last partial group).  Measured FLOPs of one decode step agree between the
    native and harness full cache (639k on the toy model).  Harness SnapKV has more measured
    FLOPs than official SnapKV at the same memory (head-wise evicted rows remain masked
    within the union of kept columns); analytic FLOPs are equal.  Peak memory is NaN on CPU.
  * Metric code differential-tested against the official LongBench metrics.py / eval.py and
    kvpress' RULER scorer: 0 mismatches (4000 random cases per metric; 10 task scorers).
  * The toy model scores 0 on most synthetic tasks: the smoke tests the engine, not quality.
  * NOT run: real Hugging Face datasets and real LLMs (this environment cannot reach the
    Hub).  Dataset ids and columns follow kvpress' evaluation code; check the first run.
  No real-LLM result exists yet; every claim is a hypothesis for the decision rules.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import gc
import json
import logging
import math
import sys
import time
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
from transformers import AutoModelForCausalLM, AutoTokenizer

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
logger = logging.getLogger("tilt_kv")


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


# KV_LDT_v12_2 reference PPL (WikiText-2, window 1024 / stride 512): (no_reuse, CLA rf2_ex20 `full`)
PRIOR_PPL = {
    "SmolLM2-360M": (11.030923, 1386.651678), "Qwen2.5-1.5B": (8.732245, 2693.617741),
    "Llama-3.2-1B": (9.378719, 3133.117293), "Qwen2.5-3B": (7.557348, 85.652599),
    "Llama-3.2-3B": (7.505574, 1647.151265), "SmolLM3-3B-Base": (7.553664, 85.850000),
    "Qwen3-4B-Base": (7.429858, 615.288763),
}

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
    """A compaction rule applied to every completed chunk of every (layer, KV head)."""
    name: str
    family: str                    # "tilt" | "evict" | "mean" | "am"
    target_bits: float             # bits per K/V element of the compacted region (<=)
    order: int = 2                 # cumulant order of the atom's log-mass (1 or 2)
    tilt: bool = True              # query-dependent (tilted) atom value
    project: bool = True           # Prop. 4 / 4' feasible projections
    phase: bool = True             # Prop. 6 freezing (REM) correction of the log-mass
    rank: int = 1                  # rank of the Sigma and Sigma_vk factors
    atoms: int = 1                 # atoms per chunk (bulk split along the principal axis)
    selection: str = "attention"   # "attention" (observed-query share) | "random"
    adaptive: bool = False         # per (chunk, head): atoms vs extra tails, by observed-query error
                                   # (ablation only: did not generalise in the --smoke run)

    def __post_init__(self):
        if self.family not in ("tilt", "evict", "mean", "am"):
            raise ValueError(f"unknown family {self.family!r}")
        if self.order not in (1, 2) or self.selection not in ("attention", "random"):
            raise ValueError("order must be 1 or 2; selection 'attention' or 'random'")


@dataclass
class Config:
    OUTPUT_DIR: str = str(Path(os.environ.get("KV_TILT_OUTPUT_DIR", PROJECT_ROOT / "tilt_results")))
    MODELS: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M", "SmolLM", 8),
        ModelConfig("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", "Qwen", 4),
        ModelConfig("Llama-3.2-1B", "meta-llama/Llama-3.2-1B", "Llama", 4),
        ModelConfig("Qwen2.5-3B", "Qwen/Qwen2.5-3B", "Qwen", 4),
        ModelConfig("Llama-3.2-3B", "meta-llama/Llama-3.2-3B", "Llama", 4),
        ModelConfig("SmolLM3-3B-Base", "HuggingFaceTB/SmolLM3-3B-Base", "SmolLM", 4),
        ModelConfig("Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", "Qwen", 2),
        ModelConfig("Qwen3-8B-Base", "Qwen/Qwen3-8B-Base", "Qwen", 2),
        ModelConfig("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", "Llama", 2),
    ])

    # ── Cache layout (all methods) ────────────────────────────────────
    N_SINK: int = 4
    WINDOW: int = 32               # exact recent window (incl. current token)
    CHUNK: int = 128               # tokens per compacted chunk
    W_OBS: int = 32                # observed queries used when a chunk is compacted
    KEEP_BITS: int = 8             # precision of every stored token-like vector (tails, AM keys/values)

    # ── Operating points ──────────────────────────────────────────────
    TARGET_BITS: Tuple[float, ...] = (3.0, 1.5, 0.75)
    PRIMARY_BITS: float = 1.5
    KIVI_BITS: Tuple[int, ...] = (2, 4)
    KIVI_GROUP: int = 32
    CLA_REUSE: int = 2
    CLA_EXEMPT: float = 0.20
    AM_ITERS: int = 200
    AM_RIDGE: float = 1e-3

    # ── Evaluation ────────────────────────────────────────────────────
    PPL_WINDOW: int = 1024
    PPL_STRIDE: int = 512
    PPL_MAX_WINDOWS: int = 200
    LAMBADA_N: int = 500
    PASSKEY_LENGTHS: Tuple[int, ...] = (2048, 4096)
    PASSKEY_DEPTHS: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    PASSKEY_REPS: int = 4
    DIAG_WINDOWS: int = 8          # windows for the attention-level theory test (H1)
    DIAG_QUERIES: int = 32         # queries sampled per compacted chunk in H1
    ATTN_CHUNK: int = 256
    DECODE_TEST_LEN: int = 384
    RUN_LAMBADA: bool = True
    RUN_PASSKEY: bool = True

    # ── Long-context benchmarks (Hugging Face) ────────────────────────
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
    RULER_HF: str = "simonjegou/ruler"
    RULER_LENGTHS: Tuple[int, ...] = (4096, 8192, 16384)
    RULER_MAX_PER_TASK: Optional[int] = 100
    LONG_PPL_DATASETS: Tuple[str, ...] = ("c4", "pg19")
    LONG_PPL_LEN: int = 16384
    LONG_PPL_N: int = 20
    LONG_PPL_BINS: Tuple[int, ...] = (0, 1024, 4096, 8192)
    C4_FILE: str = "en/c4-validation.00000-of-00008.json.gz"
    PG19_HF: str = "emozilla/pg19-test"
    KEEP_FRACTIONS: Tuple[float, ...] = (0.25, 0.125, 0.0625)   # 4x, 8x, 16x smaller context cache
    PRIMARY_KEEP: float = 0.125
    SNAPKV_WINDOW: int = 64                              # kvpress SnapKVPress defaults (v0.5.5)
    SNAPKV_KERNEL: int = 5
    # official NVIDIA kvpress presses run as SOTA baselines on the native HF path (if installed)
    KVPRESS_PRESSES: Tuple[str, ...] = ("snapkv", "pyramidkv", "adakv_snapkv", "expected_attention",
                                        "tova", "knorm", "streaming_llm")
    MEASURE_FLOPS: bool = True                           # FlopCounterMode over the first decode step
    MAX_CONTEXT_TOKENS: int = 32000
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
        if self.PRIMARY_BITS not in self.TARGET_BITS:
            raise ValueError("PRIMARY_BITS must be one of TARGET_BITS")
        if self.PRIMARY_KEEP not in self.KEEP_FRACTIONS:
            raise ValueError("PRIMARY_KEEP must be one of KEEP_FRACTIONS")
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, "results")
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

    def chunk_methods(self) -> List[ChunkMethod]:
        out = []
        for b in self.TARGET_BITS:
            out += [ChunkMethod(f"tilt@{b:g}", "tilt", b), ChunkMethod(f"evict@{b:g}", "evict", b),
                    ChunkMethod(f"mean@{b:g}", "mean", b), ChunkMethod(f"am@{b:g}", "am", b)]
        b0 = self.PRIMARY_BITS
        out += [ChunkMethod(f"moment1@{b0:g}", "tilt", b0, order=1),
                ChunkMethod(f"tilt_notilt@{b0:g}", "tilt", b0, tilt=False),
                ChunkMethod(f"tilt_noproj@{b0:g}", "tilt", b0, project=False),
                ChunkMethod(f"tilt_nophase@{b0:g}", "tilt", b0, phase=False),
                ChunkMethod(f"tilt_adaptive@{b0:g}", "tilt", b0, adaptive=True),
                ChunkMethod(f"tilt_randsel@{b0:g}", "tilt", b0, selection="random")]
        return out

    def kivi_effective_bits(self, bits: int) -> float:
        return bits + 32.0 / self.KIVI_GROUP


# ════════════════════════════════════════════════════════════════════════════
# MEMORY ACCOUNTING (exact, analytic; bits per K/V element of a compacted chunk)
# ════════════════════════════════════════════════════════════════════════════

def token_bits(d: int, keep_bits: int) -> int:
    """One stored token-like (k, v) pair: 2 d values, plus an fp16 absmax scale per vector."""
    return 2 * d * keep_bits + (32 if keep_bits < 16 else 0)


def atom_floats(method: ChunkMethod, d: int) -> int:
    """fp16 numbers per atom (Prop. 5 for the full TiltKV atom)."""
    r = method.rank
    if method.family == "mean":
        return 2 * d + 1                               # mu_k, mu_v, log n
    if method.family != "tilt":
        return 0
    n = 2 * d + 1                                      # mu_k, mu_v, log n
    if method.order == 2:
        n += d + r * d + r + 1                         # diag, U, lambda, R
    if method.tilt:
        n += 2 * r * d + 1                             # A, B (Sigma_vk ~ A B^T), R_v
    return n


def chunk_budget(method: ChunkMethod, d: int, C: int, keep_bits: int) -> Dict[str, float]:
    """Number of stored tokens (tails, or AM compact tokens) that fits the target rate."""
    budget = method.target_bits * C * 2 * d
    atoms = method.atoms if method.family in ("tilt", "mean") else 0
    a_bits = 16 * atom_floats(method, d) * atoms
    per_tok = token_bits(d, keep_bits) + (16 if method.family == "am" else 0)   # AM: + bias
    m = int(math.floor((budget - a_bits) / per_tok))
    m = max(-1, min(m, C))
    if method.family in ("tilt", "mean") and m >= C:
        m, atoms, a_bits = C, 0, 0
    used = max(m, 0) * per_tok + a_bits
    plan = {"tokens": m, "atoms": atoms, "bits_per_element": used / (C * 2 * d), "feasible": m >= 0}
    if method.family == "tilt" and method.adaptive:
        ev = chunk_budget(ChunkMethod("alt", "evict", method.target_bits), d, C, keep_bits)
        plan["alt_tokens"] = ev["tokens"]
        plan["bits_per_element"] = max(plan["bits_per_element"], ev["bits_per_element"])
    return plan


# ════════════════════════════════════════════════════════════════════════════
# ATTENTION ROUTING
# ════════════════════════════════════════════════════════════════════════════

TILT_ATTN = "tilt_router"
_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]


class Router:
    active = None


def tilt_attention(module, query, key, value, attention_mask, *args, **kwargs):
    ctl = Router.active
    if ctl is None:
        return _SDPA(module, query, key, value, attention_mask, *args, **kwargs)
    return ctl.attend(module, query, key, value, attention_mask, kwargs)


AttentionInterface.register(TILT_ATTN, tilt_attention)
AttentionMaskInterface.register(TILT_ATTN, sdpa_mask)


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
# MOMENT ATOMS (Props 1-4)
# ════════════════════════════════════════════════════════════════════════════

def safe_eigh(M: torch.Tensor):
    """Symmetric eigendecomposition in float64 with a scale-relative jitter (rank-deficient
    covariances of small bulks have many repeated zero eigenvalues)."""
    M64 = 0.5 * (M + M.transpose(-1, -2)).double()
    d = M64.shape[-1]
    scale = M64.diagonal(dim1=-2, dim2=-1).abs().mean(-1).clamp_min(1e-30)[..., None, None]
    eye = torch.eye(d, dtype=M64.dtype, device=M64.device)
    jit = 1e-9 * scale * torch.arange(1, d + 1, dtype=M64.dtype, device=M64.device).diag()  # breaks ties
    try:
        lam, V = torch.linalg.eigh(M64 + jit)
    except Exception:
        lam, V = torch.linalg.eigh(M64 + 1e-6 * scale * eye + jit)
    return lam.to(M.dtype), V.to(M.dtype)


@torch.no_grad()
def moment_atoms(kb: torch.Tensor, vb: torch.Tensor, qref: torch.Tensor, method: ChunkMethod) -> Dict:
    """
    Exponential-family atoms of a bulk set.  kb, vb (B, H, n, d); qref (B, H, nq, d) scaled
    observed queries.  With method.atoms = p > 1 the bulk is cut into p equal slices along
    its principal (query-weighted) axis.  Returns stacked tensors with an atom axis.
    """
    B, H, n, d = kb.shape
    r = method.rank
    Eq = qref.pow(2).mean(2)                                          # (B,H,d) query energy
    Dm = Eq.clamp_min(Eq.amax(-1, keepdim=True) * 1e-6 + 1e-30).sqrt()
    p = method.atoms if n >= method.atoms else 1
    if p > 1:
        kc = kb - kb.mean(2, keepdim=True)
        M = torch.einsum("bhnd,bhne->bhde", kc * Dm[:, :, None], kc * Dm[:, :, None]) / n
        axis = safe_eigh(M)[1][..., -1] * Dm                 # principal axis in key space
        order = torch.argsort(torch.einsum("bhnd,bhd->bhn", kc, axis), dim=-1)
        kb = torch.gather(kb, 2, order[..., None].expand_as(kb))
        vb = torch.gather(vb, 2, order[..., None].expand_as(vb))
    bounds = [round(i * n / p) for i in range(p + 1)]
    out = defaultdict(list)
    for a in range(p):
        k, v = kb[:, :, bounds[a]:bounds[a + 1]], vb[:, :, bounds[a]:bounds[a + 1]]
        na = k.shape[2]
        mu_k, mu_v = k.mean(2), v.mean(2)
        kc, vc = k - mu_k[:, :, None], v - mu_v[:, :, None]
        out["mu_k"].append(mu_k)
        out["mu_v"].append(mu_v)
        out["logn"].append(torch.full((B, H), math.log(na), device=k.device))
        if method.order == 2:
            S = torch.einsum("bhnd,bhne->bhde", kc, kc) / na
            lam, V = safe_eigh(Dm[..., :, None] * S * Dm[..., None, :])
            lam, V = lam[..., -r:].clamp_min(0.0), V[..., -r:]
            U = V / Dm[..., :, None]                                   # u_i = D^-1 v_i
            diag = (S.diagonal(dim1=-2, dim2=-1) - (U.pow(2) * lam[..., None, :]).sum(-1)).clamp_min(0.0)
            out["U"].append(U)
            out["lam"].append(lam)
            out["diag"].append(diag)
            out["R"].append(kc.norm(dim=-1).amax(-1))
        if method.tilt:
            Svk = torch.einsum("bhnd,bhne->bhde", vc, kc) / na         # Cov(v, k)
            P, s, Qh = torch.linalg.svd((Svk * Dm[..., None, :]).double(), full_matrices=False)
            P, s, Qh = P.to(Svk.dtype), s.to(Svk.dtype), Qh.to(Svk.dtype)
            out["A"].append(P[..., :, :r] * s[..., None, :r])
            out["Bk"].append(Qh[..., :r, :].transpose(-1, -2) / Dm[..., :, None])
            out["Rv"].append(vc.norm(dim=-1).amax(-1))
    res = {k: torch.stack(v, 2) for k, v in out.items()}               # atom axis = 2
    res["order"], res["tilt"], res["project"] = method.order, method.tilt, method.project
    res["phase"] = method.phase
    return res


def atom_read(qs: torch.Tensor, mom: Dict, rep: int):
    """
    Logits and value terms of moment atoms for scaled queries qs (B, Hq, c, d).
    Returns logit (B,Hq,c,NA), cfac (B,Hq,c,NA) and tB (B,Hq,c,NA,r) (None if no tilt).
    """
    mu = rep_heads(mom["mu_k"], rep)
    logit = torch.einsum("bhcd,bhad->bhca", qs, mu) + rep_heads(mom["logn"], rep)[:, :, None]
    cfac = torch.ones_like(logit)
    if mom["order"] == 2:
        proj = torch.einsum("bhcd,bhadr->bhcar", qs, rep_heads(mom["U"], rep))
        v2 = torch.einsum("bhcd,bhad->bhca", qs.pow(2), rep_heads(mom["diag"], rep)) + \
            (proj.pow(2) * rep_heads(mom["lam"], rep)[:, :, None]).sum(-1)
        if mom.get("phase", True):
            # Prop. 6 (REM freezing): a block of n tokens follows the cumulant expansion only
            # while the logit spread sigma <= sigma_c = sqrt(2 ln n); beyond, its log-mass is
            # carried by its extreme tokens and grows linearly: sigma sigma_c - ln n (C^1 at sigma_c).
            # log n >= 0 for a real atom; disabled atoms carry log n = -inf, which must only mask
            # the logit (below), never enter sqrt(2 ln n)
            L = rep_heads(mom["logn"], rep)[:, :, None].clamp_min(0.0)
            sig = v2.clamp_min(0).sqrt()
            sc = (2 * L).sqrt()
            hot = sig <= sc
            g = torch.where(hot, 0.5 * v2, sig * sc - L)
            cfac = torch.where(hot, cfac, sc / sig.clamp_min(1e-30))
        else:
            g = 0.5 * v2
        if mom["project"]:
            mb = qs.norm(dim=-1, keepdim=True) * rep_heads(mom["R"], rep)[:, :, None]
            cfac = torch.where(g <= mb, cfac, mb / v2.clamp_min(1e-30))
            g = torch.minimum(g, mb)
        logit = logit + g
    tB = None
    if mom["tilt"]:
        tB = torch.einsum("bhcd,bhadr->bhcar", qs, rep_heads(mom["Bk"], rep))
        # Value projection (Prop. 4'): the exact tilted mean lies in conv{v_j}, inside the ball
        # B(mu_v, R_v); projecting the tilt onto that ball can only reduce the error.
        # ||c A tB|| is computed through the r x r Gram A^T A, without materialising vectors.
        A = rep_heads(mom["A"], rep)
        gram = A.transpose(-1, -2) @ A                                   # (B,Hq,NA,r,r)
        nrm = (cfac * torch.einsum("bhcar,bhars,bhcas->bhca", tB, gram, tB).clamp_min(0).sqrt())
        cfac = cfac * torch.clamp(rep_heads(mom["Rv"], rep)[:, :, None] / nrm.clamp_min(1e-30), max=1.0)
    return logit, cfac, tB


# ════════════════════════════════════════════════════════════════════════════
# CHUNK FINALISATION (shared by the simulation, the streaming decoder and diagnostics)
# ════════════════════════════════════════════════════════════════════════════

def fake_quant_tokens(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-vector symmetric absmax quantisation (the fp16 scale is charged in token_bits)."""
    if bits >= 16:
        return x
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (x / s).round().clamp(-qmax, qmax) * s


def group_queries(q: torch.Tensor, Hkv: int) -> torch.Tensor:
    """(B, Hq, t, d) -> (B, Hkv, rep * t, d): pool the query heads served by each KV head."""
    B, Hq, t, d = q.shape
    return q.view(B, Hkv, Hq // Hkv, t, d).reshape(B, Hkv, (Hq // Hkv) * t, d)


@torch.no_grad()
def finalize_chunk(k: torch.Tensor, v: torch.Tensor, qref: torch.Tensor, method: ChunkMethod,
                   plan: Dict, keep_bits: int, cfg: "Config", key: Tuple) -> Dict:
    """
    k, v (B, Hkv, C, d) fp32 post-RoPE chunk; qref (B, Hkv, nq, d) scaled observed queries.
    Returns token-like atoms {k, v, b, idx} and (TiltKV / mean) moment atoms.
    """
    B, H, C, d = k.shape
    m = plan["tokens"]
    scores = torch.einsum("bhqd,bhcd->bhqc", qref, k)
    share = scores.softmax(-1).mean(2)                                  # within-chunk attention share
    if method.selection == "random":                   # key: one tuple, or one tuple per stacked chunk
        keys = key if isinstance(key, list) else [key]
        share = torch.cat([torch.as_tensor(_rng("randsel", *kk).random((B // len(keys), H, C)), dtype=k.dtype)
                           for kk in keys]).to(k.device)
    out = {"tok": None, "mom": None}
    if method.family == "am":
        idx = torch.topk(share, m, dim=-1).indices.sort(-1).values
        out["tok"] = attention_matching(k, v, qref, scores, idx, cfg)
        out["tok"]["idx"] = idx
        out["tok"]["k"] = fake_quant_tokens(out["tok"]["k"], keep_bits)
        out["tok"]["v"] = fake_quant_tokens(out["tok"]["v"], keep_bits)
        return out
    order = torch.argsort(share, dim=-1, descending=True)
    keep = order[..., :m].sort(-1).values
    bulk = order[..., m:].sort(-1).values
    if m > 0:
        gk = torch.gather(k, 2, keep[..., None].expand(B, H, m, d))
        gv = torch.gather(v, 2, keep[..., None].expand(B, H, m, d))
        out["tok"] = {"k": fake_quant_tokens(gk, keep_bits), "v": fake_quant_tokens(gv, keep_bits),
                      "b": torch.zeros((B, H, m), device=k.device), "idx": keep}
    nb = C - m
    if nb > 0 and plan["atoms"] > 0 and method.family in ("tilt", "mean"):
        kb = torch.gather(k, 2, bulk[..., None].expand(B, H, nb, d))
        vb = torch.gather(v, 2, bulk[..., None].expand(B, H, nb, d))
        if method.family == "mean":       # first-order, query-independent: a token-like atom
            mt = {"k": kb.mean(2, keepdim=True), "v": vb.mean(2, keepdim=True),
                  "b": torch.full((B, H, 1), math.log(nb), device=k.device), "idx": None}
            out["tok"] = mt if out["tok"] is None else {
                x: (torch.cat([out["tok"][x], mt[x]], 2) if x != "idx" else out["tok"]["idx"]) for x in mt}
        else:
            out["mom"] = moment_atoms(kb, vb, qref, method)
            if method.adaptive and plan.get("alt_tokens", 0) > m and qref.shape[2] >= 4:
                use = adaptive_decision(k, v, qref, method, m, plan["alt_tokens"], keep_bits)
                out = adaptive_build(k, v, order, m, plan["alt_tokens"], out["mom"], use, keep_bits)
    return out


def _chunk_error(k, v, q, tk, tv, in_m, mom) -> Tuple[torch.Tensor, torch.Tensor]:
    """Errors of the atoms and tails candidates on queries q: |Z_hat/Z - 1| + ||N_hat - N||/||N||."""
    s_true = torch.einsum("bhqd,bhcd->bhqc", q, k)
    shift = s_true.amax(-1, keepdim=True)
    e_true = (s_true - shift).exp()
    Z, N = e_true.sum(-1), e_true @ v
    e_tok = (torch.einsum("bhqd,bhtd->bhqt", q, tk) - shift).exp()
    al, cf, tB = atom_read(q, mom, 1)
    e_at = (al - shift).exp()
    vat = mom["mu_v"][:, :, None] + (cf[..., None] * torch.einsum("bhqar,bhadr->bhqad", tB, mom["A"])
                                     if mom["tilt"] else 0.0)

    def err(z_hat, n_hat):
        return ((z_hat / Z - 1).abs() + (n_hat - N).norm(dim=-1) / N.norm(dim=-1).clamp_min(1e-30)).mean(-1)
    e_m = e_tok * in_m[:, :, None, :]
    return (err(e_m.sum(-1) + e_at.sum(-1), e_m @ tv + (e_at[..., None] * vat).sum(-2)),
            err(e_tok.sum(-1), e_tok @ tv))


def _top_masks(order, m, m_alt, C):
    top = order[..., :m_alt].sort(-1).values
    rank_of = torch.empty_like(order)
    rank_of.scatter_(-1, order, torch.arange(C, device=order.device).expand_as(order))
    return top, torch.gather(rank_of, -1, top) < m                      # top-m within the top-m_alt


@torch.no_grad()
def adaptive_decision(k, v, qref, method, m, m_alt, keep_bits) -> torch.Tensor:
    """
    Per (chunk, KV head): do exponential-family atoms or extra exact tails (same storage)
    represent the chunk better?  Cross-fitted to avoid selection bias: tails are selected and
    atoms built on the even observed queries, both candidates are scored on the odd ones
    (scoring tails on the queries that selected them is optimistically biased).
    """
    B, H, C, d = k.shape
    qa, qb = qref[:, :, 0::2], qref[:, :, 1::2]
    order = torch.argsort(torch.einsum("bhqd,bhcd->bhqc", qa, k).softmax(-1).mean(2), -1, descending=True)
    top, in_m = _top_masks(order, m, m_alt, C)
    bulk = order[..., m:].sort(-1).values
    nb = C - m
    mom = moment_atoms(torch.gather(k, 2, bulk[..., None].expand(B, H, nb, d)),
                       torch.gather(v, 2, bulk[..., None].expand(B, H, nb, d)), qa, method)
    tk = fake_quant_tokens(torch.gather(k, 2, top[..., None].expand(B, H, m_alt, d)), keep_bits)
    tv = fake_quant_tokens(torch.gather(v, 2, top[..., None].expand(B, H, m_alt, d)), keep_bits)
    err_atoms, err_tails = _chunk_error(k, v, qb, tk, tv, in_m, mom)
    return err_atoms <= err_tails                                       # (B, H)


def adaptive_build(k, v, order, m, m_alt, mom, use_atoms, keep_bits) -> Dict:
    """Both candidates share the top-m_alt token vectors; the unused entries get log-mass -inf."""
    B, H, C, d = k.shape
    top, in_m = _top_masks(order, m, m_alt, C)
    tk = fake_quant_tokens(torch.gather(k, 2, top[..., None].expand(B, H, m_alt, d)), keep_bits)
    tv = fake_quant_tokens(torch.gather(v, 2, top[..., None].expand(B, H, m_alt, d)), keep_bits)
    b = torch.where(use_atoms[..., None] & ~in_m, float("-inf"), 0.0)
    mom = dict(mom)
    mom["logn"] = torch.where(use_atoms[..., None], mom["logn"], torch.full_like(mom["logn"], float("-inf")))
    return {"tok": {"k": tk, "v": tv, "b": b, "idx": top, "use_atoms": use_atoms}, "mom": mom}


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


def concat_atoms(parts: List[Dict], kind: str):
    items = [(p[kind], tau) for p, tau in parts if p[kind] is not None]
    if not items:
        return None
    if kind == "tok":
        res = {x: torch.cat([it[x] for it, _ in items], 2) for x in ("k", "v", "b")}
        res["tau"] = torch.cat([torch.full((it["k"].shape[2],), tau, dtype=torch.long) for it, tau in items])
        return res
    keys = [k for k in items[0][0] if isinstance(items[0][0][k], torch.Tensor)]
    res = {x: torch.cat([it[x] for it, _ in items], 2) for x in keys}
    res.update({f: items[0][0][f] for f in ("order", "tilt", "project", "phase")})
    res["tau"] = torch.cat([torch.full((it["mu_k"].shape[2],), tau, dtype=torch.long) for it, tau in items])
    return res


def chunked_attention(q, scaling, q_pos, K, V, kv_pos, hide_at, tok, mom, chunk) -> torch.Tensor:
    """
    Attention over three kinds of entries, one softmax:
      exact tokens  (K, V at kv_pos), readable by query m iff kv_pos <= m < hide_at
      token atoms   logit q.k + b, value v, visible iff tau <= m
      moment atoms  logit / value from atom_read, visible iff tau <= m
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
    if mom is not None:
        mv, mA = rep_heads(mom["mu_v"], rep), (rep_heads(mom["A"], rep) if mom["tilt"] else None)
        mtau = mom["tau"].to(dev)
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
        if mom is not None:
            ml, cfac, tB = atom_read(qs, mom, rep)
            blocks.append(ml.masked_fill(~(mtau[None, :] <= qp[:, None]), float("-inf")))
        p = torch.cat(blocks, -1).softmax(-1)
        T0 = Kr.shape[2]
        o = p[..., :T0] @ Vr
        off = T0
        if tok is not None:
            nt = tk.shape[2]
            o = o + p[..., off:off + nt] @ tv
            off += nt
        if mom is not None:
            pm = p[..., off:]
            o = o + pm @ mv
            if mA is not None:
                o = o + torch.einsum("bhcar,bhadr->bhcd", (pm * cfac)[..., None] * tB, mA)
        out[:, :, s:s + chunk] = o
    return out.transpose(1, 2).to(q.dtype).contiguous()


# ════════════════════════════════════════════════════════════════════════════
# LAYER METHODS (one object per evaluated configuration)
# ════════════════════════════════════════════════════════════════════════════

class LayerState:
    """
    Compressed cache of one layer after a (possibly frozen) prefill, ready for decoding:
    exact rows in a pre-allocated buffer with per-KV-head liveness (hide), plus the visible
    token atoms and moment atoms.  New tokens are appended exact (all methods), as in the
    compress-the-context-once protocol of KV-compression benchmarks.
    """

    def __init__(self, K, V, pos, hide, tok, mom, extra: int, shared: Optional[int] = None):
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
        self.mom = None if mom is None else {**mom, "tau": torch.full_like(mom["tau"], -1)}

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
                                 self.hide[:, :, :n], self.tok, self.mom, chunk)


class LayerMethod:
    """
    A KV-cache policy.  entries() returns the cache a layer reads during a single forward:
    exact K/V with hide_at (when a row stops being readable) plus token / moment atoms.
    freeze_at = H freezes compression at position H (benchmarks: the end of the context);
    with capture_extra set, the state at the end of the forward is kept for decoding.
    """
    name = "full"
    family = "global"
    one_shot = False                # compresses once at the horizon (needs freeze_at)

    def __init__(self, cfg: "Config"):
        self.cfg = cfg
        self.freeze_at: Optional[int] = None
        self.capture_extra: Optional[int] = None
        self.states: Dict[int, LayerState] = {}

    def bits_per_element(self, d: int) -> float:
        return 16.0

    def exact_tokens(self) -> int:
        """Tokens kept exact beyond the compacted region (worst case), for memory fractions."""
        return 0

    def begin(self):
        self.states = {}

    def horizon(self, T: int) -> int:
        return T if self.freeze_at is None else min(T, self.freeze_at)

    def entries(self, li, q, k, v, scaling, pos):
        return k, v, torch.full((k.shape[2],), BIG, dtype=torch.long), None, None

    def entry_bits(self, kind: str, d: int) -> float:
        """Storage of one entry of a kind ('exact', 'tok', 'mom') per KV head, in bits."""
        return 2 * d * 16.0 if kind == "exact" else 0.0

    def layer(self, li, q, k, v, scaling, pos) -> torch.Tensor:
        K, V, hide, tok, mom = self.entries(li, q, k, v, scaling, pos)
        if self.capture_extra is not None:
            self.capture(li, K, V, pos, hide, tok, mom)
        return chunked_attention(q, scaling, pos, K, V, pos, hide, tok, mom, self.cfg.ATTN_CHUNK)

    def capture(self, li, K, V, pos, hide, tok, mom, shared=None):
        T = K.shape[2]
        B, H = K.shape[0], K.shape[1]
        hide = hide.to(K.device)
        hide = hide.expand(B, H, T) if hide.dim() == 1 else hide
        live = hide >= BIG                                          # readable from now on
        alive_hide = torch.where(live, torch.full_like(hide, BIG), torch.zeros_like(hide))
        if tok is not None:                                         # keep atoms visible at T-1
            vis = (tok["tau"].to(K.device) <= T - 1)
            tok = {"k": tok["k"][:, :, vis], "v": tok["v"][:, :, vis], "b": tok["b"][:, :, vis],
                   "tau": tok["tau"][vis.cpu()]}
            if tok["k"].shape[2] == 0:
                tok = None
        self.states[li] = LayerState(K, V, pos, alive_hide, tok, mom, self.capture_extra, shared)

    def state_bits(self, d: int, context_len: int) -> Dict[str, float]:
        """Stored bits for the CONTEXT part of the captured state (all layers, KV heads)."""
        exact = tok = mom = 0.0
        for li, st in self.states.items():
            if st.shared is not None:
                continue
            ctx = (st.pos[:st.n] < context_len)[None, None, :] & (st.hide[:, :, :st.n] >= BIG)
            exact += float(ctx.sum()) * self.entry_bits("exact", d)
            if st.tok is not None:
                tok += float(torch.isfinite(st.tok["b"]).sum()) * self.entry_bits("tok", d)
            if st.mom is not None:
                mom += float(torch.isfinite(st.mom["logn"]).sum()) * self.entry_bits("mom", d)
        return {"exact": exact, "tok": tok, "mom": mom, "total": exact + tok + mom}


class KIVIMethod(LayerMethod):
    family = "kivi"

    def __init__(self, cfg, bits: int):
        super().__init__(cfg)
        self.bits, self.name = bits, f"kivi{bits}"

    def bits_per_element(self, d):
        return self.cfg.kivi_effective_bits(self.bits)

    def exact_tokens(self):
        return self.cfg.N_SINK + self.cfg.WINDOW

    def entry_bits(self, kind, d):
        return 2 * d * (16.0 if kind == "exact" else self.cfg.kivi_effective_bits(self.bits))

    def entries(self, li, q, k, v, scaling, pos):
        c = self.cfg
        H = self.horizon(k.shape[2])
        kh = kivi_fake_quant(k, self.bits, c.KIVI_GROUP, True, c.N_SINK)
        vh = kivi_fake_quant(v, self.bits, c.KIVI_GROUP, False, c.N_SINK)
        # a token is read exact until it leaves the window (n + W), then from its quantised
        # copy; with a horizon H, tokens still inside the window at H stay exact for good
        pc = pos.cpu()
        q_at = pc + c.WINDOW
        quant = (pc >= c.N_SINK) & (q_at <= H - 1)
        hide = torch.where(quant, q_at, torch.full_like(pc, BIG))
        b = torch.zeros(kh.shape[:3], device=k.device)
        b[:, :, ~quant.to(k.device)] = float("-inf")
        tok = {"k": kh, "v": vh, "b": b, "tau": torch.where(quant, q_at, torch.full_like(pc, BIG))}
        return k, v, hide, tok, None


def kivi_fake_quant(x, bits, group, along_tokens, n_sink):
    """KIVI (Liu et al., ICML 2024): keys per channel over token groups, values per token."""
    y = x.float().clone()
    qmax = 2 ** bits - 1

    def qz(t, dim):
        lo, hi = t.amin(dim, keepdim=True), t.amax(dim, keepdim=True)
        sc = (hi - lo).clamp_min(1e-8) / qmax
        return ((t - lo) / sc).round().clamp(0, qmax) * sc + lo

    T, d = y.shape[2], y.shape[3]
    if along_tokens:
        for s in range(n_sink, T, group):
            y[:, :, s:s + group] = qz(y[:, :, s:s + group], 2)
    else:
        g = group if d % group == 0 else d
        body = y[:, :, n_sink:]
        y[:, :, n_sink:] = qz(body.reshape(*body.shape[:-1], d // g, g), -1).reshape(body.shape)
    return y


class CLAMethod(LayerMethod):
    """Training-free cross-layer sharing, exactly KV_LDT_v12_2's `full` (all positions)."""
    family = "cla"
    name = "cla"

    def __init__(self, cfg, L: int):
        super().__init__(cfg)
        ec = min(int(math.ceil(L * cfg.CLA_EXEMPT)), L - 1)
        self.map = {t: ec + ((t - ec) // cfg.CLA_REUSE) * cfg.CLA_REUSE
                    for t in range(ec, L) if (t - ec) % cfg.CLA_REUSE}
        self.L = L
        self.cache = {}

    def bits_per_element(self, d):
        return 16.0 * (1 - len(self.map) / self.L)

    def begin(self):
        super().begin()
        self.cache = {}

    def layer(self, li, q, k, v, scaling, pos):
        if li in set(self.map.values()):
            self.cache[li] = (k, v)
        if li in self.map:
            k, v = self.cache[self.map[li]]
        K, V, hide, tok, mom = LayerMethod.entries(self, li, q, k, v, scaling, pos)
        if self.capture_extra is not None:
            self.capture(li, K, V, pos, hide, tok, mom, shared=self.map.get(li))
        return chunked_attention(q, scaling, pos, K, V, pos, hide, tok, mom, self.cfg.ATTN_CHUNK)


class ChunkedMethod(LayerMethod):
    """TiltKV and the chunk-local baselines (evict / mean / moment1 / AM) at matched memory."""
    family = "chunked"

    def __init__(self, cfg, method: ChunkMethod, d: int, record: bool = False):
        super().__init__(cfg)
        self.m, self.name, self.family = method, method.name, method.family
        self.plan = chunk_budget(method, d, cfg.CHUNK, cfg.KEEP_BITS)
        self.record = record
        self.kept = {}
        self.atom_use = []

    def bits_per_element(self, d):
        return self.plan["bits_per_element"]

    def exact_tokens(self):
        return self.cfg.N_SINK + self.cfg.WINDOW + self.cfg.CHUNK - 1

    def entry_bits(self, kind, d):
        if kind == "exact":
            return 2 * d * 16.0
        if kind == "mom":
            return 16.0 * atom_floats(self.m, d)
        if self.m.family == "mean":       # tails are token-like; the mean atom is stored as one too
            return token_bits(d, self.cfg.KEEP_BITS)
        return token_bits(d, self.cfg.KEEP_BITS) + (16 if self.m.family == "am" else 0)

    def state_bits(self, d, context_len):
        out = super().state_bits(d, context_len)
        if self.m.family == "mean":       # correct the one mean atom per (chunk, head)
            # chunks compacted before the horizon (st.n counts stored rows, not positions)
            n_atoms = sum(len(chunk_plan(context_len, self.cfg, context_len)) for st in self.states.values()) \
                * next(iter(self.states.values())).K.shape[0] * next(iter(self.states.values())).K.shape[1]
            out["tok"] += n_atoms * (16.0 * atom_floats(self.m, d) - token_bits(d, self.cfg.KEEP_BITS))
            out["total"] = out["exact"] + out["tok"] + out["mom"]
        return out

    def build(self, li, q, k, v, scaling, plan_T):
        """
        Finalise every chunk of this layer in ONE batched call (chunks are stacked on the batch
        axis; finalize_chunk is batch-generic), then unfold the atoms in chunk order.  Each
        chunk's result is the same as finalising it alone (the streaming decoder does that).
        """
        c = self.cfg
        B, Hkv, n, d = k.shape
        hide = torch.full((n,), BIG, dtype=torch.long)
        if not plan_T:
            return None, None, hide
        nch, C = len(plan_T), c.CHUNK
        kc = torch.cat([k[:, :, s:e + 1] for s, e, _ in plan_T]).float()          # (nch*B, H, C, d)
        vc = torch.cat([v[:, :, s:e + 1] for s, e, _ in plan_T]).float()
        qref = torch.cat([group_queries(q[:, :, tau - c.W_OBS + 1: tau + 1].float() * scaling, Hkv)
                          for _, _, tau in plan_T])
        res = finalize_chunk(kc, vc, qref, self.m, self.plan, c.KEEP_BITS, c,
                             [(li, ci) for ci in range(nch)])
        taus = torch.tensor([tau for _, _, tau in plan_T], dtype=torch.long)
        for s_, e_, tau in plan_T:
            hide[s_:e_ + 1] = tau

        def unfold(x):                       # (nch*B, H, a, ...) -> (B, H, nch*a, ...)
            y = x.reshape(nch, B, *x.shape[1:])
            y = y.permute(1, 2, 0, *range(3, y.dim()))
            return y.reshape(B, Hkv, nch * x.shape[2], *x.shape[3:])
        tok = mom = None
        if res["tok"] is not None:
            t = res["tok"]
            if self.record and t.get("idx") is not None:
                for ci in range(nch):
                    self.kept[(li, ci)] = t["idx"][ci * B:(ci + 1) * B].cpu()
            if "use_atoms" in t:
                self.atom_use.append(float(t["use_atoms"].float().mean()))
            tok = {x: unfold(t[x]) for x in ("k", "v", "b")}
            tok["tau"] = taus.repeat_interleave(t["k"].shape[2])
        if res["mom"] is not None:
            mm = res["mom"]
            mom = {x: unfold(y) for x, y in mm.items() if isinstance(y, torch.Tensor)}
            mom.update({f: mm[f] for f in ("order", "tilt", "project", "phase")})
            mom["tau"] = taus.repeat_interleave(mm["mu_k"].shape[2])
        return tok, mom, hide

    def entries(self, li, q, k, v, scaling, pos):
        plan_T = chunk_plan(k.shape[2], self.cfg, self.freeze_at)
        tok, mom, hide = self.build(li, q, k, v, scaling, plan_T)
        return k, v, hide, tok, mom


class SnapKVMethod(LayerMethod):
    """
    SnapKV (Li et al., NeurIPS 2024), replicated from NVIDIA kvpress' SnapKVPress (v0.5.5):
    the last SNAPKV_WINDOW context queries attend to the context (causal, fp32 softmax); the
    window columns are dropped; scores = mean over the window queries, avg_pool1d (kernel
    SNAPKV_KERNEL, padding k//2, count_include_pad as torch's default), averaged over the
    query heads of each KV group; the window gets max + 1 so it is kept; the top n_kept =
    max(1, int(P * keep)) positions per KV head survive (kvpress' compute_n_kept).  Queries
    before the horizon use full attention (compression happens after prefill).
    """
    family = "snapkv"
    one_shot = True

    def __init__(self, cfg, keep: int, name: str):
        super().__init__(cfg)
        self.keep, self.name = keep, name

    def entries(self, li, q, k, v, scaling, pos):
        c = self.cfg
        T = k.shape[2]
        H = self.horizon(T)
        B, Hkv = k.shape[0], k.shape[1]
        w = c.SNAPKV_WINDOW
        if self.freeze_at is None or self.keep >= H or H <= w:
            return LayerMethod.entries(self, li, q, k, v, scaling, pos)
        rep = q.shape[1] // Hkv
        qo = q[:, :, H - w:H].float()
        kk = rep_heads(k[:, :, :H].float(), rep)
        att = (qo @ kk.transpose(-1, -2)) * scaling                         # (B,Hq,w,H)
        att = att + torch.triu(torch.full_like(att, float("-inf")), diagonal=H - w + 1)
        att = att.softmax(-1)[..., :-w]                                    # (B,Hq,w,H-w)
        sc = att.mean(-2)
        sc = F.avg_pool1d(sc, kernel_size=c.SNAPKV_KERNEL, padding=c.SNAPKV_KERNEL // 2, stride=1)
        sc = sc.view(B, Hkv, rep, H - w).mean(2)
        sc = F.pad(sc, (0, w), value=float(sc.max()) + 1)
        top = sc.topk(self.keep, dim=-1).indices
        kept = torch.zeros((B, Hkv, T), dtype=torch.bool, device=k.device)
        kept.scatter_(-1, top, True)
        kept[..., H:] = True                                               # post-horizon tokens
        hide = torch.where(kept, torch.full((B, Hkv, T), BIG, dtype=torch.long, device=k.device),
                           torch.full((B, Hkv, T), H, dtype=torch.long, device=k.device))
        return k, v, hide, None, None


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
        return k, v, hide, None, None


class Sim:
    """Routes every attention call of a forward to a LayerMethod (fp32 attention for all)."""

    def __init__(self, method: LayerMethod):
        self.method = method

    def begin(self, positions):
        self.pos = positions
        self.method.begin()

    def attend(self, module, q, k, v, mask, kwargs):
        if getattr(module, "sliding_window", None):
            raise RuntimeError("sliding-window attention layers are not supported")
        scaling = kwargs.get("scaling") or module.scaling
        return self.method.layer(module.layer_idx, q, k, v, scaling, self.pos), None


class StreamingChunked:
    """
    Real streaming decoder for a ChunkedMethod: per layer it stores exact tokens until their
    chunk is compacted at step tau, then DISCARDS them and keeps only the atoms.  Uses the
    same finalize_chunk and chunked_attention as the simulation.
    """

    def __init__(self, runner: "Runner", cm: ChunkedMethod):
        self.runner, self.cm, self.cfg = runner, cm, runner.cfg
        L = runner.L
        self.K = [dict() for _ in range(L)]
        self.qbuf = [dict() for _ in range(L)]
        self.parts = [[] for _ in range(L)]
        self.kept = {}
        self.plan = None

    def begin(self, t: int, T: int):
        self.t = t
        if self.plan is None:
            self.plan = chunk_plan(T, self.cfg)

    def attend(self, module, q, k, v, mask, kwargs):
        li, t, c = module.layer_idx, self.t, self.cfg
        scaling = kwargs.get("scaling") or module.scaling
        self.K[li][t] = (k, v)
        self.qbuf[li][t] = q
        for p in [p for p in self.qbuf[li] if p <= t - c.W_OBS]:
            del self.qbuf[li][p]
        for ci, (s, e, tau) in enumerate(self.plan):
            if tau != t:
                continue
            kc = torch.cat([self.K[li][p][0] for p in range(s, e + 1)], 2).float()
            vc = torch.cat([self.K[li][p][1] for p in range(s, e + 1)], 2).float()
            qs = torch.cat([self.qbuf[li][p] for p in sorted(self.qbuf[li])], 2).float() * scaling
            res = finalize_chunk(kc, vc, group_queries(qs, k.shape[1]), self.cm.m, self.cm.plan,
                                 c.KEEP_BITS, c, (li, ci))
            if res["tok"] is not None and res["tok"].get("idx") is not None:
                self.kept[(li, ci)] = res["tok"]["idx"].cpu()
            self.parts[li].append((res, tau))
            for p in range(s, e + 1):
                del self.K[li][p]                                    # exact tokens discarded
        pos = sorted(self.K[li])
        Ks = torch.cat([self.K[li][p][0] for p in pos], 2)
        Vs = torch.cat([self.K[li][p][1] for p in pos], 2)
        kv_pos = torch.tensor(pos)
        return chunked_attention(q, scaling, torch.tensor([t], device=DEVICE), Ks, Vs, kv_pos,
                                 torch.full((len(pos),), BIG, dtype=torch.long),
                                 concat_atoms(self.parts[li], "tok"), concat_atoms(self.parts[li], "mom"),
                                 c.ATTN_CHUNK), None

    def stored_bits(self, d: int) -> Dict[str, float]:
        exact = sum(len(layer) for layer in self.K) * self.runner.Hkv * 2 * d * 16
        compact = 0.0                      # only entries that are actually used (finite log-mass)
        for layer in self.parts:
            for res, _ in layer:
                if res["tok"] is not None:
                    nt = float(torch.isfinite(res["tok"]["b"]).sum())
                    compact += nt * (token_bits(d, self.cfg.KEEP_BITS) + (16 if self.cm.m.family == "am" else 0))
                    if self.cm.m.family == "mean":
                        compact += self.runner.Hkv * (16 * atom_floats(self.cm.m, d)
                                                      - token_bits(d, self.cfg.KEEP_BITS))
                if res["mom"] is not None:
                    compact += float(torch.isfinite(res["mom"]["logn"]).sum()) * 16 * atom_floats(self.cm.m, d)
        return {"exact_bits": exact, "compact_bits": compact}


# ════════════════════════════════════════════════════════════════════════════
# MODEL RUNNER
# ════════════════════════════════════════════════════════════════════════════

def bos_prefix(tokenizer) -> List[int]:
    bos = tokenizer.bos_token_id if tokenizer is not None else None
    return [bos] if bos is not None and tokenizer("a").input_ids[:1] == [bos] else []


class Runner:
    def __init__(self, mc: ModelConfig, cfg: Config, model=None, tokenizer=None):
        self.mc, self.cfg = mc, cfg
        if model is None:
            logger.info(f"Loading {mc.name} ({mc.model_id})")
            tokenizer = AutoTokenizer.from_pretrained(mc.model_id)
            model = AutoModelForCausalLM.from_pretrained(mc.model_id, attn_implementation=TILT_ATTN,
                                                         dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True)
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
# SELF-TESTS: harness identities and the theory (Props 2-4) on synthetic data
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def theory_tests():
    """
    Synthetic checks of the propositions, run before any model:
      (a) Gaussian block, full rank: second-order log-mass/value error << first-order.
      (b) heavy-tailed block (Student-t, df = 3): truth inside [0, ||q|| R] (Prop. 4) and
          the projected estimate is never worse than the unprojected one.
    """
    g = torch.Generator().manual_seed(SEED)
    d, n, nq = 16, 4096, 64
    A = torch.randn(2 * d, 2 * d, generator=g) / math.sqrt(2 * d)
    X = torch.randn(n, 2 * d, generator=g) @ A.T * 0.8
    k, v = X[:, :d][None, None], X[:, d:][None, None]
    q = torch.randn(1, 1, nq, d, generator=g) * 0.9
    true_lse = torch.logsumexp(q[0, 0] @ k[0, 0].T, -1)
    true_out = (q[0, 0] @ k[0, 0].T).softmax(-1) @ v[0, 0]
    full = ChunkMethod("t", "tilt", 1.0, rank=d)
    mom = moment_atoms(k, v, q, full)
    m2, cfac, tB = atom_read(q, mom, 1)
    m1 = torch.einsum("bhcd,bhad->bhca", q, mom["mu_k"]) + mom["logn"][:, :, None]
    val2 = mom["mu_v"][:, :, None] + cfac[..., None] * torch.einsum("bhcar,bhadr->bhcad", tB, mom["A"])
    e1 = (m1[0, 0, :, 0] - true_lse).abs().median()
    e2 = (m2[0, 0, :, 0] - true_lse).abs().median()
    ev1 = ((mom["mu_v"][0, 0, 0] - true_out).norm(dim=-1) / true_out.norm(dim=-1)).median()
    ev2 = ((val2[0, 0, :, 0] - true_out).norm(dim=-1) / true_out.norm(dim=-1)).median()
    # (b) heavy tails
    t3 = torch.distributions.StudentT(3.0).sample((n, d))
    kt = (t3 * 0.6)[None, None]
    qt = torch.randn(1, 1, nq, d, generator=g) * 1.5
    lse_t = torch.logsumexp(qt[0, 0] @ kt[0, 0].T, -1)
    mt = moment_atoms(kt, kt, qt, full)
    lp, _, _ = atom_read(qt, mt, 1)
    lu, _, _ = atom_read(qt, {**mt, "project": False}, 1)
    vt = torch.distributions.StudentT(3.0).sample((n, d)) * 0.6 + kt[0, 0] @ (A[:d, :d] * 0.5)
    mv_ = moment_atoms(kt, vt[None, None], qt, full)
    _, cf_p, tb_p = atom_read(qt, mv_, 1)
    raw = torch.einsum("bhcar,bhadr->bhcad", tb_p, mv_["A"])[0, 0, :, 0]
    cf_raw, _, _ = atom_read(qt, {**mv_, "tilt": False}, 1)          # cfac before value projection
    _, cf_lp, _ = atom_read(qt, mv_, 1)
    true_vt = (qt[0, 0] @ kt[0, 0].T).softmax(-1) @ vt
    base = mv_["mu_v"][0, 0, 0]
    e_proj = (base + cf_lp[0, 0, :, 0, None] * raw - true_vt).norm(dim=-1)
    e_raw = (base + cf_raw[0, 0, :, 0, None] * raw - true_vt).norm(dim=-1)
    vproj_ok = bool((e_proj <= e_raw + 1e-4).all())
    kappa = lse_t - (qt[0, 0] @ mt["mu_k"][0, 0, 0] + mt["logn"][0, 0, 0])
    bound = qt[0, 0].norm(dim=-1) * mt["R"][0, 0, 0]
    inside = bool(((kappa >= -1e-4) & (kappa <= bound + 1e-4)).all())
    proj_ok = bool(((lp[0, 0, :, 0] - lse_t).abs() <= (lu[0, 0, :, 0] - lse_t).abs() + 1e-5).all())
    # (c) small Gaussian blocks under sharp queries: the frozen phase of Prop. 6
    kr = torch.randn(1, 1, 32, d, generator=g)
    qr = torch.randn(1, 1, nq, d, generator=g) * 2.5                  # sigma >> sqrt(2 ln 32)
    lse_r = torch.logsumexp(qr[0, 0] @ kr[0, 0].T, -1)
    mr = moment_atoms(kr, kr, qr, full)
    lph, _, _ = atom_read(qr, {**mr, "project": False}, 1)
    lga, _, _ = atom_read(qr, {**mr, "project": False, "phase": False}, 1)
    e_ph, e_ga = (lph[0, 0, :, 0] - lse_r).abs().median(), (lga[0, 0, :, 0] - lse_r).abs().median()
    res = {"rem_lse_err_phase": float(e_ph), "rem_lse_err_gaussian": float(e_ga),
           "gauss_lse_err_first": float(e1), "gauss_lse_err_second": float(e2),
           "gauss_val_err_first": float(ev1), "gauss_val_err_second": float(ev2),
           "heavy_truth_in_interval": inside, "heavy_projection_never_worse": proj_ok,
           "heavy_value_projection_never_worse": vproj_ok,
           "heavy_val_err_projected": float(e_proj.median()), "heavy_val_err_unprojected": float(e_raw.median()),
           "heavy_lse_err_projected": float((lp[0, 0, :, 0] - lse_t).abs().median()),
           "heavy_lse_err_unprojected": float((lu[0, 0, :, 0] - lse_t).abs().median())}
    logger.info("theory tests: " + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                             for k, v in res.items()))
    if not (e2 < 0.25 * e1 and ev2 < 0.5 * ev1 and inside and proj_ok and vproj_ok and e_ph < 0.25 * e_ga):
        raise RuntimeError("theory self-tests failed")
    return res


@torch.no_grad()
def harness_tests(runner: Runner, ids: torch.Tensor):
    """
    (1) fp32 simulation without compression reproduces SDPA (negative control);
    (2) a chunked method that keeps every token at 16 bit reproduces the full cache;
    (3) a chunk plan that never completes a chunk reproduces the full cache.
    """
    c = runner.cfg
    base = runner.forward(ids, None).logits.float()
    full = runner.forward(ids, LayerMethod(c)).logits.float()
    keep_all = ChunkMethod("keep_all", "evict", 16.0 + 1.0)
    cm = ChunkedMethod(c, keep_all, runner.d)
    c16 = c.KEEP_BITS
    c.KEEP_BITS = 16
    cm.plan = {"tokens": c.CHUNK, "atoms": 0, "bits_per_element": 16.0, "feasible": True}
    ident = runner.forward(ids, cm).logits.float()
    c.KEEP_BITS = c16
    tol = 1e-3 if COMPUTE_DTYPE == torch.float32 else 0.25
    d1, d2 = float((base - full).abs().max()), float((full - ident).abs().max())
    logger.info(f"  harness tests: fp32 sim vs SDPA {d1:.2e}; keep-all chunked vs full {d2:.2e}")
    if d1 > tol or d2 > tol:
        raise RuntimeError("harness self-tests failed")


# ════════════════════════════════════════════════════════════════════════════
# H1: ATTENTION-LEVEL TEST OF THE THEORY ON THE REAL MODEL
# ════════════════════════════════════════════════════════════════════════════

class DiagCtl:
    """Full attention; at every layer evaluates first- vs second-order atoms on real chunks."""

    def __init__(self, runner: Runner, method: ChunkMethod):
        self.runner, self.m, self.cfg = runner, method, runner.cfg
        self.plan = chunk_budget(method, runner.d, self.cfg.CHUNK, self.cfg.KEEP_BITS)
        self.rows = []

    def begin(self, positions):
        self.pos = positions

    @torch.no_grad()
    def attend(self, module, q, k, v, mask, kwargs):
        li, c = module.layer_idx, self.cfg
        scaling = kwargs.get("scaling") or module.scaling
        B, Hkv, T, d = k.shape
        rep = q.shape[1] // Hkv
        m = self.plan["tokens"]
        for ci, (s, e, tau) in enumerate(chunk_plan(T, c)):
            kc, vc = k[:, :, s:e + 1].float(), v[:, :, s:e + 1].float()
            qref = group_queries(q[:, :, max(0, tau - c.W_OBS + 1): tau + 1].float() * scaling, Hkv)
            share = torch.einsum("bhqd,bhcd->bhqc", qref, kc).softmax(-1).mean(2)
            bulk = torch.argsort(share, -1, descending=True)[..., m:].sort(-1).values
            nb = bulk.shape[-1]
            if nb < 2:
                continue
            kb = torch.gather(kc, 2, bulk[..., None].expand(B, Hkv, nb, d))
            vb = torch.gather(vc, 2, bulk[..., None].expand(B, Hkv, nb, d))
            mom = moment_atoms(kb, vb, qref, self.m)
            qpos = torch.linspace(tau, T - 1, min(c.DIAG_QUERIES, T - tau)).round().long().unique()
            qs = q[:, :, qpos.to(q.device)].float() * scaling              # future queries
            sc = torch.einsum("bhcd,bhnd->bhcn", qs, rep_heads(kb, rep))
            true_lse = torch.logsumexp(sc, -1)
            true_out = sc.softmax(-1) @ rep_heads(vb, rep)
            l2, cfac, tB = atom_read(qs, mom, rep)
            lu, _, _ = atom_read(qs, {**mom, "project": False}, rep)
            lg, _, _ = atom_read(qs, {**mom, "phase": False}, rep)
            l1 = torch.einsum("bhcd,bhad->bhca", qs, rep_heads(mom["mu_k"], rep)) + \
                rep_heads(mom["logn"], rep)[:, :, None]
            mv = rep_heads(mom["mu_v"], rep)[:, :, None, 0]
            v2 = mv + cfac[..., 0, None] * torch.einsum("bhcr,bhdr->bhcd", tB[..., 0, :],
                                                         rep_heads(mom["A"], rep)[:, :, 0])
            den = true_out.norm(dim=-1).clamp_min(1e-8)
            sag = true_lse - l1[..., 0]
            pred = l2[..., 0] - l1[..., 0]
            self.rows.append({
                "layer": li, "chunk": ci, "n_bulk": nb,
                "lse_err_first": float((l1[..., 0] - true_lse).abs().mean()),
                "lse_err_second_proj": float((l2[..., 0] - true_lse).abs().mean()),
                "lse_err_second_noproj": float((lu[..., 0] - true_lse).abs().mean()),
                "lse_err_second_nophase": float((lg[..., 0] - true_lse).abs().mean()),
                "overshoot_phase": float((l2[..., 0] - true_lse).clamp_min(0).mean()),
                "overshoot_nophase": float((lg[..., 0] - true_lse).clamp_min(0).mean()),
                "val_err_mean": float(((mv - true_out).norm(dim=-1) / den).mean()),
                "val_err_tilted": float(((v2 - true_out).norm(dim=-1) / den).mean()),
                "sag_mean": float(sag.mean()), "sag_predicted_mean": float(pred.mean()),
                "sag_spearman": float(stats.spearmanr(sag.flatten().cpu(), pred.flatten().cpu())[0])
                if sag.numel() > 3 else float("nan")})
        return _SDPA(module, q, k, v, mask, **kwargs)


def attention_level_test(runner: Runner, windows: List[torch.Tensor], cfg: Config) -> Tuple[pd.DataFrame, Dict]:
    method = ChunkMethod("diag", "tilt", cfg.PRIMARY_BITS)
    ctl = DiagCtl(runner, method)
    for x in windows:
        ctl.begin(None)
        with torch.no_grad(), routed(ctl):
            runner.model(input_ids=x.to(DEVICE), use_cache=False, logits_to_keep=1)
    df = pd.DataFrame(ctl.rows)
    if df.empty:
        return df, {}
    summ = {}
    for a, b, name in (("lse_err_second_proj", "lse_err_first", "H1a_logmass"),
                       ("val_err_tilted", "val_err_mean", "H1b_value"),
                       ("lse_err_second_proj", "lse_err_second_noproj", "H1c_projection"),
                       ("lse_err_second_proj", "lse_err_second_nophase", "H1d_freezing"),
                       ("overshoot_phase", "overshoot_nophase", "H1e_overshoot")):
        dlt = (df[a] - df[b]).to_numpy()
        nz = dlt[dlt != 0]
        p = float(stats.binomtest(int((nz < 0).sum()), len(nz), 0.5).pvalue) if len(nz) else 1.0
        summ[name] = {"median_A": float(df[a].median()), "median_B": float(df[b].median()),
                      "fraction_A_better": float((dlt < 0).mean()), "sign_test_p": p, "n_chunks": int(len(dlt))}
    summ["sag_spearman_median"] = float(df.sag_spearman.median())
    logger.info("  H1 attention-level: " + "; ".join(
        f"{k}: {v['median_A']:.4f} vs {v['median_B']:.4f} (A better in {v['fraction_A_better']:.2%}, p={v['sign_test_p']:.2g})"
        for k, v in summ.items() if isinstance(v, dict)) + f"; sag Spearman {summ['sag_spearman_median']:.3f}")
    return df, summ


# ════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════════════════

def wikitext_windows(ids: List[int], cfg: Config) -> List[Tuple[int, int, int]]:
    """(begin, end, n_scored) exactly as KV_LDT_v12_2.DownstreamEvaluator._perplexity."""
    out, prev_end = [], 0
    for begin in range(0, len(ids), cfg.PPL_STRIDE):
        end = min(begin + cfg.PPL_WINDOW, len(ids))
        out.append((begin, end, end - prev_end))
        prev_end = end
        if end == len(ids) or len(out) == cfg.PPL_MAX_WINDOWS:
            break
    return out


@torch.no_grad()
def eval_wikitext(runner: Runner, method: LayerMethod, ids: List[int]) -> pd.DataFrame:
    cfg, bos = runner.cfg, runner.bos
    groups = defaultdict(list)
    for w, (b, e, ns) in enumerate(wikitext_windows(ids, cfg)):
        groups[(e - b, ns)].append((w, b, e))
    rows = []
    for (length, ns), members in groups.items():
        for i in range(0, len(members), runner.mc.eval_batch):
            chunk = members[i:i + runner.mc.eval_batch]
            x = torch.tensor([bos + ids[b:e] for _, b, e in chunk], device=DEVICE)
            T = x.shape[1]
            start = max(1, T - ns)
            lg = runner.forward(x, method, logits_to_keep=T - start + 1).logits.float()
            nll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, start:].reshape(-1),
                                  reduction="none").view(len(chunk), -1).sum(1)
            rows += [{"window": w, "nll": float(val), "n_tok": T - start} for (w, _, _), val in zip(chunk, nll)]
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


@torch.no_grad()
def eval_lambada(runner: Runner, method: LayerMethod, texts: List[str]) -> pd.DataFrame:
    tok, rows = runner.tokenizer, []
    for i, text in enumerate(texts):
        ctx, last = text.rsplit(" ", 1)
        c = runner.bos + tok(ctx, add_special_tokens=False).input_ids
        t = tok(" " + last, add_special_tokens=False).input_ids
        x = torch.tensor([c + t], device=DEVICE)
        lp = runner.forward(x, method, logits_to_keep=len(t) + 1).logits[0, :-1].float().log_softmax(-1)
        tgt = torch.tensor(t, device=DEVICE)
        rows.append({"item": i, "correct": int(bool((lp.argmax(-1) == tgt).all())),
                     "nll": float(-lp.gather(1, tgt[:, None]).sum())})
    return pd.DataFrame(rows)


def passkey_items(runner: Runner, cfg: Config) -> List[Dict]:
    """Passkey retrieval (Mohtashami & Jaggi, 2023); scored teacher-forced (= greedy exact match)."""
    tok = runner.tokenizer
    filler = tok(" The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again.",
                 add_special_tokens=False).input_ids
    items = []
    for length in cfg.PASSKEY_LENGTHS:
        if length > runner.max_pos:
            continue
        for depth in cfg.PASSKEY_DEPTHS:
            for r in range(cfg.PASSKEY_REPS):
                key = int(_rng("passkey", length, depth, r).integers(10000, 99999))
                needle = tok(f" The pass key is {key}. Remember it. {key} is the pass key.",
                             add_special_tokens=False).input_ids
                question = tok(" What is the pass key? The pass key is", add_special_tokens=False).input_ids
                answer = tok(f" {key}", add_special_tokens=False).input_ids
                n_fill = max(1, (length - len(runner.bos) - len(needle) - len(question) - len(answer)) // len(filler))
                at = int(round(depth * n_fill))
                items.append({"length": length, "depth": depth, "rep": r, "answer": answer,
                              "prompt": runner.bos + filler * at + needle + filler * (n_fill - at) + question})
    return items


@torch.no_grad()
def eval_passkey(runner: Runner, method: LayerMethod, items: List[Dict]) -> pd.DataFrame:
    rows = []
    for i, it in enumerate(items):
        x = torch.tensor([it["prompt"] + it["answer"]], device=DEVICE)
        a = len(it["answer"])
        lp = runner.forward(x, method, logits_to_keep=a + 1).logits[0, :-1].float().log_softmax(-1)
        tgt = torch.tensor(it["answer"], device=DEVICE)
        rows.append({"item": i, "length": it["length"], "depth": it["depth"],
                     "correct": int(bool((lp.argmax(-1) == tgt).all())),
                     "nll": float(-lp.gather(1, tgt[:, None]).sum())})
    return pd.DataFrame(rows)


@torch.no_grad()
def decode_equivalence(runner: Runner, method: ChunkMethod, ids: torch.Tensor) -> Dict:
    """Single-pass simulation vs a streaming decoder that discards compacted exact tokens."""
    T = ids.shape[1]
    cm = ChunkedMethod(runner.cfg, method, runner.d, record=True)
    sim_logits = runner.forward(ids, cm).logits[0].float()
    st = StreamingChunked(runner, cm)
    dec = []
    for t in range(T):
        st.begin(t, T)
        with routed(st):
            out = runner.model(input_ids=ids[:, t:t + 1], position_ids=torch.tensor([[t]], device=DEVICE),
                               use_cache=False)
        dec.append(out.logits[0, -1].float())
    dec = torch.stack(dec)
    same = [bool(torch.equal(cm.kept[k_], st.kept[k_])) for k_ in cm.kept if k_ in st.kept]
    full_bits = runner.L * runner.Hkv * 2 * runner.d * T * 16
    mem = st.stored_bits(runner.d)
    res = {"T": T, "max_abs_logit_diff": float((sim_logits - dec).abs().max()),
           "logit_scale": float(sim_logits.abs().max()),
           "argmax_agreement": float((sim_logits.argmax(-1) == dec.argmax(-1)).float().mean()),
           "kept_set_agreement": float(np.mean(same)) if same else float("nan"),
           "n_compacted_chunks": len(same), "stored_fraction_measured": (mem["exact_bits"] + mem["compact_bits"]) / full_bits,
           **mem}
    logger.info(f"  decode equivalence (T={T}): max|dlogit|={res['max_abs_logit_diff']:.2e} "
                f"(scale {res['logit_scale']:.1f}), argmax agreement {res['argmax_agreement']:.4f}, "
                f"kept-set agreement {res['kept_set_agreement']:.4f} over {len(same)} chunk-layers; "
                f"measured stored fraction {res['stored_fraction_measured']:.3f}")
    return res


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


def paired_nll_test(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, n_perm: int, seed: int,
                    block: int = 1) -> Dict:
    """
    A vs B on identical units.  Estimate = (sum nll_A - sum nll_B) / sum n_tok (negative: A
    better).  CI: paired bootstrap over units; p: paired sign-flip randomisation test.
    block > 1 resamples / flips blocks of consecutive units (WikiText windows share context).
    """
    m = a.merge(b, on=a.columns[0], suffixes=("_a", "_b"))
    ntok = m["n_tok_a"].to_numpy() if "n_tok_a" in m else np.ones(len(m))
    d = m["nll_a"].to_numpy() - m["nll_b"].to_numpy()
    nb = int(math.ceil(len(d) / block))
    bid = np.arange(len(d)) // block
    dB = np.bincount(bid, d, nb)
    tB = np.bincount(bid, ntok, nb)
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, nb, size=(n_boot, nb))
    boot = dB[bi].sum(1) / tB[bi].sum(1)
    null = (rng.choice([-1.0, 1.0], size=(n_perm, nb)) * dB).sum(1)
    p = (1 + np.sum(np.abs(null) >= abs(dB.sum()) - 1e-12)) / (n_perm + 1)
    return {"estimate": d.sum() / ntok.sum(), "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "p_value": float(p), "n_units": len(d), "block": block}


def mcnemar(a: pd.Series, b: pd.Series) -> float:
    b01, b10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    return 1.0 if b01 + b10 == 0 else float(stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue)


def contrasts(cfg: Config) -> List[Tuple[str, str, str, str]]:
    """Pre-specified (name, A, B, question); claim: A has lower WikiText-2 NLL at rate_A <= rate_B."""
    b0 = f"{cfg.PRIMARY_BITS:g}"
    lo, hi = f"{min(cfg.TARGET_BITS):g}", f"{max(cfg.TARGET_BITS):g}"
    out = [("T1_vs_evict", f"tilt@{b0}", f"evict@{b0}", "atoms beat spending the bytes on more tokens"),
           ("T2_vs_mean_merge", f"tilt@{b0}", f"mean@{b0}", "second order beats first-order merging"),
           ("T3_vs_attention_matching", f"tilt@{b0}", f"am@{b0}", "vs fitted query-independent biases"),
           ("T4_variance_term", f"tilt@{b0}", f"moment1@{b0}", "query-dependent mass (Prop. 2)"),
           ("T5_tilted_value", f"tilt@{b0}", f"tilt_notilt@{b0}", "tilted value (Prop. 1)"),
           ("T6_projection", f"tilt@{b0}", f"tilt_noproj@{b0}", "feasible projection (Prop. 4)"),
           ("T7_selection", f"tilt@{b0}", f"tilt_randsel@{b0}", "observed-query tail selection"),
           ("T10_freezing", f"tilt@{b0}", f"tilt_nophase@{b0}", "REM freezing correction (Prop. 6)"),
           ("T11_adaptive", f"tilt_adaptive@{b0}", f"tilt@{b0}", "does observed-query validation help?"),
           ("T8_low_rate_vs_cla", f"tilt@{lo}", "cla", "predecessor: coding vs substitution")]
    kv = [b for b in cfg.KIVI_BITS if f"{cfg.kivi_effective_bits(b):g}" == hi]
    if kv:
        out.append(("T9_vs_kivi", f"tilt@{hi}", f"kivi{kv[0]}", "matched-rate quantisation baseline"))
    return out


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def load_text_data(runner: Runner, cfg: Config):
    from datasets import load_dataset
    tok = runner.tokenizer
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    ids = tok("\n\n".join(wiki["text"]), add_special_tokens=False).input_ids
    ids = ids[: cfg.PPL_WINDOW + cfg.PPL_STRIDE * (cfg.PPL_MAX_WINDOWS - 1)]
    lam = []
    if cfg.RUN_LAMBADA:
        lam = list(load_dataset("EleutherAI/lambada_openai", "default", split="test")["text"][: cfg.LAMBADA_N])
    return ids, lam


def run_model(mc: ModelConfig, cfg: Config, prebuilt=None) -> Optional[pd.DataFrame]:
    out_dir = os.path.join(cfg.RESULTS_DIR, mc.name)
    os.makedirs(out_dir, exist_ok=True)
    summary_path = os.path.join(out_dir, "summary.csv")
    if cfg.RESUME and os.path.exists(summary_path):
        logger.info(f"RESUME {mc.name}")
        return pd.read_csv(summary_path)
    t_model = time.time()
    if prebuilt is None:
        runner = Runner(mc, cfg)
        test_ids, lam_texts = load_text_data(runner, cfg)
    else:
        model, test_ids = prebuilt
        runner = Runner(mc, cfg, model=model)
        lam_texts = []
    win = [torch.tensor([runner.bos + test_ids[b:e]]) for b, e, _ in wikitext_windows(test_ids, cfg)[: cfg.DIAG_WINDOWS]]
    harness_tests(runner, win[0][:, : min(win[0].shape[1], 2 * cfg.CHUNK + cfg.WINDOW + cfg.N_SINK + 8)].to(DEVICE))

    diag, h1 = attention_level_test(runner, win, cfg)
    diag.to_csv(os.path.join(out_dir, "attention_level_H1.csv"), index=False)
    with open(os.path.join(out_dir, "attention_level_H1.json"), "w") as f:
        json.dump(h1, f, indent=2)

    items = passkey_items(runner, cfg) if (cfg.RUN_PASSKEY and runner.tokenizer is not None) else []
    results, per_unit = [], {}

    def evaluate(method: LayerMethod, extra=None):
        t0 = time.time()
        wt = eval_wikitext(runner, method, test_ids)
        lam = eval_lambada(runner, method, lam_texts) if lam_texts else None
        pk = eval_passkey(runner, method, items) if items else None
        rate = method.bits_per_element(runner.d)
        row = {"method": method.name, "family": method.family, "bits_per_element": rate,
               "wikitext2_nll": wt.nll.sum() / wt.n_tok.sum()}
        row["wikitext2_ppl"] = math.exp(row["wikitext2_nll"])
        for T in (4096, 32768, 131072):
            ex = min(T, method.exact_tokens()) if method.family not in ("global", "cla") else 0
            row[f"mem_fraction_T{T}"] = (rate * (T - ex) + 16.0 * ex) / (16.0 * T)
        if lam is not None:
            row.update(lambada_acc=lam.correct.mean(), lambada_nll=lam.nll.mean())
        if pk is not None and len(pk):
            row["passkey_acc"] = pk.correct.mean()
            for L_, g in pk.groupby("length"):
                row[f"passkey_acc_{L_}"] = g.correct.mean()
        row.update(extra or {})
        row["eval_seconds"] = time.time() - t0
        results.append(row)
        per_unit[method.name] = (wt, lam, pk)
        wt.assign(method=method.name).to_csv(os.path.join(out_dir, f"wikitext_{method.name}.csv"), index=False)
        for nm, df in (("lambada", lam), ("passkey", pk)):
            if df is not None:
                df.assign(method=method.name).to_csv(os.path.join(out_dir, f"{nm}_{method.name}.csv"), index=False)
        logger.info(f"  {method.name:<20} bits={rate:6.3f} ppl={row['wikitext2_ppl']:.4f}"
                    + (f" lambada={row['lambada_acc']:.3f}" if "lambada_acc" in row else "")
                    + (f" passkey={row['passkey_acc']:.3f}" if "passkey_acc" in row else ""))

    evaluate(LayerMethod(cfg))
    evaluate(CLAMethod(cfg, runner.L))
    for b in cfg.KIVI_BITS:
        evaluate(KIVIMethod(cfg, b))
    if mc.name in PRIOR_PPL:
        got = {r["method"]: r["wikitext2_ppl"] for r in results}
        ref_full, ref_cla = PRIOR_PPL[mc.name]
        logger.info(f"  reproduction vs KV_LDT_v12_2: full {got['full']:.4f} vs {ref_full:.4f} "
                    f"({got['full'] / ref_full - 1:+.2%}); CLA {got['cla']:.2f} vs {ref_cla:.2f}")
    for m in cfg.chunk_methods():
        cm = ChunkedMethod(cfg, m, runner.d)
        if not cm.plan["feasible"]:
            logger.warning(f"  {m.name}: infeasible at d={runner.d}, CHUNK={cfg.CHUNK} (atom exceeds budget); skipped")
            continue
        evaluate(cm, {"tokens_per_chunk": cm.plan["tokens"], "atoms_per_chunk": cm.plan["atoms"],
                      "atom_floats": atom_floats(m, runner.d)})
        if cm.atom_use:
            results[-1]["atom_use_fraction"] = float(np.mean(cm.atom_use))
            logger.info(f"      atoms chosen in {results[-1]['atom_use_fraction']:.1%} of (chunk, head) pairs")
    prim = ChunkMethod(f"tilt@{cfg.PRIMARY_BITS:g}", "tilt", cfg.PRIMARY_BITS)
    if ChunkedMethod(cfg, prim, runner.d).plan["feasible"]:
        T = min(cfg.DECODE_TEST_LEN, len(test_ids))
        ids = torch.tensor([runner.bos + list(test_ids[:T - len(runner.bos)])], device=DEVICE)
        with open(os.path.join(out_dir, "decode_equivalence.json"), "w") as f:
            json.dump(decode_equivalence(runner, prim, ids), f, indent=2)

    summary = pd.DataFrame(results)
    full_nll = float(summary.loc[summary.method == "full", "wikitext2_nll"].iloc[0])
    summary["delta_nll_vs_full"] = summary.wikitext2_nll - full_nll
    summary.insert(0, "model", mc.name)
    summary.insert(1, "model_family", mc.family)

    rows = []
    block = max(1, cfg.PPL_WINDOW // cfg.PPL_STRIDE)
    for name, A, B, question in contrasts(cfg):
        if A not in per_unit or B not in per_unit:
            continue
        r = paired_nll_test(per_unit[A][0], per_unit[B][0], cfg.N_BOOT, cfg.N_PERM, SEED, block)
        ra = float(summary.loc[summary.method == A, "bits_per_element"].iloc[0])
        rb = float(summary.loc[summary.method == B, "bits_per_element"].iloc[0])
        ta, tb = (float(x.split("@")[1]) if "@" in x else float("nan") for x in (A, B))
        row = {"model": mc.name, "model_family": mc.family, "contrast": name, "question": question,
               "A": A, "B": B, "rate_A": ra, "rate_B": rb, "budget_A": ta, "budget_B": tb, **r}
        for k_, idx in (("lambada", 1), ("passkey", 2)):
            ua, ub = per_unit[A][idx], per_unit[B][idx]
            if ua is not None and ub is not None and len(ua):
                row[f"{k_}_acc_diff"] = ua.correct.mean() - ub.correct.mean()
                row[f"{k_}_mcnemar_p"] = mcnemar(ua.correct, ub.correct)
        rows.append(row)
    ct = pd.DataFrame(rows)
    if len(ct):
        ct["p_holm"] = holm(ct.p_value)
        # never credit a win bought with bits: either A uses no more measured bits than B, or both
        # are chunked methods at the SAME budget and A stays within it (matched-budget design;
        # whole-token granularity can leave part of B's budget unused)
        same_budget = (ct.budget_A == ct.budget_B) & (ct.rate_A <= ct.budget_A + 1e-9)
        ct["rate_ok"] = (ct.rate_A <= ct.rate_B + 1e-9) | same_budget
        ct["supported"] = (ct.estimate < 0) & (ct.p_holm < cfg.ALPHA) & ct.rate_ok
        ct.to_csv(os.path.join(out_dir, "contrasts.csv"), index=False)
    summary["total_seconds"] = time.time() - t_model
    summary.to_csv(summary_path, index=False)
    if prebuilt is None:
        runner.release()
    return summary


def cross_model(cfg: Config, summaries: List[pd.DataFrame]):
    if not summaries:
        return
    allsum = pd.concat(summaries, ignore_index=True)
    allsum.to_csv(os.path.join(cfg.OUTPUT_DIR, "all_summary.csv"), index=False)
    h1 = []
    for p in Path(cfg.RESULTS_DIR).glob("*/attention_level_H1.json"):
        with open(p) as f:
            js = json.load(f)
        for k, v in js.items():
            if isinstance(v, dict):
                h1.append({"model": p.parent.name, "test": k, **v})
    if h1:
        h1 = pd.DataFrame(h1)
        h1["supported"] = (h1.fraction_A_better > 0.5) & (h1.sign_test_p < cfg.ALPHA)
        h1.to_csv(os.path.join(cfg.OUTPUT_DIR, "attention_level_H1_all.csv"), index=False)
    cts = [pd.read_csv(p) for p in Path(cfg.RESULTS_DIR).glob("*/contrasts.csv")]
    if not cts:
        return
    ct = pd.concat(cts, ignore_index=True)
    ct.to_csv(os.path.join(cfg.OUTPUT_DIR, "all_contrasts.csv"), index=False)
    rows = []
    for name, g in ct.groupby("contrast", sort=False):
        n = len(g)
        need = int(math.ceil(cfg.DECISION_MIN_MODEL_FRACTION * n))
        met = int(g.supported.sum())
        pw = float(stats.wilcoxon(g.estimate, alternative="less").pvalue) if n >= cfg.MIN_MODELS_WILCOXON else float("nan")
        verdict = ("insufficient models" if n < cfg.MIN_MODELS_WILCOXON else
                   "supported" if (met >= need and pw < cfg.ALPHA) else "not supported")
        lofo = {}
        for fam in g.model_family.unique():
            h = g[g.model_family != fam]
            lofo[f"lofo_{fam}"] = bool(len(h) and h.supported.sum() >= math.ceil(cfg.DECISION_MIN_MODEL_FRACTION * len(h)))
        rows.append({"contrast": name, "question": g.question.iloc[0], "models_evaluated": n,
                     "models_meeting_rule": met, "models_required": need,
                     "median_delta_nll": float(g.estimate.median()), "p_wilcoxon_models": pw,
                     "verdict": verdict, **lofo,
                     "rule": f"A better than B (block sign-flip, Holm p<{cfg.ALPHA}, measured rate_A<=rate_B) "
                             f"in >= {need}/{n} models AND one-sided Wilcoxon over models p<{cfg.ALPHA}"})
    dr = pd.DataFrame(rows)
    dr.to_csv(os.path.join(cfg.OUTPUT_DIR, "decision_rules.csv"), index=False)
    for _, r in dr.iterrows():
        logger.info(f"DECISION {r.contrast:<26} {r.verdict:<20} {r.models_meeting_rule}/{r.models_evaluated} "
                    f"median dNLL={r.median_delta_nll:+.4f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.4))
        markers = {"tilt": "o-", "evict": "v--", "mean": "s--", "am": "D--"}
        for fam, mk in markers.items():
            g = allsum[allsum.method.str.fullmatch(fam + r"@[\d.]+")]
            if len(g):
                gg = g.groupby("bits_per_element", as_index=False).delta_nll_vs_full.median()
                ax.plot(gg.bits_per_element, gg.delta_nll_vs_full, mk, label=fam)
        kv = allsum[allsum.family == "kivi"].groupby("bits_per_element", as_index=False).delta_nll_vs_full.median()
        ax.plot(kv.bits_per_element, kv.delta_nll_vs_full, "k^", label="KIVI")
        ax.set_xlabel("bits per element of the compacted region (16 = fp16)")
        ax.set_ylabel("median WikiText-2 dNLL/token vs full")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.OUTPUT_DIR, "rate_distortion.png"), dpi=200)
        plt.close(fig)
    except Exception as e:
        logger.warning(f"figure skipped: {e}")


# ════════════════════════════════════════════════════════════════════════════
# SMOKE TEST (offline, CPU): tiny models trained on a synthetic copy task
# ════════════════════════════════════════════════════════════════════════════

def smoke_setup(cfg: Config):
    from transformers import LlamaConfig, SmolLM3Config
    V, period = 128, 48
    g = torch.Generator().manual_seed(SEED)
    trans = torch.softmax(torch.randn(V, V, generator=g) * 3.0, -1)

    def sample(n, length):
        """Markov segments of `period` tokens, repeated: a repeat needs attention `period`
        positions back, i.e. into compacted chunks."""
        seg = torch.empty((n, period), dtype=torch.long)
        seg[:, 0] = torch.randint(3, V, (n,), generator=g)
        for t in range(1, period):
            seg[:, t] = torch.multinomial(trans[seg[:, t - 1]], 1, generator=g)[:, 0]
        x = seg.repeat(1, length // period + 1)[:, :length].clone()
        x[:, 0] = 0
        return x

    common = dict(vocab_size=V, hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024,
                  bos_token_id=0, eos_token_id=1, pad_token_id=2, tie_word_embeddings=True)
    builds = [("tiny-llama", "Llama", LlamaConfig(**common)),
              ("tiny-smollm3-nope", "SmolLM", SmolLM3Config(**common, no_rope_layers=[1, 1, 0, 1, 1, 0],
                                                            use_sliding_window=False))]
    for name, fam, conf in builds:
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_config(conf, attn_implementation=TILT_ATTN)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
        for _ in range(400):
            x = sample(16, 144)
            loss = model(input_ids=x, labels=x).loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        logger.info(f"smoke: trained {name}, final loss {loss.item():.3f}")
        model.eval()
        test_ids = sample(1, cfg.PPL_WINDOW + cfg.PPL_STRIDE * (cfg.PPL_MAX_WINDOWS - 1))[0].tolist()
        yield ModelConfig(name, "random-init", fam, 4), (model, test_ids)



# ════════════════════════════════════════════════════════════════════════════
# LONG-CONTEXT BENCHMARKS: LongBench, RULER, long-document perplexity (C4, PG-19)
# ════════════════════════════════════════════════════════════════════════════
#
# Protocol (as in NVIDIA kvpress and most KV-compression papers): the CONTEXT is prefilled
# and compressed; compression is then frozen; the question, the answer prefix and the
# generated tokens are appended exact for every method.  QUERY_AWARE = True instead
# compresses context + question (SnapKV's original, question-aware setting).  Memory is
# matched PER SAMPLE: every compressed method stores at most KEEP x the fp16 cache of the
# context (measured from the captured state, reported per sample).  TiltKV compacts while
# prefilling (bounded peak memory); SnapKV materialises the full cache first.
#
# LongBench prompts: split of the OFFICIAL THUDM/LongBench prompts into context / question /
# answer-prefix parts (identical to NVIDIA kvpress' Xnhyacinth/LongBench conversion; the
# concatenation reproduces the official prompt exactly for 12/16 English tasks and by
# construction for trec, triviaqa, samsum, lcc).  Generation lengths: official
# dataset2maxlen.  Metrics: official LongBench metrics.py / eval.py, verbatim.  Chat
# templates are applied to every task except the official exceptions (trec, triviaqa,
# samsum, lsht, lcc, repobench-p).  Over-long contexts are truncated in the middle (official
# rule) at the token level.

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
    import re as _re
    import string as _string

    def remove_articles(text):
        return _re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(_string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def lb_count_score(prediction, ground_truth, **kwargs):
    import re as _re
    numbers = _re.findall(r"\d+", prediction)
    right_num = sum(1 for number in numbers if str(number) == str(ground_truth))
    return float(0.0 if len(numbers) == 0 else right_num / len(numbers))


def lb_retrieval_score(prediction, ground_truth, **kwargs):
    import re as _re
    ground_truth_id = _re.findall(r'Paragraph (\d+)', ground_truth)[0]
    numbers = _re.findall(r"\d+", prediction)
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
    """RULER string_match_part (qa_*) / string_match_all (others), per sample, x 100."""
    import re as _re
    pred = _re.sub(r"[\x00-\x1f]", "", prediction.strip()).strip().lower()
    hits = [1.0 if r.lower() in pred else 0.0 for r in refs]
    return 100.0 * (max(hits) if task.split("_")[0] == "qa" else sum(hits) / len(hits))


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
        "first_line": (longbench_sample_score("trec", "\nNUM\nLOC", ["NUM"], ["NUM", "LOC"]), 100.0),
    }
    bad = {k: v for k, v in checks.items() if abs(v[0] - v[1]) > 1e-9}
    if bad:
        raise RuntimeError(f"metric self-tests failed: {bad}")
    logger.info(f"metric tests passed ({len(checks)} checks)")


# ── Datasets (Hugging Face) ──────────────────────────────────────────────────────────
def _load_hf(*args, **kw):
    from datasets import load_dataset
    return load_dataset(*args, **kw)


def load_longbench(task: str, cfg: "Config") -> List[Dict]:
    """
    One LongBench(-E) task as dicts with context / question / answer_prefix / answers /
    all_classes / length.  Order: Xnhyacinth/LongBench (parquet; kvpress), then the official
    THUDM / zai-org data.zip read directly (no loading script; works with datasets >= 4).
    """
    name = f"{task}_e" if cfg.LONGBENCH_E else task
    rows, errors = None, []
    for kw in ({"name": name}, {"data_dir": name}):
        try:
            ds = _load_hf(cfg.LONGBENCH_HF, split="test", **kw)
            rows = [dict(r) for r in ds]
            break
        except Exception as e:
            errors.append(f"{cfg.LONGBENCH_HF} {kw}: {e!r}"[:300])
    if rows is None:
        rows = _load_longbench_zip(name, task, cfg, errors)
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


def load_ruler(length: int, cfg: "Config") -> List[Dict]:
    """RULER (Hsieh et al., COLM 2024) as hosted by kvpress: simonjegou/ruler, data_dir = length."""
    ds = _load_hf(cfg.RULER_HF, data_dir=str(length), split="test")
    df = ds.to_pandas()
    out = []
    for task, g in df.groupby("task", sort=True):
        if cfg.RULER_MAX_PER_TASK:
            g = g.sample(n=min(len(g), cfg.RULER_MAX_PER_TASK), random_state=SEED)
        for i, r in g.iterrows():
            out.append({"id": f"{task}-{i}", "context": r["context"], "question": r["question"],
                        "answer_prefix": r["answer_prefix"], "answers": list(r["answer"]),
                        "max_new_tokens": int(r["max_new_tokens"]), "task": task, "length": length})
    return out


def load_long_ppl(name: str, runner: "Runner", cfg: "Config") -> List[List[int]]:
    """Long-document LM sequences: C4 (allenai/c4 validation shard, documents concatenated)
    and PG-19 (emozilla/pg19-test, one book per sequence, first LONG_PPL_LEN tokens)."""
    tok, bos = runner.tokenizer, runner.bos
    L = min(cfg.LONG_PPL_LEN, runner.max_pos) - len(bos)
    seqs = []
    if name == "c4":
        ds = _load_hf("allenai/c4", data_files={"validation": cfg.C4_FILE}, split="validation", streaming=True)
        sep = tok("\n\n", add_special_tokens=False).input_ids
        buf = []
        for r in ds:
            buf += tok(r["text"], add_special_tokens=False).input_ids + sep
            while len(buf) >= L and len(seqs) < cfg.LONG_PPL_N:
                seqs.append(bos + buf[:L])
                buf = buf[L:]
            if len(seqs) >= cfg.LONG_PPL_N:
                break
    elif name == "pg19":
        try:
            ds = _load_hf(cfg.PG19_HF, split="test", streaming=True)
        except Exception as e:
            raise RuntimeError(f"PG-19 could not be loaded from {cfg.PG19_HF} ({e!r}); set Config.PG19_HF to a "
                               f"parquet copy of the PG-19 test split with a 'text' column") from e
        for r in ds:
            ids = tok(r["text"], add_special_tokens=False).input_ids
            if len(ids) >= L:
                seqs.append(bos + ids[:L])
            if len(seqs) >= cfg.LONG_PPL_N:
                break
    else:
        raise ValueError(f"unknown long-PPL dataset {name!r}")
    return seqs


# ── Prompt assembly ─────────────────────────────────────────────────────────────────
def encode_item(runner: "Runner", item: Dict, max_new: int, cfg: "Config", chat_ok: bool = True) -> Dict:
    """
    Token ids of context and question parts (kvpress style): with a chat template, the user
    turn contains context + question and the generation prompt follows; the answer prefix is
    appended after it.  Without a template the model BOS (if any) starts the context.
    Over-long contexts are truncated in the middle; returns ids, context_len and flags.
    """
    tok = runner.tokenizer
    ctx, q, ap = item["context"], item["question"], item["answer_prefix"]
    use_chat = bool(chat_ok and cfg.USE_CHAT_TEMPLATE and getattr(tok, "chat_template", None))
    if use_chat:
        sep = "<<<TILTKV_SEPARATOR_7f3a>>>"
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
    q_ids = tok.encode(q_text, add_special_tokens=False)
    budget = min(cfg.MAX_CONTEXT_TOKENS, runner.max_pos - len(q_ids) - max_new - 1)
    truncated = len(ctx_ids) > budget
    if truncated:
        half = budget // 2
        ctx_ids = ctx_ids[:half] + ctx_ids[len(ctx_ids) - (budget - half):]
    ids = ctx_ids + q_ids
    return {"ids": ids, "context_len": len(ids) if cfg.QUERY_AWARE else len(ctx_ids),
            "chat": use_chat, "truncated": truncated}


# ── Matched-memory methods per sample ───────────────────────────────────────────────
@dataclass(frozen=True)
class BenchSpec:
    name: str
    family: str                        # full | kivi | cla | snapkv | streamingllm | chunk | native | kvpress
    keep: Optional[float] = None       # memory budget: fraction of the fp16 context cache
    kivi_bits: Optional[int] = None
    chunk: Optional[ChunkMethod] = None
    press: Optional[str] = None        # kvpress press name (official implementation)


@lru_cache(maxsize=1)
def kvpress_available() -> bool:
    try:
        import kvpress  # noqa: F401
        return True
    except Exception as e:                  # ImportError, or Python 3.13 + fire 0.6 ('pipes')
        logger.warning(f"kvpress unavailable ({e!r}); official SOTA presses are skipped")
        return False


def build_press(name: str, keep: float):
    """Official kvpress presses, constructed as in kvpress' evaluate_registry.py (v0.5.5)."""
    import kvpress as K
    ratio = 1.0 - keep
    presses = {"snapkv": lambda: K.SnapKVPress(), "pyramidkv": lambda: K.PyramidKVPress(),
               "adakv_snapkv": lambda: K.AdaKVPress(K.SnapKVPress()),
               "expected_attention": lambda: K.AdaKVPress(K.ExpectedAttentionPress(epsilon=1e-2)),
               "tova": lambda: K.TOVAPress(), "knorm": lambda: K.KnormPress(),
               "streaming_llm": lambda: K.StreamingLLMPress()}
    press = presses[name]()
    press.compression_ratio = ratio            # as evaluate.py: set after construction (AdaKV delegates)
    return press


MASKING_PRESSES = ("adakv_snapkv", "expected_attention")   # AdaKV masks keys: memory is nominal


def bench_specs(cfg: "Config", for_ppl: bool = False) -> List[BenchSpec]:
    out = [BenchSpec("full", "full")] + [BenchSpec(f"kivi{b}", "kivi", kivi_bits=b) for b in cfg.KIVI_BITS]
    if not for_ppl:
        out.append(BenchSpec("native_full", "native"))       # HF SDPA + DynamicCache reference
    presses = list(cfg.KVPRESS_PRESSES) if (not for_ppl and cfg.KVPRESS_PRESSES and kvpress_available()) else []
    for r in cfg.KEEP_FRACTIONS:
        t = f"{r:g}"
        out += [BenchSpec(f"kvpress:{pn}@{t}", "kvpress", r, press=pn) for pn in presses]
        if not for_ppl:
            out.append(BenchSpec(f"snapkv@{t}", "snapkv", r))
        out.append(BenchSpec(f"streamingllm@{t}", "streamingllm", r))
        for fam in ("tilt", "evict", "mean", "am"):
            out.append(BenchSpec(f"{fam}@{t}", "chunk", r, chunk=ChunkMethod(f"{fam}@{t}", fam, 1.0)))
    r0 = f"{cfg.PRIMARY_KEEP:g}"
    out += [BenchSpec(f"tilt_nophase@{r0}", "chunk", cfg.PRIMARY_KEEP,
                      chunk=ChunkMethod(f"tilt_nophase@{r0}", "tilt", 1.0, phase=False)),
            BenchSpec(f"moment1@{r0}", "chunk", cfg.PRIMARY_KEEP,
                      chunk=ChunkMethod(f"moment1@{r0}", "tilt", 1.0, order=1))]
    if cfg.BENCH_INCLUDE_CLA:
        out.append(BenchSpec("cla", "cla"))
    if cfg.BENCH_METHODS:                                  # e.g. ("full", "tilt@", "snapkv@")
        out = [s_ for s_ in out if s_.name == "full" or any(s_.name.startswith(m) for m in cfg.BENCH_METHODS)]
    return out


def chunk_rate_for_budget(P: int, keep: float, method: ChunkMethod, d: int, cfg: "Config"):
    """
    Bits per element of the compacted chunks such that sinks + pending + window (exact) plus
    the chunks fit keep x the fp16 cache of a P-token context.  Falls back to the cheapest
    feasible chunk code if the budget is below it (flagged; the measured memory is reported).
    """
    n_ch = len(chunk_plan(P, cfg, P))
    if n_ch == 0:
        return None, "no chunk completes (context too short)"
    exact = P - n_ch * cfg.CHUNK
    r = 16.0 * (keep * P - exact) / (n_ch * cfg.CHUNK)
    floor = max(1e-3, 16.0 * atom_floats(method, d) * (method.atoms if method.family in ("tilt", "mean") else 0)
                / (cfg.CHUNK * 2 * d))
    if method.family in ("evict", "am"):
        floor = (token_bits(d, cfg.KEEP_BITS) + (16 if method.family == "am" else 0)) / (cfg.CHUNK * 2 * d)
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
    if spec.family == "snapkv":                             # kvpress' compute_n_kept
        keep = max(1, int(P * spec.keep))
        return SnapKVMethod(cfg, keep, spec.name), ("" if P > cfg.SNAPKV_WINDOW else "context <= window")
    if spec.family == "streamingllm":
        keep = max(cfg.N_SINK + 1, int(P * spec.keep))
        return StreamingLLMMethod(cfg, keep, spec.name), ("" if keep < P else "budget >= context")
    r, note = chunk_rate_for_budget(P, spec.keep, spec.chunk, runner.d, cfg)
    if r is None:
        return LayerMethod(cfg), note
    m = ChunkedMethod(cfg, replace(spec.chunk, target_bits=r), runner.d)
    m.name, m.family = spec.name, spec.chunk.family
    if not m.plan["feasible"]:
        return LayerMethod(cfg), "infeasible chunk code"
    return m, note


# ── Systems measurements ────────────────────────────────────────────────────────────
#
# Fidelity (stated in every output row):
#   payload_bytes      MEASURED: the compressed context cache is materialised in its storage
#                      format (fp16 exact rows; bit-packed integer codes + fp16 scales/zeros;
#                      fp16 atoms) and the tensors' nbytes are summed.  analytic_bytes is the
#                      same quantity from the bit formulas (a consistency check).
#   sim_state_bytes    MEASURED allocator size of the reference harness' decode state (dense,
#                      dequantised, compute dtype): what THIS implementation holds, not a kernel.
#   peak_*_bytes       MEASURED torch.cuda.max_memory_allocated per phase (NaN on CPU).
#   *_s, tokens_per_s  MEASURED wall clock with device synchronisation; tokenisation timed
#                      separately; data loading excluded.  Harness timings measure the fp32
#                      reference implementation (no fused kernels); native_* / kvpress:* rows
#                      measure Hugging Face SDPA + DynamicCache (+ official press hooks).
#   attn_flops_*       ANALYTIC attention FLOPs per decoded token from the actual cache state
#                      (multiply-add = 2 FLOPs); flops_decode_step_measured is torch's
#                      FlopCounterMode over one full decode step (matmul-class ops only).

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


def _absmax_codes(xhat: torch.Tensor, bits: int):
    """Integer codes + fp16 scales of per-vector absmax quantisation (exact for fake_quant_tokens)."""
    qmax = 2 ** (bits - 1) - 1
    sc = xhat.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    return (xhat / sc).round().clamp(-qmax, qmax).to(torch.int64) + qmax, sc.half()


@torch.no_grad()
def payload_bytes(method: LayerMethod, k_all: Dict[int, Tuple[torch.Tensor, torch.Tensor]], context_len: int) -> int:
    """MEASURED bytes of the compressed context cache in its storage format (see header)."""
    total = 0
    c = method.cfg
    for li, st in method.states.items():
        if st.shared is not None:
            continue
        n = st.n
        live = (st.hide[:, :, :n] >= BIG) & (st.pos[:n] < context_len)[None, None]
        total += tensor_bytes(st.K[:, :, :n][live].half()) + tensor_bytes(st.V[:, :, :n][live].half())
        if isinstance(method, KIVIMethod) and li in k_all:
            k, v = k_all[li]
            total += kivi_payload_bytes(k, v, method.bits, c, context_len)
            continue
        if st.tok is not None:
            fin = torch.isfinite(st.tok["b"])
            tk, tv, tb = st.tok["k"][fin], st.tok["v"][fin], st.tok["b"][fin]
            if isinstance(method, ChunkedMethod) and method.m.family == "mean":
                atom = tb != 0                                     # mean atoms carry log n > 0
                total += tensor_bytes(tk[atom].half()) + tensor_bytes(tv[atom].half()) + tensor_bytes(tb[atom].half())
                tk, tv, tb = tk[~atom], tv[~atom], tb[~atom]
            if tk.numel():
                for x in (tk, tv):
                    if c.KEEP_BITS >= 16:
                        total += tensor_bytes(x.half())
                    else:
                        codes, sc = _absmax_codes(x, c.KEEP_BITS)
                        total += tensor_bytes(pack_codes(codes, c.KEEP_BITS)) + tensor_bytes(sc)
                if isinstance(method, ChunkedMethod) and method.m.family == "am":
                    total += tensor_bytes(tb.half())
        if st.mom is not None:
            fin = torch.isfinite(st.mom["logn"])
            for key_, x in st.mom.items():
                if isinstance(x, torch.Tensor) and key_ != "tau":
                    total += tensor_bytes(x[fin].half())
    return int(total)


def kivi_payload_bytes(k, v, bits, cfg, context_len) -> int:
    """Packed KIVI codes + fp16 scale/zero for the context rows read quantised at the horizon."""
    T = k.shape[2]
    n_q = max(0, min(T, context_len) - cfg.WINDOW - cfg.N_SINK)         # rows with n + W <= H - 1
    if n_q <= 0:
        return 0
    G = cfg.KIVI_GROUP
    B, H, _, d = k.shape
    rows = slice(cfg.N_SINK, cfg.N_SINK + n_q)
    kq, vq = k[:, :, rows].float(), v[:, :, rows].float()
    qmax = 2 ** bits - 1
    n_groups = int(math.ceil(n_q / G))
    pad = n_groups * G - n_q
    kg = F.pad(kq, (0, 0, 0, pad)).view(B, H, n_groups, G, d)
    lo, hi = kg.amin(3), kg.amax(3)                                     # per (group, channel)
    kcodes = ((kg - lo[:, :, :, None]) / ((hi - lo).clamp_min(1e-8)[:, :, :, None] / qmax)).round().clamp(0, qmax)
    gv = G if d % G == 0 else d
    vg = vq.reshape(B, H, n_q, d // gv, gv)
    vlo, vhi = vg.amin(-1), vg.amax(-1)                                 # per (token, channel group)
    vcodes = ((vg - vlo[..., None]) / ((vhi - vlo).clamp_min(1e-8)[..., None] / qmax)).round().clamp(0, qmax)
    kcodes = kcodes.reshape(B, H, n_groups * G, d)[:, :, :n_q]          # drop the padding rows
    return (tensor_bytes(pack_codes(kcodes, bits))
            + tensor_bytes(pack_codes(vcodes, bits))
            + tensor_bytes(lo.half()) + tensor_bytes(hi.half()) + tensor_bytes(vlo.half()) + tensor_bytes(vhi.half()))


def attn_flops_per_token(method: LayerMethod, Hq: int, d: int) -> float:
    """ANALYTIC attention FLOPs of one decoded token over the captured state (all layers)."""
    total = 0.0
    for li, st in method.states.items():
        src = method.states[st.shared] if st.shared is not None else st
        Hkv = src.K.shape[1]
        rep = Hq // Hkv
        n_exact = float((src.hide[:, :, :src.n] >= BIG).sum()) / src.K.shape[0]   # summed over KV heads
        n_tok = float(torch.isfinite(src.tok["b"]).sum()) / src.K.shape[0] if src.tok is not None else 0.0
        f = (4 * d + 5) * (n_exact + n_tok)                                # QK, PV, softmax
        if src.mom is not None:
            m = src.mom
            na = float(torch.isfinite(m["logn"]).sum()) / src.K.shape[0]
            r = m["U"].shape[-1] if "U" in m else (m["A"].shape[-1] if "A" in m else 0)
            # per query head and atom, counted from atom_read + the output sum (A^T A is
            # query-independent and is not charged): q.mu + log n, weight x mu_v, softmax
            per = 4 * d + 4
            if m["order"] == 2:
                per += 2 * r * d + 3 * d + 3 * r + 12                      # U^T q, q^2.diag, lam, phase, bound
            if m["tilt"]:
                per += 4 * r * d + d + 2 * r * r + 2 * r + 3               # B^T q, A (c tB), value-ball norm
            f += per * na
        total += rep * f
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


@torch.no_grad()
def generate(runner: "Runner", method: LayerMethod, ids: List[int], context_len: int, max_new: int,
             stop_ids: set, stop_fn=None, measure: bool = True) -> Tuple[List[int], Dict]:
    """
    Greedy decoding on the reference harness.  One single-pass forward over the prompt with
    compression frozen at context_len (state captured), then token-by-token steps that read
    the compressed state.  Returns systems measurements (see "Systems measurements").
    """
    P = len(ids)
    method.freeze_at, method.capture_extra = context_len, max_new + 1
    kv_keep = {}
    if measure and isinstance(method, KIVIMethod):            # raw K/V of the context for the payload
        orig = method.entries

        def entries_and_keep(li, q, k, v, scaling, pos, _o=orig):
            kv_keep[li] = (k[:, :, :context_len], v[:, :, :context_len])
            return _o(li, q, k, v, scaling, pos)
        method.entries = entries_and_keep
    _reset_peak()
    _sync()
    t0 = time.perf_counter()
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
            "prefill_s": prefill_s, "ttft_s": prefill_s, "peak_prefill_bytes": peak_prefill,
            "path": "harness", "model_dtype": str(next(runner.model.parameters()).dtype),
            "attention_dtype": "float32 (reference harness)"}
    if measure:
        info["payload_bytes"] = payload_bytes(method, kv_keep, context_len)
        info["sim_state_bytes"] = tensor_bytes([[st.K[:, :, :st.n], st.V[:, :, :st.n], st.tok, st.mom]
                                                for st in states.values() if st.shared is None])
        info["attn_flops_per_token"] = attn_flops_per_token(method, runner.Hq, runner.d)
        info["attn_flops_per_token_full"] = (4.0 * runner.d + 5) * P * runner.Hq * runner.L
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
            info["flops_decode_step_measured"] = measured_step_flops(step)
            for li, st in states.items():                           # undo the measured step's append
                st.n = snapshot[li]
        lg = step()
        nxt = int(lg.float().argmax())
    _sync()
    decode_s = time.perf_counter() - t1
    info.update(n_generated=len(out), decode_s=decode_s, peak_decode_bytes=_peak(),
                tokens_per_s=(len(out) - 1) / decode_s if len(out) > 1 and decode_s > 0 else float("nan"),
                seconds=prefill_s + decode_s)
    method.states = {}
    if "entries" in method.__dict__:
        del method.entries
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
            "memory_fraction_measured": measured_fraction,
            "attn_flops_per_token": (4.0 * runner.d + 5) * runner.Hq * (sum(lens) + runner.L * (len(ids) - context_len)),
            "attn_flops_per_token_full": (4.0 * runner.d + 5) * len(ids) * runner.Hq * runner.L}
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
                info["flops_decode_step_measured"] = measured_step_flops(
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
                seconds=ttft_s + decode_s, payload_bytes=float(cache_bytes))
    del cache
    free_memory()
    return out, info


def run_spec(runner, spec: BenchSpec, enc: Dict, max_new: int, stop: set, stop_fn, cfg) -> Tuple[List[int], Dict, str]:
    """One (sample, method) generation on the right path."""
    if spec.family == "native":
        gen, info = native_generate(runner, enc["ids"], enc["context_len"], max_new, stop, stop_fn)
        return gen, info, ""
    if spec.family == "kvpress":
        nominal = max(1, int(enc["context_len"] * spec.keep)) / enc["context_len"]
        try:
            press = build_press(spec.press, spec.keep)
            gen, info = native_generate(runner, enc["ids"], enc["context_len"], max_new, stop, stop_fn, press,
                                        nominal_keep=nominal if spec.press in MASKING_PRESSES else None)
        except Exception as e:                        # e.g. SnapKV asserts context > window
            return [], {"memory_fraction": float("nan"), "path": "native"}, f"kvpress error: {e!r}"[:200]
        return gen, info, ("masked keys: nominal memory" if spec.press in MASKING_PRESSES else "")
    method, note = make_method(spec, enc["context_len"], runner, cfg)
    gen, info = generate(runner, method, enc["ids"], enc["context_len"], max_new, stop, stop_fn)
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
    cache it must also match Hugging Face's own cached greedy generate().
    """
    rows = []
    harness_gen = {}
    for spec in specs:
        if spec.family in ("native", "kvpress"):
            continue
        m, note = make_method(spec, context_len, runner, cfg)
        gen, _ = generate(runner, m, ids, context_len, n_new, set())
        harness_gen[spec.name] = gen
        ref_m, _ = make_method(spec, context_len, runner, cfg)
        ref_m.freeze_at = context_len
        x = torch.tensor([ids + gen[:-1]], device=DEVICE)
        lg = runner.forward(x, ref_m, logits_to_keep=len(gen)).logits[0].float()
        agree = float((lg.argmax(-1).cpu() == torch.tensor(gen)).float().mean())
        rows.append({"method": spec.name, "argmax_agreement": agree, "n_tokens": len(gen), "note": note})
    full = [r for r in rows if r["method"] == "full"]
    if full:
        hf = runner.model.generate(torch.tensor([ids], device=DEVICE), max_new_tokens=n_new, do_sample=False,
                                   num_beams=1, eos_token_id=None, pad_token_id=0)[0, len(ids):].tolist()
        m, _ = make_method(BenchSpec("full", "full"), context_len, runner, cfg)
        gen, _ = generate(runner, m, ids, context_len, n_new, set())
        full[0]["hf_generate_agreement"] = float(np.mean([a == b for a, b in zip(gen, hf)]))
    for spec in specs:
        if spec.family not in ("native", "kvpress"):
            continue
        press = build_press(spec.press, spec.keep) if spec.family == "kvpress" else None
        gen, _ = native_generate(runner, ids, context_len, n_new, set(), press=press)
        row = {"method": spec.name, "n_tokens": len(gen), "note": "native path"}
        # harness counterpart: full (fp32 harness vs model-dtype SDPA) and the SnapKV /
        # StreamingLLM replicas vs the official presses (same kept set => same tokens)
        twin = "full" if spec.family == "native" else \
            {"snapkv": "snapkv", "streaming_llm": "streamingllm"}.get(spec.press, "") + f"@{spec.keep:g}"
        if twin in harness_gen:
            row["harness_agreement"] = float(np.mean([a == b for a, b in zip(gen, harness_gen[twin])]))
            row["harness_twin"] = twin
        row.update(official_pipeline_agreement(runner, ids, context_len, n_new, spec))
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
    if not kvpress_available():
        return {}
    from kvpress import KVPressTextGenerationPipeline
    pipe = KVPressTextGenerationPipeline(model=runner.model, tokenizer=runner.tokenizer)
    eos = runner.model.generation_config.eos_token_id
    eos = set(eos) if isinstance(eos, (list, tuple)) else ({int(eos)} if eos is not None else set())
    q = ids[context_len:]
    if not q:                                         # the official pipeline needs question tokens
        return {"official_pipeline_agreement": float("nan")}
    press = build_press(spec.press, spec.keep) if spec.family == "kvpress" else None
    ans = pipe._forward({"context_ids": torch.tensor([ids[:context_len]]),
                         "questions_ids": [torch.tensor([q])]}, max_new_tokens=n_new, press=press)[0]
    press = build_press(spec.press, spec.keep) if spec.family == "kvpress" else None
    gen, _ = native_generate(runner, ids, context_len, n_new, eos, press=press)
    mine = runner.tokenizer.decode(gen, skip_special_tokens=True)
    return {"official_pipeline_agreement": float(mine == ans)}


# ── Benchmark evaluation ────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_long_ppl(runner: "Runner", method: LayerMethod, seqs: List[List[int]], cfg: "Config") -> pd.DataFrame:
    """Streaming single pass per sequence (no freeze); NLL per position bin."""
    rows = []
    dec = runner.model.get_decoder()
    head = runner.model.get_output_embeddings()
    bins = list(cfg.LONG_PPL_BINS) + [BIG]
    for si, ids in enumerate(seqs):
        x = torch.tensor([ids], device=DEVICE)
        pos = torch.arange(x.shape[1], device=DEVICE)
        sim = Sim(method)
        sim.begin(pos)
        with routed(sim):
            h = dec(input_ids=x, position_ids=pos[None], use_cache=False).last_hidden_state[0]
        nll = torch.empty(x.shape[1] - 1)
        for s in range(0, x.shape[1] - 1, 1024):
            lg = head(h[s:min(s + 1024, x.shape[1] - 1)]).float()
            nll[s:s + lg.shape[0]] = F.cross_entropy(lg, x[0, s + 1:s + 1 + lg.shape[0]], reduction="none").cpu()
        tpos = torch.arange(1, x.shape[1])
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (tpos >= lo) & (tpos < hi)
            if sel.any():
                rows.append({"item": si, "bin": f"{lo}-{hi if hi < BIG else 'end'}",
                             "nll": float(nll[sel].sum()), "n_tok": int(sel.sum())})
        rows.append({"item": si, "bin": "all", "nll": float(nll.sum()), "n_tok": int(nll.numel())})
    return pd.DataFrame(rows)


def run_benchmarks(runner: "Runner", cfg: "Config", out_dir: str, suites: Sequence[str], synthetic=None) -> pd.DataFrame:
    """Runs the selected long-context suites; per-sample rows (score, measured memory, timing)."""
    os.makedirs(out_dir, exist_ok=True)
    specs = bench_specs(cfg)
    stop = stop_ids_for(runner)
    rows = []
    rows_path = os.path.join(out_dir, "bench_samples.csv")
    done = set()
    if cfg.RESUME and os.path.exists(rows_path):
        prev = pd.read_csv(rows_path)
        rows = prev.to_dict("records")
        done = {(r["benchmark"], r["task"], str(r["item"]), r["method"]) for r in rows}

    def flush():
        pd.DataFrame(rows).to_csv(rows_path, index=False)

    def run_items(bench, task, items, scorer, max_new_of, chat_ok):
        sub = items if not cfg.MAX_SAMPLES else [items[i] for i in sorted(
            _rng("sub", bench, task).choice(len(items), min(len(items), cfg.MAX_SAMPLES), replace=False))]
        pred_dir = os.path.join(out_dir, "pred", bench)
        for it in sub:
            max_new = max_new_of(it)
            t_tok = time.perf_counter()
            enc = encode_item(runner, it, max_new, cfg, chat_ok)
            tokenize_s = time.perf_counter() - t_tok
            first_line = task in LB_FIRST_LINE_TASKS and bench.startswith("longbench")

            def stop_fn(gen, _tok=runner.tokenizer):
                return first_line and "\n" in _tok.decode(gen, skip_special_tokens=True).lstrip("\n")
            for spec in specs:
                key = (bench, task, str(it["id"]), spec.name)
                if key in done:
                    continue
                gen, info, note = run_spec(runner, spec, enc, max_new, stop, stop_fn, cfg)
                pred = runner.tokenizer.decode(gen, skip_special_tokens=True)
                score = float("nan") if note.startswith("kvpress error") else scorer(it, pred)
                rows.append({"benchmark": bench, "task": task, "item": str(it["id"]), "method": spec.name,
                             "family": spec.family if spec.family != "chunk" else spec.chunk.family,
                             "keep_budget": spec.keep, "score": score, "note": note, "length": it.get("length"),
                             "truncated": enc["truncated"], "chat": enc["chat"], "tokenize_s": tokenize_s, **info})
                os.makedirs(os.path.join(pred_dir, spec.name), exist_ok=True)
                with open(os.path.join(pred_dir, spec.name, f"{task}.jsonl"), "a", encoding="utf-8") as f:
                    f.write(json.dumps({"pred": pred, "answers": it["answers"], "all_classes": it.get("all_classes"),
                                        "length": it.get("length")}, ensure_ascii=False) + "\n")
            flush()
            logger.info(f"    {bench}/{task} item {it['id']}: P={len(enc['ids'])} " + ", ".join(
                f"{r['method']}={r['score']:.0f}" for r in rows[-len(specs):] if r["item"] == str(it["id"])))

    if "longbench" in suites:
        bench = "longbench-e" if cfg.LONGBENCH_E else "longbench"
        for task in cfg.LONGBENCH_TASKS:
            items = synthetic["longbench"][task] if synthetic else load_longbench(task, cfg)
            logger.info(f"  {bench}/{task}: {len(items)} items")
            run_items(bench, task, items,
                      lambda it, pred, _t=task: longbench_sample_score(_t, pred, it["answers"], it.get("all_classes")),
                      lambda it, _t=task: LB_MAX_NEW_TOKENS[_t], chat_ok=task not in LB_NO_CHAT_TASKS)
    if "ruler" in suites:
        for length in cfg.RULER_LENGTHS:
            if length > runner.max_pos:
                logger.warning(f"  RULER {length} exceeds the model context ({runner.max_pos}); skipped")
                continue
            items = synthetic["ruler"] if synthetic else load_ruler(length, cfg)
            for task in sorted({it["task"] for it in items}):
                run_items(f"ruler-{length}", task, [it for it in items if it["task"] == task],
                          lambda it, pred: ruler_sample_score(it["task"], pred, it["answers"]),
                          lambda it: it["max_new_tokens"], chat_ok=True)
    if "longppl" in suites:
        for name in cfg.LONG_PPL_DATASETS:
            seqs = synthetic["longppl"] if synthetic else load_long_ppl(name, runner, cfg)
            if not seqs:
                continue
            L = len(seqs[0])
            for spec in bench_specs(cfg, for_ppl=True):
                if (f"longppl-{name}", "lm", "all", spec.name) in done:
                    continue
                method, note = make_method(spec, L, runner, cfg)
                t0 = time.time()
                df = eval_long_ppl(runner, method, seqs, cfg)
                if spec.family == "streamingllm":                    # streaming policy: keep tokens
                    mem = min(1.0, method.keep / L)
                else:                                                 # measured from the end state
                    method.capture_extra, method.freeze_at = 0, L
                    runner.forward(torch.tensor([seqs[0]], device=DEVICE), method, logits_to_keep=1)
                    mem = method.state_bits(runner.d, L)["total"] / (L * runner.L * runner.Hkv * 2 * runner.d * 16.0)
                    method.states, method.capture_extra = {}, None
                for r in df.itertuples():
                    rows.append({"benchmark": f"longppl-{name}", "task": r.bin, "item": str(r.item),
                                 "method": spec.name, "family": spec.family if spec.family != "chunk"
                                 else spec.chunk.family, "keep_budget": spec.keep, "score": -r.nll / r.n_tok,
                                 "nll": r.nll, "n_tok": r.n_tok, "note": note, "memory_fraction": mem,
                                 "context_len": L, "seconds": time.time() - t0})
                flush()
                g = df[df.bin == "all"]
                logger.info(f"    longppl-{name} {spec.name:<20} ppl={math.exp(g.nll.sum() / g.n_tok.sum()):.4f} "
                            f"memory={mem:.3f} {note}")
    flush()
    return pd.DataFrame(rows)


def bench_contrasts(cfg: "Config") -> List[Tuple[str, str, str]]:
    out = []
    for r in cfg.KEEP_FRACTIONS:
        t = f"{r:g}"
        out += [(f"tilt_vs_snapkv@{t}", f"tilt@{t}", f"snapkv@{t}"),
                (f"tilt_vs_streamingllm@{t}", f"tilt@{t}", f"streamingllm@{t}"),
                (f"tilt_vs_evict@{t}", f"tilt@{t}", f"evict@{t}"),
                (f"tilt_vs_mean@{t}", f"tilt@{t}", f"mean@{t}"),
                (f"tilt_vs_am@{t}", f"tilt@{t}", f"am@{t}")]
    t0 = f"{cfg.PRIMARY_KEEP:g}"
    out += [(f"freezing@{t0}", f"tilt@{t0}", f"tilt_nophase@{t0}"),
            (f"variance@{t0}", f"tilt@{t0}", f"moment1@{t0}")]
    out += [(f"tilt@{t0}_vs_kivi{b}", f"tilt@{t0}", f"kivi{b}") for b in cfg.KIVI_BITS]
    presses = [s_.press for s_ in bench_specs(cfg) if s_.family == "kvpress" and s_.keep == cfg.KEEP_FRACTIONS[0]]
    for r in cfg.KEEP_FRACTIONS:                   # official kvpress implementations (SOTA baselines)
        t = f"{r:g}"
        out += [(f"tilt_vs_kvpress:{pn}@{t}", f"tilt@{t}", f"kvpress:{pn}@{t}") for pn in presses]
        if "snapkv" in presses:                    # replica vs official SnapKV (harness check)
            out.append((f"{HARNESS_CHECK}snapkv@{t}", f"snapkv@{t}", f"kvpress:snapkv@{t}"))
    out.append((f"{HARNESS_CHECK}full", "full", "native_full"))
    return out


HARNESS_CHECK = "harness_check:"    # two-sided implementation checks; never enter decision rules


def paired_score_test(d: np.ndarray, strata: np.ndarray, n_boot: int, n_perm: int, seed: int) -> Dict:
    """
    Higher is better.  d = per-unit paired differences A - B; strata = task of each unit.
    The estimand is the benchmark's official aggregate: the MACRO mean over tasks of the
    per-task mean difference (LongBench / RULER average task scores).  CI: bootstrap that
    resamples units within each task; p: two-sided paired sign-flip test of the same statistic
    (exact under H0 of exchangeable signs within units).
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
SYSTEMS_COLS = ("tokenize_s", "prefill_s", "ttft_s", "decode_s", "tokens_per_s", "peak_prefill_bytes",
                "peak_decode_bytes", "payload_bytes", "analytic_bytes", "sim_state_bytes", "cache_bytes_measured",
                "full_cache_bytes", "attn_flops_per_token", "attn_flops_per_token_full",
                "flops_decode_step_measured", "memory_fraction", "memory_fraction_measured")


def bench_analysis(df: pd.DataFrame, cfg: "Config", model: str, family: str, out_dir: str) -> pd.DataFrame:
    """
    Task scores (official aggregation: per-task mean, macro average over tasks; LongBench-E
    also per length bucket), systems measurements, and pre-specified paired contrasts.
    A contrast uses only BUDGET-COMPLIANT units: at a shared budget both methods store
    <= KEEP (1 + BUDGET_TOL) on that sample; against an unbudgeted code (KIVI) memory_A <=
    memory_B (1 + BUDGET_TOL).  No method wins by holding more cache; excluded units are
    counted and the all-unit estimate is reported beside it.  Undefined scores are dropped.
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
    agg = pd.concat(parts, ignore_index=True)
    agg.insert(0, "model", model)
    agg.to_csv(os.path.join(out_dir, "bench_task_scores.csv"), index=False)

    sys_cols = [c_ for c_ in SYSTEMS_COLS if c_ in df]
    if sys_cols:
        gen = df[~df.benchmark.str.startswith("longppl")]
        sysd = gen.groupby(["benchmark", "method"], as_index=False)[sys_cols].median(numeric_only=True)
        if "path" in gen:
            sysd = sysd.merge(gen.groupby(["benchmark", "method"], as_index=False).path.first(), on=["benchmark", "method"])
        sysd.insert(0, "model", model)
        sysd["statistic"] = "median over samples"
        sysd.to_csv(os.path.join(out_dir, "bench_systems.csv"), index=False)

    rows = []
    for bench, g in df.groupby("benchmark"):
        lm = bench.startswith("longppl")
        g = g[g.task == "all"] if lm else g
        g = g.assign(unit=g.task.astype(str) + "/" + g["item"].astype(str))
        for name, A, B in bench_contrasts(cfg):
            ga, gb = g[g.method == A], g[g.method == B]
            if ga.empty or gb.empty:
                continue
            m = ga[["unit", "task", "score", "memory_fraction", "keep_budget"]].merge(
                gb[["unit", "score", "memory_fraction", "keep_budget"]], on="unit", suffixes=("_a", "_b"))
            check = name.startswith(HARNESS_CHECK)
            tol = 1 + cfg.BUDGET_TOL
            if check:
                ok = np.ones(len(m), bool)
            elif m.keep_budget_a.notna().all() and m.keep_budget_b.notna().all():   # same budget: both within it
                ok = ((m.memory_fraction_a <= m.keep_budget_a * tol + 1e-9) &
                      (m.memory_fraction_b <= m.keep_budget_b * tol + 1e-9)).to_numpy()
            else:                                       # vs an unbudgeted code (KIVI): A no larger than B
                ok = (m.memory_fraction_a <= m.memory_fraction_b * tol + 1e-9).to_numpy()
            d = (m.score_a - m.score_b).to_numpy()
            strata = m.task.astype(str).to_numpy()
            r = paired_score_test(d[ok], strata[ok], cfg.N_BOOT, cfg.N_PERM, SEED)
            r_all = paired_score_test(d, strata, 1, 1, SEED)
            rows.append({"model": model, "model_family": family, "benchmark": bench, "contrast": name, "A": A,
                         "B": B, "role": "check" if check else "hypothesis",
                         "memory_A": float(m.memory_fraction_a[ok].mean()) if ok.any() else float("nan"),
                         "memory_B": float(m.memory_fraction_b[ok].mean()) if ok.any() else float("nan"),
                         "score_A": float(m.score_a[ok].mean()) if ok.any() else float("nan"),
                         "score_B": float(m.score_b[ok].mean()) if ok.any() else float("nan"),
                         "n_excluded_over_budget": int((~ok).sum()), "estimate_all_units": r_all["estimate"], **r})
    ct = pd.DataFrame(rows)
    if len(ct):
        ct["p_holm"] = np.nan
        hyp = ct.role == "hypothesis"
        for bench, idx in ct[hyp].groupby("benchmark").groups.items():
            ct.loc[idx, "p_holm"] = holm(ct.loc[idx, "p_value"])
        ct["supported"] = hyp & (ct.estimate > 0) & (ct.p_holm < cfg.ALPHA) & (ct.n_units > 0)
        ct.to_csv(os.path.join(out_dir, "bench_contrasts.csv"), index=False)
        for r in ct[~hyp].itertuples():
            logger.info(f"  {r.benchmark} {r.contrast}: diff={r.estimate:+.3f} p={r.p_value:.3g} (two-sided check)")
    return ct


def bench_cross_model(cfg: "Config"):
    cts = [pd.read_csv(p) for p in Path(cfg.RESULTS_DIR).glob("*/bench/bench_contrasts.csv")]
    if not cts:
        return
    ct = pd.concat(cts, ignore_index=True)
    ct.to_csv(os.path.join(cfg.OUTPUT_DIR, "bench_all_contrasts.csv"), index=False)
    if "role" in ct:                                        # harness checks never enter decisions
        ct = ct[ct.role == "hypothesis"]
    sysd = [pd.read_csv(p) for p in Path(cfg.RESULTS_DIR).glob("*/bench/bench_systems.csv")]
    if sysd:
        pd.concat(sysd, ignore_index=True).to_csv(os.path.join(cfg.OUTPUT_DIR, "bench_all_systems.csv"), index=False)
    scores = [pd.read_csv(p) for p in Path(cfg.RESULTS_DIR).glob("*/bench/bench_task_scores.csv")]
    if scores:
        pd.concat(scores, ignore_index=True).to_csv(os.path.join(cfg.OUTPUT_DIR, "bench_all_task_scores.csv"), index=False)
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
    dr.to_csv(os.path.join(cfg.OUTPUT_DIR, "bench_decision_rules.csv"), index=False)
    for r in dr.itertuples():
        logger.info(f"BENCH DECISION {r.benchmark:<14} {r.contrast:<28} {r.verdict:<20} "
                    f"{r.models_meeting_rule}/{r.models_evaluated} median={r.median_estimate:+.3f}")


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
    eq_specs = [s for s in bench_specs(cfg) if s.name in
                ("full", "kivi2", f"snapkv@{cfg.PRIMARY_KEEP:g}", f"streamingllm@{cfg.PRIMARY_KEEP:g}",
                 f"tilt@{cfg.PRIMARY_KEEP:g}", f"am@{cfg.PRIMARY_KEEP:g}", f"mean@{cfg.PRIMARY_KEEP:g}",
                 "native_full", f"kvpress:snapkv@{cfg.PRIMARY_KEEP:g}", f"kvpress:streaming_llm@{cfg.PRIMARY_KEEP:g}")]
    generation_equivalence(runner, eq_specs, enc["ids"], enc["context_len"], 8, cfg).to_csv(
        os.path.join(out_dir, "generation_equivalence.csv"), index=False)
    if synthetic:                      # positive control: gold answers must score 100 end to end
        for task, items in synthetic["longbench"].items():
            for it in items:
                sc = longbench_sample_score(task, it["answers"][0], it["answers"], it.get("all_classes"))
                if abs(sc - 100.0) > 1e-9:
                    raise RuntimeError(f"gold answer scored {sc} on {task}")
        for it in synthetic["ruler"]:
            if ruler_sample_score(it["task"], " ".join(it["answers"]), it["answers"]) != 100.0:
                raise RuntimeError("gold answer did not score 100 on RULER")
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
    model = AutoModelForCausalLM.from_config(conf, attn_implementation=TILT_ATTN)
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
                 "longppl": [tok(text_sample(60)).input_ids[:600] for _ in range(3)]}
    return model, tok, synthetic


def main():
    ap = argparse.ArgumentParser(description="TiltKV: exponential-family compaction of the KV cache")
    ap.add_argument("--suite", nargs="+", default=["core"], choices=["core", "longbench", "ruler", "longppl"],
                    help="core = WikiText-2/LAMBADA/passkey + theory tests; longbench / ruler / longppl = "
                         "long-context benchmarks with generation from the compressed cache")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=None, help="LongBench tasks (default: Config.LONGBENCH_TASKS)")
    ap.add_argument("--longbench-e", action="store_true")
    ap.add_argument("--ruler-lengths", nargs="*", type=int, default=None)
    ap.add_argument("--keep", nargs="*", type=float, default=None, help="memory budgets (fractions of fp16 cache)")
    ap.add_argument("--max-samples", type=int, default=None, help="per task")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="benchmark method name prefixes, e.g. tilt@ snapkv@ streamingllm@ (full always runs)")
    ap.add_argument("--query-aware", action="store_true", help="compress the question with the context")
    ap.add_argument("--long-ppl-len", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-passkey", action="store_true")
    ap.add_argument("--no-lambada", action="store_true")
    a = ap.parse_args()
    kw = {}
    if a.output:
        kw["OUTPUT_DIR"] = a.output
    if a.tasks:
        kw["LONGBENCH_TASKS"] = tuple(a.tasks)
    if a.ruler_lengths:
        kw["RULER_LENGTHS"] = tuple(a.ruler_lengths)
    if a.keep:
        kw["KEEP_FRACTIONS"] = tuple(a.keep)
        kw["PRIMARY_KEEP"] = sorted(a.keep)[len(a.keep) // 2]
    if a.long_ppl_len:
        kw["LONG_PPL_LEN"] = a.long_ppl_len
    kw.update(MAX_SAMPLES=a.max_samples, QUERY_AWARE=a.query_aware, LONGBENCH_E=a.longbench_e)
    if a.methods:
        kw["BENCH_METHODS"] = tuple(a.methods)
    if a.smoke:
        kw.update(OUTPUT_DIR=a.output or str(PROJECT_ROOT / "tilt_smoke"), SMOKE=True, CHUNK=32, WINDOW=8,
                  W_OBS=8, KIVI_GROUP=8, TARGET_BITS=(6.0, 3.0, 2.0), PRIMARY_BITS=3.0, PPL_WINDOW=192,
                  PPL_STRIDE=96, PPL_MAX_WINDOWS=24, DIAG_WINDOWS=4, N_BOOT=500, N_PERM=2000,
                  DECODE_TEST_LEN=128, RUN_LAMBADA=False, RUN_PASSKEY=False, MIN_MODELS_WILCOXON=2,
                  RESUME=False, AM_ITERS=100, KEEP_FRACTIONS=(0.5, 0.25), PRIMARY_KEEP=0.25,
                  MAX_CONTEXT_TOKENS=1500, LONG_PPL_BINS=(0, 128, 256), RULER_LENGTHS=(1024,),
                  LONG_PPL_DATASETS=("synthetic",))
    cfg = Config(**kw)
    if a.no_resume:
        cfg.RESUME = False
    cfg.RUN_PASSKEY &= not a.no_passkey
    cfg.RUN_LAMBADA &= not a.no_lambada
    fh = logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, "tilt_kv.log"))
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.json"), "w") as f:
        json.dump({k: ([asdict(m) for m in v] if k in ("MODELS", "BENCH_MODELS") else v)
                   for k, v in cfg.__dict__.items()}, f, indent=2, default=str)
    with open(os.path.join(cfg.OUTPUT_DIR, "theory_tests.json"), "w") as f:
        json.dump(theory_tests(), f, indent=2)
    metric_tests()
    bench_suites = [x for x in a.suite if x != "core"]
    summaries = []
    if cfg.SMOKE:
        if "core" in a.suite:
            for mc, prebuilt in smoke_setup(cfg):
                summaries.append(run_model(mc, cfg, prebuilt))
        if bench_suites:
            run_bench_model(ModelConfig("tiny-char-llama", "random-init", "Llama", 1), cfg, bench_suites,
                            smoke_bench_setup(cfg))
    else:
        catalog = {m.name: m for m in cfg.MODELS + cfg.BENCH_MODELS}
        if a.models:
            missing = [m for m in a.models if m not in catalog]
            if missing:
                raise SystemExit(f"unknown model(s) {missing}; add them to Config.MODELS / BENCH_MODELS")
        if "core" in a.suite:
            for mc in ([catalog[m] for m in a.models] if a.models else cfg.MODELS):
                try:
                    summaries.append(run_model(mc, cfg))
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"{mc.name}: out of memory ({e}); skipped")
                    free_memory()
        if bench_suites:
            for mc in ([catalog[m] for m in a.models] if a.models else cfg.BENCH_MODELS):
                try:
                    run_bench_model(mc, cfg, bench_suites)
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"{mc.name}: out of memory in benchmarks ({e}); skipped")
                    free_memory()
    cross_model(cfg, [s_ for s_ in summaries if s_ is not None])
    if bench_suites:
        bench_cross_model(cfg)


if __name__ == "__main__":
    main()
