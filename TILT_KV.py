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

Usage
    python TILT_KV.py --smoke                       # offline CPU test (tiny models)
    python TILT_KV.py --models Llama-3.2-1B         # real model (GPU, Hugging Face access)
Requires torch >= 2.4, transformers >= 4.56 (tested on 5.19), scipy, pandas,
datasets (not with --smoke), matplotlib (optional).

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
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
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

    # ── Statistics ────────────────────────────────────────────────────
    N_BOOT: int = 2000
    N_PERM: int = 10000
    ALPHA: float = 0.05
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


def tilt_attention(module, query, key, value, attention_mask, **kwargs):
    ctl = Router.active
    if ctl is None:
        return _SDPA(module, query, key, value, attention_mask, **kwargs)
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
    if method.selection == "random":
        share = torch.as_tensor(_rng("randsel", *key).random((B, H, C)), device=k.device, dtype=k.dtype)
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


def chunk_plan(T: int, cfg: "Config") -> List[Tuple[int, int, int]]:
    """(start, end, tau) of every chunk that is compacted before the end of the sequence."""
    out, s = [], cfg.N_SINK
    while s + cfg.CHUNK - 1 < T:
        e = s + cfg.CHUNK - 1
        tau = e + cfg.WINDOW
        if tau > T - 1:
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
    q (B, Hq, Tq, d) -> out (B, Tq, Hq, d).
    """
    B, Hq, Tq, d = q.shape
    rep = Hq // K.shape[1]
    Kr, Vr = rep_heads(K.float(), rep), rep_heads(V.float(), rep)
    dev = q.device
    kv_pos, hide_at = kv_pos.to(dev), hide_at.to(dev)
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
        readable = (kv_pos[None, :] <= qp[:, None]) & (qp[:, None] < hide_at[None, :])
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

class LayerMethod:
    name = "full"
    family = "global"

    def __init__(self, cfg: "Config"):
        self.cfg = cfg

    def bits_per_element(self, d: int) -> float:
        return 16.0

    def exact_tokens(self) -> int:
        """Tokens kept exact beyond the compacted region (worst case), for memory fractions."""
        return 0

    def begin(self):
        pass

    def layer(self, li, q, k, v, scaling, pos) -> torch.Tensor:
        n = k.shape[2]
        return chunked_attention(q, scaling, pos, k, v, pos, torch.full((n,), BIG, dtype=torch.long),
                                 None, None, self.cfg.ATTN_CHUNK)


class KIVIMethod(LayerMethod):
    family = "kivi"

    def __init__(self, cfg, bits: int):
        super().__init__(cfg)
        self.bits, self.name = bits, f"kivi{bits}"

    def bits_per_element(self, d):
        return self.cfg.kivi_effective_bits(self.bits)

    def exact_tokens(self):
        return self.cfg.N_SINK + self.cfg.WINDOW

    def layer(self, li, q, k, v, scaling, pos):
        c = self.cfg
        kh = kivi_fake_quant(k, self.bits, c.KIVI_GROUP, True, c.N_SINK)
        vh = kivi_fake_quant(v, self.bits, c.KIVI_GROUP, False, c.N_SINK)
        n = k.shape[2]
        # exact for sinks and the window; quantised copies as token atoms visible elsewhere is
        # expressed directly: exact tokens hidden once outside the window, quantised copy shown.
        hide = torch.where(pos < c.N_SINK, torch.full_like(pos, BIG), pos + c.WINDOW).cpu()
        tok = {"k": kh, "v": vh, "b": torch.zeros(kh.shape[:3], device=k.device), "tau": hide.clone()}
        tok["b"][:, :, pos.cpu() < c.N_SINK] = float("-inf")
        return chunked_attention(q, scaling, pos, k, v, pos, hide, tok, None, c.ATTN_CHUNK)


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
        self.cache = {}

    def layer(self, li, q, k, v, scaling, pos):
        if li in set(self.map.values()):
            self.cache[li] = (k, v)
        if li in self.map:
            k, v = self.cache[self.map[li]]
        return super().layer(li, q, k, v, scaling, pos)


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

    def build(self, li, q, k, v, scaling, plan_T):
        """Finalise every chunk of this layer (single pass); returns atoms and hide_at."""
        c = self.cfg
        Hkv = k.shape[1]
        n = k.shape[2]
        hide = torch.full((n,), BIG, dtype=torch.long)
        parts = []
        for ci, (s, e, tau) in enumerate(plan_T):
            qref = group_queries(q[:, :, max(0, tau - c.W_OBS + 1): tau + 1].float() * scaling, Hkv)
            res = finalize_chunk(k[:, :, s:e + 1].float(), v[:, :, s:e + 1].float(), qref, self.m,
                                 self.plan, c.KEEP_BITS, c, (li, ci))
            if self.record and res["tok"] is not None and res["tok"].get("idx") is not None:
                self.kept[(li, ci)] = res["tok"]["idx"].cpu()
            if res["tok"] is not None and "use_atoms" in res["tok"]:
                self.atom_use.append(float(res["tok"]["use_atoms"].float().mean()))
            parts.append((res, tau))
            hide[s:e + 1] = tau
        return concat_atoms(parts, "tok"), concat_atoms(parts, "mom"), hide

    def layer(self, li, q, k, v, scaling, pos):
        plan_T = chunk_plan(k.shape[2], self.cfg)
        tok, mom, hide = self.build(li, q, k, v, scaling, plan_T)
        return chunked_attention(q, scaling, pos, k, v, pos, hide, tok, mom, self.cfg.ATTN_CHUNK)


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


def main():
    ap = argparse.ArgumentParser(description="TiltKV: exponential-family compaction of the KV cache")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--output", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--no-passkey", action="store_true")
    ap.add_argument("--no-lambada", action="store_true")
    a = ap.parse_args()
    kw = {}
    if a.output:
        kw["OUTPUT_DIR"] = a.output
    if a.smoke:
        kw.update(OUTPUT_DIR=a.output or str(PROJECT_ROOT / "tilt_smoke"), SMOKE=True, CHUNK=32, WINDOW=8,
                  W_OBS=8, KIVI_GROUP=8, TARGET_BITS=(6.0, 3.0, 2.0), PRIMARY_BITS=3.0, PPL_WINDOW=192,
                  PPL_STRIDE=96, PPL_MAX_WINDOWS=24, DIAG_WINDOWS=4, N_BOOT=500, N_PERM=2000,
                  DECODE_TEST_LEN=128, RUN_LAMBADA=False, RUN_PASSKEY=False, MIN_MODELS_WILCOXON=2,
                  RESUME=False, AM_ITERS=100)
    cfg = Config(**kw)
    if a.no_resume:
        cfg.RESUME = False
    cfg.RUN_PASSKEY &= not a.no_passkey
    cfg.RUN_LAMBADA &= not a.no_lambada
    fh = logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, "tilt_kv.log"))
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.json"), "w") as f:
        json.dump({k: ([asdict(m) for m in v] if k == "MODELS" else v) for k, v in cfg.__dict__.items()},
                  f, indent=2, default=str)
    with open(os.path.join(cfg.OUTPUT_DIR, "theory_tests.json"), "w") as f:
        json.dump(theory_tests(), f, indent=2)
    summaries = []
    if cfg.SMOKE:
        for mc, prebuilt in smoke_setup(cfg):
            summaries.append(run_model(mc, cfg, prebuilt))
    else:
        for mc in cfg.MODELS:
            if a.models and mc.name not in a.models:
                continue
            try:
                summaries.append(run_model(mc, cfg))
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"{mc.name}: out of memory ({e}); skipped")
                free_memory()
    cross_model(cfg, [s for s in summaries if s is not None])


if __name__ == "__main__":
    main()
