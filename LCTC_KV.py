"""
LCTC_KV.py — Loss-aware, Closed-loop Transform Coding of the KV cache (training-free).

This script follows the training-free cross-layer KV-sharing study (KV_LDT_v12_2.py).
It turns that study's negative results into design constraints for a compression method
and tests the method with the same statistical discipline: exact negative controls,
paired randomisation tests, Holm correction and fixed cross-model decision rules.

----------------------------------------------------------------------------------------
1. WHAT THE PRIOR STUDY FOUND (merged.zip, 7 models) AND WHAT IT IMPLIES
----------------------------------------------------------------------------------------
  E1  Coordinate mismatch.  Adjacent-layer K/V have high linear CKA (median 0.85-0.93)
      but a raw relative distance ||[K;V]_s - [K;V]_t|| / ||[K;V]_t|| of 1.37-1.52,
      i.e. ~sqrt(2): in raw coordinates neighbouring caches are about as far apart as
      two unrelated vectors of equal norm.  CKA is invariant to rotation/isotropic
      scaling; substitution is not.  The layers SHARE information but not coordinates.
  E2  Substitution is catastrophic.  Training-free CLA-2 (first 20% exempt) raises
      WikiText-2 PPL from 7.4-11.0 to 86-3133 while saving only ~39% of the cache.
  E3  Partner selection is uninformative.  CKA, KV distance, query-aware fidelity and
      a KVSharer-style search all perform like random layer sets (H3b/H3d/H3e,
      percentile-rank tests).  Similarity proxies do not predict functional damage.
  E4  Representation/generation dissociation.  A retrained probe recovers lexical
      status that a frozen readout loses, while generation collapses (answer mass lost
      in 7/7 models, H8b).  The information is still present but moved, so the readout
      that matters (the downstream layers / the LM loss) is what must be protected.
  E5  Early layers are fragile; damage grows monotonically with the number of
      substituted layers (H1d, rho = 1.0).

  Implications: (i) replace substitution by an optimal *linear* prediction of a
  layer's cache from the previous layer's reconstructed cache (E1, E2); (ii) never
  select partners by similarity; measure distortion in units of the LM loss instead
  (E3, E4); (iii) let a single rate-distortion allocation decide how many bits each
  layer receives, so fragile layers get more (E5).

----------------------------------------------------------------------------------------
2. CLOSEST PRIOR WORK (verify before claiming novelty; field moves monthly)
----------------------------------------------------------------------------------------
  * AQUA-KV (Shutova et al., ICML 2025, arXiv:2501.19392): linear inter-layer
    predictors trained on RECONSTRUCTED inputs + residual quantisation (HIGGS);
    K_l <- K_{l-1}, V_l <- (V_{l-1}, K_l).  Our predictor wiring follows AQUA-KV and
    is NOT claimed as novel.
  * EchoKV (2026, arXiv:2603.22910): group anchor + fine-tuned linear predictor.
  * xKV (Chang et al., 2025, arXiv:2503.18893): shared cross-layer SVD subspace.
  * SVDq (2025), KV-COBRA (2026), JoLT (2026), RDKV (2026), RateQuant (2026),
    "KV cache compression through the lens of transform coding" (2026): KLT/SVD +
    rate/rank allocation.  KLT + reverse water-filling is textbook (Cover & Thomas,
    Thm 10.3.3) and is NOT claimed as novel.
  * KVQuant (Hooper et al., 2024): pre-RoPE keys, Fisher-weighted non-uniform
    datatypes, dense-and-sparse outliers.  Palu (2025): low-rank latent, Fisher ranks.
  * RoPE-exact complex per-frequency PCA was tested publicly and lost to pre-RoPE PCA
    (github.com/PseudoHunt/rope-kv-compression, 2.69x output error at 25% budget):
    key low-rankness is mostly cross-frequency.  We therefore code keys PRE-RoPE.

----------------------------------------------------------------------------------------
3. RESEARCH GAPS ADDRESSED (each is an ablation with a pre-specified contrast)
----------------------------------------------------------------------------------------
  G1  The objective of the transform.  KV codecs choose their basis by reconstruction
      MSE (SVD/KLT/whitening) or attention error and use loss sensitivity, at most,
      to allocate bits.  E3/E4 say local similarity is the wrong yardstick.
      -> Fisher-metric KLT: the basis itself minimises the second-order increase of
         the LM loss, dL ~= 1/2 e^T F e, with per-(layer, KV-head) empirical Fisher
         blocks F (K-FAC-style block-diagonal approximation; Martens & Grosse 2015).
  G2  K and V in incommensurable units.  Existing methods split the budget between
      keys and values heuristically ("more bits for keys") or with separate targets.
      -> Because every distortion is in loss units, K and V components of all
         layers compete in ONE Lagrangian allocation (common slope theta;
         Shoham & Gersho 1988) under one memory budget.
  G3  Calibration/inference distribution shift that compounds with depth.  Codecs are
      calibrated on caches of the UNCOMPRESSED model, but at inference layer l sees
      hidden states perturbed by the compression of layers < l (prior study:
      hidden-state cosine falls to 0.5-0.8 under sharing).
      -> Propagation-aware sequential calibration: layer l's predictor, basis,
         clipping and bits are fitted on caches produced while layers < l already run
         compressed (GPTQ-style sequential calibration, applied to a KV codec).
  G4  Simulation vs deployment.  Many papers evaluate "fake-quantised" prefill caches
      that no streaming decoder can produce.
      -> A single-pass masked simulation that is mathematically identical to
         autoregressive decoding with a compressed cache + attention sinks + an exact
         recent window, and a test that checks it against a real streaming decoder
         that stores integer codes.
  G5  Evidence standards.  Every comparison is paired (same windows/items), tested
      by sign-flip randomisation or exact McNemar tests, Holm-corrected, and decided
      across models by a fixed rule, mirroring KV_LDT_v12_2.  The second-order loss
      model behind G1/G2 is itself tested: predicted dNLL vs measured dNLL.

  What is claimed as possibly novel (to be checked by a targeted literature search):
  the COMBINATION G1+G2+G3 in one closed-form, training-free codec, and the
  streaming-exact evaluation G4.  Nothing here is claimed as the first use of
  inter-layer prediction, KLT, Fisher information or bit allocation.

  Not addressed (and why): training-time memory.  The KV cache is an inference
  structure.  During training, K/V are a small share of the activation memory saved
  for backward (~2*d_kv per token per layer vs ~34*d_model for a standard block,
  Korthikanti et al. 2022; a few percent for GQA models).  Activation checkpointing,
  ZeRO/FSDP and low-precision optimiser states are the effective tools there.

----------------------------------------------------------------------------------------
4. THE CODEC (per layer l, per token; keys handled pre-RoPE)
----------------------------------------------------------------------------------------
      x_l       = pre-RoPE key (or value) row of all KV heads, x in R^n, n = H_kv * d
      z_l       = [K^_{l-1}, V^_{l-1}]  (keys)   |   [K^_{l-1}, V^_{l-1}, K^_l]  (values)
      p_l       = P_l (z_l - mu_z)                 LMMSE (ridge) prediction, closed loop
      y_l       = (x_l - mu_l - p_l) S_l           S_l = blockdiag_h F_{l,h}^{1/2}
      a_l       = y_l U_l                          KLT of the whitened residual
      a^_l      = Q_{b,c}(a_l)                     per-component static uniform quantiser
      x^_l      = mu_l + p_l + a^_l U_l^T S_l^{-1}
    Bits b_k come from  argmin_b D_k(b) + theta * b  with D_k the measured
    (operational) distortion on calibration data; theta is searched so the average
    rate meets the budget.  Rows whose whitened residual is extreme (calibrated
    quantile) are stored exactly and their cost is charged to the method.
    Attention sinks (first N_SINK positions) and an exact recent window (WINDOW)
    are kept in 16-bit for EVERY method compared.

Usage
    python LCTC_KV.py --smoke                      # offline CPU test (tiny random models)
    python LCTC_KV.py                              # all models in Config.MODELS
    python LCTC_KV.py --models Llama-3.2-1B Qwen2.5-1.5B
Requires torch >= 2.4, transformers >= 4.56 (tested on 5.19), scipy, pandas, datasets
(not needed with --smoke), matplotlib (optional, figures).

Validation status (be explicit in any write-up)
  Verified with --smoke only (CPU; two tiny Llama / SmolLM3-NoPE models trained on a
  synthetic copy task): harness identities (RoPE inverse, simulation == SDPA), rate
  control, streaming-decode equivalence (argmax 100%, max |dlogit| < 1e-3; ~1-2% of
  (layer, token) rows differ by a floating-point bin flip), statistics and decision
  rules.  On the toy, LCTC at ~1 bit beat KIVI-2 (3 bits) and CLA, but G1, G2 and G3
  showed NO measurable benefit, and the second-order model ranked configurations well
  (Spearman 0.74-0.90) while under-predicting dNLL by 20-70x.  No real-LLM result exists
  yet: every claim above is a hypothesis for the full run to accept or reject.
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
# GLOBALS: logging, seeds, device
# ════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(os.environ.get("KV_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
SEED = 42
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("lctc_kv")


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


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in keys])


# Reference values from KV_LDT_v12_2 (merged.zip, WikiText-2, window 1024 / stride 512):
# a reproduction anchor for the evaluation harness (no_reuse) and for CLA (rf2_ex20 `full`).
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
class Variant:
    """One LCTC configuration.  Ablations switch off exactly one component."""
    name: str
    budget: float                  # average bits per cached K/V element (16 = uncompressed)
    metric: str = "fisher"         # "fisher" (loss units, G1) | "euclid" (MSE units)
    joint_kv: bool = True          # one Lagrangian allocation over K and V (G2)
    xlayer: bool = True            # inter-layer prediction from layer l-1
    propagation: bool = True       # propagation-aware sequential calibration (G3)

    def __post_init__(self):
        if self.metric not in ("fisher", "euclid"):
            raise ValueError(f"unknown metric {self.metric!r}")
        if self.metric == "euclid" and self.joint_kv:
            raise ValueError("a joint K/V allocation needs commensurable (loss) units")


@dataclass
class Config:
    OUTPUT_DIR: str = str(Path(os.environ.get("KV_LCTC_OUTPUT_DIR", PROJECT_ROOT / "lctc_results")))
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

    # ── Cache layout shared by EVERY method (fair comparison) ─────────
    N_SINK: int = 4                 # attention-sink positions kept in 16 bit (StreamingLLM)
    WINDOW: int = 32                # most recent positions (incl. the current one) kept exact

    # ── Calibration (WikiText-2 train; disjoint from the test split) ──
    CALIB_DATASET: Tuple[str, str, str] = ("Salesforce/wikitext", "wikitext-2-raw-v1", "train")
    CALIB_SEQ_LEN: int = 512
    CALIB_N_SEQ: int = 128          # ~65k coded tokens for the sequential fit
    CALIB_BATCH: int = 8
    SEARCH_TOKENS: int = 16384      # rows per layer for the open-loop theta search
    FIT_TOKENS: int = 65536         # rows for the per-layer sequential fit
    DIAG_ROWS: int = 4096           # rows for CKA diagnostics (as CALIB_MAX_ROWS in v12_2)
    FISHER_N_SEQ: int = 32
    FISHER_BATCH: int = 2
    FISHER_SHRINK: float = 0.05     # F <- (1-e) F + e tr(F)/d I  (per KV head)
    RIDGE_GRID: Tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)
    BITS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 8)
    CLIP_MULTS: int = 24            # clip grid: geomspace(0.5, 8) x RMS
    RD_ROWS: int = 8192
    THETA_ITERS: int = 22
    BUDGET_TOL: float = 0.02        # achieved rate within +-2% of the target
    SECANT_STEPS: int = 2
    OUTLIER_QUANTILE: Optional[float] = 0.999   # rows above it are stored exactly (charged)

    # ── Methods ───────────────────────────────────────────────────────
    KIVI_BITS: Tuple[int, ...] = (2, 4)
    KIVI_GROUP: int = 32            # fp16 scale + zero per group -> +32/G bits/element
    BUDGETS: Tuple[float, ...] = (1.5, 2.0, 3.0, 5.0)
    PRIMARY_BUDGET: float = 3.0     # = effective rate of KIVI-2 with G = 32
    CLA_REUSE: int = 2
    CLA_EXEMPT: float = 0.20

    # ── Evaluation ────────────────────────────────────────────────────
    PPL_WINDOW: int = 1024          # KV_LDT_v12_2 protocol
    PPL_STRIDE: int = 512
    PPL_MAX_WINDOWS: int = 200
    LAMBADA_N: int = 500
    PASSKEY_LENGTHS: Tuple[int, ...] = (2048, 4096)
    PASSKEY_DEPTHS: Tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
    PASSKEY_REPS: int = 4
    ATTN_CHUNK: int = 256
    DECODE_TEST_LEN: int = 96
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
        if self.WINDOW < 1:
            raise ValueError("WINDOW >= 1: the current token's own K/V is always exact at decode")
        if self.PRIMARY_BUDGET not in self.BUDGETS:
            raise ValueError("PRIMARY_BUDGET must be one of BUDGETS")
        if self.BITS[0] != 0 or max(self.BITS) > 8:
            raise ValueError("BITS must start at 0 and stay <= 8 (uint8 codes)")
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, "results")
        os.makedirs(self.RESULTS_DIR, exist_ok=True)

    def variants(self) -> List[Variant]:
        b0 = self.PRIMARY_BUDGET
        out = [Variant(f"lctc@{b:g}", b) for b in self.BUDGETS]
        out += [Variant(f"lctc_no_xlayer@{b0:g}", b0, xlayer=False),
                Variant(f"lctc_sep_kv@{b0:g}", b0, joint_kv=False),
                Variant(f"lctc_euclid_sep@{b0:g}", b0, metric="euclid", joint_kv=False),
                Variant(f"lctc_open_loop@{b0:g}", b0, propagation=False)]
        return out

    def kivi_effective_bits(self, bits: int) -> float:
        return bits + 32.0 / self.KIVI_GROUP


# ════════════════════════════════════════════════════════════════════════════
# RoPE GEOMETRY (HF rotate_half convention: dims i and i + d/2 form one complex pair)
# ════════════════════════════════════════════════════════════════════════════

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_apply(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """y = x e^{j phi} per complex pair (cos, sin already carry the attention scaling)."""
    return x * cos + rotate_half(x) * sin


def rope_invert(y: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Exact inverse of rope_apply: x = y conj(c) / |c|^2 with c = cos + j sin."""
    return (y * cos - rotate_half(y) * sin) / (cos * cos + sin * sin)


# ════════════════════════════════════════════════════════════════════════════
# ATTENTION ROUTING
# ════════════════════════════════════════════════════════════════════════════
#
# Models are loaded with attn_implementation=LCTC_ATTN.  With no active controller the
# call is exactly SDPA.  The (key, value) tensors received are post-norm, post-RoPE and
# pre-GQA-repeat: precisely what a KV cache stores.

LCTC_ATTN = "lctc_router"
_SDPA = ALL_ATTENTION_FUNCTIONS["sdpa"]


class Router:
    active = None


def lctc_attention(module, query, key, value, attention_mask, **kwargs):
    ctl = Router.active
    if ctl is None:
        return _SDPA(module, query, key, value, attention_mask, **kwargs)
    return ctl.attend(module, query, key, value, attention_mask, kwargs)


AttentionInterface.register(LCTC_ATTN, lctc_attention)
AttentionMaskInterface.register(LCTC_ATTN, sdpa_mask)


@contextmanager
def routed(controller):
    previous = Router.active
    Router.active = controller
    try:
        yield controller
    finally:
        Router.active = previous


class StopForward(Exception):
    """Raised by a controller once the layer it needs has been captured (early exit)."""


def masked_attention(q, k, v, k_hat, v_hat, scaling: float, n_sink: int, window: int,
                     q_pos: torch.Tensor, kv_pos: torch.Tensor, chunk: int) -> torch.Tensor:
    """
    Causal attention in which visible key n is read EXACT for query m iff
    kv_pos[n] < n_sink or q_pos[m] - kv_pos[n] < window, and from (k_hat, v_hat)
    otherwise (k_hat None = exact everywhere).

    Streaming equivalence (G4): in autoregressive decoding, token m attends to the exact
    K/V of the sinks and of the last `window` tokens and to the decoded cache of every
    other earlier token; the K/V of token n at every layer are a deterministic function
    of tokens <= n and are encoded when token n is processed.  This masked single pass
    computes, for every m, exactly that attention, and every hidden state of token n is
    computed from tokens <= n under the same rule.  Hence the single pass equals
    streaming decoding (up to floating-point reassociation; see decode_equivalence).
    q (b, Hq, Tq, d); k, v, k_hat, v_hat (b, Hkv, Tk, d).  Returns (b, Tq, Hq, d).
    """
    b, Hq, Tq, d = q.shape
    rep = Hq // k.shape[1]

    def expand(x):
        return None if x is None else x.float().repeat_interleave(rep, dim=1)

    K, V, Kh, Vh = expand(k), expand(v), expand(k_hat), expand(v_hat)
    out = torch.empty((b, Hq, Tq, d), dtype=torch.float32, device=q.device)
    for s in range(0, Tq, chunk):
        qs = q[:, :, s:s + chunk].float() * scaling
        qp = q_pos[s:s + chunk]
        visible = kv_pos[None, :] <= qp[:, None]
        scores = qs @ K.transpose(-1, -2)
        if Kh is not None:
            exact = (kv_pos[None, :] < n_sink) | ((qp[:, None] - kv_pos[None, :]) < window)
            scores = torch.where(exact, scores, qs @ Kh.transpose(-1, -2))
        p = scores.masked_fill(~visible, float("-inf")).softmax(-1)
        if Kh is None:
            out[:, :, s:s + chunk] = p @ V
        else:
            out[:, :, s:s + chunk] = p.masked_fill(~exact, 0.0) @ V + p.masked_fill(exact, 0.0) @ Vh
    return out.transpose(1, 2).to(q.dtype).contiguous()


# ════════════════════════════════════════════════════════════════════════════
# QUANTISERS
# ════════════════════════════════════════════════════════════════════════════

def uq_codes(a: torch.Tensor, bits: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    """Symmetric mid-rise uniform quantiser on [-clip, clip], 2^bits levels per column."""
    levels = torch.pow(2.0, bits.to(a.dtype))
    step = 2.0 * clip / levels
    idx = torch.floor((a + clip) / step).clamp_(min=0.0)
    return torch.minimum(idx, levels - 1.0)


def uq_values(idx: torch.Tensor, bits: torch.Tensor, clip: torch.Tensor) -> torch.Tensor:
    levels = torch.pow(2.0, bits.to(clip.dtype))
    step = 2.0 * clip / levels
    return -clip + (idx.to(clip.dtype) + 0.5) * step


@torch.no_grad()
def operational_rd(A: torch.Tensor, bits_set: Sequence[int], n_mults: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Operational rate-distortion table of every column of A (rows = calibration tokens):
    D[k, j] = min over clip c of mean (a_k - Q_{b_j, c}(a_k))^2, with the clip searched on
    a geometric grid of multiples of the column RMS (handles heavy tails empirically).
    D[k, 0] = mean a_k^2 (component dropped).  Returns D (n, nb) and the clips (n, nb).
    """
    n = A.shape[1]
    rms = A.pow(2).mean(0).sqrt().clamp_min(1e-20)
    D = torch.empty((n, len(bits_set)), dtype=torch.float64, device=A.device)
    C = torch.zeros((n, len(bits_set)), dtype=A.dtype, device=A.device)
    D[:, 0] = A.pow(2).mean(0).double()
    mults = torch.logspace(math.log10(0.5), math.log10(8.0), n_mults, device=A.device, dtype=A.dtype)
    for j, b in enumerate(bits_set[1:], 1):
        bits = torch.full((n,), float(b), device=A.device, dtype=A.dtype)
        best = torch.full((n,), float("inf"), dtype=torch.float64, device=A.device)
        bestc = torch.zeros(n, dtype=A.dtype, device=A.device)
        for m in mults:
            clip = m * rms
            err = (A - uq_values(uq_codes(A, bits, clip), bits, clip)).pow(2).mean(0).double()
            better = err < best
            best = torch.where(better, err, best)
            bestc = torch.where(better, clip, bestc)
        D[:, j], C[:, j] = best, bestc
    return D, C


def kivi_fake_quant(x: torch.Tensor, bits: int, group: int, along_tokens: bool, n_sink: int) -> torch.Tensor:
    """
    KIVI (Liu et al., 2024): keys per channel over groups of `group` tokens, values per
    token over groups of `group` channels; asymmetric min/max, b bits.  Sinks are left
    out of the groups (they are exact for every method).  x (b, H, T, d).
    """
    y = x.float().clone()
    qmax = 2 ** bits - 1

    def q(t, dim):
        lo, hi = t.amin(dim, keepdim=True), t.amax(dim, keepdim=True)
        scale = (hi - lo).clamp_min(1e-8) / qmax
        return ((t - lo) / scale).round().clamp(0, qmax) * scale + lo

    T, d = y.shape[2], y.shape[3]
    if along_tokens:
        for s in range(n_sink, T, group):
            y[:, :, s:s + group] = q(y[:, :, s:s + group], 2)
    else:
        g = group if d % group == 0 else d
        body = y[:, :, n_sink:]
        shp = body.shape
        y[:, :, n_sink:] = q(body.reshape(*shp[:-1], d // g, g), -1).reshape(shp)
    return y


# ════════════════════════════════════════════════════════════════════════════
# THE CODEC
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class Part:
    """Coder of one tensor (K or V) of one layer, acting on rows x in R^n."""
    mu: torch.Tensor                       # (n,)
    S: torch.Tensor                        # (n, n) metric square root (whitening)
    E: torch.Tensor                        # (n, r) analysis  = S U[:, active]
    Fd: torch.Tensor                       # (r, n) synthesis = U[:, active]^T S^{-1}
    bits: torch.Tensor                     # (r,)  bits of the active components (>= 1)
    clip: torch.Tensor                     # (r,)
    tau: float                             # rows with ||(x - mu - p) S||^2 > tau are exact
    mu_z: Optional[torch.Tensor] = None    # (m,)
    P: Optional[torch.Tensor] = None       # (m, n) predictor

    def predict(self, z: Optional[torch.Tensor]):
        if self.P is None or z is None:
            return 0.0
        return (z - self.mu_z) @ self.P

    def encode(self, x: torch.Tensor, z: Optional[torch.Tensor]):
        pred = self.predict(z)
        r = x - self.mu - pred
        flag = (r @ self.S).pow(2).sum(-1) > self.tau
        idx = uq_codes(r @ self.E, self.bits, self.clip)
        x_hat = self.mu + pred + uq_values(idx, self.bits, self.clip) @ self.Fd
        x_hat = torch.where(flag[:, None], x, x_hat)
        return idx.to(torch.uint8), flag, x_hat

    def decode(self, idx: torch.Tensor, z: Optional[torch.Tensor], flag: torch.Tensor, exact: torch.Tensor):
        x_hat = self.mu + self.predict(z) + uq_values(idx, self.bits, self.clip) @ self.Fd
        return torch.where(flag[:, None], exact, x_hat)

    @property
    def rate(self) -> int:
        return int(self.bits.sum())

    def n_params(self) -> int:
        return sum(int(t.numel()) for t in (self.mu, self.S, self.E, self.Fd, self.bits, self.clip,
                                            self.mu_z, self.P) if t is not None)


@dataclass
class LayerCodec:
    k: Part
    v: Part
    xlayer: bool                           # K/V of layer l-1 are predictor inputs

    def _zk(self, prev):
        return torch.cat(prev, -1) if (self.xlayer and prev is not None) else None

    def _zv(self, prev, xk_hat):
        return torch.cat([*prev, xk_hat], -1) if (self.xlayer and prev is not None) else xk_hat

    def encode(self, xk, xv, prev):
        ck, fk, xk_hat = self.k.encode(xk, self._zk(prev))
        cv, fv, xv_hat = self.v.encode(xv, self._zv(prev, xk_hat))
        return (ck, fk, xk_hat), (cv, fv, xv_hat)

    def decode(self, ck, fk, ek, cv, fv, ev, prev):
        xk_hat = self.k.decode(ck, self._zk(prev), fk, ek)
        xv_hat = self.v.decode(cv, self._zv(prev, xk_hat), fv, ev)
        return xk_hat, xv_hat


def metric_sqrt(blocks: Optional[torch.Tensor], n: int, shrink: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Metric square root S = blockdiag_h F_h^{1/2} and its inverse, with shrinkage toward the
    head's scaled identity (Ledoit-Wolf-style) so noisy Fisher directions cannot dominate.
    blocks None -> Euclidean metric.
    """
    if blocks is None:
        eye = torch.eye(n, device=device)
        return eye, eye
    H, d, _ = blocks.shape
    S = torch.zeros((n, n), device=device, dtype=torch.float64)
    Si = torch.zeros_like(S)
    for h in range(H):
        Fh = blocks[h].to(device, torch.float64)
        Fh = 0.5 * (Fh + Fh.T)
        Fh = (1 - shrink) * Fh + shrink * torch.trace(Fh) / d * torch.eye(d, device=device, dtype=torch.float64)
        w, U = torch.linalg.eigh(Fh)
        w = w.clamp_min(max(float(w.max()), 1e-300) * 1e-10)
        S[h * d:(h + 1) * d, h * d:(h + 1) * d] = (U * w.sqrt()) @ U.T
        Si[h * d:(h + 1) * d, h * d:(h + 1) * d] = (U / w.sqrt()) @ U.T
    return S.float(), Si.float()


def ridge_fit(Z: torch.Tensor, X: torch.Tensor, grid: Sequence[float], alpha: Optional[float] = None):
    """
    LMMSE predictor of X from Z with ridge chosen on an interleaved 20% hold-out.  The
    LMMSE error covariance is minimal in the Loewner order, so the same predictor is
    optimal for ANY quadratic error metric (it does not depend on S).  Returns
    (mu_z, P, alpha, holdout_fvu).
    """
    mu_z = Z.mean(0)
    Zc = Z - mu_z
    m = Zc.shape[1]
    # contiguous last 20% of rows: rows are grouped by sequence, so this holds out whole
    # sequences instead of tokens adjacent to training rows
    hold = torch.arange(Zc.shape[0], device=Z.device) >= int(0.8 * Zc.shape[0])

    def solve(Zs, Xs, a):
        G = Zs.T @ Zs
        G.diagonal().add_(a * torch.trace(G) / m + 1e-12)
        return torch.linalg.solve(G, Zs.T @ Xs)

    fvu = float("nan")
    if alpha is None:
        best = (float("inf"), grid[0])
        den = (X[hold] - X[~hold].mean(0)).pow(2).sum()
        for a in grid:
            Pa = solve(Zc[~hold], X[~hold], a)
            err = float((X[hold] - Zc[hold] @ Pa).pow(2).sum() / den)
            if err < best[0]:
                best = (err, a)
        fvu, alpha = best
    return mu_z, solve(Zc, X, alpha), alpha, fvu


@torch.no_grad()
def fit_part(X: torch.Tensor, Z: Optional[torch.Tensor], S: torch.Tensor, Si: torch.Tensor,
             theta: float, cfg: Config, alpha: Optional[float] = None):
    """Fit one coder on calibration rows X (N, n) with predictor inputs Z (N, m) or None."""
    mu = X.mean(0)
    Xc = X - mu
    info = {"alpha": float("nan"), "holdout_fvu": float("nan")}
    if Z is not None:
        mu_z, P, alpha, fvu = ridge_fit(Z, Xc, cfg.RIDGE_GRID, alpha)
        pred = (Z - mu_z) @ P
        info.update(alpha=alpha, holdout_fvu=fvu)
    else:
        mu_z, P, pred = None, None, 0.0
    Y = (Xc - pred) @ S
    C = (Y.T @ Y / Y.shape[0]).double()
    lam, U = torch.linalg.eigh(0.5 * (C + C.T))
    order = torch.argsort(lam, descending=True)
    U = U[:, order].float()
    A = Y @ U
    rows = torch.linspace(0, A.shape[0] - 1, min(cfg.RD_ROWS, A.shape[0]), device=A.device).long()
    D, Cl = operational_rd(A[rows], cfg.BITS, cfg.CLIP_MULTS)
    bits_set = torch.tensor(cfg.BITS, dtype=torch.float64, device=A.device)
    j = torch.argmin(D + theta * bits_set[None, :], dim=1)
    bits_all = bits_set[j]
    active = bits_all > 0
    bits = bits_all[active].float()
    clip = Cl[torch.arange(len(j), device=A.device), j][active]
    E = (S @ U)[:, active]
    Fd = (U.T @ Si)[active]
    norms = Y.pow(2).sum(-1)
    tau = float(torch.quantile(norms[rows].float(), cfg.OUTLIER_QUANTILE)) if cfg.OUTLIER_QUANTILE else float("inf")
    part = Part(mu, S, E, Fd, bits, clip, tau, mu_z, P)
    flag = norms > tau
    x_hat = mu + pred + uq_values(uq_codes(A[:, active], bits, clip), bits, clip) @ Fd
    x_hat = torch.where(flag[:, None], X, x_hat)
    info.update(rate=int(bits.sum()), rank=int(active.sum()),
                D=float(D[torch.arange(len(j), device=A.device), j].sum()),
                var_in=float((Xc @ S).pow(2).sum(-1).mean()), var_res=float(norms.mean()),
                outlier_rate=float(flag.float().mean()))
    return part, x_hat, info


# ════════════════════════════════════════════════════════════════════════════
# CONTROLLERS AND COMPRESSORS
# ════════════════════════════════════════════════════════════════════════════

class Compressor:
    """compress(li, k, v, cos, sin) -> (k_hat, v_hat) post-RoPE, or None (layer exact)."""
    full_substitution = False      # True: the returned K/V replace the layer's own everywhere

    def reset(self):
        pass

    def compress(self, li, k, v, cos, sin):
        return None

    def extra_bits(self) -> float:
        return 0.0


class KIVICompressor(Compressor):
    def __init__(self, bits: int, cfg: Config):
        self.bits, self.cfg = bits, cfg

    def compress(self, li, k, v, cos, sin):
        c = self.cfg
        return (kivi_fake_quant(k, self.bits, c.KIVI_GROUP, True, c.N_SINK).to(k.dtype),
                kivi_fake_quant(v, self.bits, c.KIVI_GROUP, False, c.N_SINK).to(v.dtype))


class CLACompressor(Compressor):
    """Training-free cross-layer sharing exactly as `full` in KV_LDT_v12_2 (all positions)."""
    full_substitution = True

    def __init__(self, source_map: Dict[int, int]):
        self.map, self.sources = source_map, set(source_map.values())
        self.cache = {}

    def reset(self):
        self.cache = {}

    def compress(self, li, k, v, cos, sin):
        if li in self.sources:
            self.cache[li] = (k, v)
        src = self.map.get(li)
        return None if src is None else self.cache[src]


def cla_source_map(L: int, reuse: int, exempt: float) -> Dict[int, int]:
    ec = min(int(math.ceil(L * exempt)), L - 1)
    return {t: ec + ((t - ec) // reuse) * reuse for t in range(ec, L) if (t - ec) % reuse}


class LCTCCompressor(Compressor):
    """Applies fitted LayerCodecs in layer order; keeps the previous layer's reconstruction."""

    def __init__(self, codecs: Dict[int, LayerCodec], rope: List[bool], record: bool = False):
        self.codecs, self.rope, self.record = codecs, rope, record
        self.rows = 0
        self.flagged = defaultdict(int)          # part -> count of exact (outlier) rows
        self.reset()

    def reset(self):
        self.prev = None
        self.codes = {}

    def compress(self, li, k, v, cos, sin):
        codec = self.codecs.get(li)
        if codec is None:
            return None
        b, H, T, d = k.shape
        kp = rope_invert(k.float(), cos, sin) if cos is not None else k.float()
        xk = kp.transpose(1, 2).reshape(b * T, H * d)
        xv = v.float().transpose(1, 2).reshape(b * T, H * d)
        (ck, fk, xk_hat), (cv, fv, xv_hat) = codec.encode(xk, xv, self.prev)
        self.prev = (xk_hat, xv_hat)
        self.rows += b * T
        self.flagged["k"] += int(fk.sum())
        self.flagged["v"] += int(fv.sum())
        if self.record:
            self.codes[li] = (ck, cv, fk, fv)
        kh = xk_hat.view(b, T, H, d).transpose(1, 2)
        if cos is not None:
            kh = rope_apply(kh, cos, sin)
        vh = xv_hat.view(b, T, H, d).transpose(1, 2)
        return kh.to(k.dtype), vh.to(v.dtype)

    def extra_bits(self) -> float:
        """Measured cost of exact outlier rows, in bits per K/V element (16 bit + 32-bit index)."""
        if not self.rows or not self.codecs:
            return 0.0
        n = next(iter(self.codecs.values())).k.mu.numel()
        per_row = (16.0 * n + 32.0) / n         # bits per element of an exact row (+ its index)
        # self.rows counts (layer, token) rows; K and V each have one row per (layer, token)
        return per_row * (self.flagged["k"] + self.flagged["v"]) / (2.0 * self.rows)


class Capture:
    """
    Stores, at selected rows, the pre-RoPE K and V of the requested layers, the
    compressor's reconstruction of the previous layer (predictor inputs) and, for a
    few rows, post-RoPE K/V (diagnostics).
    """

    def __init__(self, layers, dtype=torch.float32, store_prev=False, post_rows=0):
        self.layers, self.dtype, self.store_prev, self.post_rows = set(layers), dtype, store_prev, post_rows
        self.select = None
        self.data = defaultdict(lambda: defaultdict(list))
        self.n_post = defaultdict(int)

    def take(self, li, k, v, cos, sin, prev):
        b, H, T, d = k.shape
        sel = self.select

        def flat(x):
            return x.transpose(1, 2).reshape(b, T, H * d)[sel]

        kp = rope_invert(k.float(), cos, sin) if cos is not None else k.float()
        rec = self.data[li]
        rec["xk"].append(flat(kp).to(self.dtype).cpu())
        rec["xv"].append(flat(v.float()).to(self.dtype).cpu())
        if self.store_prev and prev is not None:
            rec["pk"].append(prev[0].view(b, T, -1)[sel].to(self.dtype).cpu())
            rec["pv"].append(prev[1].view(b, T, -1)[sel].to(self.dtype).cpu())
        if self.n_post[li] < self.post_rows:
            pk = flat(k.float())[: self.post_rows - self.n_post[li]]
            pv = flat(v.float())[: pk.shape[0]]
            rec["post"].append(torch.cat([pk, pv], -1).cpu())
            self.n_post[li] += pk.shape[0]

    def get(self, li, key) -> Optional[torch.Tensor]:
        chunks = self.data[li].get(key)
        return torch.cat(chunks) if chunks else None


class Sim:
    """
    Routes every attention call of one forward.  With a compressor, the layer reads its
    reconstructed cache outside sinks + window (masked_attention); `shadow` computes the
    reconstruction (so later layers receive the right predictor inputs) but attends
    exactly, which is the open-loop calibration of ablation G3.
    """

    def __init__(self, runner: "Runner", compressor: Optional[Compressor] = None, shadow: bool = False,
                 capture: Optional[Capture] = None, stop_after: Optional[int] = None):
        self.runner, self.compressor, self.shadow = runner, compressor, shadow
        self.capture, self.stop_after = capture, stop_after

    def begin(self, positions: torch.Tensor):
        self.pos = positions
        self.cos, self.sin = self.runner.cos_sin(positions)
        if self.compressor is not None:
            self.compressor.reset()

    def attend(self, module, q, k, v, mask, kwargs):
        li = module.layer_idx
        if getattr(module, "sliding_window", None):
            raise RuntimeError("sliding-window attention layers are not supported")
        cos, sin = (self.cos, self.sin) if self.runner.rope[li] else (None, None)
        if self.capture is not None and li in self.capture.layers:
            self.capture.take(li, k, v, cos, sin, getattr(self.compressor, "prev", None))
        if self.stop_after is not None and li >= self.stop_after:
            raise StopForward
        k_hat = v_hat = None
        if self.compressor is not None:
            res = self.compressor.compress(li, k, v, cos, sin)
            if res is not None:
                if self.compressor.full_substitution:
                    k, v = res
                elif not self.shadow:
                    k_hat, v_hat = res
        scaling = kwargs.get("scaling") or module.scaling
        c = self.runner.cfg
        return masked_attention(q, k, v, k_hat, v_hat, scaling, c.N_SINK, c.WINDOW, self.pos, self.pos,
                                c.ATTN_CHUNK), None


class FisherProbe:
    """Adds zero leaves to pre-RoPE K and to V so that backward yields dL/dK_pre, dL/dV."""

    def __init__(self, runner: "Runner"):
        self.runner = runner
        self.delta = {}

    def begin(self, positions):
        self.cos, self.sin = self.runner.cos_sin(positions)
        self.delta = {}

    def attend(self, module, q, k, v, mask, kwargs):
        li = module.layer_idx
        dk = torch.zeros_like(k, requires_grad=True)
        dv = torch.zeros_like(v, requires_grad=True)
        self.delta[li] = (dk, dv)
        rk = rope_apply(dk.float(), self.cos, self.sin).to(k.dtype) if self.runner.rope[li] else dk
        return _SDPA(module, q, k + rk, v + dv, mask, **kwargs)


class StreamingLCTC:
    """
    Real streaming decoder for G4: per layer it stores uint8 codes of every token, exact
    K/V only for sinks, the recent window and outlier rows, and decodes the rest of the
    cache from codes at every step (layer order carries the previous-layer reconstruction).
    """

    def __init__(self, runner: "Runner", codecs: Dict[int, LayerCodec]):
        self.runner, self.codecs, self.cfg = runner, codecs, runner.cfg
        L = runner.L
        self.codes = [[] for _ in range(L)]          # per token: (ck, cv, fk, fv)
        self.outlier = [dict() for _ in range(L)]    # token -> (xk, xv) exact pre-RoPE rows
        self.exact = [dict() for _ in range(L)]      # token -> (k, v) post-RoPE, sinks + window

    def begin(self, t: int):
        self.t = t
        self.cos, self.sin = self.runner.cos_sin(torch.tensor([t], device=DEVICE))
        self.cur_prev = None
        self.past_prev = None

    def attend(self, module, q, k, v, mask, kwargs):
        li, t, c = module.layer_idx, self.t, self.cfg
        codec = self.codecs[li]
        rope = self.runner.rope[li]
        H, d = k.shape[1], k.shape[3]
        kp = rope_invert(k.float(), self.cos, self.sin) if rope else k.float()
        xk, xv = kp.transpose(1, 2).reshape(1, H * d), v.float().transpose(1, 2).reshape(1, H * d)
        (ck, fk, xk_hat), (cv, fv, xv_hat) = codec.encode(xk, xv, self.cur_prev)
        self.cur_prev = (xk_hat, xv_hat)
        self.codes[li].append((ck, cv, fk, fv))
        if bool(fk[0]) or bool(fv[0]):
            self.outlier[li][t] = (xk, xv)
        self.exact[li][t] = (k, v)
        for p in [p for p in self.exact[li] if p >= c.N_SINK and t - p >= c.WINDOW]:
            del self.exact[li][p]                    # left the window: only codes remain
        comp = list(range(c.N_SINK, t - c.WINDOW + 1))
        keys, vals, pos = [], [], []
        if comp:
            ck_, cv_, fk_, fv_ = (torch.cat([self.codes[li][p][i] for p in comp]) for i in range(4))
            n = H * d
            ek = torch.zeros((len(comp), n), device=DEVICE)
            ev = torch.zeros_like(ek)
            for j, p in enumerate(comp):
                if p in self.outlier[li]:
                    ek[j], ev[j] = self.outlier[li][p][0][0], self.outlier[li][p][1][0]
            xk_c, xv_c = codec.decode(ck_, fk_, ek, cv_, fv_, ev, self.past_prev)
            self.past_prev = (xk_c, xv_c)
            kc = xk_c.view(1, len(comp), H, d).transpose(1, 2)
            if rope:
                cs, sn = self.runner.cos_sin(torch.tensor(comp, device=DEVICE))
                kc = rope_apply(kc, cs, sn)
            keys.append(kc.to(k.dtype))
            vals.append(xv_c.view(1, len(comp), H, d).transpose(1, 2).to(v.dtype))
            pos += comp
        ex = sorted(self.exact[li])
        keys.append(torch.cat([self.exact[li][p][0] for p in ex], 2))
        vals.append(torch.cat([self.exact[li][p][1] for p in ex], 2))
        pos += ex
        K, V = torch.cat(keys, 2), torch.cat(vals, 2)
        order = torch.argsort(torch.tensor(pos))
        K, V = K[:, :, order.to(K.device)], V[:, :, order.to(V.device)]
        kv_pos = torch.tensor(sorted(pos), device=DEVICE)
        scaling = kwargs.get("scaling") or module.scaling
        return masked_attention(q, K, V, None, None, scaling, c.N_SINK, c.WINDOW,
                                torch.tensor([t], device=DEVICE), kv_pos, c.ATTN_CHUNK), None

    def stored_bytes(self) -> Dict[str, float]:
        codes = sum(int(x.numel()) for layer in self.codes for tok in layer for x in tok[:2])
        exact = sum(int(k.numel() + v.numel()) * 2 for layer in self.exact for k, v in layer.values())
        outl = sum(int(a.numel() + b.numel()) * 2 for layer in self.outlier for a, b in layer.values())
        packed = sum(float(self.codecs[li].k.rate + self.codecs[li].v.rate) / 8.0 * len(self.codes[li])
                     for li in range(len(self.codes)))
        return {"codes_bytes_uint8": codes, "codes_bytes_packed": packed, "exact_bytes": exact,
                "outlier_bytes": outl}


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
            model = AutoModelForCausalLM.from_pretrained(mc.model_id, attn_implementation=LCTC_ATTN,
                                                         dtype=COMPUTE_DTYPE, low_cpu_mem_usage=True)
        self.tokenizer, self.model = tokenizer, model.to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        dec = self.model.get_decoder()
        self.layers, self.rotary = dec.layers, dec.rotary_emb
        self.L = len(self.layers)
        conf = self.model.config
        self.Hq, self.Hkv = conf.num_attention_heads, conf.num_key_value_heads
        self.d = self.layers[0].self_attn.head_dim
        self.n = self.Hkv * self.d
        self.max_pos = int(getattr(conf, "max_position_embeddings", 4096))
        for i, layer in enumerate(self.layers):
            if getattr(layer.self_attn, "layer_idx", None) != i:
                raise RuntimeError(f"{mc.name}: layer {i} has no matching self_attn.layer_idx")
        self.rope = [bool(getattr(layer.self_attn, "use_rope", True)) for layer in self.layers]
        cos, _ = self.cos_sin(torch.arange(2, device=DEVICE))
        if cos.shape[-1] != self.d:
            raise RuntimeError(f"{mc.name}: partial rotary ({cos.shape[-1]} of {self.d} dims) not supported")
        self.bos = bos_prefix(tokenizer)
        logger.info(f"  layers={self.L} Hq={self.Hq} Hkv={self.Hkv} d={self.d} "
                    f"nope_layers={[i for i, r in enumerate(self.rope) if not r]}")

    def cos_sin(self, positions: torch.Tensor):
        x = torch.zeros(1, device=DEVICE, dtype=torch.float32)
        cos, sin = self.rotary(x, positions.to(DEVICE)[None])
        return cos[0].float(), sin[0].float()

    def forward(self, ids: torch.Tensor, ctl, **kw):
        pos = torch.arange(ids.shape[1], device=DEVICE)
        if ctl is None:
            return self.model(input_ids=ids, use_cache=False, **kw)
        ctl.begin(pos)
        with routed(ctl):
            return self.model(input_ids=ids, position_ids=pos[None].expand(ids.shape[0], -1),
                              use_cache=False, **kw)

    def release(self):
        del self.model
        free_memory()


# ════════════════════════════════════════════════════════════════════════════
# CALIBRATION: Fisher, captures, theta search, propagation-aware sequential fit
# ════════════════════════════════════════════════════════════════════════════

def batches(x: torch.Tensor, size: int):
    for i in range(0, x.shape[0], size):
        yield i, x[i:i + size]


def row_selection(n_seq: int, T: int, n_sink: int, budget: int, seq_offset: int, b: int, key: str) -> torch.Tensor:
    """Deterministic selection of non-sink positions, independent of the batch plan."""
    q = min(1.0, budget / max(1, n_seq * (T - n_sink)))
    sel = torch.zeros((b, T), dtype=torch.bool)
    for j in range(b):
        u = _rng(key, seq_offset + j).random(T) < q
        u[:n_sink] = False
        sel[j] = torch.as_tensor(u)
    return sel.to(DEVICE)


@torch.no_grad()
def run_capture(runner: Runner, seqs: torch.Tensor, capture: Capture, budget: int, key: str,
                compressor=None, shadow=False, stop_after=None):
    sim = Sim(runner, compressor, shadow, capture, stop_after)
    for i, ids in batches(seqs, runner.cfg.CALIB_BATCH):
        capture.select = row_selection(seqs.shape[0], seqs.shape[1], runner.cfg.N_SINK, budget, i, ids.shape[0], key)
        try:
            runner.forward(ids.to(DEVICE), sim, logits_to_keep=1)
        except StopForward:
            pass


def compute_fisher(runner: Runner, seqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    """
    Empirical Fisher of the summed next-token NLL w.r.t. pre-RoPE keys and values, per
    (layer, KV head) d x d block, averaged over non-sink token positions:
        F_{l,h} = (1/N) sum_t g_t g_t^T,  g_t = dL/dx_{l,h,t}.
    Then  dNLL/token ~= 1/2 E_t[e_t^T F e_t]  (second order; Fisher ~ Gauss-Newton;
    cross-token and cross-layer Hessian terms neglected — tested by P5).
    """
    c = runner.cfg
    L, H, d = runner.L, runner.Hkv, runner.d
    Fk = torch.zeros((L, H, d, d), dtype=torch.float64, device=DEVICE)
    Fv = torch.zeros_like(Fk)
    count = 0
    probe = FisherProbe(runner)
    for _, ids in batches(seqs[: c.FISHER_N_SEQ], c.FISHER_BATCH):
        ids = ids.to(DEVICE)
        with torch.enable_grad():
            logits = runner.forward(ids, probe).logits.float()
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
                                   reduction="sum")
            loss.backward()
        for li, (dk, dv) in probe.delta.items():
            for Fm, g in ((Fk, dk.grad), (Fv, dv.grad)):
                g = g[:, :, c.N_SINK:].double().permute(1, 0, 2, 3).reshape(H, -1, d)
                Fm[li] += torch.einsum("hnd,hne->hde", g, g)
        count += ids.shape[0] * (ids.shape[1] - c.N_SINK)
        probe.delta = {}
        del logits, loss
        free_memory()
    Fk /= count
    Fv /= count
    prof = pd.DataFrame({"layer": range(L),
                         "fisher_trace_k": [float(Fk[l].diagonal(dim1=-2, dim2=-1).sum()) for l in range(L)],
                         "fisher_trace_v": [float(Fv[l].diagonal(dim1=-2, dim2=-1).sum()) for l in range(L)]})
    return Fk.float(), Fv.float(), prof


class Calibrator:
    """Fits LayerCodecs for a Variant (theta search on clean captures, then the sequential fit)."""

    def __init__(self, runner: Runner, seqs: torch.Tensor, clean: Capture, Fk, Fv):
        self.runner, self.seqs, self.clean = runner, seqs, clean
        self.cfg = runner.cfg
        n = runner.n
        eye = metric_sqrt(None, n, 0.0, DEVICE)
        self.metric = {"euclid": [(eye, eye)] * runner.L,
                       "fisher": [(metric_sqrt(Fk[l], n, self.cfg.FISHER_SHRINK, DEVICE),
                                   metric_sqrt(Fv[l], n, self.cfg.FISHER_SHRINK, DEVICE)) for l in range(runner.L)]}
        self.alpha_cache: Dict[Tuple, float] = {}

    def _targets(self, v: Variant) -> Dict[str, float]:
        """Coded bits per token.  Aimed BELOW the budget so that the achieved rate, after the
        +-BUDGET_TOL convergence band and the expected cost of exact outlier rows, does not
        exceed the budget of the baseline it is compared with."""
        n, c = self.runner.n, self.cfg
        outlier = (1 - c.OUTLIER_QUANTILE) * (16.0 * n + 32.0) / n if c.OUTLIER_QUANTILE else 0.0
        per = max(0.05, v.budget * (1 - c.BUDGET_TOL) - outlier) * n * self.runner.L
        return {"kv": 2 * per} if v.joint_kv else {"k": per, "v": per}

    def _fit_layer(self, li, Xk, Xv, prev, thetas, v: Variant, cache_alpha=True):
        (Sk, Ski), (Sv, Svi) = self.metric[v.metric][li]
        zk = torch.cat(prev, -1) if (v.xlayer and prev is not None) else None
        tk = thetas.get("kv", thetas.get("k"))
        tv = thetas.get("kv", thetas.get("v"))
        ka = (v.metric, v.xlayer, li, "k")
        pk, xk_hat, ik = fit_part(Xk, zk, Sk, Ski, tk, self.cfg, self.alpha_cache.get(ka))
        zv = torch.cat([*prev, xk_hat], -1) if (v.xlayer and prev is not None) else xk_hat
        va = (v.metric, v.xlayer, li, "v")
        pv, xv_hat, iv = fit_part(Xv, zv, Sv, Svi, tv, self.cfg, self.alpha_cache.get(va))
        if cache_alpha:
            self.alpha_cache.setdefault(ka, ik["alpha"])
            self.alpha_cache.setdefault(va, iv["alpha"])
        return LayerCodec(pk, pv, v.xlayer), xk_hat, xv_hat, ik, iv

    def open_loop_rates(self, thetas, v: Variant) -> Dict[str, float]:
        prev, rk, rv = None, 0, 0
        for li in range(self.runner.L):
            Xk = self.clean.get(li, "xk").to(DEVICE).float()
            Xv = self.clean.get(li, "xv").to(DEVICE).float()
            _, xk_hat, xv_hat, ik, iv = self._fit_layer(li, Xk, Xv, prev, thetas, v)
            prev = (xk_hat, xv_hat)
            rk, rv = rk + ik["rate"], rv + iv["rate"]
        return {"kv": rk + rv, "k": rk, "v": rv}

    def search(self, v: Variant):
        """Bisection on log(theta) per budget part on clean captures; returns thetas and history."""
        targets = self._targets(v)
        X0 = self.clean.get(0, "xk").to(DEVICE).float()
        S0 = self.metric[v.metric][0][0][0]
        scale = float(((X0 - X0.mean(0)) @ S0).pow(2).mean())
        thetas = {p: scale * 1e-3 for p in targets}
        history = {p: [] for p in targets}
        for part in targets:               # rate is non-increasing in theta: keep rate(hi) <= target
            lo, hi = math.log(scale * 1e-12), math.log(scale * 1e3)
            for _ in range(self.cfg.THETA_ITERS):
                mid = 0.5 * (lo + hi)
                thetas[part] = math.exp(mid)
                rate = self.open_loop_rates(thetas, v)[part]
                history[part].append((mid, math.log(max(rate, 1))))
                if rate > targets[part]:
                    lo = mid
                else:
                    hi = mid
                if abs(rate - targets[part]) <= 0.005 * targets[part]:
                    hi = mid
                    break
            thetas[part] = math.exp(hi)
        return thetas, history

    def sequential(self, thetas, v: Variant):
        """Layer-by-layer fit; layer l's rows come from a pass with layers < l compressed (G3)."""
        L, c = self.runner.L, self.cfg
        codecs, infos = {}, []
        for li in range(L):
            cap = Capture([li], torch.float32, store_prev=True)
            comp = LCTCCompressor(codecs, self.runner.rope)
            run_capture(self.runner, self.seqs, cap, c.FIT_TOKENS, "fit", comp,
                        shadow=not v.propagation, stop_after=li)
            Xk, Xv = cap.get(li, "xk").to(DEVICE), cap.get(li, "xv").to(DEVICE)
            prev = None if li == 0 else (cap.get(li, "pk").to(DEVICE), cap.get(li, "pv").to(DEVICE))
            codec, _, _, ik, iv = self._fit_layer(li, Xk, Xv, prev, thetas, v, cache_alpha=False)
            codecs[li] = codec
            infos.append({"layer": li, **{f"k_{a}": b for a, b in ik.items()}, **{f"v_{a}": b for a, b in iv.items()}})
            del cap, Xk, Xv, prev
            free_memory()
        return codecs, pd.DataFrame(infos)

    def fit(self, v: Variant):
        t0 = time.time()
        targets = self._targets(v)
        thetas, history = self.search(v)
        for step in range(self.cfg.SECANT_STEPS + 1):
            codecs, info = self.sequential(thetas, v)
            achieved = {"kv": float(info.k_rate.sum() + info.v_rate.sum()),
                        "k": float(info.k_rate.sum()), "v": float(info.v_rate.sum())}
            off = {p: achieved[p] / targets[p] - 1 for p in targets}
            logger.info(f"    {v.name}: sequential pass {step}: rate error " +
                        ", ".join(f"{p}={o:+.3f}" for p, o in off.items()))
            if all(abs(o) <= self.cfg.BUDGET_TOL for o in off.values()) or step == self.cfg.SECANT_STEPS:
                break
            for p in targets:
                # Newton step on log rate = a + s log theta, slope s < 0 from the open-loop curve:
                # log theta_new = log theta + (log target - log achieved) / s.
                xs = np.array([a for a, _ in history[p]])
                ys = np.array([b for _, b in history[p]])
                slope = -0.5
                if len(xs) >= 3:
                    near = np.argsort(np.abs(xs - math.log(thetas[p])))[:4]
                    if np.ptp(xs[near]) > 0:
                        slope = min(-0.05, float(np.polyfit(xs[near], ys[near], 1)[0]))
                if achieved[p] > 0:
                    thetas[p] = math.exp(math.log(thetas[p])
                                         + (math.log(targets[p]) - math.log(achieved[p])) / slope)
                else:
                    thetas[p] *= 0.1
        info["variant"] = v.name
        meta = {"thetas": thetas, "targets": targets, "achieved": achieved,
                "rate_bits_per_element": achieved.get("kv", achieved.get("k", 0) + achieved.get("v", 0))
                / (2 * self.runner.n * self.runner.L),
                "pred_dnll_per_token": 0.5 * float(info.k_D.sum() + info.v_D.sum()) if v.metric == "fisher"
                else float("nan"),
                "params_millions": sum(cd.k.n_params() + cd.v.n_params() for cd in codecs.values()) / 1e6,
                "fit_seconds": time.time() - t0}
        return codecs, info, meta


# ════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS: coordinate mismatch vs linear predictability (explains E1-E3)
# ════════════════════════════════════════════════════════════════════════════

def unbiased_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA with the unbiased HSIC estimator (Song et al., 2012); as in KV_LDT_v12_2."""
    X = X.to(DEVICE, torch.float64)
    Y = Y.to(DEVICE, torch.float64)
    n = X.shape[0]
    if n < 8:
        return float("nan")
    X, Y = X - X.mean(0), Y - Y.mean(0)

    def hsic(K, M):
        K, M = K.clone(), M.clone()
        K.fill_diagonal_(0.0)
        M.fill_diagonal_(0.0)
        return ((K * M).sum() + K.sum() * M.sum() / ((n - 1) * (n - 2))
                - 2.0 * (K.sum(0) @ M.sum(1)) / (n - 2)) / (n * (n - 3))

    K, M = X @ X.T, Y @ Y.T
    den = torch.sqrt(hsic(K, K) * hsic(M, M))
    return float(hsic(K, M) / den) if den > 0 else float("nan")


@torch.no_grad()
def coordinate_mismatch(runner: Runner, clean: Capture, cfg: Config) -> pd.DataFrame:
    """
    For every adjacent pair (l-1 -> l):
      raw_rel_distance   ||[K;V]_{l-1} - [K;V]_l|| / ||[K;V]_l||  (post-RoPE; v12_2's quantity)
      cka                unbiased linear CKA of post-RoPE [K;V] rows
      fvu_identity       variance of [K;V]_l unexplained by substituting [K;V]_{l-1} (pre-RoPE)
      fvu_linear         variance unexplained by the ridge predictor, OUT OF SAMPLE
                         (fitted on the first half of the sequences, scored on the rest), pre-RoPE
    E1 predicts: high CKA, fvu_identity >= 1, fvu_linear << 1.
    """
    rows = []
    for li in range(1, runner.L):
        Ps, Pt = clean.get(li - 1, "post").float(), clean.get(li, "post").float()
        m = min(len(Ps), len(Pt))
        Ps, Pt = Ps[:m].to(DEVICE), Pt[:m].to(DEVICE)
        Zs = torch.cat([clean.get(li - 1, "xk"), clean.get(li - 1, "xv")], -1).to(DEVICE).float()
        Xt = torch.cat([clean.get(li, "xk"), clean.get(li, "xv")], -1).to(DEVICE).float()
        half = Xt.shape[0] // 2                 # rows are grouped by sequence: disjoint sequences
        tr, te = slice(0, half), slice(half, None)
        _, P, alpha, _ = ridge_fit(Zs[tr], Xt[tr] - Xt[tr].mean(0), cfg.RIDGE_GRID)
        pred = (Zs[te] - Zs[tr].mean(0)) @ P + Xt[tr].mean(0)
        var = (Xt[te] - Xt[tr].mean(0)).pow(2).sum()
        rows.append({"source": li - 1, "target": li,
                     "raw_rel_distance": float((Ps - Pt).norm() / Pt.norm()),
                     "cka": unbiased_linear_cka(Pt, Ps),
                     "fvu_identity": float((Xt[te] - Zs[te]).pow(2).sum() / var),
                     "fvu_linear": float((Xt[te] - pred).pow(2).sum() / var), "ridge_alpha": alpha,
                     "rope_pair": f"{int(runner.rope[li - 1])}{int(runner.rope[li])}"})
    df = pd.DataFrame(rows)
    if len(df) >= 4:
        rho = stats.spearmanr(df.cka, 1 - df.fvu_linear)
        logger.info(f"  coordinate mismatch: median raw rel. distance {df.raw_rel_distance.median():.3f}, "
                    f"median CKA {df.cka.median():.3f}, median FVU identity {df.fvu_identity.median():.3f} "
                    f"vs linear {df.fvu_linear.median():.3f}; Spearman(CKA, 1-FVU_linear) = {rho[0]:.3f}")
    return df


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
def eval_wikitext(runner: Runner, comp: Optional[Compressor], ids: List[int]) -> pd.DataFrame:
    cfg, bos = runner.cfg, runner.bos
    plan = wikitext_windows(ids, cfg)
    groups = defaultdict(list)
    for w, (b, e, ns) in enumerate(plan):
        groups[(e - b, ns)].append((w, b, e))
    sim = Sim(runner, comp)
    rows = []
    for (length, ns), members in groups.items():
        for i in range(0, len(members), runner.mc.eval_batch):
            chunk = members[i:i + runner.mc.eval_batch]
            x = torch.tensor([bos + ids[b:e] for _, b, e in chunk], device=DEVICE)
            T = x.shape[1]
            start = max(1, T - ns)
            keep = T - start + 1
            lg = runner.forward(x, sim, logits_to_keep=keep).logits.float()
            nll = F.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), x[:, start:].reshape(-1),
                                  reduction="none").view(len(chunk), -1).sum(1)
            rows += [{"window": w, "nll": float(v), "n_tok": T - start} for (w, _, _), v in zip(chunk, nll)]
    return pd.DataFrame(rows).sort_values("window").reset_index(drop=True)


@torch.no_grad()
def eval_lambada(runner: Runner, comp: Optional[Compressor], texts: List[str]) -> pd.DataFrame:
    sim, tok, rows = Sim(runner, comp), runner.tokenizer, []
    for i, text in enumerate(texts):
        ctx, last = text.rsplit(" ", 1)
        c = runner.bos + tok(ctx, add_special_tokens=False).input_ids
        t = tok(" " + last, add_special_tokens=False).input_ids
        x = torch.tensor([c + t], device=DEVICE)
        lp = runner.forward(x, sim, logits_to_keep=len(t) + 1).logits[0, :-1].float().log_softmax(-1)
        tgt = torch.tensor(t, device=DEVICE)
        rows.append({"item": i, "correct": int(bool((lp.argmax(-1) == tgt).all())),
                     "nll": float(-lp.gather(1, tgt[:, None]).sum())})
    return pd.DataFrame(rows)


def passkey_items(runner: Runner, cfg: Config) -> List[Dict]:
    """Passkey retrieval (Mohtashami & Jaggi, 2023) at fixed lengths and needle depths."""
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
                body = filler * at + needle + filler * (n_fill - at)
                items.append({"length": length, "depth": depth, "rep": r,
                              "prompt": runner.bos + body + question, "answer": answer})
    return items


@torch.no_grad()
def eval_passkey(runner: Runner, comp: Optional[Compressor], items: List[Dict]) -> pd.DataFrame:
    """Teacher-forced: greedy decoding returns the key iff every answer token is the argmax."""
    sim, rows = Sim(runner, comp), []
    for i, it in enumerate(items):
        x = torch.tensor([it["prompt"] + it["answer"]], device=DEVICE)
        a = len(it["answer"])
        lp = runner.forward(x, sim, logits_to_keep=a + 1).logits[0, :-1].float().log_softmax(-1)
        tgt = torch.tensor(it["answer"], device=DEVICE)
        rows.append({"item": i, "length": it["length"], "depth": it["depth"],
                     "correct": int(bool((lp.argmax(-1) == tgt).all())),
                     "nll": float(-lp.gather(1, tgt[:, None]).sum())})
    return pd.DataFrame(rows)


@torch.no_grad()
def decode_equivalence(runner: Runner, codecs: Dict[int, LayerCodec], ids: torch.Tensor) -> Dict:
    """G4: single-pass masked simulation vs a real streaming decoder that stores uint8 codes."""
    T = ids.shape[1]
    comp = LCTCCompressor(codecs, runner.rope, record=True)
    sim_logits = runner.forward(ids, Sim(runner, comp)).logits[0].float()
    st = StreamingLCTC(runner, codecs)
    dec_logits = []
    for t in range(T):
        st.begin(t)
        with routed(st):
            out = runner.model(input_ids=ids[:, t:t + 1], position_ids=torch.tensor([[t]], device=DEVICE),
                               use_cache=False)
        dec_logits.append(out.logits[0, -1].float())
    dec_logits = torch.stack(dec_logits)
    # A (layer, token) row "agrees" when all its K and V codes are identical.  Rare
    # disagreements are quantiser-bin flips caused by floating-point reassociation (batched
    # vs single-token matmuls); a flip then propagates along that token's depth chain.
    agree = []
    for li in range(runner.L):
        rows = torch.ones(T, dtype=torch.bool, device=DEVICE)
        for j in (0, 1):
            sim_c = comp.codes[li][j]
            dec_c = torch.cat([st.codes[li][t][j] for t in range(T)])
            if sim_c.numel():
                rows &= (sim_c == dec_c).all(1)
        agree.append(float(rows.float().mean()))
    full_bytes = runner.L * 2 * runner.n * T * 2
    mem = st.stored_bytes()
    res = {"T": T, "max_abs_logit_diff": float((sim_logits - dec_logits).abs().max()),
           "logit_scale": float(sim_logits.abs().max()),
           "argmax_agreement": float((sim_logits.argmax(-1) == dec_logits.argmax(-1)).float().mean()),
           "min_layer_row_agreement": min(agree), "mean_row_agreement": float(np.mean(agree)),
           "fp16_cache_bytes": full_bytes, **mem,
           "stored_fraction_packed": (mem["codes_bytes_packed"] + mem["exact_bytes"] + mem["outlier_bytes"])
           / full_bytes}
    logger.info(f"  decode equivalence (T={T}): max|dlogit|={res['max_abs_logit_diff']:.2e} "
                f"(scale {res['logit_scale']:.1f}), argmax agreement {res['argmax_agreement']:.4f}, "
                f"row-code agreement min/mean {res['min_layer_row_agreement']:.4f}/{res['mean_row_agreement']:.4f}")
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


def paired_nll_test(a: pd.DataFrame, b: pd.DataFrame, n_boot: int, n_perm: int, seed: int) -> Dict:
    """
    A vs B on the same units (windows or items).  Estimate: (sum nll_A - sum nll_B) / sum n_tok
    (negative = A better).  95% CI: paired bootstrap over units.  p: two-sided paired
    sign-flip randomisation test of the per-unit difference (exchangeability of the two
    method labels within a unit under H0).
    """
    m = a.merge(b, on=a.columns[0], suffixes=("_a", "_b"))
    ntok = m["n_tok_a"].to_numpy() if "n_tok_a" in m else np.ones(len(m))
    d = m["nll_a"].to_numpy() - m["nll_b"].to_numpy()
    est = d.sum() / ntok.sum()
    rng = np.random.default_rng(seed)
    bi = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot = d[bi].sum(1) / ntok[bi].sum(1)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    null = (signs * d).sum(1)
    p = (1 + np.sum(np.abs(null) >= abs(d.sum()) - 1e-12)) / (n_perm + 1)
    return {"estimate": est, "ci_low": float(np.percentile(boot, 2.5)),
            "ci_high": float(np.percentile(boot, 97.5)), "p_value": float(p), "n_units": len(d)}


def mcnemar(a: pd.Series, b: pd.Series) -> float:
    b01, b10 = int(((a == 0) & (b == 1)).sum()), int(((a == 1) & (b == 0)).sum())
    return 1.0 if b01 + b10 == 0 else float(stats.binomtest(min(b01, b10), b01 + b10, 0.5).pvalue)


# Pre-specified contrasts: (name, A, B, gap).  Directional claim: A has lower WikiText-2 NLL.
def contrasts(cfg: Config) -> List[Tuple[str, str, str, str]]:
    b0 = f"{cfg.PRIMARY_BUDGET:g}"
    low = f"{min(cfg.BUDGETS):g}"
    out = [("lctc_vs_kivi2", f"lctc@{b0}", f"kivi{min(cfg.KIVI_BITS)}", "matched-rate SOTA baseline")]
    hi_kivi = max(cfg.KIVI_BITS)
    if f"{cfg.kivi_effective_bits(hi_kivi):g}" in [f"{b:g}" for b in cfg.BUDGETS]:
        out.append(("lctc_vs_kivi4", f"lctc@{cfg.kivi_effective_bits(hi_kivi):g}", f"kivi{hi_kivi}",
                    "matched-rate SOTA baseline"))
    out += [("G1_loss_metric", f"lctc_sep_kv@{b0}", f"lctc_euclid_sep@{b0}", "G1 Fisher-metric transform"),
            ("G2_joint_kv", f"lctc@{b0}", f"lctc_sep_kv@{b0}", "G2 joint K/V allocation"),
            ("G3_propagation", f"lctc@{b0}", f"lctc_open_loop@{b0}", "G3 propagation-aware calibration"),
            ("xlayer_prediction", f"lctc@{b0}", f"lctc_no_xlayer@{b0}", "inter-layer prediction (AQUA-KV-style)"),
            ("lowest_rate_vs_cla", f"lctc@{low}", "cla", "E2: coding at far lower rate vs substitution")]
    return out


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

def load_text_data(runner: Runner, cfg: Config):
    from datasets import load_dataset
    tok, bos = runner.tokenizer, runner.bos
    train = load_dataset(*cfg.CALIB_DATASET)
    tr_ids = tok("\n\n".join(train["text"]), add_special_tokens=False).input_ids
    span = cfg.CALIB_SEQ_LEN - len(bos)
    starts = _rng("calib").choice(len(tr_ids) - span, size=cfg.CALIB_N_SEQ, replace=False)
    seqs = torch.tensor([bos + tr_ids[s:s + span] for s in sorted(starts)])
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    test_ids = tok("\n\n".join(wiki["text"]), add_special_tokens=False).input_ids
    test_ids = test_ids[: cfg.PPL_WINDOW + cfg.PPL_STRIDE * (cfg.PPL_MAX_WINDOWS - 1)]
    lam = []
    if cfg.RUN_LAMBADA:
        lam = list(load_dataset("EleutherAI/lambada_openai", "default", split="test")["text"][: cfg.LAMBADA_N])
    return seqs, test_ids, lam


def summarize_method(name, comp, rate, mem_extra, wt, lam, pk, cfg: Config, runner: Runner) -> Dict:
    nll = wt.nll.sum() / wt.n_tok.sum()
    row = {"method": name, "bits_per_element": rate, "wikitext2_ppl": math.exp(nll), "wikitext2_nll": nll}
    for T in (4096, 32768):
        exact = min(T, cfg.N_SINK + cfg.WINDOW)
        row[f"mem_fraction_T{T}"] = (rate * (T - exact) + 16.0 * exact) / (16.0 * T) if name != "cla" else rate / 16
    row.update(mem_extra)
    if lam is not None and len(lam):
        row.update(lambada_acc=lam.correct.mean(), lambada_nll=lam.nll.mean())
    if pk is not None and len(pk):
        row["passkey_acc"] = pk.correct.mean()
        for L_, g in pk.groupby("length"):
            row[f"passkey_acc_{L_}"] = g.correct.mean()
    return row


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
        seqs, test_ids, lam_texts = load_text_data(runner, cfg)
    else:
        model, seqs, test_ids = prebuilt
        runner = Runner(mc, cfg, model=model, tokenizer=None)
        lam_texts = []
    self_tests(runner, seqs)

    logger.info("  Fisher blocks (per layer, KV head)")
    Fk, Fv, fprof = compute_fisher(runner, seqs)
    fprof.to_csv(os.path.join(out_dir, "fisher_profile.csv"), index=False)

    logger.info("  clean capture (theta search + diagnostics)")
    clean = Capture(range(runner.L), torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
                    post_rows=cfg.DIAG_ROWS)
    run_capture(runner, seqs, clean, cfg.SEARCH_TOKENS, "search")
    coordinate_mismatch(runner, clean, cfg).to_csv(os.path.join(out_dir, "coordinate_mismatch.csv"), index=False)

    items = passkey_items(runner, cfg) if (cfg.RUN_PASSKEY and runner.tokenizer is not None) else []
    results, per_unit = [], {}

    def evaluate(name, comp, rate, extra=None):
        t0 = time.time()
        wt = eval_wikitext(runner, comp, test_ids)
        lam = eval_lambada(runner, comp, lam_texts) if lam_texts else None
        pk = eval_passkey(runner, comp, items) if items else None
        rate_eff = rate + (comp.extra_bits() if comp is not None else 0.0)
        row = summarize_method(name, comp, rate_eff, extra or {}, wt, lam, pk, cfg, runner)
        row["eval_seconds"] = time.time() - t0
        results.append(row)
        per_unit[name] = (wt, lam, pk)
        wt.assign(method=name).to_csv(os.path.join(out_dir, f"wikitext_{name}.csv"), index=False)
        if lam is not None:
            lam.assign(method=name).to_csv(os.path.join(out_dir, f"lambada_{name}.csv"), index=False)
        if pk is not None:
            pk.assign(method=name).to_csv(os.path.join(out_dir, f"passkey_{name}.csv"), index=False)
        logger.info(f"  {name:<24} bits={rate_eff:6.3f} ppl={row['wikitext2_ppl']:.4f}"
                    + (f" lambada={row['lambada_acc']:.3f}" if "lambada_acc" in row else "")
                    + (f" passkey={row['passkey_acc']:.3f}" if "passkey_acc" in row else ""))

    evaluate("full", None, 16.0)
    cmap = cla_source_map(runner.L, cfg.CLA_REUSE, cfg.CLA_EXEMPT)
    evaluate("cla", CLACompressor(cmap), 16.0 * (1 - len(cmap) / runner.L))
    for b in cfg.KIVI_BITS:
        evaluate(f"kivi{b}", KIVICompressor(b, cfg), cfg.kivi_effective_bits(b))
    if mc.name in PRIOR_PPL:
        ref_full, ref_cla = PRIOR_PPL[mc.name]
        got = {r["method"]: r["wikitext2_ppl"] for r in results}
        logger.info(f"  reproduction vs KV_LDT_v12_2: no_reuse {got['full']:.4f} vs {ref_full:.4f} "
                    f"({got['full'] / ref_full - 1:+.2%}); CLA {got['cla']:.2f} vs {ref_cla:.2f}")

    calib = Calibrator(runner, seqs, clean, Fk, Fv)
    layer_tables, metas = [], {}
    for v in cfg.variants():
        logger.info(f"  fitting {v.name}")
        codecs, info, meta = calib.fit(v)
        layer_tables.append(info)
        metas[v.name] = meta
        comp = LCTCCompressor(codecs, runner.rope)
        evaluate(v.name, comp, meta["rate_bits_per_element"],
                 {"pred_dnll_per_token": meta["pred_dnll_per_token"], "params_millions": meta["params_millions"],
                  "fit_seconds": meta["fit_seconds"]})
        if v.name == f"lctc@{cfg.PRIMARY_BUDGET:g}":
            T = min(cfg.DECODE_TEST_LEN, test_ids.__len__())
            ids = torch.tensor([runner.bos + list(test_ids[:T - len(runner.bos)])], device=DEVICE)
            with open(os.path.join(out_dir, "decode_equivalence.json"), "w") as f:
                json.dump(decode_equivalence(runner, codecs, ids), f, indent=2)
        del codecs, comp
        free_memory()
    pd.concat(layer_tables).to_csv(os.path.join(out_dir, "codec_layers.csv"), index=False)
    with open(os.path.join(out_dir, "codec_meta.json"), "w") as f:
        json.dump(metas, f, indent=2, default=float)

    summary = pd.DataFrame(results)
    full_nll = float(summary.loc[summary.method == "full", "wikitext2_nll"].iloc[0])
    summary["delta_nll_vs_full"] = summary.wikitext2_nll - full_nll
    summary.insert(0, "model", mc.name)
    summary.insert(1, "family", mc.family)

    # P5: is the second-order loss model predictive?  (Fisher-metric LCTC fits only.)
    fis = summary.dropna(subset=["pred_dnll_per_token"]) if "pred_dnll_per_token" in summary else summary.iloc[:0]
    if len(fis) >= 4:
        rho, p = stats.spearmanr(fis.pred_dnll_per_token, fis.delta_nll_vs_full)
        logger.info(f"  P5 second-order validity: Spearman(pred dNLL, observed dNLL) = {rho:.3f} "
                    f"(n={len(fis)}, p={p:.3g}); median observed/predicted ratio "
                    f"{np.median(fis.delta_nll_vs_full / fis.pred_dnll_per_token):.2f}")
        summary["P5_spearman"] = rho

    rows = []
    for name, A, B, gap in contrasts(cfg):
        if A not in per_unit or B not in per_unit:
            continue
        r = paired_nll_test(per_unit[A][0], per_unit[B][0], cfg.N_BOOT, cfg.N_PERM, SEED)
        rate_a = float(summary.loc[summary.method == A, "bits_per_element"].iloc[0])
        rate_b = float(summary.loc[summary.method == B, "bits_per_element"].iloc[0])
        row = {"model": mc.name, "family": mc.family, "contrast": name, "gap": gap, "A": A, "B": B,
               "rate_A": rate_a, "rate_B": rate_b, **r}
        for k, idx in (("lambada", 1), ("passkey", 2)):
            ua, ub = per_unit[A][idx], per_unit[B][idx]
            if ua is not None and ub is not None and len(ua):
                row[f"{k}_acc_diff"] = ua.correct.mean() - ub.correct.mean()
                row[f"{k}_mcnemar_p"] = mcnemar(ua.correct, ub.correct)
        rows.append(row)
    ct = pd.DataFrame(rows)
    if len(ct):
        ct["p_holm"] = holm(ct.p_value)
        # a method never gets credit for winning with more bits than its comparator
        ct["rate_ok"] = ct.rate_A <= ct.rate_B + 1e-9
        ct["supported"] = (ct.estimate < 0) & (ct.p_holm < cfg.ALPHA) & ct.rate_ok
        ct.to_csv(os.path.join(out_dir, "contrasts.csv"), index=False)
    summary["total_seconds"] = time.time() - t_model
    summary.to_csv(summary_path, index=False)
    if prebuilt is None:
        runner.release()
    return summary


def self_tests(runner: Runner, seqs: torch.Tensor):
    """
    (1) rope_invert(rope_apply(x)) = x and rope_apply equals the model's own RoPE;
    (2) the masked simulation without a compressor reproduces SDPA (negative control);
    (3) a lossless 'compressor' (identity K/V) leaves logits unchanged.
    """
    pos = torch.arange(seqs.shape[1], device=DEVICE)
    cos, sin = runner.cos_sin(pos)
    x = torch.randn(2, runner.Hkv, len(pos), runner.d, device=DEVICE)
    err_inv = float((rope_invert(rope_apply(x, cos, sin), cos, sin) - x).abs().max())
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    ref, _ = apply_rotary_pos_emb(x, x, cos[None], sin[None])
    err_rope = float((ref - rope_apply(x, cos, sin)).abs().max())
    ids = seqs[:2, :128].to(DEVICE)
    with torch.no_grad():
        base = runner.forward(ids, None).logits.float()
        sim = runner.forward(ids, Sim(runner)).logits.float()

        class Identity(Compressor):
            def compress(self, li, k, v, cos_, sin_):
                return k, v
        ident = runner.forward(ids, Sim(runner, Identity())).logits.float()
    # bf16 models: SDPA runs in bf16 while the simulation accumulates in fp32
    tol = 1e-3 if COMPUTE_DTYPE == torch.float32 else 0.25
    d_sim = float((base - sim).abs().max())
    d_id = float((sim - ident).abs().max())
    logger.info(f"  self-tests: rope inverse {err_inv:.1e}, rope vs model {err_rope:.1e}, "
                f"sim vs SDPA {d_sim:.2e}, identity compressor {d_id:.1e}")
    if err_inv > 1e-4 or err_rope > 1e-4 or d_sim > tol or d_id > tol:
        raise RuntimeError("self-tests failed: the evaluation harness is not an exact identity")


def cross_model(cfg: Config, summaries: List[pd.DataFrame]):
    if not summaries:
        return
    allsum = pd.concat(summaries, ignore_index=True)
    allsum.to_csv(os.path.join(cfg.OUTPUT_DIR, "all_summary.csv"), index=False)
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
        if n >= cfg.MIN_MODELS_WILCOXON:
            pw = float(stats.wilcoxon(g.estimate, alternative="less").pvalue)
        else:
            pw = float("nan")
        verdict = ("insufficient models" if n < cfg.MIN_MODELS_WILCOXON else
                   "supported" if (met >= need and pw < cfg.ALPHA) else "not supported")
        lofo = {}
        for fam in g.family.unique():
            h = g[g.family != fam]
            lofo[f"lofo_{fam}"] = bool(len(h) and h.supported.sum() >= math.ceil(cfg.DECISION_MIN_MODEL_FRACTION * len(h)))
        rows.append({"contrast": name, "gap": g.gap.iloc[0], "models_evaluated": n, "models_meeting_rule": met,
                     "models_required": need, "median_delta_nll": float(g.estimate.median()),
                     "p_wilcoxon_models": pw, "verdict": verdict, **lofo,
                     "rule": f"A better than B (Holm p<{cfg.ALPHA}, measured rate_A<=rate_B) in >= {need}/{n} models "
                             f"AND one-sided Wilcoxon over models p<{cfg.ALPHA} (needs >= {cfg.MIN_MODELS_WILCOXON})"})
    dr = pd.DataFrame(rows)
    dr.to_csv(os.path.join(cfg.OUTPUT_DIR, "decision_rules.csv"), index=False)
    for _, r in dr.iterrows():
        logger.info(f"DECISION {r.contrast:<22} {r.verdict:<20} {r.models_meeting_rule}/{r.models_evaluated} "
                    f"median dNLL={r.median_delta_nll:+.4f}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for model, g in allsum.groupby("model"):
            lc = g[g.method.str.fullmatch(r"lctc@[\d.]+")].sort_values("bits_per_element")
            line, = ax.plot(lc.bits_per_element, lc.delta_nll_vs_full, "o-", label=f"{model} LCTC")
            kv = g[g.method.str.startswith("kivi")]
            ax.plot(kv.bits_per_element, kv.delta_nll_vs_full, "s", color=line.get_color(), mfc="none")
        ax.set_xlabel("bits per cached K/V element (16 = fp16)")
        ax.set_ylabel("WikiText-2 dNLL per token vs full cache")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(fontsize=7)
        ax.set_title("Rate-distortion (squares: KIVI)")
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.OUTPUT_DIR, "rate_distortion.png"), dpi=200)
        plt.close(fig)
    except Exception as e:                       # figures are optional
        logger.warning(f"figure skipped: {e}")


# ════════════════════════════════════════════════════════════════════════════
# SMOKE TEST (offline, CPU): tiny models trained briefly on a synthetic Markov source
# ════════════════════════════════════════════════════════════════════════════

def smoke_setup(cfg: Config):
    from transformers import LlamaConfig, SmolLM3Config
    V, T = 128, 192
    g = torch.Generator().manual_seed(SEED)
    trans = torch.softmax(torch.randn(V, V, generator=g) * 3.0, -1)

    period = 48

    def sample(n, length):
        """Markov segments of `period` tokens, repeated: predicting a repeat needs attention
        `period` positions back, i.e. beyond the exact window, so KV damage is measurable."""
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
    out = []
    for name, fam, conf in builds:
        torch.manual_seed(SEED)
        model = AutoModelForCausalLM.from_config(conf, attn_implementation=LCTC_ATTN)
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
        seqs = sample(cfg.CALIB_N_SEQ, T)
        test_ids = sample(1, cfg.PPL_WINDOW + cfg.PPL_STRIDE * (cfg.PPL_MAX_WINDOWS - 1))[0].tolist()
        out.append((ModelConfig(name, "random-init", fam, 4), (model, seqs, test_ids)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
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
        kw.update(OUTPUT_DIR=a.output or str(PROJECT_ROOT / "lctc_smoke"), SMOKE=True, CALIB_N_SEQ=24,
                  CALIB_BATCH=8, SEARCH_TOKENS=3000, FIT_TOKENS=4000, DIAG_ROWS=512, FISHER_N_SEQ=8,
                  RD_ROWS=2048, THETA_ITERS=14, CLIP_MULTS=12, PPL_WINDOW=160, PPL_STRIDE=80,
                  PPL_MAX_WINDOWS=24, N_BOOT=500, N_PERM=2000, WINDOW=8, DECODE_TEST_LEN=48,
                  RUN_LAMBADA=False, RUN_PASSKEY=False, MIN_MODELS_WILCOXON=2, RESUME=False,
                  BUDGETS=(0.5, 1.0, 1.5, 3.0, 5.0), PRIMARY_BUDGET=1.0)
    cfg = Config(**kw)
    if a.no_resume:
        cfg.RESUME = False
    cfg.RUN_PASSKEY &= not a.no_passkey
    cfg.RUN_LAMBADA &= not a.no_lambada
    fh = logging.FileHandler(os.path.join(cfg.OUTPUT_DIR, "lctc_kv.log"))
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    with open(os.path.join(cfg.OUTPUT_DIR, "config.json"), "w") as f:
        json.dump({k: (v if not isinstance(v, list) or not v or not hasattr(v[0], "__dict__")
                       else [asdict(m) for m in v]) for k, v in cfg.__dict__.items()}, f, indent=2, default=str)
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
