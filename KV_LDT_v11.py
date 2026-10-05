"""
KV_LDT_v11.py — Lexical fidelity of training-free cross-layer KV sharing in LLMs.

Question
--------
Does training-free cross-layer KV-cache sharing preserve the lexical information
that frozen decoder-only LLMs use to separate words from nonwords?  Does the
damage depend on word frequency once subword token count is controlled, on where
in depth the sharing happens, and on how the sharing partner is chosen?

Scope (stated explicitly, construct validity)
---------------------------------------------
We measure *representational lexical status*: how well lexicality can be decoded
from the hidden state of the final subword of a letter string presented in a
fixed neutral carrier ("Here is a letter string: <string>").  We do not measure
the model *performing* a lexical decision task.

Design decisions, mapped to the audit (B = bug, C = confound, S = statistics)
------------------------------------------------------------------------------
Mechanism
  * KV sharing is implemented at the attention-function level (a registered
    `AttentionInterface` + `AttentionMaskInterface` entry).  The tensors that are
    substituted are exactly the tensors a KV cache stores: post-k_norm, post-RoPE
    keys and values, before GQA head repetition.  This is architecture-agnostic
    (Llama, Qwen2/2.5, Qwen3 with q/k-norm, SmolLM3 with NoPE layers) and replaces
    the v10 BypassableLinear / fused-QKV / single-vs-two-pass machinery.
  * C1  Every deployable policy uses identical LIVE semantics: a target layer
        reads the K/V its source layer produced in the same forward pass (the
        CLA / MiniCache / KVSharer inference semantics).  Sources are always
        earlier, non-target layers, so no reuse chains exist.
  * C2  Non-causal sharing (a lower layer reading a higher layer) is only
        possible with a clean pre-pass.  It is run as an explicitly labelled
        COUNTERFACTUAL pair (cf_prev_head vs cf_next_head) with identical target
        sets and mirrored source distance, both under CLEAN semantics.
        `full` vs `cf_prev_head` measures how much the semantics alone matters.
        v10 `adjacent_previous` is removed: under live semantics it collapses to a
        single shared group, and at reuse factor 2 it is identical to `full`.
  * C3  Gating similarity uses all K and V channels (no truncation), only
        stimulus-token rows (carrier and BOS rows are identical across items and
        would dominate), the unbiased HSIC estimator, and split-half reliability.
  * C4  The carrier guarantees every stimulus token attends over a multi-token
        context, so key substitution is never a softmax-over-one-key no-op, and
        BOS/no-BOS tokenizers no longer differ in a degenerate way.  Batches are
        length-bucketed, so there is no padding anywhere.
  * C5  Gated policies select EXACTLY k targets (top-k / bottom-k), and every
        gated policy has a depth-matched random control (R draws) as well as a
        uniform rate-matched random control (R draws).
  * G7  Three partner criteria are measured per layer pair: CKA
        (representational similarity, rotation invariant), relative Euclidean KV
        distance (the quantity KVSharer ranks layer pairs by), and query-aware
        attention-output fidelity (how much the target's own attention output
        changes when it reads the source's K/V).
        CKA is invariant to rotations of K, attention logits q.k are not, so CKA
        is theoretically mismatched to substitution; `fidelity_high` tests the
        query-aware criterion directly.
  * B2/B3  Wall-clock efficiency is not measured (hooks cannot show real
        savings).  Each policy reports its exact analytic KV-cache memory fraction
        (cached layers / all layers), the quantity CLA-style papers report.
  * B4  References corrected (KVSharer = Yang et al., 2024, arXiv:2410.18517).
Probing
  * S7  Primary probe is L2-regularised logistic regression (convex, unique
        optimum, full-batch L-BFGS on GPU); lambda is selected per layer on the
        no_reuse validation split and then held fixed for every policy.
        MLP probes are a capacity check (Task 5).
  * S6  Two readouts per run: RETRAINED (probe refit on policy states:
        recoverability) and FROZEN (no_reuse probe applied to policy states:
        preservation).  Their gap separates information loss from re-encoding.
  * G6  Control-task selectivity (Hewitt & Liang, 2019) per probe architecture
        and online-code MDL (Voita & Titov, 2020).
  * The test split is fixed across seeds, policies and models (same items).
Statistics (items and models are the units of inference, never layers)
  * S4  Threshold-free AUC for HF-words-vs-nonwords and LF-words-vs-nonwords
        (plus d' with log-linear correction), and a token-count-matched AUC that
        only compares words and nonwords with the same number of subword tokens
        (removes the "nonwords split into more tokens" shortcut).
  * S3  Primary H1 test: pre-registered depth bands; statistic = mean over band
        layers of AUC(policy) - AUC(no_reuse); stratified paired item bootstrap
        that resamples items jointly across layers (keeps layer dependence).
        Cross-model claims: Wilcoxon signed-rank with models as units.
  * S2  Policy x frequency interaction = difference-in-differences of AUC
        (bootstrap) and an item-level paired regression (first differences remove
        the item random intercept exactly).
  * S1  n_subword_tokens (per tokenizer) enters every frequency analysis as a
        covariate, as a stratification variable and as an exact matching variable.
  * S5/B1  Task 4: HF/LF test words are optimally matched within n_token strata on
        length, OrthoN (and mean bigram frequency if present) - NOT on human
        accuracy - and tested with an exact-form sign-flip permutation test,
        p = (1 + #{|null| >= |obs|}) / (1 + n_perm).
  * S8  Representation damage vs generative damage (WikiText-2 PPL, LAMBADA)
        Spearman correlation per model, Wilcoxon over models.
  * H5/G4  Single-layer substitution fragility scan: substitute K/V at ONE layer
        (source = previous layer), read out with the frozen probe downstream.
        Layers after the readout give an exact negative control (delta == 0).
        The n_token slope of item damage tests the detokenization mechanism.
  * Pre-registered decision rules (H1-H7, RQ8) are evaluated by code and
        written to `decision_rules.csv`.

Requirements: python>=3.10, torch>=2.1, transformers>=4.53, scikit-learn, scipy,
statsmodels, pandas, matplotlib; `datasets` for Task 3 (optional).
Inputs: ELP words (Word, Length, Log_Freq_HAL, Ortho_N[, BG_Mean]) and ELP
nonwords (Word, Length, Ortho_N[, BG_Mean]).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import json
import logging
import math
import re
import sys
import time
import zlib
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from packaging.version import Version
from scipy import stats
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

if Version(transformers.__version__) < Version("4.53"):
    raise RuntimeError("transformers>=4.53 is required (AttentionMaskInterface, "
                       f"SmolLM3, Qwen3); found {transformers.__version__}")
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

try:
    from datasets import load_dataset
    HF_DATASETS_AVAILABLE = True
except ImportError:
    HF_DATASETS_AVAILABLE = False

torch.set_num_threads(1)

# ════════════════════════════════════════════════════════════════════════════
# GLOBALS: logging, seeds, device
# ════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT = Path(os.environ.get("KV_PROJECT_ROOT", Path(__file__).resolve().parent)).resolve()
SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("kv_ldt_v11.log")])
logger = logging.getLogger("kv_ldt_v11")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


os.environ["PYTHONHASHSEED"] = str(SEED)
seed_everything(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
    # bf16 is the training dtype of every model in the sweep and cannot overflow
    # the residual stream the way fp16 does (Qwen2.5 is known to overflow in fp16).
    COMPUTE_DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    DEVICE = torch.device("cpu")
    COMPUTE_DTYPE = torch.float32
logger.info(f"device={DEVICE} compute_dtype={COMPUTE_DTYPE} "
            f"torch={torch.__version__} transformers={transformers.__version__}")


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    name: str
    model_id: str
    batch_size: int = 128          # upper bound; halved automatically on OOM


@dataclass
class ExperimentConfig:
    WORDS_PATH: str = str(Path(os.environ.get("KV_WORDS_PATH", PROJECT_ROOT / "Items.csv")))
    NONWORDS_PATH: str = str(Path(os.environ.get("KV_NONWORDS_PATH", PROJECT_ROOT / "NonWord.csv")))
    OUTPUT_DIR: str = str(Path(os.environ.get("KV_OUTPUT_DIR", PROJECT_ROOT / "kv_results_v11")))

    MODELS: List[ModelConfig] = field(default_factory=lambda: [
        ModelConfig("SmolLM2-360M", "HuggingFaceTB/SmolLM2-360M", 512),
        ModelConfig("Qwen2.5-1.5B", "Qwen/Qwen2.5-1.5B", 256),
        ModelConfig("Llama-3.2-1B", "meta-llama/Llama-3.2-1B", 256),
        ModelConfig("Qwen2.5-3B", "Qwen/Qwen2.5-3B", 256),
        ModelConfig("Llama-3.2-3B", "meta-llama/Llama-3.2-3B", 256),
        ModelConfig("SmolLM3-3B-Base", "HuggingFaceTB/SmolLM3-3B-Base", 256),
        ModelConfig("Qwen3-4B-Base", "Qwen/Qwen3-4B-Base", 128),
        ModelConfig("Qwen3-8B-Base", "Qwen/Qwen3-8B-Base", 128),
        ModelConfig("Llama-3.1-8B", "meta-llama/Llama-3.1-8B", 128),
    ])

    # ── Stimuli ───────────────────────────────────────────────────────
    # Must end with "{stimulus}": the readout token is then the stimulus' final
    # subword (the locus where multi-token words are assembled).
    STIMULUS_TEMPLATE: str = "Here is a letter string: {stimulus}"
    MAX_ITEMS_PER_CLASS: Optional[int] = None      # None = all balanced items
    HIGH_FREQ_PERCENTILE: float = 66.0             # extreme tertiles for HF / LF
    LOW_FREQ_PERCENTILE: float = 33.0
    TOKEN_STRATA: Tuple[int, ...] = (1, 2, 3, 4)   # last stratum = ">= 4"

    # ── Splits / probes ───────────────────────────────────────────────
    TEST_SIZE: float = 0.15
    VAL_SIZE: float = 0.15
    PROBE_SEEDS: List[int] = field(default_factory=lambda: [42, 123, 2024])
    PROBE_L2_GRID: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2, 1e-1])
    PROBE_MAX_ITER: int = 500
    NONFINITE_MAX_FRACTION: float = 0.01           # layer is invalid above this

    # ── KV sharing ────────────────────────────────────────────────────
    REUSE_EXEMPT_FRACTION: float = 0.20
    REUSE_FACTORS: List[int] = field(default_factory=lambda: [2])   # 2 = CLA2
    GATE_FRACTION: float = 1.0 / 3.0               # gated policies keep k = round(f*|T|)
    GATED_POLICIES: List[str] = field(default_factory=lambda: ["cka_high", "cka_low", "fidelity_high"])
    N_RANDOM_DRAWS: int = 3
    DEPTH_BINS: int = 3
    COUNTERFACTUAL_ENABLED: bool = True
    CALIB_N_ITEMS: int = 1024                      # drawn from the TRAIN split
    CALIB_MAX_ROWS: int = 4096

    # ── Inference ─────────────────────────────────────────────────────
    N_BANDS: int = 3                               # early / middle / late eligible
    N_BOOTSTRAP: int = 2000
    N_BOOTSTRAP_FRAGILITY: int = 1000
    N_PERMUTATIONS: int = 10000
    ALPHA: float = 0.05
    DECISION_MIN_MODEL_FRACTION: float = 7.0 / 9.0
    H5_EARLY_DEPTH: float = 0.60

    # ── Task 4: matching ──────────────────────────────────────────────
    MATCH_COVARIATES: List[str] = field(default_factory=lambda: ["length", "ortho_n", "bg_mean"])
    MATCH_CALIPER_SD: float = 0.25                 # per-covariate SD units

    # ── Task 5: probe reliability ─────────────────────────────────────
    RELIABILITY_ENABLED: bool = True
    RELIABILITY_POLICIES: List[str] = field(default_factory=lambda: ["no_reuse", "full"])
    MLP_ARCHITECTURES: Dict[str, Tuple[int, ...]] = field(
        default_factory=lambda: {"mlp_1x256": (256,), "mlp_512x256": (512, 256)})
    MLP_EPOCHS: int = 30
    MLP_PATIENCE: int = 5
    MLP_LR: float = 1e-3
    MLP_WEIGHT_DECAY: float = 1e-2
    MLP_DROPOUT: float = 0.3
    MLP_BATCH: int = 256
    MDL_FRACTIONS: List[float] = field(default_factory=lambda: [
        0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.0625, 0.125, 0.25, 0.5, 1.0])

    # ── Task 6: mechanistic diagnostics ───────────────────────────────
    MECH_N_ITEMS: int = 2048

    # ── Fragility scan ────────────────────────────────────────────────
    FRAGILITY_ENABLED: bool = True

    # ── Task 3: downstream ────────────────────────────────────────────
    DOWNSTREAM_ENABLED: bool = True
    DOWNSTREAM_MODELS: List[str] = field(default_factory=list)   # empty = all
    PPL_WINDOW: int = 1024
    PPL_STRIDE: int = 512
    PPL_MAX_WINDOWS: int = 200
    LAMBADA_N: int = 500

    RESUME: bool = True
    DPI: int = 300

    def __post_init__(self):
        self.RESULTS_DIR = os.path.join(self.OUTPUT_DIR, "results")
        self.PAPER_DIR = os.path.join(self.OUTPUT_DIR, "paper_artifacts")
        for d in (self.OUTPUT_DIR, self.RESULTS_DIR, self.PAPER_DIR):
            os.makedirs(d, exist_ok=True)

    def model_dir(self, model_name: str) -> str:
        d = os.path.join(self.RESULTS_DIR, safe_name(model_name))
        os.makedirs(d, exist_ok=True)
        return d


# ════════════════════════════════════════════════════════════════════════════
# KV SUBSTITUTION ENGINE (attention-interface level)
# ════════════════════════════════════════════════════════════════════════════
#
# Every attention call of a model loaded with attn_implementation=KV_ATTN_IMPL is
# routed through `kv_substitution_attention`.  With no active controller it is
# exactly SDPA.  The (key, value) tensors it receives are post-norm, post-RoPE and
# pre-GQA-repeat, i.e. what a KV cache stores, so substituting them is the
# faithful model of cross-layer KV-cache sharing.

KV_ATTN_IMPL = "kv_substitution"
_BASE_ATTENTION = ALL_ATTENTION_FUNCTIONS["sdpa"]


class KVRouter:
    active = None


def kv_substitution_attention(module, query, key, value, attention_mask, **kwargs):
    ctl = KVRouter.active
    if ctl is None:
        return _BASE_ATTENTION(module, query, key, value, attention_mask, **kwargs)
    return ctl.attend(module, query, key, value, attention_mask, kwargs)


AttentionInterface.register(KV_ATTN_IMPL, kv_substitution_attention)
# Without a mask registration, transformers builds NO mask for a custom
# implementation; registering the SDPA mask keeps causal/padding masks identical.
AttentionMaskInterface.register(KV_ATTN_IMPL, sdpa_mask)


@contextmanager
def routed(controller):
    KVRouter.active = controller
    try:
        yield controller
    finally:
        KVRouter.active = None


class LiveSubstitution:
    """Target reads the K/V its (earlier) source produced in the same pass."""

    def __init__(self, source_map: Dict[int, int]):
        self.source_map = source_map
        self.sources = frozenset(source_map.values())
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def attend(self, module, query, key, value, mask, kwargs):
        li = module.layer_idx
        src = self.source_map.get(li)
        if src is not None:
            key, value = self.cache[src]
        if li in self.sources:
            self.cache[li] = (key, value)
        return _BASE_ATTENTION(module, query, key, value, mask, **kwargs)


class CleanCapture:
    """Pass 1 of CLEAN semantics: record source K/V of an unperturbed pass."""

    def __init__(self, sources):
        self.sources = frozenset(sources)
        self.cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def attend(self, module, query, key, value, mask, kwargs):
        if module.layer_idx in self.sources:
            self.cache[module.layer_idx] = (key, value)
        return _BASE_ATTENTION(module, query, key, value, mask, **kwargs)


class CleanInjection:
    """Pass 2 of CLEAN semantics: targets read the recorded clean source K/V."""

    def __init__(self, source_map: Dict[int, int], cache):
        self.source_map = source_map
        self.cache = cache

    def attend(self, module, query, key, value, mask, kwargs):
        src = self.source_map.get(module.layer_idx)
        if src is not None:
            key, value = self.cache[src]
        return _BASE_ATTENTION(module, query, key, value, mask, **kwargs)


class CriterionMeter:
    """
    Calibration pass (no substitution).  For every (target, source) pair it
    records K and V rows at stimulus-token positions, and accumulates the
    query-aware attention-output error  ||O(q_t, K_s, V_s) - O(q_t, K_t, V_t)||^2.
    """

    def __init__(self, pairs: Dict[int, int]):
        self.pairs = pairs
        self.sources = frozenset(pairs.values())
        self.row_mask: Optional[torch.Tensor] = None
        self.batch_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.rows: Dict[int, Dict[str, List[torch.Tensor]]] = defaultdict(lambda: {"k": [], "v": []})
        self.err_num: Dict[int, float] = defaultdict(float)
        self.err_den: Dict[int, float] = defaultdict(float)

    def begin_batch(self, row_mask: torch.Tensor):
        self.row_mask = row_mask
        self.batch_kv.clear()

    def _rows(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t, d = x.shape
        return x.permute(0, 2, 1, 3).reshape(b, t, h * d)[self.row_mask].float().cpu()

    def attend(self, module, query, key, value, mask, kwargs):
        li = module.layer_idx
        out = _BASE_ATTENTION(module, query, key, value, mask, **kwargs)
        if li in self.sources or li in self.pairs:
            self.rows[li]["k"].append(self._rows(key))
            self.rows[li]["v"].append(self._rows(value))
        if li in self.sources:
            self.batch_kv[li] = (key, value)
        src = self.pairs.get(li)
        if src is not None:
            ks, vs = self.batch_kv[src]
            o_sub = _BASE_ATTENTION(module, query, ks, vs, mask, **kwargs)[0]
            o_own = out[0]
            diff = (o_sub - o_own)[self.row_mask].float()
            self.err_num[li] += float(diff.pow(2).sum())
            self.err_den[li] += float(o_own[self.row_mask].float().pow(2).sum())
        return out


# ════════════════════════════════════════════════════════════════════════════
# POLICIES
# ════════════════════════════════════════════════════════════════════════════

def exempt_cutoff(num_layers: int, fraction: float) -> int:
    return min(int(math.ceil(num_layers * fraction)), num_layers - 1)


def group_heads(num_layers: int, reuse_factor: int, ec: int) -> Dict[int, int]:
    """CLA-style grouping of eligible layers [ec, L): each layer -> its group head."""
    return {li: ec + ((li - ec) // reuse_factor) * reuse_factor for li in range(ec, num_layers)}


def eligible_bands(num_layers: int, ec: int, n_bands: int) -> Dict[str, List[int]]:
    """Pre-registered contiguous equal-count bands over eligible layers + 'all'."""
    names = ["early", "middle", "late"] if n_bands == 3 else [f"band{i + 1}" for i in range(n_bands)]
    eligible = np.arange(ec, num_layers)
    bands = {n: [int(x) for x in part] for n, part in zip(names, np.array_split(eligible, n_bands))}
    bands["all"] = [int(x) for x in eligible]
    return bands


@dataclass
class PolicySpec:
    name: str                       # run name (unique), e.g. "random_same_ratio#2"
    family: str                     # policy family, e.g. "random_same_ratio"
    source_map: Dict[int, int]      # target layer -> source layer
    semantics: str                  # "live" | "clean"
    deployable: bool                # realisable at inference with a shared cache
    num_layers: int
    draw: int = 0

    def __post_init__(self):
        targets, sources = set(self.source_map), set(self.source_map.values())
        if targets & sources:
            raise ValueError(f"{self.name}: a source is also a target (reuse chain)")
        if self.semantics == "live" and any(s >= t for t, s in self.source_map.items()):
            raise ValueError(f"{self.name}: live semantics needs every source < target")
        if any(not (0 <= x < self.num_layers) for x in targets | sources):
            raise ValueError(f"{self.name}: layer index out of range")

    @property
    def targets(self) -> List[int]:
        return sorted(self.source_map)

    @property
    def kv_memory_fraction(self) -> float:
        """Exact fraction of layers whose K/V must be cached."""
        return 1.0 - len(self.source_map) / self.num_layers

    def to_json(self) -> Dict:
        return {"name": self.name, "family": self.family, "semantics": self.semantics,
                "deployable": self.deployable, "draw": self.draw,
                "num_layers": self.num_layers, "n_targets": len(self.source_map),
                "kv_memory_fraction": self.kv_memory_fraction,
                "source_map": {str(t): int(s) for t, s in sorted(self.source_map.items())}}


def _rng(*keys) -> np.random.Generator:
    return np.random.default_rng([SEED] + [zlib.crc32(str(k).encode()) for k in keys])


NO_REUSE = "no_reuse"


def no_reuse_spec(num_layers: int) -> PolicySpec:
    return PolicySpec(NO_REUSE, NO_REUSE, {}, "live", True, num_layers)


def build_policy_specs(num_layers: int, reuse_factor: int, ec: int,
                       criteria: pd.DataFrame, cfg: ExperimentConfig) -> List[PolicySpec]:
    """
    All non-control policies for one (model, reuse factor).  `criteria` has one
    row per target of the `full` map with columns target, source, cka,
    attn_out_rel_err.  Every gated / random policy keeps exactly k targets.
    """
    L = num_layers
    gm = group_heads(L, reuse_factor, ec)
    full_map = {t: s for t, s in gm.items() if t != s}
    targets = sorted(full_map)
    heads = sorted(set(gm.values()))
    specs = [PolicySpec("full", "full", dict(full_map), "live", True, L)]
    if not targets:
        return specs

    k = max(1, int(round(cfg.GATE_FRACTION * len(targets))))
    crit = criteria.set_index("target")
    score = {"cka_high": crit["cka"], "cka_low": -crit["cka"],
             "fidelity_high": -crit["attn_out_rel_err"]}
    bins = [set(int(x) for x in b) for b in np.array_split(np.arange(ec, L), cfg.DEPTH_BINS)]
    selections = {}
    if len(targets) >= 3:
        for fam in cfg.GATED_POLICIES:
            ranked = score[fam].reindex(targets).sort_values(ascending=False, kind="mergesort")
            sel = sorted(int(t) for t in ranked.index[:k])
            selections[fam] = sel
            specs.append(PolicySpec(fam, fam, {t: full_map[t] for t in sel}, "live", True, L))
    else:
        logger.warning(f"  only {len(targets)} targets: gated policies skipped")

    for d in range(1, cfg.N_RANDOM_DRAWS + 1):
        rng = _rng("random_same_ratio", L, reuse_factor, d)
        sel = sorted(int(t) for t in rng.choice(targets, size=k, replace=False))
        specs.append(PolicySpec(f"random_same_ratio#{d}", "random_same_ratio",
                                {t: full_map[t] for t in sel}, "live", True, L, d))
        for fam, gsel in selections.items():
            rng = _rng("depth_matched", fam, L, reuse_factor, d)
            dm = []
            for b in bins:
                pool = sorted(set(targets) & b)
                need = len(set(gsel) & b)
                dm += [int(t) for t in rng.choice(pool, size=need, replace=False)] if need else []
            specs.append(PolicySpec(f"rdm_{fam}#{d}", f"rdm_{fam}",
                                    {t: full_map[t] for t in sorted(dm)}, "live", True, L, d))
        rng = _rng("random_head_source", L, reuse_factor, d)
        topo = {}
        for t in targets:
            pool = [h for h in heads if h < t and h != full_map[t]]
            topo[t] = int(rng.choice(pool)) if pool else full_map[t]
        specs.append(PolicySpec(f"random_head_source#{d}", "random_head_source",
                                topo, "live", True, L, d))

    if cfg.COUNTERFACTUAL_ENABLED:
        nxt = {t: next((h for h in heads if h > t), None) for t in targets}
        cf_targets = [t for t in targets if nxt[t] is not None]
        if cf_targets:
            specs.append(PolicySpec("cf_prev_head", "cf_prev_head",
                                    {t: full_map[t] for t in cf_targets}, "clean", False, L))
            specs.append(PolicySpec("cf_next_head", "cf_next_head",
                                    {t: nxt[t] for t in cf_targets}, "clean", False, L))
    return specs


def single_layer_spec(num_layers: int, target: int) -> PolicySpec:
    """Fragility scan: substitute K/V at one layer only, from the previous layer."""
    return PolicySpec(f"single_{target}", "single_layer", {target: target - 1}, "live", True, num_layers)


# ════════════════════════════════════════════════════════════════════════════
# STIMULI AND SPLITS
# ════════════════════════════════════════════════════════════════════════════

def load_items(cfg: ExperimentConfig) -> pd.DataFrame:
    """
    Balanced word / nonword item table.  Only lowercase alphabetic strings are
    kept (nonwords are letter strings), duplicates are removed within a class and
    strings present in both classes are removed (their label is ambiguous and
    they would leak across the train/test split).
    """
    def read(path: str, is_word: int) -> pd.DataFrame:
        df = pd.read_csv(path)

        def num(col):
            if col not in df.columns:
                return np.nan
            return pd.to_numeric(df[col].replace("#", np.nan), errors="coerce").to_numpy()

        out = pd.DataFrame({"stimulus": df["Word"].astype(str).str.strip().str.lower()})
        out["is_word"] = is_word
        out["log_freq"] = num("Log_Freq_HAL") if is_word else np.nan
        out["ortho_n"] = num("Ortho_N")
        out["bg_mean"] = num("BG_Mean")
        out = out[out["stimulus"].str.fullmatch(r"[a-z]+")]
        return out.drop_duplicates("stimulus")

    words = read(cfg.WORDS_PATH, 1)
    nonwords = read(cfg.NONWORDS_PATH, 0)
    shared = set(words["stimulus"]) & set(nonwords["stimulus"])
    words = words[~words["stimulus"].isin(shared)]
    nonwords = nonwords[~nonwords["stimulus"].isin(shared)]
    n = min(len(words), len(nonwords))
    if cfg.MAX_ITEMS_PER_CLASS:
        n = min(n, int(cfg.MAX_ITEMS_PER_CLASS))
    words = words.sample(n=n, random_state=SEED)
    nonwords = nonwords.sample(n=n, random_state=SEED)
    items = pd.concat([words, nonwords], ignore_index=True)
    items["length"] = items["stimulus"].str.len().astype(float)

    items["freq_group"] = np.where(items["is_word"] == 1, "unknown", "nonword")
    fm = (items["is_word"] == 1) & items["log_freq"].notna()
    hi = np.percentile(items.loc[fm, "log_freq"], cfg.HIGH_FREQ_PERCENTILE)
    lo = np.percentile(items.loc[fm, "log_freq"], cfg.LOW_FREQ_PERCENTILE)
    items.loc[fm, "freq_group"] = "mid"
    items.loc[fm & (items["log_freq"] >= hi), "freq_group"] = "high"
    items.loc[fm & (items["log_freq"] <= lo), "freq_group"] = "low"

    items = items.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    items.insert(0, "item_id", np.arange(len(items)))
    logger.info(f"Items: {len(items)} ({n} words / {n} nonwords); "
                f"HF>={hi:.3f}: {(items.freq_group == 'high').sum()}, "
                f"LF<={lo:.3f}: {(items.freq_group == 'low').sum()}, "
                f"removed cross-class duplicates: {len(shared)}")
    return items


@dataclass
class Splits:
    test: np.ndarray
    train: Dict[int, np.ndarray]
    val: Dict[int, np.ndarray]


def make_splits(items: pd.DataFrame, cfg: ExperimentConfig) -> Splits:
    """Test split fixed by SEED (same items for every seed, policy and model)."""
    strat = np.where(items["is_word"] == 0, "nonword", "word_" + items["freq_group"].astype(str))
    idx = np.arange(len(items))
    rest, test = train_test_split(idx, test_size=cfg.TEST_SIZE, stratify=strat, random_state=SEED)
    vsz = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
    train, val = {}, {}
    for s in cfg.PROBE_SEEDS:
        tr, va = train_test_split(rest, test_size=vsz, stratify=strat[rest], random_state=s)
        train[s], val[s] = np.sort(tr), np.sort(va)
    return Splits(np.sort(test), train, val)


# ════════════════════════════════════════════════════════════════════════════
# MODEL RUNNER
# ════════════════════════════════════════════════════════════════════════════

def unbiased_linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Linear CKA with the unbiased HSIC estimator (Song et al., 2012). Rows = samples."""
    X = X.to(DEVICE, torch.float64)
    Y = Y.to(DEVICE, torch.float64)
    n = X.shape[0]
    if n < 8:
        return float("nan")
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)

    def hsic(K, M):
        K = K.clone(); M = M.clone()
        K.fill_diagonal_(0.0); M.fill_diagonal_(0.0)
        t1 = (K * M).sum()
        t2 = K.sum() * M.sum() / ((n - 1) * (n - 2))
        t3 = 2.0 * (K.sum(0) @ M.sum(1)) / (n - 2)
        return (t1 + t2 - t3) / (n * (n - 3))

    K, M = X @ X.T, Y @ Y.T
    den = torch.sqrt(hsic(K, K) * hsic(M, M))
    return float(hsic(K, M) / den) if den > 0 else float("nan")


class ModelRunner:
    """One frozen CausalLM with length-bucketed, padding-free batched forwards."""

    def __init__(self, mc: ModelConfig, cfg: ExperimentConfig, items: pd.DataFrame):
        self.mc, self.cfg = mc, cfg
        logger.info(f"Loading {mc.name} ({mc.model_id})")
        self.tokenizer = AutoTokenizer.from_pretrained(mc.model_id)
        dtype_kw = ({"dtype": COMPUTE_DTYPE} if Version(transformers.__version__) >= Version("4.56")
                    else {"torch_dtype": COMPUTE_DTYPE})
        self.model = AutoModelForCausalLM.from_pretrained(
            mc.model_id, attn_implementation=KV_ATTN_IMPL, low_cpu_mem_usage=True, **dtype_kw)
        self.model.to(DEVICE).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.layers = self.model.get_decoder().layers
        self.L = len(self.layers)
        self.H = int(self.model.config.hidden_size)
        for i, layer in enumerate(self.layers):
            if getattr(layer.self_attn, "layer_idx", None) != i:
                raise RuntimeError(f"{mc.name}: layer {i} has no matching self_attn.layer_idx")
        self.store_dtype = COMPUTE_DTYPE
        self.batch_size = mc.batch_size
        bos = self.tokenizer.bos_token_id
        self.adds_bos = bool(bos is not None and self.tokenizer("a").input_ids[:1] == [bos])
        self._encode(items["stimulus"].tolist())
        logger.info(f"  layers={self.L} hidden={self.H} adds_bos={self.adds_bos} "
                    f"valid_items={int(self.valid.sum())}/{len(self.valid)} "
                    f"n_tokens: " + ", ".join(f"{k}:{v}" for k, v in
                                              sorted(pd.Series(self.n_tokens[self.valid]).value_counts().items())))

    # ── Tokenisation ──────────────────────────────────────────────────
    def _encode(self, stimuli: List[str]):
        template = self.cfg.STIMULUS_TEMPLATE
        if not template.endswith("{stimulus}") or template.count("{stimulus}") != 1:
            raise ValueError("STIMULUS_TEMPLATE must contain '{stimulus}' once, at the end")
        prefix = template[: -len("{stimulus}")]
        start = len(prefix)
        enc = self.tokenizer([prefix + s for s in stimuli], add_special_tokens=True,
                             return_offsets_mapping=True, return_special_tokens_mask=True)
        self.ids, n_tok = [], []
        for ids, offs, special, s in zip(enc["input_ids"], enc["offset_mapping"],
                                         enc["special_tokens_mask"], stimuli):
            end = start + len(s)
            inside = [(not sp) and o[1] > start and o[0] < end for o, sp in zip(offs, special)]
            n = int(sum(inside))
            contiguous = n >= 1 and all(inside[len(ids) - n:])
            self.ids.append(ids)
            n_tok.append(n if contiguous else -1)
        self.n_tokens = np.asarray(n_tok)
        self.seq_len = np.asarray([len(x) for x in self.ids])
        self.valid = self.n_tokens >= 1
        n_bad = int((~self.valid).sum())
        if n_bad:
            logger.warning(f"  {n_bad} items have no clean stimulus/carrier token boundary "
                           f"for this tokenizer and are excluded for this model")

    # ── Batching ──────────────────────────────────────────────────────
    def run_batches(self, item_idx: np.ndarray, step: Callable[[np.ndarray, torch.Tensor], None]):
        """Equal-length batches (no padding); halves the batch size on CUDA OOM."""
        lengths = self.seq_len[item_idx]
        queue = []
        for length in np.unique(lengths):
            pos = np.flatnonzero(lengths == length)
            queue += [pos[i:i + self.batch_size] for i in range(0, len(pos), self.batch_size)]
        while queue:
            pos = queue.pop(0)
            ids = torch.tensor([self.ids[item_idx[p]] for p in pos], dtype=torch.long, device=DEVICE)
            try:
                step(pos, ids)
            except torch.cuda.OutOfMemoryError:
                free_memory()
                if len(pos) == 1:
                    raise
                half = len(pos) // 2
                self.batch_size = max(1, half)
                logger.warning(f"  OOM: batch size -> {self.batch_size}")
                queue[:0] = [pos[:half], pos[half:]]

    def forward(self, ids: torch.Tensor, spec: Optional[PolicySpec], logits_to_keep: int = 1):
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids),
                  use_cache=False, logits_to_keep=logits_to_keep)
        if spec is None or not spec.source_map:
            return self.model(**kw)
        if spec.semantics == "live":
            with routed(LiveSubstitution(spec.source_map)):
                return self.model(**kw)
        with routed(CleanCapture(spec.source_map.values())) as cap:
            self.model(**{**kw, "logits_to_keep": 1})
        with routed(CleanInjection(spec.source_map, cap.cache)):
            return self.model(**kw)

    @contextmanager
    def recording(self, layer_ids: Sequence[int]):
        """Forward hooks storing each decoder layer's output at the last position."""
        rec: Dict[int, torch.Tensor] = {}

        def make(i):
            def hook(_m, _inp, out):
                h = out[0] if isinstance(out, tuple) else out
                rec[i] = h[:, -1, :].detach()
            return hook

        handles = [self.layers[i].register_forward_hook(make(i)) for i in layer_ids]
        try:
            yield rec
        finally:
            for h in handles:
                h.remove()

    # ── Readout extraction ────────────────────────────────────────────
    @torch.no_grad()
    def collect(self, item_idx: np.ndarray, spec: Optional[PolicySpec],
                layer_ids: Optional[Sequence[int]] = None):
        """
        Last-stimulus-token state of every requested layer for every item.
        Returns ({layer: CPU tensor (n, H)}, {layer: n_nonfinite_rows}).
        Non-finite rows are counted BEFORE they are zeroed.
        """
        layer_ids = list(range(self.L)) if layer_ids is None else list(layer_ids)
        n = len(item_idx)
        store = {li: torch.empty((n, self.H), dtype=self.store_dtype) for li in layer_ids}
        nonfinite = {li: 0 for li in layer_ids}

        def step(pos, ids):
            with self.recording(layer_ids) as rec:
                self.forward(ids, spec)
            tpos = torch.as_tensor(pos)
            for li in layer_ids:
                h = rec[li].float()
                bad = ~torch.isfinite(h).all(-1)
                if bad.any():
                    nonfinite[li] += int(bad.sum())
                    h[bad] = 0.0
                store[li][tpos] = h.to(self.store_dtype).cpu()

        self.run_batches(item_idx, step)
        return store, nonfinite

    # ── Calibration of partner criteria ───────────────────────────────
    @torch.no_grad()
    def calibrate(self, item_idx: np.ndarray, pairs: Dict[int, int]) -> Tuple[pd.DataFrame, Dict]:
        """
        Per (target, source) pair of the `full` map, measured on a clean pass:
          cka_k, cka_v, cka = mean   unbiased linear CKA over stimulus-token rows
          rel_kv_distance            ||[K_s;V_s]-[K_t;V_t]|| / ||[K_t;V_t]||
          attn_out_rel_err           ||O(q_t,K_s,V_s)-O(q_t,K_t,V_t)|| / ||O(q_t,K_t,V_t)||
        plus split-half reliability of the CKA ranking and agreement between criteria.
        """
        meter = CriterionMeter(pairs)

        def step(pos, ids):
            mask = torch.zeros(ids.shape, dtype=torch.bool, device=DEVICE)
            for b, p in enumerate(pos):
                mask[b, ids.shape[1] - self.n_tokens[item_idx[p]]:] = True
            meter.begin_batch(mask)
            with routed(meter):
                self.model(input_ids=ids, attention_mask=torch.ones_like(ids),
                           use_cache=False, logits_to_keep=1)

        self.run_batches(item_idx, step)
        rows = {li: {kv: torch.cat(v) for kv, v in d.items()} for li, d in meter.rows.items()}
        n_rows = next(iter(rows.values()))["k"].shape[0]
        rng = np.random.default_rng(SEED)
        sel = np.sort(rng.choice(n_rows, size=min(n_rows, self.cfg.CALIB_MAX_ROWS), replace=False))
        perm = rng.permutation(sel)
        halves = (np.sort(perm[: len(perm) // 2]), np.sort(perm[len(perm) // 2:]))

        out = []
        for t, s in sorted(pairs.items()):
            Kt, Vt, Ks, Vs = rows[t]["k"], rows[t]["v"], rows[s]["k"], rows[s]["v"]
            cka_k = unbiased_linear_cka(Kt[sel], Ks[sel])
            cka_v = unbiased_linear_cka(Vt[sel], Vs[sel])
            half = [0.5 * (unbiased_linear_cka(Kt[h], Ks[h]) + unbiased_linear_cka(Vt[h], Vs[h]))
                    for h in halves]
            num = (Ks[sel] - Kt[sel]).pow(2).sum() + (Vs[sel] - Vt[sel]).pow(2).sum()
            den = Kt[sel].pow(2).sum() + Vt[sel].pow(2).sum()
            out.append({"target": t, "source": s, "cka_k": cka_k, "cka_v": cka_v,
                        "cka": 0.5 * (cka_k + cka_v), "cka_half1": half[0], "cka_half2": half[1],
                        "rel_kv_distance": float(torch.sqrt(num / den)),
                        "attn_out_rel_err": math.sqrt(meter.err_num[t] / meter.err_den[t])})
        df = pd.DataFrame(out)
        summary = {"n_rows": int(len(sel)), "n_pairs": int(len(df))}
        if len(df) >= 3:
            summary.update({
                "cka_split_half_spearman": float(stats.spearmanr(df.cka_half1, df.cka_half2)[0]),
                "kendall_cka_vs_attn_fidelity": float(stats.kendalltau(df.cka, -df.attn_out_rel_err)[0]),
                "kendall_cka_vs_kv_similarity": float(stats.kendalltau(df.cka, -df.rel_kv_distance)[0]),
                "kendall_attn_fidelity_vs_kv_similarity":
                    float(stats.kendalltau(-df.attn_out_rel_err, -df.rel_kv_distance)[0])})
        return df, summary

    # ── Mechanistic diagnostics (Task 6) ──────────────────────────────
    @torch.no_grad()
    def mechanistic(self, item_idx: np.ndarray, spec: PolicySpec) -> pd.DataFrame:
        """Clean vs policy states (all layers) and next-token distributions."""
        n = len(item_idx)
        clean = {li: torch.empty((n, self.H), dtype=self.store_dtype) for li in range(self.L)}
        pol = {li: torch.empty((n, self.H), dtype=self.store_dtype) for li in range(self.L)}
        kl = torch.empty(n)
        top1 = torch.empty(n)

        def step(pos, ids):
            tpos = torch.as_tensor(pos)
            with self.recording(range(self.L)) as rec:
                lc = self.forward(ids, None).logits[:, -1].float().log_softmax(-1)
            for li in range(self.L):
                clean[li][tpos] = rec[li].to(self.store_dtype).cpu()
            with self.recording(range(self.L)) as rec:
                lp = self.forward(ids, spec).logits[:, -1].float().log_softmax(-1)
            for li in range(self.L):
                pol[li][tpos] = rec[li].to(self.store_dtype).cpu()
            kl[tpos] = (lc.exp() * (lc - lp)).sum(-1).cpu()
            top1[tpos] = (lc.argmax(-1) == lp.argmax(-1)).float().cpu()

        self.run_batches(item_idx, step)
        rows = []
        for li in range(self.L):
            c, p = clean[li].float(), pol[li].float()
            cos = F.cosine_similarity(c, p, dim=-1)
            rel = (p - c).norm(dim=-1) / c.norm(dim=-1).clamp_min(1e-8)
            rows.append({"layer": li, "hidden_cka_unbiased": unbiased_linear_cka(c, p),
                         "cosine_mean": float(cos.mean()), "relative_shift_mean": float(rel.mean()),
                         "next_token_kl_mean": float(kl.mean()), "next_token_top1_agreement": float(top1.mean())})
        return pd.DataFrame(rows)

    def release(self):
        del self.model
        free_memory()


# ════════════════════════════════════════════════════════════════════════════
# PROBES
# ════════════════════════════════════════════════════════════════════════════

def standardizer(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    return X.mean(0), X.std(0).clamp_min(1e-6)


def fit_logistic(X: torch.Tensor, y: torch.Tensor, lam: float, max_iter: int):
    """L2-regularised logistic regression; strictly convex, zero init, full-batch L-BFGS."""
    with torch.enable_grad():
        w = torch.zeros(X.shape[1], device=X.device, requires_grad=True)
        b = torch.zeros(1, device=X.device, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=max_iter, history_size=20,
                                line_search_fn="strong_wolfe",
                                tolerance_grad=1e-7, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(X @ w + b, y) + 0.5 * lam * w.dot(w)
            loss.backward()
            return loss

        opt.step(closure)
    return w.detach(), b.detach()


@dataclass
class ProbeBank:
    """no_reuse linear probes: frozen readout for every other policy."""
    mu: np.ndarray      # (S, L, H)
    sd: np.ndarray      # (S, L, H)
    w: np.ndarray       # (S, L, H)
    b: np.ndarray       # (S, L)
    lam: np.ndarray     # (L,)

    def save(self, path: str):
        np.savez_compressed(path, mu=self.mu, sd=self.sd, w=self.w, b=self.b, lam=self.lam)

    @classmethod
    def load(cls, path: str) -> "ProbeBank":
        z = np.load(path)
        return cls(z["mu"], z["sd"], z["w"], z["b"], z["lam"])

    def logits(self, X: torch.Tensor, layer: int) -> torch.Tensor:
        """Seed-averaged frozen-probe logits (n,) for states X (n, H) on DEVICE."""
        out = torch.zeros(X.shape[0], device=X.device)
        for s in range(self.w.shape[0]):
            mu, sd, w = (torch.as_tensor(a[s, layer], device=X.device) for a in (self.mu, self.sd, self.w))
            out += ((X - mu) / sd) @ w + float(self.b[s, layer])
        return out / self.w.shape[0]


class LinearProbeRunner:
    """Primary readout: multi-seed linear probes on a fixed test split."""

    def __init__(self, cfg: ExperimentConfig, splits: Splits, y: np.ndarray):
        self.cfg, self.splits = cfg, splits
        self.y = torch.tensor(np.asarray(y, dtype=np.float32), device=DEVICE)
        self.y_test = y[splits.test]

    def select_lambda(self, X: torch.Tensor) -> float:
        s0 = self.cfg.PROBE_SEEDS[0]
        tr, va = self.splits.train[s0], self.splits.val[s0]
        mu, sd = standardizer(X[tr])
        Ztr, Zva = (X[tr] - mu) / sd, (X[va] - mu) / sd
        losses = []
        for lam in self.cfg.PROBE_L2_GRID:
            w, b = fit_logistic(Ztr, self.y[tr], lam, self.cfg.PROBE_MAX_ITER)
            losses.append(float(F.binary_cross_entropy_with_logits(Zva @ w + b, self.y[va])))
        return float(self.cfg.PROBE_L2_GRID[int(np.argmin(losses))])

    def fit_layer(self, X: torch.Tensor, lam: float):
        """Returns test logits per seed (S, n_test) and the fitted parameters."""
        te = self.splits.test
        logits, params = [], []
        for s in self.cfg.PROBE_SEEDS:
            tr = self.splits.train[s]
            mu, sd = standardizer(X[tr])
            w, b = fit_logistic((X[tr] - mu) / sd, self.y[tr], lam, self.cfg.PROBE_MAX_ITER)
            logits.append((((X[te] - mu) / sd) @ w + b).cpu().numpy())
            params.append((mu.cpu().numpy(), sd.cpu().numpy(), w.cpu().numpy(), float(b)))
        return np.stack(logits), params


class MLPProbe(nn.Module):
    def __init__(self, d_in: int, hidden: Tuple[int, ...], dropout: float):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(Ztr, ytr, Zva, yva, hidden, cfg: ExperimentConfig, seed: int) -> MLPProbe:
    seed_everything(seed)
    model = MLPProbe(Ztr.shape[1], hidden, cfg.MLP_DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.MLP_LR, weight_decay=cfg.MLP_WEIGHT_DECAY)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, patience = float("inf"), None, 0
    with torch.enable_grad():
        for _ in range(cfg.MLP_EPOCHS):
            model.train()
            for idx in torch.randperm(Ztr.shape[0], generator=gen).split(cfg.MLP_BATCH):
                if len(idx) < 2:
                    continue
                opt.zero_grad()
                F.binary_cross_entropy_with_logits(model(Ztr[idx.to(DEVICE)]), ytr[idx.to(DEVICE)]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vl = float(F.binary_cross_entropy_with_logits(model(Zva), yva))
            if vl < best - 1e-6:
                best, patience = vl, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                patience += 1
                if patience >= cfg.MLP_PATIENCE:
                    break
    model.load_state_dict(best_state)
    return model.eval()


def online_codelength(X: torch.Tensor, y: torch.Tensor, lam: float,
                      fractions: Sequence[float], max_iter: int, seed: int) -> Tuple[float, float]:
    """
    Online (prequential) code length in bits of the labels given the
    representations (Voita & Titov, 2020).  Returns (online_bits, compression),
    compression = uniform_bits / online_bits with uniform_bits = n * log2(2).
    """
    n = X.shape[0]
    order = torch.as_tensor(np.random.default_rng(seed).permutation(n), device=X.device)
    sizes = sorted({max(2, int(round(f * n))) for f in fractions} | {n})
    bits = float(sizes[0])
    for prev, cur in zip(sizes[:-1], sizes[1:]):
        tr, nxt = order[:prev], order[prev:cur]
        mu, sd = standardizer(X[tr])
        w, b = fit_logistic((X[tr] - mu) / sd, y[tr], lam, max_iter)
        z = ((X[nxt] - mu) / sd) @ w + b
        bits += float(F.binary_cross_entropy_with_logits(z, y[nxt], reduction="sum")) / math.log(2)
    return bits, n / bits


# ════════════════════════════════════════════════════════════════════════════
# PER-MODEL PIPELINE
# ════════════════════════════════════════════════════════════════════════════

class ModelPipeline:
    """
    Order per model:
      1. no_reuse control (lambda selection, probe bank, retrained = frozen readout)
      2. per reuse factor: criterion calibration -> policy specs -> one run per spec
      3. single-layer fragility scan (frozen readout)
      4. downstream generative evaluation (primary reuse factor)
    Every run is saved to disk and skipped on resume.
    """

    def __init__(self, cfg: ExperimentConfig, mc: ModelConfig, items: pd.DataFrame, splits: Splits):
        self.cfg, self.mc, self.items = cfg, mc, items
        self.dir = cfg.model_dir(mc.name)
        self.runner = ModelRunner(mc, cfg, items)
        v = self.runner.valid
        self.splits = Splits(splits.test[v[splits.test]],
                             {s: a[v[a]] for s, a in splits.train.items()},
                             {s: a[v[a]] for s, a in splits.val.items()})
        self.L = self.runner.L
        self.y = items["is_word"].to_numpy()
        self.probes = LinearProbeRunner(cfg, self.splits, self.y)
        meta = items.copy()
        meta["n_tokens"] = self.runner.n_tokens
        meta["valid"] = v
        meta.to_csv(os.path.join(self.dir, "items.csv"), index=False)
        with open(os.path.join(self.dir, "model_info.json"), "w") as f:
            json.dump({"model": mc.name, "model_id": mc.model_id, "num_layers": self.L,
                       "hidden_size": self.runner.H, "adds_bos": self.runner.adds_bos,
                       "template": cfg.STIMULUS_TEMPLATE, "compute_dtype": str(COMPUTE_DTYPE),
                       "test_items": self.splits.test.tolist()}, f)
        self.reliability_layers = sorted({self.L // 4, self.L // 2, (3 * self.L) // 4, self.L - 1})

    # ── paths ─────────────────────────────────────────────────────────
    def run_path(self, rf: Optional[int], name: str, ext: str) -> str:
        sub = self.dir if rf is None else os.path.join(self.dir, f"rf{rf}")
        os.makedirs(sub, exist_ok=True)
        return os.path.join(sub, f"{safe_name(name)}.{ext}")

    # ── orchestration ─────────────────────────────────────────────────
    def run(self):
        bank = self._run_no_reuse()
        primary_specs = []
        for rf in self.cfg.REUSE_FACTORS:
            specs = self._policy_specs(rf)
            if rf == self.cfg.REUSE_FACTORS[0]:
                primary_specs = specs
            for spec in specs:
                self._run_policy(rf, spec, bank)
        if self.cfg.FRAGILITY_ENABLED:
            self._fragility_scan(bank)
        wanted = self.cfg.DOWNSTREAM_MODELS
        if self.cfg.DOWNSTREAM_ENABLED and (not wanted or self.mc.name in wanted):
            DownstreamEvaluator(self.cfg, self.runner).run(
                [no_reuse_spec(self.L)] + primary_specs,
                self.run_path(None, f"downstream_rf{self.cfg.REUSE_FACTORS[0]}", "csv"))
        self.runner.release()

    # ── step 1: control ───────────────────────────────────────────────
    def _run_no_reuse(self) -> ProbeBank:
        npz, bank_path = self.run_path(None, NO_REUSE, "npz"), self.run_path(None, "probe_bank", "npz")
        if self.cfg.RESUME and os.path.exists(npz) and os.path.exists(bank_path):
            logger.info(f"  RESUME {self.mc.name}/{NO_REUSE}")
            return ProbeBank.load(bank_path)
        spec = no_reuse_spec(self.L)
        t0 = time.time()
        states, nonfinite = self.runner.collect(np.arange(len(self.items)), spec)
        S, H = len(self.cfg.PROBE_SEEDS), self.runner.H
        bank = ProbeBank(np.zeros((S, self.L, H), np.float32), np.ones((S, self.L, H), np.float32),
                         np.zeros((S, self.L, H), np.float32), np.zeros((S, self.L), np.float32),
                         np.full(self.L, np.nan))
        logits, auc_seed = self._probe_all(states, nonfinite, spec, bank=bank, fill_bank=True)
        bank.save(bank_path)
        self._save_run(None, spec, logits, logits, auc_seed, nonfinite, time.time() - t0)
        return bank

    # ── step 2: policies ──────────────────────────────────────────────
    def _policy_specs(self, rf: int) -> List[PolicySpec]:
        ec = exempt_cutoff(self.L, self.cfg.REUSE_EXEMPT_FRACTION)
        full_map = {t: s for t, s in group_heads(self.L, rf, ec).items() if t != s}
        crit_path = self.run_path(rf, "calibration", "csv")
        if self.cfg.RESUME and os.path.exists(crit_path):
            crit = pd.read_csv(crit_path)
        elif full_map:
            s0 = self.cfg.PROBE_SEEDS[0]
            calib_idx = np.sort(np.random.default_rng(SEED).choice(
                self.splits.train[s0], size=min(self.cfg.CALIB_N_ITEMS, len(self.splits.train[s0])),
                replace=False))
            crit, summary = self.runner.calibrate(calib_idx, full_map)
            crit.to_csv(crit_path, index=False)
            with open(self.run_path(rf, "calibration_summary", "json"), "w") as f:
                json.dump(summary, f, indent=2)
            logger.info(f"  [calib rf={rf}] {summary}")
        else:
            crit = pd.DataFrame(columns=["target", "source", "cka", "attn_out_rel_err"])
        specs = build_policy_specs(self.L, rf, ec, crit, self.cfg)
        self._check_distinct(rf, specs)
        return specs

    def _check_distinct(self, rf: int, specs: List[PolicySpec]):
        """
        Two deterministic policies with identical source maps are one experiment,
        not two.  A random draw that coincides with another set is a legitimate
        draw from its null distribution and is only logged.
        """
        seen = {}
        for s in specs:
            key = (s.semantics, tuple(sorted(s.source_map.items())))
            if key in seen:
                other = seen[key]
                level = logging.INFO if (s.draw or other.draw) else logging.WARNING
                logger.log(level, f"  [rf={rf}] {s.name} has the same source map as {other.name}"
                                  + ("" if level == logging.INFO else
                                     "; they must not be reported as distinct conditions"))
            seen.setdefault(key, s)

    def _run_policy(self, rf: int, spec: PolicySpec, bank: ProbeBank):
        npz = self.run_path(rf, spec.name, "npz")
        if self.cfg.RESUME and os.path.exists(npz):
            logger.info(f"  RESUME {self.mc.name}/rf{rf}/{spec.name}")
            return
        logger.info(f"  RUN {self.mc.name}/rf{rf}/{spec.name}: targets={spec.targets} "
                    f"semantics={spec.semantics} kv_mem={spec.kv_memory_fraction:.3f}")
        t0 = time.time()
        states, nonfinite = self.runner.collect(np.arange(len(self.items)), spec)
        logits_rt, auc_seed, logits_fz = self._probe_all(states, nonfinite, spec, bank=bank,
                                                         fill_bank=False, rf=rf)
        self._save_run(rf, spec, logits_rt, logits_fz, auc_seed, nonfinite, time.time() - t0)
        if self.cfg.MECH_N_ITEMS > 0:
            mech_idx = self.splits.test[: self.cfg.MECH_N_ITEMS]
            self.runner.mechanistic(mech_idx, spec).assign(
                model=self.mc.name, reuse_factor=rf, policy=spec.name, family=spec.family
            ).to_csv(self.run_path(rf, f"{spec.name}_mechanistic", "csv"), index=False)

    # ── probing ───────────────────────────────────────────────────────
    def _probe_all(self, states, nonfinite, spec: PolicySpec, bank: ProbeBank,
                   fill_bank: bool, rf: Optional[int] = None):
        n_test, S = len(self.splits.test), len(self.cfg.PROBE_SEEDS)
        rt = np.full((self.L, n_test), np.nan, np.float32)
        fz = np.full((self.L, n_test), np.nan, np.float32)
        auc_seed = np.full((S, self.L), np.nan)
        y_test = self.probes.y_test
        keep_for_reliability = {}
        for li in tqdm(range(self.L), desc=f"probe {self.mc.name}/{spec.name}", leave=False):
            frac = nonfinite[li] / len(self.items)
            if frac > self.cfg.NONFINITE_MAX_FRACTION or (not fill_bank and np.isnan(bank.lam[li])):
                logger.warning(f"  layer {li}: {frac:.2%} non-finite rows or invalid control "
                               f"layer -> layer excluded")
                states[li] = None
                continue
            X = states[li].to(DEVICE, torch.float32)
            if fill_bank:
                bank.lam[li] = self.probes.select_lambda(X)
            lam = float(bank.lam[li])
            seed_logits, params = self.probes.fit_layer(X, lam)
            rt[li] = seed_logits.mean(0)
            auc_seed[:, li] = [roc_auc_score(y_test, z) for z in seed_logits]
            if fill_bank:
                for s, (mu, sd, w, b) in enumerate(params):
                    bank.mu[s, li], bank.sd[s, li], bank.w[s, li], bank.b[s, li] = mu, sd, w, b
            else:
                fz[li] = bank.logits(X[torch.as_tensor(self.splits.test, device=DEVICE)], li).cpu().numpy()
            if (self.cfg.RELIABILITY_ENABLED and spec.family in self.cfg.RELIABILITY_POLICIES
                    and li in self.reliability_layers):
                keep_for_reliability[li] = (states[li], lam)
            states[li] = None
            del X
            free_memory()
        if keep_for_reliability:
            self._reliability(keep_for_reliability, spec, rf)
        return (rt, auc_seed) if fill_bank else (rt, auc_seed, fz)

    def _save_run(self, rf, spec: PolicySpec, rt, fz, auc_seed, nonfinite, seconds):
        np.savez_compressed(self.run_path(rf, spec.name, "npz"), test_items=self.splits.test,
                            logit_retrained=rt, logit_frozen=fz, auc_seed=auc_seed,
                            nonfinite=np.array([nonfinite[i] for i in range(self.L)]))
        meta = spec.to_json() | {"model": self.mc.name, "reuse_factor": rf, "seconds": seconds,
                                 "exempt_cutoff": (exempt_cutoff(self.L, self.cfg.REUSE_EXEMPT_FRACTION)
                                                   if rf is not None else None)}
        with open(self.run_path(rf, spec.name, "json"), "w") as f:
            json.dump(meta, f, indent=2)

    # ── Task 5: probe reliability ─────────────────────────────────────
    def _reliability(self, layers: Dict[int, Tuple[torch.Tensor, float]], spec: PolicySpec, rf):
        """Linear and MLP probes vs random-label controls (selectivity) + online MDL."""
        cfg, sp = self.cfg, self.splits
        y = self.probes.y
        te = torch.as_tensor(sp.test, device=DEVICE)
        rows = []
        for li, (Xcpu, lam) in layers.items():
            X = Xcpu.to(DEVICE, torch.float32)
            for s in cfg.PROBE_SEEDS:
                tr = torch.as_tensor(sp.train[s], device=DEVICE)
                va = torch.as_tensor(sp.val[s], device=DEVICE)
                g = torch.Generator(device="cpu").manual_seed(s)
                y_ctrl = y.clone()
                y_ctrl[tr] = y[tr][torch.randperm(len(tr), generator=g).to(DEVICE)]
                mu, sd = standardizer(X[tr])
                Z = (X - mu) / sd
                for task, yy in (("real", y), ("control", y_ctrl)):
                    w, b = fit_logistic(Z[tr], yy[tr], lam, cfg.PROBE_MAX_ITER)
                    rows.append(self._rel_row(li, "linear", task, s, (Z[te] @ w + b)))
                    for arch, hidden in cfg.MLP_ARCHITECTURES.items():
                        m = fit_mlp(Z[tr], yy[tr], Z[va], yy[va], hidden, cfg, s)
                        with torch.no_grad():
                            rows.append(self._rel_row(li, arch, task, s, m(Z[te])))
                        del m
                    if s == cfg.PROBE_SEEDS[0]:
                        bits, comp = online_codelength(X[tr], yy[tr], lam, cfg.MDL_FRACTIONS,
                                                       cfg.PROBE_MAX_ITER, s)
                        rows.append({"layer": li, "probe": "linear_mdl", "task": task, "seed": s,
                                     "mdl_bits": bits, "mdl_compression": comp})
            del X
            free_memory()
        df = pd.DataFrame(rows).assign(model=self.mc.name, policy=spec.name, reuse_factor=rf)
        df.to_csv(self.run_path(rf, f"{spec.name}_reliability", "csv"), index=False)

    def _rel_row(self, li, probe, task, seed, z: torch.Tensor) -> Dict:
        z = z.detach().cpu().numpy()
        yt = self.probes.y_test
        return {"layer": li, "probe": probe, "task": task, "seed": seed, "n_test": len(yt),
                "test_accuracy": float(((z > 0).astype(int) == yt).mean()),
                "test_auc": float(roc_auc_score(yt, z))}

    # ── step 3: single-layer fragility scan ───────────────────────────
    def _fragility_scan(self, bank: ProbeBank):
        path = self.run_path(None, "fragility", "npz")
        if self.cfg.RESUME and os.path.exists(path):
            logger.info(f"  RESUME {self.mc.name}/fragility")
            return
        base = np.load(self.run_path(None, NO_REUSE, "npz"))
        peak = int(np.nanargmax(np.nanmean(base["auc_seed"], 0)))
        readouts = sorted({peak, self.L - 1})
        te = self.splits.test

        def frozen_logits(states):
            return np.stack([bank.logits(states[r].to(DEVICE, torch.float32), r).cpu().numpy()
                             for r in readouts])

        base_states, _ = self.runner.collect(te, None, readouts)
        base_logits = frozen_logits(base_states)
        out = np.full((self.L, len(readouts), len(te)), np.nan, np.float32)
        for t in tqdm(range(1, self.L), desc=f"fragility {self.mc.name}"):
            st, _ = self.runner.collect(te, single_layer_spec(self.L, t), readouts)
            out[t] = frozen_logits(st)
        np.savez_compressed(path, test_items=te, readouts=np.array(readouts),
                            base_logits=base_logits, logits=out)


# ════════════════════════════════════════════════════════════════════════════
# TASK 3: DOWNSTREAM GENERATIVE EVALUATION
# ════════════════════════════════════════════════════════════════════════════

class DownstreamEvaluator:
    """WikiText-2 perplexity and LAMBADA (exact-match + target NLL) with the policy active."""

    def __init__(self, cfg: ExperimentConfig, runner: ModelRunner):
        self.cfg, self.runner, self.tok = cfg, runner, runner.tokenizer

    def _data(self):
        wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        ids = self.tok("\n\n".join(wiki["text"]), add_special_tokens=False).input_ids
        n_keep = self.cfg.PPL_WINDOW + self.cfg.PPL_STRIDE * (self.cfg.PPL_MAX_WINDOWS - 1)
        lam = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        return ids[:n_keep], list(lam["text"][: self.cfg.LAMBADA_N])

    @torch.no_grad()
    def _perplexity(self, ids: List[int], spec: PolicySpec) -> float:
        """Sliding window; each token is scored once, with up to W-S tokens of context."""
        bos = [self.tok.bos_token_id] if self.runner.adds_bos else []
        nll, ntok, prev_end = 0.0, 0, 0
        for begin in range(0, len(ids), self.cfg.PPL_STRIDE):
            end = min(begin + self.cfg.PPL_WINDOW, len(ids))
            x = torch.tensor([bos + ids[begin:end]], device=DEVICE)
            start = max(1, x.shape[1] - (end - prev_end))
            logits = self.runner.forward(x, spec, logits_to_keep=0).logits[0].float()
            nll += float(F.cross_entropy(logits[start - 1:-1], x[0, start:], reduction="sum"))
            ntok += x.shape[1] - start
            prev_end = end
            if end == len(ids):
                break
        return math.exp(nll / ntok)

    @torch.no_grad()
    def _lambada(self, texts: List[str], spec: PolicySpec) -> Tuple[float, float]:
        correct, nll, n = 0, 0.0, 0
        for text in texts:
            ctx, last = text.rsplit(" ", 1)
            c = self.tok(ctx, add_special_tokens=True).input_ids
            t = self.tok(" " + last, add_special_tokens=False).input_ids
            x = torch.tensor([c + t], device=DEVICE)
            lp = self.runner.forward(x, spec, logits_to_keep=len(t) + 1).logits[0, :-1].float().log_softmax(-1)
            tgt = torch.tensor(t, device=DEVICE)
            correct += int(bool((lp.argmax(-1) == tgt).all()))
            nll += float(-lp.gather(1, tgt[:, None]).sum())
            n += 1
        return correct / n, nll / n

    def run(self, specs: List[PolicySpec], path: str):
        if self.cfg.RESUME and os.path.exists(path):
            logger.info(f"  RESUME downstream {self.runner.mc.name}")
            return
        if not HF_DATASETS_AVAILABLE:
            logger.warning("  [Task 3] `datasets` not installed: downstream skipped")
            return
        try:
            ids, texts = self._data()
        except Exception as e:
            logger.warning(f"  [Task 3] dataset loading failed ({e}): downstream skipped")
            return
        rows = []
        for spec in tqdm(specs, desc=f"downstream {self.runner.mc.name}"):
            acc, lnll = self._lambada(texts, spec)
            rows.append({"model": self.runner.mc.name, "policy": spec.name, "family": spec.family,
                         "draw": spec.draw, "semantics": spec.semantics,
                         "kv_memory_fraction": spec.kv_memory_fraction,
                         "wikitext2_ppl": self._perplexity(ids, spec),
                         "lambada_acc": acc, "lambada_target_nll": lnll})
        pd.DataFrame(rows).to_csv(path, index=False)


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS: primitives
# ════════════════════════════════════════════════════════════════════════════

def weighted_auc_parts(scores: np.ndarray, pos: np.ndarray, W: np.ndarray,
                       chunk: int = 256) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mann-Whitney AUC under integer resampling weights, exact with ties (=1/2).
    scores (n,), pos (n,) bool, W (B, n).  Returns (num, den), AUC = num / den.
    One sort serves all B replicates: O(B n).
    """
    order = np.argsort(scores, kind="mergesort")
    s, p = scores[order], pos[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    num, den = np.empty(W.shape[0]), np.empty(W.shape[0])
    for i in range(0, W.shape[0], chunk):
        Wo = W[i:i + chunk][:, order].astype(np.float64)
        wn = np.add.reduceat(Wo * ~p, starts, axis=1)
        wp = np.add.reduceat(Wo * p, starts, axis=1)
        below = np.cumsum(wn, axis=1) - wn
        num[i:i + chunk] = (wp * (below + 0.5 * wn)).sum(1)
        den[i:i + chunk] = wp.sum(1) * wn.sum(1)
    return num, den


def bootstrap_weights(strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot + 1, n): row 0 = observed sample, rows 1.. = stratified resamples."""
    rng = np.random.default_rng(seed)
    W = np.zeros((n_boot + 1, len(strata)), np.float32)
    W[0] = 1.0
    for g in np.unique(strata):
        idx = np.flatnonzero(strata == g)
        W[1:, idx] = rng.multinomial(len(idx), np.full(len(idx), 1.0 / len(idx)), size=n_boot)
    return W


def summarize(v: np.ndarray) -> Dict[str, float]:
    """
    Point estimate (row 0), bootstrap SE, percentile 95% CI and a two-sided Wald
    p-value with the bootstrap SE (Efron & Tibshirani, 1993).  The Wald form is
    used because a percentile-count p-value cannot go below 2/(B+1), which would
    make Holm correction over many contrasts unable to reject by construction.
    """
    est, boot = float(v[0]), v[1:][np.isfinite(v[1:])]
    if not np.isfinite(est) or boot.size < 2:
        return {"estimate": np.nan, "se": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_value": np.nan}
    lo, hi = np.percentile(boot, [2.5, 97.5])
    se = float(boot.std(ddof=1))
    p = float(2 * stats.norm.sf(abs(est) / se)) if se > 0 else float(est == 0) or 0.0
    return {"estimate": est, "se": se, "ci_low": float(lo), "ci_high": float(hi), "p_value": p}


def sign_flip_test(d: np.ndarray, n_perm: int, seed: int, chunk: int = 1000) -> float:
    """Exact-form paired permutation p-value, (1 + #{|null| >= |obs|}) / (1 + n_perm)."""
    rng = np.random.default_rng(seed)
    obs, hits = abs(d.mean()), 0
    for i in range(0, n_perm, chunk):
        m = min(chunk, n_perm - i)
        signs = rng.integers(0, 2, size=(m, d.size)) * 2 - 1
        hits += int(np.sum(np.abs((signs * d).mean(1)) >= obs - 1e-12))
    return (1 + hits) / (1 + n_perm)


def dprime(hits: np.ndarray, false_alarms: np.ndarray) -> float:
    """d' with the log-linear correction (Hautus, 1995)."""
    h = (hits.sum() + 0.5) / (hits.size + 1)
    f = (false_alarms.sum() + 0.5) / (false_alarms.size + 1)
    return float(stats.norm.ppf(h) - stats.norm.ppf(f))


def adjust(df: pd.DataFrame, pcol: str, by: List[str], method: str, out: str) -> pd.DataFrame:
    df[out] = np.nan
    for _, g in df.groupby(by, dropna=False):
        g = g[g[pcol].notna()]
        if len(g):
            df.loc[g.index, out] = multipletests(g[pcol].to_numpy(), method=method)[1]
    return df


def ols_hc3(y: np.ndarray, X: pd.DataFrame):
    return sm.OLS(y, sm.add_constant(X)).fit(cov_type="HC3")


def zscore(x: np.ndarray) -> np.ndarray:
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 0 else 1.0)


CONTRASTS = [
    ("H3a_cka_high_vs_cka_low", "cka_high", "cka_low"),
    ("H3b_cka_high_vs_depth_matched_random", "cka_high", "rdm_cka_high"),
    ("H3b_cka_low_vs_depth_matched_random", "cka_low", "rdm_cka_low"),
    ("H3b_fidelity_high_vs_depth_matched_random", "fidelity_high", "rdm_fidelity_high"),
    ("H3c_fidelity_high_vs_cka_high", "fidelity_high", "cka_high"),
    ("H3d_cka_high_vs_uniform_random", "cka_high", "random_same_ratio"),
    ("H3d_cka_low_vs_uniform_random", "cka_low", "random_same_ratio"),
    ("H3d_fidelity_high_vs_uniform_random", "fidelity_high", "random_same_ratio"),
    ("H4_own_head_vs_random_head_source", "full", "random_head_source"),
    ("C1_live_vs_clean_semantics", "full", "cf_prev_head"),
    ("C2_previous_vs_next_head_source", "cf_prev_head", "cf_next_head"),
]
READOUTS = ("frozen", "retrained")


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS: per model
# ════════════════════════════════════════════════════════════════════════════

class ModelAnalysis:
    """All per-model inference.  Items are the resampling unit; layers never are."""

    def __init__(self, cfg: ExperimentConfig, model_name: str):
        self.cfg, self.model = cfg, model_name
        self.dir = cfg.model_dir(model_name)
        self.out = os.path.join(self.dir, "analysis")
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.dir, "model_info.json")) as f:
            info = json.load(f)
        self.L = int(info["num_layers"])
        self.test = np.asarray(info["test_items"])
        items = pd.read_csv(os.path.join(self.dir, "items.csv"))
        self.t = items.iloc[self.test].reset_index(drop=True)
        self.y = self.t["is_word"].to_numpy() == 1
        self.fg = self.t["freq_group"].to_numpy()
        self.ntok = self.t["n_tokens"].to_numpy()
        self.tok_stratum = np.minimum(self.ntok, max(cfg.TOKEN_STRATA))
        # Resampling strata = frequency group x token stratum, so every replicate
        # keeps the composition of every subgroup an AUC is computed on.
        self.W = bootstrap_weights(np.char.add(self.fg.astype(str), self.tok_stratum.astype(str)),
                                   cfg.N_BOOTSTRAP, SEED)
        self.runs = self._load_runs()
        self.base = (None, NO_REUSE)
        self._cache: Dict[Tuple, np.ndarray] = {}
        present = [c for c in cfg.MATCH_COVARIATES
                   if self.t.loc[self.y, c].notna().mean() >= 0.9]
        self.covariates = present

    # ── loading ───────────────────────────────────────────────────────
    def _load_runs(self) -> Dict[Tuple, Dict]:
        runs = {}
        for rf in [None] + list(self.cfg.REUSE_FACTORS):
            d = self.dir if rf is None else os.path.join(self.dir, f"rf{rf}")
            if not os.path.isdir(d):
                continue
            for fn in sorted(os.listdir(d)):
                stem = fn[:-5]
                if not fn.endswith(".json") or not os.path.exists(os.path.join(d, stem + ".npz")):
                    continue
                with open(os.path.join(d, fn)) as f:
                    meta = json.load(f)
                if rf is None and meta["name"] != NO_REUSE:
                    continue
                z = np.load(os.path.join(d, stem + ".npz"))
                if not np.array_equal(z["test_items"], self.test):
                    raise RuntimeError(f"{self.model}/{stem}: test items differ from model_info")
                runs[(rf, meta["name"])] = {"meta": meta, "retrained": z["logit_retrained"],
                                            "frozen": z["logit_frozen"], "auc_seed": z["auc_seed"]}
        if (None, NO_REUSE) not in runs:
            raise RuntimeError(f"{self.model}: no_reuse control missing")
        return runs

    def families(self, rf: int) -> Dict[str, List[Tuple]]:
        fam = defaultdict(list)
        for key, r in self.runs.items():
            if key[0] == rf:
                fam[r["meta"]["family"]].append(key)
        return dict(fam)

    def ec(self) -> int:
        return exempt_cutoff(self.L, self.cfg.REUSE_EXEMPT_FRACTION)

    # ── AUC replicates ────────────────────────────────────────────────
    def _subsets(self, group: str) -> List[np.ndarray]:
        nw = ~self.y
        if group == "all":
            return [np.ones_like(self.y)]
        if group == "hf":
            return [nw | (self.fg == "high")]
        if group == "lf":
            return [nw | (self.fg == "low")]
        if group == "tok":           # token-count-matched: pairs only within a stratum
            return [self.tok_stratum == k for k in self.cfg.TOKEN_STRATA]
        if group.startswith("tok"):
            return [self.tok_stratum == int(group[3:])]
        raise ValueError(group)

    def auc_replicates(self, scores: np.ndarray, group: str, W: np.ndarray) -> np.ndarray:
        if not np.all(np.isfinite(scores)):
            return np.full(W.shape[0], np.nan)
        num, den = np.zeros(W.shape[0]), np.zeros(W.shape[0])
        for m in self._subsets(group):
            if self.y[m].any() and (~self.y[m]).any():
                a, b = weighted_auc_parts(scores[m], self.y[m], W[:, m])
                num, den = num + a, den + b
        with np.errstate(invalid="ignore", divide="ignore"):
            return num / den

    def curve(self, key: Tuple, readout: str, group: str) -> np.ndarray:
        """(L, B+1) AUC replicates of one run across layers."""
        ck = (key, readout, group)
        if ck not in self._cache:
            logits = self.runs[key][readout]
            self._cache[ck] = np.stack([self.auc_replicates(logits[li], group, self.W)
                                        for li in range(self.L)])
        return self._cache[ck]

    def family_curve(self, keys: List[Tuple], readout: str, group: str) -> np.ndarray:
        """Random families: mean over draws (paired, same resamples)."""
        return np.mean([self.curve(k, readout, group) for k in keys], axis=0)

    def family_logits(self, keys: List[Tuple], readout: str) -> np.ndarray:
        return np.mean([self.runs[k][readout] for k in keys], axis=0)

    # ── 1. layer table ────────────────────────────────────────────────
    def layer_table(self) -> pd.DataFrame:
        rows = []
        for key, r in self.runs.items():
            meta = r["meta"]
            targets = {int(t) for t in meta["source_map"]}
            for ro in READOUTS:
                c = {g: self.curve(key, ro, g) for g in ("all", "hf", "lf", "tok")}
                base = self.curve(self.base, ro, "all")
                logits = r[ro]
                for li in range(self.L):
                    pred = logits[li] > 0
                    row = {"model": self.model, "reuse_factor": key[0], "policy": key[1],
                           "family": meta["family"], "readout": ro, "layer": li,
                           "depth": li / max(self.L - 1, 1), "is_target": li in targets,
                           "auc_all": c["all"][li, 0], "auc_hf": c["hf"][li, 0],
                           "auc_lf": c["lf"][li, 0], "auc_token_matched": c["tok"][li, 0],
                           "accuracy": float((pred == self.y).mean()) if np.isfinite(logits[li]).all() else np.nan,
                           "dprime_hf": dprime(pred[self.y & (self.fg == "high")], pred[~self.y]),
                           "dprime_lf": dprime(pred[self.y & (self.fg == "low")], pred[~self.y]),
                           "auc_seed_sd": (float(np.nanstd(r["auc_seed"][:, li]))
                                           if ro == "retrained" else np.nan)}
                    ci = np.percentile(c["all"][li, 1:], [2.5, 97.5]) if np.isfinite(c["all"][li, 0]) else (np.nan, np.nan)
                    row["auc_all_ci_low"], row["auc_all_ci_high"] = ci
                    row.update({f"freq_gap_{k}": v for k, v in summarize(c["hf"][li] - c["lf"][li]).items()})
                    if key != self.base:
                        row.update({f"delta_all_{k}": v for k, v in summarize(c["all"][li] - base[li]).items()})
                    rows.append(row)
        df = pd.DataFrame(rows)
        df = adjust(df, "freq_gap_p_value", ["reuse_factor", "policy", "readout"], "fdr_bh", "freq_gap_p_fdr")
        if "delta_all_p_value" in df:
            df = adjust(df, "delta_all_p_value", ["reuse_factor", "policy", "readout"], "fdr_bh", "delta_all_p_fdr")
        rt = df[df.readout == "retrained"].set_index(["reuse_factor", "policy", "layer"])["auc_all"]
        fz = df[df.readout == "frozen"].set_index(["reuse_factor", "policy", "layer"])["auc_all"]
        gap = (rt - fz).rename("recovery_gap").reset_index()
        return df.merge(gap, on=["reuse_factor", "policy", "layer"], how="left")

    # ── 2. band contrasts vs no_reuse (H1, H2-DiD) ────────────────────
    def band_contrasts(self) -> pd.DataFrame:
        rows = []
        for rf in self.cfg.REUSE_FACTORS:
            bands = eligible_bands(self.L, self.ec(), self.cfg.N_BANDS)
            for fam, keys in self.families(rf).items():
                mem = float(np.mean([self.runs[k]["meta"]["kv_memory_fraction"] for k in keys]))
                for ro in READOUTS:
                    d = {g: self.family_curve(keys, ro, g) - self.curve(self.base, ro, g)
                         for g in ("all", "hf", "lf", "tok")}
                    for band, layers in bands.items():
                        bd = {g: np.nanmean(v[layers], axis=0) for g, v in d.items()}
                        bd["did_lf_minus_hf"] = bd["lf"] - bd["hf"]
                        for metric, v in bd.items():
                            rows.append({"model": self.model, "reuse_factor": rf, "family": fam,
                                         "n_draws": len(keys), "kv_memory_fraction": mem,
                                         "readout": ro, "band": band, "metric": metric,
                                         **summarize(v)})
        df = pd.DataFrame(rows)
        return adjust(df, "p_value", ["reuse_factor", "readout", "metric"], "holm", "p_holm")

    def negative_controls(self) -> pd.DataFrame:
        """Layers below the first target must be unchanged: max |delta AUC| there."""
        rows = []
        for key, r in self.runs.items():
            if key == self.base or not r["meta"]["source_map"]:
                continue
            first = min(int(t) for t in r["meta"]["source_map"])
            for ro in READOUTS:
                if first == 0:
                    continue
                a = self.curve(key, ro, "all")[:first, 0]
                b = self.curve(self.base, ro, "all")[:first, 0]
                rows.append({"model": self.model, "reuse_factor": key[0], "policy": key[1],
                             "readout": ro, "layers_checked": first,
                             "max_abs_delta_auc_pre_target": float(np.nanmax(np.abs(a - b)))})
        return pd.DataFrame(rows)

    # ── 3. policy-vs-policy contrasts (H3, H4, C1, C2) ────────────────
    def policy_contrasts(self) -> pd.DataFrame:
        rows = []
        for rf in self.cfg.REUSE_FACTORS:
            fam = self.families(rf)
            bands = eligible_bands(self.L, self.ec(), self.cfg.N_BANDS)
            for label, a, b in CONTRASTS:
                if a not in fam or b not in fam:
                    continue
                for ro in READOUTS:
                    d = {g: self.family_curve(fam[a], ro, g) - self.family_curve(fam[b], ro, g)
                         for g in ("all", "hf", "lf")}
                    for band, layers in bands.items():
                        bd = {"all": np.nanmean(d["all"][layers], 0),
                              "did_lf_minus_hf": np.nanmean(d["lf"][layers] - d["hf"][layers], 0)}
                        spread = {}
                        for side, f in (("a", a), ("b", b)):
                            vals = [np.nanmean(self.curve(k, ro, "all")[layers, 0]) for k in fam[f]]
                            spread[f"draw_sd_{side}"] = float(np.std(vals)) if len(vals) > 1 else np.nan
                        for metric, v in bd.items():
                            rows.append({"model": self.model, "reuse_factor": rf, "contrast": label,
                                         "policy_a": a, "policy_b": b, "readout": ro, "band": band,
                                         "metric": metric, **summarize(v), **spread})
        df = pd.DataFrame(rows)
        return adjust(df, "p_value", ["reuse_factor", "readout", "metric"], "holm", "p_holm") if len(df) else df

    # ── 4. item-level policy x frequency regression (H2, S1, S2) ──────
    def _band_scores(self, keys, readout, layers) -> np.ndarray:
        return np.nanmean(self.family_logits(keys, readout)[layers], axis=0)

    def item_regressions(self) -> pd.DataFrame:
        words = self.y & self.t["log_freq"].notna().to_numpy()
        for c in self.covariates:
            words &= self.t[c].notna().to_numpy()
        T = self.t[words]
        base_X = pd.DataFrame({"log_freq_z": zscore(T["log_freq"].to_numpy())}
                              | {f"{c}_z": zscore(T[c].to_numpy()) for c in self.covariates})
        ntok_c = self.ntok[words] - self.ntok[words].mean()
        rows = []
        for rf in self.cfg.REUSE_FACTORS:
            bands = eligible_bands(self.L, self.ec(), self.cfg.N_BANDS)
            for fam, keys in self.families(rf).items():
                for ro in READOUTS:
                    for band, layers in bands.items():
                        dmg = (self._band_scores([self.base], ro, layers)
                               - self._band_scores(keys, ro, layers))[words]
                        if not np.isfinite(dmg).all():
                            continue
                        mA = ols_hc3(dmg, base_X)
                        XB = base_X.assign(n_tokens_c=ntok_c)
                        mB = ols_hc3(dmg, XB)
                        vif = max(variance_inflation_factor(sm.add_constant(XB).to_numpy(), i)
                                  for i in range(1, XB.shape[1] + 1))
                        bA, bB = mA.params["log_freq_z"], mB.params["log_freq_z"]
                        rows.append({"model": self.model, "reuse_factor": rf, "family": fam,
                                     "readout": ro, "band": band, "n_words": int(words.sum()),
                                     "mean_damage": float(dmg.mean()),
                                     "beta_freq_no_ntok": bA, "p_freq_no_ntok": mA.pvalues["log_freq_z"],
                                     "beta_freq": bB, "se_freq": mB.bse["log_freq_z"],
                                     "ci_freq_low": mB.conf_int().loc["log_freq_z", 0],
                                     "ci_freq_high": mB.conf_int().loc["log_freq_z", 1],
                                     "p_freq": mB.pvalues["log_freq_z"],
                                     "beta_ntok": mB.params["n_tokens_c"], "p_ntok": mB.pvalues["n_tokens_c"],
                                     "attenuation": (1 - bB / bA) if bA != 0 else np.nan,
                                     "max_vif": float(vif), "covariates": ";".join(self.covariates)})
        return pd.DataFrame(rows)

    # ── 5. token-count strata (S1) ────────────────────────────────────
    def token_strata(self) -> pd.DataFrame:
        rows = []
        for rf in self.cfg.REUSE_FACTORS:
            layers = eligible_bands(self.L, self.ec(), self.cfg.N_BANDS)["all"]
            for fam, keys in self.families(rf).items():
                for k in self.cfg.TOKEN_STRATA:
                    g = f"tok{k}"
                    m = self.tok_stratum == k
                    row = {"model": self.model, "reuse_factor": rf, "family": fam,
                           "token_stratum": f">={k}" if k == max(self.cfg.TOKEN_STRATA) else str(k),
                           "n_words": int((m & self.y).sum()), "n_nonwords": int((m & ~self.y).sum())}
                    if row["n_words"] and row["n_nonwords"]:
                        base = self.curve(self.base, "frozen", g)[layers]
                        row["base_auc"] = float(np.nanmean(base[:, 0]))
                        row.update(summarize(np.nanmean(self.family_curve(keys, "frozen", g)[layers] - base, 0)))
                    rows.append(row)
        return pd.DataFrame(rows)

    # ── 6. matched HF/LF pairs (H7, B1, S5) ───────────────────────────
    def matched_pairs(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        covs = self.covariates
        ok = self.y & np.isin(self.fg, ["high", "low"])
        for c in covs:
            ok &= self.t[c].notna().to_numpy()
        Z = np.column_stack([self.t[c].to_numpy(float) for c in covs]) if covs else np.zeros((len(self.t), 0))
        if covs:      # standardised on the HF+LF pool that is being matched
            sd = Z[ok].std(0)
            Z = (Z - Z[ok].mean(0)) / np.where(sd > 0, sd, 1.0)
        pairs = []
        for k in np.unique(self.ntok[ok]):
            hi = np.flatnonzero(ok & (self.fg == "high") & (self.ntok == k))
            lo = np.flatnonzero(ok & (self.fg == "low") & (self.ntok == k))
            if not len(hi) or not len(lo):
                continue
            cost = np.sqrt(((Z[hi][:, None, :] - Z[lo][None, :, :]) ** 2).sum(-1))
            r, c = linear_sum_assignment(cost)
            for i, j in zip(r, c):
                if np.all(np.abs(Z[hi[i]] - Z[lo[j]]) <= self.cfg.MATCH_CALIPER_SD):
                    pairs.append((hi[i], lo[j]))
        if not pairs:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        P = np.array(pairs)
        pairs_df = pd.DataFrame({"pair": np.arange(len(P)),
                                 "hf_stimulus": self.t.stimulus.to_numpy()[P[:, 0]],
                                 "lf_stimulus": self.t.stimulus.to_numpy()[P[:, 1]],
                                 "n_tokens": self.ntok[P[:, 0]]})

        bal = []
        allh, alll = ok & (self.fg == "high"), ok & (self.fg == "low")
        for c in covs + ["log_freq", "n_tokens"]:
            x = self.t[c].to_numpy(float) if c != "n_tokens" else self.ntok.astype(float)
            sd = np.sqrt((np.nanvar(x[allh]) + np.nanvar(x[alll])) / 2) or 1.0
            bal.append({"model": self.model, "covariate": c,
                        "smd_before": (np.nanmean(x[allh]) - np.nanmean(x[alll])) / sd,
                        "smd_after": (np.nanmean(x[P[:, 0]]) - np.nanmean(x[P[:, 1]])) / sd,
                        "n_pairs": len(P)})

        res = []
        for rf in self.cfg.REUSE_FACTORS:
            bands = eligible_bands(self.L, self.ec(), self.cfg.N_BANDS)
            fams = {NO_REUSE: [self.base]} | self.families(rf)
            for band, layers in bands.items():
                sb = self._band_scores([self.base], "frozen", layers)
                d_base = sb[P[:, 0]] - sb[P[:, 1]]
                for fam, keys in fams.items():
                    if fam == NO_REUSE:
                        d, label = d_base, "freq_effect_hf_minus_lf"
                    else:
                        s = self._band_scores(keys, "frozen", layers)
                        d, label = (s[P[:, 0]] - s[P[:, 1]]) - d_base, "did_vs_no_reuse"
                    if not np.isfinite(d).all():
                        continue
                    res.append({"model": self.model, "reuse_factor": rf, "family": fam, "band": band,
                                "statistic": label, "n_pairs": len(P), "mean": float(d.mean()),
                                "sd": float(d.std(ddof=1)),
                                "p_sign_flip": sign_flip_test(d, self.cfg.N_PERMUTATIONS, SEED)})
        return pairs_df, pd.DataFrame(bal), pd.DataFrame(res)

    # ── 7. single-layer fragility scan (H5, G4) ───────────────────────
    def fragility(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        path = os.path.join(self.dir, "fragility.npz")
        if not os.path.exists(path):
            return pd.DataFrame(), pd.DataFrame()
        z = np.load(path)
        readouts, base, logits = z["readouts"], z["base_logits"], z["logits"]
        W = self.W[: self.cfg.N_BOOTSTRAP_FRAGILITY + 1]
        groups = ["all", "hf", "lf"] + [f"tok{k}" for k in self.cfg.TOKEN_STRATA]
        rows, mech = [], []
        for ri, r in enumerate(readouts):
            b = {g: self.auc_replicates(base[ri], g, W) for g in groups}
            for t in range(1, self.L):
                p = {g: self.auc_replicates(logits[t, ri], g, W) for g in groups}
                d = {g: p[g] - b[g] for g in groups}
                d["did_lf_minus_hf"] = d["lf"] - d["hf"]
                row = {"model": self.model, "readout_layer": int(r), "substituted_layer": t,
                       "depth": t / max(self.L - 1, 1), "after_readout": t > r,
                       "max_abs_logit_change": float(np.nanmax(np.abs(logits[t, ri] - base[ri])))}
                for g, v in d.items():
                    s = summarize(v)
                    row.update({f"delta_{g}": s["estimate"], f"delta_{g}_ci_low": s["ci_low"],
                                f"delta_{g}_ci_high": s["ci_high"], f"delta_{g}_p": s["p_value"]})
                rows.append(row)
        df = pd.DataFrame(rows)
        # Mechanism test (G4): item damage from substitution in the first half of
        # depth, read out at the final layer, regressed on token count.
        ri = int(np.flatnonzero(readouts == self.L - 1)[0])
        early = [t for t in range(1, self.L) if t / (self.L - 1) <= 0.5]
        dmg = base[ri][None, :] - logits[early, ri]
        words = self.y & self.t["log_freq"].notna().to_numpy()
        for c in self.covariates:
            words &= self.t[c].notna().to_numpy()
        X = pd.DataFrame({"n_tokens_c": self.ntok[words] - self.ntok[words].mean(),
                          "log_freq_z": zscore(self.t.loc[words, "log_freq"].to_numpy())}
                         | {f"{c}_z": zscore(self.t.loc[words, c].to_numpy()) for c in self.covariates})
        m = ols_hc3(dmg.mean(0)[words], X)
        mech.append({"model": self.model, "readout_layer": int(self.L - 1),
                     "substituted_layers": f"1-{max(early)}", "n_words": int(words.sum()),
                     "beta_ntok": m.params["n_tokens_c"], "p_ntok": m.pvalues["n_tokens_c"],
                     "beta_freq": m.params["log_freq_z"], "p_freq": m.pvalues["log_freq_z"]})
        return df, pd.DataFrame(mech)

    # ── 8. reliability, calibration, downstream (collation) ───────────
    def reliability(self) -> pd.DataFrame:
        frames = []
        for rf in [None] + list(self.cfg.REUSE_FACTORS):
            d = self.dir if rf is None else os.path.join(self.dir, f"rf{rf}")
            frames += [pd.read_csv(os.path.join(d, f)) for f in sorted(os.listdir(d))
                       if f.endswith("_reliability.csv")] if os.path.isdir(d) else []
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True)
        acc = df[df.probe != "linear_mdl"].groupby(["model", "reuse_factor", "policy", "layer", "probe", "n_test",
                                                    "task"], dropna=False)[["test_accuracy", "test_auc"]].mean().unstack("task")
        acc.columns = [f"{m}_{t}" for m, t in acc.columns]
        acc = acc.reset_index()
        acc["selectivity_accuracy"] = acc["test_accuracy_real"] - acc["test_accuracy_control"]
        acc["selectivity_auc"] = acc["test_auc_real"] - acc["test_auc_control"]
        mdl = df[df.probe == "linear_mdl"].pivot_table(index=["model", "reuse_factor", "policy", "layer"],
                                                       columns="task", values="mdl_compression", dropna=False)
        mdl.columns = [f"mdl_compression_{t}" for t in mdl.columns]
        return acc.merge(mdl.reset_index(), on=["model", "reuse_factor", "policy", "layer"], how="left")

    def downstream(self, bands_frame: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Deltas vs no_reuse and S8/H6 Spearman between probe and generative damage."""
        rf = self.cfg.REUSE_FACTORS[0]
        path = os.path.join(self.dir, f"downstream_rf{rf}.csv")
        if not os.path.exists(path):
            return pd.DataFrame(), pd.DataFrame()
        df = pd.read_csv(path)
        base = df[df.policy == NO_REUSE].iloc[0]
        df["delta_log_ppl"] = np.log(df.wikitext2_ppl) - np.log(base.wikitext2_ppl)
        df["delta_lambada_acc"] = df.lambada_acc - base.lambada_acc
        df["delta_lambada_nll"] = df.lambada_target_nll - base.lambada_target_nll
        fam = df[df.policy != NO_REUSE].groupby("family")[
            ["delta_log_ppl", "delta_lambada_acc", "delta_lambada_nll"]].mean()
        probe = bands_frame.query("reuse_factor == @rf and readout == 'frozen' and band == 'all' "
                                  "and metric == 'all'").set_index("family")["estimate"]
        j = fam.join(probe.rename("delta_auc_probe"), how="inner")
        rows = []
        for col in ["delta_log_ppl", "delta_lambada_acc", "delta_lambada_nll"]:
            if len(j) >= 4 and j[col].nunique() > 1 and j["delta_auc_probe"].nunique() > 1:
                rho, p = stats.spearmanr(-j["delta_auc_probe"], j[col])
                rows.append({"model": self.model, "generative_metric": col, "n_policies": len(j),
                             "spearman_rho_probe_damage": rho, "p": p})
        return df, pd.DataFrame(rows)

    # ── driver ────────────────────────────────────────────────────────
    def run(self) -> Dict[str, pd.DataFrame]:
        logger.info(f"[analysis] {self.model}")
        res = {"layers": self.layer_table(), "bands": self.band_contrasts(),
               "negative_controls": self.negative_controls(), "contrasts": self.policy_contrasts(),
               "item_regression": self.item_regressions(), "token_strata": self.token_strata()}
        res["matched_pairs"], res["matching_balance"], res["matched_tests"] = self.matched_pairs()
        res["fragility"], res["fragility_mechanism"] = self.fragility()
        res["reliability"] = self.reliability()
        res["downstream"], res["downstream_correlation"] = self.downstream(res["bands"])
        frames = []
        for rf in self.cfg.REUSE_FACTORS:
            p = os.path.join(self.dir, f"rf{rf}", "calibration.csv")
            if os.path.exists(p):
                frames.append(pd.read_csv(p).assign(model=self.model, reuse_factor=rf))
        res["calibration"] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for name, df in res.items():
            if len(df):
                df.to_csv(os.path.join(self.out, f"{name}.csv"), index=False)
        return res


# ════════════════════════════════════════════════════════════════════════════
# STATISTICS: across models (models are the units) + decision rules
# ════════════════════════════════════════════════════════════════════════════

class CrossModelAnalysis:
    def __init__(self, cfg: ExperimentConfig, per_model: Dict[str, Dict[str, pd.DataFrame]]):
        self.cfg, self.per_model = cfg, per_model
        self.rf = cfg.REUSE_FACTORS[0]
        self.n = len(per_model)
        self.need = int(math.ceil(cfg.DECISION_MIN_MODEL_FRACTION * self.n))
        self.out = cfg.PAPER_DIR

    def cat(self, name: str) -> pd.DataFrame:
        frames = [r[name] for r in self.per_model.values() if name in r and len(r[name])]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # ── models-as-units tests ─────────────────────────────────────────
    def models_as_units(self) -> pd.DataFrame:
        bands = self.cat("bands")
        rows = []
        for (rf, fam, ro, band, metric), g in bands.groupby(
                ["reuse_factor", "family", "readout", "band", "metric"]):
            v = g["estimate"].dropna().to_numpy()
            p = stats.wilcoxon(v).pvalue if v.size >= 5 and np.any(v != 0) else np.nan
            rows.append({"reuse_factor": rf, "family": fam, "readout": ro, "band": band, "metric": metric,
                         "n_models": v.size, "median_estimate": float(np.median(v)) if v.size else np.nan,
                         "n_negative": int((v < 0).sum()), "n_positive": int((v > 0).sum()),
                         "p_wilcoxon_models": p})
        df = pd.DataFrame(rows)
        return adjust(df, "p_wilcoxon_models", ["reuse_factor", "readout", "metric"], "holm", "p_holm")

    def fragility_profiles(self) -> Tuple[pd.DataFrame, float]:
        """Per-model peak fragility depth (final-layer readout) and profile alignment."""
        fr = self.cat("fragility")
        if fr.empty:
            return pd.DataFrame(), np.nan
        grid = np.linspace(0, 1, 21)
        peaks, profiles = [], {}
        for m, g in fr.groupby("model"):
            g = g[g.readout_layer == g.readout_layer.max()].sort_values("depth")
            dmg = -g["delta_all"].to_numpy()
            peaks.append({"model": m, "peak_depth": float(g["depth"].to_numpy()[int(np.nanargmax(dmg))]),
                          "peak_delta_auc": float(-np.nanmax(dmg))})
            profiles[m] = np.interp(grid, g["depth"].to_numpy(), dmg)
        names = list(profiles)
        rhos = [stats.spearmanr(profiles[a], profiles[b])[0]
                for i, a in enumerate(names) for b in names[i + 1:]]
        return pd.DataFrame(peaks), float(np.nanmean(rhos)) if rhos else np.nan

    # ── pre-registered decision rules ─────────────────────────────────
    def _verdict(self, count: int, total: int) -> str:
        if total < self.need:
            return "insufficient data"
        return "supported" if count >= self.need else "not supported"

    def decision_rules(self, peaks: pd.DataFrame, alignment: float) -> pd.DataFrame:
        a, rows = self.cfg.ALPHA, []
        bands, con = self.cat("bands"), self.cat("contrasts")
        reg, mt, rel = self.cat("item_regression"), self.cat("matched_tests"), self.cat("reliability")
        dsc = self.cat("downstream_correlation")

        def add(h, rule, count, total, detail):
            rows.append({"hypothesis": h, "rule": rule, "models_meeting_rule": count,
                         "models_evaluated": total, "models_required": self.need,
                         "verdict": self._verdict(count, total), "detail": detail})

        for ro in READOUTS:
            b = bands.query("reuse_factor == @self.rf and family == 'full' and readout == @ro and metric == 'all' "
                            "and band != 'all'") if len(bands) else bands
            per = b.groupby("model").apply(lambda g: bool(((g.estimate < 0) & (g.p_holm < a)).any()),
                                           include_groups=False) if len(b) else pd.Series(dtype=bool)
            add(f"H1 ({ro})", "full lowers AUC vs no_reuse in >=1 band (Holm, item bootstrap)",
                int(per.sum()), len(per), f"models: {', '.join(per.index[per])}")

        if len(reg):
            r = reg.query("reuse_factor == @self.rf and family == 'full' and readout == 'frozen' and band == 'all'")
            hit = (r.beta_freq < 0) & (r.p_freq < a)
            hit_no = (r.beta_freq_no_ntok < 0) & (r.p_freq_no_ntok < a)
            add("H2", "damage decreases with log frequency, n_tokens controlled (HC3, frozen, all band)",
                int(hit.sum()), len(r),
                f"without n_tokens: {int(hit_no.sum())}/{len(r)} models; median attenuation "
                f"{np.nanmedian(r.attenuation):.2f}; n_tokens slope >0 & p<alpha in "
                f"{int(((r.beta_ntok > 0) & (r.p_ntok < a)).sum())} models")

        if len(con):
            c = con.query("reuse_factor == @self.rf and readout == 'frozen' and band == 'all' and metric == 'all'")
            for label in sorted(c.contrast.unique()):
                g = c[c.contrast == label]
                sig = g.p_holm < a
                add(label, "band-'all' AUC difference A-B != 0 (Holm, item bootstrap)",
                    int(sig.sum()), len(g),
                    f"A better in {int((sig & (g.estimate > 0)).sum())}, "
                    f"B better in {int((sig & (g.estimate < 0)).sum())} models")

        if len(peaks):
            early = peaks.peak_depth <= self.cfg.H5_EARLY_DEPTH
            add("H5", f"single-layer fragility peaks at normalised depth <= {self.cfg.H5_EARLY_DEPTH}",
                int(early.sum()), len(peaks), f"mean pairwise Spearman of profiles = {alignment:.3f}")

        if len(dsc):
            d = dsc[dsc.generative_metric == "delta_log_ppl"]
            v = d.spearman_rho_probe_damage.dropna().to_numpy()
            p = stats.wilcoxon(v).pvalue if v.size >= 6 and np.any(v != 0) else np.nan
            rows.append({"hypothesis": "H6", "rule": "probe damage ranks policies like log-PPL damage "
                         "(per-model Spearman, Wilcoxon over models)",
                         "models_meeting_rule": int((v > 0).sum()), "models_evaluated": v.size,
                         "models_required": 6,
                         "verdict": ("insufficient data" if v.size < 6 else
                                     "supported" if (np.median(v) > 0 and p < a) else "not supported"),
                         "detail": f"median rho = {np.median(v) if v.size else np.nan:.3f}, p = {p:.4g}"})

        if len(mt):
            base = mt.query("reuse_factor == @self.rf and family == @NO_REUSE and band == 'all'")
            add("H7a", "matched HF > LF word evidence in no_reuse (sign-flip permutation)",
                int(((base["mean"] > 0) & (base.p_sign_flip < a)).sum()), len(base),
                f"median pairs = {base.n_pairs.median() if len(base) else 0:.0f}")
            full = mt.query("reuse_factor == @self.rf and family == 'full' and band == 'all'")
            add("H7b", "on matched pairs, full damages LF more than HF (DiD > 0, sign-flip)",
                int(((full["mean"] > 0) & (full.p_sign_flip < a)).sum()), len(full), "")

        if len(rel):
            lin = rel.query("policy == @NO_REUSE and probe == 'linear'")
            tol = 3 * np.sqrt(0.25 / lin.n_test)        # 3 binomial SEs around chance
            lin = lin.assign(ok=(lin.selectivity_accuracy > 0) & ((lin.test_accuracy_control - 0.5).abs() < tol))
            ok = lin.groupby("model")["ok"].all()
            add("RQ8", "linear probe selective at every reliability layer, random-label control "
                "within 3 binomial SE of chance", int(ok.sum()), len(ok), "")
        return pd.DataFrame(rows)

    # ── figures ───────────────────────────────────────────────────────
    def _save(self, fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(self.out, f"{name}.png"), dpi=self.cfg.DPI, bbox_inches="tight")
        fig.savefig(os.path.join(self.out, f"{name}.pdf"), bbox_inches="tight")
        plt.close(fig)

    def fig_layerwise(self):
        lt = self.cat("layers")
        for m, g in lt.groupby("model"):
            g = g[(g.reuse_factor == self.rf) | (g.policy == NO_REUSE)]
            ec = exempt_cutoff(int(g.layer.max()) + 1, self.cfg.REUSE_EXEMPT_FRACTION)
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
            for ax, ro in zip(axes, READOUTS):
                fam = g[g.readout == ro].groupby(["family", "layer"])["auc_all"].mean().reset_index()
                for f, h in fam.groupby("family"):
                    ax.plot(h.layer, h.auc_all, lw=2.5 if f in (NO_REUSE, "full") else 1.2, label=f)
                ax.axvspan(-0.5, ec - 0.5, color="0.9", zorder=0)
                ax.set(xlabel="layer", ylabel="AUC (words vs nonwords)", title=f"{m}: {ro} readout")
                ax.grid(alpha=0.3)
            axes[1].legend(fontsize=7, ncol=2)
            self._save(fig, f"fig_layerwise_{safe_name(m)}")

    def fig_forest(self):
        b = self.cat("bands")
        b = b.query("reuse_factor == @self.rf and readout == 'frozen' and band == 'all' and metric == 'all'")
        if b.empty:
            return
        fams = sorted(b.family.unique())
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(fams) + 2))
        for i, f in enumerate(fams):
            g = b[b.family == f]
            ax.scatter(g.estimate, np.full(len(g), i), s=18, alpha=0.6)
            ax.plot([g.estimate.median()] * 2, [i - 0.3, i + 0.3], color="k", lw=2)
        ax.axvline(0, color="r", ls="--")
        ax.set(yticks=range(len(fams)), yticklabels=fams,
               xlabel="mean delta AUC over eligible layers vs no_reuse (frozen readout)",
               title="Per-model band effects (dots) and cross-model median (bar)")
        self._save(fig, "fig_band_forest")

    def fig_fragility(self):
        fr = self.cat("fragility")
        if fr.empty:
            return
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for m, g in fr.groupby("model"):
            g = g[g.readout_layer == g.readout_layer.max()].sort_values("depth")
            axes[0].plot(g.depth, g.delta_all, label=m)
            axes[1].plot(g.depth, g.delta_did_lf_minus_hf, label=m)
        axes[0].set(xlabel="normalised depth of the substituted layer",
                    ylabel="delta AUC at final layer", title="Single-layer KV substitution fragility")
        axes[1].set(xlabel="normalised depth of the substituted layer",
                    ylabel="delta AUC(LF) - delta AUC(HF)", title="Frequency dependence of fragility")
        for ax in axes:
            ax.axhline(0, color="k", lw=0.8)
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        self._save(fig, "fig_fragility")

    def fig_pareto(self):
        b = self.cat("bands")
        b = b.query("reuse_factor == @self.rf and readout == 'frozen' and band == 'all' and metric == 'all'")
        if b.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        for f, g in b.groupby("family"):
            ax.scatter(g.kv_memory_fraction, g.estimate, label=f, s=22)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xlabel="analytic KV-cache memory fraction (cached layers / layers)",
               ylabel="delta AUC vs no_reuse (frozen, all eligible layers)",
               title="Lexical fidelity vs KV-cache memory")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        self._save(fig, "fig_pareto")

    def fig_token_strata(self):
        ts = self.cat("token_strata")
        if ts.empty:
            return
        ts = ts[(ts.reuse_factor == self.rf) & (ts.family == "full")]
        strata = list(dict.fromkeys(ts.token_stratum))
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.8 / max(ts.model.nunique(), 1)
        for i, (m, g) in enumerate(ts.groupby("model")):
            g = g.set_index("token_stratum").reindex(strata)
            ax.bar(np.arange(len(strata)) + i * width, g.estimate, width, label=m)
        ax.axhline(0, color="k", lw=0.8)
        ax.set(xticks=np.arange(len(strata)) + 0.4, xticklabels=strata, xlabel="subword tokens",
               ylabel="delta token-matched AUC (full vs no_reuse)",
               title="Damage within token-count strata (words vs nonwords of equal length in tokens)")
        ax.legend(fontsize=7)
        self._save(fig, "fig_token_strata")

    def _latex(self, df: pd.DataFrame, name: str, caption: str):
        with open(os.path.join(self.out, f"{name}.tex"), "w") as f:
            f.write("\\begin{table}[t]\n\\centering\n\\small\n"
                    + df.to_latex(index=False, escape=True, float_format="%.3g")
                    + f"\\caption{{{caption}}}\n\\label{{tab:{name}}}\n\\end{{table}}\n")

    def run(self):
        for name in ["layers", "bands", "negative_controls", "contrasts", "item_regression",
                     "token_strata", "matched_tests", "matching_balance", "fragility",
                     "fragility_mechanism", "reliability", "downstream", "downstream_correlation",
                     "calibration"]:
            df = self.cat(name)
            if len(df):
                df.to_csv(os.path.join(self.out, f"all_{name}.csv"), index=False)
        mu = self.models_as_units()
        mu.to_csv(os.path.join(self.out, "models_as_units.csv"), index=False)
        peaks, alignment = self.fragility_profiles()
        if len(peaks):
            peaks.to_csv(os.path.join(self.out, "fragility_peaks.csv"), index=False)
        rules = self.decision_rules(peaks, alignment)
        rules.to_csv(os.path.join(self.out, "decision_rules.csv"), index=False)
        self._latex(rules[["hypothesis", "models_meeting_rule", "models_evaluated", "verdict"]],
                    "decision_rules", "Pre-registered decision rules evaluated on all models.")
        prim = mu.query("reuse_factor == @self.rf and readout == 'frozen' and metric == 'all'")
        if len(prim):
            self._latex(prim[["family", "band", "n_models", "median_estimate", "n_negative", "p_holm"]],
                        "models_as_units", "Change in AUC vs no-reuse with models as units "
                        "(Wilcoxon signed-rank, Holm over policy x band).")
        self.fig_layerwise()
        self.fig_forest()
        self.fig_fragility()
        self.fig_pareto()
        self.fig_token_strata()
        logger.info("\n" + rules[["hypothesis", "models_meeting_rule", "models_evaluated",
                                  "verdict"]].to_string(index=False))


# ════════════════════════════════════════════════════════════════════════════
# EXPERIMENT
# ════════════════════════════════════════════════════════════════════════════

REFERENCES = [
    "Brandon et al. (2024) Reducing Transformer KV Cache Size with Cross-Layer Attention. NeurIPS.",
    "Liu et al. (2024) MiniCache: KV Cache Compression in Depth Dimension for LLMs. NeurIPS.",
    "Yang et al. (2024) KVSharer: Efficient Inference via Layer-Wise Dissimilar KV Cache Sharing. arXiv:2410.18517.",
    "Wu & Tu (2024) Layer-Condensed KV Cache for Efficient Inference of LLMs. ACL.",
    "Wu, Wu & Tu (2025) A Systematic Study of Cross-Layer KV Sharing for Efficient LLM Inference. NAACL (short).",
    "Kaplan et al. (2025) From Tokens to Words: On the Inner Lexicon of LLMs. ICLR.",
    "Feucht et al. (2024) Token Erasure as a Footprint of Implicit Vocabulary Items in LLMs. EMNLP.",
    "Balota et al. (2007) The English Lexicon Project. Behavior Research Methods.",
    "Kornblith et al. (2019) Similarity of Neural Network Representations Revisited. ICML.",
    "Song et al. (2012) Feature Selection via Dependence Maximization (unbiased HSIC). JMLR.",
    "Davari et al. (2023) Reliability of CKA as a Similarity Measure in Deep Learning. ICLR.",
    "Hewitt & Liang (2019) Designing and Interpreting Probes with Control Tasks. EMNLP.",
    "Voita & Titov (2020) Information-Theoretic Probing with Minimum Description Length. EMNLP.",
    "Belinkov (2022) Probing Classifiers: Promises, Shortcomings, and Advances. Computational Linguistics.",
    "Hautus (1995) Corrections for extreme proportions in d'. Behavior Research Methods.",
    "Efron & Tibshirani (1993) An Introduction to the Bootstrap. Chapman & Hall.",
    "Holm (1979) A Simple Sequentially Rejective Multiple Test Procedure. Scand. J. Statistics.",
    "Benjamini & Hochberg (1995) Controlling the False Discovery Rate. JRSS-B.",
    "Wilcoxon (1945) Individual Comparisons by Ranking Methods. Biometrics Bulletin.",
    "MacKinnon & White (1985) Heteroskedasticity-consistent covariance matrix estimators (HC3). J. Econometrics.",
    "MacCallum et al. (2002) On the Practice of Dichotomization of Quantitative Variables. Psychological Methods.",
    "Stuart (2010) Matching Methods for Causal Inference. Statistical Science.",
    "Xiao et al. (2024) Efficient Streaming Language Models with Attention Sinks. ICLR.",
]


class Experiment:
    def __init__(self, cfg: ExperimentConfig, analysis_only: bool = False):
        self.cfg, self.analysis_only = cfg, analysis_only

    def run(self):
        t0 = datetime.now()
        items = load_items(self.cfg)
        items.to_csv(os.path.join(self.cfg.OUTPUT_DIR, "items.csv"), index=False)
        splits = make_splits(items, self.cfg)
        per_model, failed = {}, {}
        for mc in self.cfg.MODELS:
            if not self.analysis_only:
                try:
                    ModelPipeline(self.cfg, mc, items, splits).run()
                except Exception as e:
                    logger.exception(f"pipeline failed for {mc.name}")
                    failed[mc.name] = f"pipeline: {e!r}"
                    free_memory()
                    continue
                free_memory()
            if not os.path.exists(os.path.join(self.cfg.model_dir(mc.name), "model_info.json")):
                continue
            try:
                per_model[mc.name] = ModelAnalysis(self.cfg, mc.name).run()
            except Exception as e:
                logger.exception(f"analysis failed for {mc.name}")
                failed[mc.name] = f"analysis: {e!r}"
        if per_model:
            CrossModelAnalysis(self.cfg, per_model).run()
        with open(os.path.join(self.cfg.OUTPUT_DIR, "metadata.json"), "w") as f:
            json.dump({"script": "KV_LDT_v11", "started": f"{t0:%Y-%m-%d %H:%M:%S}",
                       "finished": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                       "torch": torch.__version__, "transformers": transformers.__version__,
                       "compute_dtype": str(COMPUTE_DTYPE), "models_analysed": list(per_model),
                       "failures": failed,
                       "config": {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None)))
                                      else str(v)) for k, v in vars(self.cfg).items() if k != "MODELS"},
                       "models": [vars(m) for m in self.cfg.MODELS],
                       "references": REFERENCES}, f, indent=2, default=str)
        logger.info(f"DONE in {datetime.now() - t0} -> {self.cfg.OUTPUT_DIR}")
        return per_model


def main():
    cfg = ExperimentConfig()
    if os.environ.get("KV_MODELS"):
        wanted = {m.strip() for m in os.environ["KV_MODELS"].split(",") if m.strip()}
        cfg.MODELS = [m for m in cfg.MODELS if m.name in wanted]
        if not cfg.MODELS:
            raise SystemExit(f"No model matched KV_MODELS={os.environ['KV_MODELS']!r}")
    if os.environ.get("KV_MAX_ITEMS_PER_CLASS"):
        cfg.MAX_ITEMS_PER_CLASS = int(os.environ["KV_MAX_ITEMS_PER_CLASS"])
    analysis_only = os.environ.get("KV_ANALYSIS_ONLY", "").lower() in ("1", "true", "yes")
    for path in (cfg.WORDS_PATH, cfg.NONWORDS_PATH):
        if not os.path.exists(path):
            raise SystemExit(f"Missing input file: {path}")
    return Experiment(cfg, analysis_only=analysis_only).run()


if __name__ == "__main__":
    main()
