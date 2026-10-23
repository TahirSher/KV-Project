"""
COVE_KV.py — COVE: COVerage-priced Eviction of the KV cache for query-agnostic reuse, decoded by the
model's own attention kernel.

A training-free, calibration-free KV-cache compressor for the compress-the-context protocol (the
context is compressed once, before the question is seen; NVIDIA kvpress' setting).  Every KV head
keeps the same number of tokens (sinks, the recent window, and the tokens with the largest
COVERAGE score: the expected contribution to an exact bound on the attention-output error of
eviction, under the context's OWN queries sampled over its whole length).  The compressed cache is
a dense tensor of the model's dtype: decoding is the model's fused SDPA over fewer tokens, so the
speed-up is that of attention over n_keep instead of n tokens, with no custom kernel.  The
evaluation engine (LongBench, LongBench-v2, RULER, long PPL, official kvpress baselines, paired
tests, decision rules) is the engine of STRATA_KV / QUASAR_KV / HEDGE_KV.

════════════════════════════════════════════════════════════════════════════════════════
1. WHY THIS DESIGN (measured, Llama-3.2-1B-Instruct, A100-class MIG slice, bf16, CUDA graphs)
════════════════════════════════════════════════════════════════════════════════════════
  Decode-step attention of one layer (8 KV heads, head_dim 64), microseconds:
      context      full SDPA   SDPA over 1/4 | 1/8 | 1/16 of the tokens   STRATA kernel 1/4 | 1/8 | 1/16
      4 096          16.5         11.2 | 11.4 |  9.8                         130 | 121 | 112
      16 384         33.2         15.7 | 16.4 | 12.0                         163 | 170 | 148
      65 536         89.1         31.2 | 23.4 | 18.2                         264 | 344 | 261
  * Attention over the kept tokens is 1.5-4.9x faster than the full cache at exactly the stored
    bytes; every mixed-precision kernel we wrote (QUASAR, STRATA) was 3-15x SLOWER than the full
    cache, and STRATA's time did not fall with its bytes (fixed costs dominated).  At equal bytes
    no mixed-precision cache can be faster than eviction (it reads the same bytes and must also
    decode them), and our kernels were far from that bound.  COVE therefore decodes with SDPA.
  * The ingredient that carried the accuracy of all three earlier methods was the pricing of
    tokens by the context's own queries over its whole length: +9.1 / +18.6 points (HedgeKV,
    SmolLM3-3B, 1/8 / 1/16) and +4.2 (QUASAR, Llama-3.2-1B, 1/8) over pricing by the end window
    (p <= 0.001).  The chunk-local int8 eviction baseline (which also prices every region of the
    document by its own queries) beat SnapKV by +9 to +11 points on Llama at 1/8 and 1/16.
  * Low-bit codes hurt on small models: KIVI-2 lost 8 points to the full cache on Llama-3.2-1B
    qasper; mixed precision gained 1-3 points over int8 eviction at best.
  COVE keeps the pricing, prices eviction with the right error model (section 3), and spends the
  bytes on exact tokens that the model's own kernel reads.

════════════════════════════════════════════════════════════════════════════════════════
2. METHOD (one-shot, per layer, after the prefill of the context; all on the device)
════════════════════════════════════════════════════════════════════════════════════════
  (a) Coverage queries.  COVE_QUERIES query positions sampled in every COVE_SEGMENT-token segment
      of the context, plus the last W_OBS positions (the window group).  Segment groups count only
      reads at distance >= COVE_LOCAL; the window group counts every read.  No question token is
      used.  On the deployment path the query rows are captured during the model's own cached
      prefill (no extra forward pass, unlike KVzip's reconstruction).
  (b) Score.  For a sampled query (head a) with exact probabilities p and output o, token j gets
      p_{a,j} ||v_j - o_a||_{M_a}, M_a = W_O,a^T W_O,a (the error is measured in the residual
      stream, where it acts), summed over the query heads of its KV head and averaged over the
      group's queries; each group is normalised by its total over the layer and the groups are
      averaged (every part of the context has the same say).  COVE_POOL > 1 smooths the scores by
      a moving average (SnapKV smooths its window scores with 5; the coverage score already
      averages many queries, and on the smoke model smoothing hurt: default 1, ablation cove_pool5).
  (c) Selection.  n_keep = int(P x keep) per KV head (kvpress' count, so memory matches the
      official presses token for token): N_SINK sinks + the last WINDOW context tokens + the
      n_keep - N_SINK - WINDOW highest-scoring tokens.  Same count per head, different tokens:
      the cache is dense (1, Hkv, n_keep, d) in the model dtype.
  (d) Decoding.  The model's own attention (SDPA) over the compressed cache, on the same path as
      the official kvpress presses (native_cove_generate replaces Hugging Face's DynamicCache
      layers in place); question tokens and generated tokens are appended exact.

════════════════════════════════════════════════════════════════════════════════════════
3. THEORY (cove_tests checks every identity numerically at start-up)
════════════════════════════════════════════════════════════════════════════════════════
  Prop 1 (eviction error, exact).  For one query with probabilities p, output o = sum_j p_j v_j and
  an evicted set S of mass P_S < 1, the output over the kept keys is o' = (o - sum_S p_j v_j) / (1 -
  P_S), hence  o' - o = - sum_{j in S} p_j (v_j - o) / (1 - P_S)  and, in any norm,
        ||o' - o|| <= ( sum_{j in S} p_j ||v_j - o|| ) / (1 - P_S).
  The numerator is additive over tokens: for a fixed number kept, keeping the tokens with the
  largest p_j ||v_j - o|| minimises it exactly.  The bound is tight when the evicted tokens' errors
  are coherent (their v_j - o align), which is how eviction errors behave (eviction removes mass,
  it does not add zero-mean noise); the squared per-token price of HedgeKV / QUASAR / STRATA,
  sum_S p_j^2 ||v_j - o||^2 / (1 - p_j)^2, is the independent-error model, right for quantisation
  noise, optimistic for eviction (cove_l2 is that ablation).
  Prop 2 (coverage).  With the expectation over the context's own queries in every segment, the
  score bounds the expected eviction error of a future query distributed like any mixture of the
  segments (linearity of the expectation in the mixture weights); the end-window score bounds it
  only for queries like the end window.  The window group is kept, so the question-like queries
  that the end window represents are covered too.
  Prop 3 (decode cost).  With the same number of tokens per head the cache is a dense tensor and
  decoding is attention over n_keep tokens: memory and attention FLOPs / bytes per step are exactly
  keep x those of the full cache (+ the appended question and answer), with the model's kernel.

  Prior work (venues verified Oct 2026 unless marked): H2O (NeurIPS 2023, cumulative attention, a
  causally biased whole-context score), StreamingLLM (ICLR 2024), SnapKV (NeurIPS 2024, end-window
  attention + pooling), TOVA (EMNLP 2024), PyramidKV (COLM 2025), Ada-KV (NeurIPS 2025), Expected
  Attention (arXiv 2510.00636, Gaussian future queries), KVzip (NeurIPS 2025 oral, query-agnostic by
  reconstructing the context, 2-3x prefill cost), Compactor (arXiv 2507.08143, query-agnostic
  leverage scores); value-aware scores (e.g. VATP) weight attention by value norms.  KVzip and
  Compactor are run as official kvpress baselines.  Not found in the literature as stated (searched
  Oct 2026; re-check before submission): pricing eviction by the exact bound of Prop. 1 in the
  output-projection metric under the context's own queries sampled over its whole length with
  per-segment normalisation.  The contribution is a scoring rule and its theory, NOT a new kernel
  or architecture; it must win on accuracy at equal memory and speed against KVzip, Compactor,
  Expected Attention and SnapKV to be a paper.

════════════════════════════════════════════════════════════════════════════════════════
4. METHODS COMPARED AT MATCHED MEMORY
════════════════════════════════════════════════════════════════════════════════════════
  cove            COVE (proposed), deployment path (HF SDPA, bf16), every budget
  cove_window     end-window scores only (SnapKV-type prior)              (ablation, PRIMARY_KEEP)
  cove_l2         squared independent-error price                         (ablation, PRIMARY_KEEP)
  cove_pool5      SnapKV-style smoothing of the scores (kernel 5)          (ablation, PRIMARY_KEEP)
  cove_ref        COVE on the fp32 harness (path check of cove)            (PRIMARY_KEEP)
  kvpress:*       OFFICIAL NVIDIA kvpress 0.5.5: SnapKV, PyramidKV, AdaKV-SnapKV, Expected
                  Attention, TOVA, Knorm, StreamingLLM, KVzip, Compactor (same path as cove);
                  AdaKV, Expected Attention (AdaKV-wrapped) and KVzip mask keys: their memory is
                  NOMINAL (flagged), COVE's is real
  snapkv_q2/_q4   SnapKV + KIVI-2 / KIVI-4 at matched bytes (eviction + quantisation hybrid)
  evict, am       chunk-local int8 eviction / Attention-Matching-lite
  snapkv, streamingllm, kivi2, kivi4   harness replicas / KIVI
  Pre-registered contrasts: cove vs every baseline at every budget; vs its ablations and KIVI at
  PRIMARY_KEEP; path checks cove vs cove_ref, full vs native_full, snapkv vs kvpress:snapkv.

════════════════════════════════════════════════════════════════════════════════════════
5. SYSTEMS MEASUREMENTS
════════════════════════════════════════════════════════════════════════════════════════
  cove rows (native): prefill_s (model's cached prefill), compress_s (coverage scores + selection,
  all layers, GPU-synchronised), ttft_s, decode_s, tokens_per_s, peak_*_bytes, cache_bytes_measured
  (the compressed cache tensors), memory_fraction (n_keep / P).  microbench.csv: one layer's
  decode-step attention, full SDPA vs SDPA over the COVE cache at every budget (CUDA graphs on
  GPU), and COVE's compression time for that layer.

Usage
    python COVE_KV.py --smoke                                     # self-tests + benchmark engine, CPU
    python COVE_KV.py --microbench 4096 16384 65536 131072 --micro-shape 64 8 4
    python COVE_KV.py --report run.log [bench_samples.csv ...] --output report_dir   # interim paired report
    python COVE_KV.py --suite longbench --models Llama-3.2-1B-Instruct --tasks qasper hotpotqa trec
    python COVE_KV.py --suite longbench --models Llama-3.2-1B-Instruct --tasks qasper \\
           --methods cove kvpress:kvzip kvpress:compactor kvpress:snapkv kvpress:expected_attention snapkv_q
Requires torch >= 2.5, transformers >= 4.56 (< 5.3 for kvpress 0.5.5; tested on 5.2.0), scipy, pandas,
datasets, huggingface_hub, fuzzywuzzy or rapidfuzz, rouge, kvpress == 0.5.5.

VALIDATION (be explicit in any write-up; nothing below is a real-LLM result)
  Run offline on CPU only (Python 3.13, torch 2.14.1, transformers 5.2.0, kvpress 0.5.5).  NOT run
  here: real LLMs, real Hugging Face datasets, GPU timings of COVE itself (the GPU numbers of
  section 1 are the user's STRATA microbenchmark: SDPA over 1/4-1/16 of the tokens, which is COVE's
  decode path).  Every accuracy claim is a hypothesis for the pre-registered decision rules.
  * cove_tests: Prop. 1 identity exact (8e-14 on 200 random sets) and the bound holds (largest
    ||o' - o|| / bound = 0.87); selection returns n_keep sorted distinct positions per head with
    sinks and window; a budget of the whole context keeps every token; a token read only by an early
    segment ranks in the top 5% of the coverage score (37th percentile under the end-window score)
    and is kept.
  * --smoke (tiny character-level Llama, synthetic stand-ins of every benchmark; official kvpress
    presses through the pipes shim): harness == SDPA (1.3e-5); evict keep-all and COVE keep-all ==
    full cache (0 for COVE); decoding from the compressed state == single pass for every harness
    method (1.000); COVE's deployment path (HF SDPA + compressed DynamicCache) == its harness twin
    cove_ref (1.000); full == HF generate(); native path == kvpress' official pipeline (1.000).
    kvpress:kvzip cannot run on the random-init smoke model (it loads a tokenizer by the model's
    hub name, which that model has not); the failure is recorded per item, not fatal.
  * Fidelity, prefix protocol (smoke model, budget 0.25, KL(full || method), lower is better):
    cove 0.299 | cove_l2 0.459 | cove_pool5 0.472 | cove_window 0.490 | SnapKV 1.203 |
    StreamingLLM 1.138 | SnapKV + KIVI-4 1.086 | chunk-local int8 eviction 0.189 (twice as many
    tokens at int8) | KIVI-4 0.0017 at 0.52 memory.  Every ablation moves the way section 3 predicts;
    a toy model: no accuracy claim is drawn from it.
  * Microbenchmark on CPU (eager): SDPA over the COVE cache 1.4-1.9x faster than over the full cache
    at 512-2048 tokens; compression 5-29 ms per layer on CPU (record compress_s on GPU).
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
logger = logging.getLogger("cove_kv")


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
    OUTPUT_DIR: str = str(Path(os.environ.get("KV_COVE_OUTPUT_DIR", PROJECT_ROOT / "cove_results")))

    # ── Cache layout (all methods) ────────────────────────────────────
    N_SINK: int = 4
    WINDOW: int = 32               # exact recent window (incl. current token)
    CHUNK: int = 128               # tokens per chunk (chunk-local baselines)
    W_OBS: int = 32                # observed queries per chunk (chunk-local baselines); COVE's window group
    KEEP_BITS: int = 8             # precision of the chunk-local baselines' stored tokens

    # ── COVE ──────────────────────────────────────────────────────────
    COVE_SEGMENT: int = 512        # context positions per coverage group
    COVE_QUERIES: int = 16         # sampled query positions per coverage group
    COVE_LOCAL: int = 64           # segment groups count reads at distance >= COVE_LOCAL (long-range attention)
    COVE_POOL: int = 1             # moving average of the scores over positions (1 = none; cove_pool5 ablation: 5)
    SHARED_PREFILL: bool = True    # one exact context pass per sample for all one-shot harness methods
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
                                        "tova", "knorm", "streaming_llm", "kvzip", "compactor")
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
        if self.COVE_POOL < 1 or self.COVE_POOL % 2 == 0:
            raise ValueError("COVE_POOL must be an odd positive integer")
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
# COVE 1/3: THE ERROR OF EVICTING A SET OF TOKENS (exact identity and bound)
# ════════════════════════════════════════════════════════════════════════════

def eviction_shift(p: torch.Tensor, V: torch.Tensor, evicted: torch.Tensor) -> torch.Tensor:
    """
    Prop. 1 (exact).  For one query with probabilities p over n keys, output o = p V, and a set S of
    evicted keys with mass P_S < 1, the output after eviction is o' = (o - sum_S p_j v_j) / (1 - P_S):
        o' - o = - sum_{j in S} p_j (v_j - o) / (1 - P_S).
    Hence, in any norm (here the residual-stream metric of the output projection),
        ||o' - o|| <= sum_{j in S} p_j ||v_j - o|| / (1 - P_S)                       (bound)
    and the numerator is ADDITIVE over tokens: keeping the tokens with the largest
    s_j = p_j ||v_j - o|| minimises the numerator exactly for a given number kept.  The bound holds
    for coherent errors (evicted tokens whose v_j - o point the same way, which is how eviction
    errors behave: they remove mass), unlike a sum of squared per-token errors, which assumes
    independent zero-mean errors (true for quantisation noise, not for eviction).
    Returns o' - o for p (..., n), V (n, d), evicted (n,) bool.
    """
    o = p @ V
    ps = p * evicted
    return -(ps @ V - ps.sum(-1, keepdim=True) * o) / (1 - ps.sum(-1, keepdim=True))


# ════════════════════════════════════════════════════════════════════════════
# COVE 2/3: COVERAGE SCORES (the context's own queries over its whole length)
# ════════════════════════════════════════════════════════════════════════════

def coverage_groups(H: int, cfg: "Config") -> List[np.ndarray]:
    """
    Observation positions (all < H: no question, answer or future token is ever used).  One
    group per context segment of COVE_SEGMENT positions (COVE_QUERIES positions sampled without
    replacement, seeded by (H, segment)), plus the last W_OBS context positions, the 'window'
    group (always last), which is the only group SnapKV-style scoring uses.  Positions start at
    N_SINK + COVE_LOCAL, the first that can see an evictable key at long range.
    """
    lo, win = cfg.N_SINK + cfg.COVE_LOCAL, max(cfg.N_SINK + cfg.COVE_LOCAL, H - cfg.W_OBS)
    groups = []
    for s in range(lo, win, cfg.COVE_SEGMENT):
        seg = np.arange(s, min(s + cfg.COVE_SEGMENT, win))
        n = min(len(seg), cfg.COVE_QUERIES)
        groups.append(np.sort(_rng("cover_group", H, s).choice(seg, n, replace=False)))
    groups.append(np.arange(win, H))
    return [g_ for g_ in groups if len(g_)]


@torch.no_grad()
def coverage_scores(q, k, v, scaling, H: int, rs: int, re: int, M: torch.Tensor, cfg: "Config",
                    window_only: bool = False, score: str = "l1") -> torch.Tensor:
    """
    Importance of every evictable token j in [rs, re) of one layer, for each KV head, as the
    expected contribution to the eviction bound (Prop. 1) under the context's own queries:
        l1 (COVE)   s_j = E[ sum_a p_{a,j} ||v_j - o_a||_{M_a} ]      (numerator of the bound)
        l2          s_j = E[ sum_a (p_{a,j} / (1 - p_{a,j}))^2 ||v_j - o_a||^2_{M_a} ]   (independent-error
                    model: the drop price of HedgeKV / QUASAR / STRATA; ablation)
    with p, o the exact attention of query head a at a sampled position, M_a = W_O,a^T W_O,a (the
    error is measured where it acts, in the residual stream).  Each group's scores are averaged
    over its queries and normalised by their sum over the layer (every segment of the context has
    the same say: coverage), then averaged over groups.  Reads at distance < COVE_LOCAL count only
    for the window group (it stands for the question, which reads everything).  Returns (Hkv, n);
    no host synchronisation.
    """
    Hkv, d, n = k.shape[1], k.shape[-1], re - rs
    rep = q.shape[1] // Hkv
    groups = coverage_groups(H, cfg)
    if window_only:
        groups = groups[-1:]
    dev = k.device
    out = torch.zeros((Hkv, n), device=dev)
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
        local = 1 if gi == len(groups) - 1 else cfg.COVE_LOCAL
        far = torch.arange(rs, te, device=dev) <= (pos[:, None] - local)
        pR = p[..., rs:te] * far
        oM = torch.einsum("hrqd,hrde->hrqe", o, Mg)
        u2 = (vMv[:, :, None, :nt] - 2 * (oM.reshape(Hkv, rep * nq, d) @ VR[:, :nt].transpose(-1, -2))
              .view(Hkv, rep, nq, nt) + (oM * o).sum(-1)[..., None]).clamp_min(0.0)   # ||v_j - o||^2_M
        if score == "l1":
            c = pR * u2.sqrt()
        else:
            c = (pR / (1 - pR).clamp_min(1e-6)).pow(2) * u2
        cg = c.sum((1, 2)) / nq                                                   # (Hkv, nt)
        N = cg.sum()
        out[:, :nt] += cg * torch.where(N > 0, 1.0 / N.clamp_min(1e-30), torch.zeros_like(N))
        live = live + (N > 0).float()
    return out / live.clamp_min(1.0)


# ════════════════════════════════════════════════════════════════════════════
# COVE 3/3: HEAD-SYNCHRONOUS SELECTION (dense caches the model's own attention kernel reads)
# ════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def cove_select(q, k, v, scaling, H: int, n_keep: int, M: torch.Tensor, cfg: "Config",
                window_only: bool = False, score: str = "l1", pool: Optional[int] = None) -> torch.Tensor:
    """
    The positions every KV head keeps (Hkv, n_keep), sorted: the N_SINK sinks, the last WINDOW
    context tokens, and the n_keep - N_SINK - WINDOW evictable tokens with the largest coverage
    score (optionally smoothed by a moving average of `pool` positions, as SnapKV smooths its
    window scores; the coverage score is already an average over many queries, so the default is
    no smoothing).  Every head keeps the same number of tokens (different
    tokens), so the compressed cache is a dense (1, Hkv, n_keep, d) tensor: decoding needs no
    custom kernel, and its cost is the cost of attention over n_keep tokens.
    """
    c = cfg
    Hkv, dev = k.shape[1], k.device
    rs, re = min(c.N_SINK, H), max(min(c.N_SINK, H), H - c.WINDOW)
    exact = torch.cat([torch.arange(rs, device=dev), torch.arange(re, H, device=dev)])
    if n_keep >= H:
        return torch.arange(H, device=dev).expand(Hkv, H)
    if n_keep <= len(exact) or re <= rs:                    # budget below sinks + window: the most recent tokens
        keep = torch.cat([torch.arange(rs, device=dev), torch.arange(H - max(n_keep - rs, 0), H, device=dev)])[:n_keep]
        return keep.sort().values.expand(Hkv, n_keep)
    s = coverage_scores(q, k, v, scaling, H, rs, re, M, c, window_only, score)
    w = c.COVE_POOL if pool is None else pool
    if w > 1:
        s = F.avg_pool1d(s[:, None], kernel_size=w, padding=w // 2, stride=1)[:, 0, :s.shape[-1]]
    mid = s.topk(n_keep - len(exact), dim=-1).indices + rs
    return torch.cat([exact.expand(Hkv, len(exact)), mid], 1).sort(-1).values


class CoveMethod(LayerMethod):
    """
    COVE on the reference harness (fp32 attention): one-shot at the horizon, every KV head keeps
    the positions cove_select returns.  It is the harness twin of the deployment path (cove_*
    specs run through native_cove_generate on the model's own SDPA kernel); generation
    equivalence compares the two.  Variants: window_only (end-window scores: the SnapKV-type
    prior), score 'l2' (independent-error price), pool 5 (SnapKV-style smoothing).
    """
    family = "cove"

    def __init__(self, cfg, keep: float, name: str, runner: "Runner", window_only: bool = False,
                 score: str = "l1", pool: Optional[int] = None):
        super().__init__(cfg)
        self.keep, self.name, self.runner = keep, name, runner
        self.window_only, self.score, self.pool = window_only, score, pool
        self.metrics: Dict[int, torch.Tensor] = {}

    def out_metric(self, li: int, dev) -> torch.Tensor:
        """M_a = W_O,a^T W_O,a (d x d) for every query head a of layer li (from the weights)."""
        if li not in self.metrics:
            W = self.runner.layers[li].self_attn.o_proj.weight.float()            # (hidden, Hq d)
            d, Hq = self.runner.d, self.runner.Hq
            Wa = W.view(W.shape[0], Hq, d).permute(1, 0, 2)                       # (Hq, hidden, d)
            self.metrics[li] = (Wa.transpose(-1, -2) @ Wa).to(dev)
        return self.metrics[li]

    def select(self, li: int, q, k, v, scaling, H: int) -> torch.Tensor:
        return cove_select(q, k, v, scaling, H, kvpress_n_kept(H, self.keep), self.out_metric(li, k.device),
                           self.cfg, self.window_only, self.score, self.pool)

    def entries(self, li, q, k, v, scaling, pos):
        T = k.shape[2]
        H = self.horizon(T)
        B, Hkv = k.shape[0], k.shape[1]
        if self.freeze_at is None or kvpress_n_kept(H, self.keep) >= H or B != 1:
            return LayerMethod.entries(self, li, q, k, v, scaling, pos)
        idx = self.select(li, q, k, v, scaling, H)
        kept = torch.zeros((B, Hkv, T), dtype=torch.bool, device=k.device)
        kept[0].scatter_(-1, idx, True)
        kept[..., H:] = True                                               # post-horizon tokens
        hide = torch.where(kept, torch.full((B, Hkv, T), BIG, dtype=torch.long, device=k.device),
                           torch.full((B, Hkv, T), H, dtype=torch.long, device=k.device))
        return k, v, hide, None


class CacheCapture:
    """During the model's own cached prefill (SDPA, model dtype): records the query rows at the
    kept positions of every layer and lets the attention run unchanged."""

    def __init__(self, keep_q: torch.Tensor):
        self.keep_q = keep_q
        self.Q, self.scaling = {}, {}

    def begin(self, positions):
        self.pos = positions

    def attend(self, module, q, k, v, mask, kwargs):
        li = module.layer_idx
        self.scaling[li] = kwargs.get("scaling") or module.scaling
        self.Q[li] = q[:, :, self.keep_q.to(q.device)]
        return _SDPA(module, q, k, v, mask, **kwargs)


@torch.no_grad()
def native_cove_generate(runner: "Runner", method: "CoveMethod", ids: List[int], context_len: int, max_new: int,
                         stop_ids: set, stop_fn=None) -> Tuple[List[int], Dict]:
    """
    COVE as it is deployed, on the same path as the official kvpress presses (Hugging Face SDPA +
    DynamicCache, model dtype): the context is prefilled into the model's cache (query rows kept at
    the coverage positions only), every layer's cache is replaced IN PLACE by the dense
    (1, Hkv, n_keep, d) selection, and the question and greedy decoding run through the model's own
    attention kernel.  compress_s is the selection time of all layers; ttft_s = prefill + compression
    + question; cache_bytes_measured is the size of the compressed cache tensors.
    """
    from transformers import DynamicCache
    if Router.active is not None:
        raise RuntimeError("native path must run without an active harness controller")
    model, H = runner.model, context_len
    keep_q = kept_query_positions(H, runner.cfg)
    cap = CacheCapture(keep_q)
    cap.begin(torch.arange(H, device=DEVICE))
    cache = DynamicCache()
    _reset_peak()
    _sync()
    t0 = time.perf_counter()
    with routed(cap):
        h = model.model(input_ids=torch.tensor([ids[:H]], device=DEVICE), past_key_values=cache).last_hidden_state
    _sync()
    prefill_s = time.perf_counter() - t0
    n_keep = min(kvpress_n_kept(H, method.keep), H)
    t1 = time.perf_counter()
    if n_keep < H:
        for li, lay in enumerate(cache.layers):
            K, V = lay.keys, lay.values
            idx = method.select(li, QRows(cap.Q[li], keep_q, H), K, V, cap.scaling[li], H)
            g = idx[None, :, :, None].expand(1, K.shape[1], n_keep, K.shape[-1])
            lay.keys, lay.values = K.gather(2, g).contiguous(), V.gather(2, g).contiguous()
            cap.Q[li] = None
    _sync()
    compress_s = time.perf_counter() - t1
    peak_prefill = _peak()
    cache_bytes = sum(tensor_bytes([lay.keys, lay.values]) for lay in cache.layers)
    t2 = time.perf_counter()
    if len(ids) > H:
        pos = torch.arange(H, len(ids), device=DEVICE)[None]
        logits = model(input_ids=torch.tensor([ids[H:]], device=DEVICE), past_key_values=cache,
                       position_ids=pos, logits_to_keep=1).logits[0, -1]
        pos = pos[:, -1:] + 1
    else:
        logits = model.get_output_embeddings()(h[0, -1])
        pos = torch.tensor([[H]], device=DEVICE)
    nxt = int(logits.float().argmax())
    _sync()
    ttft_s = prefill_s + compress_s + time.perf_counter() - t2
    full_bytes = H * runner.L * runner.Hkv * 2 * runner.d * next(model.parameters()).element_size()
    info = {"prompt_len": len(ids), "context_len": H, "prefill_s": prefill_s, "compress_s": compress_s,
            "ttft_s": ttft_s, "peak_prefill_bytes": peak_prefill, "path": "native", "full_cache_bytes": float(full_bytes),
            "model_dtype": str(next(model.parameters()).dtype), "attention_dtype": "model dtype (SDPA)",
            "cache_bytes_measured": float(cache_bytes), "memory_fraction": n_keep / H,
            "memory_fraction_measured": cache_bytes / full_bytes, **run_flags(),
            "attn_flops_first_step": 4.0 * runner.d * runner.Hq * runner.L * (n_keep + len(ids) - H + 1),
            "attn_flops_first_step_full": 4.0 * runner.d * (len(ids) + 1) * runner.Hq * runner.L}
    out = []
    _reset_peak()
    _sync()
    t3 = time.perf_counter()
    for i in range(max_new):
        out.append(nxt)
        if nxt in stop_ids or i == max_new - 1 or (stop_fn is not None and stop_fn(out)):
            break
        lg = model(input_ids=torch.tensor([[nxt]], device=DEVICE), past_key_values=cache, position_ids=pos).logits[0, -1]
        pos = pos + 1
        nxt = int(lg.float().argmax())
    _sync()
    decode_s = time.perf_counter() - t3
    info.update(n_generated=len(out), decode_s=decode_s, peak_decode_bytes=_peak(),
                tokens_per_s=(len(out) - 1) / decode_s if len(out) > 1 and decode_s > 0 else float("nan"),
                payload_bytes=float(cache_bytes))
    cache = None
    free_memory()
    return out, info


# ── Decode-attention microbenchmark (one layer, synthetic cache, CUDA graphs on GPU) ─────
@torch.no_grad()
def attention_microbench(cfg: "Config", d: int = 128, Hkv: int = 8, rep: int = 4, dtype=None) -> pd.DataFrame:
    """
    One decode step of attention for one layer: SDPA over the full cache vs SDPA over the COVE
    cache at every budget (the decode path COVE uses, identical to any evicting method's at the
    same token count), plus COVE's compression time for that layer (coverage scores + selection,
    GPU-synchronised wall clock) on a synthetic layer.  On CUDA the attention calls are captured in
    CUDA graphs and replayed; on CPU they are timed eagerly.
    """
    dtype = dtype or (COMPUTE_DTYPE if DEVICE.type == "cuda" else torch.float32)
    rows = []
    g = torch.Generator(device="cpu").manual_seed(SEED)
    fake = types.SimpleNamespace(d=d, Hq=Hkv * rep, layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(
        o_proj=types.SimpleNamespace(weight=torch.eye(d).repeat(1, Hkv * rep).to(DEVICE) / math.sqrt(Hkv * rep))))])
    for H in cfg.MICRO_LENGTHS:
        spec = torch.exp(-torch.arange(d) / (d / 8.0))
        Kc = (torch.randn(1, Hkv, H, d, generator=g) * spec.sqrt() + 2.0).to(DEVICE, dtype)
        Vc = torch.randn(1, Hkv, H, d, generator=g).to(DEVICE, dtype)
        keep_q = kept_query_positions(H, cfg)
        Qc = (torch.randn(1, Hkv * rep, len(keep_q), d, generator=g) * spec.sqrt()).to(DEVICE, dtype)
        q1 = torch.randn(1, Hkv * rep, 1, d, generator=g).to(DEVICE, dtype)
        full = lambda: F.scaled_dot_product_attention(q1, Kc, Vc, enable_gqa=True)
        base = _time_callable(full, cfg)
        rows.append({"context": H, "method": "full_sdpa", "us_per_step": base, "cache_bytes": tensor_bytes([Kc, Vc]),
                     "speedup_vs_full": 1.0, "compress_us": 0.0})
        for keep in cfg.KEEP_FRACTIONS:
            m = CoveMethod(cfg, keep, f"cove@{keep:g}", fake)
            _sync()
            t = time.perf_counter()
            idx = m.select(0, QRows(Qc, keep_q, H), Kc, Vc, d ** -0.5, H)
            _sync()
            comp = (time.perf_counter() - t) * 1e6
            gi = idx[None, :, :, None].expand(1, Hkv, idx.shape[-1], d)
            Ke, Ve = Kc.gather(2, gi).contiguous(), Vc.gather(2, gi).contiguous()
            us = _time_callable(lambda Ke=Ke, Ve=Ve: F.scaled_dot_product_attention(q1, Ke, Ve, enable_gqa=True), cfg)
            rows.append({"context": H, "method": f"cove@{keep:g}", "us_per_step": us, "cache_bytes": tensor_bytes([Ke, Ve]),
                         "speedup_vs_full": base / us if us > 0 else float("nan"), "compress_us": comp})
        logger.info("  microbench H=%d: " % H + ", ".join(
            f"{r['method']}={r['us_per_step']:.1f}us/{r['cache_bytes'] / 2**20:.2f}MiB (x{r['speedup_vs_full']:.2f}, "
            f"compress {r['compress_us'] / 1e3:.1f}ms)" for r in rows if r["context"] == H))
    return pd.DataFrame(rows).assign(device=str(DEVICE), dtype=str(dtype),
                                     timer="cuda_graph" if DEVICE.type == "cuda" else "eager_cpu")


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
# SELF-TESTS: Prop. 1 (exact eviction identity and bound), selection invariants, harness identities
# ════════════════════════════════════════════════════════════════════════════

def _fake_runner(d: int, Hkv: int, rep: int, W_o: Optional[torch.Tensor] = None, dev=None):
    """The attributes CoveMethod reads from a Runner (head_dim, heads, o_proj weight)."""
    dev = dev or DEVICE
    W = W_o if W_o is not None else torch.eye(d).repeat(1, Hkv * rep) / math.sqrt(Hkv * rep)
    attn = types.SimpleNamespace(o_proj=types.SimpleNamespace(weight=W.to(dev)))
    return types.SimpleNamespace(d=d, Hq=Hkv * rep, Hkv=Hkv, layers=[types.SimpleNamespace(self_attn=attn)])


@torch.no_grad()
def cove_tests(cfg: "Config") -> Dict:
    """Synthetic checks of every identity COVE relies on.  Raises on any failed identity."""
    out = {}
    g = torch.Generator().manual_seed(SEED)
    # Prop. 1: the eviction identity is exact and the bound holds, for random sets and queries
    n, d = 200, 16
    V = torch.randn(n, d, generator=g, dtype=torch.float64)
    worst_id, worst_ratio = 0.0, 0.0
    for _ in range(200):
        p = (torch.randn(n, generator=g, dtype=torch.float64) * 3).softmax(0)
        S = torch.rand(n, generator=g) < 0.6
        ref = (p * ~S) @ V / (p * ~S).sum() - p @ V                    # attention over the kept keys, minus o
        got = eviction_shift(p, V, S)
        worst_id = max(worst_id, float((ref - got).abs().max()))
        bound = float((p * S * (V - p @ V).norm(dim=-1)).sum() / (1 - (p * S).sum()))
        worst_ratio = max(worst_ratio, float(ref.norm()) / bound)
    out["prop1_identity_max_abs"], out["prop1_norm_over_bound_max"] = worst_id, worst_ratio
    if worst_id > 1e-10 or worst_ratio > 1 + 1e-9:
        raise RuntimeError(f"Prop. 1 failed: identity {worst_id}, norm / bound {worst_ratio}")
    # selection on a synthetic layer: sizes, sinks and window always kept, keep-all is the identity,
    # and the coverage score of a token that only an early segment reads is not zero while the
    # end-window score of that token is (what the window prior misses)
    Hkv, rep, dd, H = 2, 2, 16, 700
    fr = _fake_runner(dd, Hkv, rep, dev=torch.device("cpu"))
    tc = replace(cfg, COVE_SEGMENT=64, COVE_QUERIES=8, COVE_LOCAL=16)
    K = torch.randn(1, Hkv, H, dd, generator=g)
    Vx = torch.randn(1, Hkv, H, dd, generator=g)
    Q = torch.randn(1, Hkv * rep, H, dd, generator=g) * 0.3
    j_star = 150                                                         # read only by queries 200..260
    Q[:, :, 200:260] = K[0, :, j_star].repeat_interleave(rep, 0)[None, :, None] * 3.0
    m = CoveMethod(tc, 0.25, "t", fr)
    idx = m.select(0, Q, K, Vx, dd ** -0.5, H)
    n_keep = kvpress_n_kept(H, 0.25)
    if tuple(idx.shape) != (Hkv, n_keep) or not bool((idx[:, 1:] > idx[:, :-1]).all()):
        raise RuntimeError("selection must return n_keep sorted distinct positions per head")
    must = torch.cat([torch.arange(tc.N_SINK), torch.arange(H - tc.WINDOW, H)])
    if not all(bool(torch.isin(must, idx[h]).all()) for h in range(Hkv)):
        raise RuntimeError("sinks and the window must always be kept")
    rs, re = tc.N_SINK, H - tc.WINDOW
    s_cov = coverage_scores(Q, K, Vx, dd ** -0.5, H, rs, re, m.out_metric(0, K.device), tc)
    s_win = coverage_scores(Q, K, Vx, dd ** -0.5, H, rs, re, m.out_metric(0, K.device), tc, window_only=True)
    r_cov = float((s_cov[:, j_star - rs:j_star - rs + 1] >= s_cov).float().mean())
    r_win = float((s_win[:, j_star - rs:j_star - rs + 1] >= s_win).float().mean())
    out["early_read_token_rank_coverage"], out["early_read_token_rank_window"] = r_cov, r_win
    if not r_cov >= 0.9 or not r_cov > r_win:
        raise RuntimeError(f"coverage must rank the early-read token at the top ({r_cov:.3f} vs window {r_win:.3f})")
    if not bool(torch.isin(torch.tensor(j_star), idx[0])):
        raise RuntimeError("COVE must keep the token the early segment reads")
    full = CoveMethod(tc, 1.0, "all", fr).select(0, Q, K, Vx, dd ** -0.5, H)
    if not torch.equal(full, torch.arange(H).expand(Hkv, H)):
        raise RuntimeError("a budget of the whole context must keep every token")
    logger.info("cove tests: " + ", ".join(f"{k_}={v_:.4g}" for k_, v_ in out.items()))
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
    (2) identities: chunk-local eviction that keeps every token at 16 bit, and COVE with a budget
        of the whole context, reproduce the full cache.
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
    qm = CoveMethod(c, 1.0, "cove_all", runner)
    qm.freeze_at = T - 4
    out = {}
    for name, ref, other in (("sdpa_vs_fp32_sim", base, full),
                             ("evict_keepall_vs_full", full, runner.forward(x, cm).logits.float()[0]),
                             ("cove_keepall_vs_full", full, runner.forward(x, qm).logits.float()[0])):
        out[f"{name}_max_abs"] = float((ref - other).abs().max())
        out[f"{name}_argmax"] = float((ref.argmax(-1) == other.argmax(-1)).float().mean())
    logger.info("  harness tests: " + ", ".join(f"{k_}={v_:.3g}" for k_, v_ in out.items()))
    exact = COMPUTE_DTYPE == torch.float32
    for name in ("sdpa_vs_fp32_sim", "evict_keepall_vs_full", "cove_keepall_vs_full"):
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
        sep = "<<<COVE_SEPARATOR_7f3a>>>"
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
    family: str                        # full | kivi | cla | snapkv | streamingllm | mtlp | chunk | cove | cove_ref | native | kvpress
    keep: Optional[float] = None       # memory budget: fraction of the fp16 context cache
    kivi_bits: Optional[int] = None    # KIVI / MTLP code width
    chunk: Optional[ChunkMethod] = None
    press: Optional[str] = None        # kvpress press name (official implementation)
    variant: Optional[str] = None      # COVE variant (COVE_VARIANTS)

    @property
    def label(self) -> str:
        """Method family as reported in result rows."""
        if self.family == "chunk":
            return self.chunk.family
        return self.family if self.family in ("cove_ref", "mtlp") else (self.variant or self.family)


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
        logging.getLogger("kvpress").setLevel(logging.ERROR)     # KVzip warns of its 2-3x prefill on every call
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
               "streaming_llm": lambda: K.StreamingLLMPress(),
               "kvzip": lambda: K.KVzipPress(), "compactor": lambda: K.CompactorPress()}
    press = presses[name]()
    press.compression_ratio = 1.0 - keep       # as evaluate.py: set after construction (AdaKV delegates)
    return press


def kvpress_n_kept(k_len: int, keep: float) -> int:
    """kvpress' compute_n_kept(k_len, compression_ratio = 1 - keep), same float arithmetic."""
    ratio = 1.0 - keep
    return 0 if ratio >= 1.0 else max(1, int(k_len * (1 - ratio)))


MASKING_PRESSES = ("adakv_snapkv", "expected_attention", "kvzip")   # they mask keys: memory is nominal
ALWAYS_RUN = ("full", "native_full")                        # references of every contrast / path control


COVE_VARIANTS = {"cove": {},
                 "cove_window": {"window_only": True},   # end-window scores (SnapKV-type prior)
                 "cove_l2": {"score": "l2"},              # independent-error (squared) price
                 "cove_pool5": {"pool": 5}}               # SnapKV-style smoothing of the scores


def bench_specs(cfg: "Config", for_ppl: bool = False) -> List[BenchSpec]:
    """Every method at every budget (COVE's ablations and its harness twin at PRIMARY_KEEP)."""
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
        out.append(BenchSpec(f"cove@{t}", "cove", r, variant="cove"))
    r0 = f"{cfg.PRIMARY_KEEP:g}"
    out += [BenchSpec(f"{v_}@{r0}", "cove", cfg.PRIMARY_KEEP, variant=v_)
            for v_ in ("cove_window", "cove_l2", "cove_pool5")]
    out.append(BenchSpec(f"cove_ref@{r0}", "cove_ref", cfg.PRIMARY_KEEP, variant="cove"))   # harness twin
    if cfg.BENCH_INCLUDE_CLA:
        out.append(BenchSpec("cla", "cla"))
    if cfg.BENCH_METHODS:                                  # e.g. ("cove@", "snapkv@")
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
    if spec.family in ("cove", "cove_ref"):
        note = "" if P > cfg.N_SINK + cfg.WINDOW else "context <= sinks + window"
        return CoveMethod(cfg, spec.keep, spec.name, runner, **COVE_VARIANTS[spec.variant]), note
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
    bias); SnapKV + KIVI-b: as KIVI.  (COVE keeps exact rows only.)
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
        info["attn_flops_first_step"] = attn_flops_first_step(method, runner.Hq, runner.d)
        info["attn_flops_first_step_full"] = 4.0 * runner.d * (P + 1) * runner.Hq * runner.L
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
    full_bytes = context_len * runner.L * runner.Hkv * 2 * runner.d * next(model.parameters()).element_size()
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
    if spec.family == "cove":
        gen, info = native_cove_generate(runner, method, enc["ids"], enc["context_len"], max_new, stop, stop_fn)
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
        if spec.family in ("native", "kvpress", "cove"):
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
        if spec.family == "cove":
            m, _ = make_method(spec, context_len, runner, cfg)
            gen, inf = native_cove_generate(runner, m, ids, context_len, n_new, set())
            twin = f"cove_ref@{spec.keep:g}"
            row = {"method": spec.name, "n_tokens": len(gen), "note": "deployment path (HF SDPA, compressed cache)",
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
# A one-shot method (full cache, COVE, SnapKV, SnapKV + KIVI-b, StreamingLLM) compresses only at the horizon
# H: every query before H reads the exact cache.  Its keys, values and queries over the context
# are therefore the SAME for every such method, so the context is prefilled once per sample
# (fp32 attention, exactly as each method's own single pass computes it), and each method only
# compresses the captured cache and runs the question and the answer from its own state.
# generation_equivalence() checks this path against each method's single pass
# (shared_path_agreement).  Methods that compress while prefilling (KIVI, chunk-local evict /
# AM, CLA) keep the single pass.

SHARED_FAMILIES = ("full", "cove", "cove_ref", "snapkv", "streamingllm", "mtlp")


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
    """Every context position whose query a one-shot method reads: COVE's coverage groups and the
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
    out = {"nll": float(nll.sum()), "n_tok": int(nll.numel()), "memory_fraction": mem}
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
                try:                          # a transient OOM (shared GPU) never ends the run
                    gen, info, note = run_spec(runner, spec, enc, max_new, stop, stop_fn, cfg, pre)
                except torch.cuda.OutOfMemoryError:
                    free_memory()
                    try:
                        gen, info, note = run_spec(runner, spec, enc, max_new, stop, stop_fn, cfg, pre)
                    except torch.cuda.OutOfMemoryError:
                        free_memory()
                        logger.warning(f"    {spec.name}: out of memory twice on item {it['id']}; "
                                       f"left for a resumed run")
                        continue
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
                            "context_len": H, "path": "harness", "harness_seconds": time.time() - t0}
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
        out += [(f"cove_vs_kvpress:{pn}@{t}", f"cove@{t}", f"kvpress:{pn}@{t}") for pn in cfg.KVPRESS_PRESSES]
        out += [(f"cove_vs_{b_}@{t}", f"cove@{t}", f"{b_}@{t}")
                for b_ in ["snapkv", "streamingllm", "evict", "am"] + [f"snapkv_q{b}" for b in cfg.MTLP_BITS]]
        out.append((f"{HARNESS_CHECK}snapkv@{t}", f"snapkv@{t}", f"kvpress:snapkv@{t}"))
    t0 = f"{cfg.PRIMARY_KEEP:g}"
    out += [(f"cove_vs_{b_}@{t0}", f"cove@{t0}", f"{b_}@{t0}") for b_ in ("cove_window", "cove_l2", "cove_pool5")]
    out += [(f"cove@{t0}_vs_kivi{b}", f"cove@{t0}", f"kivi{b}") for b in cfg.KIVI_BITS]
    out.append((f"{HARNESS_CHECK}cove@{t0}", f"cove@{t0}", f"cove_ref@{t0}"))
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
                ("full", "kivi2", f"snapkv@{t0}", f"streamingllm@{t0}", f"evict@{t0}", f"am@{t0}", f"snapkv_q4@{t0}",
                 f"cove_ref@{t0}", f"cove@{t0}", "native_full", f"kvpress:snapkv@{t0}", f"kvpress:streaming_llm@{t0}")]
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


# ── Interim report from a running (or finished) benchmark: paired statistics, not single lines ──
ITEM_LINE = re.compile(r" - INFO -\s+(\S+)/(\S+) item (\S+): P=\d+ (.*)$")


def read_progress(paths: Sequence[str]) -> pd.DataFrame:
    """(benchmark, task, item, method, score) rows from bench_samples.csv files and / or run logs
    (the per-item log lines); the last record of every (benchmark, task, item, method) wins."""
    parts = []
    for p in paths:
        if p.endswith(".csv"):
            df = pd.read_csv(p, dtype={"benchmark": str, "task": str, "item": str, "method": str})
            parts.append(df[["benchmark", "task", "item", "method", "score"]])
            continue
        recs = []
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = ITEM_LINE.search(line)
                if not m:
                    continue
                bench, task, item, rest = m.groups()
                for kv in rest.split(", "):
                    name, _, val = kv.rpartition("=")
                    try:
                        recs.append((bench, task, item, name, float(val)))
                    except ValueError:
                        continue
        parts.append(pd.DataFrame(recs, columns=["benchmark", "task", "item", "method", "score"]))
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return df.drop_duplicates(["benchmark", "task", "item", "method"], keep="last") if len(df) else df


def progress_report(paths: Sequence[str], cfg: "Config", out_dir: str) -> pd.DataFrame:
    """
    What a partial run supports: per task, the mean score and number of items of every method, the
    share of items the FULL cache already fails (score 0: those items cannot discriminate between
    compressors), and every pre-registered contrast as a paired test over all items both methods
    have (macro over tasks; bootstrap CI; sign-flip p; Holm per benchmark).  Individual log lines
    are single draws of a noisy per-item metric (one F1 per item) and must never be read as results.
    """
    df = read_progress(paths)
    if df.empty:
        raise SystemExit(f"no per-item results found in {list(paths)}")
    df = df[df.score.notna()]
    table = df.pivot_table(index="method", columns=["benchmark", "task"], values="score", aggfunc="mean")
    counts = df.groupby(["benchmark", "task"]).item.nunique()
    full0 = df[df.method == "full"].groupby(["benchmark", "task"]).score.apply(lambda s_: float((s_ == 0).mean()))
    rows = []
    for bench, g in df.groupby("benchmark"):
        g = g.assign(unit=g.task + "/" + g["item"])
        S = g.pivot_table(index="unit", columns="method", values="score")
        task_of = S.index.str.split("/").str[0].to_numpy()
        for name, A, B in bench_contrasts(cfg):
            if A not in S or B not in S:
                continue
            ok = S[A].notna() & S[B].notna()
            if not ok.any():
                continue
            d = (S[A] - S[B])[ok].to_numpy()
            r = paired_score_test(d, task_of[ok.to_numpy()], cfg.N_BOOT, cfg.N_PERM, SEED)
            rows.append({"benchmark": bench, "contrast": name, "A": A, "B": B,
                         "score_A": float(S[A][ok].mean()), "score_B": float(S[B][ok].mean()), **r})
    ct = pd.DataFrame(rows)
    if len(ct):
        ct["p_holm"] = np.nan
        hyp = ~ct.contrast.str.startswith(HARNESS_CHECK)
        for _, idx in ct[hyp].groupby("benchmark").groups.items():
            ct.loc[idx, "p_holm"] = holm(ct.loc[idx, "p_value"])
    os.makedirs(out_dir, exist_ok=True)
    table.round(2).to_csv(os.path.join(out_dir, "report_scores.csv"))
    ct.to_csv(os.path.join(out_dir, "report_contrasts.csv"), index=False)
    for key, n in counts.items():
        logger.info(f"REPORT {key[0]}/{key[1]}: {n} items; full cache scores 0 on {full0.get(key, float('nan')):.0%}")
    macro = table.T.groupby(level=0).mean().T                  # macro over the tasks of each benchmark
    for bench in macro.columns:
        col = macro[bench].dropna().sort_values(ascending=False)
        logger.info(f"REPORT {bench} macro mean over tasks:\n" + "\n".join(f"    {m_:<34} {v_:6.2f}" for m_, v_ in col.items()))
    for r in ct.itertuples():
        logger.info(f"REPORT {r.benchmark} {r.contrast:<38} {r.estimate:+6.2f} [{r.ci_low:+.2f}, {r.ci_high:+.2f}] "
                    f"p={r.p_value:.4f} holm={r.p_holm:.4f} n={r.n_units}")
    logger.info(f"REPORT written to {out_dir}")
    return ct


def main():
    ap = argparse.ArgumentParser(description="COVE: coverage-priced eviction of the KV cache")
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
                    help="benchmark method name prefixes, e.g. cove evict snapkv@ (full and native_full always run)")
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
    ap.add_argument("--report", nargs="+", default=None, metavar="PATH",
                    help="interim paired report from bench_samples.csv files and / or run logs, then exit")
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
        kw.update(OUTPUT_DIR=a.output or str(PROJECT_ROOT / "cove_smoke"), SMOKE=True, CHUNK=32, WINDOW=8,
                  W_OBS=8, KIVI_GROUP=8, N_BOOT=500, N_PERM=2000, MIN_MODELS_WILCOXON=2, RESUME=False,
                  AM_ITERS=100, KEEP_FRACTIONS=(0.5, 0.25), PRIMARY_KEEP=0.25, LB_MAX_PROMPT_TOKENS=1500,
                  LBV2_MAX_PROMPT_TOKENS=1500, RULER_LENGTHS=(1024,), LONG_PPL_DATASETS=("synthetic",),
                  LONG_PPL_LEN=600, LONG_PPL_EVAL=120, COVE_SEGMENT=64, COVE_QUERIES=8, COVE_LOCAL=16,
                  MICRO_LENGTHS=(512, 2048), MICRO_REPEATS=5, ALLOW_MISSING_KVPRESS=True)
    if a.microbench is not None and a.microbench:
        kw["MICRO_LENGTHS"] = tuple(a.microbench)
    cfg = Config(**kw)
    if a.report:
        progress_report(a.report, cfg, a.output or os.path.join(cfg.RESULTS_DIR, "report"))
        return
    if a.check_data:
        problems = preflight_data(cfg, a.suite)
        raise SystemExit(("data check FAILED:\n  " + "\n  ".join(problems)) if problems else 0)
    fh = logging.FileHandler(os.path.join(cfg.RESULTS_DIR, "cove_kv.log"))
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(fh)
    logger.info(f"run {cfg.RUN_ID}: results in {cfg.RESULTS_DIR}")
    with open(os.path.join(cfg.RESULTS_DIR, "config.json"), "w") as f:
        json.dump({"source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   "torch": torch.__version__, "transformers": transformers.__version__,
                   **{k: ([asdict(m) for m in v] if k == "BENCH_MODELS" else v)
                      for k, v in cfg.__dict__.items()}}, f, indent=2, default=str)
    with open(os.path.join(cfg.RESULTS_DIR, "cove_tests.json"), "w") as f:
        json.dump(cove_tests(cfg), f, indent=2)
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
