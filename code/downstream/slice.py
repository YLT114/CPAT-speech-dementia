# -*- coding: utf-8 -*-
"""
slice_mmse_weight1_loss.py
conda activate slice_mmse
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v2/slice.py

将新版官方 SLiCE 作为新的下游时序模型，接入现有
Whisper / M-P-TAPT -> MMSE 回归框架。

正式模型：
    Whisper feature [B,T,C]
        -> Linear(C,128)
        -> BD-SLiCELayer x 2
        -> Mean Pooling
        -> LayerNorm
        -> Linear(128,128)
        -> GELU
        -> Linear(128,1)
        -> segment-level MMSE
        -> audio-level aggregation

关键原则：
1. 使用官方 torch-slices 的 SLiCELayer，不自行重写 SLiCE recurrence。
2. SLICE_BLOCK_SIZE=4 且 SLICE_DIAGONAL_DENSE=False，因此为 BD-SLiCE。
3. SLICE_USE_PARALLEL=True，请求官方 parallel-in-time associative scan。
4. official_train 全部训练；official_test 仅 monitoring/reporting，不选 epoch/checkpoint。
5. early stopping 仅监控 official_train train_loss；不恢复历史最优模型；
   正常停止时使用最后一个完整 epoch；若出现数值异常，则恢复上一完整有限 epoch。
6. 复用原脚本的 train-only target 标准化、PROCESS-2 mixed loss、
   audio-level 聚合与多 seed 统计思想。

"""

# =============================================================================
# 0. 标准库 + GPU（必须在 import torch 前设置）
# =============================================================================
import os
import sys
import re
import json
import copy
import random
import warnings
import importlib.metadata
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

# =============================================================================
# 1. ========================== 参数设置区 ====================================
# =============================================================================
CONFIG_ROOT_DIR = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v2"
CONFIG_FEATURE_ROOT = os.path.join(CONFIG_ROOT_DIR, "feature_tensor")
CONFIG_OUTPUT_ROOT = os.path.join(CONFIG_ROOT_DIR, "outputs_repeat")

# ---------- GPU ----------
GPU_ID = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# ---------- 数据集 ----------
# "ADReSS-IS2020", "ADReSSo21", "Pitt", "PROCESS-2"
DATASET_OPTIONS = [ "ADReSS-IS2020" ]

# ---------- 上游特征 ----------
FEAT_MODEL_NAME = "whisper_small"   # hubert_base    wav2vec2_base  wavlm_base_plus   whisper_small
FT_MODEL_NAME = "ft_ptapt_lr5e-5_epo5"
FEATURE_SOURCE_MODES = ["freeze"]       # ["freeze"] / ["finetune"] / 两者
ENABLE_RECURSIVE_BATCH = True

SINGLE_LAYERS = [12]
UNFREE_LAYER_NUM = [1]

# 单跑兜底（ENABLE_RECURSIVE_BATCH=False 时使用）
FEATURE_MODEL_TAG = ""
FEATURE_SELECTION_TAG = ""
FEATURE_DATA_DIR_OVERRIDE = None
FEATURE_FILE_TAG = "whisper_single12_0.3_30s"

# ---------- 下游 ----------
MODEL_NAME = "slice"

# ---------- seed ----------
RANDOM_SEED = 400
RUN_SINGLE_SEED_400 = False
RUN_MULTI_SEEDS = True
MULTI_SEEDS = [401, 402, 403, 404, 405]

# ---------- 统一训练协议 ----------
BATCH_SIZE = 16

# 最大训练轮数为 10。
# 注意：这里不是“固定训练满 10 轮”。
# USE_EARLY_STOPPING=True 时，若 train_loss 连续 EARLY_STOPPING_STEPS
# 个 epoch 没有达到 EARLY_STOPPING_MIN_DELTA 的改进，则会提前停止。
TARGET_EPOCHS = 10

# 与最终版 LRU 方案对齐：开启 train-loss early stopping。
USE_EARLY_STOPPING = True
EARLY_STOPPING_STEPS = 3
EARLY_STOPPING_MONITOR = "train_loss"
EARLY_STOPPING_MIN_DELTA = 1e-4

# 不恢复历史最优 checkpoint。
# 正常停止使用最后完整 epoch；数值异常时回退到上一完整有限 epoch。
EARLY_STOPPING_RESTORE_BEST = False

EPOCH_SELECTION_POLICY = "last_epoch_or_train_loss_early_stop"
TEST_MONITOR_ONLY = True

WARMUP_EPOCHS = 2
COSINE_ALPHA = 0.05

# ---------- SLiCE 优化器 ----------
# 使用 Adam、无 weight decay，保留梯度裁剪；早停依据训练损失。
LR_SLICE = 1e-4

SLICE_OPTIMIZER_NAME = "adam"

# 显式记录为 0，仅用于配置日志；
# torch.optim.Adam 中不会传入非零 weight_decay。
SLICE_WEIGHT_DECAY = 0.0

# 保留 SLiCE 的梯度稳定性保护。
SLICE_GRAD_CLIP_NORM = 1.0

# 数值稳定性保护：
# 1) forward / loss / gradient / parameter 出现 NaN/Inf 时立即定位；
# 2) 不再让 NaN 模型继续跑到 r2_score 才报错；
# 3) 数值失败时恢复“上一完整有限 epoch”的模型。
SLICE_NUMERICAL_GUARD = True

SLICE_BATCH_SIZE = BATCH_SIZE
SLICE_TARGET_EPOCHS = TARGET_EPOCHS
SLICE_WARMUP_EPOCHS = WARMUP_EPOCHS
SLICE_COSINE_ALPHA = COSINE_ALPHA

# ---------- SLiCE 网络 ----------
SLICE_HIDDEN_DIM = 128
SLICE_NUM_LAYERS = 2

# block_size > 1 且 diagonal_dense=False => BD-SLiCE
SLICE_BLOCK_SIZE = 4
SLICE_DIAGONAL_DENSE = False

SLICE_BIAS = True
SLICE_INIT_STD = 0.01
# SLiCE 内部驱动缩放。
# 当前缩放为 0.1，仅作用于 SLiCE 内部；共享上游特征文件保持原样。
SLICE_SCALE = 0.1
SLICE_INPUT_DEPENDENT_INIT = False

# 正式实验仍开启 parallel-in-time。
SLICE_USE_PARALLEL = True

# 并行扫描的 chunk 长度。
SLICE_CHUNK_SIZE = 128

SLICE_PATH_MODE = "values"
SLICE_TRANSITION_MODE = "euler"

SLICE_NORM_TYPE = "rmsnorm"
SLICE_PRENORM = True
SLICE_SECOND_NORM = True

SLICE_FF_STYLE = "mlp"
SLICE_FF_ACTIVATION = "gelu"

# 第一版设 1，使 FFN hidden_dim=128，和当前小数据集设置更接近
SLICE_FF_MULT = 1

SLICE_DROPOUT_RATE = 0.0
SLICE_DROPOUT_POSITION = "residual"
SLICE_NORM_EPS = 1e-6

# ---------- MMSE readout ----------
# 可选："mean" / "last" / "meanmax"
SLICE_READOUT_POOL = "mean"
SLICE_MLP_HIDDEN_DIM = 128

# ---------- 预处理 ----------
FEATURE_STANDARDIZE = False
INCLUDE_TIME = False
T_MAX = 1.0

TARGET_STANDARDIZE = True
TARGET_STANDARDIZE_EPS = 1e-6

# ---------- audio 聚合 ----------
AUDIO_AGG_PRIMARY_METHOD = "mean"
AUDIO_AGG_TRIM_RATIO = 0.10

# ---------- PROCESS-2 weighting ----------
# 与当前最终实验协议对齐：
# 仅在 dataset == "PROCESS-2" 时真正启用；
# 其他数据集仍然自动使用普通 MSE。
USE_PROCESS2_MMSE_BIN_WEIGHT = True
PROCESS2_MMSE_BIN_EDGES = [0.0, 21.0, 26.0, 31.0]
PROCESS2_MMSE_BIN_LABELS = ["[0,21)", "[21,26)", "[26,31)"]
PROCESS2_MMSE_BIN_WEIGHT_BETA = 0.99
PROCESS2_MMSE_BIN_WEIGHT_MAX = 2.0
PROCESS2_WEIGHT_EPS = 1e-8
PROCESS2_MIXED_LOSS_ALPHA = 0.3

# ---------- 调试 ----------
# 正式多 seed 前可打开，比较 recurrent / parallel 同参数输出。
SLICE_RUN_PARALLEL_RECURRENT_CHECK = False
SLICE_CHECK_BATCH_SIZE = 2

# 只做少量 forward/backward 后退出。
SLICE_SMOKE_TEST_ONLY = False
SLICE_SMOKE_TEST_NUM_BATCHES = 2

FILE_TAG_MAP = {
    "ADReSS-IS2020": "whisper_single12_0.3_30s",
    "ADReSSo21": "whisper_single12_0.3_30s",
    "Pitt": "whisper_single12_0.3_30s",
    "PROCESS-2": "whisper_single12_0.3_30s",
}

# =============================================================================
# 2. 第三方库
# =============================================================================
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score

try:
    from slices import SLiCELayer
except Exception as exc:
    raise ImportError(
        "无法导入 torch-slices。请先：pip install torch-slices\n"
        f"原始错误：{exc}"
    ) from exc

# =============================================================================
# 3. 数据结构
# =============================================================================
@dataclass
class RunContext:
    source_mode: str
    dataset: str
    feature_model_tag: str
    feature_selection_tag: str
    feature_file_tag: str
    feature_data_dir: str
    output_dir: str

@dataclass
class FeatureBundle:
    X_train: np.ndarray
    X_val: np.ndarray
    y_train_norm: np.ndarray
    y_val_norm: np.ndarray
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    idx_train_audio: np.ndarray
    idx_val_audio: np.ndarray
    train_segment_weights: np.ndarray
    target_normalizer: Dict[str, Any]
    process2_weight_info: Dict[str, Any]
    data_dim: int

# =============================================================================
# 4. 通用工具
# =============================================================================
def _is_blank(value) -> bool:
    return value is None or str(value).strip() in {"", "None", "none", "null", "Null"}

def _json_safe(obj):
    if obj is None or isinstance(obj, (str, bool, int)):
        return obj
    if isinstance(obj, float):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    return str(obj)

def save_json(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_json_safe(obj), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def seed_everything(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def count_trainable_parameters(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def _safe_torch_load(path):
    # 本地上游特征属于可信文件；显式 weights_only=False 兼容 indices 等对象。
    return torch.load(path, map_location="cpu", weights_only=False)

def _to_numpy(x, dtype=None):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x

# =============================================================================
# 5. selection / 目录发现
# =============================================================================
def _extract_single_layer(selection):
    if _is_blank(selection):
        return None
    first = str(selection).strip().replace("\\", "/").split("/")[0]
    m = re.match(r"^single(\d+)(?:_|$)", first, flags=re.IGNORECASE)
    return None if m is None else int(m.group(1))

def _extract_unfree(selection):
    if _is_blank(selection):
        return None
    first = str(selection).strip().replace("\\", "/").split("/")[0]
    m = re.search(r"_unfree(\d+)(?:_|$)", first, flags=re.IGNORECASE)
    return None if m is None else int(m.group(1))

def _matches_layers(selection):
    allowed = {int(x) for x in SINGLE_LAYERS if int(x) > 0}
    return True if not allowed else _extract_single_layer(selection) in allowed

def _matches_unfree(selection):
    allowed = {int(x) for x in UNFREE_LAYER_NUM if int(x) > 0}
    return True if not allowed else _extract_unfree(selection) in allowed

def _clean_selection(selection):
    if _is_blank(selection):
        return "unknown_selection"
    return str(selection).strip().replace("\\", "/").split("/")[0]

def _parse_file_tag(train_path):
    m = re.match(r"feature_(.+)_train\.pt$", Path(train_path).name)
    return None if m is None else m.group(1)

def _has_bundle(feature_dir, file_tag):
    names = [
        f"feature_{file_tag}_train.pt",
        f"feature_{file_tag}_test.pt",
        f"labels1_{file_tag}_train.pt",
        f"labels2_{file_tag}_test.pt",
        f"indices_{file_tag}_train.pt",
        f"indices_{file_tag}_test.pt",
    ]
    return all(os.path.exists(os.path.join(feature_dir, n)) for n in names)

def build_output_dir(dataset, feature_model, selection):
    return os.path.join(
        CONFIG_OUTPUT_ROOT,
        str(dataset).strip(),
        str(feature_model).strip(),
        MODEL_NAME,
        _clean_selection(selection),
    )

def _discover_root(dataset, root, source_mode, require_unfree):
    contexts = []
    if not os.path.isdir(root):
        print(f"[DISCOVER] {source_mode} root 不存在，跳过: {root}")
        return contexts

    for train_path in sorted(Path(root).rglob("feature_*_train.pt")):
        feature_dir_path = train_path.parent

        # 不读取旧 fold 子目录
        if re.fullmatch(r"fold\d+", feature_dir_path.name, flags=re.IGNORECASE):
            continue

        file_tag = _parse_file_tag(str(train_path))
        if _is_blank(file_tag):
            continue

        feature_dir = str(feature_dir_path)
        if not _has_bundle(feature_dir, file_tag):
            continue

        rel = os.path.relpath(feature_dir, root).replace("\\", "/")
        if rel in {".", ""}:
            rel = feature_dir_path.name

        if not _matches_layers(rel):
            continue
        if require_unfree and not _matches_unfree(rel):
            continue

        selection = _clean_selection(rel)
        feature_model = FEAT_MODEL_NAME if source_mode == "freeze" else FT_MODEL_NAME

        contexts.append(
            RunContext(
                source_mode=source_mode,
                dataset=dataset,
                feature_model_tag=feature_model,
                feature_selection_tag=selection,
                feature_file_tag=str(file_tag),
                feature_data_dir=feature_dir,
                output_dir=build_output_dir(dataset, feature_model, selection),
            )
        )
    return contexts

def build_all_contexts():
    all_contexts = []
    modes = {str(x).strip().lower() for x in FEATURE_SOURCE_MODES}

    for dataset in DATASET_OPTIONS:
        dataset = str(dataset).strip()

        if ENABLE_RECURSIVE_BATCH:
            contexts = []
            if "freeze" in modes:
                contexts += _discover_root(
                    dataset,
                    os.path.join(CONFIG_FEATURE_ROOT, dataset, FEAT_MODEL_NAME),
                    "freeze",
                    require_unfree=False,
                )
            if "finetune" in modes:
                contexts += _discover_root(
                    dataset,
                    os.path.join(CONFIG_FEATURE_ROOT, dataset, FT_MODEL_NAME),
                    "finetune",
                    require_unfree=True,
                )
        else:
            # 单跑：优先 override，否则精确拼路径
            if not _is_blank(FEATURE_DATA_DIR_OVERRIDE):
                feature_dir = str(FEATURE_DATA_DIR_OVERRIDE).strip()
                feature_model = (
                    FEATURE_MODEL_TAG
                    if not _is_blank(FEATURE_MODEL_TAG)
                    else Path(feature_dir).parent.name
                )
                selection = _clean_selection(Path(feature_dir).name)
            else:
                if _is_blank(FEATURE_MODEL_TAG) or _is_blank(FEATURE_SELECTION_TAG):
                    raise ValueError(
                        "单跑模式必须填写 FEATURE_MODEL_TAG / FEATURE_SELECTION_TAG，"
                        "或直接给 FEATURE_DATA_DIR_OVERRIDE。"
                    )
                feature_model = FEATURE_MODEL_TAG
                selection = _clean_selection(FEATURE_SELECTION_TAG)
                feature_dir = os.path.join(
                    CONFIG_FEATURE_ROOT, dataset, feature_model, selection
                )

            if not os.path.isdir(feature_dir):
                raise FileNotFoundError(f"单跑特征目录不存在: {feature_dir}")

            file_tag = None if _is_blank(FEATURE_FILE_TAG) else str(FEATURE_FILE_TAG)
            if file_tag is None or not _has_bundle(feature_dir, file_tag):
                file_tag = None
                for p in sorted(Path(feature_dir).glob("feature_*_train.pt")):
                    candidate = _parse_file_tag(str(p))
                    if candidate and _has_bundle(feature_dir, candidate):
                        file_tag = candidate
                        break
            if file_tag is None:
                raise FileNotFoundError(f"目录中未找到完整特征 bundle: {feature_dir}")

            contexts = [
                RunContext(
                    source_mode="manual",
                    dataset=dataset,
                    feature_model_tag=str(feature_model),
                    feature_selection_tag=selection,
                    feature_file_tag=str(file_tag),
                    feature_data_dir=feature_dir,
                    output_dir=build_output_dir(dataset, feature_model, selection),
                )
            ]

        # 去重
        seen = set()
        for ctx in contexts:
            key = (
                ctx.source_mode, ctx.dataset, ctx.feature_model_tag,
                ctx.feature_selection_tag, ctx.feature_file_tag, ctx.feature_data_dir
            )
            if key in seen:
                continue
            seen.add(key)
            all_contexts.append(ctx)

        print(f"[DISCOVER] dataset={dataset}, contexts={len(contexts)}")
        for ctx in contexts:
            print(
                f"  source={ctx.source_mode} | feature={ctx.feature_model_tag} | "
                f"selection={ctx.feature_selection_tag} | file_tag={ctx.feature_file_tag}"
            )

    if not all_contexts:
        raise RuntimeError("没有发现任何可运行的 SLiCE 上游特征。")
    return all_contexts

# =============================================================================
# 6. 预处理 / 标签标准化
# =============================================================================
def preprocess_data(features):
    """严格沿用原脚本：每个样本沿时间轴 axis=1 标准化。"""
    features = np.asarray(features, dtype=np.float32)
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return ((features - mean) / (std + 1e-8)).astype(np.float32)

def ensure_targets(y):
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 0:
        y = y.reshape(1)
    if y.ndim > 1:
        if y.shape[-1] == 1:
            y = y.reshape(-1)
        else:
            raise ValueError(f"回归标签应为一维，当前 shape={y.shape}")
    return y.astype(np.float32)

def fit_target_normalizer_train_only(y):
    y = ensure_targets(y)
    return {
        "enabled": bool(TARGET_STANDARDIZE),
        "mean": float(np.mean(y)),
        "std": max(float(np.std(y)), float(TARGET_STANDARDIZE_EPS)),
        "eps": float(TARGET_STANDARDIZE_EPS),
    }

def normalize_targets(y, norm):
    y = ensure_targets(y)
    if not norm.get("enabled", False):
        return y.astype(np.float32)
    return ((y - norm["mean"]) / norm["std"]).astype(np.float32)

def denormalize_targets(y, norm):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if not norm.get("enabled", False):
        return y.astype(np.float32)
    return (y * norm["std"] + norm["mean"]).astype(np.float32)

def _validate_numpy_finite(name, arr):
    """
    在进入模型前检查数据中是否已经存在 NaN / Inf。
    不能用 nan_to_num 静默修复，因为那会改变正式实验数据。
    """
    arr = np.asarray(arr)

    finite_mask = np.isfinite(arr)
    if not bool(finite_mask.all()):
        bad_count = int((~finite_mask).sum())
        first_bad = np.argwhere(~finite_mask)[0].tolist()
        raise ValueError(
            f"[NONFINITE DATA] {name} 中存在 NaN/Inf | "
            f"bad_count={bad_count} | first_bad_index={first_bad} | "
            f"shape={arr.shape}"
        )


def validate_feature_shapes(X_train, X_val):
    for name, X in [
        ("train", X_train),
        ("official_test_monitor", X_val),
    ]:
        X = np.asarray(X)

        if X.ndim != 3:
            raise ValueError(
                f"[SLiCE] {name} 必须为 [N,T,C]，当前={X.shape}"
            )

        _validate_numpy_finite(
            f"X_{name}",
            X,
        )

        print(
            f"[SLiCE][{name}] "
            f"N={X.shape[0]}, T={X.shape[1]}, C={X.shape[2]} | "
            f"min={float(X.min()):.6f}, max={float(X.max()):.6f}, "
            f"mean={float(X.mean()):.6f}, std={float(X.std()):.6f}"
        )

    if X_train.shape[-1] != X_val.shape[-1]:
        raise ValueError(
            "train / official_test_monitor 特征维度不一致。"
        )

# =============================================================================
# 7. PROCESS-2 weighting
# =============================================================================
def _is_process2(dataset):
    return str(dataset).strip().lower() == "process-2"

def _process2_bin(v):
    v = float(v)
    if 0 <= v < 21:
        return "[0,21)"
    if 21 <= v < 26:
        return "[21,26)"
    if 26 <= v < 31:
        return "[26,31)"
    return "out_of_range"

def _effective_bin_weights(counts):
    raw = {}
    beta = float(PROCESS2_MMSE_BIN_WEIGHT_BETA)
    for k, n in counts.items():
        if int(n) > 0:
            eff = 1.0 - beta ** int(n)
            raw[k] = (1.0 - beta) / max(eff, PROCESS2_WEIGHT_EPS)

    total = sum(int(n) for n in counts.values() if int(n) > 0)
    if total <= 0 or not raw:
        return {k: 1.0 for k in counts}

    mean_w = sum(counts[k] * raw[k] for k in raw) / max(total, PROCESS2_WEIGHT_EPS)
    return {
        k: (
            min(raw[k] / max(mean_w, PROCESS2_WEIGHT_EPS), PROCESS2_MMSE_BIN_WEIGHT_MAX)
            if k in raw else 0.0
        )
        for k in counts
    }

def build_process2_segment_weights(y_raw, audio_ids):
    y_raw = ensure_targets(y_raw)
    audio_ids = np.asarray(audio_ids, dtype=np.int64).reshape(-1)

    audio_mmse, audio_nseg = {}, {}
    for aid in sorted(set(audio_ids.tolist())):
        mask = audio_ids == aid
        audio_mmse[aid] = float(y_raw[mask][0])
        audio_nseg[aid] = int(mask.sum())

    counts = {label: 0 for label in PROCESS2_MMSE_BIN_LABELS}
    audio_bin = {}
    for aid, mmse in audio_mmse.items():
        b = _process2_bin(mmse)
        audio_bin[aid] = b
        if b in counts:
            counts[b] += 1

    bin_weights = _effective_bin_weights(counts)
    weights = np.zeros(len(y_raw), dtype=np.float32)

    for i, aid in enumerate(audio_ids):
        b = audio_bin[int(aid)]
        aw = float(bin_weights.get(b, 1.0))
        weights[i] = aw / max(audio_nseg[int(aid)], 1)

    weights = weights / max(float(weights.mean()), PROCESS2_WEIGHT_EPS)

    info = {
        "enabled": True,
        "bin_counts_audio": counts,
        "bin_weights_audio": bin_weights,
        "beta": PROCESS2_MMSE_BIN_WEIGHT_BETA,
        "max_weight": PROCESS2_MMSE_BIN_WEIGHT_MAX,
        "segment_weight_min": float(weights.min()),
        "segment_weight_max": float(weights.max()),
        "segment_weight_mean": float(weights.mean()),
        "segment_weight_std": float(weights.std()),
    }
    return weights.astype(np.float32), info

# =============================================================================
# 8. 读取上游特征
# =============================================================================
def get_bundle_paths(ctx):
    tag, root = ctx.feature_file_tag, ctx.feature_data_dir
    paths = {
        "feature_train": os.path.join(root, f"feature_{tag}_train.pt"),
        "feature_test": os.path.join(root, f"feature_{tag}_test.pt"),
        "label_train": os.path.join(root, f"labels1_{tag}_train.pt"),
        "label_test": os.path.join(root, f"labels2_{tag}_test.pt"),
        "index_train": os.path.join(root, f"indices_{tag}_train.pt"),
        "index_test": os.path.join(root, f"indices_{tag}_test.pt"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"[{k}] 不存在: {p}")
    return paths

def load_feature_bundle(ctx):
    paths = get_bundle_paths(ctx)

    print("\n[FEATURE BUNDLE]")
    print("dataset   =", ctx.dataset)
    print("feature   =", ctx.feature_model_tag)
    print("selection =", ctx.feature_selection_tag)
    print("dir       =", ctx.feature_data_dir)

    X_train = _to_numpy(_safe_torch_load(paths["feature_train"]), np.float32)
    X_val = _to_numpy(_safe_torch_load(paths["feature_test"]), np.float32)

    y_train_raw = ensure_targets(_to_numpy(_safe_torch_load(paths["label_train"]), np.float32))
    y_val_raw = ensure_targets(_to_numpy(_safe_torch_load(paths["label_test"]), np.float32))

    idx_train = _to_numpy(_safe_torch_load(paths["index_train"]))
    idx_val = _to_numpy(_safe_torch_load(paths["index_test"]))

    if idx_train.ndim != 2 or idx_train.shape[1] < 2:
        raise ValueError(f"index_train 应为 (N,2)，当前={idx_train.shape}")
    if idx_val.ndim != 2 or idx_val.shape[1] < 2:
        raise ValueError(f"index_test 应为 (N,2)，当前={idx_val.shape}")

    audio_train = idx_train[:, 0].astype(np.int64)
    audio_val = idx_val[:, 0].astype(np.int64)

    if not (len(X_train) == len(y_train_raw) == len(audio_train)):
        raise ValueError("official_train 的 X / y / audio_id 长度不一致。")
    if not (len(X_val) == len(y_val_raw) == len(audio_val)):
        raise ValueError("official_test 的 X / y / audio_id 长度不一致。")

    if FEATURE_STANDARDIZE:
        X_train = preprocess_data(X_train)
        X_val = preprocess_data(X_val)

    if INCLUDE_TIME:
        seq_len = X_train.shape[1]
        t = (float(T_MAX) / float(seq_len)) * np.arange(seq_len, dtype=np.float32)
        X_train = np.concatenate(
            [np.repeat(t[None, :, None], len(X_train), axis=0), X_train], axis=2
        )
        X_val = np.concatenate(
            [np.repeat(t[None, :, None], len(X_val), axis=0), X_val], axis=2
        )

    validate_feature_shapes(X_train, X_val)

    norm = fit_target_normalizer_train_only(y_train_raw)
    y_train_norm = normalize_targets(y_train_raw, norm)
    y_val_norm = normalize_targets(y_val_raw, norm)

    # 标签也必须在训练前确认全部有限。
    _validate_numpy_finite("y_train_raw", y_train_raw)
    _validate_numpy_finite("y_val_raw", y_val_raw)
    _validate_numpy_finite("y_train_norm", y_train_norm)
    _validate_numpy_finite("y_val_norm", y_val_norm)

    weights = np.ones(len(y_train_norm), dtype=np.float32)
    weight_info = {"enabled": False, "reason": "disabled_or_not_PROCESS-2"}

    if USE_PROCESS2_MMSE_BIN_WEIGHT and _is_process2(ctx.dataset):
        weights, weight_info = build_process2_segment_weights(
            y_train_raw,
            audio_train,
        )

    _validate_numpy_finite(
        "train_segment_weights",
        weights,
    )

    print("[TARGET NORMALIZER] =", norm)
    print("[PROCESS-2 WEIGHT]   =", weight_info)

    return FeatureBundle(
        X_train=np.asarray(X_train, np.float32),
        X_val=np.asarray(X_val, np.float32),
        y_train_norm=np.asarray(y_train_norm, np.float32),
        y_val_norm=np.asarray(y_val_norm, np.float32),
        y_train_raw=np.asarray(y_train_raw, np.float32),
        y_val_raw=np.asarray(y_val_raw, np.float32),
        idx_train_audio=np.asarray(audio_train, np.int64),
        idx_val_audio=np.asarray(audio_val, np.int64),
        train_segment_weights=np.asarray(weights, np.float32),
        target_normalizer=norm,
        process2_weight_info=weight_info,
        data_dim=int(X_train.shape[-1]),
    )

# =============================================================================
# 9. ========================== SLiCE 模型 ====================================
# =============================================================================
def _assert_finite_tensor(tensor, stage):
    """
    对模型中间张量做严格有限值检查。
    发现 NaN/Inf 立即报出真正发生的位置，而不是等到 sklearn.r2_score。
    """
    if not SLICE_NUMERICAL_GUARD:
        return

    finite_mask = torch.isfinite(tensor)

    if bool(finite_mask.all().item()):
        return

    bad_count = int(
        (~finite_mask).sum().detach().cpu().item()
    )

    finite_values = tensor.detach()[finite_mask]

    if finite_values.numel() > 0:
        finite_min = float(
            finite_values.min().cpu().item()
        )
        finite_max = float(
            finite_values.max().cpu().item()
        )
    else:
        finite_min = float("nan")
        finite_max = float("nan")

    raise FloatingPointError(
        f"[SLiCE NONFINITE] stage={stage} | "
        f"shape={tuple(tensor.shape)} | "
        f"bad_count={bad_count} | "
        f"finite_min={finite_min:.6e} | "
        f"finite_max={finite_max:.6e}"
    )


def _nonfinite_parameter_names(model, check_grad=False):
    """
    返回出现 NaN/Inf 的 parameter 或 gradient 名称。
    """
    bad_names = []

    for name, param in model.named_parameters():
        tensor = param.grad if check_grad else param

        if tensor is None:
            continue

        if not bool(
            torch.isfinite(tensor).all().item()
        ):
            bad_names.append(name)

    return bad_names


class SLiCEMMSERegressor(nn.Module):
    """
    输入  : [B,T,C]
    输出  : [B]

    input_proj -> 官方 SLiCELayer x N -> pooling -> MMSE 回归头。
    """

    def __init__(self, input_dim, use_parallel=None):
        super().__init__()

        if use_parallel is None:
            use_parallel = SLICE_USE_PARALLEL

        if (
            not SLICE_DIAGONAL_DENSE
            and SLICE_HIDDEN_DIM % SLICE_BLOCK_SIZE != 0
        ):
            raise ValueError(
                "BD-SLiCE 要求 SLICE_HIDDEN_DIM 能被 SLICE_BLOCK_SIZE 整除。"
            )

        self.pool_mode = str(SLICE_READOUT_POOL).lower()

        # Whisper feature C -> hidden H
        self.input_proj = nn.Linear(int(input_dim), int(SLICE_HIDDEN_DIM))

        # 官方 SLiCELayer。
        # block_size=4 + diagonal_dense=False => BD-SLiCE。
        self.layers = nn.ModuleList(
            [
                SLiCELayer(
                    input_dim=int(SLICE_HIDDEN_DIM),
                    bias=bool(SLICE_BIAS),
                    block_size=int(SLICE_BLOCK_SIZE),
                    diagonal_dense=bool(SLICE_DIAGONAL_DENSE),
                    init_std=float(SLICE_INIT_STD),
                    scale=float(SLICE_SCALE),
                    input_dependent_init=bool(SLICE_INPUT_DEPENDENT_INIT),
                    use_parallel=bool(use_parallel),
                    chunk_size=int(SLICE_CHUNK_SIZE),
                    dropout_rate=float(SLICE_DROPOUT_RATE),
                    path_mode=str(SLICE_PATH_MODE),
                    transition_mode=str(SLICE_TRANSITION_MODE),
                    norm_type=str(SLICE_NORM_TYPE),
                    prenorm=bool(SLICE_PRENORM),
                    second_norm=bool(SLICE_SECOND_NORM),
                    ff_style=str(SLICE_FF_STYLE),
                    ff_activation=str(SLICE_FF_ACTIVATION),
                    ff_mult=int(SLICE_FF_MULT),
                    dropout_position=str(SLICE_DROPOUT_POSITION),
                    norm_eps=float(SLICE_NORM_EPS),
                )
                for _ in range(int(SLICE_NUM_LAYERS))
            ]
        )

        pooled_dim = (
            int(SLICE_HIDDEN_DIM) * 2
            if self.pool_mode == "meanmax"
            else int(SLICE_HIDDEN_DIM)
        )

        # MMSE 回归头：Pooling -> LayerNorm -> Linear -> GELU -> Linear(1)
        self.pooled_norm = nn.LayerNorm(pooled_dim)
        self.fc1 = nn.Linear(pooled_dim, int(SLICE_MLP_HIDDEN_DIM))
        self.fc2 = nn.Linear(int(SLICE_MLP_HIDDEN_DIM), 1)

    def _pool(self, x):
        if self.pool_mode == "mean":
            return x.mean(dim=1)
        if self.pool_mode == "last":
            return x[:, -1, :]
        if self.pool_mode == "meanmax":
            return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        raise ValueError(f"未知 pooling: {self.pool_mode}")

    def forward(self, x):
        if x.ndim != 3:
            raise ValueError(
                f"SLiCE 输入必须为 [B,T,C]，当前={tuple(x.shape)}"
            )

        _assert_finite_tensor(
            x,
            "model_input",
        )

        x = self.input_proj(x)
        _assert_finite_tensor(
            x,
            "after_input_proj",
        )

        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            _assert_finite_tensor(
                x,
                f"after_slice_layer_{layer_idx}",
            )

        x = self._pool(x)
        _assert_finite_tensor(
            x,
            "after_pool",
        )

        x = self.pooled_norm(x)
        _assert_finite_tensor(
            x,
            "after_pooled_norm",
        )

        x = self.fc1(x)
        x = F.gelu(x)
        _assert_finite_tensor(
            x,
            "after_fc1_gelu",
        )

        x = self.fc2(x)
        _assert_finite_tensor(
            x,
            "after_fc2",
        )

        return x.squeeze(-1)

    def effective_parallel_flags(self):
        """
        官方实现可能在 associative_scan 不可用时自动回退。
        因此正式日志一定要看这个实际值。
        """
        return [
            bool(getattr(getattr(layer, "slice", None), "use_parallel", False))
            for layer in self.layers
        ]

def create_slice_model(input_dim, device, use_parallel=None):
    return SLiCEMMSERegressor(
        input_dim=input_dim,
        use_parallel=use_parallel,
    ).to(device)


# =============================================================================
# 10. DataLoader / loss / learning rate
# =============================================================================
def build_loader(X, y, weights=None, batch_size=16, shuffle=False, seed=400):
    X = np.asarray(X, dtype=np.float32)
    y = ensure_targets(y)

    if X.ndim != 3:
        raise ValueError(f"DataLoader 输入必须为 [N,T,C]，当前={X.shape}")

    if weights is None:
        weights = np.ones(len(y), dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32).reshape(-1)

    if len(X) != len(y) or len(y) != len(weights):
        raise ValueError("X / y / weights 长度不一致。")

    ds = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(weights),
    )

    g = torch.Generator()
    g.manual_seed(int(seed))

    return DataLoader(
        ds,
        batch_size=int(max(1, batch_size)),
        shuffle=bool(shuffle),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=g,
    )

def mixed_mse_loss_torch(pred, target, sample_weight, mixed_loss_alpha):
    """
    ordinary MSE + weighted MSE 混合。
    """
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    sample_weight = sample_weight.reshape(-1)

    sq_error = (pred - target) ** 2
    ordinary_mse = sq_error.mean()

    sample_weight = torch.clamp(sample_weight, min=float(PROCESS2_WEIGHT_EPS))
    denom = torch.clamp(sample_weight.sum(), min=float(PROCESS2_WEIGHT_EPS))
    weighted_mse = (sample_weight * sq_error).sum() / denom

    alpha = float(np.clip(mixed_loss_alpha, 0.0, 1.0))
    loss = (1.0 - alpha) * ordinary_mse + alpha * weighted_mse

    return loss, ordinary_mse, weighted_mse

def get_warmup_cosine_lr(global_step, total_steps, warmup_steps, base_lr, cosine_alpha):
    """
    与当前 PyTorch MLP baseline 保持同一形式。
    """
    global_step = int(max(global_step, 0))
    total_steps = int(max(total_steps, 1))
    warmup_steps = int(max(warmup_steps, 0))
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))

    base_lr = float(base_lr)
    cosine_alpha = float(np.clip(cosine_alpha, 0.0, 1.0))

    if warmup_steps > 0 and global_step < warmup_steps:
        return float(base_lr * global_step / max(warmup_steps, 1))

    decay_steps = max(total_steps - warmup_steps, 1)
    decay_step = max(global_step - warmup_steps, 0)
    progress = float(np.clip(decay_step / decay_steps, 0.0, 1.0))
    cosine_value = 0.5 * (1.0 + np.cos(np.pi * progress))

    return float(
        base_lr * (
            cosine_alpha
            + (1.0 - cosine_alpha) * cosine_value
        )
    )

# =============================================================================
# 11. 推理 / 训练
# =============================================================================
def predict_loader_normalized(model, loader, device):
    """
    返回 normalized 空间：
        pred [N]
        true [N]

    正式指标不允许通过过滤 / nan_to_num 来“修复”预测。
    若模型产生 NaN/Inf，立即报出。
    """
    model.eval()
    preds, labels = [], []

    with torch.no_grad():
        for batch_idx, (xb, yb, _) in enumerate(
            loader,
            start=1,
        ):
            xb = xb.to(
                device,
                non_blocking=True,
            )

            pred = model(xb)

            if not bool(
                torch.isfinite(pred).all().item()
            ):
                bad_count = int(
                    (~torch.isfinite(pred))
                    .sum()
                    .detach()
                    .cpu()
                    .item()
                )

                raise FloatingPointError(
                    "[SLiCE INFERENCE NONFINITE] "
                    f"batch={batch_idx} | "
                    f"bad_pred_count={bad_count}"
                )

            preds.append(
                pred.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            labels.append(
                yb.numpy().reshape(-1)
            )

    if not preds:
        return (
            np.array([], np.float32),
            np.array([], np.float32),
        )

    pred_all = np.concatenate(
        preds
    ).astype(np.float32)

    label_all = np.concatenate(
        labels
    ).astype(np.float32)

    _validate_numpy_finite(
        "inference_predictions",
        pred_all,
    )

    _validate_numpy_finite(
        "inference_labels",
        label_all,
    )

    return (
        pred_all,
        label_all,
    )


def evaluate_monitor_mse(model, loader, device):
    pred, true = predict_loader_normalized(model, loader, device)
    if len(true) == 0:
        return float("nan")
    return float(np.mean((pred - true) ** 2))

def train_slice_model(
    model,
    train_loader,
    val_loader,
    device,
    mixed_loss_alpha,
):
    """
    数值稳定版 SLiCE 训练。

    与当前训练协议保持一致：
    - 最大 10 epoch；
    - train_loss early stopping；
    - Adam；
    - 无 weight decay；
    - warmup + cosine；
    - official_test 仅 monitor；
    - restore_best=False。

    额外增加的只是“数值故障保护”：
    - 数据 / forward / loss / gradient / parameter 全链路检查；
    - NaN/Inf 一旦出现立即停止当前 epoch；
    - 恢复到“上一完整且有限的 epoch”，而不是继续使用已经污染的模型；
    - numerical stop 与普通 early stop 分开记录。
    """

    if (
        str(SLICE_OPTIMIZER_NAME)
        .strip()
        .lower()
        != "adam"
    ):
        raise ValueError(
            "当前协议要求 SLICE_OPTIMIZER_NAME='adam'。"
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(LR_SLICE),
    )

    steps_per_epoch = len(train_loader)

    if steps_per_epoch <= 0:
        raise RuntimeError(
            "train_loader 为空。"
        )

    total_steps = int(
        SLICE_TARGET_EPOCHS
        * steps_per_epoch
    )

    warmup_steps = int(
        SLICE_WARMUP_EPOCHS
        * steps_per_epoch
    )

    warmup_steps = min(
        warmup_steps,
        max(
            total_steps - 1,
            0,
        ),
    )

    best_train_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(
        model.state_dict()
    )
    no_improve = 0

    global_step = 0
    history = []

    stopped_early = False
    numerical_stop = False
    numerical_stop_message = None

    stop_reason = (
        "max_epochs_reached"
    )

    print(
        "\n[SLiCE TRAIN]"
    )

    print(
        f"optimizer=Adam | "
        f"weight_decay=0 | "
        f"grad_clip_norm={SLICE_GRAD_CLIP_NORM} | "
        f"early_stopping={USE_EARLY_STOPPING} | "
        f"scale={SLICE_SCALE} | "
        f"chunk_size={SLICE_CHUNK_SIZE} | "
        f"steps_per_epoch={steps_per_epoch} | "
        f"total_steps={total_steps} | "
        f"warmup_steps={warmup_steps} | "
        f"base_lr={LR_SLICE}"
    )

    for epoch in range(
        1,
        int(SLICE_TARGET_EPOCHS) + 1,
    ):
        model.train()

        # -----------------------------------------------------
        # 保存“当前 epoch 开始前”的有限模型。
        #
        # 若本 epoch 中途发生 NaN/Inf，
        # 恢复到这里。
        #
        # 这不是 restore_best：
        # 它恢复的是上一完整 epoch，而不是历史最优 epoch。
        # -----------------------------------------------------
        epoch_start_state = copy.deepcopy(
            model.state_dict()
        )

        batch_losses = []
        last_lr = float("nan")

        epoch_numerical_failure = False
        epoch_failure_message = None

        for batch_idx, (xb, yb, wb) in enumerate(
            train_loader,
            start=1,
        ):
            xb = xb.to(
                device,
                non_blocking=True,
            )
            yb = yb.to(
                device,
                non_blocking=True,
            )
            wb = wb.to(
                device,
                non_blocking=True,
            )

            current_lr = get_warmup_cosine_lr(
                global_step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                base_lr=LR_SLICE,
                cosine_alpha=SLICE_COSINE_ALPHA,
            )

            for group in optimizer.param_groups:
                group["lr"] = current_lr

            try:
                # =============================================
                # 1. Forward
                # =============================================
                pred = model(xb)

                if not bool(
                    torch.isfinite(pred).all().item()
                ):
                    raise FloatingPointError(
                        "prediction 出现 NaN/Inf"
                    )

                # =============================================
                # 2. Loss
                # =============================================
                (
                    loss,
                    plain_mse,
                    weighted_mse,
                ) = mixed_mse_loss_torch(
                    pred,
                    yb,
                    wb,
                    mixed_loss_alpha,
                )

                if not bool(
                    torch.isfinite(loss).item()
                ):
                    raise FloatingPointError(
                        "loss 出现 NaN/Inf | "
                        f"plain_mse="
                        f"{float(plain_mse.detach().cpu())} | "
                        f"weighted_mse="
                        f"{float(weighted_mse.detach().cpu())}"
                    )

                # =============================================
                # 3. Backward
                # =============================================
                optimizer.zero_grad(
                    set_to_none=True
                )

                loss.backward()

                bad_grad_names = (
                    _nonfinite_parameter_names(
                        model,
                        check_grad=True,
                    )
                )

                if bad_grad_names:
                    raise FloatingPointError(
                        "backward 后 gradient 出现 NaN/Inf | "
                        f"parameters={bad_grad_names[:20]}"
                    )

                # =============================================
                # 4. Gradient clipping
                #
                # 关键修改：
                # error_if_nonfinite=True。
                #
                # 原代码默认 False，
                # 即使总梯度范数已经 NaN/Inf，
                # 也不会在这里及时阻断。
                # =============================================
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(
                            SLICE_GRAD_CLIP_NORM
                        ),
                        error_if_nonfinite=True,
                    )
                )

                if not bool(
                    torch.isfinite(grad_norm).item()
                ):
                    raise FloatingPointError(
                        "gradient total norm 非有限"
                    )

                # =============================================
                # 5. Optimizer step
                # =============================================
                optimizer.step()

                bad_param_names = (
                    _nonfinite_parameter_names(
                        model,
                        check_grad=False,
                    )
                )

                if bad_param_names:
                    raise FloatingPointError(
                        "optimizer.step 后 parameter "
                        "出现 NaN/Inf | "
                        f"parameters={bad_param_names[:20]}"
                    )

            except (
                FloatingPointError,
                RuntimeError,
            ) as exc:
                epoch_numerical_failure = True

                epoch_failure_message = (
                    f"epoch={epoch}, "
                    f"batch={batch_idx}/{steps_per_epoch}, "
                    f"global_step={global_step}, "
                    f"lr={current_lr:.10e}, "
                    f"error={type(exc).__name__}: {exc}"
                )

                print(
                    "\n[SLiCE NUMERICAL STOP] "
                    + epoch_failure_message
                )

                break

            batch_losses.append(
                float(
                    loss.detach()
                    .cpu()
                    .item()
                )
            )

            last_lr = float(
                current_lr
            )

            global_step += 1

        # -----------------------------------------------------
        # 本 epoch 中途出现 NaN/Inf：
        # 恢复上一完整有限 epoch。
        # -----------------------------------------------------
        if epoch_numerical_failure:
            model.load_state_dict(
                epoch_start_state
            )

            numerical_stop = True
            numerical_stop_message = (
                epoch_failure_message
            )

            stop_reason = (
                "numerical_nonfinite"
            )

            if len(history) == 0:
                raise RuntimeError(
                    "[SLiCE] 第 1 个 epoch "
                    "尚未完整结束就发生数值异常；"
                    "没有可恢复的上一完整 epoch。\n"
                    + str(
                        numerical_stop_message
                    )
                )

            print(
                "[SLiCE NUMERICAL RECOVERY] "
                f"已恢复到上一完整有限 epoch="
                f"{history[-1]['epoch']}。"
            )

            break

        # -----------------------------------------------------
        # 完整 epoch train loss
        # -----------------------------------------------------
        if len(batch_losses) == 0:
            raise RuntimeError(
                f"epoch={epoch} 没有有效 batch loss。"
            )

        train_loss_epoch = float(
            np.mean(
                batch_losses
            )
        )

        if not np.isfinite(
            train_loss_epoch
        ):
            model.load_state_dict(
                epoch_start_state
            )

            numerical_stop = True
            numerical_stop_message = (
                f"epoch={epoch} 的 train_loss 非有限: "
                f"{train_loss_epoch}"
            )
            stop_reason = "numerical_nonfinite"

            if len(history) == 0:
                raise RuntimeError(
                    numerical_stop_message
                )

            print(
                "[SLiCE NUMERICAL RECOVERY] "
                + numerical_stop_message
            )

            break

        # -----------------------------------------------------
        # official_test 仅 monitor
        # -----------------------------------------------------
        try:
            val_monitor_mse = (
                evaluate_monitor_mse(
                    model,
                    val_loader,
                    device,
                )
            )
        except FloatingPointError as exc:
            model.load_state_dict(
                epoch_start_state
            )

            numerical_stop = True
            numerical_stop_message = (
                f"epoch={epoch} official_test monitor "
                f"出现非有限预测: {exc}"
            )
            stop_reason = "numerical_nonfinite"

            if len(history) == 0:
                raise RuntimeError(
                    numerical_stop_message
                )

            print(
                "[SLiCE NUMERICAL RECOVERY] "
                + numerical_stop_message
            )

            break

        if not np.isfinite(
            val_monitor_mse
        ):
            model.load_state_dict(
                epoch_start_state
            )

            numerical_stop = True
            numerical_stop_message = (
                f"epoch={epoch} val_monitor_mse 非有限: "
                f"{val_monitor_mse}"
            )
            stop_reason = "numerical_nonfinite"

            if len(history) == 0:
                raise RuntimeError(
                    numerical_stop_message
                )

            print(
                "[SLiCE NUMERICAL RECOVERY] "
                + numerical_stop_message
            )

            break

        # -----------------------------------------------------
        # train-loss early stopping
        # -----------------------------------------------------
        improved = (
            train_loss_epoch
            <
            best_train_loss
            -
            float(
                EARLY_STOPPING_MIN_DELTA
            )
        )

        if improved:
            best_train_loss = float(
                train_loss_epoch
            )
            best_epoch = int(
                epoch
            )
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improve = 0
        else:
            no_improve += 1

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(
                    train_loss_epoch
                ),
                "val_monitor_mse": float(
                    val_monitor_mse
                ),
                "lr": float(
                    last_lr
                ),
                "train_loss_improved": int(
                    improved
                ),
                "best_train_epoch": int(
                    best_epoch
                ),
                "no_improve_epochs": int(
                    no_improve
                ),
                "numerical_failure": 0,
            }
        )

        print(
            f"[SLiCE] "
            f"epoch={epoch:02d}/"
            f"{SLICE_TARGET_EPOCHS} "
            f"train_loss="
            f"{train_loss_epoch:.6f} "
            f"official_test_monitor_mse="
            f"{val_monitor_mse:.6f} "
            f"lr={last_lr:.8f} "
            f"best_train_epoch="
            f"{best_epoch}"
        )

        if (
            USE_EARLY_STOPPING
            and
            no_improve
            >= int(
                EARLY_STOPPING_STEPS
            )
        ):
            stopped_early = True
            stop_reason = (
                "train_loss_early_stop"
            )

            print(
                f"[SLiCE] early stop: "
                f"epoch={epoch}, "
                f"best_train_epoch={best_epoch}, "
                f"best_train_loss="
                f"{best_train_loss:.6f}"
            )

            break

    if not history:
        raise RuntimeError(
            "未生成任何完整且有限的训练 epoch。"
        )

    last_epoch = int(
        history[-1]["epoch"]
    )

    # ---------------------------------------------------------
    # 数值停止：
    # 当前模型已恢复为上一完整有限 epoch。
    #
    # 这不是普通 early stopping，
    # 因此单独记录 selection policy。
    # ---------------------------------------------------------
    if numerical_stop:
        selected_epoch = int(
            last_epoch
        )
        selection_policy = (
            "last_finite_epoch_after_"
            "numerical_nonfinite"
        )
        selected_by = (
            "last_finite_epoch_"
            "numerical_guard"
        )

    elif (
        USE_EARLY_STOPPING
        and
        EARLY_STOPPING_RESTORE_BEST
    ):
        model.load_state_dict(
            best_state
        )

        selected_epoch = int(
            best_epoch
        )
        selection_policy = (
            "train_loss_early_stop_"
            "restore_best"
        )
        selected_by = (
            "best_train_loss_checkpoint"
        )

    else:
        selected_epoch = int(
            last_epoch
        )

        selection_policy = (
            "last_epoch_or_"
            "train_loss_early_stop"
            if USE_EARLY_STOPPING
            else
            "last_epoch_fixed_budget"
        )

        selected_by = (
            "last_epoch_or_"
            "train_loss_early_stop_"
            "no_test_selection"
            if USE_EARLY_STOPPING
            else
            "last_epoch_no_test_selection"
        )

    model.eval()

    # 最终保存前再做一次参数检查。
    bad_final_params = (
        _nonfinite_parameter_names(
            model,
            check_grad=False,
        )
    )

    if bad_final_params:
        raise FloatingPointError(
            "[SLiCE] 最终模型仍含 NaN/Inf 参数: "
            f"{bad_final_params[:20]}"
        )

    return {
        "model": model,
        "best_epoch": int(
            best_epoch
        ),
        "best_train_loss": float(
            best_train_loss
        ),
        "selected_epoch": int(
            selected_epoch
        ),
        "last_epoch": int(
            last_epoch
        ),
        "selection_policy": (
            selection_policy
        ),
        "selected_by": (
            selected_by
        ),
        "early_stopping_used": int(
            bool(
                USE_EARLY_STOPPING
            )
        ),
        "stopped_early": int(
            stopped_early
        ),
        "numerical_stop": int(
            numerical_stop
        ),
        "numerical_stop_message": (
            numerical_stop_message
        ),
        "stop_reason": (
            stop_reason
        ),
        "history": history,
        "steps_per_epoch": int(
            steps_per_epoch
        ),
        "total_steps": int(
            total_steps
        ),
        "warmup_steps": int(
            warmup_steps
        ),
    }


# =============================================================================
# 12. 回归 / audio-level 指标
# =============================================================================
def regression_metrics(y_true, y_pred):
    """
    输入必须是原始 MMSE 空间。

    正式指标禁止：
    - nan_to_num
    - 删除 NaN 样本
    - 只对有限子集算指标

    因为这些做法会掩盖模型数值失败。
    """
    y_true = np.asarray(
        y_true,
        np.float32,
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred,
        np.float32,
    ).reshape(-1)

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true / y_pred 长度不一致: "
            f"{len(y_true)} vs {len(y_pred)}"
        )

    if len(y_true) == 0:
        return {
            "count": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson": float("nan"),
        }

    _validate_numpy_finite(
        "metric_y_true",
        y_true,
    )

    _validate_numpy_finite(
        "metric_y_pred",
        y_pred,
    )

    rmse = float(
        np.sqrt(
            np.mean(
                (
                    y_pred
                    -
                    y_true
                ) ** 2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                y_pred
                -
                y_true
            )
        )
    )

    r2 = (
        float(
            r2_score(
                y_true,
                y_pred,
            )
        )
        if (
            len(y_true) >= 2
            and
            float(
                np.std(y_true)
            )
            > 1e-12
        )
        else
        float("nan")
    )

    pearson = (
        float(
            np.corrcoef(
                y_true,
                y_pred,
            )[0, 1]
        )
        if (
            len(y_true) >= 2
            and
            float(
                np.std(y_true)
            )
            > 1e-12
            and
            float(
                np.std(y_pred)
            )
            > 1e-12
        )
        else
        float("nan")
    )

    return {
        "count": int(
            len(y_true)
        ),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def trimmed_mean(values, ratio):
    arr = np.asarray(values, np.float32).reshape(-1)
    if len(arr) <= 1:
        return float(arr[0]) if len(arr) == 1 else float("nan")

    ratio = float(np.clip(ratio, 0.0, 0.49))
    k = int(np.floor(len(arr) * ratio))

    if k <= 0 or 2 * k >= len(arr):
        return float(np.mean(arr))

    arr = np.sort(arr)[k: len(arr) - k]
    return float(np.mean(arr))

def reduce_segments(values, method):
    arr = np.asarray(values, np.float32).reshape(-1)
    if method == "mean":
        return float(np.mean(arr))
    if method == "median":
        return float(np.median(arr))
    if method == "trimmed_mean":
        return trimmed_mean(arr, AUDIO_AGG_TRIM_RATIO)
    raise ValueError(f"未知聚合方式: {method}")

def process2_bin_metrics(y_true_audio, y_pred_audio):
    rows, rmses, maes = [], [], []

    for label in PROCESS2_MMSE_BIN_LABELS:
        left = float(label.split(",")[0].replace("[", ""))
        right = float(label.split(",")[1].replace(")", ""))

        mask = (y_true_audio >= left) & (y_true_audio < right)

        if int(mask.sum()) == 0:
            row = {"mmse_bin": label, "count": 0, "rmse": float("nan"), "mae": float("nan")}
        else:
            yt, yp = y_true_audio[mask], y_pred_audio[mask]
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            mae = float(np.mean(np.abs(yp - yt)))
            row = {"mmse_bin": label, "count": int(mask.sum()), "rmse": rmse, "mae": mae}
            rmses.append(rmse)
            maes.append(mae)

        rows.append(row)

    macro = {
        "macro_bin_rmse": float(np.mean(rmses)) if rmses else float("nan"),
        "macro_bin_mae": float(np.mean(maes)) if maes else float("nan"),
        "macro_bin_nonempty_count": int(len(rmses)),
    }

    return {"per_bin": rows, "macro": macro}

def aggregate_audio(dataset, audio_ids, segment_preds_raw, segment_true_raw):
    """
    同时计算 mean / median / trimmed_mean。
    主结果由 AUDIO_AGG_PRIMARY_METHOD 指定。
    """
    audio_ids = np.asarray(audio_ids, np.int64).reshape(-1)
    preds = np.asarray(segment_preds_raw, np.float32).reshape(-1)
    trues = np.asarray(segment_true_raw, np.float32).reshape(-1)

    if not (len(audio_ids) == len(preds) == len(trues)):
        raise ValueError("audio_ids / preds / trues 长度不一致。")

    pred_map, true_map = {}, {}

    for aid, pred, true in zip(audio_ids, preds, trues):
        aid = int(aid)
        pred_map.setdefault(aid, []).append(float(pred))
        true_map.setdefault(aid, []).append(float(true))

    sorted_ids = np.asarray(sorted(pred_map), np.int64)
    y_true_audio = np.asarray(
        [np.mean(true_map[int(aid)]) for aid in sorted_ids],
        np.float32,
    )

    all_methods = {}

    for method in ["mean", "median", "trimmed_mean"]:
        y_pred_audio = np.asarray(
            [reduce_segments(pred_map[int(aid)], method) for aid in sorted_ids],
            np.float32,
        )

        metrics = regression_metrics(y_true_audio, y_pred_audio)

        per_bin = None
        if _is_process2(dataset):
            per_bin = process2_bin_metrics(y_true_audio, y_pred_audio)
            metrics.update(per_bin["macro"])

        all_methods[method] = {
            "sorted_audio_ids": sorted_ids.copy(),
            "y_true_audio": y_true_audio.copy(),
            "y_pred_audio": y_pred_audio,
            "metrics": metrics,
            "process2_per_bin_metrics": None if per_bin is None else per_bin["per_bin"],
        }

    selected = all_methods[str(AUDIO_AGG_PRIMARY_METHOD).lower()]

    return {
        "aggregation_method": str(AUDIO_AGG_PRIMARY_METHOD).lower(),
        "sorted_audio_ids": selected["sorted_audio_ids"],
        "y_true_audio": selected["y_true_audio"],
        "y_pred_audio": selected["y_pred_audio"],
        "metrics": selected["metrics"],
        "process2_per_bin_metrics": selected["process2_per_bin_metrics"],
        "all_methods": all_methods,
    }

# =============================================================================
# 13. recurrent / parallel 一致性检查
# =============================================================================
def check_parallel_recurrent(bundle, device, seed):
    """
    两个模型加载完全相同 state_dict，
    分别使用 recurrent / parallel，对同一个 batch 前向。
    """
    print("\n[CHECK] recurrent vs parallel")

    seed_everything(seed)
    rec_model = create_slice_model(bundle.data_dim, device, use_parallel=False)
    common_state = copy.deepcopy(rec_model.state_dict())

    seed_everything(seed)
    par_model = create_slice_model(bundle.data_dim, device, use_parallel=True)
    par_model.load_state_dict(common_state)

    rec_model.eval()
    par_model.eval()

    n = min(int(SLICE_CHECK_BATCH_SIZE), len(bundle.X_train))
    x = torch.from_numpy(bundle.X_train[:n]).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        y_rec = rec_model(x)
        y_par = par_model(x)

    diff = (y_rec - y_par).abs()

    print("recurrent flags =", rec_model.effective_parallel_flags())
    print("parallel flags  =", par_model.effective_parallel_flags())
    print("max_abs_diff    =", float(diff.max().cpu()))
    print("mean_abs_diff   =", float(diff.mean().cpu()))

    del rec_model, par_model, x, y_rec, y_par
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =============================================================================
# 14. 单 seed 输出
# =============================================================================
def audio_prediction_df(seed, audio_result):
    return pd.DataFrame({
        "seed": int(seed),
        "audio_id": audio_result["sorted_audio_ids"],
        "y_true_mmse": audio_result["y_true_audio"],
        "pred_mean": audio_result["all_methods"]["mean"]["y_pred_audio"],
        "pred_median": audio_result["all_methods"]["median"]["y_pred_audio"],
        "pred_trimmed_mean": audio_result["all_methods"]["trimmed_mean"]["y_pred_audio"],
    })

def save_seed_outputs(
    ctx,
    bundle,
    seed,
    train_info,
    model,
    train_pred_norm,
    train_true_norm,
    val_pred_norm,
    val_true_norm,
):
    seed_dir = os.path.join(ctx.output_dir, f"seed_{int(seed)}")
    os.makedirs(seed_dir, exist_ok=True)

    # 反标准化回原始 MMSE
    train_pred_raw = denormalize_targets(train_pred_norm, bundle.target_normalizer)
    train_true_raw = denormalize_targets(train_true_norm, bundle.target_normalizer)
    val_pred_raw = denormalize_targets(val_pred_norm, bundle.target_normalizer)
    val_true_raw = denormalize_targets(val_true_norm, bundle.target_normalizer)

    train_seg_metrics = regression_metrics(train_true_raw, train_pred_raw)
    val_seg_metrics = regression_metrics(val_true_raw, val_pred_raw)

    train_audio = aggregate_audio(
        ctx.dataset,
        bundle.idx_train_audio,
        train_pred_raw,
        train_true_raw,
    )
    val_audio = aggregate_audio(
        ctx.dataset,
        bundle.idx_val_audio,
        val_pred_raw,
        val_true_raw,
    )

    # checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": int(seed),
            "input_dim": int(bundle.data_dim),
            "selected_epoch": int(train_info["selected_epoch"]),
            "selection_policy": train_info["selection_policy"],
            "slice_config": {
                "hidden_dim": SLICE_HIDDEN_DIM,
                "num_layers": SLICE_NUM_LAYERS,
                "block_size": SLICE_BLOCK_SIZE,
                "diagonal_dense": SLICE_DIAGONAL_DENSE,
                "use_parallel_requested": SLICE_USE_PARALLEL,
                "use_parallel_effective_per_layer": model.effective_parallel_flags(),
                "chunk_size": SLICE_CHUNK_SIZE,
                "path_mode": SLICE_PATH_MODE,
                "transition_mode": SLICE_TRANSITION_MODE,
                "ff_mult": SLICE_FF_MULT,
                "readout_pool": SLICE_READOUT_POOL,
            },
            "target_normalizer": bundle.target_normalizer,
        },
        os.path.join(seed_dir, "selected_last_model.pt"),
    )

    # history
    pd.DataFrame(train_info["history"]).to_csv(
        os.path.join(seed_dir, "training_history.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # segment predictions
    pd.DataFrame({
        "seed": int(seed),
        "audio_id": bundle.idx_train_audio,
        "y_true_mmse": train_true_raw,
        "y_pred_mmse": train_pred_raw,
        "y_true_norm": train_true_norm,
        "y_pred_norm": train_pred_norm,
    }).to_csv(
        os.path.join(seed_dir, "train_segment_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame({
        "seed": int(seed),
        "audio_id": bundle.idx_val_audio,
        "y_true_mmse": val_true_raw,
        "y_pred_mmse": val_pred_raw,
        "y_true_norm": val_true_norm,
        "y_pred_norm": val_pred_norm,
    }).to_csv(
        os.path.join(seed_dir, "val_official_test_segment_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # audio predictions
    train_audio_df = audio_prediction_df(seed, train_audio)
    val_audio_df = audio_prediction_df(seed, val_audio)

    train_audio_df.to_csv(
        os.path.join(seed_dir, "train_audio_predictions_all_aggs.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    val_audio_df.to_csv(
        os.path.join(seed_dir, "val_official_test_audio_predictions_all_aggs.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    # PROCESS-2 per-bin
    if val_audio["process2_per_bin_metrics"] is not None:
        pd.DataFrame(val_audio["process2_per_bin_metrics"]).to_csv(
            os.path.join(seed_dir, "val_official_test_process2_per_bin_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    row = {
        "seed": int(seed),
        "dataset": ctx.dataset,
        "source_mode": ctx.source_mode,
        "feature_model": ctx.feature_model_tag,
        "selection": ctx.feature_selection_tag,
        "file_tag": ctx.feature_file_tag,

        "selected_epoch": int(train_info["selected_epoch"]),
        "last_epoch": int(train_info["last_epoch"]),
        "best_train_epoch": int(train_info["best_epoch"]),
        "best_train_loss": float(train_info["best_train_loss"]),
        "stopped_early": int(train_info["stopped_early"]),
        "numerical_stop": int(train_info.get("numerical_stop", 0)),
        "numerical_stop_message": train_info.get("numerical_stop_message"),
        "stop_reason": train_info["stop_reason"],
        "selection_policy": train_info["selection_policy"],

        "use_parallel_requested": int(bool(SLICE_USE_PARALLEL)),
        "use_parallel_effective_all_layers": int(all(model.effective_parallel_flags())),
        "trainable_parameters": count_trainable_parameters(model),

        "train_segment_rmse": train_seg_metrics["rmse"],
        "train_segment_mae": train_seg_metrics["mae"],
        "train_segment_r2": train_seg_metrics["r2"],
        "train_segment_pearson": train_seg_metrics["pearson"],

        "train_audio_rmse": train_audio["metrics"]["rmse"],
        "train_audio_mae": train_audio["metrics"]["mae"],
        "train_audio_r2": train_audio["metrics"]["r2"],
        "train_audio_pearson": train_audio["metrics"]["pearson"],

        "val_segment_rmse": val_seg_metrics["rmse"],
        "val_segment_mae": val_seg_metrics["mae"],
        "val_segment_r2": val_seg_metrics["r2"],
        "val_segment_pearson": val_seg_metrics["pearson"],

        "val_audio_rmse": val_audio["metrics"]["rmse"],
        "val_audio_mae": val_audio["metrics"]["mae"],
        "val_audio_r2": val_audio["metrics"]["r2"],
        "val_audio_pearson": val_audio["metrics"]["pearson"],
    }

    # PROCESS-2 macro bin
    for k in ["macro_bin_rmse", "macro_bin_mae", "macro_bin_nonempty_count"]:
        if k in val_audio["metrics"]:
            row[f"val_{k}"] = val_audio["metrics"][k]

    save_json(row, os.path.join(seed_dir, "metrics.json"))

    return {
        "metrics_row": row,
        "val_audio_df": val_audio_df,
    }

# =============================================================================
# 15. 一个 seed 的完整运行
# =============================================================================
def run_one_seed(ctx, bundle, seed):
    print("\n" + "#" * 96)
    print(
        f"[RUN] dataset={ctx.dataset} | selection={ctx.feature_selection_tag} | seed={seed}"
    )
    print("#" * 96)

    seed_everything(seed)
    device = get_device()

    train_loader = build_loader(
        bundle.X_train,
        bundle.y_train_norm,
        bundle.train_segment_weights,
        batch_size=SLICE_BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    train_eval_loader = build_loader(
        bundle.X_train,
        bundle.y_train_norm,
        bundle.train_segment_weights,
        batch_size=SLICE_BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    val_loader = build_loader(
        bundle.X_val,
        bundle.y_val_norm,
        np.ones(len(bundle.y_val_norm), np.float32),
        batch_size=SLICE_BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    model = create_slice_model(bundle.data_dim, device)

    print("[MODEL] trainable_parameters =", count_trainable_parameters(model))
    print("[MODEL] requested use_parallel =", SLICE_USE_PARALLEL)
    print("[MODEL] effective flags        =", model.effective_parallel_flags())

    if SLICE_USE_PARALLEL and not all(model.effective_parallel_flags()):
        warnings.warn(
            "请求了 use_parallel=True，但至少一层实际回退为 recurrent。",
            RuntimeWarning,
        )

    # smoke test
    if SLICE_SMOKE_TEST_ONLY:
        # Smoke test 与正式训练保持一致：Adam、无 weight decay。
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(LR_SLICE),
        )
        model.train()

        for i, (xb, yb, wb) in enumerate(train_loader):
            if i >= int(SLICE_SMOKE_TEST_NUM_BATCHES):
                break

            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            pred = model(xb)

            alpha = (
                PROCESS2_MIXED_LOSS_ALPHA
                if USE_PROCESS2_MMSE_BIN_WEIGHT and _is_process2(ctx.dataset)
                else 0.0
            )

            loss, _, _ = mixed_mse_loss_torch(pred, yb, wb, alpha)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            print(
                f"[SMOKE] batch={i} X={tuple(xb.shape)} "
                f"pred={tuple(pred.shape)} loss={float(loss.detach().cpu()):.6f}"
            )

        raise SystemExit("SLICE_SMOKE_TEST_ONLY=True，smoke test 完成。")

    alpha = (
        float(PROCESS2_MIXED_LOSS_ALPHA)
        if USE_PROCESS2_MMSE_BIN_WEIGHT and _is_process2(ctx.dataset)
        else 0.0
    )

    train_info = train_slice_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        mixed_loss_alpha=alpha,
    )

    model = train_info["model"]

    train_pred_norm, train_true_norm = predict_loader_normalized(
        model, train_eval_loader, device
    )
    val_pred_norm, val_true_norm = predict_loader_normalized(
        model, val_loader, device
    )

    result = save_seed_outputs(
        ctx=ctx,
        bundle=bundle,
        seed=seed,
        train_info=train_info,
        model=model,
        train_pred_norm=train_pred_norm,
        train_true_norm=train_true_norm,
        val_pred_norm=val_pred_norm,
        val_true_norm=val_true_norm,
    )

    row = result["metrics_row"]

    print(
        f"[FINAL] seed={seed} selected_epoch={row['selected_epoch']} | "
        f"official_test_monitor_segment_rmse={row['val_segment_rmse']:.4f} | "
        f"official_test_monitor_audio_rmse={row['val_audio_rmse']:.4f} | "
        f"val_audio_mae={row['val_audio_mae']:.4f} | "
        f"val_audio_r2={row['val_audio_r2']:.4f} | "
        f"val_audio_pearson={row['val_audio_pearson']:.4f}"
    )

    del train_loader, train_eval_loader, val_loader, model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result

# =============================================================================
# 16. 多 seed 汇总
# =============================================================================
def mean_std_sem(values):
    arr = np.asarray(values, np.float64)
    arr = arr[np.isfinite(arr)]

    if len(arr) == 0:
        return {"mean": float("nan"), "std": float("nan"), "sem": float("nan"), "count": 0}

    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) >= 2 else 0.0
    sem = float(std / np.sqrt(len(arr)))

    return {"mean": mean, "std": std, "sem": sem, "count": int(len(arr))}

def build_summary_df(metrics_df):
    cols = [
        "selected_epoch",
        "train_segment_rmse",
        "train_segment_mae",
        "train_audio_rmse",
        "train_audio_mae",
        "val_segment_rmse",
        "val_segment_mae",
        "val_segment_r2",
        "val_segment_pearson",
        "val_audio_rmse",
        "val_audio_mae",
        "val_audio_r2",
        "val_audio_pearson",
        "val_macro_bin_rmse",
        "val_macro_bin_mae",
    ]

    rows = []
    for col in cols:
        if col in metrics_df.columns:
            s = mean_std_sem(metrics_df[col].to_numpy())
            rows.append({"metric": col, **s})

    return pd.DataFrame(rows)

# =============================================================================
# 17. training_config.json
# =============================================================================
def build_training_config(ctx, bundle):
    try:
        slices_version = importlib.metadata.version("torch-slices")
    except Exception:
        slices_version = "unknown"

    return {
        "task": "MMSE regression",
        "downstream_model": "SLiCE",
        "slice_variant": (
            "BD-SLiCE"
            if (not SLICE_DIAGONAL_DENSE and SLICE_BLOCK_SIZE > 1)
            else "other_SLICE_variant"
        ),
        "context": {
            "dataset": ctx.dataset,
            "source_mode": ctx.source_mode,
            "feature_model": ctx.feature_model_tag,
            "selection": ctx.feature_selection_tag,
            "file_tag": ctx.feature_file_tag,
            "feature_dir": ctx.feature_data_dir,
            "output_dir": ctx.output_dir,
        },
        "split_protocol": {
            "train": "official_train",
            "val_monitor": "official_test",
            "official_test_used_for_selection": False,
            "early_stopping_monitor": "train_loss",
            "restore_best": EARLY_STOPPING_RESTORE_BEST,
            "selection_policy": EPOCH_SELECTION_POLICY,
            "numerical_failure_policy": "last_finite_epoch_after_numerical_nonfinite",
            "monitor_semantics": "official_test_monitor",
        },
        "dataset_info": {
            "X_train_shape": list(bundle.X_train.shape),
            "X_val_shape": list(bundle.X_val.shape),
            "data_dim": int(bundle.data_dim),
            "num_train_audio": int(len(np.unique(bundle.idx_train_audio))),
            "num_val_audio": int(len(np.unique(bundle.idx_val_audio))),
        },
        "preprocess": {
            "FEATURE_STANDARDIZE": FEATURE_STANDARDIZE,
            "INCLUDE_TIME": INCLUDE_TIME,
            "TARGET_STANDARDIZE": TARGET_STANDARDIZE,
            "target_normalizer": bundle.target_normalizer,
        },
        "slice_config": {
            "hidden_dim": SLICE_HIDDEN_DIM,
            "num_layers": SLICE_NUM_LAYERS,
            "block_size": SLICE_BLOCK_SIZE,
            "diagonal_dense": SLICE_DIAGONAL_DENSE,
            "bias": SLICE_BIAS,
            "init_std": SLICE_INIT_STD,
            "scale": SLICE_SCALE,
            "input_dependent_init": SLICE_INPUT_DEPENDENT_INIT,
            "use_parallel_requested": SLICE_USE_PARALLEL,
            "chunk_size": SLICE_CHUNK_SIZE,
            "path_mode": SLICE_PATH_MODE,
            "transition_mode": SLICE_TRANSITION_MODE,
            "norm_type": SLICE_NORM_TYPE,
            "prenorm": SLICE_PRENORM,
            "second_norm": SLICE_SECOND_NORM,
            "ff_style": SLICE_FF_STYLE,
            "ff_activation": SLICE_FF_ACTIVATION,
            "ff_mult": SLICE_FF_MULT,
            "dropout_rate": SLICE_DROPOUT_RATE,
            "dropout_position": SLICE_DROPOUT_POSITION,
            "readout_pool": SLICE_READOUT_POOL,
            "mlp_hidden_dim": SLICE_MLP_HIDDEN_DIM,
        },
        "training": {
            "optimizer": SLICE_OPTIMIZER_NAME,
            "lr": LR_SLICE,

            # Adam 中不使用 weight decay。
            "weight_decay": 0.0,

            # 本版保留 gradient clipping。
            "grad_clip_norm": SLICE_GRAD_CLIP_NORM,

            "batch_size": SLICE_BATCH_SIZE,

            # 这是最大 epoch，不是固定训练满。
            "max_epochs": SLICE_TARGET_EPOCHS,

            "warmup_epochs": SLICE_WARMUP_EPOCHS,
            "cosine_alpha": SLICE_COSINE_ALPHA,

            "early_stopping": USE_EARLY_STOPPING,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "patience": EARLY_STOPPING_STEPS,
            "min_delta": EARLY_STOPPING_MIN_DELTA,
            "restore_best": EARLY_STOPPING_RESTORE_BEST,
            "numerical_guard": SLICE_NUMERICAL_GUARD,
        },
        "process2_weighting": {
            "enabled": USE_PROCESS2_MMSE_BIN_WEIGHT,
            "beta": PROCESS2_MMSE_BIN_WEIGHT_BETA,
            "max_weight": PROCESS2_MMSE_BIN_WEIGHT_MAX,
            "mixed_loss_alpha": PROCESS2_MIXED_LOSS_ALPHA,
            "actual_weight_info": bundle.process2_weight_info,
        },
        "audio_aggregation": {
            "primary": AUDIO_AGG_PRIMARY_METHOD,
            "trim_ratio": AUDIO_AGG_TRIM_RATIO,
        },
        "seeds": {
            "RUN_SINGLE_SEED_400": RUN_SINGLE_SEED_400,
            "RANDOM_SEED": RANDOM_SEED,
            "RUN_MULTI_SEEDS": RUN_MULTI_SEEDS,
            "MULTI_SEEDS": MULTI_SEEDS,
        },
        "environment": {
            "python": sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "torch_slices_version": slices_version,
            "cuda_available": torch.cuda.is_available(),
            "configured_GPU_ID": GPU_ID,
            "gpu_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
    }

# =============================================================================
# 18. 打印配置
# =============================================================================
def print_active_config(ctx, bundle):
    print("\n" + "=" * 96)
    print("[SLiCE ACTIVE CONFIG]")
    print("=" * 96)

    print("dataset                  =", ctx.dataset)
    print("source_mode              =", ctx.source_mode)
    print("feature_model            =", ctx.feature_model_tag)
    print("selection                =", ctx.feature_selection_tag)
    print("feature_dir              =", ctx.feature_data_dir)
    print("output_dir               =", ctx.output_dir)

    print("X_train                  =", bundle.X_train.shape)
    print("X_val/official_test_monitor =", bundle.X_val.shape)
    print("[protocol] official_test 仅监测和报告；数值异常时使用上一完整有限 epoch。")
    print("input_dim                =", bundle.data_dim)

    print("SLICE_HIDDEN_DIM         =", SLICE_HIDDEN_DIM)
    print("SLICE_NUM_LAYERS         =", SLICE_NUM_LAYERS)
    print("SLICE_BLOCK_SIZE         =", SLICE_BLOCK_SIZE)
    print("SLICE_DIAGONAL_DENSE     =", SLICE_DIAGONAL_DENSE)

    print("SLICE_USE_PARALLEL       =", SLICE_USE_PARALLEL)
    print("SLICE_CHUNK_SIZE         =", SLICE_CHUNK_SIZE)
    print("SLICE_PATH_MODE          =", SLICE_PATH_MODE)
    print("SLICE_TRANSITION_MODE    =", SLICE_TRANSITION_MODE)

    print("SLICE_FF_MULT            =", SLICE_FF_MULT)
    print("SLICE_DROPOUT_RATE       =", SLICE_DROPOUT_RATE)
    print("SLICE_READOUT_POOL       =", SLICE_READOUT_POOL)

    print("LR_SLICE                 =", LR_SLICE)
    print("SLICE_OPTIMIZER_NAME     =", SLICE_OPTIMIZER_NAME)
    print("SLICE_WEIGHT_DECAY       =", SLICE_WEIGHT_DECAY, "(disabled)")
    print("SLICE_GRAD_CLIP_NORM     =", SLICE_GRAD_CLIP_NORM, "(enabled)")
    print("BATCH_SIZE               =", SLICE_BATCH_SIZE)
    print("TARGET_EPOCHS            =", SLICE_TARGET_EPOCHS)
    print("WARMUP_EPOCHS            =", SLICE_WARMUP_EPOCHS)
    print("COSINE_ALPHA             =", SLICE_COSINE_ALPHA)

    print("USE_EARLY_STOPPING       =", USE_EARLY_STOPPING)
    print("EARLY_STOPPING_STEPS     =", EARLY_STOPPING_STEPS)
    print("EARLY_STOPPING_MIN_DELTA =", EARLY_STOPPING_MIN_DELTA)
    print("RESTORE_BEST             =", EARLY_STOPPING_RESTORE_BEST)

    print("PROCESS2_WEIGHT          =", USE_PROCESS2_MMSE_BIN_WEIGHT)
    print("PROCESS2_MIXED_ALPHA     =", PROCESS2_MIXED_LOSS_ALPHA)

    print("GPU_ID                   =", GPU_ID)
    print("torch                    =", torch.__version__)
    print("torch.version.cuda       =", torch.version.cuda)
    print("CUDA available           =", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name                 =", torch.cuda.get_device_name(0))

    print("=" * 96)

# =============================================================================
# 19. 一个 context 的完整运行
# =============================================================================
def run_context(ctx):
    os.makedirs(ctx.output_dir, exist_ok=True)

    bundle = load_feature_bundle(ctx)
    print_active_config(ctx, bundle)

    save_json(
        build_training_config(ctx, bundle),
        os.path.join(ctx.output_dir, "training_config.json"),
    )

    save_json(
        bundle.process2_weight_info,
        os.path.join(ctx.output_dir, "process2_mmse_bin_weight_info.json"),
    )

    if SLICE_RUN_PARALLEL_RECURRENT_CHECK:
        check_parallel_recurrent(bundle, get_device(), RANDOM_SEED)

    seeds = []

    if RUN_SINGLE_SEED_400:
        seeds.append(int(RANDOM_SEED))

    if RUN_MULTI_SEEDS:
        for seed in MULTI_SEEDS:
            if int(seed) not in seeds:
                seeds.append(int(seed))

    if not seeds:
        raise ValueError("没有任何 seed 被启用。")

    rows = []
    val_audio_dfs = []

    for seed in seeds:
        result = run_one_seed(ctx, bundle, seed)
        rows.append(result["metrics_row"])
        val_audio_dfs.append(result["val_audio_df"])

    metrics_df = pd.DataFrame(rows)

    metrics_path = os.path.join(ctx.output_dir, "slice_seed_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    summary_df = build_summary_df(metrics_df)
    summary_path = os.path.join(ctx.output_dir, "slice_seed_summary_mean_std_sem.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    if val_audio_dfs:
        pd.concat(val_audio_dfs, ignore_index=True).to_csv(
            os.path.join(
                ctx.output_dir,
                "val_official_test_audio_predictions_all_seeds.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

    print("\n[CONTEXT FINISHED]")
    print("metrics =", metrics_path)
    print("summary =", summary_path)

    if "val_audio_rmse" in metrics_df.columns:
        s = mean_std_sem(metrics_df["val_audio_rmse"].to_numpy())
        print(
            f"official_test_monitor_audio_rmse mean={s['mean']:.6f}, "
            f"std={s['std']:.6f}, sem={s['sem']:.6f}, n={s['count']}"
        )

# =============================================================================
# 20. main
# =============================================================================
def main():
    print("\n" + "=" * 96)
    print("SLiCE -> Whisper / M-P-TAPT -> MMSE Regression")
    print("=" * 96)

    print("CONFIG_ROOT_DIR        =", CONFIG_ROOT_DIR)
    print("CONFIG_FEATURE_ROOT     =", CONFIG_FEATURE_ROOT)
    print("CONFIG_OUTPUT_ROOT      =", CONFIG_OUTPUT_ROOT)
    print("CUDA_VISIBLE_DEVICES    =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.__version__       =", torch.__version__)
    print("torch.version.cuda      =", torch.version.cuda)
    print("torch.cuda.is_available =", torch.cuda.is_available())

    if (
        not SLICE_DIAGONAL_DENSE
        and SLICE_HIDDEN_DIM % SLICE_BLOCK_SIZE != 0
    ):
        raise ValueError(
            "BD-SLiCE 配置非法：SLICE_HIDDEN_DIM 必须能被 SLICE_BLOCK_SIZE 整除。"
        )

    if SLICE_BLOCK_SIZE > 1 and not SLICE_DIAGONAL_DENSE:
        print("[MODEL VARIANT] BD-SLiCE")
    elif SLICE_BLOCK_SIZE == 1 and not SLICE_DIAGONAL_DENSE:
        print("[MODEL VARIANT] Diagonal SLiCE")
    else:
        print("[MODEL VARIANT] D-DE-SLiCE")

    contexts = build_all_contexts()
    print("[TOTAL CONTEXTS] =", len(contexts))

    for i, ctx in enumerate(contexts, start=1):
        print("\n" + "*" * 96)
        print(f"[CONTEXT {i}/{len(contexts)}]")
        print("*" * 96)
        run_context(ctx)

    print("\nALL SLiCE EXPERIMENTS FINISHED.")


if __name__ == "__main__":
    main()
