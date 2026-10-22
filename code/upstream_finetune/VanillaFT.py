# -*- coding: utf-8 -*-
"""Whisper MMSE 监督微调与单层特征导出。
精简版保留原参数、随机数调用顺序、训练/损失/评估/导出及保存路径。
Pitt 使用 original_test_subjects_holdout：原 test 中存在有效 visit 的受试者，
其全部有效 visits 归入 final test；其余受试者归入 final train。
当前默认：Pitt、第 12 层、解冻 1–12 层、segment MAE、3 个 epoch、last checkpoint。

conda activate audiojax_gpu
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune/ft_vanilla/VanillaFT.py
"""


import os
import re
import json
import math
import copy
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "5")
os.environ["PYTHONHASHSEED"] = "400"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

from transformers import WhisperConfig, WhisperModel, WhisperProcessor

SEED = 400

def set_seed(seed: int = 400) -> None:
    """统一设置 Python / NumPy / PyTorch 的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE =", DEVICE)
print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
print("torch version =", torch.__version__)

# 数据集：ADReSS-IS2020 / ADReSSo21 / Pitt / PROCESS-2；默认仅 Pitt。
DATASET_KEYS = ["ADReSS-IS2020"]

DATASET_KEY = DATASET_KEYS[0]

# Pitt 受试者互斥划分及原有严格计数校验。
PITT_ENABLE_SUBJECT_DISJOINT_SPLIT = True
PITT_SPLIT_POLICY = "original_test_subjects_holdout"
PITT_STRICT_COUNT_CHECK = True
PITT_EXPECTED_CANDIDATE_VISITS = 477
PITT_EXPECTED_VALID_VISITS = 398
PITT_EXPECTED_UNIQUE_SUBJECTS = 252
PITT_EXPECTED_VALID_TEST_SUBJECTS = 68

# 模型与保存标签：保持原值及路径命名规则。
MODEL_NAME = "whisper_small"  # 真实 Whisper base model 名，不要改

FEATURE_MODEL_SAVE_TAG = "vanilla_lr1e-5_epo3_repeat"

BASE_WHISPER_MODEL = os.path.join("/home/juanyu/whisper_lru_3", MODEL_NAME)
LOCAL_FILES_ONLY = True

# 导出层；空列表表示遍历所有 encoder 层。
SINGLE_LAYERS = [12]

RUN_PROTOCOL_TAG = "official_train_test"

# 从导出层向前连续解冻的层数；按原顺序批量运行。
UNFREEZE_ENCODER_LAYERS_LIST = [7]

UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# 旧 manifest 兼容字段；当前入口不执行 fold 划分。
FOLD_IDS = [1]

# 项目、特征、模型与元数据根目录。
PROJECT_ROOT = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT, "feature_tensor")
MODEL_ROOT_DIR = os.path.join(PROJECT_ROOT, "models")
SPLIT_ROOT_DIR = os.path.join(PROJECT_ROOT, "split")

FINETUNED_WHISPER_DIR = ""
FEATURE_SAVE_PATH = ""
TRAIN_VAL_SPLIT_PATH = ""  # 旧协议兼容占位；当前不用于训练划分
LOCAL_FEATURE_SPLIT_SAVE_PATH = ""     # 导出阶段写回当前特征目录的本地 metadata json
ACTIVE_FOLD_ID = None
ACTIVE_LAYER_NAME = ""

# 音频采样率、切段长度、重叠率与波形归一化。
SAMPLING_RATE = 16000
SEGMENT_SEC = 30
SEGMENT_LENGTH = SEGMENT_SEC * SAMPLING_RATE
OVERLAP_RATE = 0.3
MAX_SEGMENTS_PER_AUDIO = 0   # 0 表示不限制
NORMALIZE_AUDIO = True       # 是否先做波形峰值归一化

VAL_RATIO = 0.25

# 训练参数；完整音频 batch 内保留原 segment 顺序。
BATCH_SIZE = 8
AUDIO_BATCH_SIZE = 2

# 阶段配置保留；默认 Stage 1 为 0 轮、Stage 2 为 3 轮。
TEMPORAL_FINETUNE_MODE = "two_stage"
TEMPORAL_STAGE1_EPOCHS = 0
TEMPORAL_STAGE1_HEAD_LR = 0.0001

TEMPORAL_STAGE2_EPOCHS = 3
NUM_EPOCHS = int(TEMPORAL_STAGE1_EPOCHS + TEMPORAL_STAGE2_EPOCHS)
TEMPORAL_STAGE2_WHISPER_LR = 1e-5
TEMPORAL_STAGE2_HEAD_LR = 3e-5

LR = TEMPORAL_STAGE2_WHISPER_LR

WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0

# 上游仅训练 final train / official_train；固定轮数后使用 last checkpoint。
WHISPER_FINETUNE_USE_VAL = False
WHISPER_FINETUNE_CHECKPOINT_SELECTION = "last"
WHISPER_FINETUNE_PATH_TAG = ""  # 仅用于 checkpoint 目录名

TEMPORAL_HEAD_TYPE = "pool_linear"   # pool_linear
TEMPORAL_HIDDEN_DIM = 128            # pool_linear 不使用，可保留兼容路径命名
TEMPORAL_NUM_LAYERS = 1              # pool_linear 不使用
TEMPORAL_DROPOUT = 0.2

# 监督粒度：segment / segment_pred_mean / segment_audio_balanced / audio。
UPSTREAM_MMSE_SUPERVISION_LEVEL = "segment"

TEMPORAL_SEGMENT_POOL = "mean"

TEMPORAL_DELTA_CONSISTENCY_WEIGHT = 0.0

# 目标标准化仅拟合训练音频；报告指标反归一化到 MMSE 原尺度。
TEMPORAL_TARGET_NORMALIZE = True
TEMPORAL_TARGET_NORMALIZE_EPS = 1e-6

RESOLVED_UNFREEZE_ENCODER_LAYERS = []

POOLING_TYPE = "mean"
ATTENTION_HIDDEN_DIM = 128
REGRESSOR_DROPOUT = 0.2
CONSISTENCY_WEIGHT = 0.0
CONSISTENCY_LAYER = 0  # <=0 时使用 SINGLE_LAYER
CONSISTENCY_LOSS_TYPE = "mse"
AUDIO_FEATURE_CONSISTENCY_WEIGHT = 0.0
AUDIO_PREDICTION_CONSISTENCY_WEIGHT = 0.0

# 回归损失：mse / mae / smooth_l1。
REGRESSION_LOSS_TYPE = "mae"
SMOOTH_L1_BETA = 1.0

ENABLE_TUNING_TAG_IN_PATH = False
ENABLE_TRAINING_HPARAM_TAG_IN_PATH = False

# 路径附加参数：[] 或 none 不附加；all 或 None 全部附加；也可指定参数名列表。
SAVE_PARAMS_IN_PATH = []

FEATURE_PREFIX = "whisper"
SINGLE_LAYER = 1

# 导出特征归一化：none / layernorm / std / ln_post。
NORM_FOR_SINGLE = "ln_post"   # 可选："none" / "layernorm" / "std" / "ln_post"

PROTOTYPE_SMOOTHING_ALPHA = 1

# 导出开关；导出仅前向，不更新 Whisper 参数。
SKIP_FEATURE_EXPORT = False

# 仅重导出开关及已有 Whisper checkpoint 路径。
REEXPORT_ONLY = False

REEXPORT_FROM_FINETUNED_DIR = ""

def make_default_split_file_tag(feature_prefix: str, single_layer: int, overlap_rate: float, segment_sec: int) -> str:
    ov_str = f"{float(overlap_rate):.1f}".rstrip("0").rstrip(".") if float(overlap_rate) % 1 != 0 else str(int(overlap_rate))
    if "." not in ov_str:
        ov_str = f"{ov_str}.0"
    return f"{feature_prefix}_single{int(single_layer)}_{ov_str}_{int(segment_sec)}s"

def get_whisper_encoder_layer_count(whisper_model_name_or_path: str, local_files_only: bool = True) -> int:
    """读取 Whisper config，返回 encoder 总层数。"""
    cfg = WhisperConfig.from_pretrained(
        whisper_model_name_or_path,
        local_files_only=local_files_only,
    )
    num_layers = getattr(cfg, "encoder_layers", None)
    if num_layers is None:
        raise AttributeError("WhisperConfig 中缺少 encoder_layers，无法自动解析层数")
    return int(num_layers)

def resolve_single_layers(single_layers: Sequence[int], whisper_model_name_or_path: str, local_files_only: bool) -> List[int]:
    """解析本次批量实验要跑的 SINGLE_LAYER 列表。"""
    cleaned = sorted({int(x) for x in list(single_layers) if int(x) > 0})
    if len(cleaned) > 0:
        return cleaned

    total_layers = get_whisper_encoder_layer_count(
        whisper_model_name_or_path=whisper_model_name_or_path,
        local_files_only=local_files_only,
    )
    return list(range(1, total_layers + 1))

def resolve_unfreeze_layers_for_single_layer(
    single_layer: int,
    total_layers: int,
    unfreeze_layer_count: int,
) -> List[int]:
    """根据 SINGLE_LAYER 和“解冻层数”自动得到当前运行实际使用的解冻层列表。"""
    single_layer = int(single_layer)
    total_layers = int(total_layers)
    unfreeze_layer_count = int(unfreeze_layer_count)

    if not (1 <= single_layer <= total_layers):
        raise ValueError(
            f"single_layer={single_layer} 超出范围 [1, {total_layers}]"
        )

    if unfreeze_layer_count == 0:
        return []

    if unfreeze_layer_count < 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS 必须 >= 0")

    start_layer = max(1, single_layer - unfreeze_layer_count + 1)
    end_layer = min(single_layer, total_layers)

    resolved = list(range(start_layer, end_layer + 1))
    if len(resolved) == 0:
        raise ValueError(
            f"single_layer={single_layer}, unfreeze_layer_count={unfreeze_layer_count} 无法解析有效的解冻层"
        )

    return resolved

def _format_float_for_path(x: float) -> str:
    """把浮点数转成适合路径名的紧凑字符串："""
    x = float(x)
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    if "." not in s:
        s = f"{s}.0"
    return s

PATH_PARAM_DEFAULT_ORDER = [
    "TEMPORAL_HEAD_TYPE",
    "TEMPORAL_HIDDEN_DIM",
    "TEMPORAL_DROPOUT",

    "TEMPORAL_FINETUNE_MODE",
    "TEMPORAL_STAGE1_EPOCHS",
    "TEMPORAL_STAGE2_EPOCHS",
    "TEMPORAL_STAGE1_HEAD_LR",
    "TEMPORAL_STAGE2_WHISPER_LR",
    "TEMPORAL_STAGE2_HEAD_LR",

    "CONSISTENCY_WEIGHT",
    "TEMPORAL_DELTA_CONSISTENCY_WEIGHT",
    "TEMPORAL_TARGET_NORMALIZE",
    "AUDIO_FEATURE_CONSISTENCY_WEIGHT",
    "AUDIO_PREDICTION_CONSISTENCY_WEIGHT",

    "AUDIO_BATCH_SIZE",
]

PATH_PARAM_ALIASES = {
    "CONS": "CONSISTENCY_WEIGHT",
    "CONSISTENCY": "CONSISTENCY_WEIGHT",
    "DELTA": "TEMPORAL_DELTA_CONSISTENCY_WEIGHT",
    "TNORM": "TEMPORAL_TARGET_NORMALIZE",
    "TARGET_NORMALIZE": "TEMPORAL_TARGET_NORMALIZE",
    "LRW2": "TEMPORAL_STAGE2_WHISPER_LR",
    "LRH2": "TEMPORAL_STAGE2_HEAD_LR",
    "LRH1": "TEMPORAL_STAGE1_HEAD_LR",
    "S1E": "TEMPORAL_STAGE1_EPOCHS",
    "S2E": "TEMPORAL_STAGE2_EPOCHS",
    "ABS": "AUDIO_BATCH_SIZE",
}

def resolve_save_params_in_path() -> set:
    """解析 SAVE_PARAMS_IN_PATH。"""
    raw = globals().get("SAVE_PARAMS_IN_PATH", None)

    all_params = set(PATH_PARAM_DEFAULT_ORDER)

    if raw is None:
        return all_params

    if isinstance(raw, str):
        raw_clean = raw.strip()
        if raw_clean == "" or raw_clean.lower() == "all":
            return all_params
        if raw_clean.lower() in {"none", "null", "false", "off"}:
            return set()
        raw = [raw_clean]

    cleaned = set()
    for item in list(raw):
        name = str(item).strip().upper()
        if name == "":
            continue
        name = PATH_PARAM_ALIASES.get(name, name)
        cleaned.add(name)

    known = set(PATH_PARAM_DEFAULT_ORDER)
    unknown = sorted([x for x in cleaned if x not in known])
    if len(unknown) > 0:
        print(
            "[WARN] SAVE_PARAMS_IN_PATH 中存在当前未支持的参数名，已忽略：",
            unknown,
        )

    return cleaned.intersection(known)

def path_wants_param(param_name: str, save_params: Optional[set] = None) -> bool:
    """判断某个参数是否需要写入路径名。"""
    if save_params is None:
        save_params = resolve_save_params_in_path()
    return str(param_name).strip().upper() in save_params

def path_wants_any(param_names: Sequence[str], save_params: Optional[set] = None) -> bool:
    """判断一组参数中是否至少有一个需要写入路径名。"""
    if save_params is None:
        save_params = resolve_save_params_in_path()
    return any(str(x).strip().upper() in save_params for x in param_names)

def build_layer_name(single_layer: int, unfreeze_layer_count: int) -> str:
    """构造当前 single_layer / unfreeze / temporal 微调策略对应的目录名。"""
    save_params = resolve_save_params_in_path()

    base_name = f"single{int(single_layer)}_unfree{int(unfreeze_layer_count)}"
    parts = [base_name]

    if path_wants_any(
        ["TEMPORAL_HEAD_TYPE", "TEMPORAL_HIDDEN_DIM", "TEMPORAL_DROPOUT"],
        save_params,
    ):
        temporal_type = str(globals().get("TEMPORAL_HEAD_TYPE", "none")).strip()
        temporal_hidden = int(globals().get("TEMPORAL_HIDDEN_DIM", 0))
        temporal_dropout = float(globals().get("TEMPORAL_DROPOUT", 0.0))

        if temporal_type not in ["", "none", "None", "null", "Null"]:
            parts.append(
                f"temporal_{temporal_type}_h{temporal_hidden}_d{_format_float_for_path(temporal_dropout)}"
            )

    finetune_mode = str(
        globals().get("TEMPORAL_FINETUNE_MODE", "single_stage")
    ).strip().lower()

    if path_wants_any(
        ["TEMPORAL_FINETUNE_MODE", "TEMPORAL_STAGE1_EPOCHS", "TEMPORAL_STAGE2_EPOCHS"],
        save_params,
    ):
        if finetune_mode == "two_stage":
            s1e = int(globals().get("TEMPORAL_STAGE1_EPOCHS", 0))
            s2e = int(globals().get("TEMPORAL_STAGE2_EPOCHS", 0))
            parts.append(f"twostage_s1e{s1e}_s2e{s2e}")
        else:
            parts.append("single_stage")

    if path_wants_param("TEMPORAL_STAGE1_HEAD_LR", save_params):
        lrh1 = _format_float_for_path(globals().get("TEMPORAL_STAGE1_HEAD_LR", 0.0))
        parts.append(f"lrh1_{lrh1}")

    if path_wants_param("TEMPORAL_STAGE2_WHISPER_LR", save_params):
        lrw2 = _format_float_for_path(globals().get("TEMPORAL_STAGE2_WHISPER_LR", 0.0))
        parts.append(f"lrw2_{lrw2}")

    if path_wants_param("TEMPORAL_STAGE2_HEAD_LR", save_params):
        lrh2 = _format_float_for_path(globals().get("TEMPORAL_STAGE2_HEAD_LR", 0.0))
        parts.append(f"lrh2_{lrh2}")

    if path_wants_param("CONSISTENCY_WEIGHT", save_params):
        cons_str = _format_float_for_path(globals().get("CONSISTENCY_WEIGHT", 0.0))
        parts.append(f"cons{cons_str}")

    if path_wants_param("TEMPORAL_DELTA_CONSISTENCY_WEIGHT", save_params):
        delta_str = _format_float_for_path(
            globals().get("TEMPORAL_DELTA_CONSISTENCY_WEIGHT", 0.0)
        )
        parts.append(f"delta{delta_str}")

    if path_wants_param("TEMPORAL_TARGET_NORMALIZE", save_params):
        tnorm_tag = "tnorm1" if bool(globals().get("TEMPORAL_TARGET_NORMALIZE", False)) else "tnorm0"
        parts.append(tnorm_tag)

    if path_wants_param("AUDIO_FEATURE_CONSISTENCY_WEIGHT", save_params):
        feat_str = _format_float_for_path(globals().get("AUDIO_FEATURE_CONSISTENCY_WEIGHT", 0.0))
        parts.append(f"feat{feat_str}")

    if path_wants_param("AUDIO_PREDICTION_CONSISTENCY_WEIGHT", save_params):
        pred_str = _format_float_for_path(globals().get("AUDIO_PREDICTION_CONSISTENCY_WEIGHT", 0.0))
        parts.append(f"pred{pred_str}")

    if path_wants_param("AUDIO_BATCH_SIZE", save_params):
        audio_bs = int(globals().get("AUDIO_BATCH_SIZE", BATCH_SIZE))
        parts.append(f"abs{audio_bs}")

    return "_".join(parts)

    if path_wants_any(
        ["TEMPORAL_HEAD_TYPE", "TEMPORAL_HIDDEN_DIM", "TEMPORAL_DROPOUT"],
        save_params,
    ):
        temporal_type = str(globals().get("TEMPORAL_HEAD_TYPE", "none")).strip()
        temporal_hidden = int(globals().get("TEMPORAL_HIDDEN_DIM", 0))
        temporal_dropout = float(globals().get("TEMPORAL_DROPOUT", 0.0))

        if temporal_type not in ["", "none", "None", "null", "Null"]:
            parts.append(
                f"temporal_{temporal_type}_h{temporal_hidden}_d{_format_float_for_path(temporal_dropout)}"
            )

    finetune_mode = str(globals().get("TEMPORAL_FINETUNE_MODE", "single_stage")).strip().lower()

    if path_wants_any(
        ["TEMPORAL_FINETUNE_MODE", "TEMPORAL_STAGE1_EPOCHS", "TEMPORAL_STAGE2_EPOCHS"],
        save_params,
    ):
        if finetune_mode == "two_stage":
            s1e = int(globals().get("TEMPORAL_STAGE1_EPOCHS", 0))
            s2e = int(globals().get("TEMPORAL_STAGE2_EPOCHS", 0))
            parts.append(f"twostage_s1e{s1e}_s2e{s2e}")
        else:
            parts.append("single_stage")

    if path_wants_param("TEMPORAL_STAGE1_HEAD_LR", save_params):
        lrh1 = _format_float_for_path(globals().get("TEMPORAL_STAGE1_HEAD_LR", 0.0))
        parts.append(f"lrh1_{lrh1}")

    if path_wants_param("TEMPORAL_STAGE2_WHISPER_LR", save_params):
        lrw2 = _format_float_for_path(globals().get("TEMPORAL_STAGE2_WHISPER_LR", 0.0))
        parts.append(f"lrw2_{lrw2}")

    if path_wants_param("TEMPORAL_STAGE2_HEAD_LR", save_params):
        lrh2 = _format_float_for_path(globals().get("TEMPORAL_STAGE2_HEAD_LR", 0.0))
        parts.append(f"lrh2_{lrh2}")

    if bool(ENABLE_TUNING_TAG_IN_PATH):
        if path_wants_param("CONSISTENCY_WEIGHT", save_params):
            cons_str = _format_float_for_path(CONSISTENCY_WEIGHT)
            parts.append(f"cons{cons_str}")

        if path_wants_param("TEMPORAL_DELTA_CONSISTENCY_WEIGHT", save_params):
            delta_str = _format_float_for_path(
                globals().get("TEMPORAL_DELTA_CONSISTENCY_WEIGHT", 0.0)
            )
            parts.append(f"delta{delta_str}")

        if path_wants_param("TEMPORAL_TARGET_NORMALIZE", save_params):
            tnorm_tag = "tnorm1" if bool(globals().get("TEMPORAL_TARGET_NORMALIZE", False)) else "tnorm0"
            parts.append(tnorm_tag)

        if path_wants_param("AUDIO_FEATURE_CONSISTENCY_WEIGHT", save_params):
            feat_str = _format_float_for_path(AUDIO_FEATURE_CONSISTENCY_WEIGHT)
            parts.append(f"feat{feat_str}")

        if path_wants_param("AUDIO_PREDICTION_CONSISTENCY_WEIGHT", save_params):
            pred_str = _format_float_for_path(AUDIO_PREDICTION_CONSISTENCY_WEIGHT)
            parts.append(f"pred{pred_str}")

    if bool(ENABLE_TRAINING_HPARAM_TAG_IN_PATH):
        if path_wants_param("AUDIO_BATCH_SIZE", save_params):
            audio_bs = int(globals().get("AUDIO_BATCH_SIZE", BATCH_SIZE))
            parts.append(f"abs{audio_bs}")

    return "_".join(parts)

def build_feature_save_path(
    dataset_key: str,
    model_name: str,
    layer_name: str,
    fold_id: Optional[int] = None,
) -> str:
    """微调特征保存目录。"""
    return os.path.join(
        FEATURE_ROOT_DIR,
        dataset_key,
        f"ft_{model_name}",
        layer_name,
    )

def build_feature_local_split_path(
    dataset_key: str,
    model_name: str,
    layer_name: str,
    fold_id: Optional[int],
    feature_prefix: str,
    single_layer: int,
    overlap_rate: float,
    segment_sec: int,
) -> str:
    """当前特征目录下保存 metadata 的路径。"""
    feature_save_path = build_feature_save_path(
        dataset_key=dataset_key,
        model_name=model_name,
        layer_name=layer_name,
        fold_id=None,
    )
    file_tag = make_default_split_file_tag(
        feature_prefix=feature_prefix,
        single_layer=single_layer,
        overlap_rate=overlap_rate,
        segment_sec=segment_sec,
    )
    return os.path.join(feature_save_path, f"official_train_test_metadata_{file_tag}.json")

def build_finetuned_whisper_dir(
    dataset_key: str,
    model_name: str,
    layer_name: str,
    fold_id: Optional[int] = None,
) -> str:
    """微调后的 Whisper checkpoint 保存目录。"""
    return os.path.join(
        MODEL_ROOT_DIR,
        dataset_key,
        f"ft_{model_name}",
        layer_name,
        RUN_PROTOCOL_TAG,
    )

def apply_run_context(single_layer: int, fold_id: Optional[int] = None) -> Dict[str, object]:
    """根据当前 single_layer 刷新全局运行上下文。"""
    global SINGLE_LAYER
    global RESOLVED_UNFREEZE_ENCODER_LAYERS
    global TRAIN_VAL_SPLIT_PATH
    global TRAIN_VAL_SPLIT_PATH_RESOLVED
    global DEFAULT_SPLIT_FILE_TAG
    global DEFAULT_TRAIN_VAL_SPLIT_PATH
    global FEATURE_SAVE_PATH
    global FINETUNED_WHISPER_DIR
    global NOTEBOOK_LOG_ROOT
    global ACTIVE_FOLD_ID
    global ACTIVE_LAYER_NAME
    global LOCAL_FEATURE_SPLIT_SAVE_PATH

    total_layers = get_whisper_encoder_layer_count(
        whisper_model_name_or_path=BASE_WHISPER_MODEL,
        local_files_only=LOCAL_FILES_ONLY,
    )

    SINGLE_LAYER = int(single_layer)
    RESOLVED_UNFREEZE_ENCODER_LAYERS = resolve_unfreeze_layers_for_single_layer(
        SINGLE_LAYER,
        total_layers,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
    )

    ACTIVE_FOLD_ID = None

    ACTIVE_LAYER_NAME = build_layer_name(SINGLE_LAYER, UNFREEZE_ENCODER_LAYERS)

    print("[PATH PARAMS] SAVE_PARAMS_IN_PATH =", globals().get("SAVE_PARAMS_IN_PATH", None))
    print("[PATH PARAMS] resolved =", sorted(resolve_save_params_in_path(), key=lambda x: PATH_PARAM_DEFAULT_ORDER.index(x)))
    print("[PATH PARAMS] TEMPORAL_STAGE2_WHISPER_LR =", globals().get("TEMPORAL_STAGE2_WHISPER_LR", None))
    print("[PATH PARAMS] CONSISTENCY_WEIGHT =", globals().get("CONSISTENCY_WEIGHT", None))
    print("[PATH PARAMS] ACTIVE_LAYER_NAME =", ACTIVE_LAYER_NAME)
    print("[PROTOCOL] RUN_PROTOCOL_TAG =", RUN_PROTOCOL_TAG)
    print("[PROTOCOL] fold is disabled; official_train is used as the full finetune train set.")

    DEFAULT_SPLIT_FILE_TAG = make_default_split_file_tag(
        FEATURE_PREFIX,
        SINGLE_LAYER,
        OVERLAP_RATE,
        SEGMENT_SEC,
    )

    DEFAULT_TRAIN_VAL_SPLIT_PATH = None
    TRAIN_VAL_SPLIT_PATH = None
    TRAIN_VAL_SPLIT_PATH_RESOLVED = None

    FEATURE_SAVE_PATH = build_feature_save_path(
        dataset_key=DATASET_KEY,
        model_name=FEATURE_MODEL_SAVE_TAG,
        layer_name=ACTIVE_LAYER_NAME,
        fold_id=None,
    )

    FINETUNED_WHISPER_DIR = build_finetuned_whisper_dir(
        dataset_key=DATASET_KEY,
        model_name=FEATURE_MODEL_SAVE_TAG,
        layer_name=ACTIVE_LAYER_NAME,
        fold_id=None,
    )

    LOCAL_FEATURE_SPLIT_SAVE_PATH = build_feature_local_split_path(
        dataset_key=DATASET_KEY,
        model_name=FEATURE_MODEL_SAVE_TAG,
        layer_name=ACTIVE_LAYER_NAME,
        fold_id=None,
        feature_prefix=FEATURE_PREFIX,
        single_layer=SINGLE_LAYER,
        overlap_rate=OVERLAP_RATE,
        segment_sec=SEGMENT_SEC,
    )

    NOTEBOOK_LOG_ROOT = os.path.join(FEATURE_SAVE_PATH, "notebook_logs")

    os.makedirs(FEATURE_SAVE_PATH, exist_ok=True)
    os.makedirs(FINETUNED_WHISPER_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOCAL_FEATURE_SPLIT_SAVE_PATH), exist_ok=True)

    print("[PATH] FEATURE_SAVE_PATH =", FEATURE_SAVE_PATH)
    print("[PATH] FINETUNED_WHISPER_DIR =", FINETUNED_WHISPER_DIR)
    print("[PATH] TRAIN_VAL_SPLIT_PATH_RESOLVED =", TRAIN_VAL_SPLIT_PATH_RESOLVED)
    print("[PATH] LOCAL_FEATURE_SPLIT_SAVE_PATH =", LOCAL_FEATURE_SPLIT_SAVE_PATH)

    return {
        "single_layer": int(SINGLE_LAYER),
        "resolved_unfreeze_encoder_layers": list(RESOLVED_UNFREEZE_ENCODER_LAYERS),
        "active_layer_name": str(ACTIVE_LAYER_NAME),
        "run_protocol": str(RUN_PROTOCOL_TAG),
        "active_fold_id": None,
        "feature_save_path": str(FEATURE_SAVE_PATH),
        "finetuned_whisper_dir": str(FINETUNED_WHISPER_DIR),
        "train_val_split_path": None,
        "train_val_split_path_resolved": None,
        "local_feature_split_save_path": str(LOCAL_FEATURE_SPLIT_SAVE_PATH),
    }

DEFAULT_SINGLE_LAYERS_RESOLVED = resolve_single_layers(
    SINGLE_LAYERS,
    whisper_model_name_or_path=BASE_WHISPER_MODEL,
    local_files_only=LOCAL_FILES_ONLY,
)

DEFAULT_RUN_CONTEXT = apply_run_context(
    single_layer=DEFAULT_SINGLE_LAYERS_RESOLVED[0],
    fold_id=None,
)

ENABLE_NOTEBOOK_LOG = True
NOTEBOOK_LOG_ROOT = os.path.join(FEATURE_SAVE_PATH, "notebook_logs")
NOTEBOOK_LOG_RUN_NAME = "audio_cut_mmse_regression_finetune"

os.makedirs(FINETUNED_WHISPER_DIR, exist_ok=True)
os.makedirs(FEATURE_SAVE_PATH, exist_ok=True)

print("DATASET_KEY =", DATASET_KEY)
print("FINETUNED_WHISPER_DIR =", FINETUNED_WHISPER_DIR)
print("FEATURE_SAVE_PATH =", FEATURE_SAVE_PATH)

import io
import sys
import types
import traceback
import contextlib
from IPython import get_ipython

def _nb_log_safe_json_value(obj):
    """尝试把对象转成适合写入 JSON 的形式。"""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [_nb_log_safe_json_value(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _nb_log_safe_json_value(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return repr(obj)

def _nb_log_collect_uppercase_params(global_ns):
    """收集当前全局命名空间里“看起来像参数配置”的变量。"""
    params = {}
    for k, v in global_ns.items():
        if not isinstance(k, str):
            continue
        if not k.isupper():
            continue
        if hasattr(v, "__module__") and callable(v):
            continue
        params[k] = _nb_log_safe_json_value(v)
    return dict(sorted(params.items(), key=lambda x: x[0]))

if not ENABLE_NOTEBOOK_LOG:
    print("[NOTEBOOK LOG] ENABLE_NOTEBOOK_LOG = False，跳过日志初始化。")
else:
    shell = get_ipython()
    if shell is None:
        print("[NOTEBOOK LOG] 当前环境不是 IPython/Jupyter，无法安装 cell 日志钩子。")
    elif getattr(shell, "_oa_nb_log_installed", False):
        print("[NOTEBOOK LOG] 日志钩子已经安装，无需重复安装。")
        print("[NOTEBOOK LOG] 当前日志目录：", shell._oa_nb_log_dir)
    else:
        _nb_log_time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        _nb_log_dir = Path(NOTEBOOK_LOG_ROOT) / f"{NOTEBOOK_LOG_RUN_NAME}_{_nb_log_time_tag}"
        _nb_log_dir.mkdir(parents=True, exist_ok=True)

        _nb_log_file = _nb_log_dir / "cell_output.log"
        _nb_params_file = _nb_log_dir / "params_snapshot.json"
        _nb_meta_file = _nb_log_dir / "run_meta.json"
        _nb_fig_dir = _nb_log_dir / "figures"
        _nb_fig_dir.mkdir(parents=True, exist_ok=True)

        def _nb_log_write_text(text: str) -> None:
            """向统一日志文件追加文本。"""
            with open(_nb_log_file, "a", encoding="utf-8") as f:
                f.write(text)
                if not text.endswith("\n"):
                    f.write("\n")

        def _nb_log_dump_params_snapshot() -> None:
            """把当前参数快照保存成 JSON。"""
            params = _nb_log_collect_uppercase_params(shell.user_ns)
            with open(_nb_params_file, "w", encoding="utf-8") as f:
                json.dump(params, f, ensure_ascii=False, indent=2)

        _nb_meta = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "log_dir": str(_nb_log_dir),
            "log_file": str(_nb_log_file),
            "params_snapshot_file": str(_nb_params_file),
            "figure_dir": str(_nb_fig_dir),
            "feature_save_path": str(FEATURE_SAVE_PATH),
            "finetuned_whisper_dir": str(FINETUNED_WHISPER_DIR),
            "dataset_key": str(DATASET_KEY),
        }
        with open(_nb_meta_file, "w", encoding="utf-8") as f:
            json.dump(_nb_meta, f, ensure_ascii=False, indent=2)

        _nb_log_dump_params_snapshot()
        _nb_log_write_text("=" * 120)
        _nb_log_write_text(f"[NOTEBOOK LOG START] {datetime.now().isoformat(timespec='seconds')}")
        _nb_log_write_text(f"[LOG DIR] {_nb_log_dir}")
        _nb_log_write_text("=" * 120)

        class _NotebookLogTee:
            """把输出同时写到："""
            def __init__(self, *streams):
                self.streams = streams

            def write(self, data):
                for s in self.streams:
                    s.write(data)
                return len(data)

            def flush(self):
                for s in self.streams:
                    s.flush()

            @property
            def encoding(self):
                s = self.streams[0] if len(self.streams) > 0 else None
                return getattr(s, "encoding", "utf-8")

            def isatty(self):
                return any(getattr(s, "isatty", lambda: False)() for s in self.streams)

        _nb_log_original_run_cell = shell.run_cell

        def _nb_log_run_cell_wrapper(self, raw_cell, *args, **kwargs):
            """包装后的 run_cell："""
            self._oa_nb_log_cell_counter = int(getattr(self, "_oa_nb_log_cell_counter", 0)) + 1
            cell_idx = int(self._oa_nb_log_cell_counter)

            _nb_log_write_text("")
            _nb_log_write_text("=" * 120)
            _nb_log_write_text(
                f"[CELL {cell_idx:04d}] {datetime.now().isoformat(timespec='seconds')}"
            )
            _nb_log_write_text("-" * 120)
            _nb_log_write_text("[SOURCE]")
            _nb_log_write_text(raw_cell.rstrip())

            stdout_buffer = io.StringIO()
            stderr_buffer = io.StringIO()

            tee_stdout = _NotebookLogTee(sys.stdout, stdout_buffer)
            tee_stderr = _NotebookLogTee(sys.stderr, stderr_buffer)

            with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
                result = _nb_log_original_run_cell(raw_cell, *args, **kwargs)

            stdout_text = stdout_buffer.getvalue()
            stderr_text = stderr_buffer.getvalue()

            if stdout_text.strip():
                _nb_log_write_text("[STDOUT]")
                _nb_log_write_text(stdout_text.rstrip())

            if stderr_text.strip():
                _nb_log_write_text("[STDERR]")
                _nb_log_write_text(stderr_text.rstrip())

            error_before_exec = getattr(result, "error_before_exec", None)
            error_in_exec = getattr(result, "error_in_exec", None)
            if error_before_exec is not None or error_in_exec is not None:
                _nb_log_write_text("[EXCEPTION]")
                err_obj = error_in_exec if error_in_exec is not None else error_before_exec
                _nb_log_write_text(
                    "".join(traceback.format_exception(type(err_obj), err_obj, err_obj.__traceback__)).rstrip()
                )

            result_obj = getattr(result, "result", None)
            if result_obj is not None:
                try:
                    result_repr = repr(result_obj)
                except Exception:
                    result_repr = "<repr failed>"
                if len(result_repr) > 5000:
                    result_repr = result_repr[:5000] + " ...<truncated>"
                _nb_log_write_text("[RESULT REPR]")
                _nb_log_write_text(result_repr)

            _nb_log_dump_params_snapshot()
            return result

        shell.run_cell = types.MethodType(_nb_log_run_cell_wrapper, shell)
        shell._oa_nb_log_installed = True
        shell._oa_nb_log_dir = str(_nb_log_dir)
        shell._oa_nb_log_file = str(_nb_log_file)
        shell._oa_nb_log_cell_counter = 0

        print("[NOTEBOOK LOG] 已启用。")
        print("[NOTEBOOK LOG] 日志目录 =", _nb_log_dir)
        print("[NOTEBOOK LOG] 参数快照 =", _nb_params_file)
        print("[NOTEBOOK LOG] 文本日志 =", _nb_log_file)
        print("[NOTEBOOK LOG] 提示：从这个 cell 之后运行的 code cell，都会被自动记录。")

# 各数据集音频目录。
DATASETS: Dict[str, Dict[str, str]] = {
    "PROCESS-2": {
        "train_cc": "/home/juanyu/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/juanyu/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/juanyu/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/juanyu/Downloads/PROCESS-2/test/cd",
    },
    "NCMMSC2021": {
        "train_cc": "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/HC",
        "train_cd": "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/AD_combine",
        "test_cc": "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/HC",
        "test_cd": "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/AD_combine",
    },
    "ADReSSo21": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "Pitt": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence",
        "test_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence",
    },
    "ADReSS_m": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/train_wav/HC",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/train_wav/AD",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr/HC",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr/AD",
    },
}

# MMSE 元数据路径与列名。
MMSE_META_PATHS = {
    "PROCESS-2": {
        "meta": "/home/juanyu/Downloads/PROCESS-2/meta-info.csv",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/cc_meta_data.txt",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/cd_meta_data.txt",
        "test_all": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/ADReSS 2020 test result.txt",
    },
    "ADReSSo21": {
        "train_all": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/adresso-train-mmse-scores.csv",
        "test_all": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/test_results_task2_groundtruth.csv",
    },
    "Pitt": {
        "all_meta": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/PItt-data(1).xlsx",
        "sheet_name": "data",
        "header_row": 2,
        "id_col": "id",
    },
    "ADReSS_m": {
        "train_all": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/training-groundtruth.csv",
        "test_all": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr-groundtruth.csv",
    },
}

for _dataset_key in DATASET_KEYS:
    assert _dataset_key in DATASETS, f"未知 DATASET_KEY={_dataset_key}"
    assert _dataset_key in MMSE_META_PATHS, f"当前 notebook 暂未配置 {_dataset_key} 的 MMSE 元数据"

assert DATASET_KEY in DATASETS, f"未知 DATASET_KEY={DATASET_KEY}"
assert DATASET_KEY in MMSE_META_PATHS, f"当前 notebook 暂未配置 {DATASET_KEY} 的 MMSE 元数据"

print("DATASET_KEYS =", DATASET_KEYS)
print("DEFAULT DATASET_KEY =", DATASET_KEY)
print("DEFAULT DATASET PATHS:")
for k, v in DATASETS[DATASET_KEY].items():
    print("  ", k, "->", v)

# 文件、切段与 MMSE 元数据工具
def natural_key(path_str: str) -> List[object]:
    """自然排序："""
    name = Path(path_str).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", name)]

def canonical_wav_path(wav_path: str) -> str:
    """统一 wav 路径表示，避免同一文件因为相对路径 / 绝对路径不同而无法对齐。"""
    return str(Path(wav_path).expanduser().resolve())

def _is_process2_ctd_wav_path(wav_path: str, dataset_key: str = None) -> bool:
    """PROCESS-2 专用过滤："""
    if dataset_key is None:
        dataset_key = globals().get("DATASET_KEY", "")

    if str(dataset_key).strip().lower() != "process-2":
        return True

    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None

def list_wav_files(audio_dir: str, dataset_key: str = None) -> List[str]:
    """读取目录中的 .wav 文件，并按自然顺序排序。"""
    if dataset_key is None:
        dataset_key = globals().get("DATASET_KEY", "")

    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"音频目录不存在: {audio_dir}")

    files = []
    for name in os.listdir(audio_dir):
        if not str(name).lower().endswith(".wav"):
            continue

        wav_path = str(Path(audio_dir) / name)

        if not _is_process2_ctd_wav_path(wav_path, dataset_key=dataset_key):
            continue

        files.append(wav_path)

    files = sorted(files, key=natural_key)

    if len(files) == 0:
        raise RuntimeError(
            f"目录中没有符合条件的 wav 文件: {audio_dir}, dataset_key={dataset_key}"
        )

    if str(dataset_key).strip().lower() == "process-2":
        print(f"[PROCESS-2][CTD] {audio_dir}: 只读取 CTD wav，文件数 = {len(files)}")

    return files

def split_segment_boundaries(audio_len: int, segment_length: int, overlap_rate: float) -> List[Tuple[int, int]]:
    """按固定长度 + 固定重叠率生成每一段的 [start, end) 边界。"""
    step = int(segment_length * (1.0 - overlap_rate))
    if step <= 0:
        raise ValueError("overlap_rate 设置不合理，导致步长 step<=0")

    total = int(audio_len / step) + (1 if (audio_len % step) > overlap_rate * segment_length else 0)
    total = max(total, 1)
    return [(i * step, i * step + segment_length) for i in range(total)]

# MMSE 标签解析
def _pick_first_existing_column(df: pd.DataFrame, candidates):
    """从候选列名里选第一个真正存在的列。"""
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"未找到候选列 {candidates}，当前列有：{list(df.columns)}")

def _safe_to_float(x):
    """把文本安全地转换成 float；空值 / NA / None 返回 None。"""
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.upper() in {"NA", "N/A", "NAN", "NONE"}:
        return None
    try:
        return float(s)
    except Exception:
        return None

def _extract_pitt_subject_visit_from_name(raw_name):
    """解析 Pitt 文件名中的 subject 和 visit："""
    if raw_name is None or pd.isna(raw_name):
        return None, None

    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None

    subject_key = str(int(m.group(1)))
    visit_index = int(m.group(2))
    return subject_key, visit_index

def _extract_pitt_subject_visit_from_wav_path(wav_path: str):
    """从完整 wav_path 提取 Pitt 的 (subject_key, visit_index)。"""
    return _extract_pitt_subject_visit_from_name(Path(wav_path).stem)

def _pitt_visit_to_mmse_col(visit_index: int) -> str:
    """Pitt readme 对应关系："""
    visit_index = int(visit_index)
    if visit_index < 0:
        raise ValueError(f"非法 visit_index={visit_index}")
    if visit_index == 0:
        return "mms"
    return f"mmse{visit_index + 1}"

def _normalize_mmse_key(dataset_key: str, raw_id):
    """统一元数据 ID 与音频文件名 stem 的格式，便于做字典匹配。"""
    if raw_id is None or pd.isna(raw_id):
        return None

    s = str(raw_id).strip()
    s = Path(s).stem

    dataset_key_norm = str(dataset_key).strip().lower()

    if dataset_key_norm == "pitt":
        m = re.search(r"\d+", s)
        if m is None:
            return None
        return str(int(m.group()))

    if dataset_key_norm == "process-2":
        s = re.sub(r"__(SFT|PFT|CTD)$", "", s, flags=re.IGNORECASE)
        return s.lower()

    return s.lower()

def _build_map_from_rows(df: pd.DataFrame, dataset_key: str, id_col: str, mmse_col: str):
    """把 DataFrame 的两列（ID, MMSE）转成 {规范化ID: mmse} 字典。"""
    mmse_map = {}
    for _, row in df.iterrows():
        key = _normalize_mmse_key(dataset_key, row[id_col])
        val = _safe_to_float(row[mmse_col])
        if key is None or val is None:
            continue
        mmse_map[key] = float(val)
    return mmse_map

def _read_adress_is2020_mmse(cfg: dict):
    """读取 ADReSS-IS2020 的 txt 格式 MMSE 元数据。"""
    def _read_txt(txt_path: str):
        mmse_map = {}
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (not line) or line.startswith("ID"):
                    continue
                parts = [x.strip() for x in line.split(";")]
                sid = _normalize_mmse_key("ADReSS-IS2020", parts[0])
                mmse_val = _safe_to_float(parts[-1])
                if sid is None or mmse_val is None:
                    continue
                mmse_map[sid] = float(mmse_val)
        return mmse_map

    train_cc_map = _read_txt(cfg["train_cc"])
    train_cd_map = _read_txt(cfg["train_cd"])
    test_map = _read_txt(cfg["test_all"])
    return {
        "train_cc": train_cc_map,
        "train_cd": train_cd_map,
        "test_cc": test_map,
        "test_cd": test_map,
    }

def _read_adresso21_mmse(cfg: dict):
    """读取 ADReSSo21 的 csv 格式 MMSE 元数据。"""
    train_df = pd.read_csv(cfg["train_all"])
    test_df = pd.read_csv(cfg["test_all"])

    train_id_col = _pick_first_existing_column(train_df, ["adressfname", "addressfname", "id"])
    train_mmse_col = _pick_first_existing_column(train_df, ["mmse", "MMSE"])
    train_dx_col = _pick_first_existing_column(train_df, ["dx", "label", "group"])

    test_id_col = _pick_first_existing_column(test_df, ["ID", "id", "adressfname", "addressfname"])
    test_mmse_col = _pick_first_existing_column(test_df, ["MMSE", "mmse"])

    dx_norm = train_df[train_dx_col].astype(str).str.strip().str.lower()
    train_cc_df = train_df[dx_norm.isin(["cn", "control", "cc"])]
    train_cd_df = train_df[dx_norm.isin(["ad", "probablead", "cd"])]

    train_cc_map = _build_map_from_rows(train_cc_df, "ADReSSo21", train_id_col, train_mmse_col)
    train_cd_map = _build_map_from_rows(train_cd_df, "ADReSSo21", train_id_col, train_mmse_col)
    test_map = _build_map_from_rows(test_df, "ADReSSo21", test_id_col, test_mmse_col)

    return {
        "train_cc": train_cc_map,
        "train_cd": train_cd_map,
        "test_cc": test_map,
        "test_cd": test_map,
    }

def _read_pitt_mmse(cfg: dict):
    """读取 Pitt 的 Excel 元数据，构建 visit-level MMSE 映射："""
    df = pd.read_excel(
        cfg["all_meta"],
        sheet_name=cfg.get("sheet_name", "data"),
        header=int(cfg.get("header_row", 2)),
    )

    id_col = _pick_first_existing_column(df, [cfg.get("id_col", "id"), "id"])

    col_map = {str(c).strip().lower(): c for c in df.columns}

    all_map = {}

    for _, row in df.iterrows():
        raw_id = row[id_col]
        if raw_id is None or pd.isna(raw_id):
            continue

        try:
            subject_key = str(int(raw_id))
        except Exception:
            m = re.search(r"\d+", str(raw_id))
            if m is None:
                continue
            subject_key = str(int(m.group(0)))

        for visit_index in range(0, 7):
            col_name = _pitt_visit_to_mmse_col(visit_index).lower()
            if col_name not in col_map:
                continue

            mmse_val = _safe_to_float(row[col_map[col_name]])
            if mmse_val is None:
                continue

            all_map[(subject_key, int(visit_index))] = float(mmse_val)

    return {
        "train_cc": all_map,
        "train_cd": all_map,
        "test_cc": all_map,
        "test_cd": all_map,
    }

def _read_adress_m_mmse(cfg: dict):
    """读取 ADReSS-M 的 groundtruth csv。"""
    train_df = pd.read_csv(cfg["train_all"])
    test_df = pd.read_csv(cfg["test_all"])

    train_id_col = _pick_first_existing_column(train_df, ["adressfname", "addressfname", "id", "ID"])
    train_mmse_col = _pick_first_existing_column(train_df, ["mmse", "MMSE"])
    train_dx_col = _pick_first_existing_column(train_df, ["dx", "label", "group"])

    test_id_col = _pick_first_existing_column(test_df, ["addressfname", "adressfname", "id", "ID"])
    test_mmse_col = _pick_first_existing_column(test_df, ["mmse", "MMSE"])

    dx_norm = train_df[train_dx_col].astype(str).str.strip().str.lower()
    train_cc_df = train_df[dx_norm.isin(["control", "cn", "cc"])]
    train_cd_df = train_df[dx_norm.isin(["probablead", "ad", "cd"])]

    train_cc_map = _build_map_from_rows(train_cc_df, "ADReSS_m", train_id_col, train_mmse_col)
    train_cd_map = _build_map_from_rows(train_cd_df, "ADReSS_m", train_id_col, train_mmse_col)
    test_map = _build_map_from_rows(test_df, "ADReSS_m", test_id_col, test_mmse_col)

    return {
        "train_cc": train_cc_map,
        "train_cd": train_cd_map,
        "test_cc": test_map,
        "test_cd": test_map,
    }

def _read_process2_mmse(cfg: dict):
    """读取 PROCESS-2 的 meta-info.csv，构建四个子集的 MMSE 映射。"""
    meta_path = cfg["meta"]
    df = pd.read_csv(meta_path)

    id_col = _pick_first_existing_column(df, ["IDs", "ID", "id", "dir_name"])
    diagnosis_col = _pick_first_existing_column(df, ["diagnosis", "Diagnosis", "dx", "label"])
    mmse_col = _pick_first_existing_column(df, ["MMSE", "mmse"])
    split_col = _pick_first_existing_column(df, ["Split", "split", "partition"])

    split_norm = df[split_col].astype(str).str.strip().str.lower()
    diagnosis_norm = df[diagnosis_col].astype(str).str.strip().str.lower()

    is_train = split_norm.isin(["train", "training"])
    is_test = split_norm.isin(["test", "testing"])

    is_cc = diagnosis_norm.isin(["hc", "healthy", "healthy control", "control", "cn", "cc"])
    is_cd = diagnosis_norm.isin(["mci", "dementia", "ad", "probablead", "cd"])

    train_cc_df = df[is_train & is_cc]
    train_cd_df = df[is_train & is_cd]
    test_cc_df = df[is_test & is_cc]
    test_cd_df = df[is_test & is_cd]

    train_cc_map = _build_map_from_rows(train_cc_df, "PROCESS-2", id_col, mmse_col)
    train_cd_map = _build_map_from_rows(train_cd_df, "PROCESS-2", id_col, mmse_col)
    test_cc_map = _build_map_from_rows(test_cc_df, "PROCESS-2", id_col, mmse_col)
    test_cd_map = _build_map_from_rows(test_cd_df, "PROCESS-2", id_col, mmse_col)

    print("[PROCESS-2 MMSE] meta rows:", len(df))
    print("[PROCESS-2 MMSE] train_cc rows:", len(train_cc_df), "valid MMSE:", len(train_cc_map))
    print("[PROCESS-2 MMSE] train_cd rows:", len(train_cd_df), "valid MMSE:", len(train_cd_map))
    print("[PROCESS-2 MMSE] test_cc rows :", len(test_cc_df), "valid MMSE:", len(test_cc_map))
    print("[PROCESS-2 MMSE] test_cd rows :", len(test_cd_df), "valid MMSE:", len(test_cd_map))

    return {
        "train_cc": train_cc_map,
        "train_cd": train_cd_map,
        "test_cc": test_cc_map,
        "test_cd": test_cd_map,
    }

def load_mmse_maps_for_dataset(dataset_key: str):
    """统一读取指定数据集的 MMSE 映射。"""
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"暂未配置数据集 {dataset_key} 的 MMSE 元数据路径")

    cfg = MMSE_META_PATHS[dataset_key]

    if dataset_key == "PROCESS-2":
        mmse_maps = _read_process2_mmse(cfg)
    elif dataset_key == "ADReSS-IS2020":
        mmse_maps = _read_adress_is2020_mmse(cfg)
    elif dataset_key == "ADReSSo21":
        mmse_maps = _read_adresso21_mmse(cfg)
    elif dataset_key == "Pitt":
        mmse_maps = _read_pitt_mmse(cfg)
    elif dataset_key == "ADReSS_m":
        mmse_maps = _read_adress_m_mmse(cfg)
    else:
        raise ValueError(f"暂未实现数据集 {dataset_key} 的 MMSE 解析逻辑")

    print(
        f"[MMSE] {dataset_key} 映射加载完成："
        f"train_cc={len(mmse_maps['train_cc'])}, "
        f"train_cd={len(mmse_maps['train_cd'])}, "
        f"test_cc={len(mmse_maps['test_cc'])}, "
        f"test_cd={len(mmse_maps['test_cd'])}"
    )
    return mmse_maps

def get_mmse_for_wav(dataset_key: str, wav_path: str, mmse_map) -> Optional[float]:
    """根据音频路径去查该音频对应的 MMSE 分数。"""
    if dataset_key == "Pitt":
        subject_key, visit_index = _extract_pitt_subject_visit_from_wav_path(wav_path)
        if subject_key is None or visit_index is None:
            return None
        return mmse_map.get((subject_key, int(visit_index)), None)

    sid = _normalize_mmse_key(dataset_key, Path(wav_path).stem)
    if sid is None:
        return None
    return mmse_map.get(sid, None)

def make_missing_mmse_record_for_wts(dataset_key: str, subset_name: str, wav_path: str) -> dict:
    """为 whisper_teacher_student.py 构造缺失 MMSE 的音频记录。"""
    record = {
        "dataset_key": str(dataset_key),
        "subset_name": str(subset_name),
        "wav_name": Path(wav_path).name,
        "wav_path": canonical_wav_path(wav_path),
        "reason": "missing_mmse",
    }

    if str(dataset_key).strip().lower() == "pitt":
        try:
            subject_key, visit_index = _extract_pitt_subject_visit_from_wav_path(wav_path)
            record["subject_id"] = subject_key
            record["visit_index"] = visit_index

            if subject_key is not None and visit_index is not None:
                expected_col = _pitt_visit_to_mmse_col(int(visit_index))
                record["expected_mmse_col"] = expected_col
                record["reason"] = f"missing_{expected_col}"
            else:
                record["expected_mmse_col"] = None
                record["reason"] = "unparsed_pitt_subject_visit"
        except Exception:
            record["subject_id"] = None
            record["visit_index"] = None
            record["expected_mmse_col"] = None
            record["reason"] = "pitt_parse_error"

    return record

def save_missing_mmse_records_for_wts(missing_records, save_path: str, file_tag: str) -> Optional[str]:
    """保存 whisper_teacher_student.py 中被跳过的缺失 MMSE 音频。"""
    if missing_records is None or len(missing_records) == 0:
        print("[MMSE] 本次没有缺失 MMSE 的音频，无需导出 CSV。")
        return None

    os.makedirs(save_path, exist_ok=True)

    csv_path = os.path.join(save_path, f"missing_mmse_{file_tag}.csv")

    df = pd.DataFrame(missing_records)
    dedup_cols = [c for c in ["subset_name", "wav_path"] if c in df.columns]
    if len(dedup_cols) > 0:
        df = df.drop_duplicates(subset=dedup_cols, keep="first")

    front_cols = [
        "dataset_key",
        "subset_name",
        "wav_name",
        "wav_path",
        "reason",
        "subject_id",
        "visit_index",
        "expected_mmse_col",
    ]
    cols = [c for c in front_cols if c in df.columns] + [c for c in df.columns if c not in front_cols]
    df = df[cols]

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[MMSE] 缺失 MMSE 音频完整列表已保存到: {csv_path}")
    print(f"[MMSE] 缺失音频总数（去重后）: {len(df)}")

    return csv_path

def save_json_file(payload: Dict[str, object], save_path: str) -> None:
    """把字典保存成 utf-8 json。"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

# Pitt 受试者互斥划分与审计
def _pitt_collect_and_make_subject_disjoint_split_for_vanilla(
    dataset_paths: Dict[str, str],
    mmse_map: Dict[Tuple[str, int], float],
):
    """只针对 Pitt：从原 train/test 四个目录收集全部 candidate visits，先按 visit-level MMSE 过滤，"""
    if PITT_SPLIT_POLICY != "original_test_subjects_holdout":
        raise ValueError(f"不支持的 PITT_SPLIT_POLICY={PITT_SPLIT_POLICY}")

    source_subsets = ["train_cc", "train_cd", "test_cc", "test_cd"]
    records = []
    missing_records = []
    seen_visit_keys = {}
    candidate_count = 0

    for source_subset in source_subsets:
        audio_dir = dataset_paths[source_subset]
        files = list_wav_files(audio_dir, dataset_key="Pitt")
        candidate_count += len(files)
        source_split, group = source_subset.split("_", 1)

        for fp in files:
            subject_id, visit_index = _extract_pitt_subject_visit_from_wav_path(fp)
            if subject_id is None or visit_index is None:
                rec = make_missing_mmse_record_for_wts("Pitt", source_subset, fp)
                rec["reason"] = "unparsed_pitt_subject_visit"
                missing_records.append(rec)
                continue

            visit_key = (str(subject_id), int(visit_index))
            if visit_key not in mmse_map:
                missing_records.append(
                    make_missing_mmse_record_for_wts("Pitt", source_subset, fp)
                )
                continue

            if visit_key in seen_visit_keys:
                raise RuntimeError(
                    "[Pitt split] 发现重复的 subject-visit key："
                    f"{visit_key}\n"
                    f"  first={seen_visit_keys[visit_key]}\n"
                    f"  second={fp}"
                )
            seen_visit_keys[visit_key] = str(fp)

            records.append({
                "subject_id": str(subject_id),
                "visit_index": int(visit_index),
                "mmse": float(mmse_map[visit_key]),
                "group": str(group),
                "source_subset": str(source_subset),
                "source_split": str(source_split),
                "wav_name": Path(fp).name,
                "wav_path": canonical_wav_path(fp),
            })

    valid_visit_count = len(records)
    valid_subjects = {r["subject_id"] for r in records}
    original_test_subjects = {
        r["subject_id"] for r in records if r["source_split"] == "test"
    }
    original_train_subjects = {
        r["subject_id"] for r in records if r["source_split"] == "train"
    }

    final_test_subjects = set(original_test_subjects)
    final_train_subjects = set(valid_subjects) - final_test_subjects

    overlap = final_train_subjects & final_test_subjects
    if overlap:
        raise RuntimeError(
            f"[Pitt split] subject-disjoint 检查失败，overlap={sorted(overlap)}"
        )

    final_files = {
        "train_cc": [],
        "train_cd": [],
        "test_cc": [],
        "test_cd": [],
    }

    for rec in records:
        final_split = "test" if rec["subject_id"] in final_test_subjects else "train"
        final_subset = f"{final_split}_{rec['group']}"
        rec["final_split"] = final_split
        rec["final_subset"] = final_subset
        final_files[final_subset].append(rec["wav_path"])

    for subset_name in final_files:
        final_files[subset_name] = sorted(
            final_files[subset_name],
            key=lambda p: (natural_key(p), str(p)),
        )

    final_train_ids = {
        r["subject_id"] for r in records if r["final_split"] == "train"
    }
    final_test_ids = {
        r["subject_id"] for r in records if r["final_split"] == "test"
    }
    final_overlap = final_train_ids & final_test_ids

    if final_overlap:
        raise RuntimeError(
            f"[Pitt split] 最终 train/test 仍存在 subject overlap: {sorted(final_overlap)}"
        )
    if len(final_train_ids | final_test_ids) != len(valid_subjects):
        raise RuntimeError("[Pitt split] 最终 subject union 与有效 subject 集不一致。")
    if sum(len(v) for v in final_files.values()) != valid_visit_count:
        raise RuntimeError("[Pitt split] 最终 visit 总数与有效 visit 总数不一致。")

    expected_checks = [
        ("candidate_visits", candidate_count, PITT_EXPECTED_CANDIDATE_VISITS),
        ("valid_visits", valid_visit_count, PITT_EXPECTED_VALID_VISITS),
        ("unique_subjects", len(valid_subjects), PITT_EXPECTED_UNIQUE_SUBJECTS),
        (
            "valid_original_test_subjects",
            len(original_test_subjects),
            PITT_EXPECTED_VALID_TEST_SUBJECTS,
        ),
    ]
    for name, actual, expected in expected_checks:
        if int(actual) != int(expected):
            msg = f"[Pitt split] {name}={actual}，预期={expected}"
            if PITT_STRICT_COUNT_CHECK:
                raise RuntimeError(msg + "。为避免无意改变实验数据，已停止。")
            print("[WARN]", msg)

    visit_df = pd.DataFrame(records).sort_values(
        ["final_split", "subject_id", "visit_index", "wav_name"],
        kind="stable",
    ).reset_index(drop=True)

    subject_rows = []
    for subject_id in sorted(valid_subjects, key=lambda x: int(x)):
        rows = [r for r in records if r["subject_id"] == subject_id]
        src_splits = {r["source_split"] for r in rows}
        subject_rows.append({
            "subject_id": subject_id,
            "original_membership": (
                "both" if src_splits == {"train", "test"}
                else ("test" if src_splits == {"test"} else "train")
            ),
            "final_split": "test" if subject_id in final_test_subjects else "train",
            "valid_visit_count": len(rows),
            "cc_visit_count": sum(r["group"] == "cc" for r in rows),
            "cd_visit_count": sum(r["group"] == "cd" for r in rows),
            "original_train_visit_count": sum(r["source_split"] == "train" for r in rows),
            "original_test_visit_count": sum(r["source_split"] == "test" for r in rows),
        })
    subject_df = pd.DataFrame(subject_rows)

    original_overlap = original_train_subjects & original_test_subjects

    print("\n" + "=" * 100)
    print("[Pitt subject-disjoint final split | VanillaFT]")
    print(f"policy = {PITT_SPLIT_POLICY}")
    print(f"candidate visits = {candidate_count}")
    print(f"valid-MMSE visits = {valid_visit_count}")
    print(f"missing-MMSE/unusable-for-regression visits = {len(missing_records)}")
    print(f"valid unique subjects = {len(valid_subjects)}")
    print(
        "original valid subject counts | "
        f"train={len(original_train_subjects)}, "
        f"test={len(original_test_subjects)}, "
        f"overlap={len(original_overlap)}"
    )
    print(
        "final subject counts | "
        f"train={len(final_train_ids)}, "
        f"test={len(final_test_ids)}, "
        f"overlap={len(final_overlap)}"
    )
    print(
        "final visit counts | "
        f"train_cc={len(final_files['train_cc'])}, "
        f"train_cd={len(final_files['train_cd'])}, "
        f"test_cc={len(final_files['test_cc'])}, "
        f"test_cd={len(final_files['test_cd'])}, "
        f"total={sum(len(v) for v in final_files.values())}"
    )
    print("=" * 100 + "\n")

    split_info = {
        "split_strategy": str(PITT_SPLIT_POLICY),
        "subject_disjoint": True,
        "train_test_subject_overlap": 0,
        "candidate_visits": int(candidate_count),
        "valid_visits": int(valid_visit_count),
        "unique_subjects": int(len(valid_subjects)),
        "train_subjects": int(len(final_train_ids)),
        "test_subjects": int(len(final_test_ids)),
        "train_visits": int(len(final_files["train_cc"]) + len(final_files["train_cd"])),
        "test_visits": int(len(final_files["test_cc"]) + len(final_files["test_cd"])),
        "original_valid_subject_overlap": int(len(original_overlap)),
    }

    return final_files, visit_df, subject_df, missing_records, split_info

def _save_pitt_split_manifests_for_vanilla(
    visit_df: pd.DataFrame,
    subject_df: pd.DataFrame,
) -> Tuple[str, str]:
    """保存与 Whisper_pitt_new.py 同字段口径的 Pitt final split 审计文件。"""
    split_root = os.path.join(FEATURE_ROOT_DIR, "Pitt", "subject_disjoint_split")
    os.makedirs(split_root, exist_ok=True)

    visit_path = os.path.join(
        split_root,
        "pitt_valid_visits_subject_disjoint_manifest.csv",
    )
    subject_path = os.path.join(
        split_root,
        "pitt_subject_assignment.csv",
    )

    visit_df.to_csv(visit_path, index=False, encoding="utf-8-sig")
    subject_df.to_csv(subject_path, index=False, encoding="utf-8-sig")

    print(f"[Pitt split] visit manifest -> {visit_path}")
    print(f"[Pitt split] subject assignment -> {subject_path}")
    return visit_path, subject_path

def build_experiment_config_snapshot() -> Dict[str, object]:
    """在训练开始前保存一份独立的实验配置快照。"""
    return {
        "format_version": 1,
        "task_type": "mmse_regression_finetune",
        "created_at": datetime.now().isoformat(timespec="seconds"),

        "dataset_key": DATASET_KEY,
        "model_name": MODEL_NAME,
        "base_whisper_model": str(Path(BASE_WHISPER_MODEL).resolve()),
        "local_files_only": bool(LOCAL_FILES_ONLY),

        "active_fold_id": ACTIVE_FOLD_ID,
        "active_layer_name": ACTIVE_LAYER_NAME,
        "single_layer": int(SINGLE_LAYER),
        "single_layers_batch": [int(x) for x in DEFAULT_SINGLE_LAYERS_RESOLVED],

        "finetuned_whisper_dir": str(Path(FINETUNED_WHISPER_DIR).resolve()),
        "feature_save_path": str(Path(FEATURE_SAVE_PATH).resolve()),
        "train_val_split_path_resolved": str(TRAIN_VAL_SPLIT_PATH_RESOLVED),

        "sampling_rate": int(SAMPLING_RATE),
        "segment_sec": int(SEGMENT_SEC),
        "segment_length": int(SEGMENT_LENGTH),
        "overlap_rate": float(OVERLAP_RATE),
        "max_segments_per_audio": int(MAX_SEGMENTS_PER_AUDIO),
        "normalize_audio": bool(NORMALIZE_AUDIO),
        "val_ratio": float(VAL_RATIO),

        "pitt_subject_disjoint_enabled": bool(
            DATASET_KEY == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
        ),
        "pitt_split_policy": (
            str(PITT_SPLIT_POLICY) if DATASET_KEY == "Pitt" else None
        ),

        "seed": int(SEED),
        "device": str(DEVICE),
        "batch_size": int(BATCH_SIZE),
        "num_epochs": int(NUM_EPOCHS),
        "lr": float(LR),

        "temporal_finetune_mode": str(TEMPORAL_FINETUNE_MODE),
        "temporal_stage1_epochs": int(TEMPORAL_STAGE1_EPOCHS),
        "temporal_stage2_epochs": int(TEMPORAL_STAGE2_EPOCHS),
        "temporal_stage1_head_lr": float(TEMPORAL_STAGE1_HEAD_LR),
        "temporal_stage2_whisper_lr": float(TEMPORAL_STAGE2_WHISPER_LR),
        "temporal_stage2_head_lr": float(TEMPORAL_STAGE2_HEAD_LR),
        "weight_decay": float(WEIGHT_DECAY),
        "grad_clip_norm": float(GRAD_CLIP_NORM),
        "upstream_mmse_supervision_level": str(globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")),

        "temporal_target_normalize": bool(TEMPORAL_TARGET_NORMALIZE),
        "temporal_target_normalize_eps": float(TEMPORAL_TARGET_NORMALIZE_EPS),

        "unfreeze_encoder_layers": int(UNFREEZE_ENCODER_LAYERS),
        "resolved_unfreeze_encoder_layers": [int(x) for x in RESOLVED_UNFREEZE_ENCODER_LAYERS],

        "pooling_type": str(POOLING_TYPE),
        "attention_hidden_dim": int(ATTENTION_HIDDEN_DIM),
        "regressor_dropout": float(REGRESSOR_DROPOUT),
        "regression_loss_type": str(REGRESSION_LOSS_TYPE),
        "smooth_l1_beta": float(SMOOTH_L1_BETA),

        "consistency_weight": float(CONSISTENCY_WEIGHT),
        "consistency_layer": int(CONSISTENCY_LAYER),
        "consistency_loss_type": str(CONSISTENCY_LOSS_TYPE),
        "audio_feature_consistency_weight": float(AUDIO_FEATURE_CONSISTENCY_WEIGHT),
        "audio_prediction_consistency_weight": float(AUDIO_PREDICTION_CONSISTENCY_WEIGHT),

        "feature_prefix": str(FEATURE_PREFIX),
        "norm_for_single": str(NORM_FOR_SINGLE),
        "prototype_smoothing_alpha": float(PROTOTYPE_SMOOTHING_ALPHA),
        "skip_feature_export": bool(SKIP_FEATURE_EXPORT),

        "enable_tuning_tag_in_path": bool(ENABLE_TUNING_TAG_IN_PATH),
        "enable_training_hparam_tag_in_path": bool(ENABLE_TRAINING_HPARAM_TAG_IN_PATH),
        "save_params_in_path": (
            "all"
            if globals().get("SAVE_PARAMS_IN_PATH", None) is None
            else globals().get("SAVE_PARAMS_IN_PATH", None)
        ),

        "enable_notebook_log": bool(ENABLE_NOTEBOOK_LOG),
        "notebook_log_root": str(NOTEBOOK_LOG_ROOT),
        "notebook_log_run_name": str(NOTEBOOK_LOG_RUN_NAME),
    }

mmse_maps = load_mmse_maps_for_dataset(DATASET_KEY)

# 音频数据集、collate 与模型
@dataclass
class AudioSequenceRecord:
    """audio-level 样本记录。"""
    wav_path: str
    target: float
    audio_id: int
    boundaries: List[Tuple[int, int]]

def load_all_segments_for_audio_record(
    record: AudioSequenceRecord,
    sampling_rate: int,
    segment_length: int,
    normalize_audio: bool,
    overlap_rate: float,
) -> np.ndarray:
    """一次性读取一条音频，并切出该音频的全部 segments。"""
    y, _ = librosa.load(
        record.wav_path,
        sr=sampling_rate,
        mono=True,
        dtype=np.float32,
    )

    if normalize_audio:
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-8:
            y = y / peak

    segments = []

    for seg_idx, (start, end) in enumerate(record.boundaries):
        seg = y[start:end]

        if len(seg) < segment_length and start > 0:
            need = segment_length - len(seg)
            step = int(segment_length * (1.0 - overlap_rate))
            prev_end = start + segment_length - step
            prev_start = max(0, prev_end - need)
            pad_values = y[prev_start:prev_end]

            if len(pad_values) < need:
                pad_values = np.concatenate(
                    [
                        pad_values,
                        np.zeros((need - len(pad_values),), dtype=np.float32),
                    ],
                    axis=0,
                )

            seg = np.concatenate([seg, pad_values], axis=0)

        if len(seg) < segment_length:
            seg = np.concatenate(
                [
                    seg,
                    np.zeros((segment_length - len(seg),), dtype=np.float32),
                ],
                axis=0,
            )
        elif len(seg) > segment_length:
            seg = seg[:segment_length]

        segments.append(seg.astype(np.float32))

    if len(segments) == 0:
        raise RuntimeError(f"音频没有有效 segment: {record.wav_path}")

    return np.stack(segments, axis=0).astype(np.float32)

class AudioSequenceRegressionDataset(Dataset):
    """audio-level temporal 微调数据集。"""
    def __init__(
        self,
        records: Sequence[AudioSequenceRecord],
        sampling_rate: int,
        segment_length: int,
        normalize_audio: bool,
        overlap_rate: float,
    ) -> None:
        self.records = list(records)
        self.sampling_rate = int(sampling_rate)
        self.segment_length = int(segment_length)
        self.normalize_audio = bool(normalize_audio)
        self.overlap_rate = float(overlap_rate)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        segments = load_all_segments_for_audio_record(
            record=rec,
            sampling_rate=self.sampling_rate,
            segment_length=self.segment_length,
            normalize_audio=self.normalize_audio,
            overlap_rate=self.overlap_rate,
        )
        return segments, np.float32(rec.target), int(rec.audio_id)

def build_audio_sequence_records_from_files_with_mmse(
    wav_files: Sequence[str],
    dataset_key: str,
    mmse_map: Dict[str, float],
    start_audio_id: int,
    sampling_rate: int,
    segment_length: int,
    overlap_rate: float,
    max_segments_per_audio: int,
) -> Tuple[List[AudioSequenceRecord], int]:
    """把音频文件列表构造成 audio-level 样本。"""
    records: List[AudioSequenceRecord] = []
    audio_id = int(start_audio_id)
    skipped_files = []

    for wav_path in tqdm(wav_files, desc="[build_audio_sequence_records_with_mmse]", ncols=100):
        mmse = get_mmse_for_wav(dataset_key, wav_path, mmse_map)
        if mmse is None:
            skipped_files.append(Path(wav_path).name)
            continue

        y, _ = librosa.load(wav_path, sr=sampling_rate, mono=True, dtype=np.float32)
        boundaries = split_segment_boundaries(len(y), segment_length, overlap_rate)

        if max_segments_per_audio > 0:
            boundaries = boundaries[:max_segments_per_audio]

        if len(boundaries) == 0:
            skipped_files.append(Path(wav_path).name)
            continue

        records.append(
            AudioSequenceRecord(
                wav_path=wav_path,
                target=float(mmse),
                audio_id=int(audio_id),
                boundaries=[(int(s), int(e)) for s, e in boundaries],
            )
        )
        audio_id += 1

    if len(skipped_files) > 0:
        print(f"[WARN] 有 {len(skipped_files)} 个音频没有 MMSE 或没有有效 segment，已跳过。示例：{skipped_files[:10]}")

    return records, audio_id

def build_audio_sequence_collate_fn(processor: WhisperProcessor, sampling_rate: int):
    """audio-level collate_fn。"""
    def _collate(batch):
        all_segments = []
        targets = []
        audio_ids = []
        segment_counts = []

        for segments, target, audio_id in batch:
            segments = np.asarray(segments, dtype=np.float32)
            if segments.ndim != 2:
                raise ValueError(f"期望 segments shape=[S,L]，当前 shape={segments.shape}")

            segment_counts.append(int(segments.shape[0]))
            targets.append(float(target))
            audio_ids.append(int(audio_id))

            for i in range(segments.shape[0]):
                all_segments.append(segments[i])

        proc = processor(
            all_segments,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        return (
            proc.input_features,                                      # [sum_S, mel, time]
            torch.tensor(targets, dtype=torch.float32),               # [B]
            torch.tensor(audio_ids, dtype=torch.long),                # [B]
            torch.tensor(segment_counts, dtype=torch.long),           # [B]
        )

    return _collate

def select_encoder_hidden_state(enc_out, layer_index: int) -> torch.Tensor:
    """从 Whisper encoder 的 hidden_states 中按 1-based 层号取某一层。"""
    hs_all = list(enc_out.hidden_states) if enc_out.hidden_states is not None else []
    hs = hs_all[1:] if len(hs_all) >= 2 else hs_all
    if not (1 <= int(layer_index) <= len(hs)):
        raise ValueError(f"layer_index={layer_index} 超出范围 [1, {len(hs)}]")
    return hs[int(layer_index) - 1]

def freeze_module(module: nn.Module) -> None:
    """冻结 teacher 模型参数，并切到 eval 模式。"""
    module.eval()
    for p in module.parameters():
        p.requires_grad = False

class TemporalAttentionPooling(nn.Module):
    """轻量 attention pooling。"""
    def __init__(self, d_model: int, attention_hidden_dim: int) -> None:
        super().__init__()
        hidden_dim = max(int(attention_hidden_dim), 1)
        self.score_proj = nn.Linear(int(d_model), hidden_dim)
        self.score_out = nn.Linear(hidden_dim, 1)

    def forward(self, hidden_states: torch.Tensor):
        score_hidden = torch.tanh(self.score_proj(hidden_states))
        scores = self.score_out(score_hidden).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        pooled = torch.sum(hidden_states * weights.unsqueeze(-1), dim=1)
        return pooled, weights

def pack_flat_segment_embeddings_to_audio_sequence(
    flat_embeddings: torch.Tensor,
    segment_counts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """把 flat segment embeddings 还原成 audio-level 序列。"""
    segment_counts = segment_counts.to(flat_embeddings.device).long()
    batch_size = int(segment_counts.shape[0])
    max_segments = int(segment_counts.max().item())
    feat_dim = int(flat_embeddings.shape[-1])

    seq = flat_embeddings.new_zeros((batch_size, max_segments, feat_dim))
    mask = torch.zeros((batch_size, max_segments), dtype=torch.bool, device=flat_embeddings.device)

    offset = 0
    for b in range(batch_size):
        n = int(segment_counts[b].item())
        if n <= 0:
            raise ValueError("segment_counts 中出现 <=0 的音频。")
        seq[b, :n, :] = flat_embeddings[offset: offset + n]
        mask[b, :n] = True
        offset += n

    if offset != int(flat_embeddings.shape[0]):
        raise ValueError(
            f"segment_counts 之和={offset} 与 flat_embeddings 数量={flat_embeddings.shape[0]} 不一致。"
        )

    return seq, mask

class AudioPoolingLinearHead(nn.Module):
    """pool_linear / linear_pool 回归头。"""

    def __init__(
        self,
        d_model: int,
        dropout: float,
        frame_pooling_type: str = "mean",
        attention_hidden_dim: Optional[int] = None,
        prediction_level: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.d_model = int(d_model)
        self.frame_pooling_type = str(frame_pooling_type).strip().lower()

        if self.frame_pooling_type not in {"mean", "attention"}:
            raise ValueError(
                f"不支持的 frame_pooling_type={frame_pooling_type}，"
                "可选：'mean' / 'attention'"
            )

        if prediction_level is None:
            prediction_level = globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")

        self.prediction_level = str(prediction_level).strip().lower()

        if self.prediction_level in {"audio_level", "audio-level"}:
            self.prediction_level = "audio"

        if self.prediction_level in {
            "segment",
            "segment_level",
            "segment-level",
            "segment_loss",
            "segment_direct",
            "chunk",
            "chunk_level",
            "chunk_loss",
            "chunk_direct",
        }:
            self.prediction_level = "segment"

        if self.prediction_level in {
            "segment_audio_balanced",
            "audio_balanced_segment",
            "segment-balanced",
            "segment_audio_balance",
        }:
            self.prediction_level = "segment_audio_balanced"

        if self.prediction_level in {
            "segment_pred_mean",
            "segment-pred-mean",
            "prediction_mean",
            "prediction-mean",
            "segment_prediction_mean",
            "audio_mean",
            "audio-level-mean",
        }:
            self.prediction_level = "segment_pred_mean"

        if self.prediction_level not in {
            "audio",
            "segment",
            "segment_audio_balanced",
            "segment_pred_mean",
        }:
            raise ValueError(
                f"未知 UPSTREAM_MMSE_SUPERVISION_LEVEL={prediction_level}，"
                "可选：'audio' / 'segment' / 'segment_audio_balanced' / 'segment_pred_mean'"
            )

        attn_hidden = (
            int(attention_hidden_dim)
            if attention_hidden_dim is not None
            else max(int(d_model) // 2, 1)
        )

        self.frame_attention_pool = TemporalAttentionPooling(
            d_model=int(d_model),
            attention_hidden_dim=attn_hidden,
        )

        self.dropout = nn.Dropout(float(dropout))
        self.regressor = nn.Linear(int(d_model), 1)

    def pool_frames(self, hidden_states: torch.Tensor):
        """hidden_states: [sum_S, T, C]"""
        if self.frame_pooling_type == "attention":
            segment_embeddings, frame_weights = self.frame_attention_pool(hidden_states)
            return segment_embeddings, frame_weights

        segment_embeddings = hidden_states.mean(dim=1)
        frame_weights = None
        return segment_embeddings, frame_weights

    @staticmethod
    def _make_segment_audio_indices(segment_counts: torch.Tensor, device: torch.device) -> torch.Tensor:
        """根据 segment_counts 构造每个 flat segment 属于 batch 内哪条音频的索引。"""
        segment_counts = segment_counts.to(device=device).long()
        batch_size = int(segment_counts.numel())
        audio_indices = torch.arange(batch_size, device=device, dtype=torch.long)
        return torch.repeat_interleave(audio_indices, segment_counts)

    def forward(
        self,
        hidden_states: torch.Tensor,
        segment_counts: torch.Tensor,
        return_aux: bool = False,
    ):
        """hidden_states:"""
        segment_embeddings, frame_weights = self.pool_frames(hidden_states)

        seq_embeddings, mask = pack_flat_segment_embeddings_to_audio_sequence(
            flat_embeddings=segment_embeddings,
            segment_counts=segment_counts,
        )

        mask_f = mask.to(dtype=seq_embeddings.dtype)
        denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)

        audio_embedding = torch.sum(
            seq_embeddings * mask_f.unsqueeze(-1),
            dim=1,
        ) / denom

        segment_audio_indices = self._make_segment_audio_indices(
            segment_counts=segment_counts,
            device=segment_embeddings.device,
        )

        if self.prediction_level in {"segment", "segment_audio_balanced", "segment_pred_mean"}:
            preds = self.regressor(self.dropout(segment_embeddings)).squeeze(-1)
            output_level = "segment"
        else:
            preds = self.regressor(self.dropout(audio_embedding)).squeeze(-1)
            output_level = "audio"

        if not return_aux:
            return preds

        segment_weights = mask_f / denom

        aux = {
            "prediction_level": output_level,
            "supervision_level": self.prediction_level,

            "segment_embeddings": segment_embeddings,
            "seq_embeddings": seq_embeddings,
            "mask": mask,
            "pooled": audio_embedding,
            "audio_embedding": audio_embedding,
            "temporal_weights": segment_weights,
            "frame_weights": frame_weights,

            "segment_audio_indices": segment_audio_indices,
            "segment_counts": segment_counts,
        }

        return preds, aux

class GRUAttentionTemporalHead(nn.Module):
    """轻量 temporal head："""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=int(input_dim),
            hidden_size=int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            dropout=float(dropout) if int(num_layers) > 1 else 0.0,
            bidirectional=False,
        )

        self.attn_score = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.Tanh(),
            nn.Linear(int(hidden_dim), 1),
        )

        self.dropout = nn.Dropout(float(dropout))
        self.regressor = nn.Linear(int(hidden_dim), 1)

    def forward(self, seq_embeddings: torch.Tensor, mask: torch.Tensor):
        """seq_embeddings: [B, S, D]"""
        gru_out, _ = self.gru(seq_embeddings)  # [B, S, H]

        scores = self.attn_score(gru_out).squeeze(-1)  # [B, S]
        scores = scores.masked_fill(~mask, -1e9)

        weights = torch.softmax(scores, dim=1)  # [B, S]
        pooled = torch.sum(gru_out * weights.unsqueeze(-1), dim=1)  # [B, H]

        pred = self.regressor(self.dropout(pooled)).squeeze(-1)  # [B]
        return pred, {
            "temporal_weights": weights,
            "temporal_hidden": gru_out,
            "pooled": pooled,
        }

class WhisperAudioTemporalMMSERegressor(nn.Module):
    """Whisper + 轻量 temporal head 的 audio-level MMSE 回归模型。"""
    def __init__(
        self,
        whisper_model_name_or_path: str,
        local_files_only: bool,
        feature_layer: int,
        temporal_head_type: str,
        temporal_hidden_dim: int,
        temporal_num_layers: int,
        temporal_dropout: float,
        segment_pool: str = "mean",
    ) -> None:
        super().__init__()

        self.whisper = WhisperModel.from_pretrained(
            whisper_model_name_or_path,
            local_files_only=local_files_only,
        )

        self.feature_layer = int(feature_layer)
        self.segment_pool = str(segment_pool).strip().lower()

        d_model = int(self.whisper.config.d_model)

        head_type = str(temporal_head_type).strip().lower()
        self.temporal_head_type = head_type

        self.upstream_mmse_supervision_level = str(
            globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")
        ).strip().lower()

        if self.upstream_mmse_supervision_level in {"audio_level", "audio-level"}:
            self.upstream_mmse_supervision_level = "audio"

        if self.upstream_mmse_supervision_level in {
            "segment",
            "segment_level",
            "segment-level",
            "segment_loss",
            "segment_direct",
            "chunk",
            "chunk_level",
            "chunk_loss",
            "chunk_direct",
        }:
            self.upstream_mmse_supervision_level = "segment"

        if self.upstream_mmse_supervision_level in {
            "segment_audio_balanced",
            "audio_balanced_segment",
            "segment-balanced",
            "segment_audio_balance",
        }:
            self.upstream_mmse_supervision_level = "segment_audio_balanced"

        if self.upstream_mmse_supervision_level in {
            "segment_pred_mean",
            "segment-pred-mean",
            "prediction_mean",
            "prediction-mean",
            "segment_prediction_mean",
            "audio_mean",
            "audio-level-mean",
        }:
            self.upstream_mmse_supervision_level = "segment_pred_mean"

        if self.upstream_mmse_supervision_level not in {
            "audio",
            "segment",
            "segment_audio_balanced",
            "segment_pred_mean",
        }:
            raise ValueError(
                f"未知 UPSTREAM_MMSE_SUPERVISION_LEVEL={self.upstream_mmse_supervision_level}，"
                "可选：'audio' / 'segment' / 'segment_audio_balanced' / 'segment_pred_mean'"
            )

        if head_type == "gru_attn":
            if self.upstream_mmse_supervision_level != "audio":
                raise NotImplementedError(
                    "当前 whisper_vanilla_mmse_mean.py 中 GRUAttentionTemporalHead "
                    "仍然只支持 audio-level prediction；"
                    "若要使用 segment_pred_mean，请保持 TEMPORAL_HEAD_TYPE='pool_linear'。"
                )

            self.temporal_head = GRUAttentionTemporalHead(
                input_dim=d_model,
                hidden_dim=int(temporal_hidden_dim),
                num_layers=int(temporal_num_layers),
                dropout=float(temporal_dropout),
            )

        elif head_type in {"pool_linear", "linear_pool", "mean_linear", "attn_linear"}:
            self.temporal_head = AudioPoolingLinearHead(
                d_model=d_model,
                dropout=float(temporal_dropout),
                frame_pooling_type=self.segment_pool,
                attention_hidden_dim=ATTENTION_HIDDEN_DIM,
                prediction_level=self.upstream_mmse_supervision_level,
            )

        else:
            raise ValueError(
                f"未知 TEMPORAL_HEAD_TYPE={temporal_head_type}，"
                "可选：'gru_attn' / 'pool_linear'"
            )

    def pool_segment_frames(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """把每个 segment 的 frame-level hidden states 压成 segment embedding。"""
        if self.segment_pool == "mean":
            return hidden_states.mean(dim=1)

        raise ValueError(f"未知 TEMPORAL_SEGMENT_POOL={self.segment_pool}")

    def forward(
        self,
        input_features: torch.Tensor,
        segment_counts: torch.Tensor,
        return_aux: bool = False,
    ):
        """input_features:"""
        enc = self.whisper.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        selected_hidden = select_encoder_hidden_state(enc, self.feature_layer)  # [sum_S, T, C]

        if self.temporal_head_type == "gru_attn":
            segment_embeddings = self.pool_segment_frames(selected_hidden)      # [sum_S, C]

            seq_embeddings, mask = pack_flat_segment_embeddings_to_audio_sequence(
                flat_embeddings=segment_embeddings,
                segment_counts=segment_counts,
            )

            preds, temporal_aux = self.temporal_head(seq_embeddings, mask)

            if not return_aux:
                return preds

            aux = {
                "selected_hidden": selected_hidden,
                "segment_embeddings": segment_embeddings,
                "seq_embeddings": seq_embeddings,
                "mask": mask,
            }
            aux.update(temporal_aux)
            return preds, aux

        else:
            preds, head_aux = self.temporal_head(
                selected_hidden,
                segment_counts=segment_counts,
                return_aux=True,
            )

            if not return_aux:
                return preds

            aux = {
                "selected_hidden": selected_hidden,
            }
            aux.update(head_aux)
            return preds, aux

        if not return_aux:
            return preds

        aux = {
            "selected_hidden": selected_hidden,
            "segment_embeddings": segment_embeddings,
            "seq_embeddings": seq_embeddings,
            "mask": mask,
        }
        aux.update(temporal_aux)
        return preds, aux

def configure_whisper_trainable_layers(model: nn.Module, unfreeze_encoder_layers: Sequence[int]) -> List[int]:
    """设置 Whisper 哪些 encoder 层可训练。"""
    if not hasattr(model, "whisper"):
        raise AttributeError("model 缺少 whisper 属性，无法设置 Whisper 可训练层。")

    for p in model.whisper.parameters():
        p.requires_grad = False

    total_layers = len(model.whisper.encoder.layers)

    requested_layers = list(unfreeze_encoder_layers) if unfreeze_encoder_layers is not None else []
    resolved_layers = sorted({
        int(x) for x in requested_layers
        if 1 <= int(x) <= total_layers
    })

    for layer_idx_1based in resolved_layers:
        layer = model.whisper.encoder.layers[layer_idx_1based - 1]
        for p in layer.parameters():
            p.requires_grad = True

    if len(resolved_layers) > 0:
        for p in model.whisper.encoder.layer_norm.parameters():
            p.requires_grad = True

    if hasattr(model, "segment_head"):
        for p in model.segment_head.parameters():
            p.requires_grad = True

    if hasattr(model, "temporal_head"):
        for p in model.temporal_head.parameters():
            p.requires_grad = True

    return resolved_layers

# 损失、指标与目标标准化
def build_regression_criterion(loss_type: str, beta: float = 1.0):
    """构建回归损失函数。"""
    loss_type = str(loss_type).strip().lower()
    if loss_type == "mse":
        return nn.MSELoss()
    if loss_type == "mae":
        return nn.L1Loss()
    if loss_type == "smooth_l1":
        return nn.SmoothL1Loss(beta=float(beta))
    raise ValueError(f"未知的 REGRESSION_LOSS_TYPE: {loss_type}")

def compute_regression_loss_elementwise(
    preds: torch.Tensor,
    targets: torch.Tensor,
    loss_type: Optional[str] = None,
    beta: Optional[float] = None,
) -> torch.Tensor:
    """返回逐样本回归 loss，不做 reduction。"""
    if loss_type is None:
        loss_type = globals().get("REGRESSION_LOSS_TYPE", "mae")
    if beta is None:
        beta = globals().get("SMOOTH_L1_BETA", 1.0)

    loss_type = str(loss_type).strip().lower()

    if loss_type == "mse":
        return F.mse_loss(preds, targets, reduction="none")

    if loss_type == "mae":
        return F.l1_loss(preds, targets, reduction="none")

    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(
            preds,
            targets,
            beta=float(beta),
            reduction="none",
        )

    raise ValueError(f"未知的 REGRESSION_LOSS_TYPE: {loss_type}")

def repeat_audio_targets_to_segments(
    targets_audio: torch.Tensor,
    segment_counts: torch.Tensor,
) -> torch.Tensor:
    """把 audio-level target 复制到对应 segments。"""
    targets_audio = targets_audio.view(-1)
    segment_counts = segment_counts.to(device=targets_audio.device).long().view(-1)

    if int(targets_audio.numel()) != int(segment_counts.numel()):
        raise ValueError(
            f"targets_audio 数量={targets_audio.numel()} 与 segment_counts 数量={segment_counts.numel()} 不一致。"
        )

    return torch.repeat_interleave(targets_audio, segment_counts)

def make_segment_audio_indices_from_counts(
    segment_counts: torch.Tensor,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """根据 segment_counts 构造每个 segment 所属的 batch 内 audio index。"""
    if device is None:
        device = segment_counts.device

    segment_counts = segment_counts.to(device=device).long().view(-1)
    batch_size = int(segment_counts.numel())
    audio_indices = torch.arange(batch_size, device=device, dtype=torch.long)
    return torch.repeat_interleave(audio_indices, segment_counts)

def compute_audio_balanced_segment_regression_loss(
    segment_preds: torch.Tensor,
    targets_audio: torch.Tensor,
    segment_counts: torch.Tensor,
    loss_type: Optional[str] = None,
    beta: Optional[float] = None,
):
    """segment-level prediction + audio-balanced loss。"""
    segment_preds = segment_preds.view(-1)
    targets_audio = targets_audio.view(-1)
    segment_counts = segment_counts.to(device=segment_preds.device).long().view(-1)

    targets_segment = repeat_audio_targets_to_segments(
        targets_audio=targets_audio,
        segment_counts=segment_counts,
    ).to(device=segment_preds.device)

    if int(segment_preds.numel()) != int(targets_segment.numel()):
        raise ValueError(
            f"segment_preds 数量={segment_preds.numel()} 与 repeated targets 数量={targets_segment.numel()} 不一致。"
        )

    elementwise_loss = compute_regression_loss_elementwise(
        preds=segment_preds,
        targets=targets_segment,
        loss_type=loss_type,
        beta=beta,
    )

    segment_audio_indices = make_segment_audio_indices_from_counts(
        segment_counts=segment_counts,
        device=segment_preds.device,
    )

    batch_size = int(segment_counts.numel())
    per_audio_loss_sum = segment_preds.new_zeros((batch_size,), dtype=elementwise_loss.dtype)
    per_audio_loss_sum.scatter_add_(0, segment_audio_indices, elementwise_loss)

    denom = segment_counts.to(dtype=elementwise_loss.dtype).clamp_min(1.0)
    per_audio_loss = per_audio_loss_sum / denom

    balanced_loss = per_audio_loss.mean()
    return balanced_loss, targets_segment, segment_audio_indices, per_audio_loss

def compute_direct_segment_regression_loss(
    segment_preds: torch.Tensor,
    targets_audio: torch.Tensor,
    segment_counts: torch.Tensor,
    loss_type: Optional[str] = None,
    beta: Optional[float] = None,
):
    """stage3-style segment-level regression loss。"""
    segment_preds = segment_preds.view(-1)
    targets_audio = targets_audio.view(-1)
    segment_counts = segment_counts.to(device=segment_preds.device).long().view(-1)

    targets_segment = repeat_audio_targets_to_segments(
        targets_audio=targets_audio,
        segment_counts=segment_counts,
    ).to(device=segment_preds.device)

    if int(segment_preds.numel()) != int(targets_segment.numel()):
        raise ValueError(
            f"segment_preds 数量={segment_preds.numel()} 与 targets_segment 数量={targets_segment.numel()} 不一致。"
        )

    elementwise_loss = compute_regression_loss_elementwise(
        preds=segment_preds,
        targets=targets_segment,
        loss_type=loss_type,
        beta=beta,
    )

    segment_audio_indices = make_segment_audio_indices_from_counts(
        segment_counts=segment_counts,
        device=segment_preds.device,
    )

    segment_loss = elementwise_loss.mean()
    return segment_loss, targets_segment, segment_audio_indices, elementwise_loss

def aggregate_segment_values_to_audio_mean(
    segment_values: torch.Tensor,
    segment_counts: torch.Tensor,
) -> torch.Tensor:
    """把 segment-level prediction 聚合成 audio-level mean prediction。"""
    segment_values = segment_values.view(-1)
    segment_counts = segment_counts.to(device=segment_values.device).long().view(-1)

    segment_audio_indices = make_segment_audio_indices_from_counts(
        segment_counts=segment_counts,
        device=segment_values.device,
    )

    batch_size = int(segment_counts.numel())
    audio_sum = segment_values.new_zeros((batch_size,), dtype=segment_values.dtype)
    audio_sum.scatter_add_(0, segment_audio_indices, segment_values)

    denom = segment_counts.to(dtype=segment_values.dtype).clamp_min(1.0)
    return audio_sum / denom

def compute_segment_pred_mean_audio_regression_loss(
    segment_preds: torch.Tensor,
    targets_audio: torch.Tensor,
    segment_counts: torch.Tensor,
    loss_type: Optional[str] = None,
    beta: Optional[float] = None,
):
    """segment_pred_mean loss。"""
    segment_preds = segment_preds.view(-1)
    targets_audio = targets_audio.view(-1)
    segment_counts = segment_counts.to(device=segment_preds.device).long().view(-1)

    audio_preds = aggregate_segment_values_to_audio_mean(
        segment_values=segment_preds,
        segment_counts=segment_counts,
    )

    if int(audio_preds.numel()) != int(targets_audio.numel()):
        raise ValueError(
            f"audio_preds 数量={audio_preds.numel()} 与 targets_audio 数量={targets_audio.numel()} 不一致。"
        )

    audio_elementwise_loss = compute_regression_loss_elementwise(
        preds=audio_preds,
        targets=targets_audio,
        loss_type=loss_type,
        beta=beta,
    )

    audio_loss = audio_elementwise_loss.mean()

    return audio_loss, audio_preds, audio_elementwise_loss

def compute_hidden_state_consistency_loss(student_h: torch.Tensor, teacher_h: torch.Tensor, loss_type: str) -> torch.Tensor:
    """teacher / student 隐状态一致性损失。"""
    loss_type = str(loss_type).strip().lower()
    if loss_type == "mse":
        return F.mse_loss(student_h, teacher_h)
    if loss_type == "smooth_l1":
        return F.smooth_l1_loss(student_h, teacher_h)
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(student_h, teacher_h, dim=-1).mean()
    raise ValueError(f"未知 consistency loss type: {loss_type}")

def compute_temporal_delta_consistency_loss(
    student_seq: torch.Tensor,
    teacher_seq: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """保护 segment 序列的动态结构。"""
    losses = []

    batch_size = int(student_seq.shape[0])

    for b in range(batch_size):
        valid_idx = torch.where(mask[b])[0]
        n = int(valid_idx.numel())

        if n <= 1:
            continue

        s = student_seq[b, :n, :]
        t = teacher_seq[b, :n, :]

        s_delta = s[1:, :] - s[:-1, :]
        t_delta = t[1:, :] - t[:-1, :]

        losses.append(F.mse_loss(s_delta, t_delta))

    if len(losses) == 0:
        return torch.zeros((), device=student_seq.device)

    return torch.stack(losses).mean()

def compute_regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """计算回归指标："""
    preds = preds.detach().float().view(-1)
    targets = targets.detach().float().view(-1)

    mae = torch.mean(torch.abs(preds - targets)).item()
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()

    pred_center = preds - preds.mean()
    tgt_center = targets - targets.mean()
    denom = (torch.sqrt(torch.sum(pred_center ** 2)) * torch.sqrt(torch.sum(tgt_center ** 2))).item()
    if denom < 1e-12:
        pearson = 0.0
    else:
        pearson = torch.sum(pred_center * tgt_center).item() / denom

    ss_res = torch.sum((targets - preds) ** 2).item()
    ss_tot = torch.sum((targets - targets.mean()) ** 2).item()
    r2 = 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "pearson": float(pearson),
        "r2": float(r2),
    }

def compute_target_normalizer_from_audio_records(
    records: Sequence[AudioSequenceRecord],
    eps: float = 1e-6,
) -> Dict[str, float]:
    """根据当前 fold_train 的 audio-level MMSE 标签计算 target normalizer。"""
    if records is None or len(records) == 0:
        raise ValueError("records 为空，无法计算 target normalizer。")

    targets = np.asarray([float(r.target) for r in records], dtype=np.float32)

    mean = float(np.mean(targets))
    std = float(np.std(targets))

    eps = float(eps)
    if std < eps:
        print(
            f"[WARN] target std={std:.8f} < eps={eps:.8f}，"
            "将 std 设置为 1.0，避免除零。"
        )
        std = 1.0

    return {
        "enabled": True,
        "mean": mean,
        "std": std,
        "eps": eps,
        "n_audio": int(len(targets)),
        "min": float(np.min(targets)),
        "max": float(np.max(targets)),
    }

def normalize_mmse_targets(
    targets: torch.Tensor,
    target_normalizer: Dict[str, float],
) -> torch.Tensor:
    """把原始 MMSE target 转成 normalized target。"""
    if target_normalizer is None or not bool(target_normalizer.get("enabled", False)):
        return targets

    mean = torch.tensor(
        float(target_normalizer["mean"]),
        dtype=targets.dtype,
        device=targets.device,
    )
    std = torch.tensor(
        float(target_normalizer["std"]),
        dtype=targets.dtype,
        device=targets.device,
    ).clamp_min(float(target_normalizer.get("eps", 1e-6)))

    return (targets - mean) / std

def denormalize_mmse_predictions(
    preds_norm: torch.Tensor,
    target_normalizer: Dict[str, float],
) -> torch.Tensor:
    """把 normalized prediction 反变换回原始 MMSE 尺度。"""
    if target_normalizer is None or not bool(target_normalizer.get("enabled", False)):
        return preds_norm

    mean = torch.tensor(
        float(target_normalizer["mean"]),
        dtype=preds_norm.dtype,
        device=preds_norm.device,
    )
    std = torch.tensor(
        float(target_normalizer["std"]),
        dtype=preds_norm.dtype,
        device=preds_norm.device,
    ).clamp_min(float(target_normalizer.get("eps", 1e-6)))

    return preds_norm * std + mean

# 当前训练循环
def build_temporal_optimizer_for_current_stage(
    model: WhisperAudioTemporalMMSERegressor,
    stage_name: str,
    stage1_head_lr: float,
    stage2_whisper_lr: float,
    stage2_head_lr: float,
    weight_decay: float,
):
    """为二阶段 temporal 微调构造 optimizer。"""
    stage_name = str(stage_name).strip().lower()

    if stage_name == "stage1":
        params = [p for p in model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError("Stage 1 没有可训练参数，请检查 temporal_head 是否可训练。")
        return torch.optim.AdamW(
            params,
            lr=float(stage1_head_lr),
            weight_decay=float(weight_decay),
        )

    if stage_name == "stage2":
        whisper_params = []
        head_params = []

        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("whisper."):
                whisper_params.append(p)
            else:
                head_params.append(p)

        param_groups = []
        if len(whisper_params) > 0:
            param_groups.append(
                {
                    "params": whisper_params,
                    "lr": float(stage2_whisper_lr),
                    "weight_decay": float(weight_decay),
                }
            )
        if len(head_params) > 0:
            param_groups.append(
                {
                    "params": head_params,
                    "lr": float(stage2_head_lr),
                    "weight_decay": float(weight_decay),
                }
            )

        if len(param_groups) == 0:
            raise RuntimeError("Stage 2 没有可训练参数，请检查解冻层和 temporal_head。")

        return torch.optim.AdamW(param_groups)

    raise ValueError(f"未知 stage_name={stage_name}")

def run_one_temporal_audio_epoch(
    model: WhisperAudioTemporalMMSERegressor,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    target_normalizer: Dict[str, float],
    device: torch.device,
    epoch_index: int,
    total_epochs: int,
    stage_name: str,
    grad_clip_norm: float,
    teacher_model: Optional[WhisperModel],
    consistency_weight: float,
    consistency_layer: Optional[int],
    consistency_loss_type: str,
    temporal_delta_consistency_weight: float,
):
    """跑一个 temporal audio-level epoch。"""
    model.train()

    whisper_has_trainable_params = any(p.requires_grad for p in model.whisper.parameters())
    if not whisper_has_trainable_params:
        model.whisper.eval()

    use_hidden_consistency = (teacher_model is not None) and (float(consistency_weight) > 0.0)
    use_delta_consistency = (teacher_model is not None) and (float(temporal_delta_consistency_weight) > 0.0)

    resolved_consistency_layer = None if consistency_layer is None else int(consistency_layer)
    if (use_hidden_consistency or use_delta_consistency) and (
        resolved_consistency_layer is None or resolved_consistency_layer <= 0
    ):
        raise ValueError("启用 teacher consistency 时必须提供合法的 consistency_layer。")

    train_loss_sum = 0.0
    train_reg_loss_sum = 0.0
    train_hidden_cons_loss_sum = 0.0
    train_delta_cons_loss_sum = 0.0
    train_audio_cnt = 0

    train_preds_raw_all = []
    train_targets_raw_all = []

    for input_features, targets_raw, audio_ids, segment_counts in tqdm(
        train_loader,
        desc=f"[{stage_name}] epoch={epoch_index}/{total_epochs}",
        ncols=120,
    ):
        input_features = input_features.to(device)
        targets_raw = targets_raw.to(device)
        segment_counts = segment_counts.to(device)

        targets_for_loss = normalize_mmse_targets(
            targets_raw,
            target_normalizer,
        )

        optimizer.zero_grad(set_to_none=True)

        preds_for_loss, aux = model(
            input_features=input_features,
            segment_counts=segment_counts,
            return_aux=True,
        )

        prediction_level = str(aux.get("prediction_level", "audio")).strip().lower()

        if prediction_level == "segment":
            supervision_level = str(
                aux.get(
                    "supervision_level",
                    globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "segment"),
                )
            ).strip().lower()

            segment_direct_aliases = {
                "segment",
                "segment_level",
                "segment-level",
                "segment_loss",
                "segment_direct",
                "chunk",
                "chunk_level",
                "chunk_loss",
                "chunk_direct",
            }

            segment_audio_balanced_aliases = {
                "segment_audio_balanced",
                "audio_balanced_segment",
                "segment-balanced",
                "segment_audio_balance",
            }

            segment_pred_mean_aliases = {
                "segment_pred_mean",
                "segment-pred-mean",
                "prediction_mean",
                "prediction-mean",
                "segment_prediction_mean",
                "audio_mean",
                "audio-level-mean",
            }

            if supervision_level in segment_pred_mean_aliases:
                reg_loss, preds_audio_for_loss, per_audio_reg_loss = (
                    compute_segment_pred_mean_audio_regression_loss(
                        segment_preds=preds_for_loss,
                        targets_audio=targets_for_loss,
                        segment_counts=segment_counts,
                        loss_type=REGRESSION_LOSS_TYPE,
                        beta=SMOOTH_L1_BETA,
                    )
                )
                targets_for_loss_segment = None
                segment_audio_indices = None

            elif supervision_level in segment_audio_balanced_aliases:
                reg_loss, targets_for_loss_segment, segment_audio_indices, per_audio_reg_loss = (
                    compute_audio_balanced_segment_regression_loss(
                        segment_preds=preds_for_loss,
                        targets_audio=targets_for_loss,
                        segment_counts=segment_counts,
                        loss_type=REGRESSION_LOSS_TYPE,
                        beta=SMOOTH_L1_BETA,
                    )
                )

            elif supervision_level in segment_direct_aliases:
                reg_loss, targets_for_loss_segment, segment_audio_indices, per_audio_reg_loss = (
                    compute_direct_segment_regression_loss(
                        segment_preds=preds_for_loss,
                        targets_audio=targets_for_loss,
                        segment_counts=segment_counts,
                        loss_type=REGRESSION_LOSS_TYPE,
                        beta=SMOOTH_L1_BETA,
                    )
                )

            else:
                raise ValueError(
                    f"未知 UPSTREAM_MMSE_SUPERVISION_LEVEL={supervision_level!r}，"
                    "可选：'segment' / 'segment_audio_balanced' / 'segment_pred_mean'"
                )

        else:
            reg_loss = criterion(preds_for_loss, targets_for_loss)
            targets_for_loss_segment = None
            segment_audio_indices = None
            per_audio_reg_loss = None

        hidden_cons_loss = torch.zeros((), device=device)
        delta_cons_loss = torch.zeros((), device=device)

        if use_hidden_consistency or use_delta_consistency:
            with torch.no_grad():
                teacher_out = teacher_model.encoder(
                    input_features=input_features,
                    output_hidden_states=True,
                    return_dict=True,
                )
                teacher_hidden = select_encoder_hidden_state(
                    teacher_out,
                    resolved_consistency_layer,
                )
                teacher_segment_embeddings = teacher_hidden.mean(dim=1)
                teacher_seq, teacher_mask = pack_flat_segment_embeddings_to_audio_sequence(
                    flat_embeddings=teacher_segment_embeddings,
                    segment_counts=segment_counts,
                )

        if use_hidden_consistency:
            hidden_cons_loss = compute_hidden_state_consistency_loss(
                aux["selected_hidden"],
                teacher_hidden,
                consistency_loss_type,
            )

        if use_delta_consistency:
            delta_cons_loss = compute_temporal_delta_consistency_loss(
                student_seq=aux["seq_embeddings"],
                teacher_seq=teacher_seq,
                mask=aux["mask"],
            )

        loss = (
            reg_loss
            + float(consistency_weight) * hidden_cons_loss
            + float(temporal_delta_consistency_weight) * delta_cons_loss
        )

        loss.backward()

        if grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(grad_clip_norm),
            )

        optimizer.step()

        batch_audio_now = int(targets_raw.shape[0])
        train_loss_sum += float(loss.item()) * batch_audio_now
        train_reg_loss_sum += float(reg_loss.item()) * batch_audio_now
        train_hidden_cons_loss_sum += float(hidden_cons_loss.item()) * batch_audio_now
        train_delta_cons_loss_sum += float(delta_cons_loss.item()) * batch_audio_now
        train_audio_cnt += batch_audio_now

        preds_raw = denormalize_mmse_predictions(
            preds_for_loss.detach(),
            target_normalizer,
        )

        if prediction_level == "segment":
            preds_raw_for_metrics = aggregate_segment_values_to_audio_mean(
                segment_values=preds_raw,
                segment_counts=segment_counts,
            )
        else:
            preds_raw_for_metrics = preds_raw

        train_preds_raw_all.append(preds_raw_for_metrics.detach().cpu())
        train_targets_raw_all.append(targets_raw.detach().cpu())

    train_loss = train_loss_sum / max(train_audio_cnt, 1)
    train_reg_loss = train_reg_loss_sum / max(train_audio_cnt, 1)
    train_hidden_cons_loss = train_hidden_cons_loss_sum / max(train_audio_cnt, 1)
    train_delta_cons_loss = train_delta_cons_loss_sum / max(train_audio_cnt, 1)

    if len(train_preds_raw_all) > 0:
        train_preds_raw_cat = torch.cat(train_preds_raw_all, dim=0)
        train_targets_raw_cat = torch.cat(train_targets_raw_all, dim=0)
        train_metrics = compute_regression_metrics(
            train_preds_raw_cat,
            train_targets_raw_cat,
        )
    else:
        train_metrics = {
            "mae": 0.0,
            "rmse": float("inf"),
            "pearson": 0.0,
            "r2": 0.0,
        }

    row = {
        "stage": str(stage_name),
        "epoch": float(epoch_index),
        "upstream_mmse_supervision_level": str(globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")),
        "prediction_level": str(prediction_level if "prediction_level" in locals() else "audio"),
        "train_loss": float(train_loss),
        "train_regression_loss": float(train_reg_loss),
        "train_hidden_consistency_loss": float(train_hidden_cons_loss),
        "train_temporal_delta_consistency_loss": float(train_delta_cons_loss),
        "train_audio_mae": float(train_metrics["mae"]),
        "train_audio_rmse": float(train_metrics["rmse"]),
        "train_audio_pearson": float(train_metrics["pearson"]),
        "train_audio_r2": float(train_metrics["r2"]),
        "target_normalize": bool(target_normalizer.get("enabled", False)),
        "target_mean": float(target_normalizer.get("mean", 0.0)),
        "target_std": float(target_normalizer.get("std", 1.0)),
        "val_loss": None,
        "val_audio_rmse": None,
        "val_audio_mae": None,
        "val_audio_pearson": None,
        "val_audio_r2": None,
    }

    print(
        f"[{stage_name}][epoch={epoch_index}/{total_epochs}] "
        f"supervision={str(globals().get('UPSTREAM_MMSE_SUPERVISION_LEVEL', 'audio'))} "
        f"prediction_level={str(prediction_level if 'prediction_level' in locals() else 'audio')} "
        f"train_loss={train_loss:.6f} "
        f"train_reg_loss={train_reg_loss:.6f} "
        f"hidden_cons={train_hidden_cons_loss:.6f} "
        f"delta_cons={train_delta_cons_loss:.6f} "
        f"train_audio_rmse={train_metrics['rmse']:.4f} "
        f"train_audio_mae={train_metrics['mae']:.4f} "
        f"train_audio_pearson={train_metrics['pearson']:.4f} "
        f"train_audio_r2={train_metrics['r2']:.4f} "
        f"target_norm={bool(target_normalizer.get('enabled', False))}"
    )

    return row

def run_finetune_mmse_temporal_audio_level(
    model: WhisperAudioTemporalMMSERegressor,
    processor: WhisperProcessor,
    train_records: Sequence[AudioSequenceRecord],
    sampling_rate: int,
    segment_length: int,
    overlap_rate: float,
    normalize_audio: bool,
    audio_batch_size: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    grad_clip_norm: float,
    device: torch.device,
    regression_loss_type: str,
    smooth_l1_beta: float,
    teacher_model: Optional[WhisperModel] = None,
    consistency_weight: float = 0.0,
    consistency_layer: Optional[int] = None,
    consistency_loss_type: str = "mse",
    temporal_delta_consistency_weight: float = 0.0,
    seed: int = 0,
    target_normalize: bool = True,
    target_normalize_eps: float = 1e-6,

    finetune_mode: str = "two_stage",
    stage1_epochs: int = 5,
    stage2_epochs: int = 3,
    stage1_head_lr: float = 1e-4,
    stage2_whisper_lr: float = 1e-5,
    stage2_head_lr: float = 3e-5,
    stage2_unfreeze_encoder_layers: Optional[Sequence[int]] = None,
):
    """audio-level temporal 微调。"""
    finetune_mode = str(finetune_mode).strip().lower()
    if finetune_mode not in {"two_stage", "single_stage"}:
        raise ValueError(f"未知 finetune_mode={finetune_mode}")

    if bool(target_normalize):
        target_normalizer = compute_target_normalizer_from_audio_records(
            train_records,
            eps=float(target_normalize_eps),
        )
    else:
        target_normalizer = {
            "enabled": False,
            "mean": 0.0,
            "std": 1.0,
            "eps": float(target_normalize_eps),
            "n_audio": int(len(train_records)),
            "min": None,
            "max": None,
        }

    print("[TARGET NORMALIZER]", target_normalizer)

    train_ds = AudioSequenceRegressionDataset(
        records=train_records,
        sampling_rate=sampling_rate,
        segment_length=segment_length,
        normalize_audio=normalize_audio,
        overlap_rate=overlap_rate,
    )

    collate_fn = build_audio_sequence_collate_fn(processor, sampling_rate)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(audio_batch_size),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    criterion = build_regression_criterion(regression_loss_type, beta=smooth_l1_beta)
    history = []

    if finetune_mode == "two_stage":
        total_epochs = int(stage1_epochs) + int(stage2_epochs)

        print("\n" + "=" * 90)
        print("[TEMPORAL TWO-STAGE] Stage 1: freeze Whisper, train temporal head")
        print("=" * 90)

        configure_whisper_trainable_layers(model, [])

        stage1_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        stage1_total = sum(p.numel() for p in model.parameters())
        print(
            f"[STAGE1 MODEL] trainable={stage1_trainable} "
            f"total={stage1_total} ratio={stage1_trainable / max(stage1_total, 1):.6f}"
        )

        optimizer = build_temporal_optimizer_for_current_stage(
            model=model,
            stage_name="stage1",
            stage1_head_lr=stage1_head_lr,
            stage2_whisper_lr=stage2_whisper_lr,
            stage2_head_lr=stage2_head_lr,
            weight_decay=weight_decay,
        )

        for e in range(1, int(stage1_epochs) + 1):
            row = run_one_temporal_audio_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                target_normalizer=target_normalizer,
                device=device,
                epoch_index=e,
                total_epochs=total_epochs,
                stage_name="stage1_head_warmup",
                grad_clip_norm=grad_clip_norm,
                teacher_model=None,                 # Stage 1 不做 teacher consistency
                consistency_weight=0.0,
                consistency_layer=None,
                consistency_loss_type=consistency_loss_type,
                temporal_delta_consistency_weight=0.0,
            )
            history.append(row)

        print("\n" + "=" * 90)
        print("[TEMPORAL TWO-STAGE] Stage 2: unfreeze Whisper layer(s), adapt encoder")
        print("=" * 90)

        if stage2_unfreeze_encoder_layers is None:
            stage2_unfreeze_encoder_layers = []

        resolved_stage2_layers = configure_whisper_trainable_layers(
            model,
            stage2_unfreeze_encoder_layers,
        )

        print(f"[STAGE2 UNFREEZE] encoder layers = {resolved_stage2_layers}")

        stage2_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        stage2_total = sum(p.numel() for p in model.parameters())
        print(
            f"[STAGE2 MODEL] trainable={stage2_trainable} "
            f"total={stage2_total} ratio={stage2_trainable / max(stage2_total, 1):.6f}"
        )

        optimizer = build_temporal_optimizer_for_current_stage(
            model=model,
            stage_name="stage2",
            stage1_head_lr=stage1_head_lr,
            stage2_whisper_lr=stage2_whisper_lr,
            stage2_head_lr=stage2_head_lr,
            weight_decay=weight_decay,
        )

        for e in range(1, int(stage2_epochs) + 1):
            global_epoch = int(stage1_epochs) + e

            row = run_one_temporal_audio_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                target_normalizer=target_normalizer,
                device=device,
                epoch_index=global_epoch,
                total_epochs=total_epochs,
                stage_name="stage2_encoder_adapt",
                grad_clip_norm=grad_clip_norm,
                teacher_model=teacher_model,
                consistency_weight=consistency_weight,
                consistency_layer=consistency_layer,
                consistency_loss_type=consistency_loss_type,
                temporal_delta_consistency_weight=temporal_delta_consistency_weight,
            )
            history.append(row)

        selected_epoch = total_epochs
        selection_mode = "temporal_audio_level_two_stage_last_checkpoint"

    else:
        selected_epoch = int(num_epochs)
        selection_mode = "temporal_audio_level_single_stage_last_checkpoint"

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

        for e in range(1, int(num_epochs) + 1):
            row = run_one_temporal_audio_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                target_normalizer=target_normalizer,
                device=device,
                epoch_index=e,
                total_epochs=int(num_epochs),
                stage_name="single_stage",
                grad_clip_norm=grad_clip_norm,
                teacher_model=teacher_model,
                consistency_weight=consistency_weight,
                consistency_layer=consistency_layer,
                consistency_loss_type=consistency_loss_type,
                temporal_delta_consistency_weight=temporal_delta_consistency_weight,
            )
            history.append(row)

    return {
        "history": history,
        "selection_mode": selection_mode,
        "used_validation": False,
        "best_val_rmse": None,
        "best_epoch": int(selected_epoch),
        "selected_epoch": int(selected_epoch),
        "selected_checkpoint": "last",

        "finetune_mode": str(finetune_mode),
        "stage1_epochs": int(stage1_epochs) if finetune_mode == "two_stage" else 0,
        "stage2_epochs": int(stage2_epochs) if finetune_mode == "two_stage" else 0,
        "stage1_head_lr": float(stage1_head_lr),
        "stage2_whisper_lr": float(stage2_whisper_lr),
        "stage2_head_lr": float(stage2_head_lr),
        "stage2_unfreeze_encoder_layers": (
            [int(x) for x in stage2_unfreeze_encoder_layers]
            if stage2_unfreeze_encoder_layers is not None
            else []
        ),

        "target_normalizer": target_normalizer,

        "final_train_audio_rmse": float(history[-1]["train_audio_rmse"]) if len(history) > 0 else None,
        "final_train_audio_mae": float(history[-1]["train_audio_mae"]) if len(history) > 0 else None,
        "final_train_audio_pearson": float(history[-1]["train_audio_pearson"]) if len(history) > 0 else None,
        "final_train_audio_r2": float(history[-1]["train_audio_r2"]) if len(history) > 0 else None,
    }

# 特征归一化与导出
def apply_norm(h: torch.Tensor, mode: str, ln_post: Optional[nn.Module], skip_ln_post: bool) -> torch.Tensor:
    """对 [1, T, C] 的隐藏状态做归一化。"""
    if mode in [None, "none", "None"]:
        return h

    mode = str(mode)
    if mode == "layernorm":
        return F.layer_norm(h, h.shape[-1:])
    if mode == "std":
        std = h.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return h / std
    if mode == "ln_post":
        if skip_ln_post:
            return h
        if ln_post is None:
            raise ValueError("选择了 ln_post，但 ln_post 不可用")
        return ln_post(h)

    raise ValueError(f"未知归一化模式: {mode}")

@torch.inference_mode()
def extract_one_file_segments(
    wav_path: str,
    processor: WhisperProcessor,
    whisper_model: WhisperModel,
    ln_post: nn.Module,
    sampling_rate: int,
    segment_length: int,
    overlap_rate: float,
    single_layer: int,
    norm_for_single: str,
    normalize_audio: bool,
    device: torch.device,
    regression_head: Optional[nn.Module] = None,
    prototype_smoothing_alpha: float = 1.0,
) -> List[torch.Tensor]:
    """从单个音频提取多个 segment 的 [T, C] 特征。"""
    y, _ = librosa.load(wav_path, sr=sampling_rate, mono=True, dtype=np.float32)

    if normalize_audio:
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-8:
            y = y / peak

    bounds = split_segment_boundaries(len(y), segment_length, overlap_rate)

    out = []
    seg_preds = []

    for i, (start, end) in enumerate(bounds):
        seg = y[start:end]

        if len(seg) < segment_length and i > 0:
            need = segment_length - len(seg)
            step = int(segment_length * (1.0 - overlap_rate))
            prev_end = start + segment_length - step
            prev_start = max(0, prev_end - need)
            pad_values = y[prev_start:prev_end]
            if len(pad_values) < need:
                pad_values = np.concatenate(
                    [pad_values, np.zeros((need - len(pad_values),), dtype=np.float32)],
                    axis=0
                )
            seg = np.concatenate([seg, pad_values], axis=0)

        if len(seg) < segment_length:
            seg = np.concatenate([seg, np.zeros((segment_length - len(seg),), dtype=np.float32)], axis=0)

        inp = processor(seg, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inp.input_features.to(device)

        enc_out = whisper_model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        hs_all = list(enc_out.hidden_states)
        hs = hs_all[1:] if len(hs_all) >= 2 else hs_all

        if not (1 <= int(single_layer) <= len(hs)):
            raise ValueError(f"single_layer={single_layer} 超出范围 [1, {len(hs)}]")

        h = hs[int(single_layer) - 1]
        is_top = int(single_layer) == len(hs)
        skip_ln = (norm_for_single == "ln_post") and is_top
        h = apply_norm(h, mode=norm_for_single, ln_post=ln_post, skip_ln_post=skip_ln)
        out.append(h.squeeze(0).detach().cpu())

        if regression_head is not None:
            pred_mmse = float(regression_head(enc_out.last_hidden_state).view(-1).item())
            seg_preds.append(pred_mmse)

    alpha = float(prototype_smoothing_alpha)
    if len(out) >= 2 and 0.0 <= alpha < 1.0:
        feat_stack = torch.stack(out, dim=0)
        proto = feat_stack.mean(dim=0)

        if len(seg_preds) == len(out):
            pred_tensor = torch.tensor(seg_preds, dtype=feat_stack.dtype)
            pred_center = pred_tensor.mean()
            weights = 1.0 / (pred_tensor - pred_center).abs().clamp_min(1e-3)
            weights = weights / weights.sum().clamp_min(1e-6)
            proto = (feat_stack * weights.view(-1, 1, 1)).sum(dim=0)

        out = [alpha * feat + (1.0 - alpha) * proto for feat in feat_stack]

    return out

def stack_with_time_padding(feature_list: Sequence[torch.Tensor], target_t: Optional[int], pad_value: float = 0.0) -> torch.Tensor:
    """把若干个 [T, C] 特征补齐到相同时间长度后堆叠成 [N, T, C]。"""
    if len(feature_list) == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)

    feat_dim = int(feature_list[0].shape[1])
    if target_t is None:
        target_t = max(int(x.shape[0]) for x in feature_list)

    out = []
    for feat in feature_list:
        t = int(feat.shape[0])
        if int(feat.shape[1]) != feat_dim:
            raise ValueError("特征维度不一致，无法 stack")
        if t < target_t:
            pad = torch.full((target_t - t, feat_dim), float(pad_value), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)
        elif t > target_t:
            feat = feat[:target_t]
        out.append(feat)

    return torch.stack(out, dim=0)

def make_feature_file_tag(feature_prefix: str, single_layer: int, overlap_rate: float, segment_sec: int) -> str:
    """生成与原 notebook 尽量一致的文件标签。"""
    ov_str = f"{overlap_rate:.1f}".rstrip("0").rstrip(".") if overlap_rate % 1 != 0 else str(int(overlap_rate))
    if "." not in ov_str:
        ov_str = f"{ov_str}.0"
    sec_str = f"{segment_sec}s"
    return f"{feature_prefix}_single{single_layer}_{ov_str}_{sec_str}"

@torch.inference_mode()
def extract_and_save_mmse_features(
    dataset_key: str,
    dataset_paths: Dict[str, str],
    mmse_maps: Dict[str, Dict[str, float]],
    processor: WhisperProcessor,
    whisper_model: WhisperModel,
    regression_head: Optional[nn.Module],
    save_path: str,
    feature_prefix: str,
    single_layer: int,
    sampling_rate: int,
    segment_sec: int,
    overlap_rate: float,
    norm_for_single: str,
    normalize_audio: bool,
    device: torch.device,
    train_val_split_lookup: Optional[Dict[str, str]] = None,
    prototype_smoothing_alpha: float = 1.0,
    split_save_path: Optional[str] = None,
    dataset_file_lists: Optional[Dict[str, Sequence[str]]] = None,
    precomputed_missing_mmse_records: Optional[Sequence[dict]] = None,
):
    """导出训练集 / 测试集特征，并且标签改成 MMSE 浮点数。"""
    os.makedirs(save_path, exist_ok=True)

    segment_length = int(segment_sec * sampling_rate)
    ln_post = whisper_model.encoder.layer_norm

    split_order = [
        ("train_cc", "train"),
        ("train_cd", "train"),
        ("test_cc", "test"),
        ("test_cd", "test"),
    ]

    feats_train = []
    feats_test = []
    labels_train = []
    labels_test = []
    idx_train = []
    idx_test = []
    audio_entries_train = []
    audio_entries_test = []
    missing_mmse_records = list(precomputed_missing_mmse_records or [])

    next_audio_id_train = 0
    next_audio_id_test = 0

    for subset_key, split_name in split_order:
        if dataset_file_lists is not None and subset_key in dataset_file_lists:
            wav_files = list(dataset_file_lists[subset_key])
        else:
            wav_files = list_wav_files(dataset_paths[subset_key], dataset_key=dataset_key)

        for wav_path in tqdm(wav_files, desc=f"[extract][{split_name}] {subset_key}", ncols=100):
            mmse = get_mmse_for_wav(dataset_key, wav_path, mmse_maps[subset_key])
            if mmse is None:
                if "missing_mmse_records" not in locals():
                    missing_mmse_records = []
                missing_mmse_records.append(
                    make_missing_mmse_record_for_wts(dataset_key, subset_key, wav_path)
                )
                continue

            seg_feats = extract_one_file_segments(
                wav_path=wav_path,
                processor=processor,
                whisper_model=whisper_model,
                ln_post=ln_post,
                sampling_rate=sampling_rate,
                segment_length=segment_length,
                overlap_rate=overlap_rate,
                single_layer=single_layer,
                norm_for_single=norm_for_single,
                normalize_audio=normalize_audio,
                device=device,
                regression_head=regression_head,
                prototype_smoothing_alpha=prototype_smoothing_alpha,
            )

            wav_path_canonical = canonical_wav_path(wav_path)

            if split_name == "train":
                audio_id = next_audio_id_train
                next_audio_id_train += 1

                logical_split = "train"
                if train_val_split_lookup is not None:
                    logical_split = train_val_split_lookup.get(wav_path_canonical, "train")

                row = {
                    "audio_id": int(audio_id),
                    "target_mmse": float(mmse),
                    "split": logical_split,
                    "wav_path": wav_path_canonical,
                    "wav_name": Path(wav_path).name,
                }

                if dataset_key == "Pitt":
                    subject_key, visit_index = _extract_pitt_subject_visit_from_wav_path(wav_path)
                    row["subject_id"] = None if subject_key is None else int(subject_key)
                    row["visit_index"] = None if visit_index is None else int(visit_index)

                audio_entries_train.append(row)
                for seg_id, feat in enumerate(seg_feats):
                    feats_train.append(feat)
                    labels_train.append(float(mmse))
                    idx_train.append((int(audio_id), int(seg_id)))
            else:
                audio_id = next_audio_id_test
                next_audio_id_test += 1

                row = {
                    "audio_id": int(audio_id),
                    "target_mmse": float(mmse),
                    "split": "test",
                    "wav_path": wav_path_canonical,
                    "wav_name": Path(wav_path).name,
                }

                if dataset_key == "Pitt":
                    subject_key, visit_index = _extract_pitt_subject_visit_from_wav_path(wav_path)
                    row["subject_id"] = None if subject_key is None else int(subject_key)
                    row["visit_index"] = None if visit_index is None else int(visit_index)

                audio_entries_test.append(row)
                for seg_id, feat in enumerate(seg_feats):
                    feats_test.append(feat)
                    labels_test.append(float(mmse))
                    idx_test.append((int(audio_id), int(seg_id)))

    if len(feats_train) == 0 or len(feats_test) == 0:
        raise RuntimeError("导出的特征为空，请检查路径、MMSE 元数据和参数配置")

    target_t = max(
        max(int(x.shape[0]) for x in feats_train),
        max(int(x.shape[0]) for x in feats_test),
    )
    print(f"[PAD] 统一时间长度 target_t={target_t}")

    features_train = stack_with_time_padding(feats_train, target_t=target_t)
    features_test = stack_with_time_padding(feats_test, target_t=target_t)

    labels1 = torch.tensor(labels_train, dtype=torch.float32)
    labels2 = torch.tensor(labels_test, dtype=torch.float32)
    indices_train = np.asarray(idx_train, dtype=np.int64)
    indices_test = np.asarray(idx_test, dtype=np.int64)

    file_tag = make_feature_file_tag(feature_prefix, single_layer, overlap_rate, segment_sec)

    p_feature_train = os.path.join(save_path, f"feature_{file_tag}_train.pt")
    p_feature_test = os.path.join(save_path, f"feature_{file_tag}_test.pt")
    p_label_train = os.path.join(save_path, f"labels1_{file_tag}_train.pt")
    p_label_test = os.path.join(save_path, f"labels2_{file_tag}_test.pt")
    p_index_train = os.path.join(save_path, f"indices_{file_tag}_train.pt")
    p_index_test = os.path.join(save_path, f"indices_{file_tag}_test.pt")

    if split_save_path is None or str(split_save_path).strip() in ["", "None", "none", "null", "Null"]:
        p_split_meta = os.path.join(save_path, f"train_val_split_{file_tag}.json")
    else:
        p_split_meta = str(split_save_path).strip()
        os.makedirs(os.path.dirname(p_split_meta), exist_ok=True)
    torch.save(features_train, p_feature_train)
    torch.save(features_test, p_feature_test)
    torch.save(labels1, p_label_train)
    torch.save(labels2, p_label_test)
    torch.save(indices_train, p_index_train)
    torch.save(indices_test, p_index_test)

    split_metadata = {
        "format_version": 3 if (dataset_key == "Pitt" and dataset_file_lists is not None) else 1,
        "task_type": "mmse_regression",
        "dataset_key": dataset_key,
        "dataset": dataset_key,
        "file_tag": file_tag,
        "feature_save_path": str(Path(save_path).resolve()),
        "split_strategy": (
            PITT_SPLIT_POLICY
            if (dataset_key == "Pitt" and dataset_file_lists is not None)
            else RUN_PROTOCOL_TAG
        ),
        "subject_disjoint": bool(dataset_key == "Pitt" and dataset_file_lists is not None),
        "train_test_subject_overlap": (
            0 if (dataset_key == "Pitt" and dataset_file_lists is not None) else None
        ),
        "train_audio_ids": [int(x["audio_id"]) for x in audio_entries_train if str(x["split"]) == "train"],
        "val_audio_ids": [int(x["audio_id"]) for x in audio_entries_train if str(x["split"]) == "val"],
        "test_audio_ids": [int(x["audio_id"]) for x in audio_entries_test],
        "train_entries": audio_entries_train,
        "test_entries": audio_entries_test,
    }
    save_json_file(split_metadata, p_split_meta)

    pitt_audio_lookup_sidecar = None
    if dataset_key == "Pitt" and dataset_file_lists is not None:
        pitt_audio_lookup_sidecar = os.path.join(
            save_path, f"train_val_split_{file_tag}.json"
        )
        save_json_file(split_metadata, pitt_audio_lookup_sidecar)
        print("[Pitt split] feature audio-id sidecar ->", pitt_audio_lookup_sidecar)

    missing_mmse_csv_path = save_missing_mmse_records_for_wts(
        missing_records=missing_mmse_records,
        save_path=save_path,
        file_tag=file_tag,
    )

    print("[SAVE] ->", save_path)
    print("  -", p_feature_train)
    print("  -", p_feature_test)
    print("  -", p_label_train)
    print("  -", p_label_test)
    print("  -", p_index_train)
    print("  -", p_index_test)
    print("  -", p_split_meta)

    return {
        "file_tag": file_tag,
        "feature_train": p_feature_train,
        "feature_test": p_feature_test,
        "label_train": p_label_train,
        "label_test": p_label_test,
        "index_train": p_index_train,
        "index_test": p_index_test,
        "train_val_split_path": p_split_meta,
        "pitt_audio_lookup_sidecar": pitt_audio_lookup_sidecar,
        "missing_mmse_csv_path": missing_mmse_csv_path,
        "missing_mmse_count": int(0 if missing_mmse_records is None else len(missing_mmse_records)),
    }

# 单次实验与批量入口
def run_experiment_mmse():
    set_seed(SEED)

    dataset_paths = DATASETS[DATASET_KEY]
    mmse_maps = load_mmse_maps_for_dataset(DATASET_KEY)

    dataset_file_lists = None
    pitt_missing_mmse_records = []
    pitt_split_info = None
    pitt_visit_manifest_path = None
    pitt_subject_manifest_path = None

    if DATASET_KEY == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT:
        all_pitt_mmse_map = mmse_maps["train_cc"]  # Pitt 四个 map 共享同一个 visit-level all_map
        (
            dataset_file_lists,
            pitt_visit_df,
            pitt_subject_df,
            pitt_missing_mmse_records,
            pitt_split_info,
        ) = _pitt_collect_and_make_subject_disjoint_split_for_vanilla(
            dataset_paths=dataset_paths,
            mmse_map=all_pitt_mmse_map,
        )
        (
            pitt_visit_manifest_path,
            pitt_subject_manifest_path,
        ) = _save_pitt_split_manifests_for_vanilla(
            pitt_visit_df,
            pitt_subject_df,
        )

    print(f"[REPRO] reset seed before run_experiment_mmse -> {SEED}")

    print("\n[DATASET]")
    for k, v in dataset_paths.items():
        print(" ", k, ":", v)

    experiment_config = build_experiment_config_snapshot()

    experiment_config_path = os.path.join(FINETUNED_WHISPER_DIR, "experiment_config.json")
    save_json_file(experiment_config, experiment_config_path)
    print("[SAVE] experiment_config ->", experiment_config_path)

    feature_experiment_config_path = os.path.join(FEATURE_SAVE_PATH, "experiment_config.json")
    save_json_file(experiment_config, feature_experiment_config_path)
    print("[SAVE] feature experiment_config ->", feature_experiment_config_path)

    if dataset_file_lists is not None:
        full_train_cc_files = list(dataset_file_lists["train_cc"])
        full_train_cd_files = list(dataset_file_lists["train_cd"])
    else:
        full_train_cc_files = list_wav_files(dataset_paths["train_cc"], dataset_key=DATASET_KEY)
        full_train_cd_files = list_wav_files(dataset_paths["train_cd"], dataset_key=DATASET_KEY)

    missing_mmse_records = list(pitt_missing_mmse_records)

    def filter_wavs_with_mmse(wav_files, subset_name: str, mmse_map: dict):
        kept = []
        for wav_path in wav_files:
            mmse = get_mmse_for_wav(DATASET_KEY, wav_path, mmse_map)
            if mmse is None:
                missing_mmse_records.append(
                    make_missing_mmse_record_for_wts(DATASET_KEY, subset_name, wav_path)
                )
                continue
            kept.append(wav_path)
        return kept

    full_train_cc_files = filter_wavs_with_mmse(
        full_train_cc_files,
        "train_cc",
        mmse_maps["train_cc"],
    )
    full_train_cd_files = filter_wavs_with_mmse(
        full_train_cd_files,
        "train_cd",
        mmse_maps["train_cd"],
    )

    tr_cc = list(full_train_cc_files)
    tr_cd = list(full_train_cd_files)
    va_cc = []
    va_cd = []

    active_train_val_split_path_for_run = None

    train_val_split_lookup = {}
    for wav_path in tr_cc + tr_cd:
        train_val_split_lookup[canonical_wav_path(wav_path)] = "train"

    is_pitt_subject_disjoint = bool(
        DATASET_KEY == "Pitt" and dataset_file_lists is not None
    )
    protocol_meta = {
        "format_version": 3 if is_pitt_subject_disjoint else 2,
        "task_type": "mmse_regression",
        "dataset_key": DATASET_KEY,
        "source": "VanillaFT_pitt_new.py",
        "run_protocol": RUN_PROTOCOL_TAG,
        "split_strategy": (
            PITT_SPLIT_POLICY
            if is_pitt_subject_disjoint
            else "no_fold_official_train_all_used_for_whisper_finetune"
        ),
        "subject_disjoint": is_pitt_subject_disjoint,
        "train_test_subject_overlap": 0 if is_pitt_subject_disjoint else None,
        "train_val_split_path": None,
        "train_cc_count": int(len(tr_cc)),
        "train_cd_count": int(len(tr_cd)),
        "train_total_count": int(len(tr_cc) + len(tr_cd)),
        "test_cc_count": (
            int(len(dataset_file_lists["test_cc"])) if is_pitt_subject_disjoint else None
        ),
        "test_cd_count": (
            int(len(dataset_file_lists["test_cd"])) if is_pitt_subject_disjoint else None
        ),
        "test_total_count": (
            int(len(dataset_file_lists["test_cc"]) + len(dataset_file_lists["test_cd"]))
            if is_pitt_subject_disjoint else None
        ),
        "test_policy": "not_used_by_whisper_finetune_export_only",
        "train_files": [canonical_wav_path(p) for p in tr_cc + tr_cd],
        "val_files": [],
        "pitt_split_info": pitt_split_info,
        "pitt_visit_manifest_path": pitt_visit_manifest_path,
        "pitt_subject_manifest_path": pitt_subject_manifest_path,
        "note": (
            "Pitt uses the subject-disjoint final train pool for Whisper finetuning; "
            "the subject-disjoint final test pool is not used during Whisper finetuning and is exported only as test features."
            if is_pitt_subject_disjoint
            else
            "official_train is used completely for Whisper temporal finetuning; "
            "official_test is not used during Whisper finetuning and is exported only as test features."
        ),
    }

    os.makedirs(os.path.dirname(LOCAL_FEATURE_SPLIT_SAVE_PATH), exist_ok=True)
    save_json_file(protocol_meta, LOCAL_FEATURE_SPLIT_SAVE_PATH)

    if is_pitt_subject_disjoint:
        print("[SPLIT] Pitt subject-disjoint final train/test protocol for upstream finetune.")
        print("[SPLIT] final_train_cc -> train:", len(tr_cc))
        print("[SPLIT] final_train_cd -> train:", len(tr_cd))
        print("[SPLIT] fold_val disabled:", len(va_cc) + len(va_cd))
        print("[SPLIT] final subject-disjoint test is not used by Whisper finetune; it will be exported only.")
    else:
        print("[SPLIT] LogNCDE-style / official train-test protocol for upstream finetune.")
        print("[SPLIT] official_train_cc -> train:", len(tr_cc))
        print("[SPLIT] official_train_cd -> train:", len(tr_cd))
        print("[SPLIT] fold_val disabled:", len(va_cc) + len(va_cd))
        print("[SPLIT] official_test is not used by Whisper finetune; it will be exported only.")
    print("[SPLIT] active split path for this run ->", active_train_val_split_path_for_run)
    print("[SPLIT] local metadata path ->", LOCAL_FEATURE_SPLIT_SAVE_PATH)

    train_audio_records_cc, next_aid = build_audio_sequence_records_from_files_with_mmse(
        tr_cc,
        dataset_key=DATASET_KEY,
        mmse_map=mmse_maps["train_cc"],
        start_audio_id=0,
        sampling_rate=SAMPLING_RATE,
        segment_length=SEGMENT_LENGTH,
        overlap_rate=OVERLAP_RATE,
        max_segments_per_audio=MAX_SEGMENTS_PER_AUDIO,
    )

    train_audio_records_cd, next_aid = build_audio_sequence_records_from_files_with_mmse(
        tr_cd,
        dataset_key=DATASET_KEY,
        mmse_map=mmse_maps["train_cd"],
        start_audio_id=next_aid,
        sampling_rate=SAMPLING_RATE,
        segment_length=SEGMENT_LENGTH,
        overlap_rate=OVERLAP_RATE,
        max_segments_per_audio=MAX_SEGMENTS_PER_AUDIO,
    )

    train_audio_records = train_audio_records_cc + train_audio_records_cd

    print(
        f"[AUDIO-LEVEL TEMPORAL] train_audio_records={len(train_audio_records)} "
        f"| protocol={RUN_PROTOCOL_TAG}"
    )
    print(
        "[WHISPER TEMPORAL FINETUNE POLICY] "
        "official_train 全部参与 Whisper temporal 微调；"
        "official_test 不参与 Whisper 微调，也不参与 Whisper checkpoint selection；"
        "official_test 仅在特征导出阶段生成 test feature。"
    )
    if len(train_audio_records) == 0:
        raise RuntimeError("audio-level 训练样本数为 0，请检查数据路径、split json 和 MMSE 元数据。")

    resolved_consistency_layer = int(CONSISTENCY_LAYER) if int(CONSISTENCY_LAYER) > 0 else int(SINGLE_LAYER)

    export_info = None
    finetune_result = None
    resolved_unfreeze_layers = list(RESOLVED_UNFREEZE_ENCODER_LAYERS)

    if REEXPORT_ONLY:
        active_finetuned_dir = (
            str(REEXPORT_FROM_FINETUNED_DIR).strip()
            if str(REEXPORT_FROM_FINETUNED_DIR).strip() != ""
            else str(FINETUNED_WHISPER_DIR).strip()
        )

        if not os.path.isdir(active_finetuned_dir):
            raise FileNotFoundError(f"找不到已保存的微调 Whisper 目录: {active_finetuned_dir}")

        print("[MODE] REEXPORT_ONLY=True")
        print("[LOAD] finetuned Whisper from:", active_finetuned_dir)
        print("[SKIP] 跳过 Whisper 微调，只重新导出特征。")

        processor = WhisperProcessor.from_pretrained(
            active_finetuned_dir,
            local_files_only=LOCAL_FILES_ONLY,
        )

        whisper_for_extract = WhisperModel.from_pretrained(
            active_finetuned_dir,
            local_files_only=LOCAL_FILES_ONLY,
        ).to(DEVICE)
        whisper_for_extract.eval()

        export_info = extract_and_save_mmse_features(
            dataset_key=DATASET_KEY,
            dataset_paths=dataset_paths,
            mmse_maps=mmse_maps,
            processor=processor,
            whisper_model=whisper_for_extract,
            regression_head=None,
            save_path=FEATURE_SAVE_PATH,
            feature_prefix=FEATURE_PREFIX,
            single_layer=SINGLE_LAYER,
            sampling_rate=SAMPLING_RATE,
            segment_sec=SEGMENT_SEC,
            overlap_rate=OVERLAP_RATE,
            norm_for_single=NORM_FOR_SINGLE,
            normalize_audio=NORMALIZE_AUDIO,
            device=DEVICE,
            train_val_split_lookup=train_val_split_lookup,
            prototype_smoothing_alpha=PROTOTYPE_SMOOTHING_ALPHA,
            split_save_path=LOCAL_FEATURE_SPLIT_SAVE_PATH,
            dataset_file_lists=dataset_file_lists,
            precomputed_missing_mmse_records=pitt_missing_mmse_records,
        )

        run_manifest = {
            "mode": "reexport_only",
            "dataset_key": DATASET_KEY,
            "single_layer": int(SINGLE_LAYER),
            "active_layer_name": str(ACTIVE_LAYER_NAME),
            "finetuned_whisper_dir": active_finetuned_dir,
            "feature_save_path": str(FEATURE_SAVE_PATH),
            "feature_export": export_info,
            "reexport_only": True,
        }

        run_manifest_path = os.path.join(FINETUNED_WHISPER_DIR, "run_manifest_reexport.json")
        save_json_file(run_manifest, run_manifest_path)
        print("[SAVE] run_manifest ->", run_manifest_path)

        print("\n" + "=" * 90)
        print("已从保存好的 Whisper checkpoint 重新导出特征。")
        print(f"FEATURE_SAVE_PATH = {FEATURE_SAVE_PATH}")
        print(f"FILE_TAG          = {export_info['file_tag']}")
        print("=" * 90)

        return {
            "finetune_result": None,
            "export_info": export_info,
            "run_manifest_path": run_manifest_path,
        }

    else:
        processor = WhisperProcessor.from_pretrained(
            BASE_WHISPER_MODEL,
            local_files_only=LOCAL_FILES_ONLY,
        )

        teacher_model = None

        need_teacher_model = (
            float(CONSISTENCY_WEIGHT) > 0.0
            or float(TEMPORAL_DELTA_CONSISTENCY_WEIGHT) > 0.0
        )

        if need_teacher_model:
            teacher_model = WhisperModel.from_pretrained(
                BASE_WHISPER_MODEL,
                local_files_only=LOCAL_FILES_ONLY,
            ).to(DEVICE)
            freeze_module(teacher_model)
            print(
                f"[TEACHER] enabled | hidden_weight={CONSISTENCY_WEIGHT:.4f} "
                f"| delta_weight={TEMPORAL_DELTA_CONSISTENCY_WEIGHT:.4f} "
                f"| layer={resolved_consistency_layer} | loss={CONSISTENCY_LOSS_TYPE}"
            )
        else:
            print("[TEACHER] disabled")

        model = WhisperAudioTemporalMMSERegressor(
            whisper_model_name_or_path=BASE_WHISPER_MODEL,
            local_files_only=LOCAL_FILES_ONLY,
            feature_layer=SINGLE_LAYER,
            temporal_head_type=TEMPORAL_HEAD_TYPE,
            temporal_hidden_dim=TEMPORAL_HIDDEN_DIM,
            temporal_num_layers=TEMPORAL_NUM_LAYERS,
            temporal_dropout=TEMPORAL_DROPOUT,
            segment_pool=TEMPORAL_SEGMENT_POOL,
        ).to(DEVICE)

        resolved_unfreeze_layers = configure_whisper_trainable_layers(
            model,
            RESOLVED_UNFREEZE_ENCODER_LAYERS,
        )

        print(f"[UNFREEZE] encoder layers = {resolved_unfreeze_layers}")

        if len(resolved_unfreeze_layers) == 0:
            print(
                "[SANITY CHECK MODE] UNFREEZE_ENCODER_LAYERS=0，"
                "当前冻结整个 Whisper，只训练 temporal head。"
            )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"[MODEL] trainable={trainable} total={total} ratio={trainable / max(total, 1):.4f}")

        if bool(WHISPER_FINETUNE_USE_VAL):
            raise RuntimeError(
                "temporal 版本要求 WHISPER_FINETUNE_USE_VAL=False。"
                "fold_val 只能供下游 LRU 阶段使用，不能参与 Whisper temporal 微调。"
            )

        finetune_result = run_finetune_mmse_temporal_audio_level(
            model=model,
            processor=processor,
            train_records=train_audio_records,
            sampling_rate=SAMPLING_RATE,
            segment_length=SEGMENT_LENGTH,
            overlap_rate=OVERLAP_RATE,
            normalize_audio=NORMALIZE_AUDIO,
            audio_batch_size=AUDIO_BATCH_SIZE,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            grad_clip_norm=GRAD_CLIP_NORM,
            device=DEVICE,
            regression_loss_type=REGRESSION_LOSS_TYPE,
            smooth_l1_beta=SMOOTH_L1_BETA,
            teacher_model=teacher_model,
            consistency_weight=CONSISTENCY_WEIGHT,
            consistency_layer=resolved_consistency_layer,
            consistency_loss_type=CONSISTENCY_LOSS_TYPE,
            temporal_delta_consistency_weight=TEMPORAL_DELTA_CONSISTENCY_WEIGHT,
            seed=SEED,
            target_normalize=TEMPORAL_TARGET_NORMALIZE,
            target_normalize_eps=TEMPORAL_TARGET_NORMALIZE_EPS,

            finetune_mode=TEMPORAL_FINETUNE_MODE,
            stage1_epochs=TEMPORAL_STAGE1_EPOCHS,
            stage2_epochs=TEMPORAL_STAGE2_EPOCHS,
            stage1_head_lr=TEMPORAL_STAGE1_HEAD_LR,
            stage2_whisper_lr=TEMPORAL_STAGE2_WHISPER_LR,
            stage2_head_lr=TEMPORAL_STAGE2_HEAD_LR,
            stage2_unfreeze_encoder_layers=RESOLVED_UNFREEZE_ENCODER_LAYERS,
)
        model.whisper.save_pretrained(FINETUNED_WHISPER_DIR)
        processor.save_pretrained(FINETUNED_WHISPER_DIR)

        temporal_head_path = os.path.join(FINETUNED_WHISPER_DIR, "audio_temporal_head.pt")
        torch.save(model.temporal_head.state_dict(), temporal_head_path)

        finetune_metrics_path = os.path.join(FINETUNED_WHISPER_DIR, "finetune_metrics.json")
        with open(finetune_metrics_path, "w", encoding="utf-8") as f:
            json.dump(finetune_result, f, ensure_ascii=False, indent=2)

        print("[SAVE] FINETUNED_WHISPER_DIR ->", FINETUNED_WHISPER_DIR)
        print("[SAVE] temporal head         ->", temporal_head_path)
        print("[SAVE] finetune metrics      ->", finetune_metrics_path)
        print("[SELECT] Whisper checkpoint selection =", finetune_result["selection_mode"])
        print("[SELECT] selected_checkpoint =", finetune_result["selected_checkpoint"])
        print("[SELECT] selected_epoch =", finetune_result["selected_epoch"])
        print("[TRAIN] final_train_audio_rmse =", finetune_result["final_train_audio_rmse"])
    if not SKIP_FEATURE_EXPORT:
        whisper_for_extract = model.whisper.to(DEVICE)
        whisper_for_extract.eval()

        regression_head_for_extract = None

        export_info = extract_and_save_mmse_features(
            dataset_key=DATASET_KEY,
            dataset_paths=dataset_paths,
            mmse_maps=mmse_maps,
            processor=processor,
            whisper_model=whisper_for_extract,
            regression_head=regression_head_for_extract,
            save_path=FEATURE_SAVE_PATH,
            feature_prefix=FEATURE_PREFIX,
            single_layer=SINGLE_LAYER,
            sampling_rate=SAMPLING_RATE,
            segment_sec=SEGMENT_SEC,
            overlap_rate=OVERLAP_RATE,
            norm_for_single=NORM_FOR_SINGLE,
            normalize_audio=NORMALIZE_AUDIO,
            device=DEVICE,
            train_val_split_lookup=train_val_split_lookup,  # official_train 内全部标记为 train
            prototype_smoothing_alpha=PROTOTYPE_SMOOTHING_ALPHA,
            split_save_path=LOCAL_FEATURE_SPLIT_SAVE_PATH,
            dataset_file_lists=dataset_file_lists,
            precomputed_missing_mmse_records=pitt_missing_mmse_records,
        )
    run_manifest = {
        "format_version": 1,
        "task_type": "mmse_regression",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_key": DATASET_KEY,
        "run_protocol": RUN_PROTOCOL_TAG,
        "train_semantics": (
            "pitt_subject_disjoint_final_train" if is_pitt_subject_disjoint else "official_train"
        ),
        "val_semantics": "not_used_by_upstream_whisper_finetune",
        "test_semantics": (
            "pitt_subject_disjoint_final_test_export_only_not_used_by_upstream_finetune"
            if is_pitt_subject_disjoint
            else "official_test_export_only_not_used_by_upstream_finetune"
        ),
        "split_strategy": (PITT_SPLIT_POLICY if is_pitt_subject_disjoint else RUN_PROTOCOL_TAG),
        "subject_disjoint": is_pitt_subject_disjoint,
        "train_test_subject_overlap": 0 if is_pitt_subject_disjoint else None,
        "pitt_split_info": pitt_split_info,
        "pitt_visit_manifest_path": pitt_visit_manifest_path,
        "pitt_subject_manifest_path": pitt_subject_manifest_path,
        "active_fold_id": None,
        "base_whisper_model": str(Path(BASE_WHISPER_MODEL).resolve()),
        "finetuned_whisper_dir": str(Path(FINETUNED_WHISPER_DIR).resolve()),
        "feature_save_path": str(Path(FEATURE_SAVE_PATH).resolve()),
        "train_val_split_path": None,
        "local_feature_metadata_path": str(LOCAL_FEATURE_SPLIT_SAVE_PATH),
        "seed": SEED,

        "val_ratio": VAL_RATIO,
        "segment_sec": SEGMENT_SEC,
        "overlap_rate": OVERLAP_RATE,
        "pooling_type": POOLING_TYPE,
        "regression_loss_type": REGRESSION_LOSS_TYPE,
        "smooth_l1_beta": SMOOTH_L1_BETA,
        "unfreeze_encoder_layers": int(UNFREEZE_ENCODER_LAYERS),
        "resolved_unfreeze_encoder_layers": resolved_unfreeze_layers,
        "consistency_weight": CONSISTENCY_WEIGHT,
        "consistency_layer": resolved_consistency_layer,
        "audio_feature_consistency_weight": AUDIO_FEATURE_CONSISTENCY_WEIGHT,
        "audio_prediction_consistency_weight": AUDIO_PREDICTION_CONSISTENCY_WEIGHT,
        "single_layer_for_export": SINGLE_LAYER,
        "norm_for_single": NORM_FOR_SINGLE,
        "prototype_smoothing_alpha": PROTOTYPE_SMOOTHING_ALPHA,

        "finetune_variant": (
            "whisper_pool_linear_segment_pred_mean_audio_loss_two_stage"
            if str(globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")).strip().lower()
            in {"segment_pred_mean", "segment-pred-mean", "prediction_mean", "prediction-mean", "segment_prediction_mean"}
            else (
                "whisper_pool_linear_segment_audio_balanced_loss_two_stage"
                if str(globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")).strip().lower() == "segment_audio_balanced"
                else "whisper_audio_temporal_head_audio_level_loss_two_stage"
            )
        ),
        "upstream_mmse_supervision_level": str(globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")),
        "temporal_finetune_mode": str(TEMPORAL_FINETUNE_MODE),
        "temporal_stage1_epochs": int(TEMPORAL_STAGE1_EPOCHS),
        "temporal_stage2_epochs": int(TEMPORAL_STAGE2_EPOCHS),
        "temporal_stage1_head_lr": float(TEMPORAL_STAGE1_HEAD_LR),
        "temporal_stage2_whisper_lr": float(TEMPORAL_STAGE2_WHISPER_LR),
        "temporal_stage2_head_lr": float(TEMPORAL_STAGE2_HEAD_LR),

        "audio_batch_size": int(AUDIO_BATCH_SIZE),
        "temporal_head_type": str(TEMPORAL_HEAD_TYPE),
        "temporal_hidden_dim": int(TEMPORAL_HIDDEN_DIM),
        "temporal_num_layers": int(TEMPORAL_NUM_LAYERS),
        "temporal_dropout": float(TEMPORAL_DROPOUT),
        "temporal_segment_pool": str(TEMPORAL_SEGMENT_POOL),
        "temporal_delta_consistency_weight": float(TEMPORAL_DELTA_CONSISTENCY_WEIGHT),

        "temporal_target_normalize": bool(TEMPORAL_TARGET_NORMALIZE),
        "temporal_target_normalize_eps": float(TEMPORAL_TARGET_NORMALIZE_EPS),
        "temporal_target_normalizer": (
            finetune_result.get("target_normalizer", None)
            if isinstance(finetune_result, dict)
            else None
        ),

        "whisper_finetune_use_val": bool(WHISPER_FINETUNE_USE_VAL),
        "whisper_finetune_checkpoint_selection": str(WHISPER_FINETUNE_CHECKPOINT_SELECTION),
        "whisper_finetune_path_tag": str(WHISPER_FINETUNE_PATH_TAG),
        "finetune_result": finetune_result,
        "feature_export": export_info,
        "reexport_only": bool(REEXPORT_ONLY),
        "reexport_from_finetuned_dir": (
            str(REEXPORT_FROM_FINETUNED_DIR).strip()
            if str(REEXPORT_FROM_FINETUNED_DIR).strip() != ""
            else str(FINETUNED_WHISPER_DIR).strip()
        ),
        "save_params_in_path": (
            "all"
            if globals().get("SAVE_PARAMS_IN_PATH", None) is None
            else globals().get("SAVE_PARAMS_IN_PATH", None)
        ),
    }

    run_manifest_path = os.path.join(FINETUNED_WHISPER_DIR, "run_manifest.json")
    save_json_file(run_manifest, run_manifest_path)
    print("[SAVE] run_manifest ->", run_manifest_path)

    if export_info is not None:
        print("\n" + "=" * 90)
        print("微调 + 特征导出已完成。")
        print("后续如果你要在 JAX/LRU notebook 中使用这些特征，建议关注：")
        print(f"  FEATURE_SAVE_PATH      = {FEATURE_SAVE_PATH}")
        print(f"  FILE_TAG               = {export_info['file_tag']}")
        print(f"  METADATA_PATH          = {export_info['train_val_split_path']}")
        print("  TRAIN_VAL_SPLIT_PATH   = None  # 新 protocol 下不再使用 fold split")
        print("=" * 90)

    return {
        "finetune_result": finetune_result,
        "export_info": export_info,
        "run_manifest_path": run_manifest_path,
    }

def run_all_experiments_mmse():
    """批量执行所有实验。"""
    global DATASET_KEY
    global UNFREEZE_ENCODER_LAYERS

    dataset_keys = [str(x) for x in DATASET_KEYS]
    unfreeze_list = [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]

    if len(dataset_keys) == 0:
        raise ValueError("DATASET_KEYS 为空，至少需要一个数据集。")
    if len(unfreeze_list) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST 为空，至少需要一个解冻层数。")

    for dataset_key in dataset_keys:
        if dataset_key not in DATASETS:
            raise ValueError(f"未知 DATASET_KEY={dataset_key}")
        if dataset_key not in MMSE_META_PATHS:
            raise ValueError(f"当前 notebook 暂未配置 {dataset_key} 的 MMSE 元数据")

    resolved_single_layers = resolve_single_layers(
        SINGLE_LAYERS,
        whisper_model_name_or_path=BASE_WHISPER_MODEL,
        local_files_only=LOCAL_FILES_ONLY,
    )

    print("[BATCH] DATASET_KEYS ->", dataset_keys)
    print("[BATCH] UNFREEZE_ENCODER_LAYERS_LIST ->", unfreeze_list)
    print("[BATCH] SINGLE_LAYERS ->", resolved_single_layers)
    print("[BATCH] RUN_PROTOCOL_TAG ->", RUN_PROTOCOL_TAG)
    print("[BATCH] UPSTREAM_MMSE_SUPERVISION_LEVEL ->", UPSTREAM_MMSE_SUPERVISION_LEVEL)
    print("[BATCH] fold loop disabled")

    results = []

    for dataset_key in dataset_keys:
        for unfreeze_count in unfreeze_list:
            DATASET_KEY = str(dataset_key)
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "#" * 120)
            print("[BATCH DATASET / UNFREEZE]")
            print("  DATASET_KEY =", DATASET_KEY)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("  UPSTREAM_MMSE_SUPERVISION_LEVEL =", UPSTREAM_MMSE_SUPERVISION_LEVEL)
            print("#" * 120)

            for single_layer in resolved_single_layers:
                print("\n" + "=" * 120)
                print(
                    f"[BATCH] START | "
                    f"dataset={DATASET_KEY} | "
                    f"single_layer={int(single_layer)} | "
                    f"unfreeze={int(UNFREEZE_ENCODER_LAYERS)} | "
                    f"protocol={RUN_PROTOCOL_TAG} | "
                    f"supervision={UPSTREAM_MMSE_SUPERVISION_LEVEL}"
                )
                print("=" * 120)

                set_seed(SEED)
                print(f"[REPRO] reset seed before apply_run_context -> {SEED}")

                run_context = apply_run_context(single_layer=single_layer, fold_id=None)
                run_result = run_experiment_mmse()

                run_result["run_context"] = run_context
                run_result["dataset_key"] = DATASET_KEY
                run_result["unfreeze_encoder_layers"] = int(UNFREEZE_ENCODER_LAYERS)
                run_result["single_layer"] = int(single_layer)
                run_result["upstream_mmse_supervision_level"] = str(UPSTREAM_MMSE_SUPERVISION_LEVEL)

                results.append(run_result)

    summary_dir = os.path.join(
        FEATURE_ROOT_DIR,
        "_batch_summaries",
        f"ft_{FEATURE_MODEL_SAVE_TAG}",
    )
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "batch_run_summary_multi_dataset_unfreeze.json")

    serializable_results = []
    for item in results:
        serializable_results.append(
            {
                "dataset_key": item.get("dataset_key"),
                "single_layer": item.get("single_layer"),
                "unfreeze_encoder_layers": item.get("unfreeze_encoder_layers"),
                "upstream_mmse_supervision_level": item.get("upstream_mmse_supervision_level"),
                "run_context": item.get("run_context", {}),
                "run_manifest_path": item.get("run_manifest_path"),
                "best_val_rmse": item.get("finetune_result", {}).get("best_val_rmse"),
                "best_epoch": item.get("finetune_result", {}).get("best_epoch"),
                "feature_export": item.get("export_info"),
            }
        )

    save_json_file(
        {
            "format_version": 3,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "dataset_keys": dataset_keys,
            "model_name": MODEL_NAME,
            "feature_model_save_tag": FEATURE_MODEL_SAVE_TAG,
            "run_protocol": RUN_PROTOCOL_TAG,
            "fold_ids": [],
            "fold_policy": "disabled",
            "train_semantics": (
                "pitt_subject_disjoint_final_train"
                if dataset_keys == ["Pitt"] and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_train"
            ),
            "test_semantics": (
                "pitt_subject_disjoint_final_test_export_only_not_used_by_upstream_finetune"
                if dataset_keys == ["Pitt"] and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_test_export_only_not_used_by_upstream_finetune"
            ),
            "pitt_split_policy": (
                PITT_SPLIT_POLICY
                if "Pitt" in dataset_keys and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else None
            ),
            "finetune_variant": (
                "whisper_poolhead_segment_pred_mean_audio_loss_two_stage"
                if str(globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")).strip().lower()
                in {"segment_pred_mean", "segment-pred-mean", "prediction_mean", "prediction-mean", "segment_prediction_mean"}
                else "whisper_poolhead_segment_audio_balanced_loss_two_stage"
            ),
            "upstream_mmse_supervision_level": str(globals().get("UPSTREAM_MMSE_SUPERVISION_LEVEL", "audio")),
            "upstream_mmse_supervision_level": str(UPSTREAM_MMSE_SUPERVISION_LEVEL),
            "temporal_finetune_mode": str(TEMPORAL_FINETUNE_MODE),
            "temporal_stage1_epochs": int(TEMPORAL_STAGE1_EPOCHS),
            "temporal_stage2_epochs": int(TEMPORAL_STAGE2_EPOCHS),
            "temporal_stage1_head_lr": float(TEMPORAL_STAGE1_HEAD_LR),
            "temporal_stage2_whisper_lr": float(TEMPORAL_STAGE2_WHISPER_LR),
            "temporal_stage2_head_lr": float(TEMPORAL_STAGE2_HEAD_LR),
            "single_layers": [int(x) for x in resolved_single_layers],
            "unfreeze_encoder_layers_list": [int(x) for x in unfreeze_list],
            "results": serializable_results,
        },
        summary_path,
    )

    print("\n" + "=" * 120)
    print("[BATCH] 全部实验完成")
    print("[BATCH] summary_path =", summary_path)
    print("=" * 120)

    return {
        "results": results,
        "summary_path": summary_path,
    }

if __name__ == "__main__":
    batch_result = run_all_experiments_mmse()
    print(json.dumps({"summary_path": batch_result["summary_path"]}, ensure_ascii=False, indent=2))

pass
