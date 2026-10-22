# -*- coding: utf-8 -*-
"""CPAT Stage3：masked cluster prediction 与 MMSE 辅助回归。
精简版保留原参数、模型/损失/随机调用顺序、Lightning 训练和文件协议。
读取 Stage2 train chunks 与聚类标签，使用 Stage1 checkpoint 初始化 Whisper。
默认 segment_pred_mean 辅助回归、5 轮、last checkpoint；不使用验证集选模。

conda activate whisper_ptapt_faiss 
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune/ft_ptapt_repeat/CPAT_stage3.py
"""

import os

# 工程、中间结果与特征根目录；保留原路径。
PROJECT_ROOT = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune"
MMSE_PTAPT_ROOT = os.path.join(PROJECT_ROOT, "ft_ptapt_repeat")

PROJECT_ROOT_MAIN = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT_MAIN, "feature_tensor")

# Pitt Stage1 规范划分审计；Stage2/3 按原规则检查。
PITT_SPLIT_AUDIT_ROOT = os.path.join(FEATURE_ROOT_DIR, "Pitt", "subject_disjoint_split")
PITT_VISIT_MANIFEST_PATH = os.path.join(PITT_SPLIT_AUDIT_ROOT, "pitt_valid_visits_subject_disjoint_manifest.csv")
PITT_SUBJECT_ASSIGNMENT_PATH = os.path.join(PITT_SPLIT_AUDIT_ROOT, "pitt_subject_assignment.csv")

# Pitt：原 test 中有有效 visit 的受试者全部归入 final test，其余归入 final train。
PITT_ENABLE_SUBJECT_DISJOINT_SPLIT = True
PITT_SPLIT_POLICY = "original_test_subjects_holdout"
PITT_EXPECTED_FINAL_TRAIN_SUBJECTS = 184
PITT_EXPECTED_FINAL_TEST_SUBJECTS = 68
PITT_EXPECTED_FINAL_TRAIN_VISITS = 276
PITT_EXPECTED_FINAL_TEST_VISITS = 122

# 批量数据集；默认仅 ["ADReSS-IS2020"]，其他数据集分支仍保留。
DATASET_OPTIONS = ["ADReSS-IS2020"]

DATASET_OPTION = DATASET_OPTIONS[0]

# 各阶段层号保持联动；默认第 12 层。
LAYERS_TO_RUN = [12]

# 相对当前表示层，向前连续解冻的层数；保留遍历顺序。
UNFREEZE_ENCODER_LAYERS_LIST = [8]

UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# GPU 设置；保留各文件原值及环境变量生效顺序。
GPU_ID = 3 # 默认为4，但是segments-->mmse +mean---->audio，改为2
if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# 协议标签；当前训练不划分 fold，test 仅用于最终导出。
RUN_PROTOCOL_TAG = "official_train_test"

# 旧配置/manifest 的兼容占位；不新增 fold 目录。
PROTOCOL_FOLD_ID = 1
FOLDS_TO_RUN = [PROTOCOL_FOLD_ID]

# 跨阶段模型根标签；各脚本必须一致。
MODEL_ROOT_TAG = "ptapt_lr5e-5_epo5"

# 读取 Stage1 checkpoint 的路径后缀；与实际 Stage1 输出对应。
STAGE1_SOURCE_PATH_PARAM_TAGS =  []

# singleX_unfreeY 的附加路径标签；Stage2、Stage3、导出需一致。
PATH_PARAM_TAGS = []

STAGE1_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN

STAGE3_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN

# 相对首个可训练层的 mask 偏移；不是绝对 encoder 层号。
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]

STAGE4_OUTPUT_LAYERS_TO_RUN = LAYERS_TO_RUN

CHUNK_LEVEL = True

# 固定切段长度与重叠率；chunk key 格式保持原样。
SEGMENT_SEC = 30
OVERLAP_RATE = 0.3
CHUNK_KEY_VERSION = "wav_path_chunk_start_end_v1"

# Whisper 基模；保持原 checkpoint 与加载配置。
WHISPER_MODEL_NAME = "/home/juanyu/whisper_lru_3/whisper_small"

# 手动指定 Stage1 checkpoint；None 使用原自动查找规则。
STAGE1_CKPT_OVERRIDE = None

RESUME_STAGE3_CKPT = None

# Stage3 聚类头；与 Stage2 KS 及导出元信息一致。
EXPECTED_NUM_CLUSTERS = [32, 64]

USE_ALL_CLUS_JSON = True

TARGET_SEC = 30.0
BATCH_SIZE = 4
NUM_WORKERS = 4

# Stage3 有效 token 上的 span masking。
MASK_PROB = 0.06
MASK_SPAN = 10
LAMBDA_CLUS = 1.0

# Stage3 MMSE 辅助回归权重与 Huber 参数。
LAMBDA_MMSE = 0.05
HUBER_DELTA = 1.0

# 辅助回归粒度；默认同一音频的 chunk 预测先求均值，再计算音频损失。
STAGE3_AUX_REGRESSION_LEVEL = "segment_pred_mean"

# 音频 batch size；collate 按原顺序展开其内部 chunks。
STAGE3_AUDIO_BATCH_SIZE = 2

HEAD_DROPOUT = 0.2
L2_NORMALIZE_REPS = False

LR_ENC = 5e-5
LR_HEAD = 7e-5
WEIGHT_DECAY = 0.02

# 固定 epoch、无验证集选模，保存 last checkpoint。
STAGE3_USE_FOLD_VAL = False
STAGE3_MAX_EPOCHS = 5
STAGE3_CHECKPOINT_SELECTION = "last"

TRAINING_STEP = 3000

PRECISION = 16
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_BATCHES = 1

EARLY_PATIENCE = 5
EARLY_MIN_DELTA = 1e-4
SAVE_TOP_K = 0
CHECK_VAL_EVERY_N_EPOCH = 1

SEED = 42
SAVE_MANIFESTS = True
ENABLE_CSV_LOGGER = True

# 当前 no-val-last 流程要求 False；保留原错误检查。
ENABLE_STAGE_QUALITY_EVAL = False
QUALITY_REPORT_FILENAME = "quality_report.json"
QUALITY_SUMMARY_FILENAME = "quality_summary.csv"
QUALITY_DECISION_FILENAME = "quality_decision.txt"

import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from transformers import WhisperConfig, WhisperModel, WhisperFeatureExtractor

os.environ.pop("PL_TRAINER_AMP_BACKEND", None)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
TOK_T_FULL = (MAX_MEL_FRAMES + 1) // 2    # Whisper small 默认 1500
IGNORE_INDEX = -100
TARGET_30S_SAMPLES = int(float(TARGET_SEC) * SAMPLE_RATE)
SEGMENT_LENGTH = int(float(SEGMENT_SEC) * SAMPLE_RATE)

# 随机源与通用工具
def seed_everything_for_repro(seed: int) -> None:
    """统一设置随机种子，尽量保证实验可复现。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def ensure_dir(path: str) -> None:
    """若目录不存在，则自动创建。"""
    os.makedirs(path, exist_ok=True)

def natural_key(x: str):
    """自然排序 key。"""
    s = Path(str(x)).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def canonical_wav_path(path: str) -> str:
    """统一 wav 路径表示，避免相对/绝对路径不一致导致 Stage2 labels 匹配失败。"""
    return str(Path(str(path)).expanduser().resolve())

def save_json(obj: Dict[str, Any], path: str) -> None:
    """保存 json，并自动创建父目录。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(text: str, path: str) -> None:
    """保存纯文本，并自动创建父目录。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def compute_valid_token_length(wav_len_samples: int) -> int:
    """将“真实有效 waveform 样本点数”映射为 Whisper encoder 的有效 token 长度。"""
    if wav_len_samples <= 0:
        return 1
    n_mel_valid = int(math.ceil(wav_len_samples / float(HOP_LENGTH)))
    n_mel_valid = min(MAX_MEL_FRAMES, max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)

def _clean_path_tag(x) -> str:
    """清理路径参数标签，避免空格、斜杠、等号等字符进入目录名。"""
    import re

    if x is None:
        return ""

    s = str(x).strip()
    if s in ["", "None", "none", "null", "Null"]:
        return ""

    s = s.replace("=", "").replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_.-]+", "", s)
    s = s.strip("_-.")
    return s

def build_path_param_suffix(path_param_tags=None) -> str:
    """根据 PATH_PARAM_TAGS 构造路径后缀。"""
    if path_param_tags is None:
        path_param_tags = globals().get("PATH_PARAM_TAGS", [])

    if isinstance(path_param_tags, (str, int, float)):
        tags = [path_param_tags]
    else:
        tags = list(path_param_tags)

    tags = [_clean_path_tag(x) for x in tags]
    tags = [x for x in tags if x != ""]
    return "_".join(tags)

def build_layer_name(single_layer: int, unfreeze_layer_count: int, path_param_tags=None) -> str:
    """构造 P-TAPT 统一使用的层目录名。"""
    base = f"single{int(single_layer)}_unfree{int(unfreeze_layer_count)}"
    suffix = build_path_param_suffix(path_param_tags)
    if suffix == "":
        return base
    return f"{base}_{suffix}"

def get_whisper_encoder_layer_count(whisper_model_name_or_path: str) -> int:
    """从 Whisper config 读取 encoder 总层数。"""
    cfg = WhisperConfig.from_pretrained(whisper_model_name_or_path)
    return int(getattr(cfg, "encoder_layers"))

def resolve_unfreeze_layers_for_single_layer(single_layer: int, total_layers: int, unfreeze_layer_count: int) -> List[int]:
    """根据 single_layer 和 UNFREEZE_ENCODER_LAYERS 解析实际解冻层。"""
    single_layer = int(single_layer)
    total_layers = int(total_layers)
    unfreeze_layer_count = int(unfreeze_layer_count)

    if not (1 <= single_layer <= total_layers):
        raise ValueError(f"single_layer={single_layer} 超出范围 [1, {total_layers}]")
    if unfreeze_layer_count <= 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS 必须 >= 1")

    start_layer = max(1, single_layer - unfreeze_layer_count + 1)
    end_layer = min(single_layer, total_layers)
    return list(range(start_layer, end_layer + 1))

def resolve_mask_inject_after_layer(resolved_unfreeze_layers: Sequence[int], mask_offset: int) -> int:
    """根据“第一个可训练层”自动解析 mask 注入位置。"""
    if len(resolved_unfreeze_layers) == 0:
        raise ValueError("resolved_unfreeze_layers 不能为空。")
    mask_offset = int(mask_offset)
    if mask_offset <= 0:
        raise ValueError("STAGE3_MASK_INJECT_AFTER_LAYERS 中的值必须 >= 1。")

    first_trainable = min(int(x) for x in resolved_unfreeze_layers)
    return max(0, first_trainable - int(mask_offset))

def build_stage3_subdir(mask_offset: int) -> str:
    """Stage3 保存子目录。"""
    base = f"stage3_mptapt_maskoff{int(mask_offset)}"

    aux_level = str(globals().get("STAGE3_AUX_REGRESSION_LEVEL", "chunk")).strip().lower()

    if aux_level in {
        "segment_pred_mean",
        "segment-pred-mean",
        "prediction_mean",
        "prediction-mean",
        "segment_prediction_mean",
    }:
        return f"{base}"

    return base

def canonical_label_key(key: str) -> str:
    """规范化 Stage2/Stage3 label key。"""
    key = str(key)
    if "::chunk" in key:
        wav_part, suffix = key.split("::chunk", 1)
        return canonical_wav_path(wav_part) + "::chunk" + suffix
    return canonical_wav_path(key)

def load_audio_16k(wav_path: str) -> np.ndarray:
    """与 audio_cut_xr.py 对齐：librosa.load(sr=16000, mono=True, dtype=float32)。"""
    wav_path = canonical_wav_path(wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"[Stage3-MMSE] 找不到音频文件：{wav_path}")
    wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    return np.asarray(wav, dtype=np.float32).reshape(-1)

def load_chunk_with_padding_from_wav(y: np.ndarray, start: int, end: int, chunk_id: int) -> Tuple[np.ndarray, int]:
    """从 16k waveform 中取出一个 chunk，并复用 audio_cut_xr.py 的尾段补齐逻辑。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    start = int(start)
    end = int(end)
    chunk_id = int(chunk_id)

    segment = y[start:end]

    if len(segment) < SEGMENT_LENGTH and chunk_id > 0:
        padding_needed = SEGMENT_LENGTH - len(segment)
        step_size = int(SEGMENT_LENGTH * (1.0 - float(OVERLAP_RATE)))
        prev_end = start + SEGMENT_LENGTH - step_size
        padding_start = max(0, prev_end - padding_needed)
        padding_values = y[padding_start:prev_end]
        segment = np.concatenate((segment, padding_values), axis=0)

    if len(segment) > SEGMENT_LENGTH:
        segment = segment[:SEGMENT_LENGTH]

    valid_len = int(min(len(segment), SEGMENT_LENGTH))
    if len(segment) < SEGMENT_LENGTH:
        pad = np.zeros((SEGMENT_LENGTH - len(segment),), dtype=np.float32)
        segment = np.concatenate([segment.astype(np.float32, copy=False), pad], axis=0)

    return segment.astype(np.float32, copy=False), valid_len

def pad_or_truncate_labels(label_tensor: torch.Tensor, target_t: int = TOK_T_FULL) -> torch.Tensor:
    """把 Stage2 label tensor 统一到 [H, target_t]。"""
    if label_tensor.dim() == 1:
        label_tensor = label_tensor.unsqueeze(0)
    if label_tensor.dim() != 2:
        raise ValueError(f"label_tensor 必须是 [H,T] 或 [T]，当前形状={tuple(label_tensor.shape)}")

    H, T = label_tensor.shape
    if T == target_t:
        return label_tensor.long()
    if T > target_t:
        return label_tensor[:, :target_t].long()

    pad = torch.full((H, target_t - T), int(IGNORE_INDEX), dtype=torch.long)
    return torch.cat([label_tensor.long(), pad], dim=1)

def make_time_mask_tok_valid(valid_mask_tok: torch.Tensor, mask_prob: float, mask_span: int) -> torch.Tensor:
    """根据 token 级有效 mask 随机生成 span mask。"""
    assert valid_mask_tok.dim() == 2
    B, T = valid_mask_tok.shape
    device = valid_mask_tok.device
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)

    mask_prob = float(mask_prob)
    mask_span = int(mask_span)

    for b in range(B):
        valid_len = int(valid_mask_tok[b].long().sum().item())
        valid_len = max(1, min(valid_len, T))
        if valid_len <= 1:
            continue

        num_mask = int(math.ceil(valid_len * mask_prob))
        if num_mask <= 0:
            continue

        masked = 0
        tries = 0
        span_low = max(1, mask_span // 2)
        span_high = max(span_low + 1, int(mask_span * 1.5))

        while masked < num_mask and tries < 10 * max(1, num_mask):
            span = int(torch.randint(span_low, span_high, (1,), device=device).item())
            span = min(span, valid_len)
            start_max = max(1, valid_len - span + 1)
            start = int(torch.randint(0, start_max, (1,), device=device).item())
            end = min(valid_len, start + span)

            if mask[b, start:end].any():
                tries += 1
                continue

            mask[b, start:end] = True
            masked += (end - start)

    mask = mask & valid_mask_tok
    return mask

# MMSE 元数据路径与列配置。
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
}

def _extract_pitt_subject_visit(raw_name: str) -> Tuple[Optional[str], Optional[int]]:
    """从 Pitt 文件名解析 (subject_id, visit_index)。"""
    if raw_name is None:
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    return str(int(m.group(1))), int(m.group(2))

# Stage3 路径与输入文件配置
def build_stage3_run_config(dataset_option: str, repr_layer: int, mask_offset: int, fold_id: int = PROTOCOL_FOLD_ID) -> Dict[str, Any]:
    """构造 Stage3 运行配置。"""
    repr_layer = int(repr_layer)
    mask_offset = int(mask_offset)
    fold_id = int(PROTOCOL_FOLD_ID)

    total_layers = get_whisper_encoder_layer_count(WHISPER_MODEL_NAME)

    resolved_unfreeze_layers = resolve_unfreeze_layers_for_single_layer(
        single_layer=repr_layer,
        total_layers=total_layers,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
    )

    mask_inject_after_layer = resolve_mask_inject_after_layer(
        resolved_unfreeze_layers=resolved_unfreeze_layers,
        mask_offset=mask_offset,
    )

    layer_name = build_layer_name(
        single_layer=repr_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage1_layer_name = build_layer_name(
        single_layer=repr_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=STAGE1_SOURCE_PATH_PARAM_TAGS,
    )

    stage3_subdir = build_stage3_subdir(mask_offset)

    run_root = os.path.join(
        MMSE_PTAPT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        layer_name,
    )

    stage1_run_root = os.path.join(
        MMSE_PTAPT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        stage1_layer_name,
    )

    stage1_dir = os.path.join(stage1_run_root, "stage1_mmse")

    stage2_save_dir = os.path.join(run_root, "stage2_clusters")
    stage2_train_labelpath = os.path.join(stage2_save_dir, "train", "all-clus.json")
    stage2_train_manifest = os.path.join(stage2_save_dir, "train", "chunks_manifest.json")

    stage3_save_dir = os.path.join(run_root, stage3_subdir)
    meta_save_dir = os.path.join(run_root, "meta")

    return {
        "DATASET_OPTION": str(dataset_option),
        "REPR_LAYER": int(repr_layer),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),
        "RESOLVED_UNFREEZE_ENCODER_LAYERS": list(resolved_unfreeze_layers),
        "STAGE3_TRAIN_LAYER_START": int(min(resolved_unfreeze_layers)),
        "STAGE3_TRAIN_LAYER_END": int(max(resolved_unfreeze_layers)),
        "TOTAL_ENCODER_LAYERS": int(total_layers),
        "MASK_OFFSET": int(mask_offset),
        "MASK_INJECT_AFTER_LAYER": int(mask_inject_after_layer),

        "FOLD_ID": int(fold_id),
        "FOLD_TAG": "not_used",
        "FOLD_JSON_PATH": "",
        "RUN_PROTOCOL_TAG": str(RUN_PROTOCOL_TAG),

        "LAYER_TAG": layer_name,
        "LAYER_NAME": layer_name,
        "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(PATH_PARAM_TAGS),

        "STAGE1_LAYER_NAME": stage1_layer_name,
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(STAGE1_SOURCE_PATH_PARAM_TAGS),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(STAGE1_SOURCE_PATH_PARAM_TAGS),

        "RUN_ROOT": run_root,
        "STAGE1_RUN_ROOT": stage1_run_root,
        "STAGE1_SAVE_DIR": stage1_dir,

        "STAGE2_SAVE_DIR": stage2_save_dir,
        "STAGE2_TRAIN_LABELPATH": stage2_train_labelpath,
        "STAGE2_VAL_LABELPATH": "",
        "STAGE2_TRAIN_MANIFEST": stage2_train_manifest,
        "STAGE2_VAL_MANIFEST": "",

        "STAGE3_SAVE_DIR": stage3_save_dir,
        "STAGE3_SUBDIR": stage3_subdir,
        "META_SAVE_DIR": meta_save_dir,
    }

def _strip_state_dict_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """去掉 state_dict key 的公共前缀。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def load_stage1_ckpt_into_whisper_model(whisper_model: WhisperModel, ckpt_path: str) -> None:
    """把 Stage1 checkpoint 中的 Whisper 权重加载到当前 WhisperModel。"""
    if ckpt_path in [None, "None", ""]:
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    candidates = [
        "whisper.model.",
        "model.",
        "",
    ]

    loaded_any = False
    for pref in candidates:
        sd = _strip_state_dict_prefix(state, pref) if pref != "" else state
        missing, unexpected = whisper_model.load_state_dict(sd, strict=False)
        encoder_hit = any(k.startswith("encoder.") for k in sd.keys())
        if encoder_hit:
            print(f"[stage1_init] tried prefix='{pref}' -> missing={len(missing)} unexpected={len(unexpected)}")
            loaded_any = True
            break

    if not loaded_any:
        missing, unexpected = whisper_model.load_state_dict(state, strict=False)
        print(f"[stage1_init][warn] 前缀无法明确匹配，已 strict=False 加载，missing={len(missing)} unexpected={len(unexpected)}")

def find_stage1_checkpoint(stage1_save_dir: str, override_ckpt: Optional[str] = None) -> str:
    """no-val-last:"""
    if override_ckpt not in [None, "", "None"]:
        ckpt_path = str(override_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"STAGE1_CKPT_OVERRIDE 不存在：{ckpt_path}")
        return ckpt_path

    stage1_dir = Path(stage1_save_dir)
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage1 目录不存在：{stage1_save_dir}")

    last_ckpt = stage1_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt.resolve())

    all_ckpts = sorted(stage1_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(all_ckpts) > 0:
        return str(all_ckpts[0].resolve())

    raise FileNotFoundError(f"在 {stage1_save_dir} 下未找到任何 Stage1 checkpoint。")

# 聚类标签加载及完整性检查
def load_all_clus_json(cluster_json_path: str, expected_k_list: Optional[Sequence[int]] = None) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """读取 Stage2 的 all-clus.json。"""
    if not os.path.isfile(cluster_json_path):
        raise FileNotFoundError(f"cluster json 不存在：{cluster_json_path}")

    with open(cluster_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"cluster json 应该是 dict，当前={type(raw)}")

    keys = [k for k in raw.keys() if not str(k).startswith("__")]
    if len(keys) == 0:
        raise ValueError(f"{cluster_json_path} 没有有效样本 key。")

    cluster_label_dict: Dict[str, torch.Tensor] = {}
    max_ids: Optional[List[int]] = None

    for raw_key, value in raw.items():
        if str(raw_key).startswith("__"):
            continue

        wav_key = canonical_label_key(raw_key)

        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
            seq_lists = value
        elif isinstance(value, list):
            seq_lists = [value]
        else:
            raise ValueError(f"{raw_key} 的 label 格式非法，应该是 list 或 list[list]。")

        if max_ids is None:
            max_ids = [0 for _ in seq_lists]
        if len(seq_lists) != len(max_ids):
            raise ValueError(f"{raw_key} 的 head 数不一致：{len(seq_lists)} vs {len(max_ids)}")

        tensors = []
        for h, seq in enumerate(seq_lists):
            arr = torch.tensor(seq, dtype=torch.long)
            arr = pad_or_truncate_labels(arr, target_t=TOK_T_FULL).squeeze(0)

            valid = arr[arr != IGNORE_INDEX]
            if valid.numel() > 0:
                max_ids[h] = max(max_ids[h], int(valid.max().item()))

            tensors.append(arr)

        stacked = torch.stack(tensors, dim=0)  # [H,T]
        cluster_label_dict[wav_key] = stacked

    inferred_k_list = [int(x) + 1 for x in (max_ids or [])]

    if expected_k_list is not None and len(expected_k_list) > 0:
        if len(expected_k_list) != len(inferred_k_list):
            raise ValueError(
                f"EXPECTED_NUM_CLUSTERS 与 all-clus head 数不匹配："
                f"expected={list(expected_k_list)}, inferred={inferred_k_list}"
            )
        k_list = [int(x) for x in expected_k_list]
    else:
        k_list = inferred_k_list

    for name, lab in cluster_label_dict.items():
        for h in range(lab.size(0)):
            v = lab[h][lab[h] != IGNORE_INDEX]
            if v.numel() > 0 and int(v.max().item()) >= int(k_list[h]):
                raise ValueError(
                    f"[Stage3] {name} head{h} label max={int(v.max().item())} >= K={int(k_list[h])}"
                )

    return cluster_label_dict, k_list

def check_stage3_cluster_config(
    k_list,
    expected_num_clusters,
    train_clus_path: str,
    val_clus_path: str,
) -> None:
    """检查 Stage3 读到的 K/head 是否与 EXPECTED_NUM_CLUSTERS 一致。"""
    k_list = [int(x) for x in k_list]
    expected_num_clusters = [int(x) for x in expected_num_clusters]

    if len(k_list) != len(expected_num_clusters):
        raise RuntimeError(
            f"[Stage3] head mismatch: loaded k_list={k_list}, "
            f"EXPECTED_NUM_CLUSTERS={expected_num_clusters}, "
            f"train={train_clus_path}, val={val_clus_path}"
        )

    for loaded_k, expected_k in zip(k_list, expected_num_clusters):
        if int(loaded_k) != int(expected_k):
            raise RuntimeError(
                f"[Stage3] K mismatch: loaded k_list={k_list}, "
                f"EXPECTED_NUM_CLUSTERS={expected_num_clusters}, "
                f"train={train_clus_path}, val={val_clus_path}"
            )

def write_stage3_run_meta(
    cfg: Dict[str, Any],
    stage1_ckpt_path: str,
    train_clus_path: str,
    val_clus_path: str,
) -> None:
    """保存 Stage3 运行配置。"""
    stage3_save_dir = cfg["STAGE3_SAVE_DIR"]

    meta = {
        "DATASET_OPTION": cfg.get("DATASET_OPTION", DATASET_OPTION),
        "MODEL_ROOT_TAG": MODEL_ROOT_TAG,
        "REPR_LAYER": int(cfg.get("REPR_LAYER", -1)),
        "FOLD_ID": int(cfg.get("FOLD_ID", -1)),
        "LAYER_NAME": cfg.get("LAYER_NAME", ""),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),

        "PATH_PARAM_TAGS": list(globals().get("PATH_PARAM_TAGS", [])),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(globals().get("PATH_PARAM_TAGS", [])),
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(globals().get("STAGE1_SOURCE_PATH_PARAM_TAGS", [])),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(
            globals().get("STAGE1_SOURCE_PATH_PARAM_TAGS", [])
        ),

        "EXPECTED_NUM_CLUSTERS": [int(k) for k in EXPECTED_NUM_CLUSTERS],
        "LAMBDA_CLUS": float(LAMBDA_CLUS),
        "LAMBDA_MMSE": float(LAMBDA_MMSE),
        "MASK_PROB": float(MASK_PROB),
        "MASK_SPAN": int(MASK_SPAN),
        "LR_ENC": float(LR_ENC),
        "LR_HEAD": float(LR_HEAD),
        "TRAINING_STEP": int(TRAINING_STEP),
        "MASK_INJECT_AFTER_LAYER": int(cfg.get("MASK_INJECT_AFTER_LAYER", -1)),

        "PTAPT_METHOD": "clean_ptapt" if float(LAMBDA_MMSE) == 0.0 else "ptapt_mmse_aux",
        "PTAPT_METHOD": (
            "clean_ptapt"
            if float(LAMBDA_MMSE) == 0.0
            else "ptapt_mmse_segment_pred_mean_aux"
        ),

        "stage1_ckpt_path": str(stage1_ckpt_path),
        "train_clus_path": str(train_clus_path),
        "val_clus_path": str(val_clus_path),
    }

    save_json(meta, os.path.join(stage3_save_dir, "stage3_run_meta.json"))

def filter_entries_by_cluster_labels(
    entries: List[Dict[str, Any]],
    cluster_label_dict: Dict[str, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """只保留在 Stage2 cluster labels 中存在的 chunk entries。"""
    valid = []
    missing = []
    for row in entries:
        key = canonical_label_key(row.get("chunk_key", row.get("wav_path", "")))
        if key in cluster_label_dict:
            row = copy.deepcopy(row)
            row["chunk_key"] = key
            valid.append(row)
        else:
            row = copy.deepcopy(row)
            row["missing_reason"] = "missing_stage2_cluster_label"
            missing.append(row)
    return valid, missing

def _normalize_pitt_subject_id(x) -> str:
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        m = re.search(r"\d+", s)
        if m is None:
            raise ValueError(f"无法规范化 Pitt subject_id={x}")
        return str(int(m.group(0)))

def _load_pitt_canonical_split_audit_for_stage3() -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not os.path.isfile(PITT_VISIT_MANIFEST_PATH) or not os.path.isfile(PITT_SUBJECT_ASSIGNMENT_PATH):
        raise FileNotFoundError(
            "[Pitt Stage3] 缺少 subject-disjoint 审计文件。请先运行 SCPA_stage1_pitt_new.py。\n"
            f"visit_manifest={PITT_VISIT_MANIFEST_PATH}\n"
            f"subject_assignment={PITT_SUBJECT_ASSIGNMENT_PATH}"
        )

    visit_df = pd.read_csv(PITT_VISIT_MANIFEST_PATH)
    subject_df = pd.read_csv(PITT_SUBJECT_ASSIGNMENT_PATH)
    required_visit = {"subject_id", "visit_index", "wav_path", "mmse", "final_split"}
    required_subject = {"subject_id", "final_split"}
    if not required_visit.issubset(set(visit_df.columns)):
        raise ValueError(f"[Pitt Stage3] visit manifest 缺列：{sorted(required_visit - set(visit_df.columns))}")
    if not required_subject.issubset(set(subject_df.columns)):
        raise ValueError(f"[Pitt Stage3] subject assignment 缺列：{sorted(required_subject - set(subject_df.columns))}")

    visit_df = visit_df.copy()
    subject_df = subject_df.copy()
    visit_df["subject_id"] = visit_df["subject_id"].map(_normalize_pitt_subject_id)
    subject_df["subject_id"] = subject_df["subject_id"].map(_normalize_pitt_subject_id)
    visit_df["visit_index"] = pd.to_numeric(visit_df["visit_index"], errors="raise").astype(int)
    visit_df["mmse"] = pd.to_numeric(visit_df["mmse"], errors="raise").astype(float)
    visit_df["final_split"] = visit_df["final_split"].astype(str).str.strip().str.lower()
    subject_df["final_split"] = subject_df["final_split"].astype(str).str.strip().str.lower()

    train_subjects = set(subject_df.loc[subject_df["final_split"] == "train", "subject_id"].tolist())
    test_subjects = set(subject_df.loc[subject_df["final_split"] == "test", "subject_id"].tolist())
    if train_subjects & test_subjects:
        raise RuntimeError(f"[Pitt Stage3] canonical audit 本身存在 overlap: {sorted(train_subjects & test_subjects)}")

    checks = [
        ("train_subjects", len(train_subjects), PITT_EXPECTED_FINAL_TRAIN_SUBJECTS),
        ("test_subjects", len(test_subjects), PITT_EXPECTED_FINAL_TEST_SUBJECTS),
        ("train_visits", int((visit_df["final_split"] == "train").sum()), PITT_EXPECTED_FINAL_TRAIN_VISITS),
        ("test_visits", int((visit_df["final_split"] == "test").sum()), PITT_EXPECTED_FINAL_TEST_VISITS),
    ]
    for name, actual, expected in checks:
        if int(actual) != int(expected):
            raise RuntimeError(f"[Pitt Stage3] canonical audit {name}={actual}, expected={expected}")
    return visit_df, subject_df

def validate_pitt_stage2_chunk_manifest_subject_disjoint(entries: List[Dict[str, Any]]) -> None:
    """Stage3 入口安全检查：Stage2 chunks 必须且只能来自 canonical final train 的 276 visits。"""
    if str(DATASET_OPTION) != "Pitt" or not bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return

    visit_df, subject_df = _load_pitt_canonical_split_audit_for_stage3()
    canonical_train = visit_df[visit_df["final_split"] == "train"].copy()
    canonical_visits = {
        (_normalize_pitt_subject_id(r.subject_id), int(r.visit_index))
        for r in canonical_train.itertuples()
    }
    canonical_paths = {canonical_wav_path(p) for p in canonical_train["wav_path"].astype(str).tolist()}
    canonical_mmse = {
        (_normalize_pitt_subject_id(r.subject_id), int(r.visit_index)): float(r.mmse)
        for r in canonical_train.itertuples()
    }
    canonical_train_subjects = set(subject_df.loc[subject_df["final_split"] == "train", "subject_id"].tolist())
    canonical_test_subjects = set(subject_df.loc[subject_df["final_split"] == "test", "subject_id"].tolist())

    manifest_visits = set()
    manifest_paths = set()
    manifest_subjects = set()
    bad_mmse = []
    for row in entries:
        fp = canonical_wav_path(row.get("wav_path", ""))
        sid = row.get("subject_id", None)
        vid = row.get("visit_index", None)
        if sid is None or vid is None:
            sid, vid = _extract_pitt_subject_visit(Path(fp).stem)
        if sid is None or vid is None:
            raise RuntimeError(f"[Pitt Stage3] chunk manifest 无法解析 subject/visit: {fp}")
        sid = _normalize_pitt_subject_id(sid)
        key = (sid, int(vid))
        manifest_visits.add(key)
        manifest_paths.add(fp)
        manifest_subjects.add(sid)

        if key in canonical_mmse:
            try:
                y = float(row.get("target_mmse"))
                if abs(y - canonical_mmse[key]) > 1e-6:
                    bad_mmse.append((key, y, canonical_mmse[key]))
            except Exception:
                bad_mmse.append((key, row.get("target_mmse"), canonical_mmse[key]))

    if manifest_subjects & canonical_test_subjects:
        raise RuntimeError(
            f"[Pitt Stage3] Stage2 chunks 含 final-test subjects: {sorted(manifest_subjects & canonical_test_subjects, key=int)[:20]}"
        )
    if manifest_subjects != canonical_train_subjects:
        missing = sorted(canonical_train_subjects - manifest_subjects, key=int)
        extra = sorted(manifest_subjects - canonical_train_subjects, key=int)
        raise RuntimeError(f"[Pitt Stage3] Stage2 unique subjects 不一致：missing={missing[:20]}, extra={extra[:20]}")
    if manifest_visits != canonical_visits:
        missing = sorted(canonical_visits - manifest_visits, key=lambda x: (int(x[0]), x[1]))
        extra = sorted(manifest_visits - canonical_visits, key=lambda x: (int(x[0]), x[1]))
        raise RuntimeError(f"[Pitt Stage3] Stage2 unique visits 不一致：missing={missing[:20]}, extra={extra[:20]}")
    if manifest_paths != canonical_paths:
        missing = sorted(canonical_paths - manifest_paths)
        extra = sorted(manifest_paths - canonical_paths)
        raise RuntimeError(f"[Pitt Stage3] Stage2 unique wav paths 不一致：missing={missing[:10]}, extra={extra[:10]}")
    if bad_mmse:
        raise RuntimeError(f"[Pitt Stage3] Stage2 target_mmse 与 canonical visit MMSE 不一致：{bad_mmse[:10]}")

    print(
        "[Pitt Stage3] subject-disjoint Stage2 chunk manifest check PASSED | "
        f"unique_subjects={len(manifest_subjects)}, unique_visits={len(manifest_visits)}, "
        f"chunks={len(entries)}, final-test-overlap=0"
    )

def _is_process2_ctd_chunk_entry(row: Dict[str, Any], dataset_key: str = None) -> bool:
    """PROCESS-2 专用 chunk manifest safety filter。"""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    wav_path = row.get("wav_path", row.get("wav_name", ""))
    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None

def load_stage2_chunks_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """读取 Stage2 v3 导出的 chunks_manifest.json。"""
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Stage2 chunks_manifest 不存在：{manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and isinstance(obj.get("chunks", None), list):
        chunks = obj["chunks"]
    elif isinstance(obj, list):
        chunks = obj
    else:
        raise ValueError(f"无法解析 chunks_manifest：{manifest_path}")

    out = []
    for row in chunks:
        row = copy.deepcopy(row)
        row["wav_path"] = canonical_wav_path(row["wav_path"])
        row["wav_name"] = str(row.get("wav_name", Path(row["wav_path"]).name))
        row["chunk_key"] = canonical_label_key(row["chunk_key"])
        row["chunk_id"] = int(row["chunk_id"])
        row["chunk_start"] = int(row["chunk_start"])
        row["chunk_end"] = int(row["chunk_end"])
        row["chunk_valid_len"] = int(row.get("chunk_valid_len", SEGMENT_LENGTH))

        if not _is_process2_ctd_chunk_entry(row, dataset_key=DATASET_OPTION):
            continue

        out.append(row)

    if len(out) == 0:
        raise ValueError(f"chunks_manifest 经过 CTD 过滤后为空：{manifest_path}")

    if str(DATASET_OPTION).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] Stage3 loaded chunks_manifest: "
            f"raw_chunks={len(chunks)}, kept_ctd_chunks={len(out)}, "
            f"dropped_non_ctd={len(chunks) - len(out)}"
        )

    if str(DATASET_OPTION) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        validate_pitt_stage2_chunk_manifest_subject_disjoint(out)
    return out

def sanity_check_stage3_entries_have_target_mmse(entries: List[Dict[str, Any]]) -> None:
    """Stage3 不重复做 MMSE 过滤，只做 sanity check。"""
    bad: List[Dict[str, Any]] = []

    for row in entries:
        val = row.get("target_mmse", None)
        try:
            if val in [None, ""]:
                raise ValueError("empty target_mmse")
            _ = float(val)
        except Exception:
            bad.append({
                "wav_path": row.get("wav_path", ""),
                "wav_name": row.get("wav_name", ""),
                "chunk_key": row.get("chunk_key", ""),
                "reason": "missing_or_invalid_target_mmse",
            })

    if len(bad) > 0:
        examples = bad[:5]
        raise ValueError(
            f"Stage3 sanity check failed: {len(bad)} chunks 缺失或非法 target_mmse。"
            f" 示例={examples}"
        )

def get_stage3_audio_id_from_entry(row: Dict[str, Any]) -> str:
    """Stage3 segment_pred_mean 分组用 audio_id。"""
    if isinstance(row, dict):
        for key in ["audio_id", "utt_id", "recording_id"]:
            val = row.get(key, None)
            if val not in [None, ""]:
                return str(val)

        wav_path = row.get("wav_path", row.get("wav_name", ""))
        if wav_path not in [None, ""]:
            return canonical_wav_path(wav_path)

    raise ValueError(f"无法从 row 中解析 audio_id: {row}")

def compute_stage3_audio_level_target_stats(entries: List[Dict[str, Any]]) -> Tuple[float, float, int]:
    """用 unique audio 统计 Stage3 target normalization。"""
    audio_to_mmse: Dict[str, float] = {}

    for row in entries:
        audio_id = get_stage3_audio_id_from_entry(row)
        mmse_raw = float(row["target_mmse"])

        if audio_id in audio_to_mmse:
            prev = float(audio_to_mmse[audio_id])
            if abs(prev - mmse_raw) > 1e-5:
                raise ValueError(
                    f"同一 audio 出现不同 target_mmse: "
                    f"audio_id={audio_id}, prev={prev}, current={mmse_raw}"
                )
            continue

        audio_to_mmse[audio_id] = mmse_raw

    if len(audio_to_mmse) == 0:
        raise ValueError("无法计算 audio-level target stats：没有有效 audio。")

    values = np.asarray(list(audio_to_mmse.values()), dtype=np.float32)
    target_mean = float(values.mean())
    target_std = float(values.std(ddof=0))

    if target_std < 1e-6:
        target_std = 1.0

    return target_mean, target_std, int(len(audio_to_mmse))

# Stage3 音频/chunk 数据集
class Stage3MPTAPTDataset(data.Dataset):
    """Stage3 M-P-TAPT segment_pred_mean 数据集。"""

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        cluster_label_dict: Dict[str, torch.Tensor],
        target_mean: float,
        target_std: float,
        target_sec: float = 30.0,
    ):
        self.cluster_label_dict = cluster_label_dict
        self.target_mean = float(target_mean)
        self.target_std = float(max(1e-6, target_std))
        self.target_seqlen = int(float(target_sec) * SAMPLE_RATE)

        entries = [copy.deepcopy(x) for x in entries]
        if len(entries) == 0:
            raise ValueError("Stage3MPTAPTDataset 收到空 entries。")

        missing = []
        for row in entries:
            key = canonical_label_key(row["chunk_key"])
            if key not in self.cluster_label_dict:
                missing.append(key)

        if len(missing) > 0:
            raise ValueError(
                f"Stage3MPTAPTDataset 有 {len(missing)} 个 chunk 缺少 Stage2 label，例如：{missing[:3]}"
            )

        audio_groups: Dict[str, Dict[str, Any]] = {}

        for row in entries:
            row = copy.deepcopy(row)
            row["wav_path"] = canonical_wav_path(row["wav_path"])
            row["chunk_key"] = canonical_label_key(row["chunk_key"])
            row["chunk_id"] = int(row["chunk_id"])
            row["chunk_start"] = int(row["chunk_start"])
            row["chunk_end"] = int(row["chunk_end"])
            row["chunk_valid_len"] = int(row.get("chunk_valid_len", SEGMENT_LENGTH))

            audio_id = get_stage3_audio_id_from_entry(row)
            mmse_raw = float(row["target_mmse"])

            if audio_id not in audio_groups:
                audio_groups[audio_id] = {
                    "audio_id": audio_id,
                    "wav_path": row["wav_path"],
                    "wav_name": str(row.get("wav_name", Path(row["wav_path"]).name)),
                    "target_mmse": mmse_raw,
                    "chunks": [],
                }
            else:
                prev_mmse = float(audio_groups[audio_id]["target_mmse"])
                if abs(prev_mmse - mmse_raw) > 1e-5:
                    raise ValueError(
                        f"同一 audio 出现不同 target_mmse: "
                        f"audio_id={audio_id}, prev={prev_mmse}, current={mmse_raw}"
                    )

            audio_groups[audio_id]["chunks"].append(row)

        self.records: List[Dict[str, Any]] = []
        for audio_id, rec in audio_groups.items():
            rec["chunks"] = sorted(
                rec["chunks"],
                key=lambda x: (int(x.get("chunk_id", 0)), int(x.get("chunk_start", 0))),
            )
            self.records.append(rec)

        self.records = sorted(self.records, key=lambda x: natural_key(x["wav_path"]))

        if len(self.records) == 0:
            raise ValueError("Stage3MPTAPTDataset 分组后没有有效 audio。")

        chunk_counts = [len(x["chunks"]) for x in self.records]
        print(
            "[Stage3 Dataset][segment_pred_mean] "
            f"num_audio={len(self.records)}, "
            f"num_chunks={sum(chunk_counts)}, "
            f"min_chunks_per_audio={min(chunk_counts)}, "
            f"max_chunks_per_audio={max(chunk_counts)}, "
            f"mean_chunks_per_audio={float(np.mean(chunk_counts)):.3f}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]

        wav_path = canonical_wav_path(rec["wav_path"])
        wav_name = str(rec.get("wav_name", Path(wav_path).name))
        audio_id = str(rec["audio_id"])
        mmse_raw = float(rec["target_mmse"])
        mmse_z = (mmse_raw - self.target_mean) / self.target_std

        y = load_audio_16k(wav_path)

        wav_chunks = []
        valid_lens = []
        labels_list = []
        chunk_keys = []

        for row in rec["chunks"]:
            chunk_key = canonical_label_key(row["chunk_key"])
            chunk_id = int(row["chunk_id"])
            start = int(row["chunk_start"])
            end = int(row["chunk_end"])

            wav_30, valid_len = load_chunk_with_padding_from_wav(
                y,
                start=start,
                end=end,
                chunk_id=chunk_id,
            )

            labels = self.cluster_label_dict[chunk_key].clone().long()

            wav_chunks.append(wav_30.astype(np.float32, copy=False))
            valid_lens.append(int(valid_len))
            labels_list.append(labels)
            chunk_keys.append(chunk_key)

        if len(wav_chunks) == 0:
            raise ValueError(f"audio 没有有效 chunks: {audio_id}")

        return {
            "wav_chunks": wav_chunks,
            "valid_lens": valid_lens,
            "labels_list": labels_list,
            "mmse_z": np.float32(mmse_z),
            "mmse_raw": np.float32(mmse_raw),
            "chunk_keys": chunk_keys,
            "wav_name": wav_name,
            "audio_id": audio_id,
        }

    def repeat30s_collate(self, batch):
        """segment_pred_mean collate_fn。"""
        flat_wav_chunks = []
        flat_valid_lens = []
        flat_labels = []
        flat_chunk_keys = []

        audio_mmse_z = []
        audio_mmse_raw = []
        audio_ids = []
        wav_names = []
        chunk_to_audio = []
        chunk_counts = []

        for audio_idx, item in enumerate(batch):
            wav_chunks = item["wav_chunks"]
            valid_lens = item["valid_lens"]
            labels_list = item["labels_list"]
            chunk_keys = item["chunk_keys"]

            n_chunks = int(len(wav_chunks))
            if n_chunks <= 0:
                raise ValueError(f"collate 收到空 chunk audio: {item.get('audio_id', '')}")

            audio_mmse_z.append(float(item["mmse_z"]))
            audio_mmse_raw.append(float(item["mmse_raw"]))
            audio_ids.append(str(item["audio_id"]))
            wav_names.append(str(item["wav_name"]))
            chunk_counts.append(n_chunks)

            for wav_30, valid_len, labels, chunk_key in zip(
                wav_chunks,
                valid_lens,
                labels_list,
                chunk_keys,
            ):
                wav_30 = np.asarray(wav_30, dtype=np.float32).reshape(-1)
                labels = pad_or_truncate_labels(labels, target_t=TOK_T_FULL)

                flat_wav_chunks.append(wav_30)
                flat_valid_lens.append(int(valid_len))
                flat_labels.append(labels)
                flat_chunk_keys.append(str(chunk_key))
                chunk_to_audio.append(int(audio_idx))

        flat_wav_30 = torch.from_numpy(
            np.stack(flat_wav_chunks, axis=0).astype(np.float32, copy=False)
        )
        flat_valid_len = torch.from_numpy(
            np.asarray(flat_valid_lens, dtype=np.int64)
        )
        flat_labels = torch.stack(flat_labels, dim=0).long()

        audio_mmse_z = torch.tensor(audio_mmse_z, dtype=torch.float32)
        audio_mmse_raw = torch.tensor(audio_mmse_raw, dtype=torch.float32)
        chunk_to_audio = torch.tensor(chunk_to_audio, dtype=torch.long)
        chunk_counts = torch.tensor(chunk_counts, dtype=torch.long)

        return (
            flat_wav_30,
            flat_valid_len,
            flat_labels,
            audio_mmse_z,
            audio_mmse_raw,
            chunk_to_audio,
            chunk_counts,
            flat_chunk_keys,
            wav_names,
            audio_ids,
        )

# Stage3 模型、损失与训练
class Stage3WhisperMPTAPTModule(pl.LightningModule):
    """Stage3 LightningModule。"""
    def __init__(
        self,
        whisper_model_name: str,
        stage1_ckpt_path: str,
        num_clusters: List[int],
        repr_layer: int,
        mask_inject_after_layer: int,
        target_mean: float,
        target_std: float,
        train_layer_start: int,
        train_layer_end: int,
        train_last_n: Optional[int] = None,
        lr_enc: float = 1e-5,
        lr_head: float = 7e-5,
        weight_decay: float = 0.02,
        mask_prob: float = 0.06,
        mask_span: int = 10,
        lambda_clus: float = 1.0,
        lambda_mmse: float = 0.2,
        head_dropout: float = 0.2,
        huber_delta: float = 1.0,
        l2_normalize_reps: bool = False,
        training_step: int = 2000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = WhisperModel.from_pretrained(whisper_model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model_name)
        self.hidden_size = int(self.model.config.d_model)

        self.num_clusters = [int(x) for x in num_clusters]
        self.repr_layer = int(repr_layer)
        self.mask_inject_after_layer = int(mask_inject_after_layer)
        self.target_mean = float(target_mean)
        self.target_std = float(max(1e-6, target_std))

        self.lr_enc = float(lr_enc)
        self.lr_head = float(lr_head)
        self.weight_decay = float(weight_decay)
        self.mask_prob = float(mask_prob)
        self.mask_span = int(mask_span)
        self.lambda_clus = float(lambda_clus)
        self.lambda_mmse = float(lambda_mmse)
        self.huber_delta = float(huber_delta)
        self.l2_normalize_reps = bool(l2_normalize_reps)
        self.total_training_steps = int(training_step)

        self.mask_emb = nn.Parameter(torch.zeros(self.hidden_size))
        nn.init.normal_(self.mask_emb, mean=0.0, std=0.02)

        self.cluster_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(float(head_dropout)),
                nn.Linear(self.hidden_size, int(k)),
            )
            for k in self.num_clusters
        ])

        self.mmse_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(float(head_dropout)),
            nn.Linear(self.hidden_size, 1),
        )

        load_stage1_ckpt_into_whisper_model(self.model, stage1_ckpt_path)

        self._set_trainable_layers(
            train_layer_start=int(train_layer_start),
            train_layer_end=int(train_layer_end),
            train_last_n=train_last_n,
        )

    def _set_trainable_layers(self, train_layer_start: int, train_layer_end: int, train_last_n: Optional[int]) -> None:
        """冻结 Whisper 大部分参数，只解冻指定 encoder 层。"""
        for p in self.model.parameters():
            p.requires_grad = False

        n_layers = int(self.model.config.encoder_layers)

        if train_last_n not in [None, "None", ""]:
            last_n = int(train_last_n)
            train_layer_start = max(1, n_layers - last_n + 1)
            train_layer_end = n_layers

        train_layer_start = max(1, int(train_layer_start))
        train_layer_end = min(n_layers, int(train_layer_end))

        if train_layer_start > train_layer_end:
            raise ValueError(
                f"训练层范围非法：start={train_layer_start}, end={train_layer_end}, n_layers={n_layers}"
            )

        for layer_id in range(train_layer_start, train_layer_end + 1):
            layer = self.model.encoder.layers[layer_id - 1]
            for p in layer.parameters():
                p.requires_grad = True

        if train_layer_end == n_layers and hasattr(self.model.encoder, "layer_norm"):
            for p in self.model.encoder.layer_norm.parameters():
                p.requires_grad = True

        print(
            f"[Stage3-MMSE] Whisper trainable encoder layers: "
            f"{train_layer_start}..{train_layer_end} / total={n_layers}"
        )

    def _build_valid_mask(self, input_features: torch.Tensor, valid_wav_lens: torch.Tensor) -> torch.Tensor:
        """根据每条音频真实有效长度，构造 token 级有效 mask：[B,T]。"""
        B = int(input_features.size(0))
        mel_T = int(input_features.size(-1))
        tok_T = (mel_T + 1) // 2

        if valid_wav_lens is None:
            return torch.ones((B, tok_T), dtype=torch.bool, device=input_features.device)

        vwl = valid_wav_lens.detach().cpu().long().view(-1)
        if vwl.numel() != B:
            raise RuntimeError(f"valid_wav_lens 数量与 batch 不一致：expect={B}, got={vwl.numel()}")

        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(vwl[i].item()))
            valid_Ts.append(min(vt, tok_T))

        valid_Ts = torch.tensor(valid_Ts, device=input_features.device, dtype=torch.long)
        t = torch.arange(tok_T, device=input_features.device).view(1, tok_T).expand(B, tok_T)
        return (t < valid_Ts.view(B, 1)).bool()

    def _encoder_layer_forward(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        """兼容不同 transformers 版本的 WhisperEncoderLayer forward 签名。"""
        try:
            out = layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False,
            )
        except TypeError:
            try:
                out = layer(hidden_states, attention_mask=None, output_attentions=False)
            except TypeError:
                out = layer(hidden_states)

        if isinstance(out, tuple):
            return out[0]
        return out

    def manual_encoder_forward(
        self,
        input_features: torch.Tensor,
        valid_wav_lens: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """手动执行 Whisper encoder，以便在第 mask_inject_after_layer 层后注入 mask embedding。"""
        enc = self.model.encoder

        hidden_states = F.gelu(enc.conv1(input_features))
        hidden_states = F.gelu(enc.conv2(hidden_states))
        hidden_states = hidden_states.permute(0, 2, 1)  # [B,T,C]

        seq_len = int(hidden_states.size(1))
        pos_emb = enc.embed_positions.weight[:seq_len, :].to(hidden_states.device)
        hidden_states = hidden_states + pos_emb.unsqueeze(0)

        dropout_p = float(getattr(enc, "dropout", 0.0))
        hidden_states = F.dropout(hidden_states, p=dropout_p, training=self.training)

        valid_mask = self._build_valid_mask(input_features, valid_wav_lens)

        if mask_positions is not None:
            mask_positions = mask_positions.to(hidden_states.device).bool()
            if mask_positions.shape != valid_mask.shape:
                raise RuntimeError(
                    f"mask_positions shape 与 valid_mask 不一致：{tuple(mask_positions.shape)} vs {tuple(valid_mask.shape)}"
                )
            mask_positions = mask_positions & valid_mask

        if int(self.mask_inject_after_layer) == 0 and mask_positions is not None:
            mask_vec = self.mask_emb.to(hidden_states.dtype).view(1, 1, -1)
            hidden_states = torch.where(mask_positions.unsqueeze(-1), mask_vec, hidden_states)

        selected_repr = None
        n_layers = len(enc.layers)

        if self.repr_layer < 1 or self.repr_layer > n_layers:
            raise ValueError(f"repr_layer={self.repr_layer} 非法；Whisper encoder 层数={n_layers}")

        for layer_idx, layer in enumerate(enc.layers, start=1):
            hidden_states = self._encoder_layer_forward(layer, hidden_states)

            if int(layer_idx) == int(self.mask_inject_after_layer) and mask_positions is not None:
                mask_vec = self.mask_emb.to(hidden_states.dtype).view(1, 1, -1)
                hidden_states = torch.where(mask_positions.unsqueeze(-1), mask_vec, hidden_states)

            if int(layer_idx) == int(self.repr_layer):
                selected_repr = hidden_states

        hidden_states = enc.layer_norm(hidden_states)

        if int(self.repr_layer) == n_layers:
            selected_repr = hidden_states

        if selected_repr is None:
            raise RuntimeError(f"未能提取 repr_layer={self.repr_layer} 的 hidden states。")

        return selected_repr, valid_mask

    def extract_features(
        self,
        wav_30: torch.Tensor,
        valid_len: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从 30s waveform 中提取 masked / unmasked 的 Whisper hidden states。"""
        if wav_30.dim() == 1:
            wav_30 = wav_30.unsqueeze(0)

        wav_cpu = wav_30.detach().cpu().numpy()
        target_len = int(self.feature_extractor.n_samples)

        feats = self.feature_extractor(
            wav_cpu,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=target_len,
        )

        input_features = feats.input_features.to(self.device)
        return self.manual_encoder_forward(
            input_features=input_features,
            valid_wav_lens=valid_len,
            mask_positions=mask_positions,
        )

    def _pool_utterance(self, reps: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """对 token 表示做有效区域 mean pooling。"""
        mask = valid_mask.to(reps.device).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        pooled = (reps * mask).sum(dim=1) / denom

        if self.l2_normalize_reps:
            pooled = F.normalize(pooled, dim=-1)

        return pooled

    def _shared_step(self, batch, stage: str):
        """train / val 共用逻辑。"""
        (
            wav_30,
            valid_len,
            labels,
            mmse_z,
            mmse_raw,
            chunk_to_audio,
            chunk_counts,
            _chunk_keys,
            _wav_names,
            _audio_ids,
        ) = batch

        wav_30 = wav_30.to(self.device)
        valid_len = valid_len.to(self.device).long()
        labels = labels.to(self.device).long()                  # [N_chunks, H, T]
        mmse_z = mmse_z.to(self.device).float().view(-1)        # [B_audio]
        mmse_raw = mmse_raw.to(self.device).float().view(-1)    # [B_audio]
        chunk_to_audio = chunk_to_audio.to(self.device).long().view(-1)  # [N_chunks]
        chunk_counts = chunk_counts.to(self.device).long().view(-1)      # [B_audio]

        label_valid = (labels[:, 0, :] != IGNORE_INDEX)
        mask_positions = make_time_mask_tok_valid(
            valid_mask_tok=label_valid,
            mask_prob=self.mask_prob,
            mask_span=self.mask_span,
        )

        reps, valid_mask = self.extract_features(
            wav_30=wav_30,
            valid_len=valid_len,
            mask_positions=mask_positions,
        )

        N_chunks, T, C = reps.shape
        H = labels.size(1)
        B_audio = int(mmse_z.numel())

        if chunk_to_audio.numel() != N_chunks:
            raise RuntimeError(
                f"chunk_to_audio 数量与 flat chunks 不一致: "
                f"chunk_to_audio={chunk_to_audio.numel()}, N_chunks={N_chunks}"
            )

        if chunk_counts.numel() != B_audio:
            raise RuntimeError(
                f"chunk_counts 数量与 audio batch 不一致: "
                f"chunk_counts={chunk_counts.numel()}, B_audio={B_audio}"
            )

        if B_audio <= 0:
            raise RuntimeError("audio-level batch 中 B_audio <= 0。")

        if int(chunk_to_audio.min().item()) < 0 or int(chunk_to_audio.max().item()) >= B_audio:
            raise RuntimeError(
                f"chunk_to_audio 超出 audio batch 范围: "
                f"min={int(chunk_to_audio.min().item())}, "
                f"max={int(chunk_to_audio.max().item())}, "
                f"B_audio={B_audio}"
            )

        if int(chunk_counts.sum().item()) != int(N_chunks):
            raise RuntimeError(
                f"chunk_counts.sum() 与 N_chunks 不一致: "
                f"chunk_counts.sum={int(chunk_counts.sum().item())}, "
                f"N_chunks={N_chunks}"
            )

        if H != len(self.cluster_heads):
            raise RuntimeError(
                f"labels head 数={H}，cluster_heads 数={len(self.cluster_heads)}，不一致。"
            )

        clus_losses = []
        clus_logk_scales = []
        total_masked = 0
        num_heads = len(self.cluster_heads)

        for h, head in enumerate(self.cluster_heads):
            target = labels[:, h, :]  # [N_chunks, T]
            target_valid = (target != IGNORE_INDEX)
            loss_mask = mask_positions[:, :T] & target_valid & valid_mask[:, :T]

            n = int(loss_mask.long().sum().item())
            total_masked += n

            logits = head(reps)  # [N_chunks, T, K]

            if n > 0:
                loss_h = F.cross_entropy(
                    logits[loss_mask],
                    target[loss_mask],
                    reduction="mean",
                )

                if num_heads > 1:
                    num_clusters_h = int(logits.size(-1))
                    logk_h = math.log(float(num_clusters_h))
                    loss_h = loss_h / logk_h
                    clus_logk_scales.append(logk_h)

            else:
                loss_h = logits.sum() * 0.0

            clus_losses.append(loss_h)

        clus_loss = torch.stack(clus_losses).mean()

        if num_heads > 1 and len(clus_logk_scales) > 0:
            clus_loss = clus_loss * (sum(clus_logk_scales) / len(clus_logk_scales))

        pooled_chunks = self._pool_utterance(reps, valid_mask)  # [N_chunks, C]

        pred_chunk_z = self.mmse_head(pooled_chunks).squeeze(-1)  # [N_chunks]

        audio_pred_sum = torch.zeros(
            (B_audio,),
            dtype=pred_chunk_z.dtype,
            device=pred_chunk_z.device,
        )
        audio_pred_sum.index_add_(0, chunk_to_audio, pred_chunk_z)

        audio_count = torch.zeros(
            (B_audio,),
            dtype=pred_chunk_z.dtype,
            device=pred_chunk_z.device,
        )
        audio_count.index_add_(
            0,
            chunk_to_audio,
            torch.ones(
                (N_chunks,),
                dtype=pred_chunk_z.dtype,
                device=pred_chunk_z.device,
            ),
        )
        audio_count = audio_count.clamp_min(1.0)

        pred_z = audio_pred_sum / audio_count  # [B_audio]

        mmse_loss = F.huber_loss(
            pred_z,
            mmse_z,
            delta=float(self.huber_delta),
            reduction="mean",
        )

        pred_raw = pred_z.detach() * self.target_std + self.target_mean
        utt_mae_raw = torch.mean(torch.abs(pred_raw - mmse_raw))

        total_loss = self.lambda_clus * clus_loss + self.lambda_mmse * mmse_loss

        self.log(
            f"{stage}_loss",
            total_loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=B_audio,
        )
        self.log(
            f"{stage}_clus_loss",
            clus_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=B_audio,
        )
        self.log(
            f"{stage}_mmse_loss",
            mmse_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=B_audio,
        )
        self.log(
            f"{stage}_utt_mae_raw",
            utt_mae_raw,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=B_audio,
        )

        if stage == "train":
            self.log(
                "train_masked_tokens",
                float(total_masked),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=B_audio,
            )
            self.log(
                "train_flat_chunks",
                float(N_chunks),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=B_audio,
            )
            self.log(
                "train_audio_batch_size",
                float(B_audio),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=B_audio,
            )
            self.log(
                "train_mean_chunks_per_audio",
                float(N_chunks) / float(max(B_audio, 1)),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=B_audio,
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="valid")

    def configure_optimizers(self):
        """分组学习率："""
        enc_params = [p for p in self.model.parameters() if p.requires_grad]
        head_params = list(self.cluster_heads.parameters()) + list(self.mmse_head.parameters()) + [self.mask_emb]

        param_groups = []
        if len(enc_params) > 0:
            param_groups.append({
                "params": enc_params,
                "lr": self.lr_enc,
                "weight_decay": self.weight_decay,
            })
        param_groups.append({
            "params": head_params,
            "lr": self.lr_head,
            "weight_decay": self.weight_decay,
        })

        optimizer = torch.optim.AdamW(param_groups)

        total_steps = max(1, int(self.total_training_steps))

        def lr_lambda(step):
            step = int(step)
            if step >= total_steps:
                return 0.0
            return 0.5 * (1.0 + math.cos(math.pi * step / float(total_steps)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

# 单次实验
def run_one_experiment(run_config: Dict[str, Any]) -> None:
    """执行单个 (dataset, repr_layer, mask_after_layer, fold) 的 Stage3 实验。"""
    seed_everything_for_repro(SEED)

    ensure_dir(run_config["STAGE3_SAVE_DIR"])
    ensure_dir(run_config["META_SAVE_DIR"])

    print("\n" + "=" * 90)
    print(f"[Stage3-MMSE] DATASET = {run_config['DATASET_OPTION']}")
    print(f"[Stage3-MMSE] REPR_LAYER = {run_config['REPR_LAYER']}")
    print(f"[Stage3-MMSE] LAYER_NAME = {run_config['LAYER_NAME']}")
    print(f"[Stage3-MMSE] UNFREEZE_ENCODER_LAYERS = {run_config['UNFREEZE_ENCODER_LAYERS']}")
    print(f"[Stage3-MMSE] RESOLVED_UNFREEZE_ENCODER_LAYERS = {run_config['RESOLVED_UNFREEZE_ENCODER_LAYERS']}")
    print(f"[Stage3-MMSE] MASK_OFFSET = {run_config['MASK_OFFSET']}")
    print(f"[Stage3-MMSE] MASK_INJECT_AFTER_LAYER = {run_config['MASK_INJECT_AFTER_LAYER']}")
    print(f"[Stage3-MMSE] FOLD_ID = {run_config['FOLD_ID']}")
    print(f"[Stage3-MMSE] FOLD_JSON_PATH = {run_config['FOLD_JSON_PATH']}")
    print(f"[Stage3-MMSE] STAGE1_SAVE_DIR = {run_config['STAGE1_SAVE_DIR']}")
    print(f"[Stage3-MMSE] STAGE2_TRAIN_LABELPATH = {run_config['STAGE2_TRAIN_LABELPATH']}")
    print(f"[Stage3-MMSE] STAGE2_VAL_LABELPATH = {run_config['STAGE2_VAL_LABELPATH']}")
    print(f"[Stage3-MMSE] STAGE2_TRAIN_MANIFEST = {run_config['STAGE2_TRAIN_MANIFEST']}")
    print(f"[Stage3-MMSE] STAGE2_VAL_MANIFEST = {run_config['STAGE2_VAL_MANIFEST']}")
    print(f"[Stage3-MMSE] STAGE3_SAVE_DIR = {run_config['STAGE3_SAVE_DIR']}")
    print("=" * 90)

    train_entries_raw = load_stage2_chunks_manifest(run_config["STAGE2_TRAIN_MANIFEST"])
    sanity_check_stage3_entries_have_target_mmse(train_entries_raw)

    train_entries = train_entries_raw
    train_missing_mmse = []

    val_entries_raw = []
    val_entries = []
    val_missing_mmse = []

    if len(train_entries) == 0:
        raise ValueError(f"Stage3 train chunks 为空：train={len(train_entries)}")

    target_mean, target_std, target_audio_count = compute_stage3_audio_level_target_stats(
        train_entries
    )

    train_clus_path = run_config["STAGE2_TRAIN_LABELPATH"]

    train_cluster_dict, train_k_list = load_all_clus_json(
        train_clus_path,
        expected_k_list=EXPECTED_NUM_CLUSTERS,
    )

    check_stage3_cluster_config(
        k_list=train_k_list,
        expected_num_clusters=EXPECTED_NUM_CLUSTERS,
        train_clus_path=train_clus_path,
        val_clus_path="NO_VAL_USED",
    )

    train_entries, train_missing_cluster = filter_entries_by_cluster_labels(
        train_entries,
        train_cluster_dict,
    )

    val_cluster_dict = {}
    val_k_list = []
    val_missing_cluster = []

    if len(train_entries) == 0:
        raise ValueError(f"Stage2 label 匹配后 train 为空：train={len(train_entries)}")

    print(f"[Stage3-MMSE][NO-VAL-LAST] train_chunks={len(train_entries)}")
    print(f"[Stage3-MMSE][NO-VAL-LAST] train_missing_mmse={len(train_missing_mmse)}")
    print(f"[Stage3-MMSE][NO-VAL-LAST] train_missing_cluster={len(train_missing_cluster)}")
    print(f"[Stage3-MMSE][NO-VAL-LAST] num_clusters={train_k_list}")
    print(
        f"[Stage3-MMSE][NO-VAL-LAST] "
        f"target_mean={target_mean:.4f}, "
        f"target_std={target_std:.4f}, "
        f"target_audio_count={target_audio_count}"
    )

    stage1_ckpt_path = find_stage1_checkpoint(
        stage1_save_dir=run_config["STAGE1_SAVE_DIR"],
        override_ckpt=STAGE1_CKPT_OVERRIDE,
    )

    if not os.path.isfile(stage1_ckpt_path):
        raise FileNotFoundError(f"[Stage3-MMSE] Stage1 checkpoint 不存在：{stage1_ckpt_path}")

    print(f"[Stage3-MMSE][NO-VAL-LAST] 使用的 Stage1 checkpoint = {stage1_ckpt_path}")

    train_ds = Stage3MPTAPTDataset(
        entries=train_entries,
        cluster_label_dict=train_cluster_dict,
        target_mean=target_mean,
        target_std=target_std,
        target_sec=TARGET_SEC,
    )

    train_loader = data.DataLoader(
        train_ds,
        batch_size=int(STAGE3_AUDIO_BATCH_SIZE),
        shuffle=True,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=train_ds.repeat30s_collate,
        pin_memory=True,
        persistent_workers=(int(NUM_WORKERS) > 0),
    )

    val_loader = None

    steps_per_epoch = int(math.ceil(len(train_loader) / max(1, int(ACCUMULATE_GRAD_BATCHES))))
    effective_training_steps = max(1, steps_per_epoch * int(STAGE3_MAX_EPOCHS))
    print(
        f"[Stage3-MMSE][NO-VAL-LAST] steps_per_epoch={steps_per_epoch}, "
        f"epochs={STAGE3_MAX_EPOCHS}, effective_training_steps={effective_training_steps}"
    )

    model = Stage3WhisperMPTAPTModule(
        whisper_model_name=WHISPER_MODEL_NAME,
        stage1_ckpt_path=stage1_ckpt_path,
        num_clusters=train_k_list,
        repr_layer=int(run_config["REPR_LAYER"]),
        mask_inject_after_layer=int(run_config["MASK_INJECT_AFTER_LAYER"]),
        target_mean=target_mean,
        target_std=target_std,
        train_layer_start=int(run_config["STAGE3_TRAIN_LAYER_START"]),
        train_layer_end=int(run_config["STAGE3_TRAIN_LAYER_END"]),
        train_last_n=None,
        lr_enc=float(LR_ENC),
        lr_head=float(LR_HEAD),
        weight_decay=float(WEIGHT_DECAY),
        mask_prob=float(MASK_PROB),
        mask_span=int(MASK_SPAN),
        lambda_clus=float(LAMBDA_CLUS),
        lambda_mmse=float(LAMBDA_MMSE),
        head_dropout=float(HEAD_DROPOUT),
        huber_delta=float(HUBER_DELTA),
        l2_normalize_reps=bool(L2_NORMALIZE_REPS),
        training_step=int(effective_training_steps),
    )

    stage3_config = {
        "dataset_option": str(run_config["DATASET_OPTION"]),
        "pitt_subject_disjoint": bool(str(run_config["DATASET_OPTION"]) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
        "pitt_split_policy": (PITT_SPLIT_POLICY if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
        "pitt_expected_final_train_subjects": (PITT_EXPECTED_FINAL_TRAIN_SUBJECTS if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
        "pitt_expected_final_train_visits": (PITT_EXPECTED_FINAL_TRAIN_VISITS if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
        "repr_layer": int(run_config["REPR_LAYER"]),
        "layer_name": str(run_config["LAYER_NAME"]),
        "unfreeze_encoder_layers": int(run_config["UNFREEZE_ENCODER_LAYERS"]),
        "resolved_unfreeze_encoder_layers": list(run_config["RESOLVED_UNFREEZE_ENCODER_LAYERS"]),
        "stage3_train_layer_start": int(run_config["STAGE3_TRAIN_LAYER_START"]),
        "stage3_train_layer_end": int(run_config["STAGE3_TRAIN_LAYER_END"]),
        "mask_offset": int(run_config["MASK_OFFSET"]),
        "mask_inject_after_layer": int(run_config["MASK_INJECT_AFTER_LAYER"]),
        "fold_id": None,
        "fold_disabled": True,
        "run_protocol": str(run_config.get("RUN_PROTOCOL_TAG", "")),
        "fold_json_path": "",
        "run_root": str(run_config["RUN_ROOT"]),
        "stage1_save_dir": str(run_config["STAGE1_SAVE_DIR"]),
        "stage1_ckpt_path": str(stage1_ckpt_path),
        "stage2_train_labelpath": str(run_config["STAGE2_TRAIN_LABELPATH"]),

        "stage2_train_manifest": str(run_config["STAGE2_TRAIN_MANIFEST"]),
        "stage2_val_labelpath": None,
        "stage2_val_manifest": None,
        "stage3_use_fold_val": False,
        "stage3_max_epochs": int(STAGE3_MAX_EPOCHS),
        "stage3_checkpoint_selection": str(STAGE3_CHECKPOINT_SELECTION),
        "effective_training_steps": int(effective_training_steps),
        "stage3_save_dir": str(run_config["STAGE3_SAVE_DIR"]),
        "whisper_model_name": str(WHISPER_MODEL_NAME),
        "num_clusters": [int(x) for x in train_k_list],
        "target_mean": float(target_mean),
        "target_std": float(target_std),
        "segment_sec": float(SEGMENT_SEC),
        "overlap_rate": float(OVERLAP_RATE),
        "chunk_key_version": str(CHUNK_KEY_VERSION),
        "target_sec": float(TARGET_SEC),
        "mask_prob": float(MASK_PROB),
        "mask_span": int(MASK_SPAN),
        "lambda_clus": float(LAMBDA_CLUS),
        "lambda_mmse": float(LAMBDA_MMSE),
        "STAGE3_AUX_REGRESSION_LEVEL": str(globals().get("STAGE3_AUX_REGRESSION_LEVEL", "chunk")),
        "STAGE3_AUDIO_BATCH_SIZE": int(globals().get("STAGE3_AUDIO_BATCH_SIZE", BATCH_SIZE)),
        "STAGE3_TARGET_NORMALIZATION_LEVEL": "audio",
        "train_num_chunks": int(len(train_entries)),
        "train_missing_mmse": int(len(train_missing_mmse)),
        "train_missing_cluster": int(len(train_missing_cluster)),

        "val_num_chunks": 0,
        "val_missing_mmse": 0,
        "val_missing_cluster": 0,
        "upstream_uses_fold_val": False,
    }
    save_json(stage3_config, os.path.join(run_config["STAGE3_SAVE_DIR"], "config.json"))
    save_json(stage3_config, os.path.join(run_config["META_SAVE_DIR"], "stage3_meta.json"))

    write_stage3_run_meta(
        cfg=run_config,
        stage1_ckpt_path=stage1_ckpt_path,
        train_clus_path=run_config["STAGE2_TRAIN_LABELPATH"],
        val_clus_path="NO_VAL_USED",
    )

    save_text(stage1_ckpt_path, os.path.join(run_config["META_SAVE_DIR"], "stage1_ckpt_used_for_stage3.txt"))
    save_text(run_config["STAGE2_TRAIN_LABELPATH"], os.path.join(run_config["STAGE3_SAVE_DIR"], "stage2_train_labelpath.txt"))
    save_text(run_config["STAGE2_TRAIN_MANIFEST"], os.path.join(run_config["STAGE3_SAVE_DIR"], "stage2_train_manifest.txt"))

    save_text("NO_VAL_USED", os.path.join(run_config["STAGE3_SAVE_DIR"], "stage2_val_labelpath.txt"))
    save_text("NO_VAL_USED", os.path.join(run_config["STAGE3_SAVE_DIR"], "stage2_val_manifest.txt"))

    if SAVE_MANIFESTS:
        save_json(
            {
                "train_entries": train_entries,
                "missing_mmse": train_missing_mmse,
                "missing_cluster": train_missing_cluster,
                "stage3_use_fold_val": False,
            },
            os.path.join(run_config["STAGE3_SAVE_DIR"], "train_manifest.json"),
        )
        save_json(
            {
                "val_entries": [],
                "missing_mmse": [],
                "missing_cluster": [],
                "stage3_use_fold_val": False,
                "note": "Stage3 no-val-last: fold_val is not used by upstream training.",
            },
            os.path.join(run_config["STAGE3_SAVE_DIR"], "val_manifest.json"),
        )

    callbacks = []

    ckpt_callback = ModelCheckpoint(
        dirpath=run_config["STAGE3_SAVE_DIR"],
        filename="whisper-s3-mmse-last",
        monitor=None,
        save_top_k=0,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(ckpt_callback)

    logger = None
    if ENABLE_CSV_LOGGER:
        logger = CSVLogger(
            save_dir=run_config["STAGE3_SAVE_DIR"],
            name="csv_logs",
        )

    trainer = Trainer(
        max_epochs=int(STAGE3_MAX_EPOCHS),
        max_steps=-1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision=int(PRECISION),
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=float(GRAD_CLIP),
        accumulate_grad_batches=int(ACCUMULATE_GRAD_BATCHES),
        log_every_n_steps=1,
        deterministic=False,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        ckpt_path=RESUME_STAGE3_CKPT,
    )

    if bool(ENABLE_STAGE_QUALITY_EVAL):
        raise RuntimeError(
            "no-val-last 设置下应保持 ENABLE_STAGE_QUALITY_EVAL=False，"
            "避免 Stage3 使用 fold_val 做质量判断。"
    )

    print(f"\n[Stage3-MMSE] 完成：{run_config['STAGE3_SAVE_DIR']}")

# 批量运行入口
def main() -> None:
    """主入口："""
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS 为空，至少需要一个数据集。")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST 为空，至少需要一个解冻层数。")
    if len(STAGE3_MASK_INJECT_AFTER_LAYERS) == 0:
        raise ValueError("STAGE3_MASK_INJECT_AFTER_LAYERS 不能为空。")

    print("[CONFIG]")
    print("  PROJECT_ROOT =", PROJECT_ROOT)
    print("  MMSE_PTAPT_ROOT =", MMSE_PTAPT_ROOT)
    print("  DATASET_OPTIONS =", DATASET_OPTIONS)
    print("  RUN_PROTOCOL_TAG =", RUN_PROTOCOL_TAG)
    print("  STAGE3_REPR_LAYERS_TO_RUN =", STAGE3_REPR_LAYERS_TO_RUN)
    print("  STAGE3_MASK_INJECT_AFTER_LAYERS =", STAGE3_MASK_INJECT_AFTER_LAYERS)
    print("  UNFREEZE_ENCODER_LAYERS_LIST =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  EXPECTED_NUM_CLUSTERS =", EXPECTED_NUM_CLUSTERS)
    print("  fold disabled = True")
    print("  protocol subdir disabled = True")

    for dataset_option in [str(x) for x in DATASET_OPTIONS]:
        for unfreeze_count in [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "=" * 100)
            print("[Stage3 BATCH ITEM]")
            print("  DATASET_OPTION =", DATASET_OPTION)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("=" * 100)

            for repr_layer in [int(x) for x in STAGE3_REPR_LAYERS_TO_RUN]:
                for mask_offset in [int(x) for x in STAGE3_MASK_INJECT_AFTER_LAYERS]:
                    run_config = build_stage3_run_config(
                        dataset_option=DATASET_OPTION,
                        repr_layer=int(repr_layer),
                        mask_offset=int(mask_offset),
                        fold_id=PROTOCOL_FOLD_ID,
                    )
                    run_one_experiment(run_config)

if __name__ == "__main__":
    main()
