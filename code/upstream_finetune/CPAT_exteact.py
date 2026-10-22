#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CPAT Stage3 checkpoint 的 Whisper 单层特征导出。
精简版保留原参数、音频/segment 顺序、归一化、标签/索引/sidecar 与保存路径。
Pitt 复用原受试者互斥划分规则，分别导出 final train 和 final test 特征。
仅模型前向；不更新参数。原文件名及跨阶段文件格式保持不变。

conda activate whisper_ptapt_faiss 
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune/ft_ptapt_repeat/CPAT_exteact.py
"""

import os
import random
import numpy as np

# GPU 设置；保留各文件原值及环境变量生效顺序。
GPU_ID = 4
if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

os.environ["PYTHONHASHSEED"] = "400"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(400)
np.random.seed(400)

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel

# 工程、中间结果与特征根目录；保留原路径。
PROJECT_ROOT = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune"
MMSE_PTAPT_ROOT = os.path.join(PROJECT_ROOT, "ft_ptapt_repeat")

PROJECT_ROOT2 = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT = os.path.join(PROJECT_ROOT2, "feature_tensor")
SPLIT_ROOT = os.path.join(PROJECT_ROOT, "split")

# 批量数据集；默认仅 ["ADReSS-IS2020"]，其他数据集分支仍保留。
DATASET_OPTIONS = ["ADReSS-IS2020"]

DATASET_OPTION = DATASET_OPTIONS[0]

# Pitt：原 test 中有有效 visit 的受试者全部归入 final test，其余归入 final train。
PITT_ENABLE_SUBJECT_DISJOINT_SPLIT = True
PITT_SPLIT_POLICY = "original_test_subjects_holdout"
PITT_STRICT_COUNT_CHECK = True
PITT_EXPECTED_CANDIDATE_VISITS = 477
PITT_EXPECTED_VALID_VISITS = 398
PITT_EXPECTED_UNIQUE_SUBJECTS = 252
PITT_EXPECTED_VALID_TEST_SUBJECTS = 68
PITT_EXPECTED_FINAL_TRAIN_SUBJECTS = 184
PITT_EXPECTED_FINAL_TEST_SUBJECTS = 68
PITT_EXPECTED_FINAL_TRAIN_VISITS = 276
PITT_EXPECTED_FINAL_TEST_VISITS = 122
_PITT_SUBJECT_DISJOINT_CACHE = None

# 各阶段层号保持联动；默认第 12 层。
LAYERS_TO_RUN = [12]

# 跨阶段模型根标签；各脚本必须一致。
MODEL_ROOT_TAG = "ptapt_lr5e-5_epo5"

# 相对当前表示层，向前连续解冻的层数；保留遍历顺序。
UNFREEZE_ENCODER_LAYERS_LIST = [8]

UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# 相对首个可训练层的 mask 偏移；不是绝对 encoder 层号。
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]

# 协议标签；当前训练不划分 fold，test 仅用于最终导出。
RUN_PROTOCOL_TAG = "official_train_test"

# 旧配置/manifest 的兼容占位；不新增 fold 目录。
PROTOCOL_FOLD_ID = 1
FOLDS_TO_RUN = [PROTOCOL_FOLD_ID]

# 读取 Stage1 checkpoint 的路径后缀；与实际 Stage1 输出对应。
STAGE1_SOURCE_PATH_PARAM_TAGS =  []

# singleX_unfreeY 的附加路径标签；Stage2、Stage3、导出需一致。
PATH_PARAM_TAGS = []

# 仅用于导出元信息；必须与真实 Stage2/Stage3 配置对应。
PTAPT_KS = [32, 64]
PTAPT_EXPECTED_NUM_CLUSTERS = [32, 64]
PTAPT_STAGE3_LAMBDA_MMSE = 0.05
PTAPT_METHOD_TAG = ""

# 先加载 Whisper 基模，再加载 Stage3 checkpoint 的 Whisper 权重。
WHISPER_BASE_MODEL_NAME = "/home/juanyu/whisper_lru_3/whisper_small"
LOCAL_FILES_ONLY = True

# 手动指定 Stage3 checkpoint；None 使用原自动查找规则。
STAGE3_CKPT_OVERRIDE = None

FEATURE_PREFIX = "whisper"

SAMPLE_RATE = 16000

# 固定切段长度与重叠率；chunk key 格式保持原样。
SEGMENT_SEC = 30
SEGMENT_LENGTH = int(SEGMENT_SEC * SAMPLE_RATE)
OVERLAP_RATE = 0.3
OV_STR = "0.3"
SEC_STR = f"{SEGMENT_SEC}s"

# 导出特征归一化：None / layernorm / std / ln_post。
NORM_FOR_SINGLE = "ln_post"
DROP_INPUT_LAYER = True

ENABLE_AUDIO_NORMALIZE = True

SAVE_META = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(400)
torch.cuda.manual_seed_all(400)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def ensure_dir(path: str) -> None:
    """若目录不存在，则自动创建。"""
    os.makedirs(path, exist_ok=True)

def natural_key(p: str):
    """文件名自然排序："""
    s = Path(str(p)).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

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

def build_stage3_subdir(mask_offset: int) -> str:
    """与 whisper_stage3_mmse_v3.py 保持一致的 Stage3 checkpoint 子目录名。"""
    return f"stage3_mptapt_maskoff{int(mask_offset)}"

def get_ft_model_name(mask_offset: int) -> str:
    """构造给 LRU_mmse_fold5_v2.py 使用的 FT_MODEL_NAME。"""
    return f"{MODEL_ROOT_TAG}"

def get_feature_model_tag(mask_offset: int) -> str:
    """feature_tensor 下真实模型目录名：FT_MODEL_NAME + _ft。"""
    return f"ft_{get_ft_model_name(mask_offset)}"

def get_feature_selection_tag(single_layer: int) -> str:
    """feature_tensor 下的 selection 目录名，与 Stage3 v3 的 layer_name 对齐。"""
    return build_layer_name(
        single_layer=single_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

def canonical_wav_path(path: str) -> str:
    """规范化 wav 路径，避免相对路径 / 符号链接导致匹配不一致。"""
    return str(Path(str(path)).expanduser().resolve())

def save_json(obj: Dict[str, Any], path: str) -> None:
    """保存 JSON，并自动创建父目录。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(text: str, path: str) -> None:
    """保存纯文本，并自动创建父目录。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def torch_load_compatible(path: str, map_location="cpu"):
    """兼容 PyTorch 2.6+ 的 torch.load。"""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

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

# 数据集音频目录。
DATASETS = {
    "PROCESS-2": {
        "train_cc": "/home/juanyu/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/juanyu/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/juanyu/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/juanyu/Downloads/PROCESS-2/test/cd",
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
}

# MMSE 元数据解析
def _pick_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """从候选列名中找到 DataFrame 中真实存在的一列。"""
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"未找到候选列 {candidates}，当前列有：{list(df.columns)}")

def _safe_to_float(x):
    """安全转 float；NA / 空值返回 None。"""
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

def _normalize_mmse_key(dataset_key: str, raw_id):
    """统一 MMSE 映射键："""
    if raw_id is None or pd.isna(raw_id):
        return None

    s = str(raw_id).strip()
    s = Path(s).stem

    dataset_key_norm = str(dataset_key).strip().lower()

    if dataset_key_norm == "pitt":
        m = re.search(r"\d+", s)
        if m is None:
            return None
        return str(int(m.group(0)))

    if dataset_key_norm == "process-2":
        s = re.sub(r"__(SFT|PFT|CTD)$", "", s, flags=re.IGNORECASE)
        return s.lower()

    return s.lower()

def _extract_pitt_subject_visit(raw_name: str) -> Tuple[Optional[str], Optional[int]]:
    """从 Pitt 文件名解析 subject_id 和 visit_index。"""
    if raw_name is None or pd.isna(raw_name):
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    return str(int(m.group(1))), int(m.group(2))

def _pitt_visit_to_mmse_col(visit_index: int) -> str:
    """Pitt visit 到 MMSE 列名的映射："""
    visit_index = int(visit_index)
    if visit_index < 0:
        raise ValueError(f"非法 visit_index={visit_index}")
    if visit_index == 0:
        return "mms"
    return f"mmse{visit_index + 1}"

def _build_map_from_rows(df: pd.DataFrame, dataset_key: str, id_col: str, mmse_col: str) -> Dict[str, float]:
    """从 DataFrame 两列构建 {规范化ID: mmse}。"""
    mmse_map = {}
    for _, row in df.iterrows():
        key = _normalize_mmse_key(dataset_key, row[id_col])
        val = _safe_to_float(row[mmse_col])
        if key is None or val is None:
            continue
        mmse_map[key] = float(val)
    return mmse_map

def _read_adress_is2020_mmse(cfg: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """读取 ADReSS-IS2020 的 MMSE txt 元数据。"""
    def _read_txt(txt_path: str):
        mmse_map = {}
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (not line) or line.startswith("ID"):
                    continue
                parts = [x.strip() for x in line.split(";")]
                if len(parts) < 2:
                    continue
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

def _read_process2_mmse(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """解析 PROCESS-2 的 meta-info.csv，构建四个子集的 MMSE 映射。"""
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

def _read_adresso21_mmse(cfg: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """读取 ADReSSo21 的 MMSE csv 元数据。"""
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

def _read_pitt_mmse(cfg: Dict[str, Any]) -> Dict[str, Dict[Tuple[str, int], float]]:
    """读取 Pitt visit-level MMSE 映射。"""
    df = pd.read_excel(
        cfg["all_meta"],
        sheet_name=cfg.get("sheet_name", "data"),
        header=int(cfg.get("header_row", 2)),
    )
    id_col = _pick_first_existing_column(df, [cfg.get("id_col", "id"), "id"])
    col_map = {str(c).strip().lower(): c for c in df.columns}

    all_map: Dict[Tuple[str, int], float] = {}
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

def load_mmse_maps_for_dataset(dataset_key: str) -> Dict[str, Dict[Any, float]]:
    """加载指定数据集的 MMSE 映射。"""
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"暂未配置数据集 {dataset_key} 的 MMSE_META_PATHS")

    cfg = MMSE_META_PATHS[dataset_key]
    if dataset_key == "ADReSS-IS2020":
        return _read_adress_is2020_mmse(cfg)
    if dataset_key == "ADReSSo21":
        return _read_adresso21_mmse(cfg)
    if dataset_key == "Pitt":
        return _read_pitt_mmse(cfg)
    if dataset_key == "PROCESS-2":
        return _read_process2_mmse(cfg)

    raise ValueError(f"暂未实现数据集 {dataset_key} 的 MMSE 读取。")

def resolve_mmse_for_entry(row: Dict[str, Any], dataset_key: str, mmse_maps: Dict[str, Dict[Any, float]]) -> Tuple[Optional[float], str]:
    """为 official train / official test entry 获取 MMSE。"""
    if row.get("target_mmse", None) not in [None, ""]:
        val = _safe_to_float(row.get("target_mmse"))
        if val is not None:
            return float(val), "from_entry"

    wav_path = row.get("wav_path", "")
    wav_stem = Path(str(wav_path)).stem

    group = str(row.get("group", "")).strip().lower()
    split_name = str(row.get("split", "train")).strip().lower()

    if group not in ["cc", "cd"]:
        return None, f"invalid_group_{group}"

    is_official_test = split_name in ["test", "official_test", "test_cc", "test_cd"]

    if is_official_test:
        bucket = "test_cc" if group == "cc" else "test_cd"
    else:
        bucket = "train_cc" if group == "cc" else "train_cd"

    if dataset_key == "Pitt":
        subject_key, visit_index = _extract_pitt_subject_visit(wav_stem)
        if subject_key is None or visit_index is None:
            return None, "cannot_parse_pitt_subject_visit"

        key = (subject_key, int(visit_index))
        mmse_map = mmse_maps.get(bucket, {})
        if key not in mmse_map:
            return None, f"missing_{bucket}_{_pitt_visit_to_mmse_col(visit_index)}"

        return float(mmse_map[key]), f"from_mmse_meta_{bucket}"

    sid = _normalize_mmse_key(dataset_key, wav_stem)
    mmse_map = mmse_maps.get(bucket, {})

    if sid not in mmse_map:
        return None, f"missing_mmse_{bucket}"

    return float(mmse_map[sid]), f"from_mmse_meta_{bucket}"

def attach_mmse_and_filter(entries: List[Dict[str, Any]], dataset_key: str, mmse_maps: Dict[str, Dict[Any, float]], subset_name: str):
    """给 entries 回填 target_mmse。"""
    valid_entries = []
    missing_records = []

    for row in entries:
        mmse, source = resolve_mmse_for_entry(row, dataset_key, mmse_maps)
        if mmse is None:
            rec = {
                "dataset": dataset_key,
                "subset": subset_name,
                "wav_name": row.get("wav_name", ""),
                "wav_path": row.get("wav_path", ""),
                "group": row.get("group", ""),
                "reason": source,
            }
            missing_records.append(rec)
            continue

        row = copy.deepcopy(row)
        row["target_mmse"] = float(mmse)
        row["mmse_source"] = source
        valid_entries.append(row)

    return valid_entries, missing_records

def _is_process2_ctd_wav_path(wav_path: str, dataset_key: str = None) -> bool:
    """PROCESS-2 专用 wav 过滤："""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None

def list_wav_files(audio_dir: str, dataset_key: str = None) -> List[str]:
    """读取某个 official 子目录下的 wav 文件，并按 natural_key 排序。"""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    audio_dir = str(audio_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"official audio dir 不存在：{audio_dir}")

    files = []
    for f in os.listdir(audio_dir):
        if not str(f).lower().endswith(".wav"):
            continue

        wav_path = os.path.join(audio_dir, f)

        if not _is_process2_ctd_wav_path(wav_path, dataset_key=dataset_key):
            continue

        files.append(wav_path)

    files = sorted(files, key=natural_key)

    if len(files) == 0:
        raise ValueError(
            f"official audio dir 中没有符合条件的 wav 文件：{audio_dir}, "
            f"dataset_key={dataset_key}"
        )

    if str(dataset_key).strip().lower() == "process-2":
        print(f"[PROCESS-2][CTD] {audio_dir}: 只读取 CTD wav，文件数 = {len(files)}")

    return [canonical_wav_path(x) for x in files]

def build_entries_from_official_dir(
    audio_dir: str,
    split_name: str,
    group_name: str,
    dataset_key: str = None,
) -> List[Dict[str, Any]]:
    """把 official train_cc/train_cd/test_cc/test_cd 目录转成 entry 列表。"""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    split_name = str(split_name).strip().lower()
    group_name = str(group_name).strip().lower()

    if split_name not in ["train", "test"]:
        raise ValueError(f"split_name 必须是 train/test，当前={split_name}")
    if group_name not in ["cc", "cd"]:
        raise ValueError(f"group_name 必须是 cc/cd，当前={group_name}")

    entries: List[Dict[str, Any]] = []
    for wav_path in list_wav_files(audio_dir, dataset_key=dataset_key):
        entries.append({
            "wav_path": canonical_wav_path(wav_path),
            "wav_name": Path(wav_path).name,
            "split": split_name,
            "group": group_name,
            "source_subset": f"{split_name}_{group_name}",
            "process2_task_filter": "CTD" if str(dataset_key).strip().lower() == "process-2" else "",
        })

    return entries

# Pitt 受试者互斥划分与审计
def _pitt_make_missing_visit_record(source_subset: str, wav_path: str, subject_id=None, visit_index=None) -> Dict[str, Any]:
    """Pitt subject-disjoint 重建时记录无法进入 MMSE regression 的 visit。"""
    rec = {
        "dataset": "Pitt",
        "source_subset": str(source_subset),
        "wav_name": Path(wav_path).name,
        "wav_path": canonical_wav_path(wav_path),
        "subject_id": subject_id,
        "visit_index": visit_index,
        "expected_mmse_col": None,
        "reason": "missing_mmse",
    }
    if subject_id is not None and visit_index is not None:
        rec["expected_mmse_col"] = _pitt_visit_to_mmse_col(int(visit_index))
        rec["reason"] = f"missing_{rec['expected_mmse_col']}"
    return rec

def _save_pitt_subject_disjoint_audit_extract(
    visit_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    missing_records: List[Dict[str, Any]],
) -> Dict[str, str]:
    """保存与 Whisper_pitt_new(1).py / SCPA Stage1 同口径的 Pitt split 审计文件。"""
    split_root = os.path.join(FEATURE_ROOT, "Pitt", "subject_disjoint_split")
    ensure_dir(split_root)

    visit_path = os.path.join(split_root, "pitt_valid_visits_subject_disjoint_manifest.csv")
    subject_path = os.path.join(split_root, "pitt_subject_assignment.csv")
    missing_path = os.path.join(split_root, "pitt_missing_mmse_visits.csv")

    visit_df.to_csv(visit_path, index=False, encoding="utf-8-sig")
    subject_df.to_csv(subject_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(missing_records).to_csv(missing_path, index=False, encoding="utf-8-sig")

    print(f"[Pitt split] visit manifest -> {visit_path}")
    print(f"[Pitt split] subject assignment -> {subject_path}")
    print(f"[Pitt split] missing-MMSE audit -> {missing_path}")
    return {
        "visit_manifest": visit_path,
        "subject_assignment": subject_path,
        "missing_mmse_audit": missing_path,
    }

def _build_pitt_subject_disjoint_train_test_entries() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """与 Whisper_pitt_new(1).py 一致地重建 Pitt final train/test。"""
    global _PITT_SUBJECT_DISJOINT_CACHE

    if _PITT_SUBJECT_DISJOINT_CACHE is not None:
        cached = _PITT_SUBJECT_DISJOINT_CACHE
        return copy.deepcopy(cached["train_entries"]), copy.deepcopy(cached["test_entries"])

    if PITT_SPLIT_POLICY != "original_test_subjects_holdout":
        raise ValueError(f"不支持的 PITT_SPLIT_POLICY={PITT_SPLIT_POLICY}")

    dataset_paths = DATASETS["Pitt"]
    mmse_maps = load_mmse_maps_for_dataset("Pitt")
    all_mmse_map = mmse_maps["train_cc"]

    source_subsets = ["train_cc", "train_cd", "test_cc", "test_cd"]
    records: List[Dict[str, Any]] = []
    missing_records: List[Dict[str, Any]] = []
    seen_visit_keys: Dict[Tuple[str, int], str] = {}
    candidate_count = 0

    for source_subset in source_subsets:
        audio_dir = dataset_paths[source_subset]
        files = list_wav_files(audio_dir, dataset_key="Pitt")
        candidate_count += len(files)
        source_split, group = source_subset.split("_", 1)

        for fp in files:
            fp = canonical_wav_path(fp)
            subject_id, visit_index = _extract_pitt_subject_visit(Path(fp).stem)
            if subject_id is None or visit_index is None:
                missing_records.append(
                    _pitt_make_missing_visit_record(source_subset, fp, subject_id, visit_index)
                )
                continue

            visit_key = (str(subject_id), int(visit_index))
            if visit_key not in all_mmse_map:
                missing_records.append(
                    _pitt_make_missing_visit_record(source_subset, fp, subject_id, visit_index)
                )
                continue

            if visit_key in seen_visit_keys:
                raise RuntimeError(
                    "[Pitt split] 发现重复的 subject-visit key："
                    f"{visit_key}\n  first={seen_visit_keys[visit_key]}\n  second={fp}"
                )
            seen_visit_keys[visit_key] = fp

            records.append({
                "subject_id": str(subject_id),
                "visit_index": int(visit_index),
                "mmse": float(all_mmse_map[visit_key]),
                "group": str(group),
                "source_subset": str(source_subset),
                "source_split": str(source_split),
                "wav_name": Path(fp).name,
                "wav_path": fp,
            })

    valid_visit_count = len(records)
    valid_subjects = {r["subject_id"] for r in records}
    original_train_subjects = {r["subject_id"] for r in records if r["source_split"] == "train"}
    original_test_subjects = {r["subject_id"] for r in records if r["source_split"] == "test"}

    final_test_subjects = set(original_test_subjects)
    final_train_subjects = set(valid_subjects) - final_test_subjects
    if final_train_subjects & final_test_subjects:
        raise RuntimeError(f"[Pitt split] subject-disjoint 检查失败: {sorted(final_train_subjects & final_test_subjects)}")

    final_records: Dict[str, List[Dict[str, Any]]] = {
        "train_cc": [], "train_cd": [], "test_cc": [], "test_cd": []
    }
    for rec in records:
        rec["final_split"] = "test" if rec["subject_id"] in final_test_subjects else "train"
        rec["final_subset"] = f"{rec['final_split']}_{rec['group']}"
        final_records[rec["final_subset"]].append(rec)

    for subset_name in final_records:
        final_records[subset_name] = sorted(
            final_records[subset_name],
            key=lambda r: (natural_key(r["wav_path"]), str(r["wav_path"])),
        )

    final_train_ids = {r["subject_id"] for r in records if r["final_split"] == "train"}
    final_test_ids = {r["subject_id"] for r in records if r["final_split"] == "test"}
    final_overlap = final_train_ids & final_test_ids
    if final_overlap:
        raise RuntimeError(f"[Pitt split] 最终 train/test 仍存在 subject overlap: {sorted(final_overlap)}")
    if len(final_train_ids | final_test_ids) != len(valid_subjects):
        raise RuntimeError("[Pitt split] 最终 subject union 与有效 subject 集不一致。")
    if sum(len(v) for v in final_records.values()) != valid_visit_count:
        raise RuntimeError("[Pitt split] 最终 visit 总数与有效 visit 总数不一致。")

    final_train_visit_count = len(final_records["train_cc"]) + len(final_records["train_cd"])
    final_test_visit_count = len(final_records["test_cc"]) + len(final_records["test_cd"])
    expected_checks = [
        ("candidate_visits", candidate_count, PITT_EXPECTED_CANDIDATE_VISITS),
        ("valid_visits", valid_visit_count, PITT_EXPECTED_VALID_VISITS),
        ("unique_subjects", len(valid_subjects), PITT_EXPECTED_UNIQUE_SUBJECTS),
        ("valid_original_test_subjects", len(original_test_subjects), PITT_EXPECTED_VALID_TEST_SUBJECTS),
        ("final_train_subjects", len(final_train_ids), PITT_EXPECTED_FINAL_TRAIN_SUBJECTS),
        ("final_test_subjects", len(final_test_ids), PITT_EXPECTED_FINAL_TEST_SUBJECTS),
        ("final_train_visits", final_train_visit_count, PITT_EXPECTED_FINAL_TRAIN_VISITS),
        ("final_test_visits", final_test_visit_count, PITT_EXPECTED_FINAL_TEST_VISITS),
    ]
    for name, actual, expected in expected_checks:
        if int(actual) != int(expected):
            msg = f"[Pitt split] {name}={actual}，预期={expected}"
            if PITT_STRICT_COUNT_CHECK:
                raise RuntimeError(msg + "。为避免无意改变实验数据，已停止。")
            print("[WARN]", msg)

    visit_df = pd.DataFrame(records).sort_values(
        ["final_split", "subject_id", "visit_index", "wav_name"], kind="stable"
    ).reset_index(drop=True)

    subject_rows: List[Dict[str, Any]] = []
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
    audit_paths = _save_pitt_subject_disjoint_audit_extract(visit_df, subject_df, missing_records)

    ordered_train_records = list(final_records["train_cc"]) + list(final_records["train_cd"])
    ordered_test_records = list(final_records["test_cc"]) + list(final_records["test_cd"])

    def _to_entry(rec: Dict[str, Any], audio_id: int) -> Dict[str, Any]:
        return {
            "wav_path": canonical_wav_path(rec["wav_path"]),
            "wav_name": str(rec["wav_name"]),
            "split": str(rec["final_split"]),
            "group": str(rec["group"]),
            "source_subset": str(rec["final_subset"]),
            "original_source_subset": str(rec["source_subset"]),
            "original_source_split": str(rec["source_split"]),
            "final_subset": str(rec["final_subset"]),
            "subject_id": str(rec["subject_id"]),
            "visit_index": int(rec["visit_index"]),
            "target_mmse": float(rec["mmse"]),
            "mmse_source": "pitt_visit_level_meta_subject_disjoint",
            "official_audio_id": int(audio_id),
            "audio_id": int(audio_id),
        }

    train_entries = [_to_entry(rec, i) for i, rec in enumerate(ordered_train_records)]
    test_entries = [_to_entry(rec, i) for i, rec in enumerate(ordered_test_records)]

    original_overlap = original_train_subjects & original_test_subjects
    print("\n" + "=" * 100)
    print("[Pitt subject-disjoint final split | SCPA Extract]")
    print(f"policy = {PITT_SPLIT_POLICY}")
    print(f"candidate visits = {candidate_count}")
    print(f"valid-MMSE visits = {valid_visit_count}")
    print(f"missing-MMSE/unusable-for-regression visits = {len(missing_records)}")
    print(f"valid unique subjects = {len(valid_subjects)}")
    print(
        "original valid subject counts | "
        f"train={len(original_train_subjects)}, test={len(original_test_subjects)}, overlap={len(original_overlap)}"
    )
    print(
        "final subject counts | "
        f"train={len(final_train_ids)}, test={len(final_test_ids)}, overlap={len(final_overlap)}"
    )
    print(
        "final visit counts | "
        f"train_cc={len(final_records['train_cc'])}, train_cd={len(final_records['train_cd'])}, "
        f"test_cc={len(final_records['test_cc'])}, test_cd={len(final_records['test_cd'])}, "
        f"train_total={final_train_visit_count}, test_total={final_test_visit_count}, total={valid_visit_count}"
    )
    print("=" * 100 + "\n")

    _PITT_SUBJECT_DISJOINT_CACHE = {
        "train_entries": copy.deepcopy(train_entries),
        "test_entries": copy.deepcopy(test_entries),
        "visit_df": visit_df.copy(),
        "subject_df": subject_df.copy(),
        "missing_records": copy.deepcopy(missing_records),
        "audit_paths": dict(audit_paths),
    }
    return copy.deepcopy(train_entries), copy.deepcopy(test_entries)

def load_official_train_test_entries(dataset_option: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """构造 official train / official test entries。"""
    if dataset_option not in DATASETS:
        raise ValueError(f"DATASETS 中没有配置 DATASET_OPTION={dataset_option}")

    if str(dataset_option) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return _build_pitt_subject_disjoint_train_test_entries()

    paths = DATASETS[dataset_option]

    train_entries = (
        build_entries_from_official_dir(
            paths["train_cc"],
            split_name="train",
            group_name="cc",
            dataset_key=dataset_option,
        )
        + build_entries_from_official_dir(
            paths["train_cd"],
            split_name="train",
            group_name="cd",
            dataset_key=dataset_option,
        )
    )

    test_entries = (
        build_entries_from_official_dir(
            paths["test_cc"],
            split_name="test",
            group_name="cc",
            dataset_key=dataset_option,
        )
        + build_entries_from_official_dir(
            paths["test_cd"],
            split_name="test",
            group_name="cd",
            dataset_key=dataset_option,
        )
    )

    if str(dataset_option).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] official entries: "
            f"train={len(train_entries)}, test={len(test_entries)}"
        )

    if len(train_entries) == 0 or len(test_entries) == 0:
        raise ValueError(
            f"official train/test entries 为空：train={len(train_entries)}, test={len(test_entries)}"
        )

    for i, row in enumerate(train_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    for i, row in enumerate(test_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    return train_entries, test_entries

def build_run_root(dataset_option: str, layer_id: int, fold_id: int = PROTOCOL_FOLD_ID) -> str:
    """构造 Stage3 checkpoint 对应的 run_root。"""
    layer_name = build_layer_name(
        single_layer=int(layer_id),
        unfreeze_layer_count=int(UNFREEZE_ENCODER_LAYERS),
        path_param_tags=PATH_PARAM_TAGS,
    )

    return os.path.join(
        MMSE_PTAPT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        layer_name,
    )

def find_stage3_checkpoint(
    dataset_option: str,
    layer_id: int,
    fold_id: int,
    mask_offset: int,
    override_ckpt: Optional[str] = None,
) -> str:
    """自动查找 Stage3 v3 checkpoint。"""
    if override_ckpt not in [None, "", "None"]:
        ckpt = str(override_ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"STAGE3_CKPT_OVERRIDE 不存在：{ckpt}")
        return ckpt

    run_root = build_run_root(dataset_option, layer_id, fold_id)
    stage3_dir = Path(run_root) / build_stage3_subdir(mask_offset)
    if not stage3_dir.exists():
        raise FileNotFoundError(
            "Stage3 v3 目录不存在：{}\n"
            "请确认 whisper_stage3_mmse_v3.py 的 DATASET_OPTION / LAYERS_TO_RUN / "
            "UNFREEZE_ENCODER_LAYERS / STAGE3_MASK_INJECT_AFTER_LAYERS / FOLDS_TO_RUN "
            "与本脚本一致。".format(stage3_dir)
        )

    all_ckpts = sorted(stage3_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    non_last = [p for p in all_ckpts if p.name != "last.ckpt"]

    if len(non_last) > 0:
        return str(non_last[0].resolve())

    last_ckpt = stage3_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt.resolve())

    raise FileNotFoundError(f"在 {stage3_dir} 下未找到 Stage3 checkpoint。")

def load_stage3_ckpt_into_whisper_model(model: WhisperModel, ckpt_path: str) -> None:
    """把 Stage3 Lightning checkpoint 中的 WhisperModel 权重加载到当前 HuggingFace WhisperModel。"""
    ckpt = torch_load_compatible(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    target_sd = model.state_dict()
    candidate_prefixes = ["model.", "whisper.model.", "whisper.", ""]

    best_loaded = {}
    best_prefix = None

    for pref in candidate_prefixes:
        loaded = {}
        for k, v in state.items():
            if pref != "" and not str(k).startswith(pref):
                continue
            new_k = str(k)[len(pref):] if pref != "" else str(k)
            if new_k in target_sd and tuple(target_sd[new_k].shape) == tuple(v.shape):
                loaded[new_k] = v

        if len(loaded) > len(best_loaded):
            best_loaded = loaded
            best_prefix = pref

    if len(best_loaded) == 0:
        raise RuntimeError(f"没有从 Stage3 checkpoint 中匹配到任何 WhisperModel 参数：{ckpt_path}")

    missing, unexpected = model.load_state_dict(best_loaded, strict=False)
    print(
        f"[Stage3-CKPT] loaded checkpoint: {ckpt_path}\n"
        f"[Stage3-CKPT] prefix={best_prefix!r}, loaded_params={len(best_loaded)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )

# Whisper 特征归一化与导出
def apply_norm(h: torch.Tensor, mode: Optional[str], ln_post=None, skip_ln_post: bool = False):
    """对 [1,T,C] 特征做归一化。"""
    if mode is None:
        return h
    if mode == "layernorm":
        return F.layer_norm(h, h.shape[-1:])
    if mode == "std":
        std = h.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return h / std
    if mode in ("ln_post", "whisper_ln"):
        if skip_ln_post:
            return h
        if ln_post is None:
            raise ValueError("ln_post is None，但选择了 ln_post/whisper_ln。")
        return ln_post(h)
    raise ValueError(f"Unknown norm mode: {mode}")

@torch.inference_mode()
def get_segment_features(
    audio_file: str,
    audio_file_index: int,
    processor: WhisperProcessor,
    model: WhisperModel,
    single_layer: int,
    norm_single: Optional[str],
    ln_post,
    top_layer_index: int,
    drop_input_layer: bool = True,
    segment_length: int = SEGMENT_LENGTH,
    sampling_rate: int = SAMPLE_RATE,
    overlap_rate: float = OVERLAP_RATE,
):
    """对单个 wav 按 30s + overlap 切片，并抽取指定 single layer 特征。"""
    y, _ = librosa.load(audio_file, sr=sampling_rate, mono=True, dtype=np.float32)

    if ENABLE_AUDIO_NORMALIZE:
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-8:
            y = y / peak

    step_size = int(segment_length * (1 - overlap_rate))
    if step_size <= 0:
        raise ValueError(f"overlap_rate={overlap_rate} 导致 step_size<=0")

    total_segments = int(len(y) / step_size) + (
        1 if len(y) % step_size > overlap_rate * segment_length else 0
    )
    total_segments = max(1, total_segments)

    seg_features = []
    seg_indices = []
    warned_double_ln_top = False

    for i in range(total_segments):
        start = i * step_size
        end = i * step_size + segment_length
        segment = y[start:end]

        if len(segment) < segment_length and i > 0:
            padding_needed = segment_length - len(segment)
            prev_end = start + segment_length - step_size
            padding_start = max(0, prev_end - padding_needed)
            padding_values = y[padding_start:prev_end]
            segment = np.concatenate((segment, padding_values))

        inp = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inp.input_features.to(DEVICE)

        enc_out = model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        hs_all = list(enc_out.hidden_states)
        hs = hs_all[1:] if (drop_input_layer and len(hs_all) >= 2) else hs_all

        try:
            already_post = torch.allclose(hs[-1], enc_out.last_hidden_state, atol=1e-5, rtol=1e-5)
        except Exception:
            already_post = True

        if not (1 <= int(single_layer) <= len(hs)):
            raise ValueError(f"single_layer={single_layer} 超出范围，当前可用层数={len(hs)}")

        is_top = (int(single_layer) == len(hs))
        if (not warned_double_ln_top and is_top and already_post and norm_single in ("ln_post", "whisper_ln")):
            print(f"[WARN] {Path(audio_file).name}: 顶层已 post-LN，外部又设 ln_post，已跳过第二次 ln_post。")
            warned_double_ln_top = True

        h = hs[int(single_layer) - 1]
        skip_ln = (norm_single in ("ln_post", "whisper_ln")) and (
            is_top or int(single_layer) == int(top_layer_index)
        )
        fused = apply_norm(h, norm_single, ln_post=ln_post, skip_ln_post=skip_ln)

        fused = fused.squeeze(0).detach().cpu()  # [T,C]
        seg_features.append(fused)
        seg_indices.append((int(audio_file_index), int(i)))

    return seg_features, seg_indices

def stack_with_time_padding(feature_list: List[torch.Tensor], target_t: Optional[int] = None, pad_value: float = 0.0):
    """把一组 [T,C] 特征补齐到统一时间长度，然后 stack 成 [N,T,C]。"""
    if len(feature_list) == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)

    for i, feat in enumerate(feature_list):
        if feat.ndim != 2:
            raise ValueError(f"feature_list[{i}] 不是二维张量，当前形状={tuple(feat.shape)}")

    feat_dim = int(feature_list[0].shape[1])
    for i, feat in enumerate(feature_list):
        if int(feat.shape[1]) != feat_dim:
            raise ValueError(f"feature_list[{i}] 特征维不一致：{feat.shape[1]} != {feat_dim}")

    if target_t is None:
        target_t = max(int(feat.shape[0]) for feat in feature_list)

    padded = []
    for feat in feature_list:
        t = int(feat.shape[0])
        if t < target_t:
            pad = torch.full((target_t - t, feat_dim), float(pad_value), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)
        elif t > target_t:
            feat = feat[:target_t]
        padded.append(feat)

    return torch.stack(padded, dim=0)

def extract_entries_features(entries: List[Dict[str, Any]], subset_name: str, processor, model, single_layer: int, ln_post, top_layer_index: int):
    """对 official train / official test entries 抽取 segment-level 特征。"""
    feature_list = []
    labels_list = []
    indices_list = []
    kept_entries = []

    for compact_idx, row in enumerate(tqdm(entries, desc=f"[{subset_name}] single{single_layer}", ncols=100)):
        wav_path = canonical_wav_path(row["wav_path"])
        mmse = float(row["target_mmse"])

        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"找不到音频文件：{wav_path}")

        if row.get("official_audio_id", None) is not None:
            audio_idx = int(row["official_audio_id"])
        elif row.get("audio_id", None) is not None:
            audio_idx = int(row["audio_id"])
        else:
            audio_idx = int(compact_idx)

        seg_feats, seg_idx = get_segment_features(
            audio_file=wav_path,
            audio_file_index=int(audio_idx),
            processor=processor,
            model=model,
            single_layer=int(single_layer),
            norm_single=NORM_FOR_SINGLE,
            ln_post=ln_post,
            top_layer_index=int(top_layer_index),
            drop_input_layer=DROP_INPUT_LAYER,
            segment_length=SEGMENT_LENGTH,
            sampling_rate=SAMPLE_RATE,
            overlap_rate=OVERLAP_RATE,
        )

        feature_list.extend(seg_feats)
        indices_list.extend(seg_idx)
        labels_list.extend([float(mmse)] * len(seg_feats))

        kept = copy.deepcopy(row)
        kept["audio_index_in_subset"] = int(audio_idx)
        kept["official_audio_id"] = int(audio_idx)
        kept["audio_id"] = int(audio_idx)
        kept["compact_audio_index_after_mmse_filter"] = int(compact_idx)
        kept["num_segments"] = int(len(seg_feats))
        kept_entries.append(kept)

    return feature_list, labels_list, indices_list, kept_entries

# 下游特征目录与元信息
def get_save_root(dataset_option: str, single_layer: int, fold_id: int, mask_offset: int) -> str:
    """构造当前 ssm_mmse.py 可读取的无 fold finetune 特征目录。"""
    return os.path.join(
        FEATURE_ROOT,
        dataset_option,
        get_feature_model_tag(mask_offset),
        get_feature_selection_tag(single_layer),
    )

def write_pitt_lru_audio_lookup_sidecar(
    save_root: str,
    file_tag: str,
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
) -> Optional[str]:
    """为 Pitt SCPA 特征写出与 Whisper_pitt_new(1).py 相同协议的："""
    if str(DATASET_OPTION) != "Pitt" or not bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return None

    def _row(row: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        audio_id = row.get("audio_index_in_subset", row.get("official_audio_id", row.get("audio_id", None)))
        if audio_id is None:
            raise RuntimeError(f"[Pitt sidecar] 缺少 audio_id: {row}")
        subject_id = row.get("subject_id")
        visit_index = row.get("visit_index")
        if subject_id is None or visit_index is None:
            subject_id, visit_index = _extract_pitt_subject_visit(Path(row["wav_path"]).stem)
        if subject_id is None or visit_index is None:
            raise RuntimeError(f"[Pitt sidecar] 无法解析 subject/visit: {row['wav_path']}")
        return {
            "audio_id": int(audio_id),
            "split": str(split_name),
            "wav_path": canonical_wav_path(row["wav_path"]),
            "wav_name": str(row.get("wav_name", Path(row["wav_path"]).name)),
            "subject_id": str(subject_id),
            "visit_index": int(visit_index),
            "group": str(row.get("group", "")),
            "target_mmse": float(row["target_mmse"]),
        }

    train_rows = [_row(row, "train") for row in train_manifest]
    test_rows = [_row(row, "test") for row in test_manifest]

    train_subjects = {r["subject_id"] for r in train_rows}
    test_subjects = {r["subject_id"] for r in test_rows}
    overlap = train_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"[Pitt sidecar] train/test subject overlap != 0: {sorted(overlap)}")

    train_ids = sorted({int(r["audio_id"]) for r in train_rows})
    test_ids = sorted({int(r["audio_id"]) for r in test_rows})
    if train_ids != list(range(len(train_rows))):
        raise RuntimeError(f"[Pitt sidecar] train audio_id 非连续 0..N-1: {train_ids[:10]} ...")
    if test_ids != list(range(len(test_rows))):
        raise RuntimeError(f"[Pitt sidecar] test audio_id 非连续 0..N-1: {test_ids[:10]} ...")

    payload = {
        "format_version": 3,
        "task_type": "mmse_regression",
        "dataset": "Pitt",
        "split_strategy": PITT_SPLIT_POLICY,
        "subject_disjoint": True,
        "train_test_subject_overlap": 0,
        "train_audio_ids": train_ids,
        "val_audio_ids": [],
        "train_entries": train_rows,
        "test_entries": test_rows,
    }

    sidecar_path = os.path.join(save_root, f"train_val_split_{file_tag}.json")
    save_json(payload, sidecar_path)
    print(f"[Pitt split] feature audio-id sidecar -> {sidecar_path}")
    return sidecar_path

def save_feature_bundle(
    save_root: str,
    single_layer: int,
    fold_id: int,
    mask_offset: int,
    fold_json_path: str,
    stage3_ckpt_path: str,
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
    missing_train: List[Dict[str, Any]],
    missing_test: List[Dict[str, Any]],
):
    """保存成当前 ssm_mmse.py 可读取的 official_train / official_test 无 fold 格式。"""
    ensure_dir(save_root)

    mask_offset = int(mask_offset)
    ft_model_name = get_ft_model_name(mask_offset)
    feature_model_tag = get_feature_model_tag(mask_offset)
    feature_selection_tag = get_feature_selection_tag(single_layer)

    layer_name = build_layer_name(
        single_layer=single_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage3_subdir = build_stage3_subdir(mask_offset)

    file_tag = f"{FEATURE_PREFIX}_single{int(single_layer)}_{OV_STR}_{SEC_STR}"

    paths = {
        "feature_train": os.path.join(save_root, f"feature_{file_tag}_train.pt"),
        "feature_test": os.path.join(save_root, f"feature_{file_tag}_test.pt"),
        "labels_train": os.path.join(save_root, f"labels1_{file_tag}_train.pt"),
        "labels_test": os.path.join(save_root, f"labels2_{file_tag}_test.pt"),
        "indices_train": os.path.join(save_root, f"indices_{file_tag}_train.pt"),
        "indices_test": os.path.join(save_root, f"indices_{file_tag}_test.pt"),
    }

    torch.save(train_features, paths["feature_train"])
    torch.save(test_features, paths["feature_test"])
    torch.save(train_labels, paths["labels_train"])
    torch.save(test_labels, paths["labels_test"])
    torch.save(train_indices, paths["indices_train"])
    torch.save(test_indices, paths["indices_test"])

    pitt_sidecar_path = write_pitt_lru_audio_lookup_sidecar(
        save_root=save_root,
        file_tag=file_tag,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
    )

    if SAVE_META:
        meta = {
            "dataset_option": DATASET_OPTION,
            "run_protocol": str(RUN_PROTOCOL_TAG),
            "fold_id": None,
            "fold_disabled": True,
            "pitt_subject_disjoint": bool(str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
            "pitt_split_policy": (PITT_SPLIT_POLICY if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_train_subjects": (PITT_EXPECTED_FINAL_TRAIN_SUBJECTS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_test_subjects": (PITT_EXPECTED_FINAL_TEST_SUBJECTS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_train_visits": (PITT_EXPECTED_FINAL_TRAIN_VISITS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_test_visits": (PITT_EXPECTED_FINAL_TEST_VISITS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_lru_audio_lookup_sidecar": pitt_sidecar_path,

            "single_layer": int(single_layer),
            "unfreeze_encoder_layers": int(UNFREEZE_ENCODER_LAYERS),
            "layer_name": str(layer_name),

            "mask_offset": int(mask_offset),
            "stage3_subdir": str(stage3_subdir),
            "feature_model_tag": str(feature_model_tag),
            "ft_model_name_for_ssm": str(ft_model_name),
            "feature_selection_tag_for_ssm": str(feature_selection_tag),
            "feature_file_tag_for_ssm": file_tag,

            "stage3_ckpt_path": str(stage3_ckpt_path),
            "whisper_base_model_name": str(WHISPER_BASE_MODEL_NAME),

            "sampling_rate": int(SAMPLE_RATE),
            "segment_sec": int(SEGMENT_SEC),
            "overlap_rate": float(OVERLAP_RATE),
            "norm_for_single": str(NORM_FOR_SINGLE),
            "drop_input_layer": bool(DROP_INPUT_LAYER),

            "train_num_wavs": int(len(train_manifest)),
            "test_num_wavs": int(len(test_manifest)),
            "train_num_segments": int(train_features.shape[0]),
            "test_num_segments": int(test_features.shape[0]),
            "train_feature_shape": list(train_features.shape),
            "test_feature_shape": list(test_features.shape),
            "missing_train": int(len(missing_train)),
            "missing_test": int(len(missing_test)),

            "feature_train_semantics": (
                "pitt_subject_disjoint_final_train"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_train_all_after_missing_mmse_filter"
            ),
            "feature_test_semantics": (
                "pitt_subject_disjoint_final_test_holdout"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_test_all_after_missing_mmse_filter"
            ),
            "audio_id_policy": (
                "local_audio_id_rebuilt_after_pitt_subject_disjoint_split"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "stable_official_audio_id_before_mmse_filter"
            ),
            "saved_files": paths,

            "ssm_recommended_config": {
                "FEATURE_SOURCE_MODES": ["finetune"],
                "FT_MODEL_NAME": str(ft_model_name),
                "FEATURE_MODEL_TAG": str(feature_model_tag),
                "FEATURE_SELECTION_TAG": str(feature_selection_tag),
                "FEATURE_FILE_TAG": file_tag,
                "SINGLE_LAYERS": [int(single_layer)],
                "UNFREE_LAYER_NUM": [int(UNFREEZE_ENCODER_LAYERS)],
                "STAGE3_MASK_INJECT_AFTER_LAYERS": [int(mask_offset)],
            },

            "PTAPT_METHOD": str(PTAPT_METHOD_TAG),
            "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
            "PATH_PARAM_SUFFIX": build_path_param_suffix(PATH_PARAM_TAGS),
            "STAGE1_SOURCE_PATH_PARAM_TAGS": list(STAGE1_SOURCE_PATH_PARAM_TAGS),
            "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(STAGE1_SOURCE_PATH_PARAM_TAGS),
            "PTAPT_KS": [int(k) for k in PTAPT_KS],
            "PTAPT_EXPECTED_NUM_CLUSTERS": [int(k) for k in PTAPT_EXPECTED_NUM_CLUSTERS],
            "PTAPT_STAGE3_LAMBDA_MMSE": float(PTAPT_STAGE3_LAMBDA_MMSE),
            "STAGE3_MASK_INJECT_AFTER_LAYERS": [int(x) for x in STAGE3_MASK_INJECT_AFTER_LAYERS],
        }

        save_json(meta, os.path.join(save_root, "feature_meta.json"))
        save_json({"train_manifest": train_manifest}, os.path.join(save_root, "train_manifest.json"))
        save_json({"test_manifest": test_manifest}, os.path.join(save_root, "test_manifest.json"))

        missing = {
            "missing_train": missing_train,
            "missing_test": missing_test,
        }
        save_json(missing, os.path.join(save_root, "missing_mmse_records.json"))

        save_text(str(stage3_ckpt_path), os.path.join(save_root, "stage3_ckpt_path.txt"))

        save_text(
            (
                "Pitt subject-disjoint protocol: final train/test are rebuilt by subject; no fold split json is used by ssm_mmse.py.\n"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_train_test protocol: no fold split json is used by ssm_mmse.py.\n"
            ),
            os.path.join(save_root, "protocol_note.txt"),
        )

    print(f"[SAVE] {save_root}")
    if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT:
        print("  semantics: train=Pitt subject-disjoint final train, test=Pitt subject-disjoint final test holdout")
    else:
        print("  semantics: train=official_train_all_after_missing_mmse_filter, test=official_test_all_after_missing_mmse_filter")
    for k, p in paths.items():
        print(f"  - {k}: {p}")

# 单次导出
def run_one_fold_layer(single_layer: int, fold_id: int, mask_offset: int) -> None:
    """单个 layer + mask_offset 的完整导出流程。"""
    mask_offset = int(mask_offset)
    fold_id = int(PROTOCOL_FOLD_ID)

    fold_json_path = ""

    save_root = get_save_root(DATASET_OPTION, single_layer, fold_id, mask_offset)

    layer_name = build_layer_name(
        single_layer=single_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage3_dir = os.path.join(
        build_run_root(
            dataset_option=DATASET_OPTION,
            layer_id=int(single_layer),
            fold_id=int(PROTOCOL_FOLD_ID),
        ),
        build_stage3_subdir(mask_offset),
    )

    stage3_subdir = build_stage3_subdir(mask_offset)
    ft_model_name = get_ft_model_name(mask_offset)
    feature_model_tag = get_feature_model_tag(mask_offset)

    print("\n" + "=" * 100)
    print(f"[RUN] DATASET_OPTION = {DATASET_OPTION}")
    print(f"[RUN] RUN_PROTOCOL   = {RUN_PROTOCOL_TAG}")
    print(f"[RUN] single_layer   = {single_layer}")
    print(f"[RUN] layer_name     = {layer_name}")
    print(f"[RUN] mask_offset    = {mask_offset}")
    print(f"[RUN] stage3_subdir  = {stage3_subdir}")
    print(f"[RUN] stage3_dir     = {stage3_dir}")
    print(f"[RUN] fold disabled  = True")
    print(f"[RUN] FT_MODEL_NAME  = {ft_model_name}")
    print(f"[RUN] FEATURE_MODEL  = {feature_model_tag}")
    print(f"[RUN] save_root      = {save_root}")
    print(
        "[RUN] extract target = Pitt subject-disjoint final train + final test"
        if DATASET_OPTION == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
        else "[RUN] extract target = official train + official test"
    )
    print("=" * 100)

    train_entries_raw, test_entries_raw = load_official_train_test_entries(DATASET_OPTION)
    mmse_maps = load_mmse_maps_for_dataset(DATASET_OPTION)

    train_entries, missing_train = attach_mmse_and_filter(
        train_entries_raw,
        DATASET_OPTION,
        mmse_maps,
        subset_name="train",
    )
    test_entries, missing_test = attach_mmse_and_filter(
        test_entries_raw,
        DATASET_OPTION,
        mmse_maps,
        subset_name="test",
    )

    if len(train_entries) == 0 or len(test_entries) == 0:
        raise ValueError(
            f"MMSE 过滤后 official train/test 为空："
            f"train={len(train_entries)}, test={len(test_entries)}"
        )

    stage3_ckpt_path = find_stage3_checkpoint(
        dataset_option=DATASET_OPTION,
        layer_id=int(single_layer),
        fold_id=int(PROTOCOL_FOLD_ID),
        mask_offset=int(mask_offset),
        override_ckpt=STAGE3_CKPT_OVERRIDE,
    )

    print(f"[CKPT] Stage3 checkpoint = {stage3_ckpt_path}")
    print(f"[DATA] official train wavs raw = {len(train_entries_raw)}")
    print(f"[DATA] official test wavs raw  = {len(test_entries_raw)}")
    print(f"[DATA] official train valid MMSE = {len(train_entries)}")
    print(f"[DATA] official test valid MMSE  = {len(test_entries)}")
    print(f"[DATA] missing train = {len(missing_train)}, missing test = {len(missing_test)}")

    processor = WhisperProcessor.from_pretrained(
        WHISPER_BASE_MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = WhisperModel.from_pretrained(
        WHISPER_BASE_MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    load_stage3_ckpt_into_whisper_model(model, stage3_ckpt_path)

    model.eval().to(DEVICE)
    ln_post = model.encoder.layer_norm
    top_layer_index = int(model.config.encoder_layers)

    with torch.no_grad():
        train_feat_list, train_labels_list, train_indices_list, train_manifest = extract_entries_features(
            entries=train_entries,
            subset_name="official_train",
            processor=processor,
            model=model,
            single_layer=int(single_layer),
            ln_post=ln_post,
            top_layer_index=top_layer_index,
        )

        test_feat_list, test_labels_list, test_indices_list, test_manifest = extract_entries_features(
            entries=test_entries,
            subset_name="official_test",
            processor=processor,
            model=model,
            single_layer=int(single_layer),
            ln_post=ln_post,
            top_layer_index=top_layer_index,
        )

    if len(train_feat_list) == 0 or len(test_feat_list) == 0:
        raise ValueError(
            f"抽特征后 official train/test segment 为空："
            f"train={len(train_feat_list)}, test={len(test_feat_list)}"
        )

    global_t = max(
        max(int(x.shape[0]) for x in train_feat_list),
        max(int(x.shape[0]) for x in test_feat_list),
    )
    print(f"[PAD] global target_t = {global_t}")

    train_features = stack_with_time_padding(train_feat_list, target_t=global_t)
    test_features = stack_with_time_padding(test_feat_list, target_t=global_t)

    train_labels = torch.tensor(train_labels_list, dtype=torch.float32)
    test_labels = torch.tensor(test_labels_list, dtype=torch.float32)

    train_indices = np.asarray(train_indices_list, dtype=np.int64)
    test_indices = np.asarray(test_indices_list, dtype=np.int64)

    print(f"[SHAPE] train_features = {tuple(train_features.shape)}")
    print(f"[SHAPE] test_features  = {tuple(test_features.shape)}")
    print(f"[SHAPE] train_labels   = {tuple(train_labels.shape)}")
    print(f"[SHAPE] test_labels    = {tuple(test_labels.shape)}")
    print(f"[SHAPE] train_indices  = {tuple(train_indices.shape)}")
    print(f"[SHAPE] test_indices   = {tuple(test_indices.shape)}")

    save_feature_bundle(
        save_root=save_root,
        single_layer=int(single_layer),
        fold_id=int(PROTOCOL_FOLD_ID),
        mask_offset=int(mask_offset),
        fold_json_path=fold_json_path,
        stage3_ckpt_path=stage3_ckpt_path,
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        test_labels=test_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        missing_train=missing_train,
        missing_test=missing_test,
    )

    print(f"[DONE] official_train_test single{single_layer} maskoff{mask_offset}")

# 批量运行入口
def main() -> None:
    """遍历 DATASET_OPTIONS、UNFREEZE_ENCODER_LAYERS_LIST、LAYERS_TO_RUN"""
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS 为空，至少需要一个数据集。")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST 为空，至少需要一个解冻层数。")

    print("[CONFIG]")
    print("  PROJECT_ROOT                    =", PROJECT_ROOT)
    print("  DATASET_OPTIONS                 =", DATASET_OPTIONS)
    print("  RUN_PROTOCOL_TAG                =", RUN_PROTOCOL_TAG)
    print("  MODEL_ROOT_TAG                  =", MODEL_ROOT_TAG)
    print("  UNFREEZE_ENCODER_LAYERS_LIST    =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  STAGE3_MASK_INJECT_AFTER_LAYERS =", STAGE3_MASK_INJECT_AFTER_LAYERS)
    print("  WHISPER_BASE_MODEL              =", WHISPER_BASE_MODEL_NAME)
    print("  LAYERS_TO_RUN                   =", LAYERS_TO_RUN)
    print("  SEGMENT_SEC                     =", SEGMENT_SEC)
    print("  OVERLAP_RATE                    =", OVERLAP_RATE)
    print("  NORM_FOR_SINGLE                 =", NORM_FOR_SINGLE)
    print("  DEVICE                          =", DEVICE)
    print("  fold disabled                   = True")
    print("  protocol subdir disabled        = True")

    completed_items = []

    for dataset_option in [str(x) for x in DATASET_OPTIONS]:
        for unfreeze_count in [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "=" * 100)
            print("[EXTRACT BATCH ITEM]")
            print("  DATASET_OPTION =", DATASET_OPTION)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("=" * 100)

            for layer_id in [int(x) for x in LAYERS_TO_RUN]:
                for mask_offset in [int(x) for x in STAGE3_MASK_INJECT_AFTER_LAYERS]:
                    run_one_fold_layer(
                        single_layer=layer_id,
                        fold_id=PROTOCOL_FOLD_ID,
                        mask_offset=mask_offset,
                    )
                    completed_items.append({
                        "dataset": DATASET_OPTION,
                        "unfreeze": int(UNFREEZE_ENCODER_LAYERS),
                        "layer": int(layer_id),
                        "mask_offset": int(mask_offset),
                        "feature_root": os.path.join(
                            FEATURE_ROOT,
                            DATASET_OPTION,
                            get_feature_model_tag(mask_offset),
                            get_feature_selection_tag(layer_id),
                        ),
                    })

    print("\n" + "=" * 100)
    print("[ALL DONE]")
    print("输出目录：")
    for item in completed_items:
        print(
            f"  - dataset={item['dataset']} "
            f"unfree={item['unfreeze']} "
            f"single={item['layer']} "
            f"maskoff={item['mask_offset']} "
            f"-> {item['feature_root']}"
        )

    print()
    print("ssm_mmse.py 建议设置：")
    print('  FEATURE_SOURCE_MODES = ["finetune"]')
    print(f"  SINGLE_LAYERS = {LAYERS_TO_RUN}")
    print(f"  UNFREE_LAYER_NUM = {UNFREEZE_ENCODER_LAYERS_LIST}")
    for mask_offset in [int(x) for x in STAGE3_MASK_INJECT_AFTER_LAYERS]:
        print(f'  # maskoff{mask_offset}')
        print(f'  FT_MODEL_NAME = "{get_ft_model_name(mask_offset)}"')
    print("=" * 100)

if __name__ == "__main__":
    main()
