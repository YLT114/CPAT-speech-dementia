# -*- coding: utf-8 -*-
"""CPAT Stage1：MMSE 回归与粗粒度 MMSE 分桶监督。
精简版保留原参数、随机调用顺序、计算与文件协议；仅清理不可达旧代码和冗余说明。
默认 Pitt 受试者互斥 final train，第 12 层，解冻层数依次 1–12，训练 10 轮并保存 last。
仅使用训练音频拟合目标标准化；final test 不参与上游训练。

conda activate whisper_ptapt_faiss 
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune/ft_ptapt_repeat/CPAT_stage1.py
"""

import os
import json
import math
import copy
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import soundfile as sf

# GPU 设置；保留各文件原值及环境变量生效顺序。
GPU_ID = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import WhisperConfig, WhisperModel, WhisperFeatureExtractor

# 工程、中间结果与特征根目录；保留原路径。
PROJECT_ROOT = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune"
MMSE_PTAPT_ROOT = os.path.join(PROJECT_ROOT, "ft_ptapt_repeat")

PROJECT_ROOT_MAIN = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT_MAIN, "feature_tensor")

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

# 批量数据集；默认仅 ["ADReSS-IS2020"]，其他数据集分支仍保留。
DATASET_OPTIONS = ["ADReSS-IS2020"]

DATASET_OPTION = DATASET_OPTIONS[0]

# 各阶段层号保持联动；默认第 12 层。
LAYERS_TO_RUN = [12]

STAGE1_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN

# 相对当前表示层，向前连续解冻的层数；保留遍历顺序。
UNFREEZE_ENCODER_LAYERS_LIST = [8]

UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

RESOLVED_UNFREEZE_ENCODER_LAYERS = []

# 协议标签；当前训练不划分 fold，test 仅用于最终导出。
RUN_PROTOCOL_TAG = "official_train_test"

# 旧配置/manifest 的兼容占位；不新增 fold 目录。
PROTOCOL_FOLD_ID = 1
FOLDS_TO_RUN = [PROTOCOL_FOLD_ID]

# 跨阶段模型根标签；各脚本必须一致。
MODEL_ROOT_TAG = "ptapt_lr5e-5_epo5"

# 读取 Stage1 checkpoint 的路径后缀；与实际 Stage1 输出对应。
STAGE1_SOURCE_PATH_PARAM_TAGS = []

# singleX_unfreeY 的附加路径标签；Stage2、Stage3、导出需一致。
PATH_PARAM_TAGS = []

STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE3_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE4_OUTPUT_LAYERS_TO_RUN = LAYERS_TO_RUN

# Whisper 基模；保持原 checkpoint 与加载配置。
WHISPER_MODEL_NAME = "/home/juanyu/whisper_lru_3/whisper_small"

STAGE1_REPR_POOL = "mean"        # "mean" / "scalar_mix"
TARGET_SEC = 30.0                # 与同门脚本对齐，统一 30s 视图

# Stage1 MMSE 分桶边界，左闭右开。
MMSE_BIN_EDGES = [0.0, 11.0, 16.0, 21.0, 26.0, 31.0]

# Stage1 回归/分桶损失、头结构及训练超参数。
LAMBDA_UTT = 1.0
LAMBDA_FRAME = 0.5
HEAD_HIDDEN = 128
HEAD_DROPOUT = 0.30
HUBER_DELTA = 1.0
LABEL_SMOOTHING = 0.05

LR_HEAD = 1e-4
LR_WHISPER = 1e-5
WEIGHT_DECAY = 5e-2
WARMUP_RATIO = 0.10
FLAT_RATIO = 0.20
MIN_LR_RATIO = 0.05

BATCH_SIZE = 8
NUM_WORKERS = 4

# 固定 epoch、无验证集选模，保存 last checkpoint。
STAGE1_USE_FOLD_VAL = False
STAGE1_MAX_EPOCHS = 10
STAGE1_CHECKPOINT_SELECTION = "last"

TRAINING_STEP = 2000

PRECISION = 16                   # 16 / 32
CHECK_VAL_EVERY_N_EPOCH = 1
EARLY_STOP_PATIENCE = 0
SAVE_TOP_K = 0
ACCUMULATE_GRAD_BATCHES = 1
GRAD_CLIP = 1.0

SEED = 42
ENABLE_CSV_LOGGER = True
RESUME_CHECKPOINT = None

SAVE_MANIFESTS = True

# 当前 no-val-last 流程要求 False；保留原错误检查。
ENABLE_STAGE_QUALITY_EVAL = False
QUALITY_REPORT_FILENAME = "quality_report.json"
QUALITY_SUMMARY_FILENAME = "quality_summary.csv"
QUALITY_DECISION_FILENAME = "quality_decision.txt"

os.environ.pop("PL_TRAINER_AMP_BACKEND", None)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
TARGET_30S_SAMPLES = int(TARGET_SEC * SAMPLE_RATE)

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
    """若目录不存在则创建。"""
    os.makedirs(path, exist_ok=True)

def natural_key(x: str):
    """自然排序 key：让 fold2 排在 fold10 前面，也让 S002 排在 S010 前面。"""
    s = Path(str(x)).name
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def save_json(obj: Dict[str, Any], path: str) -> None:
    """保存 json，自动创建父目录。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(text: str, path: str) -> None:
    """保存纯文本。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

def compute_valid_token_length(wav_len_samples: int) -> int:
    """把“真实有效 waveform 长度（样本点数）”映射为 Whisper encoder 的 token 长度。"""
    if wav_len_samples <= 0:
        return 1
    n_mel_valid = int(math.ceil(wav_len_samples / float(HOP_LENGTH)))
    n_mel_valid = min(MAX_MEL_FRAMES, max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)

def bin_mmse(mmse_value: float, bin_edges: List[float]) -> int:
    """把连续 MMSE 映射到粗粒度 bin。"""
    x = float(mmse_value)
    edges = [float(e) for e in bin_edges]
    if len(edges) < 2:
        raise ValueError("MMSE_BIN_EDGES 至少需要两个边界。")
    bid = int(np.digitize([x], edges[1:-1], right=False)[0])
    bid = max(0, min(len(edges) - 2, bid))
    return bid

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

# MMSE 元数据解析
def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """在候选列名中，找到 DataFrame 里真正存在的那一列。"""
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"未找到候选列 {candidates}，当前列有：{list(df.columns)}")

def _safe_to_float(x):
    """安全把文本 / Excel 单元格转成 float。"""
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

def _normalize_mmse_key(dataset_key: str, raw_id) -> Optional[str]:
    """统一不同数据集的 MMSE 索引键。"""
    if raw_id is None or pd.isna(raw_id):
        return None

    s = str(raw_id).strip()
    s = Path(s).stem

    dataset_key_norm = str(dataset_key).strip().lower()

    if dataset_key_norm == "pitt":
        import re
        m = re.search(r"\d+", s)
        if m is None:
            return None
        return str(int(m.group(0)))

    if dataset_key_norm == "process-2":
        import re
        s = re.sub(r"__(SFT|PFT|CTD)$", "", s, flags=re.IGNORECASE)
        return s.lower()

    return s.lower()

def _extract_pitt_subject_visit(raw_name: str) -> Tuple[Optional[str], Optional[int]]:
    """从 Pitt 文件名中提取 (subject_id, visit_index)。"""
    import re
    if raw_name is None:
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    subject_key = str(int(m.group(1)))
    visit_index = int(m.group(2))
    return subject_key, visit_index

def _pitt_visit_to_mmse_col(visit_index: int) -> str:
    """Pitt readme 对应关系："""
    visit_index = int(visit_index)
    if visit_index < 0:
        raise ValueError(f"非法 visit_index={visit_index}")
    if visit_index == 0:
        return "mms"
    return f"mmse{visit_index + 1}"

def _build_map_from_rows(df: pd.DataFrame, dataset_key: str, id_col: str, mmse_col: str) -> Dict[str, float]:
    """从 DataFrame 两列构建 {规范化ID: mmse}。"""
    out = {}
    for _, row in df.iterrows():
        key = _normalize_mmse_key(dataset_key, row[id_col])
        val = _safe_to_float(row[mmse_col])
        if key is None or val is None:
            continue
        out[key] = float(val)
    return out

def _read_adress_is2020_mmse(cfg: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """读取 ADReSS-IS2020 的 txt MMSE 元数据。"""
    def _read_txt(txt_path: str) -> Dict[str, float]:
        out = {}
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
                out[sid] = float(mmse_val)
        return out

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
    """读取 ADReSSo21 的 MMSE csv。"""
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
    """读取 Pitt 的 visit-level MMSE。"""
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
            import re
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
    """按数据集加载 MMSE 映射。"""
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"当前未配置数据集 {dataset_key} 的 MMSE 元数据路径。")

    cfg = MMSE_META_PATHS[dataset_key]
    if dataset_key == "ADReSS-IS2020":
        return _read_adress_is2020_mmse(cfg)
    if dataset_key == "ADReSSo21":
        return _read_adresso21_mmse(cfg)
    if dataset_key == "Pitt":
        return _read_pitt_mmse(cfg)
    if dataset_key == "PROCESS-2":
        return _read_process2_mmse(cfg)

    raise ValueError(f"暂未实现数据集 {dataset_key} 的 MMSE 解析。")

# 数据集音频目录。
DATASETS: Dict[str, Dict[str, str]] = {
    "PROCESS-2": {
        "train_cc": "/home/juanyu/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/juanyu/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/juanyu/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/juanyu/Downloads/PROCESS-2/test/cd",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "ADReSSo21": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "Pitt": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence",
        "test_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence",
    },
}

def canonical_wav_path(path: str) -> str:
    """规范化 wav 路径，避免相对路径 / 符号链接导致匹配不一致。"""
    return str(Path(str(path)).expanduser().resolve())

def _is_process2_ctd_wav_path(wav_path: str, dataset_key: str = None) -> bool:
    """PROCESS-2 专用 wav 过滤："""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None

def list_wav_files(audio_dir: str, dataset_key: str = None) -> List[str]:
    """读取 official 子目录中的 wav 文件，并按 natural_key 排序。"""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    audio_dir = str(audio_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"official audio dir 不存在：{audio_dir}")

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
            f"official audio dir 中没有符合条件的 wav 文件：{audio_dir}, "
            f"dataset_key={dataset_key}"
        )

    if str(dataset_key).strip().lower() == "process-2":
        print(f"[PROCESS-2][CTD] {audio_dir}: 只读取 CTD wav，文件数 = {len(files)}")

    return files

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

    if split_name not in {"train", "test"}:
        raise ValueError(f"split_name 必须是 train/test，当前={split_name}")
    if group_name not in {"cc", "cd"}:
        raise ValueError(f"group_name 必须是 cc/cd，当前={group_name}")

    entries: List[Dict[str, Any]] = []
    for wav_path in list_wav_files(audio_dir, dataset_key=dataset_key):
        wav_path = canonical_wav_path(wav_path)
        entries.append({
            "wav_path": wav_path,
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

def _save_pitt_subject_disjoint_audit(
    visit_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    missing_records: List[Dict[str, Any]],
) -> Dict[str, str]:
    """保存与 Whisper_pitt_new(1).py 同口径的 Pitt split 审计文件。"""
    split_root = os.path.join(FEATURE_ROOT_DIR, "Pitt", "subject_disjoint_split")
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
    all_mmse_map = mmse_maps["train_cc"]  # Pitt 四个 map 本来就是同一个 visit-level all_map。

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

    overlap = final_train_subjects & final_test_subjects
    if overlap:
        raise RuntimeError(f"[Pitt split] subject-disjoint 检查失败，overlap={sorted(overlap)}")

    final_records: Dict[str, List[Dict[str, Any]]] = {
        "train_cc": [], "train_cd": [], "test_cc": [], "test_cd": []
    }
    for rec in records:
        final_split = "test" if rec["subject_id"] in final_test_subjects else "train"
        final_subset = f"{final_split}_{rec['group']}"
        rec["final_split"] = final_split
        rec["final_subset"] = final_subset
        final_records[final_subset].append(rec)

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
        ["final_split", "subject_id", "visit_index", "wav_name"],
        kind="stable",
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
    audit_paths = _save_pitt_subject_disjoint_audit(visit_df, subject_df, missing_records)

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
    print("[Pitt subject-disjoint final split | SCPA Stage1]")
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

def load_official_train_entries(dataset_option: str) -> List[Dict[str, Any]]:
    """构造 official_train entries。"""
    if dataset_option not in DATASETS:
        raise ValueError(f"DATASETS 中没有配置 DATASET_OPTION={dataset_option}")

    if str(dataset_option) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        train_entries, _test_entries = _build_pitt_subject_disjoint_train_test_entries()
        return train_entries

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

    if str(dataset_option).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] official_train entries: "
            f"train_cc + train_cd = {len(train_entries)}"
        )

    if len(train_entries) == 0:
        raise ValueError(f"official_train entries 为空：dataset={dataset_option}")

    train_entries = sorted(train_entries, key=lambda x: natural_key(x.get("wav_path", "")))

    for i, row in enumerate(train_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    return train_entries

def enrich_entries_with_mmse(
    entries: List[Dict[str, Any]],
    dataset_option: str,
    mmse_maps: Dict[str, Dict[Any, float]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """给 fold entries 回填连续 MMSE。"""
    valid_entries: List[Dict[str, Any]] = []
    missing_entries: List[Dict[str, Any]] = []

    for row in entries:
        row = copy.deepcopy(row)
        wav_path = row.get("wav_path", None)
        group = str(row.get("group", "")).strip().lower()
        split_name = str(row.get("split", "")).strip().lower()

        if wav_path in [None, ""]:
            row["missing_reason"] = "missing_wav_path"
            missing_entries.append(row)
            continue

        if row.get("target_mmse", None) not in [None, ""]:
            try:
                row["target_mmse"] = float(row["target_mmse"])
                valid_entries.append(row)
                continue
            except Exception:
                pass

        wav_stem = Path(str(wav_path)).stem

        if dataset_option == "Pitt":
            subject_key, visit_index = _extract_pitt_subject_visit(wav_stem)
            if subject_key is None or visit_index is None:
                row["missing_reason"] = "cannot_parse_pitt_subject_visit"
                missing_entries.append(row)
                continue
            key = (subject_key, int(visit_index))
            split_bucket = "train_cc" if group == "cc" else "train_cd"
            mmse_map = mmse_maps.get(split_bucket, {})
            if key not in mmse_map:
                row["missing_reason"] = f"missing_{_pitt_visit_to_mmse_col(visit_index)}"
                missing_entries.append(row)
                continue
            row["target_mmse"] = float(mmse_map[key])
            valid_entries.append(row)
            continue

        key = _normalize_mmse_key(dataset_option, wav_stem)
        split_bucket = "train_cc" if group == "cc" else "train_cd"
        mmse_map = mmse_maps.get(split_bucket, {})
        if key not in mmse_map:
            row["missing_reason"] = f"missing_mmse_{split_name}_{group}"
            missing_entries.append(row)
            continue
        row["target_mmse"] = float(mmse_map[key])
        valid_entries.append(row)

    return valid_entries, missing_entries

# Stage1 数据集与模型
class MMSEFoldDataset(data.Dataset):
    """Stage1-MMSE 专用数据集。"""
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        target_mean: float,
        target_std: float,
        mmse_bin_edges: List[float],
        returnname: bool = False,
        sample_rate: int = 16000,
        target_sec: float = 30.0,
    ):
        self.entries = [copy.deepcopy(x) for x in entries]
        self.returnname = bool(returnname)
        self.sample_rate = int(sample_rate)
        self.target_seqlen = int(float(target_sec) * self.sample_rate)
        self.target_mean = float(target_mean)
        self.target_std = float(max(1e-6, target_std))
        self.mmse_bin_edges = [float(x) for x in mmse_bin_edges]

        if len(self.entries) == 0:
            raise ValueError("MMSEFoldDataset 收到空 entries。")

        self.mmse_values = [float(x["target_mmse"]) for x in self.entries]
        self.bin_ids = [bin_mmse(x, self.mmse_bin_edges) for x in self.mmse_values]
        self.num_bins = len(self.mmse_bin_edges) - 1

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        row = self.entries[index]
        wav_path = row["wav_path"]
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        wav = wav.astype(np.float32)
        wav_len = int(wav.shape[0])

        mmse_raw = float(row["target_mmse"])
        mmse_z = (mmse_raw - self.target_mean) / self.target_std
        mmse_bin = int(bin_mmse(mmse_raw, self.mmse_bin_edges))
        name = Path(wav_path).name

        if not self.returnname:
            return wav, float(mmse_z), float(mmse_raw), int(mmse_bin), int(wav_len)
        return wav, float(mmse_z), float(mmse_raw), int(mmse_bin), int(wav_len), name

    @staticmethod
    def _repeat_or_center_crop_to_length(x: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
        """与同门 Stage1 的处理方式保持一致："""
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        L = int(x.shape[0])
        if L <= 0:
            return np.zeros((target_len,), dtype=np.float32), 0

        if L >= target_len:
            start = (L - target_len) // 2
            seg = x[start:start + target_len]
            return seg.astype(np.float32, copy=False), target_len

        n_rep = target_len // L
        rem = target_len - n_rep * L
        out = np.tile(x, n_rep)
        if rem > 0:
            out = np.concatenate([out, x[:rem]], axis=0)
        if out.shape[0] != target_len:
            out = out[:target_len]
        return out.astype(np.float32, copy=False), L

    def repeat30s_collate(self, batch):
        """collate_fn："""
        target_len = self.target_seqlen

        def process(x):
            if self.returnname:
                wav, mmse_z, mmse_raw, mmse_bin, wav_len, name = x
            else:
                wav, mmse_z, mmse_raw, mmse_bin, wav_len = x

            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            wav_len = int(wav_len)
            wav_len = max(0, min(wav_len, wav.shape[0]))
            base = wav[:wav_len] if wav_len > 0 else wav

            wav_30, valid_len = self._repeat_or_center_crop_to_length(base, target_len=target_len)

            if self.returnname:
                return wav_30, float(mmse_z), float(mmse_raw), int(mmse_bin), int(valid_len), name
            return wav_30, float(mmse_z), float(mmse_raw), int(mmse_bin), int(valid_len)

        batch = list(map(process, batch))
        return default_collate(batch)

class LearnableScalarMix(nn.Module):
    """可学习的多层融合。"""
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = int(n_layers)
        self.w = nn.Parameter(torch.zeros(self.n_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        if stack.size(0) != self.n_layers:
            raise RuntimeError(f"ScalarMix 期望 L={self.n_layers}，当前得到 {stack.size(0)}。")
        weights = torch.softmax(self.w, dim=0).view(-1, 1, 1, 1)
        mixed = torch.sum(weights * stack, dim=0)
        return self.gamma * mixed

class WhisperEncoderWrapper(nn.Module):
    """Whisper encoder 封装。"""

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        unfreeze_encoder_layers: Optional[Sequence[int]] = None,
        scalar_mix_n_layers: Optional[int] = None,
    ):
        """Whisper encoder wrapper。"""
        super().__init__()
        self.model = WhisperModel.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.hidden_size = self.model.config.d_model

        if unfreeze_encoder_layers is None:
            unfreeze_encoder_layers = []
        self.unfreeze_encoder_layers = sorted({
            int(x) for x in list(unfreeze_encoder_layers)
            if int(x) > 0
        })

        self.scalar_mix = LearnableScalarMix(int(scalar_mix_n_layers)) if scalar_mix_n_layers is not None else None

        for p in self.model.parameters():
            p.requires_grad = False

        encoder_layers = self.model.encoder.layers
        total_layers = len(encoder_layers)

        valid_layers = []
        for layer_idx_1based in self.unfreeze_encoder_layers:
            if not (1 <= int(layer_idx_1based) <= total_layers):
                continue

            layer = encoder_layers[int(layer_idx_1based) - 1]
            for p in layer.parameters():
                p.requires_grad = True
            valid_layers.append(int(layer_idx_1based))

        self.unfreeze_encoder_layers = valid_layers

        if len(self.unfreeze_encoder_layers) > 0:
            for p in self.model.encoder.layer_norm.parameters():
                p.requires_grad = True

        print(f"[UNFREEZE] Stage1 Whisper encoder layers = {self.unfreeze_encoder_layers}")

    def trainable_params(self) -> List[torch.Tensor]:
        """返回当前所有 requires_grad=True 的参数。"""
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def _parse_repr_layers(repr_layers: Union[str, List[int], None]) -> Optional[List[int]]:
        """解析 repr_layers："""
        if repr_layers is None:
            return None
        if isinstance(repr_layers, str):
            s = repr_layers.strip()
            if s == "":
                return None
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return [int(p) for p in parts]
        if isinstance(repr_layers, (list, tuple)):
            return [int(x) for x in repr_layers]
        return None

    def _build_attn_mask(self, input_features: torch.Tensor, valid_wav_lens: torch.Tensor) -> Optional[torch.Tensor]:
        """构造 token 级 attention_mask：[B, tok_T]，True=可见，False=无效位置。"""
        if valid_wav_lens is None:
            return None
        if not torch.is_tensor(valid_wav_lens):
            valid_wav_lens = torch.tensor(valid_wav_lens)
        vwl = valid_wav_lens.detach().cpu().long().view(-1)
        B = input_features.size(0)
        if vwl.numel() != B:
            raise RuntimeError(f"valid_wav_lens 数量与 batch 不匹配：expect={B}, got={vwl.numel()}")

        mel_T = int(input_features.size(-1))
        tok_T = (mel_T + 1) // 2

        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(vwl[i].item()))
            valid_Ts.append(min(vt, tok_T))
        valid_Ts = torch.tensor(valid_Ts, device=input_features.device, dtype=torch.long)

        t = torch.arange(tok_T, device=input_features.device).view(1, tok_T).expand(B, tok_T)
        return (t < valid_Ts.view(B, 1)).bool()

    def forward(
        self,
        wav: torch.Tensor,
        sampling_rate: int = 16000,
        repr_layers: Union[str, List[int], None] = None,
        repr_pool: str = "mean",
        valid_wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """输入 waveform，输出 [B, T, C] 的 token 表征。"""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav_cpu = wav.detach().cpu().numpy()
        target_len = int(self.feature_extractor.n_samples)

        feats = self.feature_extractor(
            wav_cpu,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=target_len,
        )
        device = next(self.model.parameters()).device
        input_features = feats.input_features.to(device)

        attn_mask = None
        if valid_wav_lens is not None:
            attn_mask = self._build_attn_mask(input_features, valid_wav_lens)

        layers = self._parse_repr_layers(repr_layers)

        if layers is None:
            try:
                encoder_out = self.model.encoder(
                    input_features=input_features,
                    attention_mask=attn_mask,
                    return_dict=True,
                    output_hidden_states=False,
                )
            except TypeError:
                encoder_out = self.model.encoder(
                    input_features=input_features,
                    return_dict=True,
                    output_hidden_states=False,
                )
            return encoder_out.last_hidden_state

        try:
            encoder_out = self.model.encoder(
                input_features=input_features,
                attention_mask=attn_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        except TypeError:
            encoder_out = self.model.encoder(
                input_features=input_features,
                return_dict=True,
                output_hidden_states=True,
            )

        hs = encoder_out.hidden_states
        selected = []
        for l in layers:
            if l < 1 or l >= len(hs):
                raise ValueError(f"repr_layers 中层号非法：layer={l}, hidden_states_len={len(hs)}")
            selected.append(hs[l])

        stack = torch.stack(selected, dim=0)

        if repr_pool == "mean":
            return stack.mean(dim=0)

        if repr_pool == "scalar_mix":
            if self.scalar_mix is None:
                raise RuntimeError("repr_pool='scalar_mix'，但 scalar_mix 尚未初始化。")
            if self.scalar_mix.w.device != stack.device:
                self.scalar_mix = self.scalar_mix.to(stack.device)
            return self.scalar_mix(stack)

        raise ValueError(f"未知 repr_pool={repr_pool}")

class LinearHead(nn.Module):
    """通用 MLP 头："""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class AttentiveStatsPool(nn.Module):
    """带 mask 的 attentive mean + std pooling。"""
    def __init__(self, in_dim: int, attn_hidden: int = 128, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.attn(x).squeeze(-1)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            a = a.masked_fill(~mask, -1e4)

        a = F.softmax(a, dim=1)

        if mask is not None:
            a = a * mask.float()
            a = a / a.sum(dim=1, keepdim=True).clamp_min(self.eps)

        mean = torch.sum(x * a.unsqueeze(-1), dim=1)
        var = torch.sum(((x - mean.unsqueeze(1)) ** 2) * a.unsqueeze(-1), dim=1)
        std = torch.sqrt(var.clamp_min(self.eps))
        return torch.cat([mean, std], dim=-1)

class WhisperStage1MMSE(pl.LightningModule):
    """MMSE-aware Stage1。"""
    def __init__(
        self,
        train_entries: List[Dict[str, Any]],
        val_entries: List[Dict[str, Any]],
        saving_path: str,
        run_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_entries", "val_entries"])

        self.saving_path = str(saving_path)
        self.run_config = copy.deepcopy(run_config)

        self.batch_size = int(run_config["BATCH_SIZE"])
        self.num_workers = int(run_config["NUM_WORKERS"])
        self.maxstep = int(run_config["TRAINING_STEP"])

        self.lambda_utt = float(run_config["LAMBDA_UTT"])
        self.lambda_frame = float(run_config["LAMBDA_FRAME"])
        self.huber_delta = float(run_config["HUBER_DELTA"])

        self.repr_layers = run_config["CURRENT_STAGE1_REPR_LAYERS"]
        self.repr_pool = str(run_config["STAGE1_REPR_POOL"])
        self.mmse_bin_edges = [float(x) for x in run_config["MMSE_BIN_EDGES"]]

        train_mmse = np.asarray([float(x["target_mmse"]) for x in train_entries], dtype=np.float32)
        self.target_mean = float(np.mean(train_mmse))
        self.target_std = float(max(1e-6, np.std(train_mmse)))

        self.train_dataset = MMSEFoldDataset(
            entries=train_entries,
            target_mean=self.target_mean,
            target_std=self.target_std,
            mmse_bin_edges=self.mmse_bin_edges,
            returnname=True,
            sample_rate=SAMPLE_RATE,
            target_sec=float(run_config["TARGET_SEC"]),
        )
        self.val_dataset = MMSEFoldDataset(
            entries=val_entries,
            target_mean=self.target_mean,
            target_std=self.target_std,
            mmse_bin_edges=self.mmse_bin_edges,
            returnname=True,
            sample_rate=SAMPLE_RATE,
            target_sec=float(run_config["TARGET_SEC"]),
        )

        self.num_bins = int(self.train_dataset.num_bins)

        bin_counter = {}
        for bid in self.train_dataset.bin_ids:
            bin_counter[bid] = bin_counter.get(bid, 0) + 1
        weight_counts = [bin_counter.get(i, 1) for i in range(self.num_bins)]
        weights = torch.tensor(weight_counts, dtype=torch.float)
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        self.register_buffer("frame_class_weights", weights)

        try:
            self.frame_criterion = nn.CrossEntropyLoss(
                weight=self.frame_class_weights,
                label_smoothing=float(run_config["LABEL_SMOOTHING"]),
            )
        except TypeError:
            self.frame_criterion = nn.CrossEntropyLoss(weight=self.frame_class_weights)

        try:
            self.utt_criterion = nn.HuberLoss(delta=self.huber_delta)
        except Exception:
            self.utt_criterion = nn.SmoothL1Loss(beta=self.huber_delta)

        parsed_layers = WhisperEncoderWrapper._parse_repr_layers(self.repr_layers)
        scalar_mix_n_layers = None if parsed_layers is None else len(parsed_layers)

        self.whisper = WhisperEncoderWrapper(
            model_name=str(run_config["WHISPER_MODEL_NAME"]),
            unfreeze_encoder_layers=run_config["STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS"],
            scalar_mix_n_layers=scalar_mix_n_layers if self.repr_pool == "scalar_mix" else None,
        )
        d_model = int(self.whisper.hidden_size)

        self.frame_head = LinearHead(
            in_dim=d_model,
            hidden_dim=int(run_config["HEAD_HIDDEN"]),
            out_dim=self.num_bins,
            dropout=float(run_config["HEAD_DROPOUT"]),
        )
        self.utt_pool = AttentiveStatsPool(d_model, attn_hidden=int(run_config["HEAD_HIDDEN"]))
        self.utt_head = LinearHead(
            in_dim=d_model * 2,
            hidden_dim=int(run_config["HEAD_HIDDEN"]),
            out_dim=1,
            dropout=float(run_config["HEAD_DROPOUT"]),
        )

    @staticmethod
    def _build_token_mask(valid_wav_lens: torch.Tensor, T: int, device) -> torch.Tensor:
        """根据原始有效波形长度，构造 token-level 有效位置 mask。"""
        vwl = valid_wav_lens.detach().to("cpu").long().view(-1)
        B = vwl.numel()
        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(vwl[i].item()))
            valid_Ts.append(min(vt, T))
        valid_Ts = torch.tensor(valid_Ts, device=device, dtype=torch.long)
        t = torch.arange(T, device=device).view(1, T).expand(B, T)
        return (t < valid_Ts.view(B, 1))

    def forward(self, wav_batch: torch.Tensor, valid_len: Optional[torch.Tensor] = None):
        """前向输出："""
        reps = self.whisper(
            wav_batch,
            sampling_rate=SAMPLE_RATE,
            repr_layers=self.repr_layers,
            repr_pool=self.repr_pool,
            valid_wav_lens=valid_len,
        )

        token_mask = None
        if valid_len is not None:
            token_mask = self._build_token_mask(valid_len.to(reps.device), reps.size(1), reps.device)

        utt_repr = self.utt_pool(reps, mask=token_mask)
        utt_pred_z = self.utt_head(utt_repr).squeeze(-1)
        frame_logits = self.frame_head(reps)
        return frame_logits, utt_pred_z, token_mask

    def train_dataloader(self):
        """训练集 DataLoader。"""
        nw = int(self.num_workers)
        pin = torch.cuda.is_available()
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            drop_last=True,
            shuffle=True,
            collate_fn=self.train_dataset.repeat30s_collate,
            persistent_workers=(nw > 0),
            pin_memory=pin,
        )

    def val_dataloader(self):
        """验证集 DataLoader。"""
        nw = int(self.num_workers)
        pin = torch.cuda.is_available()
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            drop_last=False,
            shuffle=False,
            collate_fn=self.val_dataset.repeat30s_collate,
            persistent_workers=(nw > 0),
            pin_memory=pin,
        )

    def configure_optimizers(self):
        """优化器与学习率调度："""
        head_params: List[torch.Tensor] = []
        head_params += list(self.frame_head.parameters())
        head_params += list(self.utt_head.parameters())
        head_params += list(self.utt_pool.parameters())

        all_whisper_trainables = list(self.whisper.trainable_params())

        mix_params = []
        if getattr(self.whisper, "scalar_mix", None) is not None:
            mix_params = list(self.whisper.scalar_mix.parameters())

        mix_ids = {id(p) for p in mix_params}
        whisper_params = [p for p in all_whisper_trainables if id(p) not in mix_ids]

        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": float(self.run_config["LR_HEAD"])},
                {"params": whisper_params, "lr": float(self.run_config["LR_WHISPER"])},
                {"params": mix_params, "lr": float(self.run_config["LR_HEAD"])},
            ],
            weight_decay=float(self.run_config["WEIGHT_DECAY"]),
        )

        num_training_steps = int(self.maxstep)
        num_warmup_steps = int(float(self.run_config["WARMUP_RATIO"]) * num_training_steps)
        num_flat_steps = int(float(self.run_config["FLAT_RATIO"]) * num_training_steps)
        min_lr_ratio = float(self.run_config["MIN_LR_RATIO"])

        def lambda_lr(current_step: int):
            if num_training_steps <= 0:
                return 1.0
            if current_step < num_warmup_steps:
                return float(current_step + 1) / float(max(1, num_warmup_steps))
            if current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            decay_steps = max(1, num_training_steps - (num_warmup_steps + num_flat_steps))
            s = min(decay_steps, max(0, current_step - (num_warmup_steps + num_flat_steps)))
            progress = float(s) / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = {"scheduler": optim.lr_scheduler.LambdaLR(optimizer, lambda_lr), "interval": "step"}
        return [optimizer], [scheduler]

    @staticmethod
    def _get_lrs_from_optimizer(opt):
        """从 optimizer 的 param_groups 中解析当前学习率，方便日志记录。"""
        if opt is None or len(opt.param_groups) < 3:
            return None, None, None
        lr_head = float(opt.param_groups[0]["lr"])
        lr_whisper = float(opt.param_groups[1]["lr"])
        lr_mix = float(opt.param_groups[2]["lr"])
        return lr_head, lr_whisper, lr_mix

    def _frame_loss_on_valid_tokens(
        self,
        frame_logits: torch.Tensor,
        mmse_bin: torch.Tensor,
        valid_wav_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """只在有效 token 上计算 frame 分类损失和精度。"""
        B, T, C = frame_logits.shape
        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(valid_wav_lens[i].item()))
            valid_Ts.append(min(vt, T))
        valid_Ts = torch.tensor(valid_Ts, device=frame_logits.device, dtype=torch.long)

        t_ids = torch.arange(T, device=frame_logits.device).view(1, T).expand(B, T)
        mask = t_ids < valid_Ts.view(B, 1)

        logits_flat = frame_logits.reshape(-1, C)
        labels_flat = mmse_bin.unsqueeze(1).expand(-1, T).reshape(-1)
        mask_flat = mask.reshape(-1)

        if mask_flat.sum().item() == 0:
            loss_frame = logits_flat.sum() * 0.0
            acc = loss_frame.detach() * 0.0
            return loss_frame, acc

        loss_frame = self.frame_criterion(logits_flat[mask_flat], labels_flat[mask_flat])
        pred = torch.argmax(logits_flat[mask_flat], dim=-1)
        acc = (pred == labels_flat[mask_flat]).float().mean()
        return loss_frame, acc

    def training_step(self, batch, batch_idx):
        """训练步："""
        wav_30, mmse_z, mmse_raw, mmse_bin, valid_len, _names = batch

        frame_logits, utt_pred_z, _token_mask = self(wav_30, valid_len)

        mmse_z = mmse_z.to(utt_pred_z.device).float()
        mmse_bin = mmse_bin.to(frame_logits.device).long()
        valid_len = valid_len.to(frame_logits.device).long()

        loss_utt = self.utt_criterion(utt_pred_z, mmse_z)
        loss_frame, acc_frame = self._frame_loss_on_valid_tokens(frame_logits, mmse_bin, valid_len)
        loss = self.lambda_utt * loss_utt + self.lambda_frame * loss_frame

        utt_mae_z = torch.mean(torch.abs(utt_pred_z.detach() - mmse_z.detach()))

        lr_head, lr_whisper, lr_mix = None, None, None
        if self.trainer is not None and hasattr(self.trainer, "optimizers") and len(self.trainer.optimizers) > 0:
            lr_head, lr_whisper, lr_mix = self._get_lrs_from_optimizer(self.trainer.optimizers[0])

        log_dict = {
            "train_loss": loss,
            "train_loss_utt": loss_utt.detach(),
            "train_loss_frame": loss_frame.detach(),
            "train_frame_acc": acc_frame.detach(),
            "train_utt_mae_z": utt_mae_z.detach(),
        }
        if lr_head is not None:
            log_dict["lr_head"] = lr_head
        if lr_whisper is not None:
            log_dict["lr_whisper"] = lr_whisper
        if lr_mix is not None:
            log_dict["lr_mix"] = lr_mix

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """验证步："""
        wav_30, mmse_z, mmse_raw, mmse_bin, valid_len, _names = batch

        frame_logits, utt_pred_z, _token_mask = self(wav_30, valid_len)

        mmse_z = mmse_z.to(utt_pred_z.device).float()
        mmse_raw = mmse_raw.to(utt_pred_z.device).float()
        mmse_bin = mmse_bin.to(frame_logits.device).long()
        valid_len = valid_len.to(frame_logits.device).long()

        loss_utt = self.utt_criterion(utt_pred_z, mmse_z)
        loss_frame, acc_frame = self._frame_loss_on_valid_tokens(frame_logits, mmse_bin, valid_len)
        loss = self.lambda_utt * loss_utt + self.lambda_frame * loss_frame

        utt_pred_raw = utt_pred_z * self.target_std + self.target_mean
        utt_mae_raw = torch.mean(torch.abs(utt_pred_raw.detach() - mmse_raw.detach()))

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_loss_utt": loss_utt.detach(),
                "valid_loss_frame": loss_frame.detach(),
                "valid_frame_acc": acc_frame.detach(),
                "valid_utt_mae_raw": utt_mae_raw.detach(),
            },
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

def get_whisper_encoder_layer_count(whisper_model_name_or_path: str) -> int:
    """读取 Whisper config，返回 encoder 总层数。"""
    cfg = WhisperConfig.from_pretrained(whisper_model_name_or_path)
    num_layers = getattr(cfg, "encoder_layers", None)
    if num_layers is None:
        raise AttributeError("WhisperConfig 中缺少 encoder_layers，无法自动解析层数。")
    return int(num_layers)

def resolve_unfreeze_layers_for_single_layer(
    single_layer: int,
    total_layers: int,
    unfreeze_layer_count: int,
) -> List[int]:
    """根据当前 single_layer 和 UNFREEZE_ENCODER_LAYERS，解析实际解冻哪些 encoder 层。"""
    single_layer = int(single_layer)
    total_layers = int(total_layers)
    unfreeze_layer_count = int(unfreeze_layer_count)

    if not (1 <= single_layer <= total_layers):
        raise ValueError(f"single_layer={single_layer} 超出范围 [1, {total_layers}]")
    if unfreeze_layer_count <= 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS 必须 >= 1")

    start_layer = max(1, single_layer - unfreeze_layer_count + 1)
    end_layer = min(single_layer, total_layers)

    resolved = list(range(start_layer, end_layer + 1))
    if len(resolved) == 0:
        raise ValueError(
            f"single_layer={single_layer}, "
            f"unfreeze_layer_count={unfreeze_layer_count} 无法解析有效解冻层。"
        )

    return resolved

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

# Stage1 配置、训练与入口
def build_run_config(dataset_option: str, layer_id: int, fold_id: int = PROTOCOL_FOLD_ID) -> Dict[str, Any]:
    """构造当前 Stage1 运行配置。"""
    layer_id = int(layer_id)
    fold_id = int(PROTOCOL_FOLD_ID)

    total_layers = get_whisper_encoder_layer_count(WHISPER_MODEL_NAME)

    resolved_unfreeze_layers = resolve_unfreeze_layers_for_single_layer(
        single_layer=layer_id,
        total_layers=total_layers,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
    )

    layer_name = build_layer_name(
        single_layer=layer_id,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    protocol_tag = str(RUN_PROTOCOL_TAG)

    run_root = os.path.join(
        MMSE_PTAPT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        layer_name,
    )

    stage1_save_dir = os.path.join(run_root, "stage1_mmse")
    meta_save_dir = os.path.join(run_root, "meta")

    current_stage1_repr_layers = [layer_id]

    legacy_train_layer_start = min(resolved_unfreeze_layers)
    legacy_train_layer_end = max(resolved_unfreeze_layers)

    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "MMSE_PTAPT_ROOT": MMSE_PTAPT_ROOT,
        "DATASET_OPTION": dataset_option,
        "LAYER_ID": int(layer_id),

        "FOLD_ID": int(fold_id),
        "FOLD_TAG": "not_used",
        "FOLD_JSON_PATH": "",
        "RUN_PROTOCOL_TAG": protocol_tag,

        "LAYER_TAG": layer_name,
        "LAYER_NAME": layer_name,
        "PATH_PARAM_TAGS": list(globals().get("PATH_PARAM_TAGS", [])),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(),
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(STAGE1_SOURCE_PATH_PARAM_TAGS),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(STAGE1_SOURCE_PATH_PARAM_TAGS),

        "MODEL_ROOT_TAG": MODEL_ROOT_TAG,
        "RUN_ROOT": run_root,
        "STAGE1_SAVE_DIR": stage1_save_dir,
        "META_SAVE_DIR": meta_save_dir,

        "CURRENT_STAGE1_REPR_LAYERS": current_stage1_repr_layers,
        "STAGE1_REPR_POOL": STAGE1_REPR_POOL,

        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),
        "STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS": list(resolved_unfreeze_layers),
        "TOTAL_ENCODER_LAYERS": int(total_layers),

        "STAGE1_WHISPER_TRAIN_LAYER_START": int(legacy_train_layer_start),
        "STAGE1_WHISPER_TRAIN_LAYER_END": int(legacy_train_layer_end),
        "STAGE1_WHISPER_TRAIN_LAST_N": None,

        "WHISPER_MODEL_NAME": WHISPER_MODEL_NAME,
        "TARGET_SEC": TARGET_SEC,
        "MMSE_BIN_EDGES": MMSE_BIN_EDGES,
        "LAMBDA_UTT": LAMBDA_UTT,
        "LAMBDA_FRAME": LAMBDA_FRAME,
        "HUBER_DELTA": HUBER_DELTA,
        "HEAD_HIDDEN": HEAD_HIDDEN,
        "HEAD_DROPOUT": HEAD_DROPOUT,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "LR_HEAD": LR_HEAD,
        "LR_WHISPER": LR_WHISPER,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO,
        "FLAT_RATIO": FLAT_RATIO,
        "MIN_LR_RATIO": MIN_LR_RATIO,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,

        "STAGE1_USE_FOLD_VAL": False,
        "STAGE1_MAX_EPOCHS": int(STAGE1_MAX_EPOCHS),
        "STAGE1_CHECKPOINT_SELECTION": "last",
        "TRAINING_STEP": TRAINING_STEP,
        "PRECISION": PRECISION,
        "CHECK_VAL_EVERY_N_EPOCH": CHECK_VAL_EVERY_N_EPOCH,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "SAVE_TOP_K": SAVE_TOP_K,
        "ACCUMULATE_GRAD_BATCHES": ACCUMULATE_GRAD_BATCHES,
        "GRAD_CLIP": GRAD_CLIP,
        "SEED": SEED,
        "ENABLE_CSV_LOGGER": ENABLE_CSV_LOGGER,
        "RESUME_CHECKPOINT": RESUME_CHECKPOINT,
        "SAVE_MANIFESTS": SAVE_MANIFESTS,

        "STAGE1_REPR_LAYERS_TO_RUN": STAGE1_REPR_LAYERS_TO_RUN,
        "STAGE2_REP_LAYERS_TO_RUN": STAGE2_REP_LAYERS_TO_RUN,
        "STAGE3_REPR_LAYERS_TO_RUN": STAGE3_REPR_LAYERS_TO_RUN,
        "STAGE4_OUTPUT_LAYERS_TO_RUN": STAGE4_OUTPUT_LAYERS_TO_RUN,
    }

def build_trainer(stage1_save_dir: str, run_config: Dict[str, Any]) -> Tuple[Trainer, Optional[CSVLogger]]:
    """Stage1 no-val-last Trainer。"""
    ensure_dir(stage1_save_dir)

    callbacks = [
        ModelCheckpoint(
            dirpath=stage1_save_dir,
            filename="whisper-s1-mmse-last",
            save_top_k=0,
            save_last=True,
            monitor=None,
            verbose=True,
        )
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    pl_precision = "16-mixed" if (int(run_config["PRECISION"]) == 16 and torch.cuda.is_available()) else "32-true"

    logger = False
    enable_progress_bar = True
    if bool(run_config["ENABLE_CSV_LOGGER"]):
        logger = CSVLogger(save_dir=stage1_save_dir, name="csv_logs")
        enable_progress_bar = False

    trainer = Trainer(
        precision=pl_precision,
        callbacks=callbacks,
        max_epochs=int(run_config.get("STAGE1_MAX_EPOCHS", 15)),
        max_steps=-1,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=10,
        accumulate_grad_batches=int(run_config["ACCUMULATE_GRAD_BATCHES"]),
        gradient_clip_val=float(run_config["GRAD_CLIP"]),
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    return trainer, logger

def save_stage1_manifests(
    stage1_save_dir: str,
    meta_save_dir: str,
    run_config: Dict[str, Any],
    fold_meta: Dict[str, Any],
    train_entries: List[Dict[str, Any]],
    val_entries: List[Dict[str, Any]],
    missing_train: List[Dict[str, Any]],
    missing_val: List[Dict[str, Any]],
) -> None:
    """保存本次 Stage1 运行的重要清单与配置，方便后续核对。"""
    ensure_dir(stage1_save_dir)
    ensure_dir(meta_save_dir)

    save_json(run_config, os.path.join(stage1_save_dir, "config.json"))
    save_text(
        "official_train_test protocol: no fold json is used. Stage1 uses official_train only.\n",
        os.path.join(meta_save_dir, "source_protocol.txt"),
    )
    fold_summary = {
        "dataset_option": run_config["DATASET_OPTION"],
        "layer_id": run_config["LAYER_ID"],
        "layer_name": run_config["LAYER_NAME"],
        "fold_id": run_config["FOLD_ID"],
        "run_protocol": run_config.get("RUN_PROTOCOL_TAG", ""),
        "fold_disabled": True,
        "num_train_valid_entries": len(train_entries),
        "num_val_valid_entries": len(val_entries),
        "num_train_missing_mmse": len(missing_train),
        "num_val_missing_mmse": len(missing_val),
        "repr_layers": run_config["CURRENT_STAGE1_REPR_LAYERS"],
        "repr_pool": run_config["STAGE1_REPR_POOL"],

        "unfreeze_encoder_layers": run_config["UNFREEZE_ENCODER_LAYERS"],
        "resolved_unfreeze_encoder_layers": run_config["STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS"],
        "total_encoder_layers": run_config["TOTAL_ENCODER_LAYERS"],

        "whisper_train_layer_start": run_config["STAGE1_WHISPER_TRAIN_LAYER_START"],
        "whisper_train_layer_end": run_config["STAGE1_WHISPER_TRAIN_LAYER_END"],
    }
    save_json(fold_summary, os.path.join(meta_save_dir, "fold_summary.json"))

    if bool(run_config["SAVE_MANIFESTS"]):
        save_json({"entries": train_entries}, os.path.join(stage1_save_dir, "train_manifest.json"))
        save_json({"entries": val_entries}, os.path.join(stage1_save_dir, "val_manifest.json"))
        save_json({"entries": missing_train}, os.path.join(meta_save_dir, "missing_train_mmse_entries.json"))
        save_json({"entries": missing_val}, os.path.join(meta_save_dir, "missing_val_mmse_entries.json"))

    def _stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(entries) == 0:
            return {"count": 0}
        mmse = np.asarray([float(x["target_mmse"]) for x in entries], dtype=np.float32)
        bins = np.asarray([bin_mmse(float(x["target_mmse"]), run_config["MMSE_BIN_EDGES"]) for x in entries], dtype=np.int64)
        return {
            "count": int(len(entries)),
            "mmse_min": float(np.min(mmse)),
            "mmse_max": float(np.max(mmse)),
            "mmse_mean": float(np.mean(mmse)),
            "mmse_std": float(np.std(mmse)),
            "bin_counts": {str(i): int((bins == i).sum()) for i in range(len(run_config["MMSE_BIN_EDGES"]) - 1)},
        }

    dataset_stats = {
        "train": _stats(train_entries),
        "val": _stats(val_entries),
    }
    save_json(dataset_stats, os.path.join(meta_save_dir, "dataset_stats.json"))

def run_one_stage1_experiment(dataset_option: str, layer_id: int, fold_id: int = PROTOCOL_FOLD_ID) -> None:
    """运行单个 official_train_test Stage1-MMSE 实验。"""
    run_config = build_run_config(dataset_option=dataset_option, layer_id=layer_id, fold_id=fold_id)
    stage1_save_dir = run_config["STAGE1_SAVE_DIR"]
    meta_save_dir = run_config["META_SAVE_DIR"]

    ensure_dir(stage1_save_dir)
    ensure_dir(meta_save_dir)

    print("=" * 80)
    print(f"[Stage1-MMSE] DATASET={dataset_option} | LAYER={layer_id} | PROTOCOL={RUN_PROTOCOL_TAG}")
    print("[Stage1-MMSE] fold disabled = True")
    print(f"[Stage1-MMSE] STAGE1_SAVE_DIR = {stage1_save_dir}")
    if dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT:
        print(f"[Stage1-MMSE][Pitt] subject-disjoint policy = {PITT_SPLIT_POLICY}")
        print("[Stage1-MMSE][Pitt] Stage1 trains only on final train: expected 184 subjects / 276 visits.")

    train_entries_raw = load_official_train_entries(dataset_option)

    mmse_maps = load_mmse_maps_for_dataset(dataset_option)
    train_entries, missing_train = enrich_entries_with_mmse(
        train_entries_raw,
        dataset_option,
        mmse_maps,
    )

    if len(train_entries) == 0:
        raise ValueError(
            f"official_train 在回填 MMSE 后为空：dataset={dataset_option}, layer={layer_id}"
        )

    val_entries_placeholder = train_entries[:1]
    missing_val = []

    steps_per_epoch = max(1, len(train_entries) // max(1, int(BATCH_SIZE)))
    effective_training_steps = max(1, steps_per_epoch * int(STAGE1_MAX_EPOCHS))
    run_config["TRAINING_STEP"] = int(effective_training_steps)
    run_config["STAGE1_USE_FOLD_VAL"] = False
    run_config["STAGE1_CHECKPOINT_SELECTION"] = "last"

    print(
        f"[Stage1-MMSE][OFFICIAL-TRAIN] "
        f"raw_train_wavs={len(train_entries_raw)}, "
        f"valid_mmse_wavs={len(train_entries)}, "
        f"missing_mmse_wavs={len(missing_train)}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"epochs={STAGE1_MAX_EPOCHS}, "
        f"effective_training_steps={effective_training_steps}"
    )

    protocol_meta = {
        "protocol": str(RUN_PROTOCOL_TAG),
        "fold_id": None,
        "fold_disabled": True,
        "stage1_use_fold_val": False,
        "official_train_used": True,
        "official_test_used": False,
        "pitt_subject_disjoint": bool(dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
        "pitt_split_policy": (PITT_SPLIT_POLICY if dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT else None),
        "pitt_expected_final_train_subjects": (PITT_EXPECTED_FINAL_TRAIN_SUBJECTS if dataset_option == "Pitt" else None),
        "pitt_expected_final_train_visits": (PITT_EXPECTED_FINAL_TRAIN_VISITS if dataset_option == "Pitt" else None),
        "note": (
            "Stage1 Pitt subject-disjoint final-train only; final test is never used by Stage1."
            if dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
            else "Stage1 official_train_test: official_train is used; official_test is not used."
        ),
    }

    save_stage1_manifests(
        stage1_save_dir=stage1_save_dir,
        meta_save_dir=meta_save_dir,
        run_config=run_config,
        fold_meta=protocol_meta,
        train_entries=train_entries,
        val_entries=[],
        missing_train=missing_train,
        missing_val=missing_val,
    )

    trainer, _logger = build_trainer(stage1_save_dir, run_config)
    model = WhisperStage1MMSE(
        train_entries=train_entries,
        val_entries=val_entries_placeholder,
        saving_path=stage1_save_dir,
        run_config=run_config,
    )

    ckpt_path = run_config["RESUME_CHECKPOINT"]
    if ckpt_path in [None, "None", ""]:
        ckpt_path = None

    trainer.fit(model, ckpt_path=ckpt_path)

    if bool(ENABLE_STAGE_QUALITY_EVAL):
        raise RuntimeError(
            "official_train_test no-val-last 设置下应保持 ENABLE_STAGE_QUALITY_EVAL=False。"
        )

    save_text(
        "Stage1-MMSE official_train_test run finished successfully.\n",
        os.path.join(stage1_save_dir, "run_notes.txt"),
    )

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

# 批量运行入口
def main():
    """主入口："""
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    seed_everything_for_repro(SEED)

    if len(LAYERS_TO_RUN) == 0:
        raise ValueError("LAYERS_TO_RUN 为空，至少需要一个层号。")
    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS 为空，至少需要一个数据集。")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST 为空，至少需要一个解冻层数。")

    print("[CONFIG]")
    print("  PROJECT_ROOT =", PROJECT_ROOT)
    print("  MMSE_PTAPT_ROOT =", MMSE_PTAPT_ROOT)
    print("  DATASET_OPTIONS =", DATASET_OPTIONS)
    print("  RUN_PROTOCOL_TAG =", RUN_PROTOCOL_TAG)
    print("  LAYERS_TO_RUN =", LAYERS_TO_RUN)
    print("  MODEL_ROOT_TAG =", MODEL_ROOT_TAG)
    print("  UNFREEZE_ENCODER_LAYERS_LIST =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  WHISPER_MODEL_NAME =", WHISPER_MODEL_NAME)
    print("  STAGE1_MAX_EPOCHS =", STAGE1_MAX_EPOCHS)
    print("  fold disabled = True")
    print("  protocol subdir disabled = True")

    for dataset_option in [str(x) for x in DATASET_OPTIONS]:
        if dataset_option not in MMSE_META_PATHS:
            raise ValueError(f"不支持的 DATASET_OPTION={dataset_option}")

        for unfreeze_count in [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "=" * 100)
            print("[Stage1 BATCH ITEM]")
            print("  DATASET_OPTION =", DATASET_OPTION)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("=" * 100)

            for layer_id in LAYERS_TO_RUN:
                run_one_stage1_experiment(
                    dataset_option=DATASET_OPTION,
                    layer_id=int(layer_id),
                    fold_id=PROTOCOL_FOLD_ID,
                )

if __name__ == "__main__":
    main()
