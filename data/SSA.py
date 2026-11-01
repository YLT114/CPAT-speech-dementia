#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Whisper masked hidden-state Self-supervised fine-tuning with single layer features.
Simplified version retention of original parameters, random number call order, training/loss/evaluation/export and saving paths.
Pitt uses original_test_subjects_holdout so subjects never overlap between training and test exports.
Current default: Pitt, layer 12, encoder layers 1-12 unfrozen, secondary MSE objective, 5 epochs, final checkpoint.
MMSE metadata is used only to screen valid audio; self-supervised targets do not use MMSE labels or ASR.

conda activate whisper_ssa
python SSA.py
"""


try:
    from .paper_names import paper_dataset_name
except ImportError:
    from paper_names import paper_dataset_name

import os
import random
import numpy as np

GPU_ID = 5
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
import csv
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.dataloader import default_collate

from transformers import WhisperProcessor, WhisperModel, WhisperConfig

# Project, features, models and metadata root catalogues.
PROJECT_ROOT = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT, "feature_tensor")
MODEL_ROOT_DIR = os.path.join(PROJECT_ROOT, "models")
SPLIT_ROOT_DIR = os.path.join(PROJECT_ROOT, "split")

# Datasets: ADReSS-IS2020 / ADReSSo21 / Pitt / PROCESS-2; default only Pitt.
DATASET_KEYS = ["ADReSS-IS2020"]

DATASET_KEY = DATASET_KEYS[0]

# Pitt The testee was divided and the original strict counting was checked.
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

# SSA model and save-path labels.
MODEL_NAME = "SSA_lr3e-6_epo5_repeat"
FEATURE_PREFIX = "whisper"

BASE_WHISPER_MODEL = "/home/user/whisper_lru_3/whisper_small"
LOCAL_FILES_ONLY = True

# Export layer; an empty list shows all levels of encoder.
SINGLE_LAYERS = [12]

RUN_PROTOCOL_TAG = "official_train_test"

# Number of layers that are continuously unfrozen from the export layer forward; run in the original sequence of batches.
UNFREEZE_ENCODER_LAYERS_LIST = [1]

UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# Old manifest compatibility field; the fold division is not performed at the current entrance.
FOLD_IDS = [1]

RESOLVED_UNFREEZE_ENCODER_LAYERS: List[int] = []

# Paths attached to the list of argument names;[] only singleX_unfreeY.
PATH_PARAM_TAGS: List[str] = []

# Audio sampling rates, section lengths, overlap rates and wave shapes are unified.
SAMPLING_RATE = 16000
SEGMENT_SEC = 30
SEGMENT_LENGTH = int(SEGMENT_SEC * SAMPLING_RATE)
OVERLAP_RATE = 0.3
ENABLE_AUDIO_NORMALIZE = True

SEED = 400

# segment The mode is SSA_BATCH_SIZE; audio_balanced, and the mode is SSA_AUDIO_BATCH_SIZE.
SSA_BATCH_SIZE = 4  # segment batch size
SSA_AUDIO_BATCH_SIZE = 2

SSA_NUM_EPOCHS = 5
SSA_lr = 3e-6
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
PRECISION = "fp16"          # "fp32" / "fp16" / "bf16"

# Only randomly span masking on valid token; the original probability and random call order remain unchanged.
MASK_SPAN_RATIO = 0.06      # Effective token mask ratio
MASK_SPAN_LENGTH = 10       # Average length span
MASK_VALUE = 0.0            # Replace mask for mel/input_features

EMA_MOMENTUM = 0.995

PREDICTOR_HIDDEN_DIM = 256
PREDICTOR_DROPOUT = 0.10

# Supervise loss polymers: segment or audio_balanced; optional path remains complete.
SSA_TRAINING_LEVEL = "segment"

# Self-monitoring objectives, MSE/cosine weights and predictor/direct weights.
SSA_LOSS_TYPE = "mse"
SSA_MSE_WEIGHT = 1.0
SSA_COSINE_WEIGHT = 0.0

SSA_PRED_LOSS_WEIGHT = 1.0
SSA_DIRECT_LOSS_WEIGHT = 0.0

# Export feature to be normalized: none / Leyernorm / std / in_post.
NORM_FOR_SINGLE = "ln_post"
DROP_INPUT_LAYER = True

# Export the switch; export only forwards and does not update Whisper parameters.
SKIP_FEATURE_EXPORT = False

SAVE_META = True

ENABLE_NOTEBOOK_LOG = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything_for_repro(seed: int) -> None:
    """Standardize the setting of random seeds to ensure, to the extent possible, that experiments are re-emerging."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

def ensure_dir(path: str) -> None:
    """Automatically create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

# File, section and MMSE metadata tool
def natural_key(p: str):
    """Filename naturally sorting:"""
    s = Path(str(p)).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def canonical_wav_path(path: str) -> str:
    """Regularizes the wav path to avoid the relative path/symlink leading to inconsistent matches."""
    return str(Path(str(path)).expanduser().resolve())

def save_json(obj: Dict[str, Any], path: str) -> None:
    """Saves JSON and automatically creates a parent directory."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(text: str, path: str) -> None:
    """Saves the plain text and automatically creates the parent directory."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def write_csv_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write several CD lines as csv."""
    ensure_dir(str(Path(path).parent))
    if len(rows) == 0:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _clean_path_tag(x) -> str:
    """Clears the path parameter labels and avoids the entry of spaces, slashes, etc. into the directory name."""
    if x is None:
        return ""
    s = str(x).strip()
    if s in ["", "None", "none", "null", "Null"]:
        return ""
    s = s.replace("=", "").replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_.-]+", "", s)
    s = s.strip("_-.")
    return s

PATH_PARAM_ABBR = {
    "SSA_NUM_EPOCHS": "epo",
    "SSA_lr": "lr",
    "SSA_PRED_LOSS_WEIGHT": "predloss",
    "SSA_DIRECT_LOSS_WEIGHT": "directloss",
    "SSA_MSE_WEIGHT": "mse",
    "SSA_COSINE_WEIGHT": "cos",
    "PREDICTOR_HIDDEN_DIM": "hidden",
    "UNFREEZE_ENCODER_LAYERS": "unfree",
}

def _format_path_value(v) -> str:
    """Formats the parameter value into a string suitable for the path name."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(int(v))
    if isinstance(v, float):
        if float(v).is_integer():
            return f"{float(v):.1f}"
        return f"{float(v):g}"
    return str(v)

def resolve_path_param_tags(path_param_tags=None) -> List[str]:
    """Resolve PATH_PARAM_TAGS to a list of labels that really write to a path."""
    if path_param_tags is None:
        path_param_tags = globals().get("PATH_PARAM_TAGS", [])

    if isinstance(path_param_tags, (str, int, float)):
        raw_tags = [path_param_tags]
    else:
        raw_tags = list(path_param_tags)

    resolved = []
    for tag in raw_tags:
        tag_str = str(tag).strip()
        if tag_str in PATH_PARAM_ABBR:
            if tag_str not in globals():
                raise KeyError(f"PATH_PARAM_TAGS parameter(s) {tag_str} Not defined in the global parameters area.")
            abbr = PATH_PARAM_ABBR[tag_str]
            value = globals()[tag_str]
            resolved.append(f"{abbr}{_format_path_value(value)}")
        else:
            resolved.append(tag_str)

    resolved = [_clean_path_tag(x) for x in resolved]
    resolved = [x for x in resolved if x != ""]
    return resolved

def build_path_param_suffix(path_param_tags=None) -> str:
    """Construct the path suffix according to PATH_PARAM_TAGS."""
    tags = resolve_path_param_tags(path_param_tags)
    return "_".join(tags)

def build_layer_name(single_layer: int, unfreeze_layer_count: int, path_param_tags=None) -> str:
    """Construct selection directory name."""
    base = f"single{int(single_layer)}_unfree{int(unfreeze_layer_count)}"
    suffix = build_path_param_suffix(path_param_tags)
    if suffix == "":
        return base
    return f"{base}_{suffix}"

def get_whisper_encoder_layer_count(whisper_model_name_or_path: str) -> int:
    """Read from Whisper config encoder total layer."""
    cfg = WhisperConfig.from_pretrained(whisper_model_name_or_path, local_files_only=LOCAL_FILES_ONLY)
    num_layers = getattr(cfg, "encoder_layers", None)
    if num_layers is None:
        raise AttributeError("WhisperConfig is missing encoder_layers and cannot automatically resolve layers.")
    return int(num_layers)

def resolve_single_layers(single_layers: Optional[Sequence[int]], whisper_model_name_or_path: str) -> List[int]:
    """Resolve the list of layers that you want to go through eventually."""
    total_layers = get_whisper_encoder_layer_count(whisper_model_name_or_path)
    if single_layers is None or len(list(single_layers)) == 0:
        return list(range(1, total_layers + 1))

    out = sorted({int(x) for x in list(single_layers)})
    for x in out:
        if not (1 <= int(x) <= total_layers):
            raise ValueError(f"SINGLE_LAYERS The mid-level is illegal:{x},The legal range is [1,{total_layers}]")
    return out

def resolve_unfreeze_layers_for_single_layer(
    single_layer: int,
    total_layers: int,
    unfreeze_layer_count: int,
) -> List[int]:
    """Explains which layers are actually unfrozen, based on the current single_layer and UNFREEZE_ENCODER_LAYERS."""
    single_layer = int(single_layer)
    total_layers = int(total_layers)
    unfreeze_layer_count = int(unfreeze_layer_count)

    if not (1 <= single_layer <= total_layers):
        raise ValueError(f"single_layer={single_layer} Beyond range [1, {total_layers}]")
    if unfreeze_layer_count <= 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS Must be >= 1")

    start_layer = max(1, single_layer - unfreeze_layer_count + 1)
    end_layer = min(single_layer, total_layers)

    resolved = list(range(start_layer, end_layer + 1))
    if len(resolved) == 0:
        raise ValueError(
            f"single_layer={single_layer}, "
            f"unfreeze_layer_count={unfreeze_layer_count} Could not close temporary folder: %s"
        )
    return resolved

def make_default_split_file_tag(
    feature_prefix: str,
    single_layer: int,
    overlap_rate: float,
    segment_sec: int,
) -> str:
    """Construct a feature filename prefix that is consistent with the current feature style baseline / finetune."""
    ov = float(overlap_rate)
    ov_str = f"{ov:.1f}".rstrip("0").rstrip(".") if ov % 1 != 0 else str(int(ov))
    if "." not in ov_str:
        ov_str = f"{ov_str}.0"
    return f"{str(feature_prefix)}_single{int(single_layer)}_{ov_str}_{int(segment_sec)}s"

def build_ssa_model_dir(
    dataset_key: str,
    model_name: str,
    layer_name: str,
    fold_id: Optional[int] = None,
) -> str:
    """SSA model."""
    return os.path.join(
        MODEL_ROOT_DIR,
        str(dataset_key),
        f"ft_{str(model_name)}",
        str(layer_name),
        RUN_PROTOCOL_TAG,
    )

def build_ssa_feature_save_path(
    dataset_key: str,
    model_name: str,
    layer_name: str,
    fold_id: Optional[int] = None,
) -> str:
    """SSA Features save directory."""
    return os.path.join(
        FEATURE_ROOT_DIR,
        str(dataset_key),
        f"ft_{str(model_name)}",
        str(layer_name),
    )

def build_ssa_local_split_path(
    dataset_key: str,
    model_name: str,
    layer_name: str,
    fold_id: Optional[int],
    feature_prefix: str,
    single_layer: int,
    overlap_rate: float,
    segment_sec: int,
) -> str:
    """Path to save metadata under the current feature directory."""
    feature_save_path = build_ssa_feature_save_path(
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
    return os.path.join(
        feature_save_path,
        f"official_train_test_metadata_{file_tag}.json",
    )

def build_notebook_log_root(
    dataset_key: str,
    model_name: str,
    layer_name: str,
    fold_id: Optional[int] = None,
) -> str:
    """Light training log catalogue."""
    return os.path.join(
        MODEL_ROOT_DIR,
        str(dataset_key),
        f"ft_{str(model_name)}",
        str(layer_name),
        RUN_PROTOCOL_TAG,
        "csv_logs",
    )

ACTIVE_SINGLE_LAYER = None
ACTIVE_LAYER_NAME = None
ACTIVE_FOLD_ID = None
ACTIVE_TOTAL_ENCODER_LAYERS = None
TRAIN_VAL_SPLIT_PATH = None
SSA_MODEL_DIR = None
FEATURE_SAVE_PATH = None
LOCAL_FEATURE_SPLIT_SAVE_PATH = None
NOTEBOOK_LOG_ROOT = None

def apply_ssa_run_context(
    single_layer: int,
    fold_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Updates the context based on the current single_layer."""
    global ACTIVE_SINGLE_LAYER
    global ACTIVE_LAYER_NAME
    global ACTIVE_FOLD_ID
    global ACTIVE_TOTAL_ENCODER_LAYERS
    global TRAIN_VAL_SPLIT_PATH
    global SSA_MODEL_DIR
    global FEATURE_SAVE_PATH
    global LOCAL_FEATURE_SPLIT_SAVE_PATH
    global NOTEBOOK_LOG_ROOT
    global RESOLVED_UNFREEZE_ENCODER_LAYERS

    total_layers = get_whisper_encoder_layer_count(BASE_WHISPER_MODEL)
    resolved_unfreeze = resolve_unfreeze_layers_for_single_layer(
        single_layer=int(single_layer),
        total_layers=int(total_layers),
        unfreeze_layer_count=int(UNFREEZE_ENCODER_LAYERS),
    )

    layer_name = build_layer_name(
        single_layer=int(single_layer),
        unfreeze_layer_count=int(UNFREEZE_ENCODER_LAYERS),
        path_param_tags=PATH_PARAM_TAGS,
    )

    ACTIVE_SINGLE_LAYER = int(single_layer)
    ACTIVE_LAYER_NAME = str(layer_name)
    ACTIVE_FOLD_ID = None
    ACTIVE_TOTAL_ENCODER_LAYERS = int(total_layers)
    RESOLVED_UNFREEZE_ENCODER_LAYERS = list(resolved_unfreeze)

    TRAIN_VAL_SPLIT_PATH = None

    SSA_MODEL_DIR = build_ssa_model_dir(
        DATASET_KEY,
        MODEL_NAME,
        ACTIVE_LAYER_NAME,
        fold_id=None,
    )
    FEATURE_SAVE_PATH = build_ssa_feature_save_path(
        DATASET_KEY,
        MODEL_NAME,
        ACTIVE_LAYER_NAME,
        fold_id=None,
    )
    LOCAL_FEATURE_SPLIT_SAVE_PATH = build_ssa_local_split_path(
        dataset_key=DATASET_KEY,
        model_name=MODEL_NAME,
        layer_name=ACTIVE_LAYER_NAME,
        fold_id=None,
        feature_prefix=FEATURE_PREFIX,
        single_layer=ACTIVE_SINGLE_LAYER,
        overlap_rate=OVERLAP_RATE,
        segment_sec=SEGMENT_SEC,
    )
    NOTEBOOK_LOG_ROOT = build_notebook_log_root(
        DATASET_KEY,
        MODEL_NAME,
        ACTIVE_LAYER_NAME,
        fold_id=None,
    )

    ensure_dir(SSA_MODEL_DIR)
    ensure_dir(FEATURE_SAVE_PATH)
    ensure_dir(NOTEBOOK_LOG_ROOT)
    ensure_dir(str(Path(LOCAL_FEATURE_SPLIT_SAVE_PATH).parent))

    run_config = {
        "DATASET_KEY": DATASET_KEY,
        "MODEL_NAME": MODEL_NAME,
        "BASE_WHISPER_MODEL": BASE_WHISPER_MODEL,
        "SINGLE_LAYER": int(ACTIVE_SINGLE_LAYER),
        "LAYER_NAME": str(ACTIVE_LAYER_NAME),
        "RUN_PROTOCOL_TAG": str(RUN_PROTOCOL_TAG),
        "FOLD_ID": None,
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),
        "RESOLVED_UNFREEZE_ENCODER_LAYERS": list(RESOLVED_UNFREEZE_ENCODER_LAYERS),
        "TOTAL_ENCODER_LAYERS": int(ACTIVE_TOTAL_ENCODER_LAYERS),
        "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
        "RESOLVED_PATH_PARAM_TAGS": resolve_path_param_tags(PATH_PARAM_TAGS),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(PATH_PARAM_TAGS),
        "TRAIN_VAL_SPLIT_PATH": None,
        "SSA_MODEL_DIR": str(SSA_MODEL_DIR),
        "FEATURE_SAVE_PATH": str(FEATURE_SAVE_PATH),
        "LOCAL_FEATURE_SPLIT_SAVE_PATH": str(LOCAL_FEATURE_SPLIT_SAVE_PATH),
        "NOTEBOOK_LOG_ROOT": str(NOTEBOOK_LOG_ROOT),
    }

    print("[CONTEXT] DATASET_KEY =", DATASET_KEY)
    print("[CONTEXT] RUN_PROTOCOL_TAG =", RUN_PROTOCOL_TAG)
    print("[CONTEXT] ACTIVE_LAYER_NAME =", ACTIVE_LAYER_NAME)
    print("[CONTEXT] RESOLVED_UNFREEZE_ENCODER_LAYERS =", RESOLVED_UNFREEZE_ENCODER_LAYERS)
    print("[CONTEXT] SSA_MODEL_DIR =", SSA_MODEL_DIR)
    print("[CONTEXT] FEATURE_SAVE_PATH =", FEATURE_SAVE_PATH)
    print("[CONTEXT] LOCAL_FEATURE_SPLIT_SAVE_PATH =", LOCAL_FEATURE_SPLIT_SAVE_PATH)

    return run_config

# MMSE Metadata path and listing.
MMSE_META_PATHS = {
    "PROCESS-2": {
        "meta": "/home/user/Downloads/PROCESS-2/meta-info.csv",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/cc_meta_data.txt",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/cd_meta_data.txt",
        "test_all": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/ADReSS 2020 test result.txt",
    },
    "ADReSSo21": {
        "train_all": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/adresso-train-mmse-scores.csv",
        "test_all": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/test_results_task2_groundtruth.csv",
    },
    "Pitt": {
        "all_meta": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/PItt-data(1).xlsx",
        "sheet_name": "data",
        "header_row": 2,
        "id_col": "id",
    },
}

# Audio catalogues of data sets.
DATASETS = {
    "PROCESS-2": {
        "train_cc": "/home/user/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/user/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/user/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/user/Downloads/PROCESS-2/test/cd",
    },
    "ADReSSo21": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "Pitt": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence",
        "test_cd": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence",
    },
}

for _dataset_key in DATASET_KEYS:
    assert _dataset_key in DATASETS, f"Unknown DATASET_KEY={_dataset_key}"
    assert _dataset_key in MMSE_META_PATHS, f"Current script not configured {_dataset_key} MMSE metadata from"

assert DATASET_KEY in DATASETS, f"Unknown DATASET_KEY={DATASET_KEY}"
assert DATASET_KEY in MMSE_META_PATHS, f"Current script not configured {DATASET_KEY} MMSE metadata from"

print("DATASET_KEYS =", DATASET_KEYS)
print("DEFAULT DATASET_KEY =", DATASET_KEY)

# MMSE Tab Parsing
def _pick_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """A real line of DataFrame is found in the list of candidates."""
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"No candidate fields found {candidates},Current list:{list(df.columns)}")

def _safe_to_float(x):
    """Turn float; NA/empty returns None."""
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
    """Harmonizes the MMSE map keys."""
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
    """Parsing subject_id and visit_index from Pitt file names."""
    if raw_name is None or pd.isna(raw_name):
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    return str(int(m.group(1))), int(m.group(2))

def _pitt_visit_to_mmse_col(visit_index: int) -> str:
    """Pitt visit Map to MMSE for listing:"""
    visit_index = int(visit_index)
    if visit_index < 0:
        raise ValueError(f"Illegal visit_index={visit_index}")
    if visit_index == 0:
        return "mms"
    return f"mmse{visit_index + 1}"

def _build_map_from_rows(df: pd.DataFrame, dataset_key: str, id_col: str, mmse_col: str) -> Dict[str, float]:
    """Construct { norms ID: mmse} from DataFrame columns."""
    mmse_map = {}
    for _, row in df.iterrows():
        key = _normalize_mmse_key(dataset_key, row[id_col])
        val = _safe_to_float(row[mmse_col])
        if key is None or val is None:
            continue
        mmse_map[key] = float(val)
    return mmse_map

def _read_adress_is2020_mmse(cfg: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Read MMSE txt metadata from ADReSS-IS2020."""
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

def _read_adresso21_mmse(cfg: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Read MMSE csv metadata from ADReSSo21."""
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
    """Read Pitt visit-level MMSE map."""
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

def _read_process2_mmse(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """Read meta-info.csv for PROCESS-2, and construct a MMSE map for four subsets."""
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

def load_mmse_maps_for_dataset(dataset_key: str) -> Dict[str, Dict[Any, float]]:
    """Loads the MMSE map of the specified data set."""
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"Dataset not configured yet {dataset_key} MSME_META_PATHS")

    cfg = MMSE_META_PATHS[dataset_key]

    if dataset_key == "PROCESS-2":
        out = _read_process2_mmse(cfg)
    elif dataset_key == "ADReSS-IS2020":
        out = _read_adress_is2020_mmse(cfg)
    elif dataset_key == "ADReSSo21":
        out = _read_adresso21_mmse(cfg)
    elif dataset_key == "Pitt":
        out = _read_pitt_mmse(cfg)
    else:
        raise ValueError(f"Unrealized datasets {dataset_key} read on MMSE.")

    print(
        f"[MMSE] {dataset_key} Map load complete:"
        f"train_cc={len(out['train_cc'])}, "
        f"train_cd={len(out['train_cd'])}, "
        f"test_cc={len(out['test_cc'])}, "
        f"test_cd={len(out['test_cd'])}"
    )

    return out

def resolve_mmse_for_entry(row: Dict[str, Any], dataset_key: str, mmse_maps: Dict[str, Dict[Any, float]]) -> Tuple[Optional[float], str]:
    """For official train / official test entry, get MMSE."""
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
    bucket = ("test_cc" if group == "cc" else "test_cd") if is_official_test else ("train_cc" if group == "cc" else "train_cd")

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

def _is_process2_ctd_wav_path(wav_path: str, dataset_key: str = None) -> bool:
    """PROCESS-2 Special filter:"""
    if dataset_key is None:
        dataset_key = globals().get("DATASET_KEY", "")

    if str(dataset_key).strip().lower() != "process-2":
        return True

    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None

def list_wav_files(audio_dir: str, dataset_key: str = None) -> List[str]:
    """Reads a wav file under a official subdirectorate andsorts it to natural_key."""
    if dataset_key is None:
        dataset_key = globals().get("DATASET_KEY", "")

    audio_dir = str(audio_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"official audio dir Cannot initialise Evolution's mail component.{audio_dir}")

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
            f"official audio dir No eligible wav document:{audio_dir}, dataset_key={dataset_key}"
        )

    if str(dataset_key).strip().lower() == "process-2":
        print(f"[PROCESS-2][CTD] {audio_dir}: Read only CTD wav, number of documents = {len(files)}")

    return [canonical_wav_path(x) for x in files]

def build_entries_from_official_dir(
    audio_dir: str,
    split_name: str,
    group_name: str,
    dataset_key: str = None,
) -> List[Dict[str, Any]]:
    """Renumber official train_cc/train_cd/test_cc/test_cd as a entry list."""
    if dataset_key is None:
        dataset_key = globals().get("DATASET_KEY", "")

    split_name = str(split_name).strip().lower()
    group_name = str(group_name).strip().lower()

    if split_name not in ["train", "test"]:
        raise ValueError(f"split_name Must be train/test, current ={split_name}")
    if group_name not in ["cc", "cd"]:
        raise ValueError(f"group_name must be cc/cd, current ={group_name}")

    entries: List[Dict[str, Any]] = []
    for wav_path in list_wav_files(audio_dir, dataset_key=dataset_key):
        entries.append({
            "wav_path": canonical_wav_path(wav_path),
            "wav_name": Path(wav_path).name,
            "split": split_name,
            "group": group_name,
            "source_subset": f"{split_name}_{group_name}",
        })

    return entries

def _pitt_make_missing_visit_record(source_subset: str, wav_path: str, subject_id=None, visit_index=None) -> Dict[str, Any]:
    """Pitt subject-disjoint Unable to access MSME review visit at reconstruction."""
    rec = {
        "dataset": "Pitt",
        "paper_dataset": paper_dataset_name("Pitt"),
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
    """Keep audit documents Pitt split of the same calibre as the current Pitt subject-disjoint protocol."""
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

# Pitt Retardation and audit of candidates
def _build_pitt_subject_disjoint_train_test_entries() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Rebuilding in harmony with the current Pitt subject-disjoint protocol Pitt final train/test."""
    global _PITT_SUBJECT_DISJOINT_CACHE

    if _PITT_SUBJECT_DISJOINT_CACHE is not None:
        cached = _PITT_SUBJECT_DISJOINT_CACHE
        return copy.deepcopy(cached["train_entries"]), copy.deepcopy(cached["test_entries"])

    if PITT_SPLIT_POLICY != "original_test_subjects_holdout":
        raise ValueError(f"Unsupported PITT_SPLIT_POLICY={PITT_SPLIT_POLICY}")

    dataset_paths = DATASETS["Pitt"]
    mmse_maps = load_mmse_maps_for_dataset("Pitt")
    all_mmse_map = mmse_maps["train_cc"]  # Pitt Four Maps are always the same visit-level all_map.

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
                    "[Pitt split] Duplicate found in subject-visit key:"
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
        raise RuntimeError(f"[Pitt split] subject-disjoint Check failed, overlap={sorted(overlap)}")

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
        raise RuntimeError(f"[Pitt split] Eventually train/test still exists subject overlap: {sorted(final_overlap)}")
    if len(final_train_ids | final_test_ids) != len(valid_subjects):
        raise RuntimeError("[Pitt split] Eventually subject union is inconsistent with valid subject")
    if sum(len(v) for v in final_records.values()) != valid_visit_count:
        raise RuntimeError("[Pitt split] The final total of visit does not match the total of visit valid.")

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
            msg = f"[Pitt split] {name}={actual},Expected ={expected}"
            if PITT_STRICT_COUNT_CHECK:
                raise RuntimeError(msg + ".To avoid unintentional changes in experimental data, it has been discontinued.")
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
    print("[Pitt subject-disjoint final split | SSA]")
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

def load_official_train_test_entries(dataset_key: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Construct a certain train/ official test inductions."""
    if dataset_key not in DATASETS:
        raise ValueError(f"DATASETS No configuration DATASET_KEY={dataset_key}")

    if (
        str(dataset_key) == "Pitt"
        and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT)
    ):
        return _build_pitt_subject_disjoint_train_test_entries()

    paths = DATASETS[dataset_key]
    train_entries = (
        build_entries_from_official_dir(
            paths["train_cc"],
            split_name="train",
            group_name="cc",
            dataset_key=dataset_key,
        )
        + build_entries_from_official_dir(
            paths["train_cd"],
            split_name="train",
            group_name="cd",
            dataset_key=dataset_key,
        )
    )
    test_entries = (
        build_entries_from_official_dir(
            paths["test_cc"],
            split_name="test",
            group_name="cc",
            dataset_key=dataset_key,
        )
        + build_entries_from_official_dir(
            paths["test_cd"],
            split_name="test",
            group_name="cd",
            dataset_key=dataset_key,
        )
    )

    if len(train_entries) == 0 or len(test_entries) == 0:
        raise ValueError(f"official train/test entries : Train={len(train_entries)}, test={len(test_entries)}")

    for i, row in enumerate(train_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    for i, row in enumerate(test_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    return train_entries, test_entries

def attach_mmse_and_filter(
    entries: List[Dict[str, Any]],
    dataset_key: str,
    mmse_maps: Dict[str, Dict[Any, float]],
    subset_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Enter entries back to target_mmse."""
    valid_entries: List[Dict[str, Any]] = []
    missing_records: List[Dict[str, Any]] = []

    for row in entries:
        mmse, source = resolve_mmse_for_entry(row, dataset_key, mmse_maps)
        if mmse is None:
            rec = {
                "dataset": dataset_key,
                "paper_dataset": paper_dataset_name(dataset_key),
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

def split_segment_boundaries(num_samples: int, segment_length: int, overlap_rate: float) -> List[Tuple[int, int, int]]:
    """Resumes the 30s + overlap cutting idea of frozen.py to generate a border."""
    num_samples = int(num_samples)
    segment_length = int(segment_length)
    overlap_rate = float(overlap_rate)

    if segment_length <= 0:
        raise ValueError(f"segment_length Must be > 0, now ={segment_length}")
    step_size = int(segment_length * (1.0 - overlap_rate))
    if step_size <= 0:
        raise ValueError(f"overlap_rate={overlap_rate} Caused step_size <= 0")

    if num_samples <= 0:
        return [(0, 0, segment_length)]

    total_segments = int(num_samples / step_size) + (
        1 if num_samples % step_size > overlap_rate * segment_length else 0
    )
    total_segments = max(1, int(total_segments))

    boundaries = []
    for segment_id in range(total_segments):
        start = int(segment_id * step_size)
        end = int(start + segment_length)
        boundaries.append((int(segment_id), start, end))
    return boundaries

def load_audio_16k(wav_path: str) -> np.ndarray:
    """Read audio in baseline mode:"""
    wav_path = canonical_wav_path(wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"Could not close temporary folder: %s{wav_path}")
    wav, _ = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True, dtype=np.float32)

    if ENABLE_AUDIO_NORMALIZE:
        peak = float(np.max(np.abs(wav))) if len(wav) > 0 else 0.0
        if peak > 1e-8:
            wav = wav / peak

    return np.asarray(wav, dtype=np.float32).reshape(-1)

def load_segment_with_padding_from_wav(y: np.ndarray, start: int, end: int, segment_id: int) -> Tuple[np.ndarray, int]:
    """Remove a segment from 16k waveform and revert to the end paragraph of frozen.py to complete the logic."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    start = int(start)
    end = int(end)
    segment_id = int(segment_id)

    segment = y[start:end]

    if len(segment) < SEGMENT_LENGTH and segment_id > 0:
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

def load_segment_with_padding(wav_path: str, start: int, end: int, segment_id: int) -> Tuple[np.ndarray, int]:
    """Read wav and return to 30s waveform and active length specified segment."""
    y = load_audio_16k(wav_path)
    return load_segment_with_padding_from_wav(y, start=start, end=end, segment_id=segment_id)

def build_segment_entries_from_audio_entries(
    entries: List[Dict[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    """Expand wav-level entries into segment-level entries."""
    segment_entries: List[Dict[str, Any]] = []

    for wav_row in tqdm(entries, desc=f"Build {split_name} segments", dynamic_ncols=True):
        wav_row = copy.deepcopy(wav_row)
        wav_path = canonical_wav_path(wav_row["wav_path"])
        wav_name = str(wav_row.get("wav_name", Path(wav_path).name))

        y = load_audio_16k(wav_path)
        n_samples = int(len(y))
        boundaries = split_segment_boundaries(
            num_samples=n_samples,
            segment_length=SEGMENT_LENGTH,
            overlap_rate=OVERLAP_RATE,
        )

        for segment_id, start, end in boundaries:
            _seg, valid_len = load_segment_with_padding_from_wav(
                y, start=start, end=end, segment_id=segment_id
            )

            row = copy.deepcopy(wav_row)
            row.update({
                "split": str(split_name).lower(),
                "wav_path": wav_path,
                "wav_name": wav_name,
                "segment_id": int(segment_id),
                "segment_start": int(start),
                "segment_end": int(end),
                "segment_valid_len": int(valid_len),
                "parent_num_samples_16k": int(n_samples),
                "segment_sec": float(SEGMENT_SEC),
                "overlap_rate": float(OVERLAP_RATE),
            })
            segment_entries.append(row)

    if len(segment_entries) == 0:
        raise ValueError(f"{split_name} segment_entries Empty, please check the audio path and split.")

    return segment_entries

# segment / audio-balanced Datasets
class SegmentWaveDataset(data.Dataset):
    """Unified segment-level waveform data sets."""
    def __init__(self, entries: List[Dict[str, Any]], require_target: bool = False):
        self.entries = [copy.deepcopy(x) for x in entries]
        self.require_target = bool(require_target)

        if len(self.entries) == 0:
            raise ValueError("SegmentWaveDataset Roger that, entries.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        row = self.entries[i]
        wav_path = canonical_wav_path(row["wav_path"])
        wav_name = str(row.get("wav_name", Path(wav_path).name))
        segment_id = int(row["segment_id"])
        start = int(row["segment_start"])
        end = int(row["segment_end"])

        wav_30, valid_len = load_segment_with_padding(
            wav_path=wav_path,
            start=start,
            end=end,
            segment_id=segment_id,
        )

        if self.require_target:
            return (
                wav_30,
                int(valid_len),
                float(row["target_mmse"]),
                int(row["audio_id"]),
                int(segment_id),
                wav_name,
                wav_path,
            )

        return (
            wav_30,
            int(valid_len),
            int(row.get("audio_id", -1)),
            int(segment_id),
            wav_name,
            wav_path,
        )

    def repeat30s_collate(self, batch):
        """Compatibility with existing script naming habits: Each sample is now fixed at 30s and only does default_collate."""
        return default_collate(batch)

def get_audio_group_key_for_ssa(row: Dict[str, Any]) -> str:
    """SSA audio-balanced Group key."""
    if row.get("audio_id", None) not in [None, ""]:
        return f"audio_id::{row['audio_id']}"

    wav_path = row.get("wav_path", "")
    if wav_path not in [None, ""]:
        return f"wav_path::{canonical_wav_path(wav_path)}"

    raise ValueError(f"Could not parse audio group key from row: {row}")

class AudioBalancedSegmentWaveDataset(data.Dataset):
    """audio-balanced SSA Data sets."""

    def __init__(self, segment_entries: List[Dict[str, Any]]):
        self.segment_entries = [copy.deepcopy(x) for x in segment_entries]

        if len(self.segment_entries) == 0:
            raise ValueError("AudioBalancedSegmentWaveDataset Roger that, segment_entries.")

        groups: Dict[str, Dict[str, Any]] = {}

        for row in self.segment_entries:
            row = copy.deepcopy(row)
            row["wav_path"] = canonical_wav_path(row["wav_path"])
            row["wav_name"] = str(row.get("wav_name", Path(row["wav_path"]).name))
            row["segment_id"] = int(row["segment_id"])
            row["segment_start"] = int(row["segment_start"])
            row["segment_end"] = int(row["segment_end"])
            row["segment_valid_len"] = int(row.get("segment_valid_len", SEGMENT_LENGTH))

            audio_key = get_audio_group_key_for_ssa(row)

            if audio_key not in groups:
                groups[audio_key] = {
                    "audio_key": audio_key,
                    "audio_id": row.get("audio_id", audio_key),
                    "wav_path": row["wav_path"],
                    "wav_name": row["wav_name"],
                    "segments": [],
                }

            groups[audio_key]["segments"].append(row)

        self.records: List[Dict[str, Any]] = []
        for audio_key, rec in groups.items():
            rec["segments"] = sorted(
                rec["segments"],
                key=lambda x: (int(x.get("segment_id", 0)), int(x.get("segment_start", 0))),
            )
            self.records.append(rec)

        self.records = sorted(
            self.records,
            key=lambda x: natural_key(str(x.get("wav_path", x.get("audio_key", "")))),
        )

        if len(self.records) == 0:
            raise ValueError("AudioBalancedSegmentWaveDataset The grouping was not effective audio.")

        segment_counts = [len(x["segments"]) for x in self.records]
        print(
            "[AudioBalancedSSA Dataset] "
            f"num_audio={len(self.records)}, "
            f"num_segments={sum(segment_counts)}, "
            f"min_segments_per_audio={min(segment_counts)}, "
            f"max_segments_per_audio={max(segment_counts)}, "
            f"mean_segments_per_audio={float(np.mean(segment_counts)):.3f}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]

        wav_path = canonical_wav_path(rec["wav_path"])
        wav_name = str(rec.get("wav_name", Path(wav_path).name))
        audio_id = rec.get("audio_id", rec["audio_key"])
        audio_key = str(rec["audio_key"])

        y = load_audio_16k(wav_path)

        wav_chunks = []
        valid_lens = []
        segment_ids = []
        wav_names = []
        wav_paths = []

        for row in rec["segments"]:
            segment_id = int(row["segment_id"])
            start = int(row["segment_start"])
            end = int(row["segment_end"])

            wav_30, valid_len = load_segment_with_padding_from_wav(
                y,
                start=start,
                end=end,
                segment_id=segment_id,
            )

            wav_chunks.append(wav_30.astype(np.float32, copy=False))
            valid_lens.append(int(valid_len))
            segment_ids.append(int(segment_id))
            wav_names.append(wav_name)
            wav_paths.append(wav_path)

        if len(wav_chunks) == 0:
            raise ValueError(f"audio No validity segments: {audio_key}")

        return {
            "audio_key": audio_key,
            "audio_id": audio_id,
            "wav_name": wav_name,
            "wav_path": wav_path,
            "wav_chunks": wav_chunks,
            "valid_lens": valid_lens,
            "segment_ids": segment_ids,
            "wav_names": wav_names,
            "wav_paths": wav_paths,
        }

    def repeat30s_collate(self, batch):
        """audio-balanced collate_fn."""
        flat_wavs = []
        flat_valid_lens = []
        flat_audio_ids = []
        flat_segment_ids = []
        flat_wav_names = []
        flat_wav_paths = []

        segment_to_audio = []
        segment_counts = []
        audio_keys = []
        audio_wav_names = []

        for audio_idx, item in enumerate(batch):
            wav_chunks = item["wav_chunks"]
            valid_lens = item["valid_lens"]
            segment_ids = item["segment_ids"]
            wav_names = item["wav_names"]
            wav_paths = item["wav_paths"]

            n_seg = int(len(wav_chunks))
            if n_seg <= 0:
                raise ValueError(f"collate Roger empty: {item.get('audio_key', '')}")

            audio_keys.append(str(item["audio_key"]))
            audio_wav_names.append(str(item["wav_name"]))
            segment_counts.append(n_seg)

            for wav_30, valid_len, segment_id, wav_name, wav_path in zip(
                wav_chunks,
                valid_lens,
                segment_ids,
                wav_names,
                wav_paths,
            ):
                flat_wavs.append(np.asarray(wav_30, dtype=np.float32).reshape(-1))
                flat_valid_lens.append(int(valid_len))
                flat_audio_ids.append(str(item["audio_id"]))
                flat_segment_ids.append(int(segment_id))
                flat_wav_names.append(str(wav_name))
                flat_wav_paths.append(str(wav_path))
                segment_to_audio.append(int(audio_idx))

        flat_wav_30 = torch.from_numpy(
            np.stack(flat_wavs, axis=0).astype(np.float32, copy=False)
        )
        flat_valid_len = torch.tensor(flat_valid_lens, dtype=torch.long)
        segment_to_audio = torch.tensor(segment_to_audio, dtype=torch.long)
        segment_counts = torch.tensor(segment_counts, dtype=torch.long)

        return (
            flat_wav_30,
            flat_valid_len,
            flat_audio_ids,
            flat_segment_ids,
            flat_wav_names,
            flat_wav_paths,
            segment_to_audio,
            segment_counts,
            audio_keys,
            audio_wav_names,
        )

# Overseeing student, EMA teacher and predictor
class HiddenStatePredictor(nn.Module):
    """Lightly prediction head:"""
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class WhisperSSAModel(nn.Module):
    """Whisper-SSA Main model:"""
    def __init__(
        self,
        base_model_name: str,
        unfreeze_encoder_layers: Sequence[int],
        predictor_hidden_dim: int,
        predictor_dropout: float,
        local_files_only: bool = True,
    ):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(base_model_name, local_files_only=local_files_only)
        self.hidden_size = int(self.whisper.config.d_model)

        for p in self.whisper.parameters():
            p.requires_grad = False

        encoder_layers = self.whisper.encoder.layers
        total_layers = len(encoder_layers)
        valid_layers = []
        for layer_idx_1based in sorted({int(x) for x in unfreeze_encoder_layers if int(x) > 0}):
            if not (1 <= int(layer_idx_1based) <= total_layers):
                continue
            layer = encoder_layers[int(layer_idx_1based) - 1]
            for p in layer.parameters():
                p.requires_grad = True
            valid_layers.append(int(layer_idx_1based))

        if len(valid_layers) > 0:
            for p in self.whisper.encoder.layer_norm.parameters():
                p.requires_grad = True

        self.unfreeze_encoder_layers = valid_layers

        for p in self.whisper.decoder.parameters():
            p.requires_grad = False

        self.teacher_encoder = copy.deepcopy(self.whisper.encoder)
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False
        self.teacher_encoder.eval()

        self.predictor = HiddenStatePredictor(
            in_dim=self.hidden_size,
            hidden_dim=int(predictor_hidden_dim),
            dropout=float(predictor_dropout),
        )

    def trainable_params(self) -> List[torch.Tensor]:
        """Returns the parameters that currently need to optimize:"""
        out = [p for p in self.whisper.parameters() if p.requires_grad]
        out += [p for p in self.predictor.parameters() if p.requires_grad]
        return out

# Effective token mask and self-monitoring losses
def compute_valid_token_length_from_wav_len(wav_len_samples: int, mel_t: int) -> int:
    """Maps the `real and valid waveform sample points' into the token length of Whisper encoder."""
    if wav_len_samples <= 0:
        return 1
    n_mel_valid = int(math.ceil(wav_len_samples / 160.0))
    n_mel_valid = min(int(mel_t), max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)

def build_token_valid_mask_from_input_features(input_features: torch.Tensor, valid_wav_lens: torch.Tensor) -> torch.Tensor:
    """Construct a token level that is valid mask based on the true validity length of each set in the battling."""
    if not torch.is_tensor(valid_wav_lens):
        valid_wav_lens = torch.tensor(valid_wav_lens)

    valid_wav_lens = valid_wav_lens.detach().cpu().long().view(-1)
    B = int(input_features.size(0))
    mel_t = int(input_features.size(-1))
    tok_t = (mel_t + 1) // 2

    valid_ts = []
    for i in range(B):
        vt = compute_valid_token_length_from_wav_len(int(valid_wav_lens[i].item()), mel_t=mel_t)
        valid_ts.append(min(vt, tok_t))
    valid_ts = torch.tensor(valid_ts, device=input_features.device, dtype=torch.long)

    t = torch.arange(tok_t, device=input_features.device).view(1, tok_t).expand(B, tok_t)
    return (t < valid_ts.view(B, 1)).bool()

def token_mask_to_mel_mask(token_mask: torch.Tensor, mel_t: int) -> torch.Tensor:
    """Map token level mask onto the mel axis."""
    B, tok_t = token_mask.shape
    mel_mask = torch.zeros(B, mel_t, dtype=torch.bool, device=token_mask.device)
    for t in range(tok_t):
        left = int(2 * t)
        right = min(int(mel_t), int(2 * (t + 1)))
        if left >= mel_t:
            break
        mel_mask[:, left:right] |= token_mask[:, t].unsqueeze(1)
    return mel_mask

def make_time_mask_tok_valid(valid_mask_tok: torch.Tensor, mask_prob: float, mask_span: int) -> torch.Tensor:
    """Randomly generated span mask according to level token valid mask."""
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

        if not mask[b].any():
            pos = min(valid_len - 1, 0)
            mask[b, pos] = True

    mask = mask & valid_mask_tok
    return mask

def apply_input_time_mask(input_features: torch.Tensor, token_mask: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """To apply mask on the mel/input_features axis of Whisper."""
    x = input_features.clone()
    mel_mask = token_mask_to_mel_mask(token_mask, mel_t=int(x.size(-1)))
    x = x.masked_fill(mel_mask.unsqueeze(1), float(mask_value))
    return x

def get_encoder_hidden_from_model(model_encoder, input_features: torch.Tensor, target_layer: int) -> torch.Tensor:
    """Remove the specified layer hidden states from Whisper encoder."""
    out = model_encoder(
        input_features=input_features,
        return_dict=True,
        output_hidden_states=True,
    )
    hs = out.hidden_states
    if target_layer < 1 or target_layer >= len(hs):
        raise ValueError(f"target_layer={target_layer} illegal;hidden_states len={len(hs)}")
    return hs[int(target_layer)]

def masked_hidden_state_mse_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """returns each masked-token MSE at segment."""
    if pred.shape != target.shape:
        raise ValueError(f"pred/target Unmatched shape:{tuple(pred.shape)} vs {tuple(target.shape)}")
    if mask.dim() != 2:
        raise ValueError(f"mask The dimension should be [N,T], currently ={tuple(mask.shape)}")

    if pred.dim() != 3:
        raise ValueError(f"pred/target should read [N,T,D], current pred={tuple(pred.shape)}")

    N, T, D = pred.shape
    if mask.shape[0] != N or mask.shape[1] != T:
        raise ValueError(f"mask Shape {tuple(mask.shape)} With pred {tuple(pred.shape)} No match.")

    mask_f = mask.to(dtype=pred.dtype)

    token_mse = ((pred - target) ** 2).mean(dim=-1)

    denom = mask_f.sum(dim=1).clamp_min(1.0)
    per_sample_mse = (token_mse * mask_f).sum(dim=1) / denom

    return per_sample_mse

def audio_balanced_mean_from_segment_values(
    segment_values: torch.Tensor,
    segment_to_audio: torch.Tensor,
    num_audios: int,
) -> torch.Tensor:
    """Aggment-level values as audio-balanced catch mean."""
    segment_values = segment_values.view(-1)
    segment_to_audio = segment_to_audio.to(device=segment_values.device).long().view(-1)
    num_audios = int(num_audios)

    if segment_values.numel() != segment_to_audio.numel():
        raise ValueError(
            f"segment_values Number ={segment_values.numel()} and "
            f"segment_to_audio Number ={segment_to_audio.numel()} Inconsistent."
        )

    if num_audios <= 0:
        raise ValueError("num_audios Must be > 0.")

    if int(segment_to_audio.min().item()) < 0 or int(segment_to_audio.max().item()) >= num_audios:
        raise ValueError(
            f"segment_to_audio Exceeding: "
            f"min={int(segment_to_audio.min().item())}, "
            f"max={int(segment_to_audio.max().item())}, "
            f"num_audios={num_audios}"
        )

    audio_sum = segment_values.new_zeros((num_audios,))
    audio_sum.scatter_add_(0, segment_to_audio, segment_values)

    audio_count = segment_values.new_zeros((num_audios,))
    audio_count.scatter_add_(
        0,
        segment_to_audio,
        torch.ones_like(segment_values),
    )
    audio_count = audio_count.clamp_min(1.0)

    audio_mean = audio_sum / audio_count
    return audio_mean.mean()

def normalize_ssa_training_level(value: str) -> str:
    """Harmonize SSA_TRAINING_LEVEL writing."""
    level = str(value).strip().lower().replace("-", "_")

    if level in {"audio", "audio_level", "audio_balanced", "audio_balance"}:
        return "audio_balanced"

    if level in {"segment", "segment_level", "segment_mean", "segment_level_mean"}:
        return "segment"

    raise ValueError(
        f"Unknown SSA_TRAINING_LEVEL={value},"
        "Optional: 'audio_balanced' / 'segment'"
    )

def build_ssa_loss_formula() -> str:
    """Returns the text description for the current SSA loss to the following: experiment_config / manifest."""
    level = normalize_ssa_training_level(
        globals().get("SSA_TRAINING_LEVEL", "audio_balanced")
    )

    if level == "audio_balanced":
        return (
            "segment_loss_j = MSE(Predictor(H_masked_j)[mask], stopgrad(H_clean_j)[mask]); "
            "audio_loss_i = mean_j(segment_loss_j for same audio); "
            "batch_loss = mean_i(audio_loss_i)"
        )

    if level == "segment":
        return (
            "segment_loss_j = MSE(Predictor(H_masked_j)[mask], stopgrad(H_clean_j)[mask]); "
            "batch_loss = mean over all segment_loss_j in batch"
        )

    raise ValueError(f"Unknown SSA_TRAINING_LEVEL={level}")

def aggregate_ssa_segment_values(
    segment_values: torch.Tensor,
    segment_to_audio: Optional[torch.Tensor],
    num_audios: Optional[int],
    training_level: str,
) -> torch.Tensor:
    """Convergence under SSA_TRAINING_LEVEL segment-level loss."""
    training_level = normalize_ssa_training_level(training_level)
    segment_values = segment_values.view(-1)

    if training_level == "audio_balanced":
        if segment_to_audio is None or num_audios is None:
            raise ValueError("audio_balanced SSA Need segment_to_audio and num_audios.")

        return audio_balanced_mean_from_segment_values(
            segment_values=segment_values,
            segment_to_audio=segment_to_audio,
            num_audios=int(num_audios),
        )

    if training_level == "segment":
        return segment_values.mean()

    raise ValueError(f"Unknown SSA_TRAINING_LEVEL={training_level}")

@torch.no_grad()
def update_teacher_ema(student_encoder: nn.Module, teacher_encoder: nn.Module, momentum: float) -> None:
    """Update with EMA teacher encoder."""
    momentum = float(momentum)
    student_state = student_encoder.state_dict()
    teacher_state = teacher_encoder.state_dict()

    for k, v_teacher in teacher_state.items():
        v_student = student_state[k]
        if not torch.is_floating_point(v_teacher):
            v_teacher.copy_(v_student)
        else:
            v_teacher.mul_(momentum).add_(v_student.detach(), alpha=(1.0 - momentum))

def wave_batch_to_input_features(processor: WhisperProcessor, wav_batch: torch.Tensor) -> torch.Tensor:
    """For batch waveform read Whisper input_features"""
    wav_cpu = wav_batch.detach().cpu().numpy()
    feats = processor(
        wav_cpu,
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=int(processor.feature_extractor.n_samples),
    )
    return feats.input_features

# Current self-supervised training cycle
def run_one_epoch_ssa(
    model: WhisperSSAModel,
    processor: WhisperProcessor,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    target_layer: int,
    device: str,
    train_mode: bool,
    epoch: int,
    max_epochs: int,
) -> Dict[str, float]:
    """Run a whole SSA epoch."""
    training_level = normalize_ssa_training_level(
        globals().get("SSA_TRAINING_LEVEL", "audio_balanced")
    )

    if train_mode:
        model.train()
        model.teacher_encoder.eval()
    else:
        model.eval()
        model.teacher_encoder.eval()

    meter = {
        "total_loss": 0.0,
        "pred_loss": 0.0,
        "direct_loss": 0.0,
        "pred_masked_mse": 0.0,
        "direct_masked_mse": 0.0,
        "pred_unmasked_mse": 0.0,
        "num_batches": 0.0,
        "num_audios": 0.0,
        "num_segments": 0.0,
        "num_loss_units": 0.0,
        "num_masked_tokens": 0.0,
    }

    autocast_dtype = None
    if PRECISION == "fp16" and torch.cuda.is_available():
        autocast_dtype = torch.float16
    elif PRECISION == "bf16" and torch.cuda.is_available():
        autocast_dtype = torch.bfloat16

    scaler = None
    if train_mode and autocast_dtype == torch.float16 and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()

    phase = "SSA-Train" if train_mode else "SSA-Val"
    desc = f"{phase} [{training_level}] [Epoch {int(epoch)}/{int(max_epochs)}]"
    pbar = tqdm(loader, total=len(loader), desc=desc, dynamic_ncols=True)

    for batch in pbar:
        if training_level == "audio_balanced":
            (
                wav_30,
                valid_len,
                _audio_ids,
                _segment_ids,
                _wav_names,
                _wav_paths,
                segment_to_audio,
                segment_counts,
                _audio_keys,
                _audio_wav_names,
            ) = batch

            wav_30 = wav_30.to(device)
            valid_len = valid_len.to(device)
            segment_to_audio = segment_to_audio.to(device).long().view(-1)
            segment_counts = segment_counts.to(device).long().view(-1)

            num_segments = int(wav_30.shape[0])
            num_audios = int(segment_counts.numel())

            if num_segments <= 0 or num_audios <= 0:
                raise RuntimeError(
                    f"Illegal audio-balanced batch: "
                    f"num_segments={num_segments}, num_audios={num_audios}"
                )

            if int(segment_counts.sum().item()) != num_segments:
                raise RuntimeError(
                    f"segment_counts.sum()={int(segment_counts.sum().item())} "
                    f"With num_segments={num_segments} Inconsistent."
                )

            if segment_to_audio.numel() != num_segments:
                raise RuntimeError(
                    f"segment_to_audio Number ={segment_to_audio.numel()} "
                    f"With num_segments={num_segments} Inconsistent."
                )

            batch_loss_units = float(num_audios)

        elif training_level == "segment":
            (
                wav_30,
                valid_len,
                _audio_ids,
                _segment_ids,
                _wav_names,
                _wav_paths,
            ) = batch

            wav_30 = wav_30.to(device)
            valid_len = valid_len.to(device)

            num_segments = int(wav_30.shape[0])
            if num_segments <= 0:
                raise RuntimeError(f"Illegal segment batch: num_segments={num_segments}")

            segment_to_audio = None
            segment_counts = None

            try:
                if torch.is_tensor(_audio_ids):
                    num_audios = int(torch.unique(_audio_ids).numel())
                else:
                    num_audios = int(len(set([str(x) for x in list(_audio_ids)])))
            except Exception:
                num_audios = 0

            batch_loss_units = float(num_segments)

        else:
            raise ValueError(f"Unknown SSA_TRAINING_LEVEL={training_level}")

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        def _forward_once():
            clean_input = wave_batch_to_input_features(processor, wav_30).to(device)

            valid_mask_tok = build_token_valid_mask_from_input_features(
                clean_input,
                valid_len,
            )

            time_mask_tok = make_time_mask_tok_valid(
                valid_mask_tok,
                MASK_SPAN_RATIO,
                MASK_SPAN_LENGTH,
            )

            masked_input = apply_input_time_mask(
                clean_input,
                time_mask_tok,
                mask_value=MASK_VALUE,
            )

            with torch.no_grad():
                h_clean = get_encoder_hidden_from_model(
                    model.teacher_encoder,
                    clean_input,
                    target_layer=int(target_layer),
                ).detach()

            h_masked = get_encoder_hidden_from_model(
                model.whisper.encoder,
                masked_input,
                target_layer=int(target_layer),
            )

            h_pred = model.predictor(h_masked)

            pred_segment_loss = masked_hidden_state_mse_per_sample(
                pred=h_pred,
                target=h_clean,
                mask=time_mask_tok,
            )

            pred_loss = aggregate_ssa_segment_values(
                segment_values=pred_segment_loss,
                segment_to_audio=segment_to_audio,
                num_audios=num_audios if training_level == "audio_balanced" else None,
                training_level=training_level,
            )

            if float(SSA_DIRECT_LOSS_WEIGHT) > 0.0:
                direct_segment_loss = masked_hidden_state_mse_per_sample(
                    pred=h_masked,
                    target=h_clean,
                    mask=time_mask_tok,
                )
                direct_loss = aggregate_ssa_segment_values(
                    segment_values=direct_segment_loss,
                    segment_to_audio=segment_to_audio,
                    num_audios=num_audios if training_level == "audio_balanced" else None,
                    training_level=training_level,
                )
            else:
                direct_loss = pred_loss.detach() * 0.0

            total_loss = (
                float(SSA_PRED_LOSS_WEIGHT) * pred_loss
                + float(SSA_DIRECT_LOSS_WEIGHT) * direct_loss
            )

            unmasked_mask = valid_mask_tok & (~time_mask_tok)
            pred_unmasked_segment_mse = masked_hidden_state_mse_per_sample(
                pred=h_pred.detach(),
                target=h_clean.detach(),
                mask=unmasked_mask,
            )
            pred_unmasked_mse = aggregate_ssa_segment_values(
                segment_values=pred_unmasked_segment_mse,
                segment_to_audio=segment_to_audio,
                num_audios=num_audios if training_level == "audio_balanced" else None,
                training_level=training_level,
            )

            return {
                "total_loss": total_loss,
                "pred_loss": pred_loss,
                "direct_loss": direct_loss,
                "pred_masked_mse": pred_loss.detach(),
                "direct_masked_mse": direct_loss.detach(),
                "pred_unmasked_mse": pred_unmasked_mse.detach(),
                "num_masked_tokens": float(time_mask_tok.long().sum().item()),
            }

        if autocast_dtype is not None and torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                out = _forward_once()
        else:
            out = _forward_once()

        loss = out["total_loss"]

        if train_mode:
            if scaler is not None:
                scaler.scale(loss).backward()

                if float(GRAD_CLIP_NORM) > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.trainable_params(),
                        max_norm=float(GRAD_CLIP_NORM),
                    )

                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()

                if float(GRAD_CLIP_NORM) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.trainable_params(),
                        max_norm=float(GRAD_CLIP_NORM),
                    )

                optimizer.step()

            update_teacher_ema(
                student_encoder=model.whisper.encoder,
                teacher_encoder=model.teacher_encoder,
                momentum=float(EMA_MOMENTUM),
            )

        meter["total_loss"] += float(out["total_loss"].detach().item()) * batch_loss_units
        meter["pred_loss"] += float(out["pred_loss"].detach().item()) * batch_loss_units
        meter["direct_loss"] += float(out["direct_loss"].detach().item()) * batch_loss_units
        meter["pred_masked_mse"] += float(out["pred_masked_mse"].detach().item()) * batch_loss_units
        meter["direct_masked_mse"] += float(out["direct_masked_mse"].detach().item()) * batch_loss_units
        meter["pred_unmasked_mse"] += float(out["pred_unmasked_mse"].detach().item()) * batch_loss_units

        meter["num_batches"] += 1.0
        meter["num_audios"] += float(num_audios)
        meter["num_segments"] += float(num_segments)
        meter["num_loss_units"] += float(batch_loss_units)
        meter["num_masked_tokens"] += float(out["num_masked_tokens"])

        denom = max(1.0, meter["num_loss_units"])
        avg_total = meter["total_loss"] / denom
        avg_pred = meter["pred_loss"] / denom
        avg_direct = meter["direct_loss"] / denom
        avg_seg_per_audio = (
            meter["num_segments"] / max(1.0, meter["num_audios"])
            if meter["num_audios"] > 0
            else 0.0
        )

        pbar.set_postfix(
            total=f"{avg_total:.4f}",
            pred=f"{avg_pred:.4f}",
            direct=f"{avg_direct:.4f}",
            seg_per_audio=f"{avg_seg_per_audio:.2f}",
        )

    denom = max(1.0, meter["num_loss_units"])

    out = {
        "loss": meter["total_loss"] / denom,
        "masked_mse": meter["pred_masked_mse"] / denom,
        "unmasked_mse": meter["pred_unmasked_mse"] / denom,
        "num_masked_tokens": meter["num_masked_tokens"],

        "total_loss": meter["total_loss"] / denom,
        "pred_loss": meter["pred_loss"] / denom,
        "direct_loss": meter["direct_loss"] / denom,
        "pred_masked_mse": meter["pred_masked_mse"] / denom,
        "direct_masked_mse": meter["direct_masked_mse"] / denom,
        "pred_unmasked_mse": meter["pred_unmasked_mse"] / denom,
        "num_audios": meter["num_audios"],
        "num_segments": meter["num_segments"],
        "num_loss_units": meter["num_loss_units"],
        "mean_segments_per_audio": (
            meter["num_segments"] / max(1.0, meter["num_audios"])
            if meter["num_audios"] > 0
            else 0.0
        ),
        "ssa_training_level": str(training_level),
    }

    return out

def save_ssa_model_artifacts(
    model: WhisperSSAModel,
    processor: WhisperProcessor,
    model_dir: str,
    experiment_config: Dict[str, Any],
    ssa_metrics: List[Dict[str, Any]],
    run_manifest: Dict[str, Any],
) -> None:
    """Saves model products from SSA."""
    ensure_dir(model_dir)
    model.whisper.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)

    torch.save(model.predictor.state_dict(), os.path.join(model_dir, "ssa_prediction_head.pt"))
    save_json(experiment_config, os.path.join(model_dir, "experiment_config.json"))
    save_json({"history": ssa_metrics}, os.path.join(model_dir, "ssa_metrics.json"))
    write_csv_rows(os.path.join(model_dir, "ssa_metrics.csv"), ssa_metrics)
    save_json(run_manifest, os.path.join(model_dir, "run_manifest.json"))

def train_one_ssa_experiment(
    single_layer: int,
    fold_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Implement individual Whisper-SSA training official_train_test protocol."""
    run_config = apply_ssa_run_context(single_layer=single_layer, fold_id=None)

    train_entries_raw, test_entries_raw = load_official_train_test_entries(DATASET_KEY)
    mmse_maps = load_mmse_maps_for_dataset(DATASET_KEY)

    official_train_entries, missing_train = attach_mmse_and_filter(
        train_entries_raw,
        DATASET_KEY,
        mmse_maps,
        subset_name="train",
    )
    official_test_entries, missing_test = attach_mmse_and_filter(
        test_entries_raw,
        DATASET_KEY,
        mmse_maps,
        subset_name="test",
    )

    if len(official_train_entries) == 0:
        raise ValueError(f"{DATASET_KEY} MMSE Filter after official_train is empty.")

    print("[SSA][official_train_test]")
    print("  official_train raw   =", len(train_entries_raw))
    print("  official_train valid =", len(official_train_entries))
    print("  official_test raw    =", len(test_entries_raw))
    print("  official_test valid  =", len(official_test_entries))
    print("  missing_train        =", len(missing_train))
    print("  missing_test         =", len(missing_test))
    print("  fold disabled        = True")

    train_entries_seg = build_segment_entries_from_audio_entries(
        official_train_entries,
        split_name="train",
    )

    ssa_training_level = normalize_ssa_training_level(SSA_TRAINING_LEVEL)

    pin = torch.cuda.is_available()

    if ssa_training_level == "audio_balanced":
        train_dataset = AudioBalancedSegmentWaveDataset(train_entries_seg)

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=int(SSA_AUDIO_BATCH_SIZE),
            num_workers=int(NUM_WORKERS),
            drop_last=False,
            shuffle=True,
            collate_fn=train_dataset.repeat30s_collate,
            persistent_workers=(int(NUM_WORKERS) > 0),
            pin_memory=pin,
        )

        print(
            "[SSA audio-balanced] "
            f"num_train_audio={len(train_dataset)}, "
            f"num_train_segments={len(train_entries_seg)}, "
            f"SSA_AUDIO_BATCH_SIZE={int(SSA_AUDIO_BATCH_SIZE)}"
        )

    elif ssa_training_level == "segment":
        train_dataset = SegmentWaveDataset(
            train_entries_seg,
            require_target=False,
        )

        train_loader = data.DataLoader(
            train_dataset,
            batch_size=int(SSA_BATCH_SIZE),
            num_workers=int(NUM_WORKERS),
            drop_last=False,
            shuffle=True,
            collate_fn=train_dataset.repeat30s_collate,
            persistent_workers=(int(NUM_WORKERS) > 0),
            pin_memory=pin,
        )

        print(
            "[SSA segment-level] "
            f"num_train_segments={len(train_dataset)}, "
            f"SSA_BATCH_SIZE={int(SSA_BATCH_SIZE)}"
        )

    else:
        raise ValueError(f"Unknown SSA_TRAINING_LEVEL={ssa_training_level}")
    processor = WhisperProcessor.from_pretrained(
        BASE_WHISPER_MODEL,
        local_files_only=LOCAL_FILES_ONLY,
    )

    model = WhisperSSAModel(
        base_model_name=BASE_WHISPER_MODEL,
        unfreeze_encoder_layers=RESOLVED_UNFREEZE_ENCODER_LAYERS,
        predictor_hidden_dim=int(PREDICTOR_HIDDEN_DIM),
        predictor_dropout=float(PREDICTOR_DROPOUT),
        local_files_only=LOCAL_FILES_ONLY,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.trainable_params(),
        lr=float(SSA_lr),
        weight_decay=float(WEIGHT_DECAY),
    )

    history: List[Dict[str, Any]] = []

    t_start = time.time()
    train_metrics = None

    for epoch in range(1, int(SSA_NUM_EPOCHS) + 1):
        print("\n" + "-" * 90)
        print(
            f"[SSA] Epoch {int(epoch)}/{int(SSA_NUM_EPOCHS)} | "
            f"dataset={DATASET_KEY} | "
            f"layer={int(single_layer)} | "
            f"unfree={int(UNFREEZE_ENCODER_LAYERS)} | "
            f"protocol={RUN_PROTOCOL_TAG}"
        )
        print("-" * 90)

        train_metrics = run_one_epoch_ssa(
            model=model,
            processor=processor,
            loader=train_loader,
            optimizer=optimizer,
            target_layer=int(single_layer),
            device=DEVICE,
            train_mode=True,
            epoch=int(epoch),
            max_epochs=int(SSA_NUM_EPOCHS),
        )

        row = {
            "epoch": int(epoch),

            "train_loss": float(train_metrics["loss"]),
            "train_masked_mse": float(train_metrics["masked_mse"]),
            "train_unmasked_mse": float(train_metrics["unmasked_mse"]),
            "train_num_masked_tokens": float(train_metrics["num_masked_tokens"]),

            "train_total_loss": float(train_metrics["total_loss"]),
            "train_pred_loss": float(train_metrics["pred_loss"]),
            "train_direct_loss": float(train_metrics["direct_loss"]),
            "train_pred_masked_mse": float(train_metrics["pred_masked_mse"]),
            "train_direct_masked_mse": float(train_metrics["direct_masked_mse"]),
            "train_num_audios": float(train_metrics.get("num_audios", 0.0)),
            "train_num_segments": float(train_metrics.get("num_segments", 0.0)),
            "train_mean_segments_per_audio": float(train_metrics.get("mean_segments_per_audio", 0.0)),
            "ssa_training_level": str(normalize_ssa_training_level(SSA_TRAINING_LEVEL)),

            "val_loss": None,
            "val_masked_mse": None,
            "val_unmasked_mse": None,
            "val_num_masked_tokens": None,

            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)

        if ENABLE_NOTEBOOK_LOG:
            write_csv_rows(os.path.join(NOTEBOOK_LOG_ROOT, "metrics.csv"), history)

    train_seconds = float(time.time() - t_start)

    best_epoch = int(SSA_NUM_EPOCHS)
    best_val_loss = None
    best_train_metrics = copy.deepcopy(train_metrics)
    best_val_metrics = None

    best_student_state = copy.deepcopy(model.whisper.state_dict())
    best_teacher_state = copy.deepcopy(model.teacher_encoder.state_dict())
    best_predictor_state = copy.deepcopy(model.predictor.state_dict())

    model.whisper.load_state_dict(best_student_state, strict=True)
    model.teacher_encoder.load_state_dict(best_teacher_state, strict=True)
    model.predictor.load_state_dict(best_predictor_state, strict=True)

    experiment_config = {
        "PROJECT_ROOT": PROJECT_ROOT,
        "DATASET_KEY": DATASET_KEY,
        "DATASET_KEYS": list(DATASET_KEYS),
        "MODEL_NAME": MODEL_NAME,
        "FEATURE_PREFIX": FEATURE_PREFIX,
        "BASE_WHISPER_MODEL": BASE_WHISPER_MODEL,
        "LOCAL_FILES_ONLY": bool(LOCAL_FILES_ONLY),
        "RUN_PROTOCOL_TAG": str(RUN_PROTOCOL_TAG),
        "PITT_ENABLE_SUBJECT_DISJOINT_SPLIT": bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
        "PITT_SPLIT_POLICY": str(PITT_SPLIT_POLICY),
        "PITT_EXPECTED_VALID_VISITS": int(PITT_EXPECTED_VALID_VISITS),
        "PITT_EXPECTED_UNIQUE_SUBJECTS": int(PITT_EXPECTED_UNIQUE_SUBJECTS),

        "SINGLE_LAYERS": list(SINGLE_LAYERS),
        "SINGLE_LAYER": int(single_layer),

        "FOLD_IDS": [],
        "FOLD_ID": None,
        "FOLD_POLICY": "disabled",

        "UNFREEZE_ENCODER_LAYERS_LIST": list(UNFREEZE_ENCODER_LAYERS_LIST),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),
        "RESOLVED_UNFREEZE_ENCODER_LAYERS": list(RESOLVED_UNFREEZE_ENCODER_LAYERS),

        "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
        "RESOLVED_PATH_PARAM_TAGS": resolve_path_param_tags(PATH_PARAM_TAGS),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(PATH_PARAM_TAGS),
        "LAYER_NAME": str(ACTIVE_LAYER_NAME),

        "TRAIN_VAL_SPLIT_PATH": None,
        "SSA_MODEL_DIR": str(SSA_MODEL_DIR),
        "FEATURE_SAVE_PATH": str(FEATURE_SAVE_PATH),
        "LOCAL_FEATURE_SPLIT_SAVE_PATH": str(LOCAL_FEATURE_SPLIT_SAVE_PATH),

        "SAMPLING_RATE": int(SAMPLING_RATE),
        "SEGMENT_SEC": float(SEGMENT_SEC),
        "OVERLAP_RATE": float(OVERLAP_RATE),
        "ENABLE_AUDIO_NORMALIZE": bool(ENABLE_AUDIO_NORMALIZE),

        "SSA_BATCH_SIZE": int(SSA_BATCH_SIZE),
        "SSA_TRAINING_LEVEL": str(normalize_ssa_training_level(SSA_TRAINING_LEVEL)),
        "SSA_AUDIO_BATCH_SIZE": int(SSA_AUDIO_BATCH_SIZE),
        "SSA_LOSS_FORMULA": build_ssa_loss_formula(),
        "SSA_NUM_EPOCHS": int(SSA_NUM_EPOCHS),
        "SSA_lr": float(SSA_lr),
        "WEIGHT_DECAY": float(WEIGHT_DECAY),
        "GRAD_CLIP_NORM": float(GRAD_CLIP_NORM),
        "MASK_SPAN_RATIO": float(MASK_SPAN_RATIO),
        "MASK_SPAN_LENGTH": int(MASK_SPAN_LENGTH),
        "MASK_VALUE": float(MASK_VALUE),
        "EMA_MOMENTUM": float(EMA_MOMENTUM),
        "PREDICTOR_HIDDEN_DIM": int(PREDICTOR_HIDDEN_DIM),
        "PREDICTOR_DROPOUT": float(PREDICTOR_DROPOUT),
        "SSA_LOSS_TYPE": str(SSA_LOSS_TYPE),
        "SSA_MSE_WEIGHT": float(SSA_MSE_WEIGHT),
        "SSA_COSINE_WEIGHT": float(SSA_COSINE_WEIGHT),
        "SSA_PRED_LOSS_WEIGHT": float(SSA_PRED_LOSS_WEIGHT),
        "SSA_DIRECT_LOSS_WEIGHT": float(SSA_DIRECT_LOSS_WEIGHT),

        "NORM_FOR_SINGLE": NORM_FOR_SINGLE,
        "DROP_INPUT_LAYER": bool(DROP_INPUT_LAYER),
        "SEED": int(SEED),
        "DEVICE": str(DEVICE),
        "NUM_WORKERS": int(NUM_WORKERS),
        "PRECISION": str(PRECISION),

        "num_official_train_raw": int(len(train_entries_raw)),
        "num_official_train_valid": int(len(official_train_entries)),
        "num_official_test_raw": int(len(test_entries_raw)),
        "num_official_test_valid": int(len(official_test_entries)),
        "num_missing_train_mmse": int(len(missing_train)),
        "num_missing_test_mmse": int(len(missing_test)),
        "num_train_segments": int(len(train_entries_seg)),
    }

    run_manifest = {
        "stage": f"whisper_ssa_{normalize_ssa_training_level(SSA_TRAINING_LEVEL)}",
        "ssa_training_level": str(normalize_ssa_training_level(SSA_TRAINING_LEVEL)),
        "dataset_key": DATASET_KEY,
        "single_layer": int(single_layer),
        "layer_name": str(ACTIVE_LAYER_NAME),
        "fold_id": None,
        "fold_policy": "disabled",
        "run_protocol": str(RUN_PROTOCOL_TAG),

        "train_val_split_path": None,
        "ssa_model_dir": str(SSA_MODEL_DIR),
        "feature_save_path": str(FEATURE_SAVE_PATH),
        "local_feature_split_save_path": str(LOCAL_FEATURE_SPLIT_SAVE_PATH),

        "num_train_dataset_items": int(len(train_dataset)),
        "num_train_segments": int(len(train_entries_seg)),
        "ssa_batch_size": int(SSA_BATCH_SIZE),
        "ssa_audio_batch_size": int(SSA_AUDIO_BATCH_SIZE),
        "ssa_loss_formula": build_ssa_loss_formula(),
        "checkpoint_selection": "last",
        "upstream_uses_fold_val": False,
        "best_epoch": int(best_epoch),
        "best_val_loss": None,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": None,
        "train_seconds": float(train_seconds),
    }

    save_ssa_model_artifacts(
        model=model,
        processor=processor,
        model_dir=SSA_MODEL_DIR,
        experiment_config=experiment_config,
        ssa_metrics=history,
        run_manifest=run_manifest,
    )

    torch.save(best_teacher_state, os.path.join(SSA_MODEL_DIR, "teacher_encoder_ema.pt"))

    save_json({"entries": official_train_entries}, os.path.join(SSA_MODEL_DIR, "official_train_entries.json"))
    save_json({"entries": official_test_entries}, os.path.join(SSA_MODEL_DIR, "official_test_entries.json"))
    save_json(
        {
            "missing_train": missing_train,
            "missing_test": missing_test,
        },
        os.path.join(SSA_MODEL_DIR, "missing_mmse_records.json"),
    )
    save_text(str(SSA_MODEL_DIR), os.path.join(SSA_MODEL_DIR, "ssa_model_dir.txt"))

    print(f"[SSA] Completed:{SSA_MODEL_DIR}")
    print("  - protocol             =", RUN_PROTOCOL_TAG)
    print("  - checkpoint_selection = last")
    print("  - selected_epoch       =", best_epoch)
    print("  - best_val_loss        = None (no-val-last)")

    return {
        "run_config": run_config,
        "experiment_config": experiment_config,
        "run_manifest": run_manifest,
    }

# Normalize Characterization and Export
def apply_norm(h: torch.Tensor, mode: Optional[str], ln_post=None, skip_ln_post: bool = False):
    """Normalize [1,T,C] features."""
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
            raise ValueError("ln_post is None,But chose ln_post/whisper_ln.")
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
    sampling_rate: int = SAMPLING_RATE,
    overlap_rate: float = OVERLAP_RATE,
):
    """To press 30s + overlap for individual wav and to draw a specified single layer feature."""
    y = load_audio_16k(audio_file)

    step_size = int(segment_length * (1 - overlap_rate))
    if step_size <= 0:
        raise ValueError(f"overlap_rate={overlap_rate} Caused step_size<=0")

    total_segments = int(len(y) / step_size) + (
        1 if len(y) % step_size > overlap_rate * segment_length else 0
    )
    total_segments = max(1, total_segments)

    seg_features: List[torch.Tensor] = []
    seg_indices: List[Tuple[int, int]] = []
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
            raise ValueError(f"single_layer={single_layer} Beyond range, currently available layer ={len(hs)}")

        is_top = (int(single_layer) == len(hs))
        if (not warned_double_ln_top and is_top and already_post and norm_single in ("ln_post", "whisper_ln")):
            print(f"[WARN] {Path(audio_file).name}: The top level has been post-LN and the external level has been set ln_post, skipping a second ln_post.")
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

def stack_with_time_padding(feature_list: List[torch.Tensor], target_t: Optional[int] = None, pad_value: float = 0.0) -> torch.Tensor:
    """Fill a set of [T,C] features to a uniform time length, and then stop into [N,T,C]."""
    if len(feature_list) == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)

    for i, feat in enumerate(feature_list):
        if feat.ndim != 2:
            raise ValueError(f"feature_list[{i}] Not a 2-dimensional length. Current shape ={tuple(feat.shape)}")

    feat_dim = int(feature_list[0].shape[1])
    for i, feat in enumerate(feature_list):
        if int(feat.shape[1]) != feat_dim:
            raise ValueError(f"feature_list[{i}] Features inconsistent:{feat.shape[1]} != {feat_dim}")

    if target_t is None:
        target_t = max(int(feat.shape[0]) for feat in feature_list)

    padded = []
    for feat in feature_list:
        t = int(feat.shape[0])
        if t < target_t:
            pad = torch.full((target_t - t, feat_dim), float(pad_value), dtype=torch.float32)
            feat = torch.cat([feat.float(), pad], dim=0)
        elif t > target_t:
            feat = feat[:target_t].float()
        else:
            feat = feat.float()
        padded.append(feat)

    return torch.stack(padded, dim=0)  # [N,T,C]

def export_subset_features(
    entries: List[Dict[str, Any]],
    subset_name: str,
    processor: WhisperProcessor,
    whisper_model: WhisperModel,
    single_layer: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """Export single layer features for official train or official test."""
    all_feats: List[torch.Tensor] = []
    all_labels: List[float] = []
    all_indices: List[Tuple[int, int]] = []
    audio_manifest: List[Dict[str, Any]] = []

    ln_post = getattr(whisper_model.encoder, "layer_norm", None)
    total_layers = get_whisper_encoder_layer_count(SSA_MODEL_DIR)

    pbar = tqdm(entries, total=len(entries), desc=f"Export {subset_name} Features", dynamic_ncols=True)
    for row in pbar:
        wav_path = canonical_wav_path(row["wav_path"])
        wav_name = str(row.get("wav_name", Path(wav_path).name))
        audio_id = int(row["audio_id"])
        target_mmse = float(row["target_mmse"])

        feats, inds = get_segment_features(
            audio_file=wav_path,
            audio_file_index=audio_id,
            processor=processor,
            model=whisper_model,
            single_layer=int(single_layer),
            norm_single=NORM_FOR_SINGLE,
            ln_post=ln_post,
            top_layer_index=int(total_layers),
            drop_input_layer=bool(DROP_INPUT_LAYER),
            segment_length=SEGMENT_LENGTH,
            sampling_rate=SAMPLING_RATE,
            overlap_rate=OVERLAP_RATE,
        )

        for f in feats:
            all_feats.append(f)
            all_labels.append(float(target_mmse))
        for x in inds:
            all_indices.append((int(x[0]), int(x[1])))

        audio_manifest.append({
            "audio_id": int(audio_id),
            "wav_name": wav_name,
            "wav_path": wav_path,
            "group": str(row.get("group", "")),
            "split": str(row.get("split", subset_name)),
            "target_mmse": float(target_mmse),
            "num_segments": int(len(feats)),
        })

    feature_tensor = stack_with_time_padding(all_feats)
    labels_tensor = torch.tensor(all_labels, dtype=torch.float32)
    indices_tensor = torch.tensor(all_indices, dtype=torch.long)

    return feature_tensor, labels_tensor, indices_tensor, audio_manifest

def write_local_train_val_split_json(
    save_root: str,
    file_tag: str,
    official_train_entries: List[Dict[str, Any]],
    official_test_entries: List[Dict[str, Any]],
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
) -> str:
    """Save official_train_test metadata in the feature directory."""
    ensure_dir(save_root)

    train_rows = []
    for row in official_train_entries:
        wav_path = canonical_wav_path(row["wav_path"])
        train_rows.append({
            "audio_id": int(row["audio_id"]),
            "official_audio_id": int(row.get("official_audio_id", row["audio_id"])),
            "wav_name": str(row.get("wav_name", Path(wav_path).name)),
            "wav_path": wav_path,
            "group": str(row.get("group", "")),
            "target_mmse": float(row["target_mmse"]),
            "outer_split": "train",
        })

    test_rows = []
    for row in official_test_entries:
        wav_path = canonical_wav_path(row["wav_path"])
        test_rows.append({
            "audio_id": int(row["audio_id"]),
            "official_audio_id": int(row.get("official_audio_id", row["audio_id"])),
            "wav_name": str(row.get("wav_name", Path(wav_path).name)),
            "wav_path": wav_path,
            "group": str(row.get("group", "")),
            "target_mmse": float(row["target_mmse"]),
            "outer_split": "test",
        })

    obj = {
        "format_version": 2,
        "dataset_key": DATASET_KEY,
        "file_tag": str(file_tag),
        "feature_model_name": f"ft_{MODEL_NAME}",
        "selection_name": str(ACTIVE_LAYER_NAME),
        "run_protocol": str(RUN_PROTOCOL_TAG),
        "fold_id": None,
        "fold_policy": "disabled",
        "train_semantics": (
            "subject_disjoint_final_train"
            if str(DATASET_KEY) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT)
            else "official_train"
        ),
        "test_semantics": (
            "subject_disjoint_final_test_export_only_not_used_by_upstream_ssa"
            if str(DATASET_KEY) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT)
            else "official_test_export_only_not_used_by_upstream_ssa"
        ),
        "split_strategy": (
            PITT_SPLIT_POLICY
            if str(DATASET_KEY) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT)
            else "predefined_official_train_test"
        ),
        "subject_disjoint": (
            True if str(DATASET_KEY) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT) else None
        ),
        "train_test_subject_overlap": (
            0 if str(DATASET_KEY) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT) else None
        ),
        "train_entries": train_rows,
        "test_entries": test_rows,
        "train_manifest": train_manifest,
        "test_manifest": test_manifest,
    }

    local_split_path = os.path.join(
        save_root,
        f"official_train_test_metadata_{file_tag}.json",
    )
    save_json(obj, local_split_path)
    return local_split_path

def write_pitt_lru_audio_lookup_sidecar(
    save_root: str,
    file_tag: str,
    train_entries: List[Dict[str, Any]],
    test_entries: List[Dict[str, Any]],
) -> Optional[str]:
    """Additionally write for Pitt SSA features with the same agreement as the current Pitt subject-disjoint protocol:"""
    if str(DATASET_KEY) != "Pitt" or not bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return None

    def _row(row: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        subject_id = row.get("subject_id")
        visit_index = row.get("visit_index")
        if subject_id is None or visit_index is None:
            subject_id, visit_index = _extract_pitt_subject_visit(Path(row["wav_path"]).stem)
        if subject_id is None or visit_index is None:
            raise RuntimeError(f"[Pitt sidecar] Unable to parse subject/visit: {row['wav_path']}")
        return {
            "audio_id": int(row["audio_id"]),
            "split": str(split_name),
            "wav_path": canonical_wav_path(row["wav_path"]),
            "wav_name": str(row.get("wav_name", Path(row["wav_path"]).name)),
            "subject_id": str(subject_id),
            "visit_index": int(visit_index),
            "group": str(row.get("group", "")),
            "target_mmse": float(row["target_mmse"]),
        }

    train_rows = [_row(row, "train") for row in train_entries]
    test_rows = [_row(row, "test") for row in test_entries]

    train_subjects = {r["subject_id"] for r in train_rows}
    test_subjects = {r["subject_id"] for r in test_rows}
    overlap = train_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"[Pitt sidecar] train/test subject overlap != 0: {sorted(overlap)}")

    payload = {
        "format_version": 3,
        "task_type": "mmse_regression",
        "dataset": "Pitt",
        "paper_dataset": paper_dataset_name("Pitt"),
        "split_strategy": PITT_SPLIT_POLICY,
        "subject_disjoint": True,
        "train_test_subject_overlap": 0,
        "train_audio_ids": [int(x) for x in range(len(train_rows))],
        "val_audio_ids": [],
        "train_entries": train_rows,
        "test_entries": test_rows,
    }

    sidecar_path = os.path.join(save_root, f"train_val_split_{file_tag}.json")
    save_json(payload, sidecar_path)
    print(f"[Pitt split] feature audio-id sidecar -> {sidecar_path}")
    return sidecar_path

def export_features_for_one_run(
    single_layer: int,
    fold_id: Optional[int],
    training_summary: Dict[str, Any],
) -> Dict[str, Any]:
    """Export official_train / official_test features."""
    file_tag = make_default_split_file_tag(
        feature_prefix=FEATURE_PREFIX,
        single_layer=int(single_layer),
        overlap_rate=OVERLAP_RATE,
        segment_sec=SEGMENT_SEC,
    )

    train_entries_raw, test_entries_raw = load_official_train_test_entries(DATASET_KEY)
    mmse_maps = load_mmse_maps_for_dataset(DATASET_KEY)

    official_train_entries, missing_train = attach_mmse_and_filter(
        train_entries_raw,
        DATASET_KEY,
        mmse_maps,
        subset_name="train",
    )
    official_test_entries, missing_test = attach_mmse_and_filter(
        test_entries_raw,
        DATASET_KEY,
        mmse_maps,
        subset_name="test",
    )

    if len(official_train_entries) == 0 or len(official_test_entries) == 0:
        raise ValueError(
            f"MMSE Filter official train/test empty:"
            f"train={len(official_train_entries)}, test={len(official_test_entries)}"
        )

    processor = WhisperProcessor.from_pretrained(SSA_MODEL_DIR, local_files_only=True)
    whisper_model = WhisperModel.from_pretrained(SSA_MODEL_DIR, local_files_only=True).to(DEVICE)
    whisper_model.eval()

    feature_train, labels_train, indices_train, train_manifest = export_subset_features(
        entries=official_train_entries,
        subset_name="train",
        processor=processor,
        whisper_model=whisper_model,
        single_layer=int(single_layer),
    )
    feature_test, labels_test, indices_test, test_manifest = export_subset_features(
        entries=official_test_entries,
        subset_name="test",
        processor=processor,
        whisper_model=whisper_model,
        single_layer=int(single_layer),
    )

    ensure_dir(FEATURE_SAVE_PATH)

    paths = {
        "feature_train": os.path.join(FEATURE_SAVE_PATH, f"feature_{file_tag}_train.pt"),
        "feature_test": os.path.join(FEATURE_SAVE_PATH, f"feature_{file_tag}_test.pt"),
        "labels_train": os.path.join(FEATURE_SAVE_PATH, f"labels1_{file_tag}_train.pt"),
        "labels_test": os.path.join(FEATURE_SAVE_PATH, f"labels2_{file_tag}_test.pt"),
        "indices_train": os.path.join(FEATURE_SAVE_PATH, f"indices_{file_tag}_train.pt"),
        "indices_test": os.path.join(FEATURE_SAVE_PATH, f"indices_{file_tag}_test.pt"),
    }

    torch.save(feature_train, paths["feature_train"])
    torch.save(feature_test, paths["feature_test"])
    torch.save(labels_train, paths["labels_train"])
    torch.save(labels_test, paths["labels_test"])
    torch.save(indices_train, paths["indices_train"])
    torch.save(indices_test, paths["indices_test"])

    local_metadata_path = write_local_train_val_split_json(
        save_root=FEATURE_SAVE_PATH,
        file_tag=file_tag,
        official_train_entries=official_train_entries,
        official_test_entries=official_test_entries,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
    )

    pitt_lru_sidecar_path = write_pitt_lru_audio_lookup_sidecar(
        save_root=FEATURE_SAVE_PATH,
        file_tag=file_tag,
        train_entries=official_train_entries,
        test_entries=official_test_entries,
    )
    local_split_path = pitt_lru_sidecar_path or local_metadata_path

    if SAVE_META:
        save_json({"items": train_manifest}, os.path.join(FEATURE_SAVE_PATH, "train_manifest.json"))
        save_json({"items": test_manifest}, os.path.join(FEATURE_SAVE_PATH, "test_manifest.json"))
        save_json(
            {
                "missing_train": missing_train,
                "missing_test": missing_test,
            },
            os.path.join(FEATURE_SAVE_PATH, "missing_mmse_records.json"),
        )

        feature_meta = {
            "stage": f"whisper_ssa_{normalize_ssa_training_level(SSA_TRAINING_LEVEL)}_export",
            "dataset_key": DATASET_KEY,
            "fold_id": None,
            "fold_policy": "disabled",
            "run_protocol": str(RUN_PROTOCOL_TAG),
            "single_layer": int(single_layer),
            "layer_name": str(ACTIVE_LAYER_NAME),
            "file_tag": str(file_tag),
            "feature_root": str(FEATURE_SAVE_PATH),
            "feature_model_name": f"ft_{MODEL_NAME}",
            "feature_selection_name": str(ACTIVE_LAYER_NAME),
            "ssa_model_dir": str(SSA_MODEL_DIR),
            "source_fold_json_path": None,
            "lru_recommended_config": {
                "FEATURE_SOURCE_MODES": ["finetune"],
                "FT_MODEL_NAME": str(MODEL_NAME),
                "SINGLE_LAYERS": [int(single_layer)],
                "UNFREE_LAYER_NUM": [int(UNFREEZE_ENCODER_LAYERS)],
                "DATASET_OPTION": str(DATASET_KEY),
                "FOLD_IDS": [],
                "RUN_PROTOCOL_TAG": str(RUN_PROTOCOL_TAG),
                "path_param_tags": list(PATH_PARAM_TAGS),
                "resolved_path_param_tags": resolve_path_param_tags(PATH_PARAM_TAGS),
            },
            "semantics": {
                "feature_train_pt": (
                    "pitt_subject_disjoint_final_train_all_mmse_valid"
                    if str(DATASET_KEY) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                    else "official_train_all_mmse_valid"
                ),
                "feature_test_pt": (
                    "pitt_subject_disjoint_final_test_all_mmse_valid"
                    if str(DATASET_KEY) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                    else "official_test_all_mmse_valid"
                ),
                "outer_split_for_lru": "fold disabled; downstream should use official_train_test protocol",
            },
            "training_summary": training_summary.get("run_manifest", {}),
        }
        save_json(feature_meta, os.path.join(FEATURE_SAVE_PATH, "feature_meta.json"))
        save_text(str(SSA_MODEL_DIR), os.path.join(FEATURE_SAVE_PATH, "ssa_model_dir.txt"))

        export_manifest = {
            "stage": f"whisper_ssa_{normalize_ssa_training_level(SSA_TRAINING_LEVEL)}_feature_export",
            "dataset_key": DATASET_KEY,
            "fold_id": None,
            "fold_policy": "disabled",
            "run_protocol": str(RUN_PROTOCOL_TAG),
            "single_layer": int(single_layer),
            "layer_name": str(ACTIVE_LAYER_NAME),
            "feature_save_path": str(FEATURE_SAVE_PATH),
            "file_tag": str(file_tag),
            "num_train_audios": int(len(official_train_entries)),
            "num_test_audios": int(len(official_test_entries)),
            "num_train_segments": int(feature_train.shape[0]),
            "num_test_segments": int(feature_test.shape[0]),
            "local_split_path": str(local_split_path),
            "local_metadata_path": str(local_metadata_path),
            "pitt_lru_sidecar_path": (None if pitt_lru_sidecar_path is None else str(pitt_lru_sidecar_path)),
            "pitt_subject_disjoint": bool(str(DATASET_KEY) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
        }
        save_json(export_manifest, os.path.join(FEATURE_SAVE_PATH, "export_manifest.json"))
        save_json(
            training_summary.get("experiment_config", {}),
            os.path.join(FEATURE_SAVE_PATH, "experiment_config.json"),
        )

    print(f"[SAVE] {FEATURE_SAVE_PATH}")
    if str(DATASET_KEY) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT:
        print("  semantics: train=Pitt subject-disjoint final train, test=Pitt subject-disjoint final test")
    else:
        print("  semantics: train=official_train_all_mmse_valid, test=official_test_all_mmse_valid")
    for k, p in paths.items():
        print(f"  - {k}: {p}")

    return {
        "feature_save_path": str(FEATURE_SAVE_PATH),
        "file_tag": str(file_tag),
        "num_train_segments": int(feature_train.shape[0]),
        "num_test_segments": int(feature_test.shape[0]),
        "num_train_audios": int(len(official_train_entries)),
        "num_test_audios": int(len(official_test_entries)),
        "local_split_path": str(local_split_path),
        "local_metadata_path": str(local_metadata_path),
        "pitt_lru_sidecar_path": (None if pitt_lru_sidecar_path is None else str(pitt_lru_sidecar_path)),
        "pitt_subject_disjoint": bool(str(DATASET_KEY) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
    }

# Single experiments and batch entrances
def run_experiment_ssa(single_layer: int) -> Dict[str, Any]:
    """Complete individual processes dataset / unfreeze / single_layer:"""
    print("\n" + "=" * 100)
    print(f"[RUN] DATASET_KEY                  = {DATASET_KEY}")
    print(f"[RUN] MODEL_NAME                   = {MODEL_NAME}")
    print(f"[RUN] BASE_WHISPER_MODEL           = {BASE_WHISPER_MODEL}")
    print(f"[RUN] SINGLE_LAYER                 = {single_layer}")
    print(f"[RUN] UNFREEZE_ENCODER_LAYERS      = {UNFREEZE_ENCODER_LAYERS}")
    print(f"[RUN] PATH_PARAM_TAGS              = {PATH_PARAM_TAGS}")
    print(f"[RUN] RUN_PROTOCOL_TAG             = {RUN_PROTOCOL_TAG}")
    print(f"[RUN] FOLD_ID                      = None")
    print("=" * 100)

    training_summary = train_one_ssa_experiment(
        single_layer=single_layer,
        fold_id=None,
    )

    export_summary = None
    if not bool(SKIP_FEATURE_EXPORT):
        export_summary = export_features_for_one_run(
            single_layer=single_layer,
            fold_id=None,
            training_summary=training_summary,
        )

    return {
        "dataset_key": str(DATASET_KEY),
        "single_layer": int(single_layer),
        "unfreeze_encoder_layers": int(UNFREEZE_ENCODER_LAYERS),
        "fold_id": None,
        "layer_name": str(ACTIVE_LAYER_NAME),
        "ssa_model_dir": str(SSA_MODEL_DIR),
        "feature_save_path": None if export_summary is None else export_summary["feature_save_path"],
        "best_epoch": int(training_summary["run_manifest"]["best_epoch"]),
        "best_val_loss": training_summary["run_manifest"]["best_val_loss"],
        "num_train_segments": None if export_summary is None else int(export_summary["num_train_segments"]),
        "num_test_segments": None if export_summary is None else int(export_summary["num_test_segments"]),
        "num_train_audios": None if export_summary is None else int(export_summary["num_train_audios"]),
        "num_test_audios": None if export_summary is None else int(export_summary["num_test_audios"]),
        "run_manifest_path": os.path.join(SSA_MODEL_DIR, "run_manifest.json"),
        "feature_export": export_summary,
    }

def run_all_experiments_ssa() -> None:
    """Main entrance."""
    global DATASET_KEY
    global UNFREEZE_ENCODER_LAYERS

    seed_everything_for_repro(SEED)

    dataset_keys = [str(x) for x in DATASET_KEYS]
    unfreeze_list = [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]

    if len(dataset_keys) == 0:
        raise ValueError("DATASET_KEYS There is a minimum need for a data set to be empty.")
    if len(unfreeze_list) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST To be empty, a minimum number of unfrozen layers is required.")

    for dataset_key in dataset_keys:
        if dataset_key not in DATASETS:
            raise ValueError(f"Unknown DATASET_KEY={dataset_key}")
        if dataset_key not in MMSE_META_PATHS:
            raise ValueError(f"Current script not configured {dataset_key} MMSE metadata from")

    resolved_layers = resolve_single_layers(SINGLE_LAYERS, BASE_WHISPER_MODEL)
    summary_rows: List[Dict[str, Any]] = []

    print("[CONFIG]")
    print("  PROJECT_ROOT                 =", PROJECT_ROOT)
    print("  DATASET_KEYS                 =", dataset_keys)
    print("  MODEL_NAME                   =", MODEL_NAME)
    print("  BASE_WHISPER_MODEL           =", BASE_WHISPER_MODEL)
    print("  SINGLE_LAYERS                =", resolved_layers)
    print("  FOLD_IDS                     = []  # disabled")
    print("  RUN_PROTOCOL_TAG             =", RUN_PROTOCOL_TAG)
    print("  UNFREEZE_ENCODER_LAYERS_LIST =", unfreeze_list)
    print("  PATH_PARAM_TAGS              =", PATH_PARAM_TAGS)
    print("  SAMPLING_RATE                =", SAMPLING_RATE)
    print("  SEGMENT_SEC                  =", SEGMENT_SEC)
    print("  OVERLAP_RATE                 =", OVERLAP_RATE)
    print("  SSA_AUDIO_BATCH_SIZE        =", SSA_AUDIO_BATCH_SIZE)
    print("  SSA_NUM_EPOCHS              =", SSA_NUM_EPOCHS)
    print("  SSA_lr                      =", SSA_lr)
    print("  SSA_TRAINING_LEVEL          =", SSA_TRAINING_LEVEL)
    print("  MASK_SPAN_RATIO              =", MASK_SPAN_RATIO)
    print("  MASK_SPAN_LENGTH             =", MASK_SPAN_LENGTH)
    print("  EMA_MOMENTUM                 =", EMA_MOMENTUM)
    print("  NORM_FOR_SINGLE              =", NORM_FOR_SINGLE)
    print("  SKIP_FEATURE_EXPORT          =", SKIP_FEATURE_EXPORT)
    print("  DEVICE                       =", DEVICE)

    for dataset_key in dataset_keys:
        for unfreeze_count in unfreeze_list:
            DATASET_KEY = str(dataset_key)
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "#" * 120)
            print("[BATCH DATASET / UNFREEZE]")
            print("  DATASET_KEY =", DATASET_KEY)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("  SSA_TRAINING_LEVEL =", SSA_TRAINING_LEVEL)
            print("#" * 120)

            for single_layer in [int(x) for x in resolved_layers]:
                print("\n" + "=" * 120)
                print(
                    f"[BATCH] START | "
                    f"dataset={DATASET_KEY} | "
                    f"single_layer={int(single_layer)} | "
                    f"unfreeze={int(UNFREEZE_ENCODER_LAYERS)} | "
                    f"protocol={RUN_PROTOCOL_TAG}"
                )
                print("=" * 120)

                seed_everything_for_repro(SEED)
                print(f"[REPRO] reset seed before run -> {SEED}")

                row = run_experiment_ssa(single_layer=single_layer)
                summary_rows.append(row)

    summary_dir = os.path.join(
        FEATURE_ROOT_DIR,
        "_batch_summaries",
        f"ft_{MODEL_NAME}",
    )
    ensure_dir(summary_dir)

    summary_path_json = os.path.join(summary_dir, "batch_run_summary_multi_dataset_unfreeze.json")
    summary_path_csv = os.path.join(summary_dir, "batch_run_summary_multi_dataset_unfreeze.csv")

    save_json(
        {
            "format_version": 2,
            "dataset_keys": dataset_keys,
            "model_name": MODEL_NAME,
            "run_protocol": str(RUN_PROTOCOL_TAG),
            "fold_ids": [],
            "fold_policy": "disabled",
            "train_semantics": (
                "subject_disjoint_final_train"
                if dataset_keys == ["Pitt"] and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_train"
            ),
            "test_semantics": (
                "subject_disjoint_final_test_export_only_not_used_by_upstream_ssa"
                if dataset_keys == ["Pitt"] and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_test_export_only_not_used_by_upstream_ssa"
            ),
            "pitt_split_policy": (
                PITT_SPLIT_POLICY if "Pitt" in dataset_keys and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT else None
            ),
            "ssa_training_level": str(normalize_ssa_training_level(SSA_TRAINING_LEVEL)),
            "ssa_loss_formula": build_ssa_loss_formula(),
            "single_layers": resolved_layers,
            "unfreeze_encoder_layers_list": unfreeze_list,
            "runs": summary_rows,
        },
        summary_path_json,
    )
    write_csv_rows(summary_path_csv, summary_rows)

    print("\n" + "=" * 100)
    print("[ALL DONE]")
    print("batch summary json:", summary_path_json)
    print("batch summary csv :", summary_path_csv)
    print("Output root directory example:")
    print("  -", os.path.join(FEATURE_ROOT_DIR, dataset_keys[0], f"ft_{MODEL_NAME}"))
    print("Examples of downstream suggested settings:")
    print('  FEATURE_SOURCE_MODES = ["finetune"]')
    print(f'  FT_MODEL_NAME = "{MODEL_NAME}"')
    print(f"  SINGLE_LAYERS = {resolved_layers}")
    print(f"  UNFREE_LAYER_NUM = {unfreeze_list}")
    print("  FOLD_IDS = []  # official_train_test protocol")
    print("=" * 100)
if __name__ == "__main__":
    run_all_experiments_ssa()
