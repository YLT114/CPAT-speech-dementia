#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-stage VanillaFT MMSE regression baseline
================================

Standard supervised fine-tuning baseline:

1. Fine-tune only on official train; validation and official test are not used for model selection.
2. Fully unfreeze the Whisper encoder and train it end to end with a linear regression head.
3. Mean-pool frame-level hidden states from the selected Whisper layer.
4. Predict one MMSE score per segment with the linear head.
5. Average segment predictions per recording and compute audio-level MAE.
6. Train for ``NUM_EPOCHS`` and save the final Whisper model and regression head.
7. Export train/test features with the VanillaFT naming scheme.

Processes ADReSS-IS2020, ADReSSo21, Pitt, and PROCESS-2 in order by default. Pitt uses the
VanillaFT subject-disjoint protocol; PROCESS-2 reads only ``__CTD.wav`` files.

Environment:
    conda activate whisper_vanillaft
    python VanillaFT.py
"""


try:
    from .paper_names import paper_dataset_name
except ImportError:
    from paper_names import paper_dataset_name

# Set these variables before importing torch.
import os

GPU_ID = 5
if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

os.environ["PYTHONHASHSEED"] = "400"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import gc
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import WhisperConfig, WhisperModel, WhisperProcessor


# =============================================================================
# 1. Experiment configuration
# =============================================================================

SEED = 400

PROJECT_ROOT = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT, "feature_tensor")
MODEL_ROOT_DIR = os.path.join(PROJECT_ROOT, "models")

BASE_WHISPER_MODEL = "/home/user/whisper_lru_3/whisper_small"
LOCAL_FILES_ONLY = True

# Datasets
DATASET_KEYS = ["ADReSS-IS2020"]

# Use layer 12 features and unfreeze the last seven encoder layers.
SINGLE_LAYERS = [12]
UNFREEZE_ENCODER_LAYERS_LIST = [7]

FEATURE_MODEL_SAVE_TAG = "vanilla_baseline_lr3e-5_epo5"
RUN_PROTOCOL_TAG = "official_train_only_last_checkpoint"

NUM_EPOCHS = 5
AUDIO_BATCH_SIZE = 2
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
HEAD_DROPOUT = 0.2

TARGET_NORMALIZE = True
TARGET_NORMALIZE_EPS = 1e-6
LOSS_TYPE = "mae"

SAMPLING_RATE = 16000
SEGMENT_SEC = 30
SEGMENT_LENGTH = int(SEGMENT_SEC * SAMPLING_RATE)
OVERLAP_RATE = 0.3
MAX_SEGMENTS_PER_AUDIO = 0  # Zero disables the limit.
NORMALIZE_AUDIO = True

FEATURE_PREFIX = "whisper"
NORM_FOR_SINGLE = "ln_post"
EXPORT_FEATURES = True

# Use the VanillaFT subject-disjoint protocol and integrity checks for Pitt.
PITT_SPLIT_POLICY = "original_test_subjects_holdout"
PITT_STRICT_COUNT_CHECK = True
PITT_EXPECTED_CANDIDATE_VISITS = 477
PITT_EXPECTED_VALID_VISITS = 398
PITT_EXPECTED_UNIQUE_SUBJECTS = 252
PITT_EXPECTED_VALID_TEST_SUBJECTS = 68


DATASETS: Dict[str, Dict[str, str]] = {
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


# =============================================================================
# 2. Reproducibility and utilities
# =============================================================================


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Fail on nondeterministic operations to preserve reproducibility.
    torch.use_deterministic_algorithms(True)


def save_json(payload: object, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def natural_key(path_value: str) -> List[object]:
    name = Path(path_value).name
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.findall(r"\d+|\D+", name)
    ]


def canonical_wav_path(wav_path: str) -> str:
    return str(Path(wav_path).expanduser().resolve())


def is_process2_ctd_wav(wav_path: str, dataset_key: str) -> bool:
    if dataset_key != "PROCESS-2":
        return True
    return re.search(r"__CTD$", Path(wav_path).stem, flags=re.IGNORECASE) is not None


def list_wav_files(audio_dir: str, dataset_key: str) -> List[str]:
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

    files = [
        str(Path(audio_dir) / name)
        for name in os.listdir(audio_dir)
        if name.lower().endswith(".wav")
        and is_process2_ctd_wav(str(Path(audio_dir) / name), dataset_key)
    ]
    files.sort(key=natural_key)
    if not files:
        raise RuntimeError(f"No eligible WAV files found in directory: {audio_dir}")
    return files


def get_whisper_encoder_layer_count() -> int:
    config = WhisperConfig.from_pretrained(
        BASE_WHISPER_MODEL,
        local_files_only=LOCAL_FILES_ONLY,
    )
    return int(config.encoder_layers)


def resolve_unfreeze_layers(
    single_layer: int,
    total_layers: int,
    unfreeze_layer_count: int,
) -> List[int]:
    single_layer = int(single_layer)
    total_layers = int(total_layers)
    unfreeze_layer_count = int(unfreeze_layer_count)

    if not 1 <= single_layer <= total_layers:
        raise ValueError(f"single_layer={single_layer} is outside [1, {total_layers}]")
    if unfreeze_layer_count < 0:
        raise ValueError("unfreeze_layer_count must be >= 0")
    if unfreeze_layer_count == 0:
        return []

    start = max(1, single_layer - unfreeze_layer_count + 1)
    return list(range(start, single_layer + 1))


def split_segment_boundaries(
    audio_len: int,
    segment_length: int,
    overlap_rate: float,
) -> List[Tuple[int, int]]:
    step = int(segment_length * (1.0 - overlap_rate))
    if step <= 0:
        raise ValueError("overlap_rate produces a segment stride <= 0")

    total = int(audio_len / step)
    if (audio_len % step) > overlap_rate * segment_length:
        total += 1
    total = max(total, 1)
    return [(index * step, index * step + segment_length) for index in range(total)]


def make_file_tag(single_layer: int) -> str:
    overlap = f"{OVERLAP_RATE:.1f}"
    return f"{FEATURE_PREFIX}_single{single_layer}_{overlap}_{SEGMENT_SEC}s"


def build_run_paths(
    dataset_key: str,
    single_layer: int,
    unfreeze_count: int,
) -> Dict[str, str]:
    layer_name = f"single{single_layer}_unfree{unfreeze_count}"
    feature_dir = os.path.join(
        FEATURE_ROOT_DIR,
        dataset_key,
        f"ft_{FEATURE_MODEL_SAVE_TAG}",
        layer_name,
    )
    checkpoint_dir = os.path.join(
        MODEL_ROOT_DIR,
        dataset_key,
        f"ft_{FEATURE_MODEL_SAVE_TAG}",
        layer_name,
        RUN_PROTOCOL_TAG,
    )
    return {
        "layer_name": layer_name,
        "feature_dir": feature_dir,
        "checkpoint_dir": checkpoint_dir,
    }


# =============================================================================
# 3. MMSE metadata
# =============================================================================


def pick_column(dataframe: pd.DataFrame, candidates: Sequence[str]):
    normalized = {str(column).strip().lower(): column for column in dataframe.columns}
    for candidate in candidates:
        if str(candidate).strip().lower() in normalized:
            return normalized[str(candidate).strip().lower()]
    raise KeyError(f"Could not find any of the columns {list(candidates)}; available columns: {list(dataframe.columns)}")


def safe_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.upper() in {"NA", "N/A", "NAN", "NONE"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def extract_pitt_subject_visit(raw_name) -> Tuple[Optional[str], Optional[int]]:
    if raw_name is None or pd.isna(raw_name):
        return None, None
    match = re.match(
        r"^(\d+)-(\d+)(?:c)?(?:-|$)",
        Path(str(raw_name)).stem,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None, None
    return str(int(match.group(1))), int(match.group(2))


def pitt_mmse_column(visit_index: int) -> str:
    if int(visit_index) < 0:
        raise ValueError(f"Invalid Pitt visit_index={visit_index}")
    return "mms" if int(visit_index) == 0 else f"mmse{int(visit_index) + 1}"


def normalize_mmse_key(dataset_key: str, raw_id) -> Optional[str]:
    if raw_id is None or pd.isna(raw_id):
        return None
    text = Path(str(raw_id).strip()).stem
    if dataset_key == "PROCESS-2":
        text = re.sub(r"__(SFT|PFT|CTD)$", "", text, flags=re.IGNORECASE)
    return text.lower()


def dataframe_to_mmse_map(
    dataframe: pd.DataFrame,
    dataset_key: str,
    id_column: str,
    mmse_column: str,
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for _, row in dataframe.iterrows():
        key = normalize_mmse_key(dataset_key, row[id_column])
        value = safe_float(row[mmse_column])
        if key is not None and value is not None:
            result[key] = float(value)
    return result


def read_adress_is2020_mmse(config: Dict[str, str]):
    def read_text(path: str) -> Dict[str, float]:
        result = {}
        with open(path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line or line.startswith("ID"):
                    continue
                parts = [item.strip() for item in line.split(";")]
                key = normalize_mmse_key("ADReSS-IS2020", parts[0])
                value = safe_float(parts[-1])
                if key is not None and value is not None:
                    result[key] = float(value)
        return result

    train_cc = read_text(config["train_cc"])
    train_cd = read_text(config["train_cd"])
    test = read_text(config["test_all"])
    return {
        "train_cc": train_cc,
        "train_cd": train_cd,
        "test_cc": test,
        "test_cd": test,
    }


def read_adresso21_mmse(config: Dict[str, str]):
    train = pd.read_csv(config["train_all"])
    test = pd.read_csv(config["test_all"])

    train_id = pick_column(train, ["adressfname", "addressfname", "id"])
    train_mmse = pick_column(train, ["mmse"])
    train_diagnosis = pick_column(train, ["dx", "label", "group"])
    test_id = pick_column(test, ["id", "adressfname", "addressfname"])
    test_mmse = pick_column(test, ["mmse"])

    diagnosis = train[train_diagnosis].astype(str).str.strip().str.lower()
    train_cc_rows = train[diagnosis.isin(["cn", "control", "cc"])]
    train_cd_rows = train[diagnosis.isin(["ad", "probablead", "cd"])]

    train_cc = dataframe_to_mmse_map(train_cc_rows, "ADReSSo21", train_id, train_mmse)
    train_cd = dataframe_to_mmse_map(train_cd_rows, "ADReSSo21", train_id, train_mmse)
    test_map = dataframe_to_mmse_map(test, "ADReSSo21", test_id, test_mmse)
    return {
        "train_cc": train_cc,
        "train_cd": train_cd,
        "test_cc": test_map,
        "test_cd": test_map,
    }


def read_pitt_mmse(config: Dict[str, object]):
    dataframe = pd.read_excel(
        config["all_meta"],
        sheet_name=config.get("sheet_name", "data"),
        header=int(config.get("header_row", 2)),
    )
    id_column = pick_column(dataframe, [str(config.get("id_col", "id")), "id"])
    columns = {str(column).strip().lower(): column for column in dataframe.columns}
    result: Dict[Tuple[str, int], float] = {}

    for _, row in dataframe.iterrows():
        raw_id = row[id_column]
        if raw_id is None or pd.isna(raw_id):
            continue
        match = re.search(r"\d+", str(raw_id))
        if match is None:
            continue
        subject_id = str(int(match.group()))
        for visit_index in range(7):
            mmse_name = pitt_mmse_column(visit_index).lower()
            if mmse_name not in columns:
                continue
            value = safe_float(row[columns[mmse_name]])
            if value is not None:
                result[(subject_id, visit_index)] = float(value)

    return {
        "train_cc": result,
        "train_cd": result,
        "test_cc": result,
        "test_cd": result,
    }


def read_process2_mmse(config: Dict[str, str]):
    dataframe = pd.read_csv(config["meta"])
    id_column = pick_column(dataframe, ["IDs", "ID", "id", "dir_name"])
    diagnosis_column = pick_column(dataframe, ["diagnosis", "dx", "label"])
    mmse_column = pick_column(dataframe, ["MMSE", "mmse"])
    split_column = pick_column(dataframe, ["Split", "split", "partition"])

    split = dataframe[split_column].astype(str).str.strip().str.lower()
    diagnosis = dataframe[diagnosis_column].astype(str).str.strip().str.lower()
    train_mask = split.isin(["train", "training"])
    test_mask = split.isin(["test", "testing"])
    cc_mask = diagnosis.isin(
        ["hc", "healthy", "healthy control", "control", "cn", "cc"]
    )
    cd_mask = diagnosis.isin(["mci", "dementia", "ad", "probablead", "cd"])

    subsets = {
        "train_cc": dataframe[train_mask & cc_mask],
        "train_cd": dataframe[train_mask & cd_mask],
        "test_cc": dataframe[test_mask & cc_mask],
        "test_cd": dataframe[test_mask & cd_mask],
    }
    return {
        name: dataframe_to_mmse_map(rows, "PROCESS-2", id_column, mmse_column)
        for name, rows in subsets.items()
    }


def load_mmse_maps(dataset_key: str):
    if dataset_key == "ADReSS-IS2020":
        result = read_adress_is2020_mmse(MMSE_META_PATHS[dataset_key])
    elif dataset_key == "ADReSSo21":
        result = read_adresso21_mmse(MMSE_META_PATHS[dataset_key])
    elif dataset_key == "Pitt":
        result = read_pitt_mmse(MMSE_META_PATHS[dataset_key])
    elif dataset_key == "PROCESS-2":
        result = read_process2_mmse(MMSE_META_PATHS[dataset_key])
    else:
        raise ValueError(f"Unsupported dataset: {dataset_key}")

    print(
        f"[MMSE] {dataset_key}: "
        + ", ".join(f"{name}={len(values)}" for name, values in result.items())
    )
    return result


def get_mmse(dataset_key: str, wav_path: str, mmse_map) -> Optional[float]:
    if dataset_key == "Pitt":
        subject_id, visit_index = extract_pitt_subject_visit(Path(wav_path).stem)
        if subject_id is None or visit_index is None:
            return None
        return mmse_map.get((subject_id, visit_index))

    key = normalize_mmse_key(dataset_key, Path(wav_path).stem)
    return None if key is None else mmse_map.get(key)


def missing_mmse_record(
    dataset_key: str, subset: str, wav_path: str
) -> Dict[str, object]:
    result: Dict[str, object] = {
        "dataset_key": dataset_key,
        "subset_name": subset,
        "wav_name": Path(wav_path).name,
        "wav_path": canonical_wav_path(wav_path),
        "reason": "missing_mmse",
    }
    if dataset_key == "Pitt":
        subject_id, visit_index = extract_pitt_subject_visit(Path(wav_path).stem)
        result["subject_id"] = subject_id
        result["visit_index"] = visit_index
        if subject_id is None or visit_index is None:
            result["reason"] = "unparsed_pitt_subject_visit"
        else:
            result["expected_mmse_col"] = pitt_mmse_column(visit_index)
    return result


# =============================================================================
# 4. Pitt subject-disjoint split
# =============================================================================


def build_pitt_subject_disjoint_files(
    dataset_paths: Dict[str, str],
    mmse_map: Dict[Tuple[str, int], float],
):
    if PITT_SPLIT_POLICY != "original_test_subjects_holdout":
        raise ValueError(f"Unsupported PITT_SPLIT_POLICY={PITT_SPLIT_POLICY}")

    records = []
    missing = []
    seen = {}
    candidate_count = 0

    for source_subset in ["train_cc", "train_cd", "test_cc", "test_cd"]:
        source_split, group = source_subset.split("_", 1)
        files = list_wav_files(dataset_paths[source_subset], "Pitt")
        candidate_count += len(files)

        for wav_path in files:
            subject_id, visit_index = extract_pitt_subject_visit(Path(wav_path).stem)
            visit_key = (subject_id, visit_index)
            if subject_id is None or visit_index is None or visit_key not in mmse_map:
                missing.append(missing_mmse_record("Pitt", source_subset, wav_path))
                continue
            if visit_key in seen:
                raise RuntimeError(
                    f"Duplicate Pitt subject-visit {visit_key}: {seen[visit_key]} / {wav_path}"
                )
            seen[visit_key] = wav_path
            records.append(
                {
                    "subject_id": subject_id,
                    "visit_index": int(visit_index),
                    "mmse": float(mmse_map[visit_key]),
                    "group": group,
                    "source_split": source_split,
                    "source_subset": source_subset,
                    "wav_name": Path(wav_path).name,
                    "wav_path": canonical_wav_path(wav_path),
                }
            )

    valid_subjects = {row["subject_id"] for row in records}
    test_subjects = {
        row["subject_id"] for row in records if row["source_split"] == "test"
    }
    train_subjects = valid_subjects - test_subjects
    if train_subjects & test_subjects:
        raise RuntimeError("The Pitt subject-disjoint split still contains overlapping subjects")

    files_by_subset = {
        name: [] for name in ["train_cc", "train_cd", "test_cc", "test_cd"]
    }
    for row in records:
        final_split = "test" if row["subject_id"] in test_subjects else "train"
        final_subset = f"{final_split}_{row['group']}"
        row["final_split"] = final_split
        row["final_subset"] = final_subset
        files_by_subset[final_subset].append(row["wav_path"])

    for subset in files_by_subset:
        files_by_subset[subset].sort(key=natural_key)

    checks = [
        ("candidate_visits", candidate_count, PITT_EXPECTED_CANDIDATE_VISITS),
        ("valid_visits", len(records), PITT_EXPECTED_VALID_VISITS),
        ("unique_subjects", len(valid_subjects), PITT_EXPECTED_UNIQUE_SUBJECTS),
        ("valid_test_subjects", len(test_subjects), PITT_EXPECTED_VALID_TEST_SUBJECTS),
    ]
    for name, actual, expected in checks:
        if int(actual) != int(expected):
            message = f"Pitt {name}={actual}, expected={expected}"
            if PITT_STRICT_COUNT_CHECK:
                raise RuntimeError(message)
            print("[WARN]", message)

    visit_dataframe = pd.DataFrame(records).sort_values(
        ["final_split", "subject_id", "visit_index", "wav_name"],
        kind="stable",
    )
    subject_rows = []
    for subject_id in sorted(valid_subjects, key=int):
        rows = [row for row in records if row["subject_id"] == subject_id]
        source_splits = {row["source_split"] for row in rows}
        subject_rows.append(
            {
                "subject_id": subject_id,
                "original_membership": (
                    "both"
                    if source_splits == {"train", "test"}
                    else ("test" if source_splits == {"test"} else "train")
                ),
                "final_split": "test" if subject_id in test_subjects else "train",
                "valid_visit_count": len(rows),
                "cc_visit_count": sum(row["group"] == "cc" for row in rows),
                "cd_visit_count": sum(row["group"] == "cd" for row in rows),
            }
        )

    split_info = {
        "split_strategy": PITT_SPLIT_POLICY,
        "subject_disjoint": True,
        "train_test_subject_overlap": 0,
        "candidate_visits": candidate_count,
        "valid_visits": len(records),
        "unique_subjects": len(valid_subjects),
        "train_subjects": len(train_subjects),
        "test_subjects": len(test_subjects),
    }
    return (
        files_by_subset,
        visit_dataframe,
        pd.DataFrame(subject_rows),
        missing,
        split_info,
    )


def save_pitt_manifests(visit_dataframe: pd.DataFrame, subject_dataframe: pd.DataFrame):
    root = os.path.join(FEATURE_ROOT_DIR, "Pitt", "subject_disjoint_split")
    os.makedirs(root, exist_ok=True)
    visit_path = os.path.join(root, "pitt_valid_visits_subject_disjoint_manifest.csv")
    subject_path = os.path.join(root, "pitt_subject_assignment.csv")
    visit_dataframe.to_csv(visit_path, index=False, encoding="utf-8-sig")
    subject_dataframe.to_csv(subject_path, index=False, encoding="utf-8-sig")
    return visit_path, subject_path


# =============================================================================
# 5. Training data and model
# =============================================================================


@dataclass
class AudioRecord:
    wav_path: str
    target: float
    audio_id: int
    boundaries: List[Tuple[int, int]]


def load_audio(wav_path: str) -> np.ndarray:
    audio, _ = librosa.load(
        wav_path,
        sr=SAMPLING_RATE,
        mono=True,
        dtype=np.float32,
    )
    if NORMALIZE_AUDIO:
        peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
        if peak > 1e-8:
            audio = audio / peak
    return audio


def pad_audio_segments(
    audio: np.ndarray,
    boundaries: Sequence[Tuple[int, int]],
) -> np.ndarray:
    """Generate fixed-length segments using the VanillaFT padding rule."""

    segments = []
    step = int(SEGMENT_LENGTH * (1.0 - OVERLAP_RATE))
    for index, (start, end) in enumerate(boundaries):
        segment = audio[start:end]
        if len(segment) < SEGMENT_LENGTH and index > 0:
            need = SEGMENT_LENGTH - len(segment)
            previous_end = start + SEGMENT_LENGTH - step
            previous_start = max(0, previous_end - need)
            previous = audio[previous_start:previous_end]
            if len(previous) < need:
                previous = np.pad(previous, (0, need - len(previous)))
            segment = np.concatenate([segment, previous])
        if len(segment) < SEGMENT_LENGTH:
            segment = np.pad(segment, (0, SEGMENT_LENGTH - len(segment)))
        segments.append(segment[:SEGMENT_LENGTH].astype(np.float32))

    if not segments:
        raise RuntimeError("Audio contains no valid segments")
    return np.stack(segments)


def load_padded_segments(record: AudioRecord) -> np.ndarray:
    audio = load_audio(record.wav_path)
    return pad_audio_segments(audio, record.boundaries)


class AudioDataset(Dataset):
    def __init__(self, records: Sequence[AudioRecord]):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        return load_padded_segments(record), np.float32(record.target), record.audio_id


def build_records(
    wav_files: Sequence[str],
    dataset_key: str,
    subset: str,
    mmse_map,
    start_audio_id: int,
):
    records = []
    audio_id = int(start_audio_id)
    missing = []

    for wav_path in tqdm(wav_files, desc=f"[{dataset_key}] build records", ncols=100):
        target = get_mmse(dataset_key, wav_path, mmse_map)
        if target is None:
            missing.append(missing_mmse_record(dataset_key, subset, wav_path))
            continue
        audio = load_audio(wav_path)
        boundaries = split_segment_boundaries(len(audio), SEGMENT_LENGTH, OVERLAP_RATE)
        if MAX_SEGMENTS_PER_AUDIO > 0:
            boundaries = boundaries[:MAX_SEGMENTS_PER_AUDIO]
        records.append(
            AudioRecord(
                wav_path=wav_path,
                target=float(target),
                audio_id=audio_id,
                boundaries=boundaries,
            )
        )
        audio_id += 1
    return records, audio_id, missing


def make_collate(processor: WhisperProcessor):
    def collate(batch):
        segments = []
        targets = []
        audio_ids = []
        counts = []
        for audio_segments, target, audio_id in batch:
            counts.append(int(audio_segments.shape[0]))
            targets.append(float(target))
            audio_ids.append(int(audio_id))
            segments.extend(list(audio_segments))

        processed = processor(
            segments,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
            padding=True,
        )
        return (
            processed.input_features,
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(audio_ids, dtype=torch.long),
            torch.tensor(counts, dtype=torch.long),
        )

    return collate


def select_hidden_state(encoder_output, layer_index: int):
    hidden_states = list(encoder_output.hidden_states)
    hidden_states = hidden_states[1:] if len(hidden_states) >= 2 else hidden_states
    if not 1 <= int(layer_index) <= len(hidden_states):
        raise ValueError(f"layer_index={layer_index} is outside [1, {len(hidden_states)}]")
    return hidden_states[int(layer_index) - 1]


class VanillaWhisperRegressor(nn.Module):
    def __init__(self, single_layer: int):
        super().__init__()
        self.whisper = WhisperModel.from_pretrained(
            BASE_WHISPER_MODEL,
            local_files_only=LOCAL_FILES_ONLY,
        )
        self.single_layer = int(single_layer)
        self.dropout = nn.Dropout(HEAD_DROPOUT)
        self.regressor = nn.Linear(int(self.whisper.config.d_model), 1)

    def forward(self, input_features: torch.Tensor):
        encoder_output = self.whisper.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        selected = select_hidden_state(encoder_output, self.single_layer)
        segment_embedding = selected.mean(dim=1)
        return self.regressor(self.dropout(segment_embedding)).squeeze(-1)


def configure_trainable_layers(
    model: VanillaWhisperRegressor,
    layer_indices: Sequence[int],
) -> List[int]:
    for parameter in model.whisper.parameters():
        parameter.requires_grad = False

    total_layers = len(model.whisper.encoder.layers)
    resolved = sorted(
        {int(index) for index in layer_indices if 1 <= int(index) <= total_layers}
    )
    if resolved == list(range(1, total_layers + 1)):
        # Standard VanillaFT trains the convolutional frontend, all Transformer layers, and final layer normalization.
        # Keep Whisper positional embeddings frozen as in the Hugging Face default.
        for module in [
            model.whisper.encoder.conv1,
            model.whisper.encoder.conv2,
            model.whisper.encoder.layers,
            model.whisper.encoder.layer_norm,
        ]:
            for parameter in module.parameters():
                parameter.requires_grad = True
    else:
        for index in resolved:
            for parameter in model.whisper.encoder.layers[index - 1].parameters():
                parameter.requires_grad = True
        if resolved:
            for parameter in model.whisper.encoder.layer_norm.parameters():
                parameter.requires_grad = True
    for parameter in model.regressor.parameters():
        parameter.requires_grad = True
    return resolved


def aggregate_segment_predictions(
    segment_predictions: torch.Tensor,
    segment_counts: torch.Tensor,
) -> torch.Tensor:
    outputs = []
    start = 0
    for count in segment_counts.tolist():
        end = start + int(count)
        outputs.append(segment_predictions[start:end].mean())
        start = end
    if start != int(segment_predictions.numel()):
        raise RuntimeError("segment_counts does not match the number of segment predictions")
    return torch.stack(outputs)


def compute_target_normalizer(records: Sequence[AudioRecord]):
    values = np.asarray([record.target for record in records], dtype=np.float64)
    if not TARGET_NORMALIZE:
        return {"enabled": False, "mean": 0.0, "std": 1.0, "eps": TARGET_NORMALIZE_EPS}
    std = max(float(values.std()), float(TARGET_NORMALIZE_EPS))
    return {
        "enabled": True,
        "mean": float(values.mean()),
        "std": std,
        "eps": float(TARGET_NORMALIZE_EPS),
        "n_audio": int(len(values)),
    }


def normalize_targets(targets: torch.Tensor, normalizer: Dict[str, object]):
    if not normalizer["enabled"]:
        return targets
    return (targets - float(normalizer["mean"])) / float(normalizer["std"])


def denormalize_predictions(predictions: torch.Tensor, normalizer: Dict[str, object]):
    if not normalizer["enabled"]:
        return predictions
    return predictions * float(normalizer["std"]) + float(normalizer["mean"])


def regression_metrics(predictions: torch.Tensor, targets: torch.Tensor):
    predictions = predictions.float().cpu()
    targets = targets.float().cpu()
    error = predictions - targets
    mae = float(error.abs().mean())
    rmse = float(torch.sqrt((error**2).mean()))
    if (
        predictions.numel() >= 2
        and float(predictions.std()) > 0
        and float(targets.std()) > 0
    ):
        pearson = float(torch.corrcoef(torch.stack([predictions, targets]))[0, 1])
    else:
        pearson = 0.0
    denominator = float(((targets - targets.mean()) ** 2).sum())
    r2 = 0.0 if denominator <= 0 else 1.0 - float((error**2).sum()) / denominator
    return {"mae": mae, "rmse": rmse, "pearson": pearson, "r2": r2}


def train_model(
    model: VanillaWhisperRegressor,
    processor: WhisperProcessor,
    records: Sequence[AudioRecord],
    device: torch.device,
):
    normalizer = compute_target_normalizer(records)
    generator = torch.Generator()
    generator.manual_seed(SEED)
    loader = DataLoader(
        AudioDataset(records),
        batch_size=AUDIO_BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=make_collate(processor),
        generator=generator,
    )

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("The model has no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        audio_count = 0
        all_predictions = []
        all_targets = []

        for input_features, targets, _, segment_counts in tqdm(
            loader,
            desc=f"train epoch {epoch}/{NUM_EPOCHS}",
            ncols=100,
        ):
            input_features = input_features.to(device)
            targets = targets.to(device)
            segment_counts = segment_counts.to(device)

            optimizer.zero_grad(set_to_none=True)
            segment_predictions = model(input_features)
            audio_predictions_normalized = aggregate_segment_predictions(
                segment_predictions,
                segment_counts,
            )
            normalized_targets = normalize_targets(targets, normalizer)

            if LOSS_TYPE == "mae":
                loss = F.l1_loss(audio_predictions_normalized, normalized_targets)
            elif LOSS_TYPE == "mse":
                loss = F.mse_loss(audio_predictions_normalized, normalized_targets)
            else:
                raise ValueError(f"Unsupported LOSS_TYPE={LOSS_TYPE}")

            loss.backward()
            if GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    GRAD_CLIP_NORM,
                )
            optimizer.step()

            batch_size = int(targets.numel())
            loss_sum += loss.item() * batch_size
            audio_count += batch_size
            all_predictions.append(
                denormalize_predictions(
                    audio_predictions_normalized.detach(), normalizer
                ).cpu()
            )
            all_targets.append(targets.detach().cpu())

        metrics = regression_metrics(torch.cat(all_predictions), torch.cat(all_targets))
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(audio_count, 1),
            **{f"train_{name}": value for name, value in metrics.items()},
        }
        history.append(row)
        print("[TRAIN]", row)

    return {
        "history": history,
        "selected_checkpoint": "last",
        "selected_epoch": NUM_EPOCHS,
        "used_validation": False,
        "target_normalizer": normalizer,
    }


# =============================================================================
# 6. Feature export
# =============================================================================


def apply_feature_norm(
    hidden: torch.Tensor,
    mode: str,
    layer_norm: Optional[nn.Module],
    skip_layer_norm: bool,
):
    if mode in {None, "none", "None"}:
        return hidden
    if mode == "layernorm":
        return F.layer_norm(hidden, hidden.shape[-1:])
    if mode == "std":
        return hidden / hidden.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    if mode == "ln_post":
        if skip_layer_norm:
            return hidden
        if layer_norm is None:
            raise ValueError("ln_post mode requires layer_norm")
        return layer_norm(hidden)
    raise ValueError(f"Unknown feature-normalization mode: {mode}")


@torch.inference_mode()
def extract_audio_features(
    wav_path: str,
    processor: WhisperProcessor,
    whisper_model: WhisperModel,
    single_layer: int,
    device: torch.device,
):
    audio = load_audio(wav_path)
    features = []
    bounds = split_segment_boundaries(len(audio), SEGMENT_LENGTH, OVERLAP_RATE)
    segments = pad_audio_segments(audio, bounds)
    layer_norm = whisper_model.encoder.layer_norm

    for segment in segments:
        processed = processor(segment, sampling_rate=SAMPLING_RATE, return_tensors="pt")
        encoder_output = whisper_model.encoder(
            input_features=processed.input_features.to(device),
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = select_hidden_state(encoder_output, single_layer)
        total_layers = len(encoder_output.hidden_states) - 1
        hidden = apply_feature_norm(
            hidden,
            NORM_FOR_SINGLE,
            layer_norm,
            skip_layer_norm=(
                NORM_FOR_SINGLE == "ln_post" and single_layer == total_layers
            ),
        )
        features.append(hidden.squeeze(0).cpu())
    return features


def save_missing_mmse_csv(
    records: Sequence[Dict[str, object]], feature_dir: str, file_tag: str
):
    if not records:
        return None
    dataframe = pd.DataFrame(records).drop_duplicates(
        subset=["subset_name", "wav_path"]
    )
    path = os.path.join(feature_dir, f"missing_mmse_{file_tag}.csv")
    dataframe.to_csv(path, index=False, encoding="utf-8-sig")
    return path


@torch.inference_mode()
def export_features(
    dataset_key: str,
    dataset_paths: Dict[str, str],
    mmse_maps,
    dataset_files: Dict[str, Sequence[str]],
    processor: WhisperProcessor,
    whisper_model: WhisperModel,
    single_layer: int,
    feature_dir: str,
    device: torch.device,
    precomputed_missing: Sequence[Dict[str, object]],
    split_info: Optional[Dict[str, object]],
):
    os.makedirs(feature_dir, exist_ok=True)
    train_features = []
    test_features = []
    train_labels = []
    test_labels = []
    train_indices = []
    test_indices = []
    train_entries = []
    test_entries = []
    missing = list(precomputed_missing)
    next_ids = {"train": 0, "test": 0}

    for subset in ["train_cc", "train_cd", "test_cc", "test_cd"]:
        split = "train" if subset.startswith("train_") else "test"
        wav_files = list(dataset_files.get(subset, []))
        if not wav_files:
            wav_files = list_wav_files(dataset_paths[subset], dataset_key)

        for wav_path in tqdm(
            wav_files, desc=f"extract {dataset_key} {subset}", ncols=100
        ):
            target = get_mmse(dataset_key, wav_path, mmse_maps[subset])
            if target is None:
                missing.append(missing_mmse_record(dataset_key, subset, wav_path))
                continue

            segment_features = extract_audio_features(
                wav_path,
                processor,
                whisper_model,
                single_layer,
                device,
            )
            audio_id = next_ids[split]
            next_ids[split] += 1
            entry = {
                "audio_id": audio_id,
                "target_mmse": float(target),
                "split": split,
                "wav_path": canonical_wav_path(wav_path),
                "wav_name": Path(wav_path).name,
            }
            if dataset_key == "Pitt":
                subject_id, visit_index = extract_pitt_subject_visit(
                    Path(wav_path).stem
                )
                entry["subject_id"] = None if subject_id is None else int(subject_id)
                entry["visit_index"] = visit_index

            if split == "train":
                train_entries.append(entry)
                for segment_id, feature in enumerate(segment_features):
                    train_features.append(feature)
                    train_labels.append(float(target))
                    train_indices.append((audio_id, segment_id))
            else:
                test_entries.append(entry)
                for segment_id, feature in enumerate(segment_features):
                    test_features.append(feature)
                    test_labels.append(float(target))
                    test_indices.append((audio_id, segment_id))

    if not train_features or not test_features:
        raise RuntimeError(f"{dataset_key} exported empty train or test features")

    all_features = train_features + test_features
    target_time = max(int(feature.shape[0]) for feature in all_features)

    def stack_to_target(features):
        output = []
        for feature in features:
            if int(feature.shape[0]) < target_time:
                feature = torch.cat(
                    [
                        feature,
                        torch.zeros(
                            (
                                target_time - int(feature.shape[0]),
                                int(feature.shape[1]),
                            ),
                            dtype=feature.dtype,
                        ),
                    ],
                    dim=0,
                )
            output.append(feature[:target_time])
        return torch.stack(output)

    file_tag = make_file_tag(single_layer)
    paths = {
        "feature_train": os.path.join(feature_dir, f"feature_{file_tag}_train.pt"),
        "feature_test": os.path.join(feature_dir, f"feature_{file_tag}_test.pt"),
        "label_train": os.path.join(feature_dir, f"labels1_{file_tag}_train.pt"),
        "label_test": os.path.join(feature_dir, f"labels2_{file_tag}_test.pt"),
        "index_train": os.path.join(feature_dir, f"indices_{file_tag}_train.pt"),
        "index_test": os.path.join(feature_dir, f"indices_{file_tag}_test.pt"),
    }
    torch.save(stack_to_target(train_features), paths["feature_train"])
    torch.save(stack_to_target(test_features), paths["feature_test"])
    torch.save(torch.tensor(train_labels, dtype=torch.float32), paths["label_train"])
    torch.save(torch.tensor(test_labels, dtype=torch.float32), paths["label_test"])
    torch.save(np.asarray(train_indices, dtype=np.int64), paths["index_train"])
    torch.save(np.asarray(test_indices, dtype=np.int64), paths["index_test"])

    metadata = {
        "format_version": 3 if dataset_key == "Pitt" else 1,
        "task_type": "mmse_regression",
        "dataset_key": dataset_key,
        "dataset": dataset_key,
        "paper_dataset": paper_dataset_name(dataset_key),
        "file_tag": file_tag,
        "feature_save_path": str(Path(feature_dir).resolve()),
        "run_protocol": RUN_PROTOCOL_TAG,
        "split_strategy": (
            PITT_SPLIT_POLICY if dataset_key == "Pitt" else RUN_PROTOCOL_TAG
        ),
        "subject_disjoint": dataset_key == "Pitt",
        "train_test_subject_overlap": 0 if dataset_key == "Pitt" else None,
        "split_info": split_info,
        "train_audio_ids": [entry["audio_id"] for entry in train_entries],
        "val_audio_ids": [],
        "test_audio_ids": [entry["audio_id"] for entry in test_entries],
        "train_entries": train_entries,
        "test_entries": test_entries,
    }
    metadata_path = os.path.join(
        feature_dir, f"official_train_test_metadata_{file_tag}.json"
    )
    lookup_path = os.path.join(feature_dir, f"train_val_split_{file_tag}.json")
    save_json(metadata, metadata_path)
    save_json(metadata, lookup_path)
    paths["metadata"] = metadata_path
    paths["audio_lookup"] = lookup_path
    paths["missing_mmse"] = save_missing_mmse_csv(missing, feature_dir, file_tag)
    return paths


# =============================================================================
# 7. Single-run and batch entry points
# =============================================================================


def run_one_configuration(
    dataset_key: str,
    single_layer: int,
    unfreeze_count: int,
    total_layers: int,
):
    set_global_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = build_run_paths(dataset_key, single_layer, unfreeze_count)
    os.makedirs(paths["feature_dir"], exist_ok=True)
    os.makedirs(paths["checkpoint_dir"], exist_ok=True)

    dataset_paths = DATASETS[dataset_key]
    mmse_maps = load_mmse_maps(dataset_key)
    precomputed_missing = []
    split_info = None
    pitt_manifest_paths = None

    if dataset_key == "Pitt":
        (
            dataset_files,
            visit_dataframe,
            subject_dataframe,
            precomputed_missing,
            split_info,
        ) = build_pitt_subject_disjoint_files(dataset_paths, mmse_maps["train_cc"])
        pitt_manifest_paths = save_pitt_manifests(visit_dataframe, subject_dataframe)
    else:
        dataset_files = {
            subset: list_wav_files(directory, dataset_key)
            for subset, directory in dataset_paths.items()
        }

    train_cc, next_id, missing_cc = build_records(
        dataset_files["train_cc"], dataset_key, "train_cc", mmse_maps["train_cc"], 0
    )
    train_cd, _, missing_cd = build_records(
        dataset_files["train_cd"],
        dataset_key,
        "train_cd",
        mmse_maps["train_cd"],
        next_id,
    )
    train_records = train_cc + train_cd
    if not train_records:
        raise RuntimeError(f"{dataset_key} has no training audio available for fine-tuning")

    resolved_layers = resolve_unfreeze_layers(
        single_layer, total_layers, unfreeze_count
    )
    processor = WhisperProcessor.from_pretrained(
        BASE_WHISPER_MODEL,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = VanillaWhisperRegressor(single_layer).to(device)
    actual_trainable_layers = configure_trainable_layers(model, resolved_layers)
    full_encoder_finetuning = actual_trainable_layers == list(
        range(1, len(model.whisper.encoder.layers) + 1)
    )

    print("\n" + "=" * 100)
    print(f"[VanillaFT] dataset={dataset_key}")
    print(f"[VanillaFT] single_layer={single_layer}")
    print(f"[VanillaFT] trainable_encoder_layers={actual_trainable_layers}")
    print(f"[VanillaFT] full_encoder_finetuning={full_encoder_finetuning}")
    print(f"[VanillaFT] train_audio={len(train_records)}")
    print("[VanillaFT] official test is export-only")
    print("=" * 100)

    training_result = train_model(model, processor, train_records, device)

    model.whisper.save_pretrained(paths["checkpoint_dir"])
    processor.save_pretrained(paths["checkpoint_dir"])
    head_path = os.path.join(paths["checkpoint_dir"], "linear_mmse_head.pt")
    torch.save(model.regressor.state_dict(), head_path)

    export_result = None
    if EXPORT_FEATURES:
        model.whisper.eval()
        export_result = export_features(
            dataset_key=dataset_key,
            dataset_paths=dataset_paths,
            mmse_maps=mmse_maps,
            dataset_files=dataset_files,
            processor=processor,
            whisper_model=model.whisper,
            single_layer=single_layer,
            feature_dir=paths["feature_dir"],
            device=device,
            precomputed_missing=precomputed_missing + missing_cc + missing_cd,
            split_info=split_info,
        )

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "baseline": "single_stage_vanilla_finetuning",
        "dataset_key": dataset_key,
        "base_whisper_model": BASE_WHISPER_MODEL,
        "single_layer": single_layer,
        "unfreeze_layer_count": unfreeze_count,
        "trainable_encoder_layers": actual_trainable_layers,
        "full_encoder_finetuning": full_encoder_finetuning,
        "num_epochs": NUM_EPOCHS,
        "audio_batch_size": AUDIO_BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "loss_type": LOSS_TYPE,
        "target_normalize": TARGET_NORMALIZE,
        "seed": SEED,
        "environment": {
            "python": os.sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "librosa": librosa.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "device": str(device),
            "gpu_name": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "run_protocol": RUN_PROTOCOL_TAG,
        "official_test_used_for_training": False,
        "checkpoint_selection": "last",
        "segment_sec": SEGMENT_SEC,
        "overlap_rate": OVERLAP_RATE,
        "normalize_audio": NORMALIZE_AUDIO,
        "feature_norm": NORM_FOR_SINGLE,
        "feature_dir": paths["feature_dir"],
        "checkpoint_dir": paths["checkpoint_dir"],
        "linear_head_path": head_path,
        "pitt_manifest_paths": pitt_manifest_paths,
        "split_info": split_info,
        "training_result": training_result,
        "feature_export": export_result,
    }
    manifest_path = os.path.join(paths["checkpoint_dir"], "run_manifest.json")
    save_json(manifest, manifest_path)
    print("[SAVE] checkpoint ->", paths["checkpoint_dir"])
    print("[SAVE] features   ->", paths["feature_dir"])
    print("[SAVE] manifest   ->", manifest_path)

    del model
    del processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset_key": dataset_key,
        "single_layer": single_layer,
        "unfreeze_layer_count": unfreeze_count,
        "manifest_path": manifest_path,
        "feature_export": export_result,
    }


def main():
    unsupported = [
        key for key in DATASET_KEYS if key not in DATASETS or key not in MMSE_META_PATHS
    ]
    if unsupported:
        raise ValueError(f"Unconfigured datasets: {unsupported}")

    total_layers = get_whisper_encoder_layer_count()
    results = []
    for dataset_key in DATASET_KEYS:
        for single_layer in SINGLE_LAYERS:
            for unfreeze_count in UNFREEZE_ENCODER_LAYERS_LIST:
                results.append(
                    run_one_configuration(
                        dataset_key=dataset_key,
                        single_layer=int(single_layer),
                        unfreeze_count=int(unfreeze_count),
                        total_layers=total_layers,
                    )
                )

    summary_dir = os.path.join(FEATURE_ROOT_DIR, f"ft_{FEATURE_MODEL_SAVE_TAG}")
    summary_path = os.path.join(summary_dir, "batch_run_summary.json")
    save_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "baseline": "single_stage_vanilla_finetuning",
            "seed": SEED,
            "dataset_keys": DATASET_KEYS,
            "single_layers": SINGLE_LAYERS,
            "unfreeze_encoder_layers_list": UNFREEZE_ENCODER_LAYERS_LIST,
            "results": results,
        },
        summary_path,
    )
    print("[DONE] summary ->", summary_path)


if __name__ == "__main__":
    main()
