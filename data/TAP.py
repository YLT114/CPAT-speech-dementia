#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-Adaptive Pretraining (TAP) baseline for Whisper and MMSE regression.

This standalone script is designed to sit beside SSA.py and VanillaFT.py. It
does not import or modify either baseline.

Pipeline:
1. Use only the official training split. For Pitt, use the final training set
   produced by the ``original_test_subjects_holdout`` policy.
2. Adapt Whisper with the standard sequence-to-sequence ASR objective on
   manually annotated CHAT/CLAN transcripts and non-overlapping 30-second audio
   windows.
3. Initialize MMSE fine-tuning from the TAP checkpoint and use the same official
   training split.
4. Never use the official test split for training or checkpoint selection.
5. Export single-layer features in the same format as SSA.py and VanillaFT.py.

Only human transcripts are supported. The parser reads the *PAR and *INV main
tiers with timestamps, ignores dependent tiers such as %mor and %gra, and uses
the cleaned main-tier text as the ASR target.

Before running, set ``USER_NAME`` below or export ``TAP_USER_NAME``. Any root
path can also be overridden with the documented environment variables.

Example:
    conda activate audiojax_gpu
    export TAP_USER_NAME=your_username
    python TAP.py
"""


try:
    from .paper_names import paper_dataset_name
except ImportError:
    from paper_names import paper_dataset_name

import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    WhisperConfig,
    WhisperForConditionalGeneration,
    WhisperModel,
    WhisperProcessor,
)


# =============================================================================
# 1. Experiment configuration
# =============================================================================

GPU_ID = os.environ.get("TAP_GPU_ID", "0").strip()
if GPU_ID:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

os.environ["PYTHONHASHSEED"] = "400"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 400
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Audio and MMSE layouts are defined for four datasets. TAP transcript paths
# are supplied for ADReSS-IS2020 by default.
DATASET_KEYS = ["ADReSS-IS2020"]
# Add transcript paths to HUMAN_CHA_DIRS before enabling other datasets:
# DATASET_KEYS = ["ADReSS-IS2020", "ADReSSo21", "Pitt", "PROCESS-2"]

# Configure the account name once, or set TAP_USER_NAME in the environment.
USER_NAME = os.environ.get("TAP_USER_NAME", "YOUR_USERNAME")
USER_HOME = Path("/home") / USER_NAME
DOWNLOADS_ROOT = Path(os.environ.get("TAP_DOWNLOADS_ROOT", str(USER_HOME / "Downloads")))
PROJECT_ROOT = os.environ.get(
    "TAP_PROJECT_ROOT",
    str(USER_HOME / "Alzheimer_code_jax_mmse_speech_v4_mmse_mean"),
)
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT, "feature_tensor")
MODEL_ROOT_DIR = os.path.join(PROJECT_ROOT, "models")

# Do not add the ``ft_`` prefix; output paths add it exactly once.
MODEL_NAME = "tap_lr1e-5_epo5"
BASE_WHISPER_MODEL = os.environ.get(
    "TAP_WHISPER_MODEL",
    str(USER_HOME / "whisper_lru_3" / "whisper_small"),
)
LOCAL_FILES_ONLY = True
RUN_PROTOCOL_TAG = "official_train_test"

SAMPLING_RATE = 16000
SEGMENT_SEC = 30
SEGMENT_LENGTH = SAMPLING_RATE * SEGMENT_SEC
OVERLAP_RATE = 0.3
NORMALIZE_AUDIO = True

# TAP uses manually annotated English transcripts only.
TAP_LANGUAGE = "en"
TAP_TASK = "transcribe"

# CHAT transcript directories. Keys must match the subset keys in DATASETS.
# ADReSS-IS2020 is configured from DOWNLOADS_ROOT. Add real transcript paths
# for another dataset before selecting it; the script never guesses paths.
HUMAN_CHA_DIRS: Dict[str, Dict[str, str]] = {
    "ADReSS-IS2020": {
        "train_cc": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "train" / "transcription" / "cc"),
        "train_cd": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "train" / "transcription" / "cd"),
    },
    "ADReSSo21": {},
    "Pitt": {},
    "PROCESS-2": {},
}
HUMAN_CHA_SPEAKERS = ["PAR", "INV"]  # Full interviews contain participant and interviewer speech.
HUMAN_CHA_STRICT_MATCH = True
REUSE_TAP_MANIFEST = True

TAP_NUM_EPOCHS = 5
TAP_BATCH_SIZE = 2
TAP_GRAD_ACCUM_STEPS = 4
TAP_LR = 1e-5
TAP_WEIGHT_DECAY = 1e-2

# Match the primary VanillaFT configuration: layer 12 and a sweep over the
# number of consecutively unfrozen encoder layers.
SINGLE_LAYERS = [12]
UNFREEZE_ENCODER_LAYERS_LIST = [5]
MMSE_NUM_EPOCHS = 3
MMSE_BATCH_SIZE = 4
MMSE_WHISPER_LR = 1e-5
MMSE_HEAD_LR = 3e-5
MMSE_WEIGHT_DECAY = 1e-2
REGRESSION_LOSS_TYPE = "mae"
TARGET_NORMALIZE = True
TARGET_NORMALIZE_EPS = 1e-6

GRAD_CLIP_NORM = 1.0
NUM_WORKERS = 4
PRECISION = "fp16"  # fp32 / fp16 / bf16
FEATURE_PREFIX = "whisper"
NORM_FOR_SINGLE = "ln_post"
SKIP_FEATURE_EXPORT = False
SKIP_TAP_STAGE = False
SKIP_MMSE_STAGE = False

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


DATASETS: Dict[str, Dict[str, str]] = {
    "PROCESS-2": {
        "train_cc": str(DOWNLOADS_ROOT / "PROCESS-2" / "train" / "cc"),
        "train_cd": str(DOWNLOADS_ROOT / "PROCESS-2" / "train" / "cd"),
        "test_cc": str(DOWNLOADS_ROOT / "PROCESS-2" / "test" / "cc"),
        "test_cd": str(DOWNLOADS_ROOT / "PROCESS-2" / "test" / "cd"),
    },
    "ADReSSo21": {
        "train_cc": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSSo21" / "diagnosis" / "train" / "audio" / "cn"),
        "train_cd": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSSo21" / "diagnosis" / "train" / "audio" / "ad"),
        "test_cc": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSSo21" / "diagnosis" / "test-dist" / "Control"),
        "test_cd": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSSo21" / "diagnosis" / "test-dist" / "ProbableAD"),
    },
    "ADReSS-IS2020": {
        "train_cc": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "train" / "Full_wave_enhanced_audio" / "cc"),
        "train_cd": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "train" / "Full_wave_enhanced_audio" / "cd"),
        "test_cc": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "test" / "Full_wave_enhanced_cc_audio"),
        "test_cd": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "test" / "Full_wave_enhanced_cd_audio"),
    },
    "Pitt": {
        "train_cc": str(DOWNLOADS_ROOT / "dementiabank" / "Pitt_new" / "norm" / "train_test_split" / "train" / "cc_enhence"),
        "train_cd": str(DOWNLOADS_ROOT / "dementiabank" / "Pitt_new" / "norm" / "train_test_split" / "train" / "cd_enhence"),
        "test_cc": str(DOWNLOADS_ROOT / "dementiabank" / "Pitt_new" / "norm" / "train_test_split" / "test" / "cc_enhence"),
        "test_cd": str(DOWNLOADS_ROOT / "dementiabank" / "Pitt_new" / "norm" / "train_test_split" / "test" / "cd_enhence"),
    },
}

MMSE_META_PATHS: Dict[str, Dict[str, Any]] = {
    "PROCESS-2": {"meta": str(DOWNLOADS_ROOT / "PROCESS-2" / "meta-info.csv")},
    "ADReSS-IS2020": {
        "train_cc": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "train" / "cc_meta_data.txt"),
        "train_cd": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "train" / "cd_meta_data.txt"),
        "test_all": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSS-IS2020-data" / "test" / "ADReSS 2020 test result.txt"),
    },
    "ADReSSo21": {
        "train_all": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSSo21" / "diagnosis" / "train" / "adresso-train-mmse-scores.csv"),
        "test_all": str(DOWNLOADS_ROOT / "dementiabank" / "ADReSSo21" / "diagnosis" / "test-dist" / "test_results_task2_groundtruth.csv"),
    },
    "Pitt": {
        "all_meta": str(DOWNLOADS_ROOT / "dementiabank" / "Pitt_new" / "norm" / "train_test_split" / "PItt-data(1).xlsx"),
        "sheet_name": "data",
        "header_row": 2,
        "id_col": "id",
    },
}


# =============================================================================
# 2. Shared utilities, audio segmentation, and MMSE metadata
# =============================================================================

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def canonical_wav_path(path: str) -> str:
    return str(Path(str(path)).expanduser().resolve())


def natural_key(path: str) -> List[Any]:
    return [int(x) if x.isdigit() else x.lower() for x in re.findall(r"\d+|\D+", Path(path).name)]


def save_json(payload: Any, path: str) -> None:
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    ensure_dir(str(Path(path).parent))
    pd.DataFrame(list(rows)).to_csv(path, index=False, encoding="utf-8-sig")


def _safe_float(value: Any) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "N/A", "NAN", "NONE"}:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _pick_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    columns = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if str(candidate).strip().lower() in columns:
            return columns[str(candidate).strip().lower()]
    raise KeyError(f"None of the columns {list(candidates)} were found; available columns: {list(df.columns)}")


def _normalize_id(dataset_key: str, raw_id: Any) -> Optional[str]:
    if raw_id is None or pd.isna(raw_id):
        return None
    value = Path(str(raw_id).strip()).stem
    if dataset_key == "PROCESS-2":
        value = re.sub(r"__(SFT|PFT|CTD)$", "", value, flags=re.IGNORECASE)
    return value.lower()


def _pitt_subject_visit(path_or_name: str) -> Tuple[Optional[str], Optional[int]]:
    match = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", Path(str(path_or_name)).stem, flags=re.IGNORECASE)
    if match is None:
        return None, None
    return str(int(match.group(1))), int(match.group(2))


def _pitt_mmse_column(visit_index: int) -> str:
    return "mms" if int(visit_index) == 0 else f"mmse{int(visit_index) + 1}"


def _is_process2_ctd(path: str, dataset_key: str) -> bool:
    return dataset_key != "PROCESS-2" or re.search(r"__CTD$", Path(path).stem, re.IGNORECASE) is not None


def list_wav_files(audio_dir: str, dataset_key: str) -> List[str]:
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    paths = [
        canonical_wav_path(os.path.join(audio_dir, name))
        for name in os.listdir(audio_dir)
        if name.lower().endswith(".wav") and _is_process2_ctd(name, dataset_key)
    ]
    paths.sort(key=natural_key)
    if not paths:
        raise RuntimeError(f"No eligible .wav files were found in: {audio_dir}")
    return paths


def split_segment_boundaries(num_samples: int) -> List[Tuple[int, int, int]]:
    step = int(SEGMENT_LENGTH * (1.0 - OVERLAP_RATE))
    if step <= 0:
        raise ValueError("OVERLAP_RATE produces a non-positive segmentation step")
    total = int(num_samples / step) + (1 if num_samples % step > OVERLAP_RATE * SEGMENT_LENGTH else 0)
    total = max(total, 1)
    return [(i * step, i * step + SEGMENT_LENGTH, i) for i in range(total)]


def load_audio(path: str) -> np.ndarray:
    wav, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True, dtype=np.float32)
    if NORMALIZE_AUDIO and wav.size:
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-8:
            wav = wav / peak
    return wav


def get_padded_segment(wav: np.ndarray, start: int, end: int, segment_id: int) -> Tuple[np.ndarray, int]:
    segment = wav[start:end]
    valid_len = min(len(segment), SEGMENT_LENGTH)
    if len(segment) < SEGMENT_LENGTH and segment_id > 0:
        need = SEGMENT_LENGTH - len(segment)
        step = int(SEGMENT_LENGTH * (1.0 - OVERLAP_RATE))
        previous_end = start + SEGMENT_LENGTH - step
        previous_start = max(0, previous_end - need)
        pad = wav[previous_start:previous_end]
        if len(pad) < need:
            pad = np.pad(pad, (0, need - len(pad)))
        segment = np.concatenate([segment, pad])
    if len(segment) < SEGMENT_LENGTH:
        segment = np.pad(segment, (0, SEGMENT_LENGTH - len(segment)))
    return segment[:SEGMENT_LENGTH].astype(np.float32), int(valid_len)


def _map_from_frame(df: pd.DataFrame, dataset_key: str, id_col: str, mmse_col: str) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for _, row in df.iterrows():
        key = _normalize_id(dataset_key, row[id_col])
        value = _safe_float(row[mmse_col])
        if key is not None and value is not None:
            result[key] = value
    return result


def load_mmse_maps(dataset_key: str) -> Dict[str, Dict[Any, float]]:
    cfg = MMSE_META_PATHS[dataset_key]
    if dataset_key == "ADReSS-IS2020":
        def read_txt(path: str) -> Dict[str, float]:
            result = {}
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("ID"):
                        continue
                    parts = [x.strip() for x in line.split(";")]
                    key, value = _normalize_id(dataset_key, parts[0]), _safe_float(parts[-1])
                    if key is not None and value is not None:
                        result[key] = value
            return result
        test_map = read_txt(cfg["test_all"])
        return {
            "train_cc": read_txt(cfg["train_cc"]),
            "train_cd": read_txt(cfg["train_cd"]),
            "test_cc": test_map,
            "test_cd": test_map,
        }

    if dataset_key == "ADReSSo21":
        train_df, test_df = pd.read_csv(cfg["train_all"]), pd.read_csv(cfg["test_all"])
        train_id = _pick_column(train_df, ["adressfname", "addressfname", "id"])
        train_mmse = _pick_column(train_df, ["mmse"])
        train_dx = _pick_column(train_df, ["dx", "label", "group"])
        test_id = _pick_column(test_df, ["id", "adressfname", "addressfname"])
        test_mmse = _pick_column(test_df, ["mmse"])
        dx = train_df[train_dx].astype(str).str.strip().str.lower()
        cc = train_df[dx.isin(["cn", "control", "cc"])]
        cd = train_df[dx.isin(["ad", "probablead", "cd"])]
        test_map = _map_from_frame(test_df, dataset_key, test_id, test_mmse)
        return {
            "train_cc": _map_from_frame(cc, dataset_key, train_id, train_mmse),
            "train_cd": _map_from_frame(cd, dataset_key, train_id, train_mmse),
            "test_cc": test_map,
            "test_cd": test_map,
        }

    if dataset_key == "PROCESS-2":
        df = pd.read_csv(cfg["meta"])
        id_col = _pick_column(df, ["IDs", "ID", "id", "dir_name"])
        dx_col = _pick_column(df, ["diagnosis", "dx", "label"])
        mmse_col = _pick_column(df, ["MMSE", "mmse"])
        split_col = _pick_column(df, ["Split", "split", "partition"])
        split = df[split_col].astype(str).str.strip().str.lower()
        dx = df[dx_col].astype(str).str.strip().str.lower()
        masks = {
            "train_cc": split.isin(["train", "training"]) & dx.isin(["hc", "healthy", "healthy control", "control", "cn", "cc"]),
            "train_cd": split.isin(["train", "training"]) & dx.isin(["mci", "dementia", "ad", "probablead", "cd"]),
            "test_cc": split.isin(["test", "testing"]) & dx.isin(["hc", "healthy", "healthy control", "control", "cn", "cc"]),
            "test_cd": split.isin(["test", "testing"]) & dx.isin(["mci", "dementia", "ad", "probablead", "cd"]),
        }
        return {k: _map_from_frame(df[m], dataset_key, id_col, mmse_col) for k, m in masks.items()}

    if dataset_key == "Pitt":
        df = pd.read_excel(cfg["all_meta"], sheet_name=cfg["sheet_name"], header=int(cfg["header_row"]))
        id_col = _pick_column(df, [cfg["id_col"], "id"])
        columns = {str(c).strip().lower(): c for c in df.columns}
        all_map: Dict[Tuple[str, int], float] = {}
        for _, row in df.iterrows():
            match = re.search(r"\d+", str(row[id_col]))
            if match is None:
                continue
            subject = str(int(match.group()))
            for visit in range(7):
                name = _pitt_mmse_column(visit)
                if name in columns:
                    value = _safe_float(row[columns[name]])
                    if value is not None:
                        all_map[(subject, visit)] = value
        return {k: all_map for k in ["train_cc", "train_cd", "test_cc", "test_cd"]}

    raise ValueError(f"Dataset is not implemented: {dataset_key}")


def lookup_mmse(dataset_key: str, wav_path: str, mmse_map: Dict[Any, float]) -> Optional[float]:
    if dataset_key == "Pitt":
        subject, visit = _pitt_subject_visit(wav_path)
        return None if subject is None or visit is None else mmse_map.get((subject, visit))
    key = _normalize_id(dataset_key, Path(wav_path).stem)
    return None if key is None else mmse_map.get(key)


@dataclass
class AudioEntry:
    wav_path: str
    subset: str
    group: str
    mmse: float
    subject_id: Optional[str] = None
    visit_index: Optional[int] = None


def build_official_entries(dataset_key: str) -> Tuple[List[AudioEntry], List[AudioEntry], List[Dict[str, Any]]]:
    paths, maps = DATASETS[dataset_key], load_mmse_maps(dataset_key)
    records: List[AudioEntry] = []
    missing: List[Dict[str, Any]] = []
    seen_pitt_visits: Dict[Tuple[str, int], str] = {}
    for subset in ["train_cc", "train_cd", "test_cc", "test_cd"]:
        for wav_path in list_wav_files(paths[subset], dataset_key):
            value = lookup_mmse(dataset_key, wav_path, maps[subset])
            subject, visit = _pitt_subject_visit(wav_path) if dataset_key == "Pitt" else (None, None)
            if value is None:
                missing.append({
                    "dataset_key": dataset_key,
                    "subset_name": subset,
                    "wav_name": Path(wav_path).name,
                    "wav_path": wav_path,
                    "reason": "missing_mmse",
                    "subject_id": subject,
                    "visit_index": visit,
                })
                continue
            if dataset_key == "Pitt":
                visit_key = (str(subject), int(visit))
                if visit_key in seen_pitt_visits:
                    raise RuntimeError(
                        "[Pitt split] Duplicate subject-visit pair: "
                        f"{visit_key}; first={seen_pitt_visits[visit_key]}; second={wav_path}"
                    )
                seen_pitt_visits[visit_key] = wav_path
            records.append(AudioEntry(wav_path, subset, subset.split("_")[1], float(value), subject, visit))

    if dataset_key == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT:
        if PITT_SPLIT_POLICY != "original_test_subjects_holdout":
            raise ValueError(f"Unsupported PITT_SPLIT_POLICY={PITT_SPLIT_POLICY}")
        original_test_subjects = {r.subject_id for r in records if r.subset.startswith("test_")}
        train = [r for r in records if r.subject_id not in original_test_subjects]
        test = [r for r in records if r.subject_id in original_test_subjects]
        checks = [
            ("candidate_visits", len(records) + len(missing), PITT_EXPECTED_CANDIDATE_VISITS),
            ("valid_visits", len(records), PITT_EXPECTED_VALID_VISITS),
            ("unique_subjects", len({r.subject_id for r in records}), PITT_EXPECTED_UNIQUE_SUBJECTS),
            ("valid_original_test_subjects", len(original_test_subjects), PITT_EXPECTED_VALID_TEST_SUBJECTS),
            ("final_train_subjects", len({r.subject_id for r in train}), PITT_EXPECTED_FINAL_TRAIN_SUBJECTS),
            ("final_test_subjects", len({r.subject_id for r in test}), PITT_EXPECTED_FINAL_TEST_SUBJECTS),
            ("final_train_visits", len(train), PITT_EXPECTED_FINAL_TRAIN_VISITS),
            ("final_test_visits", len(test), PITT_EXPECTED_FINAL_TEST_VISITS),
        ]
        for name, actual, expected in checks:
            if actual != expected:
                message = f"[Pitt split] {name}={actual}; expected={expected}"
                if PITT_STRICT_COUNT_CHECK:
                    raise RuntimeError(message)
                print("[WARN]", message)
        train_subjects, test_subjects = {r.subject_id for r in train}, {r.subject_id for r in test}
        if train_subjects & test_subjects:
            raise RuntimeError("The final Pitt train/test splits contain overlapping subjects")
        return sorted(train, key=lambda x: natural_key(x.wav_path)), sorted(test, key=lambda x: natural_key(x.wav_path)), missing

    train = [r for r in records if r.subset.startswith("train_")]
    test = [r for r in records if r.subset.startswith("test_")]
    return train, test, missing


# =============================================================================
# 3. Human-transcript manifest and sequence-to-sequence adaptation
# =============================================================================

CHAT_TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")


def _clean_chat_text(raw_text: str) -> str:
    """Remove CHAT/CLAN markup while preserving spoken natural language."""
    text = CHAT_TIME_RE.sub(" ", str(raw_text))
    # [: replacement] is normalization markup; retain the preceding speech.
    text = re.sub(r"\[\s*:[^\]]*\]", " ", text)
    text = re.sub(r"\[\s*\*[^\]]*\]", " ", text)
    text = re.sub(r"\[(?:/|//|///|\?|!|=)[^\]]*\]", " ", text)
    text = re.sub(r"&=[^\s]+", " ", text)  # Non-lexical events such as laughter or coughing.
    text = re.sub(r"&\+?([A-Za-z']+)", r"\1", text)  # &uh, &um, or &+fragment.
    text = re.sub(r"\b(?:xxx|yyy|www)\b", " ", text, flags=re.IGNORECASE)
    text = text.replace("<", " ").replace(">", " ")
    text = re.sub(r"\+[/.,!?]+", " ", text)
    # Convert an(d) to and; remove parenthesized punctuation-only annotations.
    text = re.sub(r"\(([A-Za-z']+)\)", r"\1", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_chat_utterances(cha_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read timestamped *PAR/*INV main tiers and ignore dependent tiers."""
    included = {str(x).strip().upper() for x in HUMAN_CHA_SPEAKERS}
    utterances: List[Dict[str, Any]] = []
    audit: List[Dict[str, Any]] = []
    current_speaker: Optional[str] = None
    current_parts: List[str] = []
    current_line = 0

    def flush() -> None:
        nonlocal current_speaker, current_parts, current_line
        if current_speaker is None:
            return
        raw = " ".join(current_parts)
        times = CHAT_TIME_RE.findall(raw)
        text = _clean_chat_text(raw)
        if current_speaker in included and text:
            if times:
                start_ms = int(times[0][0])
                end_ms = int(times[-1][1])
                if end_ms > start_ms:
                    utterances.append({
                        "speaker": current_speaker,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "text": text,
                        "cha_line": current_line,
                    })
                else:
                    audit.append({"cha_line": current_line, "reason": "invalid_timestamp", "raw": raw})
            else:
                audit.append({"cha_line": current_line, "reason": "missing_timestamp", "raw": raw})
        current_speaker, current_parts, current_line = None, [], 0

    with open(cha_path, "r", encoding="utf-8-sig", errors="replace") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.rstrip("\r\n")
            main_match = re.match(r"^\*([^:]+):\s*(.*)$", line)
            if main_match:
                flush()
                current_speaker = main_match.group(1).strip().upper()
                current_parts = [main_match.group(2).strip()]
                current_line = line_number
            elif current_speaker is not None and line[:1].isspace():
                current_parts.append(line.strip())
            else:
                flush()
    flush()
    utterances.sort(key=lambda row: (row["start_ms"], row["end_ms"], row["cha_line"]))
    return utterances, audit


def build_human_cha_manifest(
    dataset_key: str,
    train_entries: Sequence[AudioEntry],
    save_path: str,
) -> List[Dict[str, Any]]:
    """Aggregate manual CHAT transcripts into non-overlapping TAP windows."""
    transcript_dirs = HUMAN_CHA_DIRS.get(dataset_key, {})
    if not transcript_dirs:
        raise ValueError(
            f"HUMAN_CHA_DIRS has no transcript paths for {dataset_key}. "
            "Configure the real .cha directories before selecting this dataset."
        )

    rows: List[Dict[str, Any]] = []
    audit_rows: List[Dict[str, Any]] = []
    missing_cha: List[str] = []
    for entry in tqdm(train_entries, desc="[TAP manifest] human CHAT", ncols=100):
        cha_dir = str(transcript_dirs.get(entry.subset, "")).strip()
        if not cha_dir:
            missing_cha.append(f"subset={entry.subset}: {entry.wav_path}")
            continue
        cha_path = os.path.join(cha_dir, f"{Path(entry.wav_path).stem}.cha")
        if not os.path.isfile(cha_path):
            missing_cha.append(cha_path)
            continue

        utterances, parse_audit = parse_chat_utterances(cha_path)
        for item in parse_audit:
            audit_rows.append({"wav_path": entry.wav_path, "cha_path": cha_path, **item})
        wav = load_audio(entry.wav_path)
        windows: Dict[int, List[Dict[str, Any]]] = {}
        for utterance in utterances:
            start_sample = int(round(utterance["start_ms"] * SAMPLING_RATE / 1000.0))
            if start_sample >= len(wav):
                audit_rows.append({
                    "wav_path": entry.wav_path,
                    "cha_path": cha_path,
                    "cha_line": utterance["cha_line"],
                    "reason": "timestamp_beyond_audio",
                    "raw": utterance["text"],
                })
                continue
            window_id = start_sample // SEGMENT_LENGTH
            windows.setdefault(window_id, []).append(utterance)

        for window_id in sorted(windows):
            start_sample = int(window_id * SEGMENT_LENGTH)
            end_sample = int(min(start_sample + SEGMENT_LENGTH, len(wav)))
            ordered = sorted(windows[window_id], key=lambda row: (row["start_ms"], row["end_ms"]))
            text = " ".join(row["text"] for row in ordered if row["text"]).strip()
            if not text:
                continue
            rows.append({
                "wav_path": entry.wav_path,
                "start_sample": start_sample,
                "end_sample": end_sample,
                "valid_samples": end_sample - start_sample,
                "segment_id": int(window_id),
                "text": text,
                "language": TAP_LANGUAGE,
                "label_source": "human_cha",
                "cha_path": canonical_wav_path(cha_path),
                "speakers": sorted({row["speaker"] for row in ordered}),
                "first_utterance_ms": int(ordered[0]["start_ms"]),
                "last_utterance_ms": int(max(row["end_ms"] for row in ordered)),
            })

    audit_path = os.path.join(str(Path(save_path).parent), "human_cha_audit.csv")
    save_csv(audit_rows, audit_path)
    if missing_cha:
        missing_path = os.path.join(str(Path(save_path).parent), "missing_human_cha.txt")
        with open(missing_path, "w", encoding="utf-8") as stream:
            stream.write("\n".join(missing_cha))
        message = f"Missing manual transcripts for {len(missing_cha)} audio files; see {missing_path}"
        if HUMAN_CHA_STRICT_MATCH:
            raise FileNotFoundError(message)
        print("[WARN]", message)
    if not rows:
        raise RuntimeError("No aligned training segments remained after parsing the manual .cha files")
    ensure_dir(str(Path(save_path).parent))
    with open(save_path, "w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def load_cached_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class HumanTranscriptDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, Any]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[index]
        wav = load_audio(row["wav_path"])
        start = int(row.get("start_sample", 0))
        end_value = row.get("end_sample")
        end = min(len(wav), start + SEGMENT_LENGTH) if end_value is None else min(len(wav), int(end_value))
        segment = wav[start:end]
        valid_len = min(len(segment), SEGMENT_LENGTH)
        if len(segment) < SEGMENT_LENGTH:
            segment = np.pad(segment, (0, SEGMENT_LENGTH - len(segment)))
        return {
            "waveform": segment[:SEGMENT_LENGTH].astype(np.float32),
            "valid_samples": valid_len,
            "text": str(row["text"]),
            "language": str(row.get("language", TAP_LANGUAGE)),
        }


class HumanTranscriptCollator:
    def __init__(self, processor: WhisperProcessor):
        self.processor = processor

    def __call__(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        waveforms = [x["waveform"] for x in batch]
        features = self.processor.feature_extractor(
            waveforms,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        ).input_features
        label_features = []
        for item in batch:
            self.processor.tokenizer.set_prefix_tokens(
                language=item["language"],
                task=TAP_TASK,
                predict_timestamps=False,
            )
            label_features.append({"input_ids": self.processor.tokenizer(item["text"]).input_ids})
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch.input_ids.masked_fill(labels_batch.attention_mask.ne(1), -100)
        decoder_start = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        if labels.shape[1] and bool((labels[:, 0] == decoder_start).all()):
            labels = labels[:, 1:]
        return {"input_features": features, "labels": labels}


def autocast_context():
    enabled = DEVICE.type == "cuda" and PRECISION in {"fp16", "bf16"}
    dtype = torch.float16 if PRECISION == "fp16" else torch.bfloat16
    return torch.autocast(device_type=DEVICE.type, dtype=dtype, enabled=enabled)


def train_task_adaptive_pretraining(
    dataset_key: str,
    train_entries: Sequence[AudioEntry],
    tap_model_dir: str,
) -> Dict[str, Any]:
    ensure_dir(tap_model_dir)
    processor = WhisperProcessor.from_pretrained(BASE_WHISPER_MODEL, local_files_only=LOCAL_FILES_ONLY)
    cached_manifest = os.path.join(tap_model_dir, "human_transcript_manifest.jsonl")

    if REUSE_TAP_MANIFEST and os.path.isfile(cached_manifest):
        rows = load_cached_jsonl(cached_manifest)
        invalid_sources = sorted({str(row.get("label_source", "")) for row in rows} - {"human_cha"})
        if invalid_sources:
            raise RuntimeError(
                f"Cached manifest {cached_manifest} contains unsupported label sources: {invalid_sources}"
            )
        print(f"[TAP manifest] Reusing {cached_manifest}")
    else:
        rows = build_human_cha_manifest(dataset_key, train_entries, cached_manifest)

    model = WhisperForConditionalGeneration.from_pretrained(
        BASE_WHISPER_MODEL,
        local_files_only=LOCAL_FILES_ONLY,
    ).to(DEVICE)
    model.config.use_cache = False
    model.generation_config.language = TAP_LANGUAGE
    model.generation_config.task = TAP_TASK
    loader = DataLoader(
        HumanTranscriptDataset(rows),
        batch_size=TAP_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=HumanTranscriptCollator(processor),
        pin_memory=DEVICE.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=TAP_LR, weight_decay=TAP_WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda" and PRECISION == "fp16")
    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, TAP_NUM_EPOCHS + 1):
        model.train()
        total_loss, steps = 0.0, 0
        for step, batch in enumerate(tqdm(loader, desc=f"[TAP ASR] epoch {epoch}", ncols=100), start=1):
            batch = {k: v.to(DEVICE, non_blocking=True) for k, v in batch.items()}
            with autocast_context():
                loss = model(**batch).loss / TAP_GRAD_ACCUM_STEPS
            scaler.scale(loss).backward()
            if step % TAP_GRAD_ACCUM_STEPS == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach().cpu()) * TAP_GRAD_ACCUM_STEPS
            steps += 1
        row = {"epoch": epoch, "train_loss": total_loss / max(steps, 1)}
        history.append(row)
        print("[TAP ASR]", row)

    model.save_pretrained(tap_model_dir)
    processor.save_pretrained(tap_model_dir)
    save_csv(history, os.path.join(tap_model_dir, "tap_training_history.csv"))
    result = {
        "stage": "human_transcript_adaptation",
        "method": "TAP",
        "method_full_name": "Task-Adaptive Pretraining",
        "label_source": "human_cha",
        "base_whisper_model": BASE_WHISPER_MODEL,
        "num_train_audio": len(train_entries),
        "num_tap_segments": len(rows),
        "epochs": TAP_NUM_EPOCHS,
        "learning_rate": TAP_LR,
        "selected_checkpoint": "last",
        "official_test_used": False,
        "tap_manifest": cached_manifest,
        "history": history,
    }
    save_json(result, os.path.join(tap_model_dir, "tap_training_manifest.json"))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# =============================================================================
# 4. MMSE fine-tuning from the TAP checkpoint
# =============================================================================

@dataclass
class SegmentEntry:
    wav_path: str
    mmse: float
    audio_id: int
    segment_id: int
    start: int
    end: int


def make_segment_entries(audio_entries: Sequence[AudioEntry]) -> List[SegmentEntry]:
    output = []
    for audio_id, entry in enumerate(tqdm(audio_entries, desc="[segments]", ncols=100)):
        wav = load_audio(entry.wav_path)
        for start, end, segment_id in split_segment_boundaries(len(wav)):
            output.append(SegmentEntry(entry.wav_path, entry.mmse, audio_id, segment_id, start, end))
    return output


class MMSESegmentDataset(Dataset):
    def __init__(self, entries: Sequence[SegmentEntry]):
        self.entries = list(entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.entries[index]
        wav = load_audio(row.wav_path)
        segment, valid_len = get_padded_segment(wav, row.start, row.end, row.segment_id)
        return {
            "waveform": segment,
            "valid_samples": valid_len,
            "target": row.mmse,
            "audio_id": row.audio_id,
            "segment_id": row.segment_id,
        }


def build_mmse_collator(processor: WhisperProcessor):
    def collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = processor.feature_extractor(
            [x["waveform"] for x in batch],
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        ).input_features
        return {
            "input_features": features,
            "valid_samples": torch.tensor([x["valid_samples"] for x in batch], dtype=torch.long),
            "targets": torch.tensor([x["target"] for x in batch], dtype=torch.float32),
        }
    return collate


class WhisperMMSERegressor(nn.Module):
    def __init__(self, model_path: str, single_layer: int):
        super().__init__()
        # The TAP directory is produced by save_pretrained on a conditional
        # generation model. Access .model explicitly to load its encoder and
        # decoder weights into the underlying WhisperModel without ambiguity.
        conditional = WhisperForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
        )
        self.whisper = conditional.model
        self.single_layer = int(single_layer)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(self.whisper.config.d_model, 1))

    def forward(self, input_features: torch.Tensor, valid_samples: torch.Tensor) -> torch.Tensor:
        output = self.whisper.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[self.single_layer]
        token_lengths = torch.ceil(valid_samples.float() / 320.0).long().clamp(1, hidden.shape[1])
        mask = torch.arange(hidden.shape[1], device=hidden.device)[None, :] < token_lengths[:, None]
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.head(pooled).squeeze(-1)


def configure_trainable_layers(model: WhisperMMSERegressor, single_layer: int, unfreeze_count: int) -> List[int]:
    for parameter in model.whisper.parameters():
        parameter.requires_grad = False
    start = max(1, int(single_layer) - int(unfreeze_count) + 1)
    layers = list(range(start, int(single_layer) + 1)) if unfreeze_count > 0 else []
    for layer_number in layers:
        for parameter in model.whisper.encoder.layers[layer_number - 1].parameters():
            parameter.requires_grad = True
    if layers:
        for parameter in model.whisper.encoder.layer_norm.parameters():
            parameter.requires_grad = True
    for parameter in model.head.parameters():
        parameter.requires_grad = True
    return layers


def train_mmse(
    train_entries: Sequence[AudioEntry],
    tap_model_dir: str,
    final_model_dir: str,
    single_layer: int,
    unfreeze_count: int,
) -> Tuple[WhisperMMSERegressor, WhisperProcessor, Dict[str, Any]]:
    processor = WhisperProcessor.from_pretrained(tap_model_dir, local_files_only=True)
    segments = make_segment_entries(train_entries)
    loader = DataLoader(
        MMSESegmentDataset(segments),
        batch_size=MMSE_BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=build_mmse_collator(processor),
        pin_memory=DEVICE.type == "cuda",
    )
    model = WhisperMMSERegressor(tap_model_dir, single_layer).to(DEVICE)
    resolved_layers = configure_trainable_layers(model, single_layer, unfreeze_count)
    whisper_params = [p for p in model.whisper.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": whisper_params, "lr": MMSE_WHISPER_LR},
            {"params": model.head.parameters(), "lr": MMSE_HEAD_LR},
        ],
        weight_decay=MMSE_WEIGHT_DECAY,
    )
    audio_targets = np.asarray([x.mmse for x in train_entries], dtype=np.float32)
    target_mean = float(audio_targets.mean()) if TARGET_NORMALIZE else 0.0
    target_std = float(audio_targets.std()) if TARGET_NORMALIZE else 1.0
    target_std = max(target_std, TARGET_NORMALIZE_EPS)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda" and PRECISION == "fp16")
    history = []
    for epoch in range(1, MMSE_NUM_EPOCHS + 1):
        model.train()
        total_loss, predictions, targets = 0.0, [], []
        for batch in tqdm(loader, desc=f"[MMSE] epoch {epoch}", ncols=100):
            input_features = batch["input_features"].to(DEVICE, non_blocking=True)
            valid_samples = batch["valid_samples"].to(DEVICE, non_blocking=True)
            target = batch["targets"].to(DEVICE, non_blocking=True)
            normalized = (target - target_mean) / target_std
            optimizer.zero_grad(set_to_none=True)
            with autocast_context():
                prediction_norm = model(input_features, valid_samples)
                if REGRESSION_LOSS_TYPE == "mae":
                    loss = torch.nn.functional.l1_loss(prediction_norm, normalized)
                elif REGRESSION_LOSS_TYPE == "mse":
                    loss = torch.nn.functional.mse_loss(prediction_norm, normalized)
                else:
                    loss = torch.nn.functional.smooth_l1_loss(prediction_norm, normalized)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            prediction = prediction_norm.detach() * target_std + target_mean
            total_loss += float(loss.detach().cpu())
            predictions.extend(prediction.cpu().tolist())
            targets.extend(target.cpu().tolist())
        pred_np, target_np = np.asarray(predictions), np.asarray(targets)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(loader), 1),
            "train_mae": float(np.mean(np.abs(pred_np - target_np))),
            "train_rmse": float(np.sqrt(np.mean((pred_np - target_np) ** 2))),
        }
        history.append(row)
        print("[MMSE]", row)

    ensure_dir(final_model_dir)
    model.whisper.save_pretrained(final_model_dir)
    processor.save_pretrained(final_model_dir)
    # Match VanillaFT.py: save only the temporal head state_dict under this name.
    torch.save(model.head.state_dict(), os.path.join(final_model_dir, "audio_temporal_head.pt"))
    result = {
        "stage": "tap_then_mmse_finetune",
        "num_train_audio": len(train_entries),
        "num_train_segments": len(segments),
        "single_layer": single_layer,
        "unfreeze_encoder_layers": unfreeze_count,
        "resolved_unfreeze_encoder_layers": resolved_layers,
        "target_mean": target_mean,
        "target_std": target_std,
        "epochs": MMSE_NUM_EPOCHS,
        "checkpoint_selection": "last",
        "official_test_used": False,
        "history": history,
    }
    save_json(result, os.path.join(final_model_dir, "finetune_metrics.json"))
    return model, processor, result


# =============================================================================
# 5. Feature export in the SSA.py and VanillaFT.py format
# =============================================================================

def make_file_tag(single_layer: int) -> str:
    overlap = f"{OVERLAP_RATE:.1f}".rstrip("0").rstrip(".")
    if "." not in overlap:
        overlap += ".0"
    return f"{FEATURE_PREFIX}_single{single_layer}_{overlap}_{SEGMENT_SEC}s"


@torch.inference_mode()
def extract_audio_features(
    entry: AudioEntry,
    processor: WhisperProcessor,
    whisper: WhisperModel,
    single_layer: int,
) -> List[torch.Tensor]:
    wav = load_audio(entry.wav_path)
    output = []
    total_layers = int(whisper.config.encoder_layers)
    for start, end, segment_id in split_segment_boundaries(len(wav)):
        segment, _ = get_padded_segment(wav, start, end, segment_id)
        features = processor.feature_extractor(
            segment,
            sampling_rate=SAMPLING_RATE,
            return_tensors="pt",
        ).input_features.to(DEVICE)
        encoded = whisper.encoder(input_features=features, output_hidden_states=True, return_dict=True)
        hidden = encoded.hidden_states[int(single_layer)]
        if NORM_FOR_SINGLE == "ln_post" and int(single_layer) != total_layers:
            hidden = whisper.encoder.layer_norm(hidden)
        elif NORM_FOR_SINGLE == "layernorm":
            hidden = torch.nn.functional.layer_norm(hidden, hidden.shape[-1:])
        elif NORM_FOR_SINGLE == "std":
            hidden = hidden / hidden.std(dim=-1, keepdim=True).clamp_min(1e-6)
        elif NORM_FOR_SINGLE not in {"none", "ln_post"}:
            raise ValueError(f"Unknown NORM_FOR_SINGLE={NORM_FOR_SINGLE}")
        output.append(hidden.squeeze(0).float().cpu())
    return output


def export_features(
    dataset_key: str,
    train_entries: Sequence[AudioEntry],
    test_entries: Sequence[AudioEntry],
    processor: WhisperProcessor,
    whisper: WhisperModel,
    save_dir: str,
    single_layer: int,
    missing_records: Sequence[Dict[str, Any]],
    effective_model_name: str,
) -> Dict[str, Any]:
    whisper.eval()
    feature_splits: Dict[str, List[torch.Tensor]] = {"train": [], "test": []}
    label_splits: Dict[str, List[float]] = {"train": [], "test": []}
    index_splits: Dict[str, List[Tuple[int, int]]] = {"train": [], "test": []}
    entry_splits = {"train": list(train_entries), "test": list(test_entries)}
    manifest_splits: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
    for split_name in ["train", "test"]:
        for audio_id, entry in enumerate(tqdm(entry_splits[split_name], desc=f"[export] {split_name}", ncols=100)):
            segment_features = extract_audio_features(entry, processor, whisper, single_layer)
            row = {
                "audio_id": audio_id,
                "target_mmse": entry.mmse,
                "split": split_name,
                "wav_path": entry.wav_path,
                "wav_name": Path(entry.wav_path).name,
                "group": entry.group,
                "num_segments": len(segment_features),
            }
            if dataset_key == "Pitt":
                row["subject_id"] = entry.subject_id
                row["visit_index"] = entry.visit_index
            manifest_splits[split_name].append(row)
            for segment_id, feature in enumerate(segment_features):
                feature_splits[split_name].append(feature)
                label_splits[split_name].append(entry.mmse)
                index_splits[split_name].append((audio_id, segment_id))

    if not feature_splits["train"] or not feature_splits["test"]:
        raise RuntimeError("The exported train/test feature collection is empty")
    target_t = max(x.shape[0] for values in feature_splits.values() for x in values)

    def stack(values: Sequence[torch.Tensor]) -> torch.Tensor:
        padded = [
            torch.cat([x, torch.zeros((target_t - x.shape[0], x.shape[1]), dtype=x.dtype)], dim=0)
            if x.shape[0] < target_t else x[:target_t]
            for x in values
        ]
        return torch.stack(padded)

    ensure_dir(save_dir)
    tag = make_file_tag(single_layer)
    paths = {
        "feature_train": os.path.join(save_dir, f"feature_{tag}_train.pt"),
        "feature_test": os.path.join(save_dir, f"feature_{tag}_test.pt"),
        "label_train": os.path.join(save_dir, f"labels1_{tag}_train.pt"),
        "label_test": os.path.join(save_dir, f"labels2_{tag}_test.pt"),
        "index_train": os.path.join(save_dir, f"indices_{tag}_train.pt"),
        "index_test": os.path.join(save_dir, f"indices_{tag}_test.pt"),
    }
    torch.save(stack(feature_splits["train"]), paths["feature_train"])
    torch.save(stack(feature_splits["test"]), paths["feature_test"])
    torch.save(torch.tensor(label_splits["train"], dtype=torch.float32), paths["label_train"])
    torch.save(torch.tensor(label_splits["test"], dtype=torch.float32), paths["label_test"])
    torch.save(np.asarray(index_splits["train"], dtype=np.int64), paths["index_train"])
    torch.save(np.asarray(index_splits["test"], dtype=np.int64), paths["index_test"])

    split_strategy = PITT_SPLIT_POLICY if dataset_key == "Pitt" else RUN_PROTOCOL_TAG
    metadata = {
        "format_version": 3 if dataset_key == "Pitt" else 1,
        "task_type": "mmse_regression",
        "dataset_key": dataset_key,
        "dataset": dataset_key,
        "paper_dataset": paper_dataset_name(dataset_key),
        "file_tag": tag,
        "feature_save_path": str(Path(save_dir).resolve()),
        "split_strategy": split_strategy,
        "subject_disjoint": dataset_key == "Pitt",
        "train_test_subject_overlap": 0 if dataset_key == "Pitt" else None,
        "train_audio_ids": list(range(len(train_entries))),
        "val_audio_ids": [],
        "train_entries": manifest_splits["train"],
        "test_entries": manifest_splits["test"],
    }
    metadata_path = os.path.join(save_dir, f"official_train_test_metadata_{tag}.json")
    save_json(metadata, metadata_path)
    save_json({"items": manifest_splits["train"]}, os.path.join(save_dir, "train_manifest.json"))
    save_json({"items": manifest_splits["test"]}, os.path.join(save_dir, "test_manifest.json"))
    save_csv(missing_records, os.path.join(save_dir, f"missing_mmse_{tag}.csv"))
    result = {
        "feature_save_path": save_dir,
        "file_tag": tag,
        "num_train_audios": len(train_entries),
        "num_test_audios": len(test_entries),
        "num_train_segments": len(feature_splits["train"]),
        "num_test_segments": len(feature_splits["test"]),
        "metadata_path": metadata_path,
        "paths": paths,
    }
    save_json(
        {
            "stage": "whisper_tap_mmse_finetune_export",
            "dataset_key": dataset_key,
            "fold_id": None,
            "fold_policy": "disabled",
            "run_protocol": RUN_PROTOCOL_TAG,
            "single_layer": int(single_layer),
            "file_tag": tag,
            "feature_root": str(Path(save_dir).resolve()),
            "feature_model_name": f"ft_{effective_model_name}",
            "feature_selection_name": f"single{int(single_layer)}",
            "semantics": {
                "feature_train_pt": (
                    "pitt_subject_disjoint_final_train_all_mmse_valid"
                    if dataset_key == "Pitt" else "official_train_all_mmse_valid"
                ),
                "feature_test_pt": (
                    "pitt_subject_disjoint_final_test_all_mmse_valid"
                    if dataset_key == "Pitt" else "official_test_all_mmse_valid"
                ),
                "official_test_role": "export_only_not_used_for_training_or_checkpoint_selection",
            },
            "lru_recommended_config": {
                "FEATURE_SOURCE_MODES": ["finetune"],
                "FT_MODEL_NAME": effective_model_name,
                "SINGLE_LAYERS": [int(single_layer)],
                "DATASET_OPTION": dataset_key,
                "FOLD_IDS": [],
                "RUN_PROTOCOL_TAG": RUN_PROTOCOL_TAG,
            },
        },
        os.path.join(save_dir, "feature_meta.json"),
    )
    save_json(result, os.path.join(save_dir, "export_manifest.json"))
    return result


# =============================================================================
# 6. Single-run and batch entry points
# =============================================================================

def model_save_name() -> str:
    """Normalize MODEL_NAME so output paths add the ft_ prefix exactly once."""
    name = str(MODEL_NAME).strip()
    while name.lower().startswith("ft_"):
        name = name[3:]
    if not name:
        raise ValueError("MODEL_NAME must not be empty")
    return name


def validate_config() -> None:
    if USER_NAME == "YOUR_USERNAME":
        raise ValueError(
            "Configure USER_NAME in TAP.py or export TAP_USER_NAME before running the script"
        )
    valid_datasets = set(DATASETS) & set(MMSE_META_PATHS)
    for key in DATASET_KEYS:
        if key not in valid_datasets:
            raise ValueError(f"DATASET_KEYS contains an unconfigured dataset: {key}")
        transcript_dirs = HUMAN_CHA_DIRS.get(key, {})
        missing_subsets = [
            subset for subset in ("train_cc", "train_cd")
            if not str(transcript_dirs.get(subset, "")).strip()
        ]
        if missing_subsets:
            raise ValueError(
                f"HUMAN_CHA_DIRS[{key!r}] is missing transcript paths for {missing_subsets}"
            )
    config = WhisperConfig.from_pretrained(BASE_WHISPER_MODEL, local_files_only=LOCAL_FILES_ONLY)
    for layer in SINGLE_LAYERS:
        if not 1 <= int(layer) <= int(config.encoder_layers):
            raise ValueError(f"SINGLE_LAYER={layer} is outside [1, {config.encoder_layers}]")


def run_one(
    dataset_key: str,
    single_layer: int,
    unfreeze_count: int,
) -> Dict[str, Any]:
    set_seed(SEED)
    effective_model_name = model_save_name()
    layer_name = f"single{int(single_layer)}_unfree{int(unfreeze_count)}"
    final_model_dir = os.path.join(
        MODEL_ROOT_DIR,
        dataset_key,
        f"ft_{effective_model_name}",
        layer_name,
        RUN_PROTOCOL_TAG,
    )
    tap_model_dir = os.path.join(final_model_dir, "tap_checkpoint")
    feature_dir = os.path.join(
        FEATURE_ROOT_DIR,
        dataset_key,
        f"ft_{effective_model_name}",
        layer_name,
    )
    train_entries, test_entries, missing = build_official_entries(dataset_key)
    if not train_entries or not test_entries:
        raise RuntimeError(f"{dataset_key} has no valid train/test entries")

    if SKIP_TAP_STAGE:
        if not os.path.isdir(tap_model_dir):
            raise FileNotFoundError(
                f"SKIP_TAP_STAGE=True, but the checkpoint does not exist: {tap_model_dir}"
            )
        tap_result = {"skipped": True, "checkpoint": tap_model_dir}
    else:
        tap_result = train_task_adaptive_pretraining(dataset_key, train_entries, tap_model_dir)

    if SKIP_MMSE_STAGE:
        if not os.path.isdir(final_model_dir):
            raise FileNotFoundError(
                f"SKIP_MMSE_STAGE=True, but the checkpoint does not exist: {final_model_dir}"
            )
        processor = WhisperProcessor.from_pretrained(final_model_dir, local_files_only=True)
        whisper = WhisperModel.from_pretrained(final_model_dir, local_files_only=True).to(DEVICE)
        mmse_result = {"skipped": True, "checkpoint": final_model_dir}
    else:
        regressor, processor, mmse_result = train_mmse(
            train_entries,
            tap_model_dir,
            final_model_dir,
            single_layer,
            unfreeze_count,
        )
        whisper = regressor.whisper

    export_result = None
    if not SKIP_FEATURE_EXPORT:
        export_result = export_features(
            dataset_key,
            train_entries,
            test_entries,
            processor,
            whisper,
            feature_dir,
            single_layer,
            missing,
            effective_model_name,
        )

    experiment_config = {
        "dataset_key": dataset_key,
        "base_whisper_model": BASE_WHISPER_MODEL,
        "model_name": effective_model_name,
        "method": "TAP",
        "method_full_name": "Task-Adaptive Pretraining",
        "run_protocol": RUN_PROTOCOL_TAG,
        "single_layer": single_layer,
        "unfreeze_encoder_layers": unfreeze_count,
        "label_source": "human_cha",
        "human_cha_dirs": HUMAN_CHA_DIRS.get(dataset_key, {}),
        "tap_model_dir": tap_model_dir,
        "finetuned_whisper_dir": final_model_dir,
        "feature_save_path": feature_dir,
        "official_test_used_for_training": False,
        "tap_result": tap_result,
        "mmse_result": mmse_result,
        "export_result": export_result,
    }
    save_json(experiment_config, os.path.join(final_model_dir, "run_manifest.json"))
    save_json(experiment_config, os.path.join(feature_dir, "experiment_config.json"))
    return experiment_config


def run_all_experiments() -> None:
    validate_config()
    effective_model_name = model_save_name()
    print("[CONFIG] METHOD = TAP (Task-Adaptive Pretraining)")
    print("[CONFIG] DATASET_KEYS =", DATASET_KEYS)
    print("[CONFIG] BASE_WHISPER_MODEL =", BASE_WHISPER_MODEL)
    print("[CONFIG] MODEL SAVE NAME =", f"ft_{effective_model_name}")
    print("[CONFIG] DEVICE =", DEVICE)
    rows = []
    for dataset_key in DATASET_KEYS:
        for unfreeze_count in UNFREEZE_ENCODER_LAYERS_LIST:
            for single_layer in SINGLE_LAYERS:
                print("=" * 100)
                print(
                    f"[RUN] method=TAP | dataset={dataset_key} | single={single_layer} | "
                    f"unfree={unfreeze_count} | save_name=ft_{effective_model_name}"
                )
                result = run_one(
                    dataset_key,
                    int(single_layer),
                    int(unfreeze_count),
                )
                rows.append({
                    "dataset_key": dataset_key,
                    "method": "TAP",
                    "label_source": "human_cha",
                    "model_name": effective_model_name,
                    "single_layer": int(single_layer),
                    "unfreeze_encoder_layers": int(unfreeze_count),
                    "model_dir": result["finetuned_whisper_dir"],
                    "feature_save_path": result["feature_save_path"],
                })
    summary_dir = os.path.join(
        FEATURE_ROOT_DIR,
        "_batch_summaries",
        f"ft_{effective_model_name}",
    )
    ensure_dir(summary_dir)
    summary_payload = {
        "format_version": 2,
        "dataset_keys": DATASET_KEYS,
        "method": "TAP",
        "method_full_name": "Task-Adaptive Pretraining",
        "label_source": "human_cha",
        "model_name": effective_model_name,
        "run_protocol": RUN_PROTOCOL_TAG,
        "fold_ids": [],
        "fold_policy": "disabled",
        "single_layers": SINGLE_LAYERS,
        "unfreeze_encoder_layers_list": UNFREEZE_ENCODER_LAYERS_LIST,
        "runs": rows,
    }
    save_json(
        summary_payload,
        os.path.join(summary_dir, "batch_run_summary_multi_dataset_unfreeze.json"),
    )
    save_csv(rows, os.path.join(summary_dir, "batch_run_summary_multi_dataset_unfreeze.csv"))
    print(f"[TAP DONE] {summary_dir}")


if __name__ == "__main__":
    run_all_experiments()
