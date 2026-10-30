# -*- coding: utf-8 -*-
"""
CPFT_2class_stage2.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.
============================================================

AD-aware CPFT is exported by Stage2: official train cluster and pseudo-label.

This script is aligned to CPFT_2class_stage1.py and serves as:
1. Reads from Stage1 classification output directory:
       stage1_class/last.ckpt
       stage1_class/train_manifest.json
2. Only the Whisper token tabulation of FAISS KMeans;
3. All 30 seconds for official train, 30% overlapping chunks generation of cluster pseudo labels;
4. Output Stage3 classification read directly:
       stage2_clusters/train/all-clus.json
       stage2_clusters/train/chunks_manifest.json
5. official test Not to enter Stage2 at all;
6. target_label Saved in manifest for use in Stage3 AD subsidiary categories,
   Do not enter the punctuation, distance calculation or cluster label distribution of KMeans.

Definitions of the second classification
----------
    0 = Health/Creation Group (cc)
    1 = Disease Group (cd)

Data protocol
--------
- Stage1 / Stage2 / Stage3 Use official train only;
- official test (a) The extraction of features only in the final CPFT_2class_extract.py;
- Do not create Stage2 val/test cluster directory;
- Do not read MMSE metadata;
- Do not execute missing MMSE filters.

Audio Processing
--------
- All original audio is read through librosa.load(..., sr=16000, mono=True);
- Unified to 16 kHz, single channel, float32;
- Stage2 Press 30 seconds, 30% to duplicate chunks;
- PROCESS-2 Retain only __CTD.wav.

Save Path
--------
/home/user/Alzheimer_code_jax_class/
└── finetune/ft_CPFT/
    └── <DATASET>/
        └── CPFT_lr5e-5_epo5/
            └── singleX_unfreeZ/
                ├── stage1_class/
                ├── stage2_clusters/
                │   └── train/
                └── meta/

Run Example
--------
conda activate whisper_cpft_faiss
python CPFT_2class_stage2.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.

Note: The location of the script document does not affect the result directory; the result directory is determined by CPFT_ROOT.
"""

# ============================================================
# 0) Environmental settings that must be completed before import torch
# ============================================================
import os

# A visible binder card. The external CUDA_VISIBLE_DEVICES will determine when set to None.
GPU_ID = 5
if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================
# 1) Remaining Imports
# ============================================================
import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.dataloader import default_collate

from transformers import WhisperFeatureExtractor, WhisperModel

import faiss
from tqdm import tqdm

# ============================================================
# 2) Parameters Settings: must be aligned to CPFT_2class_stage1.py / CPFT_2class_stage3.py
# ============================================================

# The current AD Second Classification Project Root Catalogue.
CLASS_PROJECT_ROOT = "/home/user/Alzheimer_code_jax_class"

# CPFT An intermediate result root directory.
PROJECT_ROOT = os.path.join(CLASS_PROJECT_ROOT, "finetune")
CPFT_ROOT = os.path.join(PROJECT_ROOT, "ft_CPFT")

# Four official data sets.
DATASET_OPTIONS = ["ADReSS-IS2020"]


def _canonical_dataset_option(value: str) -> str:
    """Map the public ADReSS name to the existing internal dataset key."""
    value = str(value).strip()
    return "ADReSS-IS2020" if value == "ADReSS" else value
DATASET_OPTION = _canonical_dataset_option(DATASET_OPTIONS[0])

# Indicates that the layer corresponds to the number of unfrozen layers Stage1 checkpoint.
LAYERS_TO_RUN = [12]
STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN

UNFREEZE_ENCODER_LAYERS_LIST = [7,8,9,10,11,12]
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# official_train only protocol; enter meta only, without adding a path layer.
RUN_PROTOCOL_TAG = "official_train_only"

# Stage1/2/3/extract There must be unity.
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5"

# Stage2 is managed separately from the Stage1 source suffix.
STAGE1_SOURCE_PATH_PARAM_TAGS: List[str] = []
PATH_PARAM_TAGS: List[str] = []

# Whisper Local model.
WHISPER_MODEL_NAME = "/home/user/whisper_lru_3/whisper_small"

# Manually assign Stage1 checkpoint; None indicates automatic search for stage1_class/last.ckpt.
STAGE1_CKPT_OVERRIDE: Optional[str] = None

# 30 Seconds, 30% overlap, must match Stage3 and final features.
SEGMENT_SEC = 30
OVERLAP_RATE = 0.3
CHUNK_KEY_VERSION = "wav_path_chunk_start_end_v1"

# Cluster head.
KS = [32, 64]

# KMeans token Sampling.
# - ratio:Each chunk sample scale;
# - per_chunk:Each chunk sample up to SAMPLE_PER_UTT token;
# - per_utt_cap:All chunks samples of the same original audio to a maximum of SAMPLE_PER_UTT token;
# - all:Use all valid token.
# Recommended per_utt_cap to avoid long audio lead clusters.
SAMPLE_MODE = "per_utt_cap"
SAMPLE_RATIO: Optional[float] = None
SAMPLE_PER_UTT = 128

TRAIN_BATCH_SIZE = 4
INFER_BATCH_SIZE = 4
NUM_WORKERS = 4

# FAISS Settings.
FAISS_GPU = False
FAISS_NREDO = 1
FAISS_MAX_POINTS_PER_CENTROID = 2000
FAISS_VERBOSE = False
KMEANS_MAX_ITER = 200

# Export settings.
SEED = 42
PAD_TO_FULL_LENGTH = True
SAVE_EXPORT_METADATA = True


# Selected cluster quality reports train-only.
# Checks only the key integrity and the collapse of the financial train,
# Do not read the official test, or do the Train/val JS diversence.
ENABLE_STAGE_QUALITY_EVAL = False
QUALITY_REPORT_FILENAME = "quality_report.json"
QUALITY_SUMMARY_FILENAME = "quality_summary.csv"
QUALITY_DECISION_FILENAME = "quality_decision.txt"

# ============================================================
# 3) Constant
# ============================================================
SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
IGNORE_INDEX = -100
SEGMENT_LENGTH = int(float(SEGMENT_SEC) * SAMPLE_RATE)
CLASS_NAMES = {
    0: "healthy_control",
    1: "ad_or_disease",
}

def seed_everything_for_repro(seed: int) -> None:
    """
    Standardize the setting of random seeds to ensure, to the extent possible, that experiments are re-emerging.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    """
    Can not create folder lock on %s: %s
    """
    os.makedirs(path, exist_ok=True)


def canonical_wav_path(path: str) -> str:
    """
    The uniform wav path indicates that avoidance of inconsistencies between relative/absolute paths leads to a failure of matching.
    """
    return str(Path(str(path)).expanduser().resolve())

def save_json(obj: Dict[str, Any], path: str) -> None:
    """
    Saves json and automatically creates a parent directory.
    """
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(text: str, path: str) -> None:
    """
    Saves the pure text.
    """
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def compute_valid_token_length(wav_len_samples: int) -> int:
    """
    Maps the `real and valid waveform sample points' into an effective token length of Whisper encoder.

    Whisper:
        mel Frame ~ ceil(len / hop)
        token Length ~ (mel_frames + 1) // 2
    """
    if wav_len_samples <= 0:
        return 1
    n_mel_valid = int(math.ceil(wav_len_samples / float(HOP_LENGTH)))
    n_mel_valid = min(MAX_MEL_FRAMES, max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)

def _clean_path_tag(x) -> str:
    """
    Clears the path parameter labels and avoids the entry of spaces, slashes, etc. into the directory name.
    """
    import re

    if x is None:
        return ""

    s = str(x).strip()
    if s in ["", "None", "none", "null", "Null"]:
        return ""

    # Common writing cleanup:
    #   "ks=64" -> "ks64"
    #   "mask prob 0.06" -> "maskprob0.06"
    s = s.replace("=", "").replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_.-]+", "", s)
    s = s.strip("_-.")
    return s

def build_path_param_suffix(path_param_tags=None) -> str:
    """
    Construct the path suffix according to PATH_PARAM_TAGS.

    Example:
        []                  -> ""
        ["ks64"]            -> "ks64"
        ["ks64", "maux005"] -> "ks64_maux005"
    """
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
    """
    Construct a layer directory name for CPFT that is used uniformly.

    Example:
        single_layer=12, unfreeze_layer_count=1, PATH_PARAM_TAGS=[]
            -> single12_unfree1

        single_layer=12, unfreeze_layer_count=1, PATH_PARAM_TAGS=["ks64"]
            -> single12_unfree1_ks64
    """
    base = f"single{int(single_layer)}_unfree{int(unfreeze_layer_count)}"
    suffix = build_path_param_suffix(path_param_tags)
    if suffix == "":
        return base
    return f"{base}_{suffix}"

def make_chunk_key(wav_path: str, chunk_id: int, start: int, end: int) -> str:
    """
    Construct chunk-level unique key.

    key contains:
    - Normative wav_path
    - chunk_id
    - 16k Sample point start/end at sampling rate

    Example:
        /path/S002.wav::chunk0001::start336000::end816000
    """
    wav_path = canonical_wav_path(wav_path)
    return f"{wav_path}::chunk{int(chunk_id):04d}::start{int(start)}::end{int(end)}"

def canonical_label_key(key: str) -> str:
    """
    Regularizes the key used to match a false label between the Step2/Stage3.

    - wav-level Old key: Direct canonical_wav_path.
    - chunk-level New key: Only the wav_path before *chunk is regularized and the suffix remains unchanged.
    """
    key = str(key)
    if "::chunk" in key:
        wav_part, suffix = key.split("::chunk", 1)
        return canonical_wav_path(wav_part) + "::chunk" + suffix
    return canonical_wav_path(key)

def split_segment_boundaries(num_samples: int, segment_length: int, overlap_rate: float) -> List[Tuple[int, int, int]]:
    """
    Resumes the 30s+ overlap cutting idea of frozen.py to generate a chunk boundary.

    Return list elements:
        (chunk_id, start, end)

    The start/end here are all sample point coordinates after re-sampling to 16k.
    Whether or not the last paragraph is retained follows the logic of frozen.py:
        len(y) % step_size > overlap_rate * segment_length , and the end paragraph is retained.
    """
    num_samples = int(num_samples)
    segment_length = int(segment_length)
    overlap_rate = float(overlap_rate)

    if segment_length <= 0:
        raise ValueError(f"segment_length Must be > 0, now ={segment_length}")
    step_size = int(segment_length * (1.0 - overlap_rate))
    if step_size <= 0:
        raise ValueError(f"overlap_rate={overlap_rate} Caused by step_size <= 0.")

    if num_samples <= 0:
        return [(0, 0, segment_length)]

    total_segments = int(num_samples / step_size) + (
        1 if num_samples % step_size > overlap_rate * segment_length else 0
    )
    total_segments = max(1, int(total_segments))

    boundaries = []
    for chunk_id in range(total_segments):
        start = int(chunk_id * step_size)
        end = int(start + segment_length)
        boundaries.append((int(chunk_id), start, end))
    return boundaries

def load_audio_16k(wav_path: str) -> np.ndarray:
    """
    Read audio as baseline: librosa.load(sr=16000,, mono=True,, dtype=float32).
    """
    wav_path = canonical_wav_path(wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"[Stage2-CLASS] Could not close temporary folder: %s {wav_path}")
    wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    return np.asarray(wav, dtype=np.float32).reshape(-1)

def load_chunk_with_padding_from_wav(y: np.ndarray, start: int, end: int, chunk_id: int) -> Tuple[np.ndarray, int]:
    """
    Remove a chunk from the entire audio sampled to 16k and follow the end of frozen.py to complete the logic.

    - General chunk: Directly take y[start:end] with a length 30s.
    - The first paragraph, which was not sufficiently audio to 30s: If you do not complete it, give it to WhisperFeatureExtractor padding, valid_len= the true length.
    - Non-first end is insufficient 30s: The previous end is audio-filled and the logic of frozen.py is consistent;
      This part of the completion is also actual audio, so valid_len is recorded as the length of the completion.
    """
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

    # valid_len Record real and valid audio length; then completes DataLoader stack with zero to fixed 30s
    valid_len = int(min(len(segment), SEGMENT_LENGTH))
    if len(segment) < SEGMENT_LENGTH:
        pad = np.zeros((SEGMENT_LENGTH - len(segment),), dtype=np.float32)
        segment = np.concatenate([segment.astype(np.float32, copy=False), pad], axis=0)

    return segment.astype(np.float32, copy=False), valid_len

def load_chunk_with_padding(wav_path: str, start: int, end: int, chunk_id: int) -> Tuple[np.ndarray, int]:
    """
    Read wav and return to 30s waveform with the active length specified chunk.
    """
    y = load_audio_16k(wav_path)
    return load_chunk_with_padding_from_wav(y, start=start, end=end, chunk_id=chunk_id)

def validate_stage1_class_manifest_entries(
    entries: List[Dict[str, Any]],
    dataset_option: str,
) -> Dict[str, Any]:
    """
    Check Stage1 classification manifest.

    It must be satisfied that:
    1. Each entry has access to wav_path;
    2. target_label For 0/1 only;
    3. audio_id Only and stable;
    4. official train (a) The two categories are covered;
    5. PROCESS-2 Only contains CTD audio.
    """
    if len(entries) == 0:
        raise ValueError("Stage1 class manifest It's empty.")

    labels: List[int] = []
    audio_ids: List[int] = []
    wav_paths: List[str] = []

    for index, row in enumerate(entries):
        wav_path = canonical_wav_path(row.get("wav_path", ""))
        if not os.path.isfile(wav_path):
            raise FileNotFoundError(
                f"Stage1 manifest Article {index} The audio does not exist:{wav_path}"
            )

        if row.get("target_label", None) in [None, ""]:
            raise ValueError(
                f"Stage1 manifest Article {index} Article missing target_label:{row}"
            )
        try:
            label = int(row["target_label"])
        except Exception as exc:
            raise ValueError(
                f"Stage1 manifest Article {index} Article target_label Unable to transfer in whole:{row}"
            ) from exc
        if label not in {0, 1}:
            raise ValueError(
                f"Stage1 manifest Article {index} The label is not 0/1:{label}"
            )

        audio_id_raw = row.get(
            "official_audio_id",
            row.get("audio_id", None),
        )
        if audio_id_raw in [None, ""]:
            raise ValueError(
                f"Stage1 manifest Article {index} Article missing audio_id:{row}"
            )
        audio_id = int(audio_id_raw)

        if str(dataset_option).strip().lower() == "process-2":
            if re.search(r"__CTD$", Path(wav_path).stem, flags=re.IGNORECASE) is None:
                raise ValueError(
                    "PROCESS-2 Stage1 manifest Include non-CTD audio:"
                    f"{wav_path}"
                )

        labels.append(label)
        audio_ids.append(audio_id)
        wav_paths.append(wav_path)

    if len(set(audio_ids)) != len(audio_ids):
        raise ValueError("Stage1 class manifest There is a repetition of audio_id.")
    if len(set(wav_paths)) != len(wav_paths):
        raise ValueError("Stage1 class manifest There is a repetition of wav_path.")

    unique_labels = sorted(set(labels))
    if unique_labels != [0, 1]:
        raise ValueError(
            f"official train Must contain both 0/1 categories, current ={unique_labels}"
        )

    stats = {
        "audio_total": int(len(labels)),
        "class0": int(sum(x == 0 for x in labels)),
        "class1": int(sum(x == 1 for x in labels)),
        "class1_ratio": float(np.mean(np.asarray(labels) == 1)),
    }
    print(
        "[Stage2-CLASS][MANIFEST] "
        f"audio_total={stats['audio_total']} | "
        f"class0={stats['class0']} | "
        f"class1={stats['class1']}"
    )
    return stats


def _is_process2_ctd_manifest_entry(
    row: Dict[str, Any],
    dataset_key: Optional[str] = None,
) -> bool:
    """PROCESS-2 Secure filtering: Only audio at the end of __CTD with the filename stem."""
    if dataset_key is None:
        dataset_key = DATASET_OPTION
    if str(dataset_key).strip().lower() != "process-2":
        return True

    wav_path = row.get("wav_path", row.get("wav_name", ""))
    return re.search(
        r"__CTD$",
        Path(str(wav_path)).stem,
        flags=re.IGNORECASE,
    ) is not None


def load_stage1_class_train_manifest(
    stage1_save_dir: str,
    dataset_option: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Reads the official train saved by the CPFT_2class_stage1.py.

    Stage2 Do not scan the dataset directory, or recreate the label, to ensure Stage1/2/3
    Use identical audio, target_label and stabilize audio_id.
    """
    manifest_path = os.path.join(
        str(stage1_save_dir),
        "train_manifest.json",
    )
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            "Stage1 class train_manifest.json Does not exist: \n"
            f"{manifest_path}\n"
            "Please run CPFT_2class_stage1.py."
        )

    with open(manifest_path, "r", encoding="utf-8") as file:
        obj = json.load(file)

    if isinstance(obj, dict) and isinstance(obj.get("entries"), list):
        raw_entries = obj["entries"]
    elif isinstance(obj, list):
        raw_entries = obj
    else:
        raise ValueError(
            f"Stage1 class manifest:{manifest_path}"
        )

    entries: List[Dict[str, Any]] = []
    dropped_non_ctd = 0
    for raw_row in raw_entries:
        if not isinstance(raw_row, dict):
            continue

        row = copy.deepcopy(raw_row)
        wav_path = row.get("wav_path", None)
        if wav_path in [None, ""]:
            continue

        row["wav_path"] = canonical_wav_path(wav_path)
        row["wav_name"] = str(
            row.get("wav_name", Path(row["wav_path"]).name)
        )
        row["split"] = "train"

        if row.get("target_label", None) in [None, ""]:
            raise ValueError(
                "Stage1 class manifest entry Missing target_label:"
                f"{row}"
            )
        row["target_label"] = int(row["target_label"])

        audio_id_raw = row.get(
            "official_audio_id",
            row.get("audio_id", None),
        )
        if audio_id_raw in [None, ""]:
            raise ValueError(
                "Stage1 class manifest entry Missing audio_id:"
                f"{row}"
            )
        audio_id = int(audio_id_raw)
        row["official_audio_id"] = audio_id
        row["audio_id"] = audio_id

        if not _is_process2_ctd_manifest_entry(
            row,
            dataset_key=dataset_option,
        ):
            dropped_non_ctd += 1
            continue

        entries.append(row)

    entries = sorted(
        entries,
        key=lambda row: int(row["audio_id"]),
    )

    stats = validate_stage1_class_manifest_entries(
        entries,
        dataset_option=dataset_option,
    )
    stats["manifest_path"] = manifest_path
    stats["raw_entries"] = int(len(raw_entries))
    stats["dropped_non_ctd"] = int(dropped_non_ctd)

    if str(dataset_option).strip().lower() == "process-2":
        print(
            "[PROCESS-2][CTD] Stage2 loaded Stage1 class manifest | "
            f"raw={len(raw_entries)} | kept={len(entries)} | "
            f"dropped_non_ctd={dropped_non_ctd}"
        )

    return entries, stats


def build_chunk_entries_from_official_entries(
    entries: List[Dict[str, Any]],
    split_name: str = "train",
) -> List[Dict[str, Any]]:
    """
    Expands the wave-level of financial transactions of Stage1 class as
    30 Seconds, 30% overlapping chunk-level entries.

    target_label Saved with audio_id only with manifest for use with Stage3;
    Stage2 KMeans , the DataLoader does not return the target_label, so the diagnostic labels are not used for clusters.
    """
    split_name = str(split_name).strip().lower()
    if split_name != "train":
        raise ValueError(
            "For the time being Stage2 only official train,"
            f"Copy that.{split_name}"
        )

    chunk_entries: List[Dict[str, Any]] = []

    for wav_row in tqdm(
        entries,
        desc="Build official train chunks",
        dynamic_ncols=True,
    ):
        wav_row = copy.deepcopy(wav_row)
        wav_path = canonical_wav_path(wav_row["wav_path"])
        wav_name = str(
            wav_row.get("wav_name", Path(wav_path).name)
        )
        target_label = int(wav_row["target_label"])
        audio_id = int(
            wav_row.get(
                "official_audio_id",
                wav_row.get("audio_id"),
            )
        )

        y = load_audio_16k(wav_path)
        num_samples = int(len(y))
        boundaries = split_segment_boundaries(
            num_samples=num_samples,
            segment_length=SEGMENT_LENGTH,
            overlap_rate=OVERLAP_RATE,
        )

        for chunk_id, start, end in boundaries:
            _segment, valid_len = load_chunk_with_padding_from_wav(
                y,
                start=start,
                end=end,
                chunk_id=chunk_id,
            )
            chunk_key = make_chunk_key(
                wav_path=wav_path,
                chunk_id=chunk_id,
                start=start,
                end=end,
            )

            row = copy.deepcopy(wav_row)
            row.update(
                {
                    "split": "train",
                    "wav_path": wav_path,
                    "wav_name": wav_name,
                    "official_audio_id": audio_id,
                    "audio_id": audio_id,
                    "target_label": target_label,
                    "chunk_key": chunk_key,
                    "chunk_id": int(chunk_id),
                    "chunk_start": int(start),
                    "chunk_end": int(end),
                    "chunk_valid_len": int(valid_len),
                    "parent_num_samples_16k": num_samples,
                    "segment_sec": float(SEGMENT_SEC),
                    "overlap_rate": float(OVERLAP_RATE),
                    "chunk_key_version": str(CHUNK_KEY_VERSION),
                }
            )
            chunk_entries.append(row)

    if len(chunk_entries) == 0:
        raise ValueError("official train chunk_entries It's empty.")

    # Check all chunks labels for the same audio.
    audio_to_label: Dict[int, int] = {}
    for row in chunk_entries:
        audio_id = int(row["audio_id"])
        label = int(row["target_label"])
        if audio_id in audio_to_label and audio_to_label[audio_id] != label:
            raise ValueError(
                f"audio_id={audio_id} The chunks label appears to be different."
            )
        audio_to_label[audio_id] = label

    chunk_labels = np.asarray(
        [int(row["target_label"]) for row in chunk_entries],
        dtype=np.int64,
    )
    print(
        "[Stage2-CLASS][CHUNKS] "
        f"total={len(chunk_entries)} | "
        f"class0={int(np.sum(chunk_labels == 0))} | "
        f"class1={int(np.sum(chunk_labels == 1))} | "
        f"audio_total={len(audio_to_label)}"
    )

    return chunk_entries

def torch_load_compatible(
    path: str,
    map_location: str = "cpu",
):
    """
    torch.load(weights_only=True) Default act compatible with PyTorch 2.6.

    Lightning checkpoint Generated by the project itself, the source is credible and requires a full dictionary.
    Therefore, display setting weights_only=False; old version PyTorch returns when this parameter is not supported.
    """
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def find_stage1_checkpoint(
    stage1_save_dir: str,
    override_ckpt: Optional[str] = None,
) -> str:
    """Read stage1_class/last.ckpt in preference, and return to the latest checkpoint if not found."""
    if override_ckpt not in [None, "", "None"]:
        checkpoint_path = str(override_ckpt)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f"STAGE1_CKPT_OVERRIDE Cannot initialise Evolution's mail component.{checkpoint_path}"
            )
        return str(Path(checkpoint_path).resolve())

    stage1_dir = Path(stage1_save_dir)
    if not stage1_dir.is_dir():
        raise FileNotFoundError(
            f"Stage1 class Directory does not exist:{stage1_save_dir}"
        )

    last_checkpoint = stage1_dir / "last.ckpt"
    if last_checkpoint.is_file():
        return str(last_checkpoint.resolve())

    checkpoints = sorted(
        stage1_dir.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if len(checkpoints) > 0:
        return str(checkpoints[0].resolve())

    raise FileNotFoundError(
        f"Yes. {stage1_save_dir} Stage1 class checkpoint not found below."
    )


def load_stage1_ckpt_into_whisper_model(
    whisper_model: WhisperModel,
    checkpoint_path: str,
) -> None:
    """
    Only load WhisperModel weights in Stage1 classification checkpoint into Stage2.

    Stage1 The utterance head, frame head and attendtive pool are ignored;
    target_label Do not participate in Stage2 KMeans.
    """
    checkpoint = torch_load_compatible(
        checkpoint_path,
        map_location="cpu",
    )
    state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(
            f"Stage1 checkpoint state_dict Not dict:{checkpoint_path}"
        )

    target_state = whisper_model.state_dict()
    candidate_prefixes = [
        "whisper.model.",
        "model.",
        "",
    ]

    best_loaded: Dict[str, torch.Tensor] = {}
    best_prefix: Optional[str] = None

    for prefix in candidate_prefixes:
        loaded: Dict[str, torch.Tensor] = {}
        for key, value in state.items():
            key = str(key)
            if prefix != "" and not key.startswith(prefix):
                continue
            new_key = key[len(prefix):] if prefix != "" else key
            if (
                new_key in target_state
                and hasattr(value, "shape")
                and tuple(target_state[new_key].shape) == tuple(value.shape)
            ):
                loaded[new_key] = value

        if len(loaded) > len(best_loaded):
            best_loaded = loaded
            best_prefix = prefix

    if len(best_loaded) == 0:
        raise RuntimeError(
            "No match from Stage1 classification checkpoint to Whisper argument: \n"
            f"{checkpoint_path}"
        )

    missing, unexpected = whisper_model.load_state_dict(
        best_loaded,
        strict=False,
    )
    encoder_loaded = sum(
        key.startswith("encoder.") for key in best_loaded
    )
    if encoder_loaded == 0:
        raise RuntimeError(
            "Stage1 checkpoint No encoder parameters were loaded."
        )

    print(
        "[Stage1-CLASS-CKPT] loaded | "
        f"prefix={best_prefix!r} | "
        f"matched={len(best_loaded)} | "
        f"encoder_matched={encoder_loaded} | "
        f"missing={len(missing)} | unexpected={len(unexpected)}"
    )

def build_stage2_run_config(
    dataset_option: str,
    rep_layer: int,
) -> Dict[str, Any]:
    """
    Construct Stage2 classification to run paths.

    Current path:
        CPFT_ROOT/<DATASET>/<MODEL_ROOT_TAG>/<singleX_unfreeZ>/
            stage1_class/
            stage2_clusters/train/
            meta/

    """
    rep_layer = int(rep_layer)

    layer_name = build_layer_name(
        single_layer=rep_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )
    stage1_layer_name = build_layer_name(
        single_layer=rep_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=STAGE1_SOURCE_PATH_PARAM_TAGS,
    )

    run_root = os.path.join(
        CPFT_ROOT,
        str(dataset_option),
        MODEL_ROOT_TAG,
        layer_name,
    )
    stage1_run_root = os.path.join(
        CPFT_ROOT,
        str(dataset_option),
        MODEL_ROOT_TAG,
        stage1_layer_name,
    )

    stage1_save_dir = os.path.join(
        stage1_run_root,
        "stage1_class",
    )
    stage2_save_dir = os.path.join(
        run_root,
        "stage2_clusters",
    )
    stage2_train_dir = os.path.join(
        stage2_save_dir,
        "train",
    )
    meta_save_dir = os.path.join(run_root, "meta")

    return {
        "task": "ad_binary_classification",
        "stage": "stage2_clustering",
        "class_mapping": {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
        },
        "CLASS_PROJECT_ROOT": CLASS_PROJECT_ROOT,
        "PROJECT_ROOT": PROJECT_ROOT,
        "CPFT_ROOT": CPFT_ROOT,
        "DATASET_OPTION": str(dataset_option),
        "REP_LAYER": rep_layer,
        "UNFREEZE_ENCODER_LAYERS": int(
            UNFREEZE_ENCODER_LAYERS
        ),
        "RUN_PROTOCOL_TAG": str(RUN_PROTOCOL_TAG),
        "official_train_used": True,
        "official_test_used": False,
        "target_label_used_by_kmeans": False,
        "LAYER_TAG": layer_name,
        "LAYER_NAME": layer_name,
        "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(
            PATH_PARAM_TAGS
        ),
        "STAGE1_LAYER_NAME": stage1_layer_name,
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(
            STAGE1_SOURCE_PATH_PARAM_TAGS
        ),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(
            STAGE1_SOURCE_PATH_PARAM_TAGS
        ),
        "RUN_ROOT": run_root,
        "STAGE1_RUN_ROOT": stage1_run_root,
        "STAGE1_SAVE_DIR": stage1_save_dir,
        "STAGE1_DIR": stage1_save_dir,
        "STAGE2_SAVE_DIR": stage2_save_dir,
        "STAGE2_TRAIN_DIR": stage2_train_dir,
        "META_SAVE_DIR": meta_save_dir,
    }

class ChunkAudioDataset(data.Dataset):
    """
    Stage2 chunk-level Data sets.

    Enter chunk-level entries, each entry containing at least:
    - wav_path
    - wav_name
    - chunk_key
    - chunk_id
    - chunk_start
    - chunk_end

    Each sample returns directly to 30s chunk and no longer does repeat / center crop.
    """
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = [copy.deepcopy(x) for x in entries]
        if len(self.entries) == 0:
            raise ValueError("ChunkAudioDataset Received empty entries, unable to construct data sets.")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        row = self.entries[i]
        wav_path = canonical_wav_path(row["wav_path"])
        wav_name = str(row.get("wav_name", Path(wav_path).name))
        chunk_key = canonical_label_key(row["chunk_key"])
        chunk_id = int(row["chunk_id"])
        start = int(row["chunk_start"])
        end = int(row["chunk_end"])

        wav_30, valid_len = load_chunk_with_padding(
            wav_path=wav_path,
            start=start,
            end=end,
            chunk_id=chunk_id,
        )

        return wav_30, int(valid_len), chunk_key, wav_name

    def repeat30s_collate(self, batch):
        """
        Compatible with old DataLoader call name.

        Each of the data in batch is now 30s chunk so only default_collate is done here.
        Return:
            wav_30      : [B, 480000] or enter before padded in the first segment of short audio
            valid_len   : [B]
            chunk_keys  : [B]
            wav_names   : [B]
        """
        return default_collate(batch)

class WhisperEncoderWrapper(nn.Module):
    """
    Outputs the hop_layer layer Hidden states, shape [B, T(=1500), C].

    Note:
    - Stage2 The default is not to train Whisper and is used only for sampling for clustering, so all parameters are frozen here.
    - Support the construction of token level attention_mask based on valid_wav_lens to avoid the repeat/pad area being considered effective token.
    """
    def __init__(self, model_name: str = "openai/whisper-small", rep_layer: int = 12):
        super().__init__()
        self.model = WhisperModel.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.hidden_size = self.model.config.d_model
        self.rep_layer = int(rep_layer)

        # Stage2 Cluster default is non-training, frozen uniformly
        for p in self.model.parameters():
            p.requires_grad = False

    def _build_attn_mask(self, input_features: torch.Tensor, valid_wav_lens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Construct token Level attention_mask: [B, tok_T], True= visible, False= invisible.
        """
        if valid_wav_lens is None:
            return None
        if not torch.is_tensor(valid_wav_lens):
            valid_wav_lens = torch.tensor(valid_wav_lens)

        vwl = valid_wav_lens.detach().cpu().long().view(-1)
        B = input_features.size(0)
        if vwl.numel() != B:
            raise RuntimeError(f"valid_wav_lens Number inconsistent with batch: expect={B}, got={vwl.numel()}")

        mel_T = int(input_features.size(-1))
        tok_T = (mel_T + 1) // 2

        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(vwl[i].item()))
            valid_Ts.append(min(vt, tok_T))
        valid_Ts = torch.tensor(valid_Ts, device=input_features.device, dtype=torch.long)

        t = torch.arange(tok_T, device=input_features.device).view(1, tok_T).expand(B, tok_T)
        return (t < valid_Ts.view(B, 1)).bool()

    def forward_layerk(
        self,
        wav: torch.Tensor,
        sampling_rate: int = 16000,
        valid_wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract the current Whisper hidden states from rep_layer.
        """
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

        enc_out = self.model.encoder(
            input_features=input_features,
            attention_mask=attn_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        hs = enc_out.hidden_states

        if self.rep_layer < 1 or self.rep_layer >= len(hs):
            raise ValueError(f"rep_layer={self.rep_layer} illegal;hidden_states len={len(hs)}")

        return hs[self.rep_layer]

def collect_train_samples(
    model: WhisperEncoderWrapper,
    loader,
    device: str,
    sample_ratio: Optional[float] = None,
    sample_per_utt: Optional[int] = None,
    sample_mode: str = "ratio",
) -> np.ndarray:
    """
    Draw from official train the form token and form a training sample matrix for KMeans X_train: [N, C].

    sample_mode:
      - "ratio":
          Original logic. Each 30s chunk random ceil (vt * sample_ratio) from valid tokens.
      - "per_chunk":
          A maximum of 30 shunk per sample_per_utt token
      - "per_utt_cap":
          All chunks from the original Article wav together shall be drawn up to a maximum of sample_per_utt token.
          It is recommended to avoid long audio / and more chunk audio-influence KMeans.
      - "all":
          Use all valid tokens.

    Current data are chunk-level:
      DataLoader returns to wav_30, valid_len, chunk_keys, wav_names.
      per_utt_cap Will take wav_path before::chunk as parent wav key from chunk_key.
    """
    sample_mode = str(sample_mode).strip().lower()
    valid_modes = {"ratio", "per_chunk", "per_utt_cap", "all"}
    if sample_mode not in valid_modes:
        raise ValueError(f"SAMPLE_MODE Must belong to {valid_modes},Current ={sample_mode}")

    if sample_ratio is not None:
        sample_ratio = float(sample_ratio)
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError(f"SAMPLE_RATIO Must be at (0,1], now ={sample_ratio}")

    if sample_per_utt is None:
        sample_per_utt = 0
    sample_per_utt = int(sample_per_utt)

    if sample_mode == "ratio" and sample_ratio is None:
        raise ValueError("SAMPLE_MODE='ratio' SAMPLE_RATIO cannot be None.")

    if sample_mode in {"per_chunk", "per_utt_cap"} and sample_per_utt <= 0:
        raise ValueError(
            f"SAMPLE_MODE='{sample_mode}' SAMPLE_PER_UTT must be > 0, current ={sample_per_utt}"
        )

    def _parent_key_from_chunk_key(chunk_key: str, wav_name: str) -> str:
        """
        Take original wav key from chunk_key.
        chunk_key Shapes like:
            /path/to/S001.wav::chunk0000::start0::end480000
        """
        ck = str(chunk_key)
        if "::chunk" in ck:
            return ck.split("::chunk", 1)[0]
        return str(wav_name)

    feats_all: List[np.ndarray] = []
    total_tokens = 0

    # per_utt_cap Use reservoir sampling.
    # Each parent wav retains a maximum of sample_per_utt token.
    reservoir: Dict[str, List[np.ndarray]] = {}
    seen_count: Dict[str, int] = {}

    pbar = tqdm(loader, total=len(loader), desc="Collection of training samples (official train)", dynamic_ncols=True)

    for wav_30, valid_len, chunk_keys, wav_names in pbar:
        wav_30 = wav_30.to(device)
        valid_len = valid_len.to(device)

        reps = model.forward_layerk(
            wav_30,
            sampling_rate=SAMPLE_RATE,
            valid_wav_lens=valid_len,
        )  # [B,T,C]

        B, T, C = reps.shape
        vwl_cpu = valid_len.detach().cpu().long().tolist()

        chunk_keys = list(chunk_keys)
        wav_names = list(wav_names)

        for i in range(B):
            vt = min(compute_valid_token_length(int(vwl_cpu[i])), T)
            if vt <= 0:
                continue

            x = reps[i, :vt, :].detach().cpu().numpy().astype(np.float32, copy=False)  # [vt,C]

            if sample_mode == "all":
                feats_all.append(x)
                total_tokens += int(vt)

            elif sample_mode == "ratio":
                n = int(math.ceil(vt * float(sample_ratio)))
                n = max(1, min(vt, n))

                if n >= vt:
                    feats_all.append(x)
                    total_tokens += int(vt)
                else:
                    idx = np.random.choice(vt, size=n, replace=False)
                    feats_all.append(x[idx])
                    total_tokens += int(n)

            elif sample_mode == "per_chunk":
                n = min(vt, int(sample_per_utt))

                if n >= vt:
                    feats_all.append(x)
                    total_tokens += int(vt)
                else:
                    idx = np.random.choice(vt, size=n, replace=False)
                    feats_all.append(x[idx])
                    total_tokens += int(n)

            elif sample_mode == "per_utt_cap":
                parent_key = _parent_key_from_chunk_key(
                    chunk_key=str(chunk_keys[i]),
                    wav_name=str(wav_names[i]),
                )

                if parent_key not in reservoir:
                    reservoir[parent_key] = []
                    seen_count[parent_key] = 0

                # The current chunk first takes a maximum of sample_per_utt token,
                # Write the candidate totken to the parent wav reserv.
                n_cand = min(vt, int(sample_per_utt))
                if n_cand < vt:
                    cand_idx = np.random.choice(vt, size=n_cand, replace=False)
                    cand = x[cand_idx]
                else:
                    cand = x

                for j in range(cand.shape[0]):
                    seen_count[parent_key] += 1
                    m = int(seen_count[parent_key])

                    if len(reservoir[parent_key]) < int(sample_per_utt):
                        reservoir[parent_key].append(cand[j].copy())
                    else:
                        r = np.random.randint(0, m)
                        if r < int(sample_per_utt):
                            reservoir[parent_key][r] = cand[j].copy()

        if sample_mode == "per_utt_cap":
            total_tokens = sum(len(v) for v in reservoir.values())
            pbar.set_postfix(wavs=len(reservoir), tokens=total_tokens)
        else:
            pbar.set_postfix(tokens=total_tokens)

    if sample_mode == "per_utt_cap":
        if len(reservoir) == 0:
            raise RuntimeError("per_utt_cap No training features have been collected to enable clustering.")

        feats_per_wav = []
        for parent_key in sorted(reservoir.keys()):
            arr = np.stack(reservoir[parent_key], axis=0).astype(np.float32)
            feats_per_wav.append(arr)

        X = np.concatenate(feats_per_wav, axis=0).astype(np.float32)

        print(
            f"[Stage2-CLASS][KMeansSample] mode=per_utt_cap, "
            f"num_parent_wavs={len(reservoir)}, "
            f"cap_per_wav={sample_per_utt}, "
            f"total_tokens={X.shape[0]}"
        )

    else:
        if len(feats_all) == 0:
            raise RuntimeError("No training features have been collected to enable clustering.")

        X = np.concatenate(feats_all, axis=0).astype(np.float32)

        print(
            f"[Stage2-CLASS][KMeansSample] mode={sample_mode}, "
            f"total_tokens={X.shape[0]}"
        )

    return np.ascontiguousarray(X, dtype=np.float32)

def _build_faiss_index_from_centroids(centroids: np.ndarray, use_gpu: bool):
    """
    Construct the nearest neighbour query index (IndexFlatL2) using a cluster centre.
    """
    centroids = np.ascontiguousarray(centroids.astype(np.float32))
    k, d = centroids.shape

    index = faiss.IndexFlatL2(d)

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "In the absence of faiss-gpu (in the absence of StandardGpuResources ), please install faiss-gpu or put FAISS_GPU=False."
            )
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(centroids)
    return index

def fit_faiss_kmeans(
    X_train: np.ndarray,
    k: int,
    seed: int,
    niter: int,
    nredo: int,
    use_gpu: bool,
    max_points_per_centroid: int,
    verbose: bool,
):
    """
    Training FAISS KMeans and returning:
        centroids, index
    """
    X_train = np.ascontiguousarray(X_train.astype(np.float32))
    d = int(X_train.shape[1])

    km = faiss.Kmeans(
        d=d,
        k=int(k),
        niter=int(niter),
        nredo=int(nredo),
        verbose=bool(verbose),
        gpu=bool(use_gpu),
    )

    try:
        km.seed = int(seed)
    except Exception:
        pass
    try:
        km.max_points_per_centroid = int(max_points_per_centroid)
    except Exception:
        pass

    km.train(X_train)
    centroids = km.centroids.reshape(int(k), d).astype(np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)

    index = _build_faiss_index_from_centroids(centroids, use_gpu=use_gpu)
    return centroids, index

@torch.no_grad()
def infer_and_export_official_train_json(
    model: WhisperEncoderWrapper,
    loader,
    index_list: List[Any],
    ks: List[int],
    device: str,
    output_dir: str,
    pad_to_full: bool,
    entries: List[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Generates a false label for all of the chanks.

    target_label Only write keys_meta/chunks_manifest for Stage3,
    Do not enter WhisperEncoderWrapper and FAISS index.
    """
    ensure_dir(output_dir)

    entries_by_key: Dict[str, Dict[str, Any]] = {}
    for row in entries:
        key = canonical_label_key(row["chunk_key"])
        if key in entries_by_key:
            raise ValueError(f"chunk_key Repeat:{key}")
        entries_by_key[key] = row

    output_by_head: List[Dict[str, List[int]]] = [
        {} for _ in ks
    ]
    combined: Dict[str, List[List[int]]] = {}
    keys_meta: Dict[str, Dict[str, Any]] = {}

    progress = tqdm(
        loader,
        total=len(loader),
        desc="Export official train cluster labels",
        dynamic_ncols=True,
    )

    for wav_30, valid_len, chunk_keys, wav_names in progress:
        wav_30 = wav_30.to(device)
        valid_len = valid_len.to(device)

        reps = model.forward_layerk(
            wav_30,
            sampling_rate=SAMPLE_RATE,
            valid_wav_lens=valid_len,
        )
        batch_size, token_count, _feature_dim = reps.shape
        valid_lengths = valid_len.detach().cpu().long().tolist()

        for index_in_batch in range(batch_size):
            chunk_key = canonical_label_key(
                str(chunk_keys[index_in_batch])
            )
            wav_name = str(wav_names[index_in_batch])
            if chunk_key not in entries_by_key:
                raise KeyError(
                    f"DataLoader Returned chunk_key not in manifest:{chunk_key}"
                )

            valid_token_count = min(
                compute_valid_token_length(
                    int(valid_lengths[index_in_batch])
                ),
                token_count,
            )
            valid_token_count = max(0, int(valid_token_count))

            labels_for_heads: List[List[int]] = []
            if valid_token_count <= 0:
                for _ in ks:
                    labels_for_heads.append(
                        [IGNORE_INDEX] * token_count
                        if pad_to_full
                        else []
                    )
            else:
                features = reps[
                    index_in_batch,
                    :valid_token_count,
                    :,
                ].detach().cpu().numpy()
                features = np.ascontiguousarray(
                    features.astype(np.float32)
                )

                for faiss_index in index_list:
                    _distance, nearest = faiss_index.search(
                        features,
                        1,
                    )
                    labels_valid = nearest[:, 0].astype(int).tolist()
                    if pad_to_full and valid_token_count < token_count:
                        labels_full = labels_valid + [IGNORE_INDEX] * (
                            token_count - valid_token_count
                        )
                    else:
                        labels_full = labels_valid
                    labels_for_heads.append(labels_full)

            for head_index in range(len(ks)):
                output_by_head[head_index][chunk_key] = (
                    labels_for_heads[head_index]
                )
            combined[chunk_key] = labels_for_heads

            row = entries_by_key[chunk_key]
            keys_meta[chunk_key] = {
                "wav_name": wav_name,
                "wav_path": str(row["wav_path"]),
                "chunk_key": chunk_key,
                "split": "train",
                "official_audio_id": int(row["official_audio_id"]),
                "audio_id": int(row["audio_id"]),
                "target_label": int(row["target_label"]),
                "chunk_id": int(row["chunk_id"]),
                "chunk_start": int(row["chunk_start"]),
                "chunk_end": int(row["chunk_end"]),
                "chunk_valid_len": int(row["chunk_valid_len"]),
            }

    if len(combined) != len(entries):
        raise ValueError(
            "The number of false labels exported does not match chunk manifest:"
            f"labels={len(combined)}, entries={len(entries)}"
        )

    paths: Dict[str, str] = {}
    for head_index, k in enumerate(ks):
        output_path = os.path.join(
            output_dir,
            f"{int(k)}-clus.json",
        )
        save_json(output_by_head[head_index], output_path)
        paths[f"K{int(k)}"] = output_path
        print(
            f"[Stage2-CLASS] Saved K={k} labels -> {output_path}"
        )

    all_path = os.path.join(output_dir, "all-clus.json")
    keys_meta_path = os.path.join(output_dir, "keys_meta.json")
    manifest_path = os.path.join(
        output_dir,
        "chunks_manifest.json",
    )

    save_json(combined, all_path)
    save_json(keys_meta, keys_meta_path)
    save_json(
        {
            "task": "ad_binary_classification",
            "split": "train",
            "official_train_used": True,
            "official_test_used": False,
            "target_label_used_by_kmeans": False,
            "class_mapping": {
                "0": CLASS_NAMES[0],
                "1": CLASS_NAMES[1],
            },
            "chunk_key_version": str(CHUNK_KEY_VERSION),
            "segment_sec": float(SEGMENT_SEC),
            "overlap_rate": float(OVERLAP_RATE),
            "num_chunks": int(len(entries)),
            "chunks": entries,
        },
        manifest_path,
    )

    paths.update(
        {
            "all_clus": all_path,
            "keys_meta": keys_meta_path,
            "chunks_manifest": manifest_path,
        }
    )
    print(f"[Stage2-CLASS] Saved ALL labels -> {all_path}")
    return paths

def _load_json_dict(path: str) -> Dict[str, Any]:
    """
    Read JSON dict.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"JSON Cannot initialise Evolution's mail component.{path}")
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"{path} Should be dig, current ={type(obj)}")
    return obj

def _load_manifest_chunk_keys(manifest_path: str) -> List[str]:
    """
    Read chunk_key in chunks_manifest.json.
    """
    obj = _load_json_dict(manifest_path)
    chunks = obj.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError(f"{manifest_path} The chunks aren't list.")
    keys = []
    for row in chunks:
        if not isinstance(row, dict):
            continue
        if row.get("chunk_key", None) in [None, ""]:
            continue
        keys.append(canonical_label_key(row["chunk_key"]))
    return keys

def _extract_head_sequence(value: Any, head_idx: int) -> List[int]:
    """
    Draws the label sequence for the designation of head from all-clus.json value.

    all-clus.json Format (current KS=[32,64]):
        key -> [[labels_K32], [labels_K64]]

    If value is a single format:
        key -> [labels_K]
    is readable at head_idx=0.
    """
    if isinstance(value, list) and len(value) > 0 and isinstance(value[0], list):
        if head_idx >= len(value):
            return []
        return value[head_idx]
    if isinstance(value, list) and head_idx == 0:
        return value
    return []

def compute_cluster_stats_from_all_clus(all_clus_path: str, ks: List[int]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, np.ndarray]]:
    """
    Statistics on the use of cluster per K in all-clus.json.

    Return:
        stats_by_k:  Statistical indicators per K
        prob_by_k:   Normalized distribution of each K for optional quality analysis
    """
    raw = _load_json_dict(all_clus_path)
    ks = [int(x) for x in ks]

    stats_by_k: Dict[str, Dict[str, Any]] = {}
    prob_by_k: Dict[str, np.ndarray] = {}

    for head_idx, k in enumerate(ks):
        counts = np.zeros((int(k),), dtype=np.float64)
        total_valid = 0
        total_ignore = 0

        for raw_key, value in raw.items():
            if str(raw_key).startswith("__"):
                continue
            seq = _extract_head_sequence(value, head_idx=head_idx)
            if len(seq) == 0:
                continue

            arr = np.asarray(seq, dtype=np.int64).reshape(-1)
            total_ignore += int((arr == IGNORE_INDEX).sum())

            valid = arr[arr != IGNORE_INDEX]
            valid = valid[(valid >= 0) & (valid < int(k))]
            if valid.size == 0:
                continue

            binc = np.bincount(valid, minlength=int(k)).astype(np.float64)
            counts += binc
            total_valid += int(valid.size)

        if total_valid > 0:
            prob = counts / max(1.0, counts.sum())
            active_clusters = int((counts > 0).sum())
            active_ratio = float(active_clusters / float(k))

            nz = prob[prob > 0]
            entropy = float(-(nz * np.log(nz)).sum()) if nz.size > 0 else 0.0
            effective_clusters = float(np.exp(entropy))

            sorted_counts = np.sort(counts)[::-1]
            top1_ratio = float(sorted_counts[0] / counts.sum()) if counts.sum() > 0 else 0.0
            top5_ratio = float(sorted_counts[: min(5, len(sorted_counts))].sum() / counts.sum()) if counts.sum() > 0 else 0.0
        else:
            prob = np.ones((int(k),), dtype=np.float64) / float(k)
            active_clusters = 0
            active_ratio = 0.0
            entropy = 0.0
            effective_clusters = 0.0
            top1_ratio = 0.0
            top5_ratio = 0.0

        stats_by_k[str(k)] = {
            "k": int(k),
            "num_valid_tokens": int(total_valid),
            "num_ignore_tokens": int(total_ignore),
            "active_clusters": int(active_clusters),
            "active_ratio": float(active_ratio),
            "entropy": float(entropy),
            "effective_clusters": float(effective_clusters),
            "top1_cluster_ratio": float(top1_ratio),
            "top5_cluster_ratio": float(top5_ratio),
        }
        prob_by_k[str(k)] = prob

    return stats_by_k, prob_by_k

def check_all_clus_heads(
    all_clus_path: str,
    expected_ks: List[int],
) -> None:
    """Check that the number of head in all-clus.json is consistent with the number of KS."""
    raw = _load_json_dict(all_clus_path)
    keys = [
        key for key in raw
        if not str(key).startswith("__")
    ]
    if len(keys) == 0:
        raise RuntimeError(
            f"all-clus.json Empty:{all_clus_path}"
        )

    expected_ks = [int(k) for k in expected_ks]
    for key in keys:
        value = raw[key]
        if not isinstance(value, list):
            raise ValueError(
                f"all-clus value ♪ Not list:key ♪{key}"
            )
        if len(value) != len(expected_ks):
            raise RuntimeError(
                f"key={key} head ={len(value)},"
                f"But KS={expected_ks}"
            )


def check_export_key_consistency(
    all_clus_path: str,
    manifest_path: str,
) -> Dict[str, Any]:
    """Strict inspection of chunks_manifest corresponds to chunk_key of all-clus."""
    all_clus = _load_json_dict(all_clus_path)
    label_keys = {
        canonical_label_key(key)
        for key in all_clus
        if not str(key).startswith("__")
    }
    manifest_keys = set(
        _load_manifest_chunk_keys(manifest_path)
    )

    missing = sorted(manifest_keys - label_keys)
    extra = sorted(label_keys - manifest_keys)
    if len(missing) > 0 or len(extra) > 0:
        raise RuntimeError(
            "Stage2 Export key inconsistent:"
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    return {
        "manifest_key_count": int(len(manifest_keys)),
        "label_key_count": int(len(label_keys)),
        "missing_count": 0,
        "extra_count": 0,
    }


def decide_stage2_k_quality(
    k: int,
    active_ratio: float,
    top1_ratio: float,
) -> Tuple[str, List[str]]:
    """
    official train Cluster mass is sifted early.

    Only cluster collapses are detected here, without using official test or comparing the train/val distribution.
    """
    k = int(k)
    warnings: List[str] = []
    decision = "go"

    if k <= 128:
        warn_active, reject_active = 0.50, 0.30
    elif k <= 256:
        warn_active, reject_active = 0.40, 0.25
    else:
        warn_active, reject_active = 0.25, 0.15

    if active_ratio < reject_active:
        warnings.append("active_ratio_too_low_reject")
        decision = "reject"
    elif active_ratio < warn_active:
        warnings.append("active_ratio_low")
        decision = "warn"

    if top1_ratio > 0.45:
        warnings.append("top1_cluster_ratio_too_high_reject")
        decision = "reject"
    elif top1_ratio > 0.30:
        warnings.append("top1_cluster_ratio_high")
        if decision != "reject":
            decision = "warn"

    return decision, warnings


def write_stage2_quality_report(
    run_config: Dict[str, Any],
    ks: List[int],
) -> None:
    """Output of official train-only cluster quality reports."""
    ks = [int(k) for k in ks]
    all_path = os.path.join(
        run_config["STAGE2_TRAIN_DIR"],
        "all-clus.json",
    )
    manifest_path = os.path.join(
        run_config["STAGE2_TRAIN_DIR"],
        "chunks_manifest.json",
    )

    key_check = check_export_key_consistency(
        all_clus_path=all_path,
        manifest_path=manifest_path,
    )
    train_stats, _probabilities = (
        compute_cluster_stats_from_all_clus(
            all_path,
            ks=ks,
        )
    )

    reports: Dict[str, Any] = {}
    rows: List[Dict[str, Any]] = []
    overall_warnings: List[str] = []

    for k in ks:
        stats = train_stats[str(k)]
        decision, warnings = decide_stage2_k_quality(
            k=k,
            active_ratio=float(stats["active_ratio"]),
            top1_ratio=float(stats["top1_cluster_ratio"]),
        )
        reports[str(k)] = {
            "k": int(k),
            "official_train": stats,
            "decision": decision,
            "warnings": warnings,
        }
        rows.append(
            {
                "stage": "stage2",
                "task": "ad_binary_classification",
                "dataset_option": run_config["DATASET_OPTION"],
                "layer_name": run_config["LAYER_NAME"],
                "rep_layer": int(run_config["REP_LAYER"]),
                "k": int(k),
                "active_ratio": float(stats["active_ratio"]),
                "effective_clusters": float(
                    stats["effective_clusters"]
                ),
                "top1_cluster_ratio": float(
                    stats["top1_cluster_ratio"]
                ),
                "top5_cluster_ratio": float(
                    stats["top5_cluster_ratio"]
                ),
                "entropy": float(stats["entropy"]),
                "decision": decision,
                "warnings": "|".join(warnings),
            }
        )
        overall_warnings.extend(
            [f"K{k}:{warning}" for warning in warnings]
        )

    decisions = [report["decision"] for report in reports.values()]
    if all(decision == "reject" for decision in decisions):
        overall_decision = "reject_stage2"
    elif any(decision in {"reject", "warn"} for decision in decisions):
        overall_decision = "warn_stage2"
    else:
        overall_decision = "go_stage3"

    report = {
        "stage": "stage2",
        "task": "ad_binary_classification",
        "protocol": "official_train_only",
        "official_test_used": False,
        "target_label_used_by_kmeans": False,
        "dataset_option": run_config["DATASET_OPTION"],
        "layer_name": run_config["LAYER_NAME"],
        "rep_layer": int(run_config["REP_LAYER"]),
        "ks": ks,
        "key_consistency": key_check,
        "k_reports": reports,
        "decision": overall_decision,
        "warnings": overall_warnings,
    }

    save_json(
        train_stats,
        os.path.join(
            run_config["STAGE2_TRAIN_DIR"],
            "cluster_stats.json",
        ),
    )
    save_json(
        report,
        os.path.join(
            run_config["STAGE2_SAVE_DIR"],
            QUALITY_REPORT_FILENAME,
        ),
    )
    pd.DataFrame(rows).to_csv(
        os.path.join(
            run_config["STAGE2_SAVE_DIR"],
            QUALITY_SUMMARY_FILENAME,
        ),
        index=False,
    )
    save_text(
        f"{overall_decision}\nwarnings={overall_warnings}\n",
        os.path.join(
            run_config["STAGE2_SAVE_DIR"],
            QUALITY_DECISION_FILENAME,
        ),
    )

    print(
        "[Stage2-CLASS][QUALITY] "
        f"decision={overall_decision} | warnings={overall_warnings}"
    )

def write_stage2_run_meta(
    run_config: Dict[str, Any],
    stage1_checkpoint_path: str,
    manifest_stats: Dict[str, Any],
    num_chunks: int,
    kmeans_stats: List[Dict[str, Any]],
    export_paths: Dict[str, str],
    key_check: Dict[str, Any],
) -> Dict[str, Any]:
    """Saves the complete configuration and semantic content of Stage2 classification."""
    meta = {
        "task": "ad_binary_classification",
        "stage": "stage2_clustering",
        "class_mapping": {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
        },
        "dataset_option": run_config["DATASET_OPTION"],
        "model_root_tag": MODEL_ROOT_TAG,
        "run_protocol": RUN_PROTOCOL_TAG,
        "official_train_used": True,
        "official_test_used": False,
        "target_label_preserved_in_manifest": True,
        "target_label_used_by_kmeans": False,
        "rep_layer": int(run_config["REP_LAYER"]),
        "unfreeze_encoder_layers": int(
            UNFREEZE_ENCODER_LAYERS
        ),
        "layer_name": run_config["LAYER_NAME"],
        "path_param_tags": list(PATH_PARAM_TAGS),
        "stage1_source_path_param_tags": list(
            STAGE1_SOURCE_PATH_PARAM_TAGS
        ),
        "stage1_save_dir": run_config["STAGE1_SAVE_DIR"],
        "stage1_checkpoint_path": stage1_checkpoint_path,
        "stage2_save_dir": run_config["STAGE2_SAVE_DIR"],
        "stage2_train_dir": run_config["STAGE2_TRAIN_DIR"],
        "whisper_model_name": WHISPER_MODEL_NAME,
        "ks": [int(k) for k in KS],
        "sample_mode": SAMPLE_MODE,
        "sample_ratio": (
            float(SAMPLE_RATIO)
            if SAMPLE_RATIO is not None
            else None
        ),
        "sample_per_utt": int(SAMPLE_PER_UTT),
        "segment_sec": float(SEGMENT_SEC),
        "overlap_rate": float(OVERLAP_RATE),
        "chunk_key_version": CHUNK_KEY_VERSION,
        "pad_to_full_length": bool(PAD_TO_FULL_LENGTH),
        "faiss_gpu_requested": bool(FAISS_GPU),
        "manifest_stats": manifest_stats,
        "train_num_chunks": int(num_chunks),
        "kmeans_stats": kmeans_stats,
        "export_paths": export_paths,
        "key_check": key_check,
    }

    save_json(
        meta,
        os.path.join(
            run_config["STAGE2_SAVE_DIR"],
            "stage2_run_meta.json",
        ),
    )
    save_json(
        meta,
        os.path.join(
            run_config["STAGE2_SAVE_DIR"],
            "clusters_meta.json",
        ),
    )
    save_json(
        meta,
        os.path.join(
            run_config["META_SAVE_DIR"],
            "stage2_meta.json",
        ),
    )
    save_text(
        stage1_checkpoint_path,
        os.path.join(
            run_config["META_SAVE_DIR"],
            "stage1_ckpt_used.txt",
        ),
    )
    return meta


def run_one_experiment(run_config: Dict[str, Any]) -> None:
    """Runs a data set, an expression layer, Stage2 of the number of unfrozen layers."""
    seed_everything_for_repro(SEED)

    ensure_dir(run_config["STAGE2_SAVE_DIR"])
    ensure_dir(run_config["STAGE2_TRAIN_DIR"])
    ensure_dir(run_config["META_SAVE_DIR"])

    print("\n" + "=" * 100)
    print(
        f"[Stage2-CLASS] DATASET={run_config['DATASET_OPTION']} | "
        f"REP_LAYER={run_config['REP_LAYER']} | "
        f"UNFREE={UNFREEZE_ENCODER_LAYERS}"
    )
    print("[Stage2-CLASS] official train used = True")
    print("[Stage2-CLASS] official test used = False")
    print("[Stage2-CLASS] target_label used by KMeans = False")
    print(
        f"[Stage2-CLASS] STAGE1_SAVE_DIR={run_config['STAGE1_SAVE_DIR']}"
    )
    print(
        f"[Stage2-CLASS] STAGE2_SAVE_DIR={run_config['STAGE2_SAVE_DIR']}"
    )
    print("=" * 100)

    # 1. Read Stage1 classification official train manifest.
    train_wav_entries, manifest_stats = (
        load_stage1_class_train_manifest(
            stage1_save_dir=run_config["STAGE1_SAVE_DIR"],
            dataset_option=run_config["DATASET_OPTION"],
        )
    )

    # 2. Expands the chunk manifest shared for Stage2/Stage3.
    train_entries = build_chunk_entries_from_official_entries(
        train_wav_entries,
        split_name="train",
    )

    # 3. Find Stage1 classification checkpoint.
    stage1_checkpoint_path = find_stage1_checkpoint(
        stage1_save_dir=run_config["STAGE1_SAVE_DIR"],
        override_ckpt=STAGE1_CKPT_OVERRIDE,
    )
    print(
        "[Stage2-CLASS] Stage1 checkpoint = "
        f"{stage1_checkpoint_path}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_faiss_gpu = bool(FAISS_GPU) and torch.cuda.is_available()
    if bool(FAISS_GPU) and not torch.cuda.is_available():
        print(
            "[Stage2-CLASS][WARN] FAISS_GPU=True But not CUDA,"
            "Refund to CPU FAISS."
        )

    model = WhisperEncoderWrapper(
        model_name=WHISPER_MODEL_NAME,
        rep_layer=int(run_config["REP_LAYER"]),
    ).to(device)
    load_stage1_ckpt_into_whisper_model(
        model.model,
        stage1_checkpoint_path,
    )
    model.eval()

    # ChunkAudioDataset Return only to waveform, valid_len, chunk_key and wav_name;
    # Do not return to target_label to prevent the label from entering KMeans.
    train_dataset = ChunkAudioDataset(train_entries)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=int(TRAIN_BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=train_dataset.repeat30s_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(NUM_WORKERS) > 0),
    )
    train_infer_loader = data.DataLoader(
        train_dataset,
        batch_size=int(INFER_BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=train_dataset.repeat30s_collate,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(int(NUM_WORKERS) > 0),
    )

    # 4. Collection of forms for official train token KMeans.
    print(
        f"[Stage2-CLASS][Layer{run_config['REP_LAYER']}] "
        "Commence collection of official train token forms."
    )
    kmeans_features = collect_train_samples(
        model=model,
        loader=train_loader,
        device=device,
        sample_ratio=SAMPLE_RATIO,
        sample_per_utt=SAMPLE_PER_UTT,
        sample_mode=SAMPLE_MODE,
    )
    print(
        "[Stage2-CLASS] KMeans features | "
        f"samples={kmeans_features.shape[0]} | "
        f"dim={kmeans_features.shape[1]}"
    )

    ks = [int(k) for k in KS]
    if len(ks) == 0:
        raise ValueError("KS Can't be empty.")
    if min(ks) <= 1:
        raise ValueError(f"KS Number of illegal clusters present:{ks}")
    if kmeans_features.shape[0] < max(ks):
        raise ValueError(
            "KMeans Fewer samples than K:"
            f"samples={kmeans_features.shape[0]}, KS={ks}"
        )

    # 5. Train each K and keep the cluster centres.
    faiss_indices: List[Any] = []
    kmeans_stats: List[Dict[str, Any]] = []

    for k in tqdm(
        ks,
        desc="Training FAISS KMeans",
        dynamic_ncols=True,
    ):
        centroids, faiss_index = fit_faiss_kmeans(
            X_train=kmeans_features,
            k=k,
            seed=int(SEED),
            niter=int(KMEANS_MAX_ITER),
            nredo=int(FAISS_NREDO),
            use_gpu=use_faiss_gpu,
            max_points_per_centroid=int(
                FAISS_MAX_POINTS_PER_CENTROID
            ),
            verbose=bool(FAISS_VERBOSE),
        )
        faiss_indices.append(faiss_index)

        centroid_path = os.path.join(
            run_config["STAGE2_TRAIN_DIR"],
            f"centroids_K{k}.npy",
        )
        np.save(
            centroid_path,
            np.asarray(centroids, dtype=np.float32),
        )
        kmeans_stats.append(
            {
                "k": k,
                "centroids_shape": [
                    int(value) for value in centroids.shape
                ],
                "feature_dim": int(centroids.shape[1]),
                "centroid_path": centroid_path,
            }
        )

    # 6. Export official train pseudo labels only.

    export_paths = infer_and_export_official_train_json(
        model=model,
        loader=train_infer_loader,
        index_list=faiss_indices,
        ks=ks,
        device=device,
        output_dir=run_config["STAGE2_TRAIN_DIR"],
        pad_to_full=bool(PAD_TO_FULL_LENGTH),
        entries=train_entries,
    )

    # 7. Strict inspection of the integrity of the export.
    check_all_clus_heads(
        export_paths["all_clus"],
        expected_ks=ks,
    )
    key_check = check_export_key_consistency(
        all_clus_path=export_paths["all_clus"],
        manifest_path=export_paths["chunks_manifest"],
    )

    # 8. Saves running syntax and configuration.
    if bool(SAVE_EXPORT_METADATA):
        write_stage2_run_meta(
            run_config=run_config,
            stage1_checkpoint_path=stage1_checkpoint_path,
            manifest_stats=manifest_stats,
            num_chunks=len(train_entries),
            kmeans_stats=kmeans_stats,
            export_paths=export_paths,
            key_check=key_check,
        )

    # 9. train-only quality reports are optional.
    if bool(ENABLE_STAGE_QUALITY_EVAL):
        write_stage2_quality_report(
            run_config=run_config,
            ks=ks,
        )

    print(
        "\n[Stage2-CLASS] Completed:"
        f"{run_config['STAGE2_SAVE_DIR']}"
    )


def main() -> None:
    """
    Batch order:
        DATASET_OPTIONS
        -> UNFREEZE_ENCODER_LAYERS_LIST
        -> STAGE2_REP_LAYERS_TO_RUN
    """
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS It's empty.")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST It's empty.")
    if len(STAGE2_REP_LAYERS_TO_RUN) == 0:
        raise ValueError("STAGE2_REP_LAYERS_TO_RUN It's empty.")

    print("[CONFIG]")
    print("  CLASS_PROJECT_ROOT =", CLASS_PROJECT_ROOT)
    print("  CPFT_ROOT =", CPFT_ROOT)
    print("  DATASET_OPTIONS =", DATASET_OPTIONS)
    print("  MODEL_ROOT_TAG =", MODEL_ROOT_TAG)
    print("  STAGE2_REP_LAYERS_TO_RUN =", STAGE2_REP_LAYERS_TO_RUN)
    print("  UNFREEZE_ENCODER_LAYERS_LIST =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  KS =", KS)
    print("  SAMPLE_MODE =", SAMPLE_MODE)
    print("  SAMPLE_PER_UTT =", SAMPLE_PER_UTT)
    print("  official_test_used = False")
    print("  target_label_used_by_kmeans = False")

    for dataset_option in [_canonical_dataset_option(x) for x in DATASET_OPTIONS]:
        for unfreeze_count in [
            int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST
        ]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = unfreeze_count

            for rep_layer in [
                int(x) for x in STAGE2_REP_LAYERS_TO_RUN
            ]:
                run_config = build_stage2_run_config(
                    dataset_option=DATASET_OPTION,
                    rep_layer=rep_layer,
                )
                run_one_experiment(run_config)


if __name__ == "__main__":
    main()
