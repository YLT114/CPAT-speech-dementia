# -*- coding: utf-8 -*-
"""
CPFT_stage2.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.

MMSE-aware CPFT stage2 (clusters and false labels).

Pitt subject-disjoint Amendments:
- Stage2 Do not randomly re-severe Pitt;
- It reads CPFT_stage1.py's valid train_manifest and conducts a peer/ visit/ wav-path consistency check;
- Only when it is confirmed that it is 184 sub-jects / 276 final-training observations and that there is no overlap with 68 final-test sub-checks are KMeans and false labels allowed.

This script is based on the original CPFT_stage2.py and the main changes are as follows:
1. No longer read the old label_x.json + "Train/Val/Test" cut, but read directly
   generate_five_folds_from_split.py Generate authoritative fold{i}.json.
2. KMeans Only the Whisper token expression is used to compile and strictly avoid leaking val information to a cluster centre.
3. Export of false labels after completion of training, for each of the following:
   mmse_cpft_runs/<DATASET>/whisper_small_mmse_cpft/layerX/foldY/stage2_clusters/train/
   mmse_cpft_runs/<DATASET>/whisper_small_mmse_cpft/layerX/foldY/stage2_clusters/val/
4. A "Show Parameter Settings" with the style of frozen.py / lru+transformer+lstm.py is provided at the top.
   DATASET / GPU / layers / folds may be modified directly, instead of referring to each relying command line.
5. Following the principle of "Variations first unbundled, default first connected":
   - Stage2 The real layer variable is STAGE2_REP_LAYERS_TO_RUN
   - Version 1 default STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN

Note:
- This script will automatically find best/last checkpoint from the Stage1 output directory and load weights into Whisper backbone.
- If you wish to specify a Stage1 ckpt manually, you may directly modify STAGE1_CKPT_OVERRIDE.
- To avoid affecting the baseline main line, this script will not be written in feature_tensor/, outputs/, split/.
"""

# ============================================================
# 0) Parameter Settings (must be placed before import torch to support visible binding cards)
# ============================================================
import os
"""
conda activate whisper_cpft_faiss
python CPFT_stage2.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.
"""

PROJECT_ROOT = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune"
MMSE_CPFT_ROOT = os.path.join(PROJECT_ROOT, "ft_CPFT")

# Pitt canonical subject-disjoint split Generated and audited by Stage1 under the current Pitt subject-disjoint protocol rules.
PROJECT_ROOT_MAIN = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT_MAIN, "feature_tensor")
PITT_SPLIT_AUDIT_ROOT = os.path.join(FEATURE_ROOT_DIR, "Pitt", "subject_disjoint_split")
PITT_VISIT_MANIFEST_PATH = os.path.join(PITT_SPLIT_AUDIT_ROOT, "pitt_valid_visits_subject_disjoint_manifest.csv")
PITT_SUBJECT_ASSIGNMENT_PATH = os.path.join(PITT_SPLIT_AUDIT_ROOT, "pitt_subject_assignment.csv")
PITT_ENABLE_SUBJECT_DISJOINT_SPLIT = True
PITT_SPLIT_POLICY = "original_test_subjects_holdout"
PITT_EXPECTED_FINAL_TRAIN_SUBJECTS = 184
PITT_EXPECTED_FINAL_TEST_SUBJECTS = 68
PITT_EXPECTED_FINAL_TRAIN_VISITS = 276
PITT_EXPECTED_FINAL_TEST_VISITS = 122

# Name of data set: optional "ADReSS-IS2020", "ADReSSo21", "Pitt", "PROCESS-2"
# Modify here when the batch runs.
# Pitt subject-disjoint The revised version will only run Pitt by default, avoiding the error of repeating other data sets together.
# Other data set codes remain intact; if a batch is to be resumed, it can be manually changed back to the original list.
DATASET_OPTIONS = ["ADReSS-IS2020"]


def _canonical_dataset_option(value: str) -> str:
    """Map the public ADReSS name to the existing internal dataset key."""
    value = str(value).strip()
    return "ADReSS-IS2020" if value == "ADReSS" else value

# Keeps a single-value variable for internal use in the old function; main() dynamically overlays it in the loop.
DATASET_OPTION = _canonical_dataset_option(DATASET_OPTIONS[0])

# ===== Public Access Control =====
# Must be consistent with Stage1 / Stage3 / CPFT_exteact.py.
LAYERS_TO_RUN = [12]

# ===== Align with Stage1 new interface: singleX_unfreeZ =
# UNFREEZE_ENCODER_LAYERS Indicates how many layers are unfrozen over the current expression layer_id.
# e.g. layer12 + UNFREEZE_ENCODER_LAYERS=1 -> single12_unfree1;
#      layer12 + UNFREEZE_ENCODER_LAYERS=4 -> single12_unfree4.
# The batch runs different amounts of unfrozen layers.
UNFREEZE_ENCODER_LAYERS_LIST = [1,2,3,4,5,6,7,8,9,10,11,12]

# Keep single-value variables for old logic such as build_layer_name / run_config / int(...).
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# ===== Visible binding card: same style as frozen.py / lru+transformer+lstm.py =====
# Set None to indicate that the card is not binding and is determined by the external environment; set to integer explicit designation is visible GPU.
GPU_ID = 3
if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# Current protocol:
# - Upper CPFT use official_train only;
# - official_test (b) Not participate in Stage1/2/3 training;
# - CPFT_exteact.py Output official_test feature only at the final export stage;
# - Stops creating the fold1/ fold2/... subdirectories.
RUN_PROTOCOL_TAG = "official_train_test"

# Keeps the placeholder field and avoids misreporting of the old log / function 's reference to FOLD_ID.
# No longer used for path and training divisions.
PROTOCOL_FOLD_ID = 1
FOLDS_TO_RUN = [PROTOCOL_FOLD_ID]

# Pre-training root label
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5"

# ============================================================
# Path Parameter Label
# ============================================================
# Stage2 / Stage3 Reads the path label used for Stage1 checkpoint.
# It is recommended that [] be maintained to indicate Stage1 path to singleX_unfreez/foldY/stage1_mmse/.
# If Stage1 itself carries ks suffix, the same label is added here.
# "ks32_64","clus1.0","mmse0.05","last","s1epo10","s3epo15","uttcap128"
STAGE1_SOURCE_PATH_PARAM_TAGS =  []

# Controls whether to add an abbreviation to the experimental parameter after singleX_unfreeZ.
#
# Example:
#   []                  -> single12_unfree1
#   ["ks64"]            -> single12_unfree1_ks64
#   ["ks64_128"]        -> single12_unfree1_ks64_128
#   ["ks64", "maux005"] -> single12_unfree1_ks64_maux005
#
# Note:
# 1. Stage2 / Stage3 / CPFT_exteact.py The PATH_PARAM_TAGS must be consistent.
# 2. Stage1 Not relying on KS, recommend PATH_PARAM_TAGS=[] and avoid duplication of training to the same Stage1.
# 3. If you want Stag1 to have ks suffix, set the PATH_PARAM_TAGS of Stage1 to the same effect.
PATH_PARAM_TAGS = []


# ===== stage-specific Layer variable (open first, connect first by default) =====
STAGE1_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN

# Stage2 Really used layer variable: which layer does the cluster mean?
STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN

# Stage3 / Stage4 Preset variable and keep cross- script parameters named
STAGE3_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]  # 1=Front level of the first trainingable level; first two levels of 2=; specific layer number is automatically parsed by Stage3
STAGE4_OUTPUT_LAYERS_TO_RUN = LAYERS_TO_RUN

# ===== Formal chunk-level settings: align to frozen.py =====
CHUNK_LEVEL = True
SEGMENT_SEC = 30
OVERLAP_RATE = 0.3
CHUNK_KEY_VERSION = "wav_path_chunk_start_end_v1"

# ===== Whisper / Stage1 checkpoint Configure =====
WHISPER_MODEL_NAME = "/home/user/whisper_lru_3/whisper_small"

# In the absence of None, use this Stage1 checkpoint on a priority basis;
# In the case of None, automatically search for best ckpt under the current stage1_mmse directory of fold and then return to last.ckpt if not found.
STAGE1_CKPT_OVERRIDE = None

# ===== Stage2 Cluster Configuration =====
KS = [32, 64]

# KMeans Sample mode:
#   "ratio"       : Original logic, each 30s chunk press SAMPLE_RATIO to draw token
#   "per_chunk"   : Up to SAMPLE_PER_UTT token per 30s chunk
#   "per_utt_cap" : All chanks of the same original wav together draw up to SAMPLE_PER_UTT token
#   "all"         : Use all valid tokens
#
# Current recommendations:
#   per_utt_cap + 128
# Role:
#   Avoid Pitt long audio / with more chunk audio leading KMeans while retaining more acoustic coverage than 64.
SAMPLE_MODE = "per_utt_cap" # k32_64_128For all, for other experiments per_utt_cap
SAMPLE_RATIO = None
SAMPLE_PER_UTT = 128

# Data length with DataLoader
# TARGET_SEC Reservation to old interface compatibility; formal cutting to SEGMENT_SEC.
TARGET_SEC = float(SEGMENT_SEC)
TRAIN_BATCH_SIZE = 4
INFER_BATCH_SIZE = 4
NUM_WORKERS = 4

# FAISS
FAISS_GPU = False
FAISS_NREDO = 1
FAISS_MAX_POINTS_PER_CENTROID = 2000
FAISS_VERBOSE = False
KMEANS_MAX_ITER = 200

# Other
SEED = 42
PAD_TO_FULL_LENGTH = True
SAVE_EXPORT_METADATA = True

# no-val-last:
# Stage2 Only fold_train is used to KMeans and only exported train/all-clus.json.
# Do not export fold_val pseudo labels; fold_val to downstream.
STAGE2_EXPORT_FOLD_VAL_LABELS = False

# ============================================================
# Stage2 Quality evaluation output
# ============================================================
# Role:
# 1. Automatically check whether train/val chunks_manifest matches key with all-clus.json;
# 2. Counts the active Cluster ratio, entropy, effactive clusters, Top1/top5 ratio for each K;
# 3. Calculate distribution train/val cluster JS divergence;
# 4. Output quality_report.json, quality_summary.csv, quality_decision.txt.
ENABLE_STAGE_QUALITY_EVAL = False
QUALITY_REPORT_FILENAME = "quality_report.json"
QUALITY_SUMMARY_FILENAME = "quality_summary.csv"
QUALITY_DECISION_FILENAME = "quality_decision.txt"


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

import numpy as np
import soundfile as sf
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.dataloader import default_collate

from transformers import WhisperModel, WhisperFeatureExtractor

import faiss
from tqdm import tqdm


# ============================================================
# 2) Constant
# ============================================================
SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
TOK_T_FULL = (MAX_MEL_FRAMES + 1) // 2
IGNORE_INDEX = -100
SEGMENT_LENGTH = int(float(SEGMENT_SEC) * SAMPLE_RATE)


# ============================================================
# 3) Generic Tool Functions
# ============================================================
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


def natural_key(x: str):
    """
    Natural ranking key.
    For example:
        layer2  < layer10
        fold2   < fold10
        S002    < S010
    """
    s = Path(str(x)).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


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
    Clears the path parameter labels to avoid the presence of characters such as spaces, slashes, etc. that are not suitable for entry into the directory name.
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
        PATH_PARAM_TAGS = []                  -> ""
        PATH_PARAM_TAGS = ["ks64"]            -> "ks64"
        PATH_PARAM_TAGS = ["ks64", "mp006"]   -> "ks64_mp006"
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

def check_all_clus_heads(all_clus_path: str, expected_ks: List[int]) -> None:
    """
    Checks if all-clus. json header numbers match KS.
    """
    with open(all_clus_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    keys = [k for k in raw.keys() if not str(k).startswith("__")]
    if len(keys) == 0:
        raise RuntimeError(f"[Stage2] empty all-clus.json: {all_clus_path}")

    expected_ks = [int(k) for k in expected_ks]
    n_heads = len(raw[keys[0]])

    if n_heads != len(expected_ks):
        raise RuntimeError(
            f"[Stage2] head mismatch: all-clus has {n_heads} heads, "
            f"but KS={expected_ks}. path={all_clus_path}"
        )

    # 10 samples before being sampled to avoid inconsistencies in the number of selected samples head
    for name in keys[:10]:
        if len(raw[name]) != len(expected_ks):
            raise RuntimeError(
                f"[Stage2] sample={name} has {len(raw[name])} heads, "
                f"but KS={expected_ks}. path={all_clus_path}"
            )


def write_stage2_run_meta(
    stage2_dir: str,
    layer_id: int,
    fold_id: int,
    stage1_ckpt_path: str,
    train_all_clus_path: str,
    val_all_clus_path: str,
) -> None:
    """
    Save Stage2 running configuration to confirm whether ks64 / ks64_128 is actually in effect.
    """
    meta = {
        "DATASET_OPTION": DATASET_OPTION,
        "MODEL_ROOT_TAG": MODEL_ROOT_TAG,
        "LAYER_ID": int(layer_id),
        "FOLD_ID": int(fold_id),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),
        "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(PATH_PARAM_TAGS),
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(STAGE1_SOURCE_PATH_PARAM_TAGS),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(STAGE1_SOURCE_PATH_PARAM_TAGS),

        "KS": [int(k) for k in KS],
        "SAMPLE_MODE": str(SAMPLE_MODE),
        "SAMPLE_RATIO": float(SAMPLE_RATIO) if SAMPLE_RATIO is not None else None,
        "SAMPLE_PER_UTT": int(SAMPLE_PER_UTT),
        "CHUNK_LEVEL": bool(CHUNK_LEVEL),


        "SEGMENT_SEC": float(SEGMENT_SEC),
        "OVERLAP_RATE": float(OVERLAP_RATE),
        "CHUNK_KEY_VERSION": str(CHUNK_KEY_VERSION),
        "stage1_ckpt_path": str(stage1_ckpt_path),
        "train_all_clus_path": str(train_all_clus_path),
        "val_all_clus_path": str(val_all_clus_path),
    }

    save_json(meta, os.path.join(stage2_dir, "stage2_run_meta.json"))


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
        raise FileNotFoundError(f"[Stage2-MMSE] Could not close temporary folder: %s {wav_path}")
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


def build_chunk_entries_from_fold_entries(entries: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """
    Expand wav-level entries in foldY.json to chunk-level entries.

    Important constraints:
    - (a) Only within fold train / fold val, as determined;
    - Do not do random_split;
    - Do not mix the same wav with the different split.
    """
    chunk_entries: List[Dict[str, Any]] = []

    for wav_row in tqdm(entries, desc=f"Build {split_name} chunks", dynamic_ncols=True):
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

        for chunk_id, start, end in boundaries:
            _seg, valid_len = load_chunk_with_padding_from_wav(
                y, start=start, end=end, chunk_id=chunk_id
            )
            chunk_key = make_chunk_key(
                wav_path=wav_path,
                chunk_id=chunk_id,
                start=start,
                end=end,
            )

            row = copy.deepcopy(wav_row)
            row.update({
                "split": str(split_name).lower(),
                "wav_path": wav_path,
                "wav_name": wav_name,
                "chunk_key": chunk_key,
                "chunk_id": int(chunk_id),
                "chunk_start": int(start),
                "chunk_end": int(end),
                "chunk_valid_len": int(valid_len),
                "parent_num_samples_16k": int(n_samples),
                "segment_sec": float(SEGMENT_SEC),
                "overlap_rate": float(OVERLAP_RATE),
                "chunk_key_version": str(CHUNK_KEY_VERSION),
            })
            chunk_entries.append(row)

    if len(chunk_entries) == 0:
        raise ValueError(f"{split_name} chunk_entries Please check the audio path and fold json.")

    return chunk_entries


def _strip_state_dict_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """
    Remove the public prefix from state_dict key.
    Common situation:
        "whisper.model.encoder.layers.0..." -> "encoder.layers.0..."
    """
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out


def load_stage1_ckpt_into_whisper_model(whisper_model: WhisperModel, ckpt_path: str) -> None:
    """
    Fill Whisper weight in Stage1 checkpoint with the current WhisperModel.

    Because LightingModule may have different key prefixes, here you try matching with multiple candidate prefixes.
    """
    if ckpt_path in [None, "None", ""]:
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    candidates = [
        "whisper.model.",  # Stage1-MMSE Most common: LightningModule.whisper.model.*
        "model.",          # Another common name
        "",                # Underground: No prefix
    ]

    loaded_any = False
    for pref in candidates:
        sd = _strip_state_dict_prefix(state, pref) if pref != "" else state
        missing, unexpected = whisper_model.load_state_dict(sd, strict=False)

        # A rough judgment of whether to hit an encoder.* as a prefix match.
        encoder_hit = any(k.startswith("encoder.") for k in sd.keys())
        if encoder_hit:
            print(f"[stage1_init] tried prefix='{pref}' -> missing={len(missing)} unexpected={len(unexpected)}")
            loaded_any = True
            break

    if not loaded_any:
        missing, unexpected = whisper_model.load_state_dict(state, strict=False)
        print(f"[stage1_init][warn] Prefix not clearly matched. Loaded, strict=False, missing={len(missing)} unexpected={len(unexpected)}")


def find_stage1_checkpoint(stage1_save_dir: str, override_ckpt: Optional[str] = None) -> str:
    """
    no-val-last:
    1) override_ckpt Non-empty direct use;
    2) Priority is given to the use of last.ckpt;
    3) If last.ckpt does not exist, return to the latest.ckpt
    """
    if override_ckpt not in [None, "", "None"]:
        ckpt_path = str(override_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"STAGE1_CKPT_OVERRIDE Cannot initialise Evolution's mail component.{ckpt_path}")
        return ckpt_path

    stage1_dir = Path(stage1_save_dir)
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage1 Directory does not exist:{stage1_save_dir}")

    last_ckpt = stage1_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt.resolve())

    all_ckpts = sorted(stage1_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(all_ckpts) > 0:
        return str(all_ckpts[0].resolve())

    raise FileNotFoundError(f"Yes. {stage1_save_dir} No Stage1 checkpoint found.")


# ============================================================
# 3.4) Pitt subject-disjoint consistency validation
# ============================================================
def _extract_pitt_subject_visit(raw_name: str) -> Tuple[Optional[str], Optional[int]]:
    """Parsing the same Pitt filename as the current Pitt subject-disjoint protocol"""
    if raw_name is None:
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    return str(int(m.group(1))), int(m.group(2))


def _normalize_pitt_subject_id(x) -> str:
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        m = re.search(r"\d+", s)
        if m is None:
            raise ValueError(f"Unable to regularize Pitt subject_id={x}")
        return str(int(m.group(0)))


def _load_pitt_canonical_split_audit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read audit documents canonical Pitt split written by Stage1.
    Stage2 Instead of redrawing, the Stage1 train manifest is mandatory to verify that the read is fully consistent with canonical final train.
    """
    if not os.path.isfile(PITT_VISIT_MANIFEST_PATH) or not os.path.isfile(PITT_SUBJECT_ASSIGNMENT_PATH):
        raise FileNotFoundError(
            "[Pitt Stage2] Missing subject-disjoint audit file. Please run CPFT_stage1.py."
            f"visit_manifest={PITT_VISIT_MANIFEST_PATH}\n"
            f"subject_assignment={PITT_SUBJECT_ASSIGNMENT_PATH}"
        )

    visit_df = pd.read_csv(PITT_VISIT_MANIFEST_PATH)
    subject_df = pd.read_csv(PITT_SUBJECT_ASSIGNMENT_PATH)

    required_visit = {"subject_id", "visit_index", "wav_path", "final_split"}
    required_subject = {"subject_id", "final_split"}
    if not required_visit.issubset(set(visit_df.columns)):
        raise ValueError(f"[Pitt Stage2] visit manifest Shortcut:{sorted(required_visit - set(visit_df.columns))}")
    if not required_subject.issubset(set(subject_df.columns)):
        raise ValueError(f"[Pitt Stage2] subject assignment Shortcut:{sorted(required_subject - set(subject_df.columns))}")

    visit_df = visit_df.copy()
    subject_df = subject_df.copy()
    visit_df["subject_id"] = visit_df["subject_id"].map(_normalize_pitt_subject_id)
    subject_df["subject_id"] = subject_df["subject_id"].map(_normalize_pitt_subject_id)
    visit_df["visit_index"] = pd.to_numeric(visit_df["visit_index"], errors="raise").astype(int)
    visit_df["final_split"] = visit_df["final_split"].astype(str).str.strip().str.lower()
    subject_df["final_split"] = subject_df["final_split"].astype(str).str.strip().str.lower()

    train_subjects = set(subject_df.loc[subject_df["final_split"] == "train", "subject_id"].tolist())
    test_subjects = set(subject_df.loc[subject_df["final_split"] == "test", "subject_id"].tolist())
    overlap = train_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"[Pitt Stage2] canonical audit Existence subproject overlap: {sorted(overlap)}")

    train_visits = visit_df[visit_df["final_split"] == "train"]
    test_visits = visit_df[visit_df["final_split"] == "test"]
    checks = [
        ("train_subjects", len(train_subjects), PITT_EXPECTED_FINAL_TRAIN_SUBJECTS),
        ("test_subjects", len(test_subjects), PITT_EXPECTED_FINAL_TEST_SUBJECTS),
        ("train_visits", len(train_visits), PITT_EXPECTED_FINAL_TRAIN_VISITS),
        ("test_visits", len(test_visits), PITT_EXPECTED_FINAL_TEST_VISITS),
    ]
    for name, actual, expected in checks:
        if int(actual) != int(expected):
            raise RuntimeError(f"[Pitt Stage2] canonical audit {name}={actual}, expected={expected}")

    return visit_df, subject_df


def validate_pitt_stage1_train_manifest(entries: List[Dict[str, Any]]) -> None:
    """
    A mandatory check of the entrance to Stage2 will happen to be canonical final train: 184 subjects / 276 visits.
    Thus, even if the old Stage1 product is misreaded, it will stop directly before KMeans and not mix final-test subject into a cluster.
    """
    if str(DATASET_OPTION) != "Pitt" or not bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return

    visit_df, subject_df = _load_pitt_canonical_split_audit()
    canonical_train = visit_df[visit_df["final_split"] == "train"].copy()
    canonical_train_visits = {
        (_normalize_pitt_subject_id(r.subject_id), int(r.visit_index))
        for r in canonical_train.itertuples()
    }
    canonical_train_paths = {canonical_wav_path(p) for p in canonical_train["wav_path"].astype(str).tolist()}
    canonical_train_subjects = {
        _normalize_pitt_subject_id(x)
        for x in subject_df.loc[subject_df["final_split"] == "train", "subject_id"].tolist()
    }
    canonical_test_subjects = {
        _normalize_pitt_subject_id(x)
        for x in subject_df.loc[subject_df["final_split"] == "test", "subject_id"].tolist()
    }

    manifest_visits = set()
    manifest_paths = set()
    manifest_subjects = set()
    duplicate_visits = []
    for row in entries:
        fp = canonical_wav_path(row.get("wav_path", ""))
        sid = row.get("subject_id", None)
        vid = row.get("visit_index", None)
        if sid is None or vid is None:
            sid, vid = _extract_pitt_subject_visit(Path(fp).stem)
        if sid is None or vid is None:
            raise RuntimeError(f"[Pitt Stage2] Stage1 manifest Unable to parse subject/visit: {fp}")
        sid = _normalize_pitt_subject_id(sid)
        key = (sid, int(vid))
        if key in manifest_visits:
            duplicate_visits.append(key)
        manifest_visits.add(key)
        manifest_paths.add(fp)
        manifest_subjects.add(sid)

    if duplicate_visits:
        raise RuntimeError(f"[Pitt Stage2] Stage1 manifest Repeated subject-visit: {duplicate_visits[:10]}")
    if manifest_subjects & canonical_test_subjects:
        raise RuntimeError(
            f"[Pitt Stage2] Stage1 manifest Including final-test subjects: {sorted(manifest_subjects & canonical_test_subjects, key=int)[:20]}"
        )
    if manifest_subjects != canonical_train_subjects:
        missing = sorted(canonical_train_subjects - manifest_subjects, key=int)
        extra = sorted(manifest_subjects - canonical_train_subjects, key=int)
        raise RuntimeError(f"[Pitt Stage2] Stage1 train subjects Inconsistent: missing={missing[:20]}, extra={extra[:20]}")
    if manifest_visits != canonical_train_visits:
        missing = sorted(canonical_train_visits - manifest_visits, key=lambda x: (int(x[0]), x[1]))
        extra = sorted(manifest_visits - canonical_train_visits, key=lambda x: (int(x[0]), x[1]))
        raise RuntimeError(f"[Pitt Stage2] Stage1 train visits Inconsistent: missing={missing[:20]}, extra={extra[:20]}")
    if manifest_paths != canonical_train_paths:
        missing = sorted(canonical_train_paths - manifest_paths)
        extra = sorted(manifest_paths - canonical_train_paths)
        raise RuntimeError(f"[Pitt Stage2] Stage1 train wav paths Inconsistent: missing={missing[:10]}, extra={extra[:10]}")

    print(
        "[Pitt Stage2] subject-disjoint Stage1 manifest check PASSED | "
        f"subjects={len(manifest_subjects)}, visits={len(manifest_visits)}, final-test-overlap=0"
    )


# ============================================================
# 3.5) official_train_test: Reading Stage1 Filtered manifest
# ============================================================
def _is_process2_ctd_manifest_entry(row: Dict[str, Any], dataset_key: str = None) -> bool:
    """
    PROCESS-2 Specialized in manifest safety filter.
    Only entry stem at the end of __CTD for wav_path / wav_name should be retained.

    Other data sets are not filtered.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    wav_path = row.get("wav_path", row.get("wav_name", ""))
    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None


def load_stage1_valid_train_manifest(stage1_save_dir: str) -> List[Dict[str, Any]]:
    """
    Read Stage1 already MMSE filtered train_manifest.json.

    PROCESS-2:
        Make CTD-only safety filter to avoid misreading the old SFT/PFT/CTD mix manifest.

    This is the entrance to Stage2:
    - Avoiding Stage2 from reconstructing inconsistent samples from official_train catalogues;
    - Ensures the use of the same effective MMSE audio by Stage1 / Stage2 / Stage3.
    """
    manifest_path = os.path.join(str(stage1_save_dir), "train_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Stage1 train_manifest.json Cannot initialise Evolution's mail component.{manifest_path}\n"
            "Please run CPFT_stage1.py."
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and isinstance(obj.get("entries", None), list):
        entries = obj["entries"]
    elif isinstance(obj, list):
        entries = obj
    else:
        raise ValueError(f"Could not close temporary folder: %s{manifest_path}")

    out: List[Dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue

        row = copy.deepcopy(row)
        if row.get("wav_path", None) in [None, ""]:
            continue

        row["wav_path"] = canonical_wav_path(row["wav_path"])
        row["wav_name"] = str(row.get("wav_name", Path(row["wav_path"]).name))
        row["split"] = "train"

        if not _is_process2_ctd_manifest_entry(row, dataset_key=DATASET_OPTION):
            continue

        out.append(row)

    if len(out) == 0:
        raise ValueError(f"Stage1 train_manifest Parse empty:{manifest_path}")

    if str(DATASET_OPTION).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] Stage2 loaded Stage1 train_manifest: "
            f"raw_entries={len(entries)}, kept_ctd_entries={len(out)}, "
            f"dropped_non_ctd={len(entries) - len(out)}"
        )

    out = sorted(out, key=lambda x: natural_key(x.get("wav_path", "")))
    if str(DATASET_OPTION) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        validate_pitt_stage1_train_manifest(out)
    return out


def filter_entries_with_target_mmse_for_stage2(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage2 Keep MMSE filters.

    This is not a re-analyse of MMSE metadata, but a check of target_mmse in Stage1 valid manifest.
    Under normal circumstances missing should read 0; if Stage1 manifest is manually destroyed, it will be intercepted again.
    """
    valid: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for row in entries:
        row = copy.deepcopy(row)
        val = row.get("target_mmse", None)
        try:
            if val in [None, ""]:
                raise ValueError("empty target_mmse")
            row["target_mmse"] = float(val)
            valid.append(row)
        except Exception:
            row["missing_reason"] = "missing_or_invalid_target_mmse_in_stage1_manifest"
            missing.append(row)

    return valid, missing


# ============================================================
# 4) fold json Parsing and Path Construction
# ============================================================
def load_fold_json(path: str) -> Dict[str, Any]:
    """
    Read authoritative fold json.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"fold json Cannot initialise Evolution's mail component.{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def split_entries_from_fold_json(fold_meta: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    parsing the trade/ val entry from the old json output of the generate_five_folds_from_split. py.

    Current priority:
        train_entries(Each of these entries has split fields)
    """
    rows = fold_meta.get("train_entries", None)
    if not isinstance(rows, list):
        raise ValueError("fold json Can not get folder: %s: %s")

    train_entries: List[Dict[str, Any]] = []
    val_entries: List[Dict[str, Any]] = []

    for row in rows:
        if not isinstance(row, dict):
            continue
        split_name = str(row.get("split", "")).strip().lower()
        wav_path = row.get("wav_path", None)
        if wav_path in [None, ""]:
            continue

        new_row = dict(row)
        new_row["wav_path"] = canonical_wav_path(wav_path)
        new_row["wav_name"] = str(new_row.get("wav_name", Path(new_row["wav_path"]).name))

        if split_name == "train":
            train_entries.append(new_row)
        elif split_name == "val":
            val_entries.append(new_row)

    if len(train_entries) == 0:
        raise ValueError("fold json Parsing train_entries is empty.")
    if len(val_entries) == 0:
        raise ValueError("fold json Parsing val_entries is empty.")

    train_entries = sorted(train_entries, key=lambda x: natural_key(x["wav_path"]))
    val_entries = sorted(val_entries, key=lambda x: natural_key(x["wav_path"]))
    return train_entries, val_entries

def check_all_clus_heads(all_clus_path: str, expected_ks: List[int]) -> None:
    """
    Checks if all-clus. json header numbers match KS.
    """
    if not os.path.isfile(all_clus_path):
        raise FileNotFoundError(f"[Stage2] all-clus.json not found: {all_clus_path}")

    with open(all_clus_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    keys = [k for k in raw.keys() if not str(k).startswith("__")]
    if len(keys) == 0:
        raise RuntimeError(f"[Stage2] empty all-clus.json: {all_clus_path}")

    expected_ks = [int(k) for k in expected_ks]
    n_heads = len(raw[keys[0]])

    if n_heads != len(expected_ks):
        raise RuntimeError(
            f"[Stage2] head mismatch: all-clus has {n_heads} heads, "
            f"but KS={expected_ks}. path={all_clus_path}"
        )

    for name in keys[:10]:
        if len(raw[name]) != len(expected_ks):
            raise RuntimeError(
                f"[Stage2] sample={name} has {len(raw[name])} heads, "
                f"but KS={expected_ks}. path={all_clus_path}"
            )



def write_stage2_run_meta(cfg: Dict[str, Any], stage1_ckpt_path: str) -> None:
    """
    Saves Stage2 running configuration.
    no-val-last:train/all-clus.json is required to exist; val/all-clus.json is not required.
    """
    stage2_save_dir = cfg["STAGE2_SAVE_DIR"]

    train_all_clus_path = os.path.join(cfg["STAGE2_TRAIN_DIR"], "all-clus.json")

    meta = {
        "DATASET_OPTION": cfg.get("DATASET_OPTION", DATASET_OPTION),
        "MODEL_ROOT_TAG": MODEL_ROOT_TAG,
        "REP_LAYER": int(cfg.get("REP_LAYER", -1)),
        "FOLD_ID": int(cfg.get("FOLD_ID", -1)),
        "LAYER_NAME": cfg.get("LAYER_NAME", ""),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),

        "PATH_PARAM_TAGS": list(globals().get("PATH_PARAM_TAGS", [])),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(globals().get("PATH_PARAM_TAGS", [])),
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(globals().get("STAGE1_SOURCE_PATH_PARAM_TAGS", [])),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(
            globals().get("STAGE1_SOURCE_PATH_PARAM_TAGS", [])
        ),

        "KS": [int(k) for k in KS],
        "SAMPLE_MODE": str(SAMPLE_MODE),
        "SAMPLE_RATIO": float(SAMPLE_RATIO) if SAMPLE_RATIO is not None else None,
        "SAMPLE_PER_UTT": int(SAMPLE_PER_UTT),
        "CHUNK_LEVEL": bool(CHUNK_LEVEL),


        "SEGMENT_SEC": float(SEGMENT_SEC),
        "OVERLAP_RATE": float(OVERLAP_RATE),
        "CHUNK_KEY_VERSION": str(CHUNK_KEY_VERSION),

        "STAGE2_EXPORT_FOLD_VAL_LABELS": False,
        "upstream_uses_fold_val": False,
        "stage1_ckpt_path": str(stage1_ckpt_path),
        "train_all_clus_path": str(train_all_clus_path),
        "val_all_clus_path": None,
    }

    save_json(meta, os.path.join(stage2_save_dir, "stage2_run_meta.json"))

def build_stage2_run_config(dataset_option: str, rep_layer: int, fold_id: int = PROTOCOL_FOLD_ID) -> Dict[str, Any]:
    """
    Construct Stage2 to run the configuration.

    Current protocol:
    - Do not read fold json;
    - Do not create subdirectories for fold1;
    - Do not create subdirectories for official_train_test;
    - Stage2 Read official_train valid MMSE samples from valid train_manifest.json at Stage1.

    Path:
        mmse_cpft_runs/<DATASET>/<MODEL_ROOT_TAG>/<singleX_unfreeZ>/stage2_clusters/
    """
    rep_layer = int(rep_layer)
    fold_id = int(PROTOCOL_FOLD_ID)

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
        MMSE_CPFT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        layer_name,
    )

    stage1_run_root = os.path.join(
        MMSE_CPFT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        stage1_layer_name,
    )

    stage1_dir = os.path.join(stage1_run_root, "stage1_mmse")

    stage2_save_dir = os.path.join(run_root, "stage2_clusters")
    stage2_train_dir = os.path.join(stage2_save_dir, "train")
    stage2_val_dir = os.path.join(stage2_save_dir, "val")
    meta_save_dir = os.path.join(run_root, "meta")

    return {
        "DATASET_OPTION": str(dataset_option),
        "REP_LAYER": int(rep_layer),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),

        # Compatibility with old fields; no longer true fold.
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
        "STAGE1_DIR": stage1_dir,

        "STAGE2_SAVE_DIR": stage2_save_dir,
        "STAGE2_TRAIN_DIR": stage2_train_dir,
        "STAGE2_VAL_DIR": stage2_val_dir,
        "META_SAVE_DIR": meta_save_dir,
    }
# ============================================================
# 5) Dataset and Whisper induction
# ============================================================
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


# ============================================================
# 6) KMeans Training and Export of Counterfeited Tags
# ============================================================
@torch.no_grad()
def collect_train_samples(
    model: WhisperEncoderWrapper,
    loader,
    device: str,
    sample_ratio: Optional[float] = None,
    sample_per_utt: Optional[int] = None,
    sample_mode: str = "ratio",
) -> np.ndarray:
    """
    Draw from fold-train the token form to form a training sample matrix for KMeans X_train: [N, C].

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

    pbar = tqdm(loader, total=len(loader), desc="Collection of training samples (fold-train)", dynamic_ncols=True)

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
            f"[Stage2-MMSE][KMeansSample] mode=per_utt_cap, "
            f"num_parent_wavs={len(reservoir)}, "
            f"cap_per_wav={sample_per_utt}, "
            f"total_tokens={X.shape[0]}"
        )

    else:
        if len(feats_all) == 0:
            raise RuntimeError("No training features have been collected to enable clustering.")

        X = np.concatenate(feats_all, axis=0).astype(np.float32)

        print(
            f"[Stage2-MMSE][KMeansSample] mode={sample_mode}, "
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
def infer_and_export_split_json(
    model: WhisperEncoderWrapper,
    loader,
    index_list: List[Any],
    ks: List[int],
    device: str,
    outputdir: str,
    split_name: str,
    pad_to_full: bool,
    entries: List[Dict[str, Any]],
) -> None:
    """
    Generates a false label for split (train or val) and export json.

    Export content:
      - {k}-clus.json: key(wav_path) -> labels_full
      - all-clus.json: key(wav_path) -> [labels_full_k1, labels_full_k2, ...]
      - keys_meta.json: key -> {wav_name, wav_path, split}
    """
    ensure_dir(outputdir)

    outputdata_list: List[Dict[str, List[int]]] = [dict() for _ in ks]
    combined_dict: Dict[str, List[List[int]]] = {}
    keys_meta: Dict[str, Dict[str, str]] = {}

    pbar = tqdm(loader, total=len(loader), desc=f"Extrapolation of the false label ({split_name})", dynamic_ncols=True)
    for wav_30, valid_len, wav_keys, wav_names in pbar:
        wav_30 = wav_30.to(device)
        valid_len = valid_len.to(device)

        reps = model.forward_layerk(wav_30, sampling_rate=SAMPLE_RATE, valid_wav_lens=valid_len)  # [B,T,C]
        B, T, C = reps.shape
        vwl_cpu = valid_len.detach().cpu().long().tolist()

        for i in range(B):
            wav_key = str(wav_keys[i])
            wav_name = str(wav_names[i])
            vt = min(compute_valid_token_length(int(vwl_cpu[i])), T)
            vt = max(0, int(vt))

            if vt <= 0:
                if pad_to_full:
                    labels_full_list = [[IGNORE_INDEX] * T for _ in ks]
                else:
                    labels_full_list = [[] for _ in ks]
            else:
                x = reps[i, :vt, :].detach().cpu().numpy()
                x = np.ascontiguousarray(x.astype(np.float32))

                labels_full_list = []
                for index in index_list:
                    _, I = index.search(x, 1)
                    labels_valid = I[:, 0].astype(int).tolist()

                    if pad_to_full and vt < T:
                        labels_full = labels_valid + [IGNORE_INDEX] * (T - vt)
                    else:
                        labels_full = labels_valid

                    labels_full_list.append(labels_full)

            for j in range(len(ks)):
                outputdata_list[j][wav_key] = labels_full_list[j]
            combined_dict[wav_key] = labels_full_list
            # entries is chunk-level manifest, which provides more complete meta-information by chunk_key.
            keys_meta[wav_key] = {
                "wav_name": wav_name,
                "chunk_key": wav_key,
                "split": str(split_name).lower(),
            }

    for j, k in enumerate(ks):
        out_path = os.path.join(outputdir, f"{k}-clus.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outputdata_list[j], f, ensure_ascii=False, indent=2)
        print(f"[Stage2-MMSE] Saved K={k} labels -> {out_path}")

    all_path = os.path.join(outputdir, "all-clus.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(combined_dict, f, ensure_ascii=False, indent=2)
    print(f"[Stage2-MMSE] Saved ALL labels -> {all_path}")

    # keys_meta Only light indexes; chunks_manifest saves the complete chunk entry for accurate reuse Stage3.
    save_json(keys_meta, os.path.join(outputdir, "keys_meta.json"))
    save_json(
        {
            "split": str(split_name).lower(),
            "chunk_key_version": str(CHUNK_KEY_VERSION),
            "segment_sec": float(SEGMENT_SEC),
            "overlap_rate": float(OVERLAP_RATE),
            "num_chunks": int(len(entries)),
            "chunks": entries,
        },
        os.path.join(outputdir, "chunks_manifest.json"),
    )


# ============================================================
# 6.5) Stage2 Quality evaluation: cluster distribution, degree of collapse, train/val consistency
# ============================================================
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

    all-clus.json Format:
        key -> [[labels_K128], [labels_K256], [labels_K512]]

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
        prob_by_k:   K for train/val JS divergence
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


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """
    Jensen-Shannon divergence,To measure differences in distribution train/val cluster.
    The smaller is the more uniform the Train/val cluster distribution.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p + float(eps)
    q = q + float(eps)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    kl_pm = float((p * np.log(p / m)).sum())
    kl_qm = float((q * np.log(q / m)).sum())
    return 0.5 * (kl_pm + kl_qm)


def decide_stage2_k_quality(k: int, active_ratio: float, top1_ratio: float, jsd: float) -> Tuple[str, List[str]]:
    """
    Recommendations for individual K go / warn / reject.

    N.B. These thresholds are early screening experience rules and are not final performance determinations.
    """
    k = int(k)
    warnings = []
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

    if jsd > 0.40:
        warnings.append("train_val_js_divergence_too_high_reject")
        decision = "reject"
    elif jsd > 0.25:
        warnings.append("train_val_js_divergence_high")
        if decision != "reject":
            decision = "warn"

    return decision, warnings


def write_stage2_quality_report(run_config: Dict[str, Any], ks: List[int]) -> None:
    """
    Stage2 Automatically output the cluster quality report after completion.

    Output position:
        stage2_clusters/quality_report.json
        stage2_clusters/quality_summary.csv
        stage2_clusters/quality_decision.txt
        stage2_clusters/train/cluster_stats.json
        stage2_clusters/val/cluster_stats.json
    """
    ks = [int(x) for x in ks]

    train_all_path = os.path.join(run_config["STAGE2_TRAIN_DIR"], "all-clus.json")
    val_all_path = os.path.join(run_config["STAGE2_VAL_DIR"], "all-clus.json")
    train_manifest_path = os.path.join(run_config["STAGE2_TRAIN_DIR"], "chunks_manifest.json")
    val_manifest_path = os.path.join(run_config["STAGE2_VAL_DIR"], "chunks_manifest.json")


    train_raw = _load_json_dict(train_all_path)
    val_raw = _load_json_dict(val_all_path)

    train_label_keys = set(canonical_label_key(k) for k in train_raw.keys() if not str(k).startswith("__"))
    val_label_keys = set(canonical_label_key(k) for k in val_raw.keys() if not str(k).startswith("__"))

    train_manifest_keys = set(_load_manifest_chunk_keys(train_manifest_path))
    val_manifest_keys = set(_load_manifest_chunk_keys(val_manifest_path))

    train_missing_label = sorted(list(train_manifest_keys - train_label_keys))[:20]
    val_missing_label = sorted(list(val_manifest_keys - val_label_keys))[:20]
    train_extra_label = sorted(list(train_label_keys - train_manifest_keys))[:20]
    val_extra_label = sorted(list(val_label_keys - val_manifest_keys))[:20]

    train_stats, train_prob = compute_cluster_stats_from_all_clus(train_all_path, ks=ks)
    val_stats, val_prob = compute_cluster_stats_from_all_clus(val_all_path, ks=ks)

    k_reports = {}
    summary_rows = []
    overall_warnings = []

    for k in ks:
        k_str = str(int(k))
        train_s = train_stats[k_str]
        val_s = val_stats[k_str]
        jsd = js_divergence(train_prob[k_str], val_prob[k_str])

        # active_ratio/top1 train, and also check if the distribution of val is extreme.
        decision, warnings = decide_stage2_k_quality(
            k=int(k),
            active_ratio=float(train_s["active_ratio"]),
            top1_ratio=float(train_s["top1_cluster_ratio"]),
            jsd=float(jsd),
        )

        if float(val_s["active_ratio"]) < 0.10:
            warnings.append("val_active_ratio_extremely_low")
            if decision != "reject":
                decision = "warn"

        k_reports[k_str] = {
            "k": int(k),
            "train": train_s,
            "val": val_s,
            "train_val_js_divergence": float(jsd),
            "decision": decision,
            "warnings": warnings,
        }

        summary_rows.append({
            "stage": "stage2",
            "dataset_option": run_config["DATASET_OPTION"],
            "fold_id": int(run_config["FOLD_ID"]),
            "layer_name": run_config["LAYER_NAME"],
            "rep_layer": int(run_config["REP_LAYER"]),
            "k": int(k),
            "train_active_ratio": float(train_s["active_ratio"]),
            "val_active_ratio": float(val_s["active_ratio"]),
            "train_effective_clusters": float(train_s["effective_clusters"]),
            "val_effective_clusters": float(val_s["effective_clusters"]),
            "train_top1_cluster_ratio": float(train_s["top1_cluster_ratio"]),
            "val_top1_cluster_ratio": float(val_s["top1_cluster_ratio"]),
            "train_val_js_divergence": float(jsd),
            "decision": decision,
            "warnings": "|".join(warnings),
        })

        for w in warnings:
            overall_warnings.append(f"K{k}:{w}")

    if len(train_missing_label) > 0 or len(val_missing_label) > 0:
        overall_warnings.append("manifest_key_missing_in_all_clus")
    if len(train_extra_label) > 0 or len(val_extra_label) > 0:
        overall_warnings.append("all_clus_has_extra_keys_not_in_manifest")

    decisions = [x["decision"] for x in k_reports.values()]
    if all(x == "reject" for x in decisions):
        overall_decision = "reject_stage2"
    elif any(x in ["reject", "warn"] for x in decisions) or len(overall_warnings) > 0:
        overall_decision = "warn_stage2"
    else:
        overall_decision = "go_stage3"

    report = {
        "stage": "stage2",
        "dataset_option": run_config["DATASET_OPTION"],
        "fold_id": int(run_config["FOLD_ID"]),
        "layer_name": run_config["LAYER_NAME"],
        "rep_layer": int(run_config["REP_LAYER"]),
        "unfreeze_encoder_layers": int(run_config["UNFREEZE_ENCODER_LAYERS"]),
        "chunk_level": bool(CHUNK_LEVEL),
        "segment_sec": float(SEGMENT_SEC),
        "overlap_rate": float(OVERLAP_RATE),
        "chunk_key_version": str(CHUNK_KEY_VERSION),
        "ks": [int(x) for x in ks],
        "num_train_manifest_chunks": int(len(train_manifest_keys)),
        "num_val_manifest_chunks": int(len(val_manifest_keys)),
        "num_train_label_keys": int(len(train_label_keys)),
        "num_val_label_keys": int(len(val_label_keys)),
        "train_missing_label_examples": train_missing_label,
        "val_missing_label_examples": val_missing_label,
        "train_extra_label_examples": train_extra_label,
        "val_extra_label_examples": val_extra_label,
        "k_reports": k_reports,
        "decision": overall_decision,
        "warnings": overall_warnings,
    }

    save_json(train_stats, os.path.join(run_config["STAGE2_TRAIN_DIR"], "cluster_stats.json"))
    save_json(val_stats, os.path.join(run_config["STAGE2_VAL_DIR"], "cluster_stats.json"))

    report_path = os.path.join(run_config["STAGE2_SAVE_DIR"], QUALITY_REPORT_FILENAME)
    summary_path = os.path.join(run_config["STAGE2_SAVE_DIR"], QUALITY_SUMMARY_FILENAME)
    decision_path = os.path.join(run_config["STAGE2_SAVE_DIR"], QUALITY_DECISION_FILENAME)

    save_json(report, report_path)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    save_text(f"{overall_decision}\nwarnings={overall_warnings}\n", decision_path)

    print(f"[Stage2-Quality] report saved -> {report_path}")
    print(f"[Stage2-Quality] decision = {overall_decision}, warnings = {overall_warnings}")

# ============================================================
# 7) Run individual (layer, fold)
# ============================================================
def run_one_experiment(run_config: Dict[str, Any]) -> None:
    """
    Execute individual Stage2 experiments (dataset, rep_layer, fold).
    """
    seed_everything_for_repro(SEED)

    ensure_dir(run_config["STAGE2_SAVE_DIR"])
    ensure_dir(run_config["STAGE2_TRAIN_DIR"])
    ensure_dir(run_config["STAGE2_VAL_DIR"])
    ensure_dir(run_config["META_SAVE_DIR"])

    print("\n" + "=" * 90)
    print(f"[Stage2-MMSE] DATASET = {run_config['DATASET_OPTION']}")
    print(f"[Stage2-MMSE] REP_LAYER = {run_config['REP_LAYER']}")
    print(f"[Stage2-MMSE] FOLD_ID = {run_config['FOLD_ID']}")
    print(f"[Stage2-MMSE] FOLD_JSON_PATH = {run_config['FOLD_JSON_PATH']}")
    print(f"[Stage2-MMSE] STAGE1_SAVE_DIR = {run_config['STAGE1_SAVE_DIR']}")
    print(f"[Stage2-MMSE] STAGE2_SAVE_DIR = {run_config['STAGE2_SAVE_DIR']}")
    print("=" * 90)

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # official_train_test:
    # Stage2 Do not read fold json.
    # Stage2 The entrance is Stage1 with a filtered train_manifest.json and is re-checked target_mmse.
    train_wav_entries_raw = load_stage1_valid_train_manifest(run_config["STAGE1_SAVE_DIR"])

    train_wav_entries, missing_train_mmse = filter_entries_with_target_mmse_for_stage2(
        train_wav_entries_raw
    )

    val_wav_entries = []

    if len(train_wav_entries) == 0:
        raise ValueError(
            f"Stage2 official_train After filtering target_mmse empty:"
            f"dataset={run_config['DATASET_OPTION']}"
        )

    train_entries = build_chunk_entries_from_fold_entries(train_wav_entries, split_name="train")
    val_entries = []

    save_json(
        {"entries": missing_train_mmse},
        os.path.join(run_config["META_SAVE_DIR"], "stage2_missing_train_mmse_entries.json"),
    )

    print(
        f"[Stage2-MMSE][OFFICIAL-TRAIN] "
        f"stage1_manifest_wavs={len(train_wav_entries_raw)}, "
        f"valid_mmse_wavs={len(train_wav_entries)}, "
        f"missing_mmse_wavs={len(missing_train_mmse)}, "
        f"train_chunks={len(train_entries)}"
    )

    print(
        f"[Stage2-MMSE][NO-VAL-LAST] chunk-level Implementation completed:"
        f"train_wavs={len(train_wav_entries)}, train_chunks={len(train_entries)}, "
    )
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    print(
        f"[Stage2-MMSE] chunk-level Implementation completed:"
        f"train_wavs={len(train_wav_entries)}, train_chunks={len(train_entries)}, "
        f"val_wavs={len(val_wav_entries)}, val_chunks={len(val_entries)}"
    )

    stage1_ckpt_path = find_stage1_checkpoint(
        stage1_save_dir=run_config["STAGE1_SAVE_DIR"],
        override_ckpt=STAGE1_CKPT_OVERRIDE,
    )
    print(f"[Stage2-MMSE] Use of Stage1 checkpoint = {stage1_ckpt_path}")

    # Equipment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_faiss_gpu = bool(FAISS_GPU) and torch.cuda.is_available()
    if bool(FAISS_GPU) and not torch.cuda.is_available():
        print("[warn] You have set FAISS_GPU=True but there is no CUDA at present, and you will return to CPU FAISS.")

    # Model
    model = WhisperEncoderWrapper(
        model_name=WHISPER_MODEL_NAME,
        rep_layer=int(run_config["REP_LAYER"]),
    ).to(device)
    model.eval()

    load_stage1_ckpt_into_whisper_model(model.model, stage1_ckpt_path)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Data
    # no-val-last: Construct only train dataset / train loaders.
    train_ds = ChunkAudioDataset(train_entries)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=int(TRAIN_BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=train_ds.repeat30s_collate,
        pin_memory=True,
        persistent_workers=(int(NUM_WORKERS) > 0),
    )

    train_infer_loader = data.DataLoader(
        train_ds,
        batch_size=int(INFER_BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=train_ds.repeat30s_collate,
        pin_memory=True,
        persistent_workers=(int(NUM_WORKERS) > 0),
    )

    val_infer_loader = None
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 1) Collecting fold-train samples for KMeans
    print(f"[Stage2-MMSE][Layer{run_config['REP_LAYER']}] Start collecting fold-train features for cluster...")
    X_train = collect_train_samples(
        model=model,
        loader=train_loader,
        device=device,
        sample_ratio=SAMPLE_RATIO,
        sample_per_utt=SAMPLE_PER_UTT,
        sample_mode=SAMPLE_MODE,
    )
    print(f"[Stage2-MMSE][Layer{run_config['REP_LAYER']}] Collection completed: sample ={X_train.shape[0]},dimension={X_train.shape[1]}")

    # 2) Train every KK KMeans
    ks = [int(x) for x in KS]
    if len(ks) == 0:
        raise ValueError("KS Can't be empty.")

    index_list = []
    kmeans_stats = []
    for k in tqdm(ks, desc="Training FAISS KMeans (Ks)", dynamic_ncols=True):
        print(f"\n[Stage2-MMSE][Layer{run_config['REP_LAYER']}] Training KMeans: K={k} (gpu={use_faiss_gpu}) ...")
        centroids, index = fit_faiss_kmeans(
            X_train=X_train,
            k=int(k),
            seed=int(SEED),
            niter=int(KMEANS_MAX_ITER),
            nredo=int(FAISS_NREDO),
            use_gpu=use_faiss_gpu,
            max_points_per_centroid=int(FAISS_MAX_POINTS_PER_CENTROID),
            verbose=bool(FAISS_VERBOSE),
        )
        index_list.append(index)
        kmeans_stats.append(
            {
                "k": int(k),
                "centroids_shape": [int(x) for x in centroids.shape],
                "feature_dim": int(centroids.shape[1]),
            }
        )

#++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 3) Export only false labels train
    infer_and_export_split_json(
        model=model,
        loader=train_infer_loader,
        index_list=index_list,
        ks=ks,
        device=device,
        outputdir=run_config["STAGE2_TRAIN_DIR"],
        split_name="train",
        pad_to_full=bool(PAD_TO_FULL_LENGTH),
        entries=train_entries,
    )

    if bool(STAGE2_EXPORT_FOLD_VAL_LABELS):
        raise RuntimeError(
            "The current target is no-val-last: Stage2 not to allow export fold_val pseudo labels."
        )
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 3.5) Stage2 export sanity check
    train_all_clus_path = os.path.join(run_config["STAGE2_TRAIN_DIR"], "all-clus.json")

    check_all_clus_heads(train_all_clus_path, KS)

    write_stage2_run_meta(
        cfg=run_config,
        stage1_ckpt_path=stage1_ckpt_path,
    )

    print(f"[Stage2][NO-VAL-LAST] checked train all-clus heads: KS={KS}")
    print(f"[Stage2][NO-VAL-LAST] wrote stage2_run_meta.json -> {os.path.join(run_config['STAGE2_SAVE_DIR'], 'stage2_run_meta.json')}")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    # 4) Save meta-information
    if bool(SAVE_EXPORT_METADATA):
        stage2_meta = {
            "dataset_option": str(run_config["DATASET_OPTION"]),
            "pitt_subject_disjoint": bool(str(run_config["DATASET_OPTION"]) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
            "pitt_split_policy": (PITT_SPLIT_POLICY if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
            "pitt_expected_final_train_subjects": (PITT_EXPECTED_FINAL_TRAIN_SUBJECTS if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
            "pitt_expected_final_train_visits": (PITT_EXPECTED_FINAL_TRAIN_VISITS if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
            "rep_layer": int(run_config["REP_LAYER"]),
            "fold_id": None,
            "fold_disabled": True,
            "run_protocol": str(run_config.get("RUN_PROTOCOL_TAG", "")),
            "fold_json_path": "",
            "run_root": str(run_config["RUN_ROOT"]),
            "stage1_save_dir": str(run_config["STAGE1_SAVE_DIR"]),
            "stage2_save_dir": str(run_config["STAGE2_SAVE_DIR"]),
            "stage1_ckpt_path": str(stage1_ckpt_path),
            "whisper_model_name": str(WHISPER_MODEL_NAME),

            "ks": [int(x) for x in KS],
            "sample_mode": str(SAMPLE_MODE),
            "sample_ratio": float(SAMPLE_RATIO) if SAMPLE_RATIO is not None else None,
            "sample_per_utt": int(SAMPLE_PER_UTT),
            "segment_sec": float(SEGMENT_SEC),

            "overlap_rate": float(OVERLAP_RATE),
            "chunk_key_version": str(CHUNK_KEY_VERSION),
            "unfreeze_encoder_layers": int(UNFREEZE_ENCODER_LAYERS),
            "layer_name": str(run_config["LAYER_NAME"]),
            "train_num_wavs": int(len(train_wav_entries)),


            "official_train_num_wavs_raw_from_stage1_manifest": int(len(train_wav_entries_raw)),
            "official_train_missing_mmse_wavs_stage2": int(len(missing_train_mmse)),
            "fold_val_num_wavs_held_out": 0,
            "train_num_chunks": int(len(train_entries)),
            "val_num_chunks": 0,
            "stage2_export_fold_val_labels": False,
            "upstream_uses_fold_val": False,

            "pad_to_full_length": bool(PAD_TO_FULL_LENGTH),
            "faiss_gpu": bool(use_faiss_gpu),
            "kmeans_stats": kmeans_stats,
        }
        save_json(stage2_meta, os.path.join(run_config["STAGE2_SAVE_DIR"], "clusters_meta.json"))
        save_json(stage2_meta, os.path.join(run_config["META_SAVE_DIR"], "stage2_meta.json"))
        save_text(stage1_ckpt_path, os.path.join(run_config["META_SAVE_DIR"], "stage1_ckpt_used.txt"))



    # 5) Stage2 Cluster quality evaluation
    # no-val-last train/val distribution is not compared to quality evaluation as fold_val is not used upstream.
    if bool(ENABLE_STAGE_QUALITY_EVAL):
        raise RuntimeError(
            "no-val-last Set to keep ENABLE_STAGE_QUALITY_EVAL=False,"
            "Avoids making quality judgements using the old_val."
        )

    print(f"\n[Stage2-MMSE] Completed:{run_config['STAGE2_SAVE_DIR']}")

# ============================================================
# 8) Main entrance: past layer / fold
# ============================================================
def main() -> None:
    """
    Main entrance:
    - Traveling through DATASET_OPTIONS;
    - Traveling through UNFREEZE_ENCODER_LAYERS_LIST;
    - Traveling through STAGE2_REP_LAYERS_TO_RUN;
    - Do not travel to fold;
    - Stage2 Use Stage1 valid train_manifest as the official_train portal.
    """
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS There is a minimum need for a data set to be empty.")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST To be empty, a minimum number of unfrozen layers is required.")

    print("[CONFIG]")
    print("  PROJECT_ROOT =", PROJECT_ROOT)
    print("  MMSE_CPFT_ROOT =", MMSE_CPFT_ROOT)
    print("  DATASET_OPTIONS =", DATASET_OPTIONS)
    print("  RUN_PROTOCOL_TAG =", RUN_PROTOCOL_TAG)
    print("  STAGE2_REP_LAYERS_TO_RUN =", STAGE2_REP_LAYERS_TO_RUN)
    print("  UNFREEZE_ENCODER_LAYERS_LIST =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  KS =", KS)
    print("  SAMPLE_MODE =", SAMPLE_MODE)
    print("  SAMPLE_PER_UTT =", SAMPLE_PER_UTT)
    print("  fold disabled = True")
    print("  protocol subdir disabled = True")

    for dataset_option in [_canonical_dataset_option(x) for x in DATASET_OPTIONS]:
        for unfreeze_count in [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "=" * 100)
            print("[Stage2 BATCH ITEM]")
            print("  DATASET_OPTION =", DATASET_OPTION)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("=" * 100)

            for rep_layer in [int(x) for x in STAGE2_REP_LAYERS_TO_RUN]:
                run_config = build_stage2_run_config(
                    dataset_option=DATASET_OPTION,
                    rep_layer=int(rep_layer),
                    fold_id=PROTOCOL_FOLD_ID,
                )
                run_one_experiment(run_config)


if __name__ == "__main__":
    main()
