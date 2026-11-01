#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPFT_3class_extract.py

Cluster Pseudo-Label Fine-Tuning (CPFT).
============================================================

PROCESS-2 HC / MCI / Dementia Export script for the final characteristic of the three categories CPFT.

Diagnos-aware CPFT for CPFT_3class_stage3.py
Stage3 checkpoint Whisper backbone to load back to HuggingFace
WhisperModel,Export of financial TRAIN/ downstream LRU_3class.py
official TEST The segment-level feature.

Definitions of the three categories
----------
    0 = HC
    1 = MCI
    2 = Dementia

Unique Label Source
------------
    /home/user/Downloads/PROCESS-2/meta-info.csv

Use only:
    IDs
    diagnosis
    Split

MMSE / age / gender Do not participate in label generation.

CPFT checkpoint
-----------------
/home/user/Alzheimer_code_jax_class/
    finetune/ft_CPFT/
    PROCESS-2/
    CPFT_lr5e-5_epo5_3class/
    singleX_unfreeZ/
    stage3_CPFT_maskoffK/
    last.ckpt

Final features
--------
/home/user/Alzheimer_code_jax_class/
    feature_tensor/
    PROCESS-2/
    ft_CPFT_lr5e-5_epo5_3class/
    singleX_unfreeZ/

For example:
    feature_tensor/
    PROCESS-2/
    ft_CPFT_lr5e-5_epo5_3class/
    single12_unfree7/

Each selection directory still contains six main files:

    feature_whisper_single12_0.3_30s_train.pt
    feature_whisper_single12_0.3_30s_test.pt

    labels1_whisper_single12_0.3_30s_train.pt
    labels2_whisper_single12_0.3_30s_test.pt

    indices_whisper_single12_0.3_30s_train.pt
    indices_whisper_single12_0.3_30s_test.pt

Data protocol
--------
1. Stage1 / Stage2 / Stage3 Use official TRAIN only.
2. official TEST Read only when this script finally exports features.
3. This script does not train any models.
4. This script does not use official TEST to select checkpoint.
6. PROCESS-2 Read *__CTD.wav only.
7. The label is determined by meta-info.csv diagnosis:
       HC        -> 0
       MCI       -> 1
       Dementia  -> 2
8. cc/cd Directory only for consistency check:
       cc -> HC
       cd -> MCI / Dementia
9. stable audio_id Use the number PROCESS-2 participant,
   Consistency with frozen.py.

Characteristic protocols
--------
1. librosa.load(sr=16000, mono=True)
2. 30 seconds
3. 30% overlap
4. single layer extraction
5. ln_post
6. features Save as torch.float32
7. labels Save as torch.long
8. indices Saved as NumPy int64 [N,2]:
       column 0 = audio_id
       column 1 = segment_id

Interface with downstream
----------
LRU_3class.py freeze baseline:

    FEATURE_MODEL_NAME = "whisper_small_3class"

Switch to category CPFT:

    FEATURE_MODEL_NAME = "ft_CPFT_lr5e-5_epo5_3class"

The remaining LRU training logic of the three categories does not need to be modified.

Run
----
conda activate whisper_cpft_faiss
python CPFT_3class_extract.py

Cluster Pseudo-Label Fine-Tuning (CPFT).
"""


try:
    from .paper_names import paper_dataset_name
except ImportError:
    from paper_names import paper_dataset_name

# =============================================================================
# 0) Environment and random seeds: must be set before import torch
# =============================================================================
import os
import random
import numpy as np

# A visible binder card; it is controlled externally by CUDA_VISIBLE_DEVICES when set to None.
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


# =============================================================================
# 1) Remaining Imports
# =============================================================================
import copy
import gc
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor


# =============================================================================
# 2) Parameters Settings: must be aligned to 3class_stage1 / 3class_stage2 / 3class_stage3
# =============================================================================

# Current PROCOESS-2 Category III Project Root Directory.
CLASS_PROJECT_ROOT = "/home/user/Alzheimer_code_jax_class"

# CPFT Stage1/2/3 An intermediate result root directory.
PROJECT_ROOT = os.path.join(CLASS_PROJECT_ROOT, "finetune")
CPFT_ROOT = os.path.join(PROJECT_ROOT, "ft_CPFT_3class")

# Final feature root directory.
FEATURE_ROOT = os.path.join(CLASS_PROJECT_ROOT, "feature_tensor")

# Only PROCESS-2 is running the current three classification experiments.
DATASET_OPTIONS = [
    "PROCESS-2",
]

# Keeps a single value variable for internal recycling.
DATASET_OPTION = DATASET_OPTIONS[0]

# The expression layer that needs to be extracted must be consistent with the training level Stage1/2/3.
LAYERS_TO_RUN = [12]

# How many layers of encoder have been unfrozen continuously relative to the current one.
# Formal batches may be used [7,8,9,10,11,12];
# A group may provisionally be [1] or [7] at the time of the first debug.
UNFREEZE_ENCODER_LAYERS_LIST = [2,3,4,5,6,7,8,9,10,11,12]
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# Stage3 mask Injection deviations must be consistent with CPFT_3class_stage3.py.
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]

# Upstream protocol tags are only written in meta and do not add to the directory hierarchy.
RUN_PROTOCOL_TAG = "official_train_test"


# Stage1 / Stage2 / Stage3 / extract There must be unity.
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5_3class"

# Path experiment label.
STAGE1_SOURCE_PATH_PARAM_TAGS: List[str] = []
PATH_PARAM_TAGS: List[str] = []

# Stage3 Configure, for meta records only, to be consistent with 3class_stage2/3class_stage3.
CPFT_KS = [32, 64]
CPFT_EXPECTED_NUM_CLUSTERS = [32, 64]
CPFT_STAGE3_LAMBDA_CLUS = 1.0
CPFT_STAGE3_LAMBDA_DIAG = 0.05
CPFT_METHOD_TAG = "diagnosis_aware_cpft"

# PROCESS-2 Three classification definitions with the sole label source.
NUM_CLASSES = 3
CLASS_NAMES = {
    0: "HC",
    1: "MCI",
    2: "Dementia",
}

PROCESS2_META_PATH = (
    "/home/user/Downloads/PROCESS-2/meta-info.csv"
)
STRICT_META_AUDIO_COVERAGE = True

# Original local directory Whisper small.
# Stage3 checkpoint is Lightning checkpoint, not HuggingFace model catalogue,
# So you create the Whisper structure here and then insert the Whisper parameter in Stage3.
WHISPER_BASE_MODEL_NAME = "/home/user/whisper_lru_3/whisper_small"
LOCAL_FILES_ONLY = True

# Manually specified Stage3 checkpoint.
# None is automatically read the stage3_CPFT_maskoffK/last.ckpt corresponding to the current configuration.
STAGE3_CKPT_OVERRIDE: Optional[str] = None

# Feature filename prefix.
FEATURE_PREFIX = "whisper"

# 30 Seconds + 30% overlap, consistent with freezing Whisper, Stage2, Stage3.
SAMPLE_RATE = 16000
SEGMENT_SEC = 30
SEGMENT_LENGTH = int(SEGMENT_SEC * SAMPLE_RATE)
OVERLAP_RATE = 0.3
OV_STR = "0.3"
SEC_STR = f"{SEGMENT_SEC}s"

# Single-layer features are normalized.
# Optional: None / "layernorm" / "std" / "ln_post"
NORM_FOR_SINGLE = "ln_post"
DROP_INPUT_LAYER = True

# Consistency with freezing of Whisper feature extraction.
ENABLE_AUDIO_NORMALIZE = True

# Whether to save feature_meta.json, manifest and protocol note.
SAVE_META = True

# The default directory specified by the user does not contain maskoff:
#     ft_CPFT_lr5e-5_epo5_3class/
#
# If more than one mask_offset is exported at a later time, it may be changed to True to prevent cross-covering,
# The directory will then become ft_CPFT_lr5e-5_epo5_3class_maskoffK/.
INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 3) Randomity and Common Tools
# =============================================================================
torch.manual_seed(400)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(400)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def ensure_dir(path: str) -> None:
    """Automatically create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def natural_key(path_value: str):
    """Sort naturally so that S2 is located before S10."""
    name = Path(str(path_value)).name
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.findall(r"\d+|\D+", name)
    ]


def _clean_path_tag(value) -> str:
    """Clears the path parameter labels."""
    if value is None:
        return ""

    text = str(value).strip()
    if text.lower() in {"", "none", "null"}:
        return ""

    text = text.replace("=", "").replace(" ", "")
    text = re.sub(r"[^0-9A-Za-z_.-]+", "", text)
    return text.strip("_-.")


def build_path_param_suffix(path_param_tags=None) -> str:
    """
    Construct the directory suffix from PATH_PARAM_TAGS.

    Example:
        []                  -> ""
        ["ks64"]            -> "ks64"
        ["ks64", "ad005"]   -> "ks64_ad005"
    """
    if path_param_tags is None:
        path_param_tags = PATH_PARAM_TAGS

    if isinstance(path_param_tags, (str, int, float)):
        tags = [path_param_tags]
    else:
        tags = list(path_param_tags)

    cleaned = [_clean_path_tag(value) for value in tags]
    cleaned = [value for value in cleaned if value != ""]
    return "_".join(cleaned)


def build_layer_name(
    single_layer: int,
    unfreeze_layer_count: int,
    path_param_tags=None,
) -> str:
    """
    Construct a uniform layer directory name.

    For example:
        single_layer=12, unfreeze=1 -> single12_unfree1
        single_layer=12, unfreeze=7 -> single12_unfree7
    """
    base = (
        f"single{int(single_layer)}_"
        f"unfree{int(unfreeze_layer_count)}"
    )
    suffix = build_path_param_suffix(path_param_tags)
    return base if suffix == "" else f"{base}_{suffix}"


def build_stage3_subdir(mask_offset: int) -> str:
    """Construct subdirectories Stage3 checkpoint."""
    return f"stage3_CPFT_maskoff{int(mask_offset)}"


def get_feature_model_tag(mask_offset: int) -> str:
    """
    Construct the model directory name under feature_tensor.

    Default:
        ft_CPFT_lr5e-5_epo5

    If INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG=True:
        ft_CPFT_lr5e-5_epo5_maskoff1
    """
    base = f"ft_{MODEL_ROOT_TAG}"

    if INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG:
        return f"{base}_maskoff{int(mask_offset)}"

    return base


def get_feature_selection_tag(single_layer: int) -> str:
    """Construct the selection directory name under feature_tensor."""
    return build_layer_name(
        single_layer=int(single_layer),
        unfreeze_layer_count=int(UNFREEZE_ENCODER_LAYERS),
        path_param_tags=PATH_PARAM_TAGS,
    )


def get_save_root(
    dataset_option: str,
    single_layer: int,
    mask_offset: int,
) -> str:
    """
    Constructs the final feature save directory.

    Default output:
        feature_tensor/<DATASET>/
        ft_CPFT_lr5e-5_epo5_3class/
        singleX_unfreeZ/
    """
    return os.path.join(
        FEATURE_ROOT,
        str(dataset_option),
        get_feature_model_tag(mask_offset),
        get_feature_selection_tag(single_layer),
    )


def canonical_wav_path(path: str) -> str:
    """Harmonizes the absolute path wav."""
    return str(Path(str(path)).expanduser().resolve())


def save_json(obj: Dict[str, Any], path: str) -> None:
    """Save UTF-8 JSON."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def save_text(text: str, path: str) -> None:
    """Saves UTF-8 text."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as file:
        file.write(str(text))


def torch_load_compatible(path: str, map_location="cpu"):
    """
    torch.load compatible with PyTorch 2.6+.

    Currently checkpoint is generated by this item Lightning and has a credible source,
    Therefore, use weights_only=False in a visible format.
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


# =============================================================================
# 4) PROCESS-2 official train / official test Audio Directory
# =============================================================================
DATASETS: Dict[str, Dict[str, str]] = {
    "PROCESS-2": {
        "train_cc": "/home/user/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/user/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/user/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/user/Downloads/PROCESS-2/test/cd",
    },
}


def normalize_process2_participant_id(value: str) -> str:
    """
    Harmonization PROCESS-2 participant ID.

    meta:
        PROCESS-2_rec__003

    wav:
        PROCESS-2_rec__003__CTD.wav

    Unified as:
        process-2_rec__003
    """
    text = Path(str(value)).stem.strip()
    text = re.sub(
        r"__(SFT|PFT|CTD)$",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = text.strip().lower()

    if text == "":
        raise ValueError(
            f"Unable to regularize PROCESS-2 participant ID:{value!r}"
        )

    return text


def process2_audio_id(participant_id: str) -> int:
    """
    Use the number PROCESS-2 participant as stabilization audio_id.

    For example:
        process-2_rec__003 -> 3

    Frozen The same rule shall be used for two categories of CPFT and three categories of baseline.
    """
    normalized = normalize_process2_participant_id(
        participant_id
    )
    match = re.search(
        r"__([0-9]+)$",
        normalized,
    )
    if match is None:
        raise ValueError(
            "Unable to extract number audio_id from PROCESS-2 participant ID:"
            f"{participant_id!r}"
        )

    return int(match.group(1))


def normalize_process2_diagnosis(
    value: str,
) -> Tuple[str, int]:
    """
    PROCESS-2 diagnosis Third Class Map:

        HC        -> 0
        MCI       -> 1
        Dementia  -> 2
    """
    text = str(value).strip().lower()

    mapping = {
        "hc": ("HC", 0),
        "mci": ("MCI", 1),
        "dementia": ("Dementia", 2),
    }

    if text not in mapping:
        raise ValueError(
            "PROCESS-2 diagnosis Allow Only "
            f"HC/MCI/Dementia,Current ={value!r}"
        )

    return mapping[text]


def normalize_process2_split(value: str) -> str:
    """Harmonize the TRAIN / TEST fields for meta-info.csv."""
    text = str(value).strip().lower()

    if text in {"train", "training"}:
        return "train"

    if text in {"test", "testing"}:
        return "test"

    raise ValueError(
        "PROCESS-2 Split Allow TRAIN/TEST only."
        f"Current ={value!r}"
    )


def load_process2_meta(
    meta_path: str = PROCESS2_META_PATH,
) -> Dict[str, Dict[str, Any]]:
    """
    Read PROCESS-2 meta-info.csv.

    The three categories are used only:
        IDs
        diagnosis
        Split

    MMSE / age / gender I don't think that's a good idea.
    MMSE Missing will not filter any participant.
    """
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            "PROCESS-2 meta-info.csv Does not exist: \n"
            f"{meta_path}"
        )

    frame = pd.read_csv(meta_path)

    required_columns = {
        "IDs",
        "diagnosis",
        "Split",
    }
    missing_columns = (
        required_columns
        - set(frame.columns)
    )
    if missing_columns:
        raise ValueError(
            "meta-info.csv Synchronising folder failed: %s: %s"
            f"{sorted(missing_columns)}"
        )

    meta_map: Dict[
        str,
        Dict[str, Any],
    ] = {}

    for row_index, row in frame.iterrows():
        participant_id = (
            normalize_process2_participant_id(
                row["IDs"]
            )
        )

        diagnosis_name, target_label = (
            normalize_process2_diagnosis(
                row["diagnosis"]
            )
        )

        split_name = (
            normalize_process2_split(
                row["Split"]
            )
        )

        if participant_id in meta_map:
            raise ValueError(
                "meta-info.csv participant ID Repeat:"
                f"{participant_id}"
            )

        meta_map[participant_id] = {
            "participant_id":
                participant_id,
            "raw_id":
                str(row["IDs"]),
            "diagnosis":
                diagnosis_name,
            "target_label":
                int(target_label),
            "split":
                split_name,
            "meta_row_index":
                int(row_index),
        }

    if len(meta_map) == 0:
        raise ValueError(
            "meta-info.csv No valid participant."
        )

    labels = [
        int(row["target_label"])
        for row in meta_map.values()
    ]
    if sorted(set(labels)) != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            "meta-info.csv Must be included at the same time. "
            "HC/MCI/Dementia Three groups,"
            f"Current ={sorted(set(labels))}"
        )

    train_labels = np.asarray(
        [
            int(row["target_label"])
            for row in meta_map.values()
            if row["split"] == "train"
        ],
        dtype=np.int64,
    )
    test_labels = np.asarray(
        [
            int(row["target_label"])
            for row in meta_map.values()
            if row["split"] == "test"
        ],
        dtype=np.int64,
    )

    print(
        "[PROCESS-2][META] "
        f"total={len(meta_map)} | "
        f"TRAIN={len(train_labels)} | "
        f"TEST={len(test_labels)}"
    )
    print(
        "[PROCESS-2][META][TRAIN] "
        f"HC={int(np.sum(train_labels == 0))} | "
        f"MCI={int(np.sum(train_labels == 1))} | "
        f"Dementia={int(np.sum(train_labels == 2))}"
    )
    print(
        "[PROCESS-2][META][TEST] "
        f"HC={int(np.sum(test_labels == 0))} | "
        f"MCI={int(np.sum(test_labels == 1))} | "
        f"Dementia={int(np.sum(test_labels == 2))}"
    )

    return meta_map


def _is_process2_ctd_wav_path(
    wav_path: str,
    dataset_key: Optional[str] = None,
) -> bool:
    """
    PROCESS-2 Only the name of the document stem should be retained at the end of __CTD wav.
    All other data sets are maintained.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    stem = Path(str(wav_path)).stem
    return re.search(
        r"__CTD$",
        stem,
        flags=re.IGNORECASE,
    ) is not None


def list_wav_files(
    audio_dir: str,
    dataset_key: Optional[str] = None,
) -> List[str]:
    """
    Reads wav in the sub-directories official and sorts it in natural order.

    Only the first level of the directory is currently scanned, consistent with the original extraction script.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    audio_dir = str(audio_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(
            f"official audio dir Cannot initialise Evolution's mail component.{audio_dir}"
        )

    files: List[str] = []

    for file_name in os.listdir(audio_dir):
        if not str(file_name).lower().endswith(".wav"):
            continue

        wav_path = os.path.join(audio_dir, file_name)

        if not _is_process2_ctd_wav_path(
            wav_path,
            dataset_key=dataset_key,
        ):
            continue

        files.append(wav_path)

    files = sorted(files, key=natural_key)

    if len(files) == 0:
        raise ValueError(
            "official audio dir No eligible wav:"
            f"dir={audio_dir}, dataset={dataset_key}"
        )

    if str(dataset_key).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] {audio_dir}: "
            f"Read only CTD wav, number of documents ={len(files)}"
        )

    return [
        canonical_wav_path(path)
        for path in files
    ]


def build_process2_official_entries(
    dataset_paths: Dict[str, str],
    meta_map: Dict[str, Dict[str, Any]],
    split_name: str,
) -> List[Dict[str, Any]]:
    """
    Scanning CTD wav from the PROCESS-2 official directory,
    The three labels are decided by the diagnosis of meta-info.csv.

    Folder only for consistency verification:

        cc -> HC
        cd -> MCI / Dementia

    Eventually target_label is not generated directly by cc/cd.
    """
    split_name = normalize_process2_split(
        split_name
    )

    if split_name == "train":
        subset_names = (
            ("train_cc", "cc"),
            ("train_cd", "cd"),
        )
    else:
        subset_names = (
            ("test_cc", "cc"),
            ("test_cd", "cd"),
        )

    entries: List[
        Dict[str, Any]
    ] = []
    seen_participants = set()

    for subset_name, source_group in subset_names:
        wav_files = list_wav_files(
            dataset_paths[subset_name],
            dataset_key="PROCESS-2",
        )

        for wav_path in wav_files:
            participant_id = (
                normalize_process2_participant_id(
                    wav_path
                )
            )

            if participant_id not in meta_map:
                raise KeyError(
                    "CTD wav Could not close temporary folder: %s"
                    f"wav={wav_path}\n"
                    f"participant_id={participant_id}"
                )

            if participant_id in seen_participants:
                raise ValueError(
                    f"{split_name} participant repeats:"
                    f"{participant_id}"
                )

            meta = meta_map[
                participant_id
            ]

            if meta["split"] != split_name:
                raise ValueError(
                    "official The list of contents is inconsistent with meta Split: \n"
                    f"participant={participant_id}\n"
                    f"directory_split={split_name}\n"
                    f"meta_split={meta['split']}"
                )

            diagnosis = str(
                meta["diagnosis"]
            )
            target_label = int(
                meta["target_label"]
            )

            if (
                source_group == "cc"
                and diagnosis != "HC"
            ):
                raise ValueError(
                    "PROCESS-2 cc Folders must be HC:\n"
                    f"participant={participant_id}\n"
                    f"diagnosis={diagnosis}\n"
                    f"wav={wav_path}"
                )

            if (
                source_group == "cd"
                and diagnosis
                not in {
                    "MCI",
                    "Dementia",
                }
            ):
                raise ValueError(
                    "PROCESS-2 cd Folders must be MCI/Dementia:\n"
                    f"participant={participant_id}\n"
                    f"diagnosis={diagnosis}\n"
                    f"wav={wav_path}"
                )

            audio_id = process2_audio_id(
                participant_id
            )

            entries.append(
                {
                    "wav_path":
                        canonical_wav_path(
                            wav_path
                        ),
                    "wav_name":
                        Path(wav_path).name,

                    "participant_id":
                        participant_id,
                    "raw_participant_id":
                        str(meta["raw_id"]),

                    "official_audio_id":
                        int(audio_id),
                    "audio_id":
                        int(audio_id),

                    "split":
                        split_name,

                    # Keep group/source_group at the same time,
                    # Allows manual checking of the syntax of the old directory.
                    "group":
                        source_group,
                    "source_group":
                        source_group,
                    "source_subset":
                        f"{split_name}_{source_group}",

                    "diagnosis":
                        diagnosis,
                    "target_label":
                        target_label,

                    "process2_task_filter":
                        "CTD",
                    "label_source":
                        "meta-info.csv:diagnosis",
                    "split_source":
                        "meta-info.csv:Split",
                }
            )

            seen_participants.add(
                participant_id
            )

    entries = sorted(
        entries,
        key=lambda row: int(
            row["audio_id"]
        ),
    )

    expected_participants = {
        participant_id
        for participant_id, row
        in meta_map.items()
        if row["split"] == split_name
    }
    actual_participants = {
        str(row["participant_id"])
        for row in entries
    }

    missing_audio = sorted(
        expected_participants
        - actual_participants
    )
    unexpected_audio = sorted(
        actual_participants
        - expected_participants
    )

    if unexpected_audio:
        raise ValueError(
            f"{split_name} Participant:"
            f"{unexpected_audio[:20]}"
        )

    if (
        STRICT_META_AUDIO_COVERAGE
        and missing_audio
    ):
        raise ValueError(
            f"{split_name} meta participant No audio from CTD found."
            f"Missing ={len(missing_audio)},"
            f"Example:{missing_audio[:20]}"
        )

    labels = np.asarray(
        [
            int(row["target_label"])
            for row in entries
        ],
        dtype=np.int64,
    )

    if sorted(
        np.unique(labels).tolist()
    ) != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            f"{split_name} official entries "
            "There are no three categories:"
            f"{sorted(np.unique(labels).tolist())}"
        )

    print(
        f"[ENTRY] {split_name} | "
        f"total={len(entries)} | "
        f"HC={int(np.sum(labels == 0))} | "
        f"MCI={int(np.sum(labels == 1))} | "
        f"Dementia={int(np.sum(labels == 2))}"
    )

    return entries


def load_official_train_test_entries(
    dataset_option: str,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """
    Construct the three categories PROCESS-2 official train / official test entries.

    Label only source:
        meta-info.csv diagnosis

    stable audio_id:
        PROCESS-2_rec__003 -> 3

    train/test Renumber enumerate separately.
    """
    if str(
        dataset_option
    ).strip() != "PROCESS-2":
        raise ValueError(
            "CPFT_3class_extract.py For the time being, only PROCESS-2 is supported."
            f"Got it.daset_option={dataset_option}"
        )

    if dataset_option not in DATASETS:
        raise ValueError(
            f"DATASETS No Configuration {dataset_option}"
        )

    meta_map = load_process2_meta(
        PROCESS2_META_PATH
    )
    paths = DATASETS[
        dataset_option
    ]

    train_entries = (
        build_process2_official_entries(
            dataset_paths=paths,
            meta_map=meta_map,
            split_name="train",
        )
    )
    test_entries = (
        build_process2_official_entries(
            dataset_paths=paths,
            meta_map=meta_map,
            split_name="test",
        )
    )

    if (
        len(train_entries) == 0
        or len(test_entries) == 0
    ):
        raise ValueError(
            "official train/test entries Empty:"
            f"train={len(train_entries)}, "
            f"test={len(test_entries)}"
        )

    # Formal number checks are fixed.
    expected = {
        "train": {
            "total": 320,
            "HC": 160,
            "MCI": 120,
            "Dementia": 40,
        },
        "test": {
            "total": 80,
            "HC": 40,
            "MCI": 30,
            "Dementia": 10,
        },
    }

    for split_name, entries in [
        ("train", train_entries),
        ("test", test_entries),
    ]:
        labels = np.asarray(
            [
                int(row["target_label"])
                for row in entries
            ],
            dtype=np.int64,
        )
        actual = {
            "total":
                int(len(labels)),
            "HC":
                int(np.sum(labels == 0)),
            "MCI":
                int(np.sum(labels == 1)),
            "Dementia":
                int(np.sum(labels == 2)),
        }

        if (
            STRICT_META_AUDIO_COVERAGE
            and actual
            != expected[split_name]
        ):
            raise ValueError(
                "PROCESS-2 official "
                f"{split_name} Number of Classes Unusual: \n"
                f"actual={actual}\n"
                f"expected={expected[split_name]}"
            )

    return (
        train_entries,
        test_entries,
    )


def summarize_audio_entries(
    entries: Sequence[Dict[str, Any]],
    split_name: str,
) -> Dict[str, int]:
    """
    Statistical distribution official audio-level HC/MCI/Dementia,
    and check that each diagnosis in entry is consistent with target_label.
    """
    if len(entries) == 0:
        raise ValueError(
            f"{split_name} entries It's empty."
        )

    expected_diagnosis = {
        0: "HC",
        1: "MCI",
        2: "Dementia",
    }

    labels: List[int] = []
    audio_ids: List[int] = []

    for row in entries:
        label = int(
            row["target_label"]
        )
        if label not in {
            0,
            1,
            2,
        }:
            raise ValueError(
                f"{split_name} (a) The emergence of illegal three-class labels:"
                f"{label}"
            )

        diagnosis = str(
            row.get(
                "diagnosis",
                "",
            )
        )
        if diagnosis != expected_diagnosis[
            label
        ]:
            raise ValueError(
                f"{split_name} diagnosis Inconsistent with label:"
                f"diagnosis={diagnosis}, "
                f"label={label}"
            )

        labels.append(label)
        audio_ids.append(
            int(
                row[
                    "official_audio_id"
                ]
            )
        )

    if len(audio_ids) != len(
        set(audio_ids)
    ):
        raise ValueError(
            f"{split_name} audio_id There is duplication."
        )

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    unique_labels = sorted(
        np.unique(
            labels_array
        ).tolist()
    )
    if unique_labels != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            f"{split_name} It is important to have a zero-one-two component at the same time."
            f"Current ={unique_labels}"
        )

    result = {
        "audio_total":
            int(len(labels_array)),

        "audio_class0":
            int(
                np.sum(
                    labels_array == 0
                )
            ),
        "audio_class1":
            int(
                np.sum(
                    labels_array == 1
                )
            ),
        "audio_class2":
            int(
                np.sum(
                    labels_array == 2
                )
            ),

        "audio_hc":
            int(
                np.sum(
                    labels_array == 0
                )
            ),
        "audio_mci":
            int(
                np.sum(
                    labels_array == 1
                )
            ),
        "audio_dementia":
            int(
                np.sum(
                    labels_array == 2
                )
            ),
    }

    print(
        f"[{split_name}] "
        f"audio_total={result['audio_total']} | "
        f"HC={result['audio_hc']} | "
        f"MCI={result['audio_mci']} | "
        f"Dementia={result['audio_dementia']}"
    )

    return result


# =============================================================================
# 5) Stage3 checkpoint Find and Load
# =============================================================================
def build_run_root(
    dataset_option: str,
    layer_id: int,
) -> str:
    """
    Construct Stage3 run root:

        CPFT_ROOT/<DATASET>/<MODEL_ROOT_TAG>/<singleX_unfreeZ>/
    """
    layer_name = build_layer_name(
        single_layer=int(layer_id),
        unfreeze_layer_count=int(UNFREEZE_ENCODER_LAYERS),
        path_param_tags=PATH_PARAM_TAGS,
    )

    return os.path.join(
        CPFT_ROOT,
        str(dataset_option),
        MODEL_ROOT_TAG,
        layer_name,
    )


def find_stage3_checkpoint(
    dataset_option: str,
    layer_id: int,
    mask_offset: int,
    override_ckpt: Optional[str] = None,
) -> str:
    """
    Find checkpoint saved at CPFT_3class_stage3.py.

    Priority:
    1. STAGE3_CKPT_OVERRIDE;
    2. stage3_CPFT_maskoffK/last.ckpt;
    3. Other updated.ckpt under directory as modified, as compatible returns.

    The current formal no-val-last protocol hits last.ckpt.
    """
    if override_ckpt not in {None, "", "None"}:
        ckpt_path = str(override_ckpt)

        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"STAGE3_CKPT_OVERRIDE Cannot initialise Evolution's mail component.{ckpt_path}"
            )

        return str(Path(ckpt_path).resolve())

    run_root = build_run_root(
        dataset_option=dataset_option,
        layer_id=int(layer_id),
    )
    stage3_dir = (
        Path(run_root)
        / build_stage3_subdir(mask_offset)
    )

    if not stage3_dir.is_dir():
        raise FileNotFoundError(
            "Stage3 Directory does not exist: \n"
            f"{stage3_dir}\n"
            "Please check DATASET_OPTIONS, LAYERS_TO_RUN,"
            "UNFREEZE_ENCODER_LAYERS_LIST,PATH_PARAM_TAGS and "
            "STAGE3_MASK_INJECT_AFTER_LAYERS Consistency with CPFT_3class_stage3.py."
        )

    last_ckpt = stage3_dir / "last.ckpt"
    if last_ckpt.is_file():
        return str(last_ckpt.resolve())

    other_ckpts = sorted(
        stage3_dir.glob("*.ckpt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if len(other_ckpts) > 0:
        print(
            "[WARN] Last.ckpt, back to the latest checkpoint:"
            f"{other_ckpts[0]}"
        )
        return str(other_ckpts[0].resolve())

    raise FileNotFoundError(
        f"Yes. {stage3_dir} No.ckpt files found below."
    )


def load_stage3_ckpt_into_whisper_model(
    model: WhisperModel,
    ckpt_path: str,
) -> Dict[str, Any]:
    """
    Loads only WhisperModel from Lighting checkpoint of CPFT_3class_stage3.py.

    Stage3 checkpoint It also includes:
        cluster_heads
        ad_head
        mask_emb

    These support modules are used only for Stage3 training and are all ignored when final features are extracted.
    """
    checkpoint = torch_load_compatible(
        ckpt_path,
        map_location="cpu",
    )
    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Stage3 checkpoint state_dict Not dictionary:{type(state_dict)}"
        )

    target_state = model.state_dict()

    # CPFT_3class_stage3.py WhisperModel attribute is called self.model,
    # Therefore, the official checkpoint usually uses the prefix "model."
    candidate_prefixes = [
        "model.",
        "whisper.model.",
        "whisper.",
        "",
    ]

    best_loaded: Dict[str, torch.Tensor] = {}
    best_prefix: Optional[str] = None

    for prefix in candidate_prefixes:
        current_loaded: Dict[str, torch.Tensor] = {}

        for key, value in state_dict.items():
            key = str(key)

            if prefix != "" and not key.startswith(prefix):
                continue

            stripped_key = (
                key[len(prefix):]
                if prefix != ""
                else key
            )

            if (
                stripped_key in target_state
                and hasattr(value, "shape")
                and tuple(target_state[stripped_key].shape)
                == tuple(value.shape)
            ):
                current_loaded[stripped_key] = value

        if len(current_loaded) > len(best_loaded):
            best_loaded = current_loaded
            best_prefix = prefix

    if len(best_loaded) == 0:
        raise RuntimeError(
            "No WhisperModel parameters matched from Stage3 checkpoint:"
            f"{ckpt_path}"
        )

    missing_keys, unexpected_keys = model.load_state_dict(
        best_loaded,
        strict=False,
    )

    encoder_matched = sum(
        1
        for key in best_loaded
        if key.startswith("encoder.")
    )
    decoder_matched = sum(
        1
        for key in best_loaded
        if key.startswith("decoder.")
    )

    report = {
        "checkpoint_path": str(ckpt_path),
        "selected_prefix": best_prefix,
        "loaded_parameter_tensors": int(len(best_loaded)),
        "encoder_matched": int(encoder_matched),
        "decoder_matched": int(decoder_matched),
        "missing_keys_count": int(len(missing_keys)),
        "unexpected_keys_count": int(len(unexpected_keys)),
    }

    print(
        "[Stage3-CLASS-CKPT] loaded\n"
        f"  checkpoint={ckpt_path}\n"
        f"  prefix={best_prefix!r}\n"
        f"  loaded={len(best_loaded)}\n"
        f"  encoder_matched={encoder_matched}\n"
        f"  decoder_matched={decoder_matched}\n"
        f"  missing={len(missing_keys)}\n"
        f"  unexpected={len(unexpected_keys)}"
    )

    if encoder_matched == 0:
        raise RuntimeError(
            "Stage3 checkpoint Although parameters are matched, encoder parameters are not matched."
        )

    return report


# =============================================================================
# 6) Whisper Single-layer feature extraction
# =============================================================================
def apply_norm(
    hidden: torch.Tensor,
    mode: Optional[str],
    ln_post=None,
    skip_ln_post: bool = False,
):
    """
    Normalize [1,T,C] features.

    Optional:
        None
        layernorm
        std
        ln_post
    """
    if mode is None:
        return hidden

    mode = str(mode).strip().lower()

    if mode == "layernorm":
        return F.layer_norm(
            hidden,
            hidden.shape[-1:],
        )

    if mode == "std":
        std = hidden.std(
            dim=(-2, -1),
            keepdim=True,
        ).clamp_min(1e-6)
        return hidden / std

    if mode in {"ln_post", "whisper_ln"}:
        if skip_ln_post:
            return hidden

        if ln_post is None:
            raise ValueError(
                "ln_post but ln_post is None."
            )

        return ln_post(hidden)

    raise ValueError(
        f"Unsupported characterizations are normalized:{mode}"
    )


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
) -> Tuple[List[torch.Tensor], List[Tuple[int, int]]]:
    """
    Slice a single wav to 30 seconds, 30% overlap and extract specified layer characteristics.

    Return:
        segment_features:List[[T,C] CPU Tensor]
        segment_indices :List[(audio_id, segment_id)]
    """
    waveform, loaded_sample_rate = librosa.load(
        audio_file,
        sr=int(sampling_rate),
        mono=True,
        dtype=np.float32,
    )

    if int(loaded_sample_rate) != int(sampling_rate):
        raise RuntimeError(
            "librosa The sampling rate after re-sampling was abnormal:"
            f"path={audio_file}, "
            f"expected={sampling_rate}, "
            f"actual={loaded_sample_rate}"
        )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    if waveform.size == 0:
        raise ValueError(
            f"Read empty audio:{audio_file}"
        )

    if not np.all(np.isfinite(waveform)):
        raise ValueError(
            f"Audio contains NaN or Inf:{audio_file}"
        )

    if ENABLE_AUDIO_NORMALIZE:
        peak = float(np.max(np.abs(waveform)))
        if peak > 1e-8:
            waveform = waveform / peak

    step_size = int(
        segment_length * (1.0 - overlap_rate)
    )
    if step_size <= 0:
        raise ValueError(
            f"overlap_rate={overlap_rate} Caused step_size<=0"
        )

    # (b) To continue the rule on the number of subparagraphs that freeze the extraction of scripts from Whisper and former CPFT.
    total_segments = int(len(waveform) / step_size) + (
        1
        if len(waveform) % step_size
        > overlap_rate * segment_length
        else 0
    )
    total_segments = max(1, total_segments)

    segment_features: List[torch.Tensor] = []
    segment_indices: List[Tuple[int, int]] = []
    warned_double_ln_top = False

    for segment_index in range(total_segments):
        start = int(segment_index * step_size)
        end = int(start + segment_length)
        segment = waveform[start:end]

        # Consistency with freezing baselines: Priority is given to pre-wave completion when the end of the non-first segment is inadequate.
        # If it's not enough, WhisperProcessor will automatically make up zero.
        if len(segment) < segment_length and segment_index > 0:
            padding_needed = int(
                segment_length - len(segment)
            )
            previous_end = int(
                start + segment_length - step_size
            )
            padding_start = max(
                0,
                previous_end - padding_needed,
            )
            padding_values = waveform[
                padding_start:previous_end
            ]
            segment = np.concatenate(
                (segment, padding_values)
            )

        processor_output = processor(
            segment,
            sampling_rate=int(sampling_rate),
            return_tensors="pt",
        )
        input_features = (
            processor_output.input_features
            .to(DEVICE)
        )

        encoder_output = model.encoder(
            input_features=input_features,
            output_hidden_states=True,
            return_dict=True,
        )

        all_hidden_states = list(
            encoder_output.hidden_states
        )
        hidden_states = (
            all_hidden_states[1:]
            if (
                drop_input_layer
                and len(all_hidden_states) >= 2
            )
            else all_hidden_states
        )

        if len(hidden_states) == 0:
            raise RuntimeError(
                f"Whisper Not returned to hidden_states:{audio_file}"
            )

        try:
            top_already_post_norm = torch.allclose(
                hidden_states[-1],
                encoder_output.last_hidden_state,
                atol=1e-5,
                rtol=1e-5,
            )
        except Exception:
            top_already_post_norm = True

        layer_id = int(single_layer)
        if not (1 <= layer_id <= len(hidden_states)):
            raise ValueError(
                f"single_layer={layer_id} It's beyond our reach."
                f"Current available layer ={len(hidden_states)}"
            )

        is_top_layer = (
            layer_id == len(hidden_states)
            or layer_id == int(top_layer_index)
        )

        if (
            not warned_double_ln_top
            and is_top_layer
            and top_already_post_norm
            and norm_single in {"ln_post", "whisper_ln"}
        ):
            print(
                f"[WARN] {Path(audio_file).name}: "
                "Top level has post-LN, skipping second ln_post."
            )
            warned_double_ln_top = True

        hidden = hidden_states[layer_id - 1]

        skip_ln_post = (
            norm_single in {"ln_post", "whisper_ln"}
            and is_top_layer
        )

        normalized = apply_norm(
            hidden,
            mode=norm_single,
            ln_post=ln_post,
            skip_ln_post=skip_ln_post,
        )

        normalized = (
            normalized
            .squeeze(0)
            .detach()
            .cpu()
            .to(torch.float32)
        )

        segment_features.append(normalized)
        segment_indices.append(
            (
                int(audio_file_index),
                int(segment_index),
            )
        )

    return segment_features, segment_indices


def stack_with_time_padding(
    feature_list: List[torch.Tensor],
    target_t: Optional[int] = None,
    pad_value: float = 0.0,
) -> torch.Tensor:
    """
    Fills several [T,C] features to a unified T, and stops into [N,T,C].
    """
    if len(feature_list) == 0:
        raise ValueError(
            "feature_list Empty, not stack."
        )

    for index, feature in enumerate(feature_list):
        if feature.ndim != 2:
            raise ValueError(
                f"feature_list[{index}] Not 2-dimensional:"
                f"shape={tuple(feature.shape)}"
            )

    feature_dim = int(feature_list[0].shape[1])

    for index, feature in enumerate(feature_list):
        if int(feature.shape[1]) != feature_dim:
            raise ValueError(
                f"feature_list[{index}] Features inconsistent:"
                f"{feature.shape[1]} != {feature_dim}"
            )

    if target_t is None:
        target_t = max(
            int(feature.shape[0])
            for feature in feature_list
        )

    padded_features: List[torch.Tensor] = []

    for feature in feature_list:
        current_t = int(feature.shape[0])

        if current_t < int(target_t):
            padding = torch.full(
                (
                    int(target_t) - current_t,
                    feature_dim,
                ),
                float(pad_value),
                dtype=feature.dtype,
            )
            feature = torch.cat(
                (feature, padding),
                dim=0,
            )

        elif current_t > int(target_t):
            feature = feature[:int(target_t)]

        padded_features.append(feature)

    return torch.stack(
        padded_features,
        dim=0,
    )


# =============================================================================
# 7) official entries Character extraction and rigorous verification
# =============================================================================
def extract_entries_features(
    entries: List[Dict[str, Any]],
    subset_name: str,
    processor: WhisperProcessor,
    model: WhisperModel,
    single_layer: int,
    ln_post,
    top_layer_index: int,
):
    """
    Draw segment-level features for official train/test entries.

    Tags are copied directly from each audio target_label:
        HC        -> 0
        MCI       -> 1
        Dementia  -> 2

    audio_id The number shall be PROCESS-2 participant,
    Can't follow the filter after entries again enemerate.
    """
    feature_list: List[
        torch.Tensor
    ] = []
    labels_list: List[int] = []
    indices_list: List[
        Tuple[int, int]
    ] = []
    manifest: List[
        Dict[str, Any]
    ] = []

    expected_diagnosis = {
        0: "HC",
        1: "MCI",
        2: "Dementia",
    }

    for row in tqdm(
        entries,
        desc=(
            f"[{subset_name}] "
            f"single{single_layer}"
        ),
        ncols=100,
    ):
        wav_path = canonical_wav_path(
            row["wav_path"]
        )
        target_label = int(
            row["target_label"]
        )

        if target_label not in {
            0,
            1,
            2,
        }:
            raise ValueError(
                f"{subset_name} (a) The emergence of illegal three-class labels:"
                f"path={wav_path}, "
                f"label={target_label}"
            )

        diagnosis = str(
            row.get(
                "diagnosis",
                "",
            )
        )
        if diagnosis != expected_diagnosis[
            target_label
        ]:
            raise ValueError(
                f"{subset_name} diagnosis Inconsistencies with labels:"
                f"diagnosis={diagnosis}, "
                f"label={target_label}, "
                f"path={wav_path}"
            )

        if not os.path.isfile(
            wav_path
        ):
            raise FileNotFoundError(
                f"Could not close temporary folder: %s{wav_path}"
            )

        audio_id_raw = row.get(
            "official_audio_id",
            row.get(
                "audio_id",
                None,
            ),
        )
        if audio_id_raw in [
            None,
            "",
        ]:
            raise ValueError(
                "Three categories extract entry missing stable audio_id:"
                f"{row}"
            )

        audio_id = int(
            audio_id_raw
        )

        participant_id = str(
            row.get(
                "participant_id",
                "",
            )
        ).strip().lower()
        if participant_id == "":
            raise ValueError(
                "Three categories extract entry missing participant_id:"
                f"{row}"
            )

        expected_audio_id = (
            process2_audio_id(
                participant_id
            )
        )
        if audio_id != expected_audio_id:
            raise ValueError(
                "entry audio_id Inconsistent with participant_id:"
                f"participant_id={participant_id}, "
                f"audio_id={audio_id}, "
                f"expected={expected_audio_id}"
            )

        (
            segment_features,
            segment_indices,
        ) = get_segment_features(
            audio_file=wav_path,
            audio_file_index=audio_id,
            processor=processor,
            model=model,
            single_layer=int(
                single_layer
            ),
            norm_single=(
                NORM_FOR_SINGLE
            ),
            ln_post=ln_post,
            top_layer_index=int(
                top_layer_index
            ),
            drop_input_layer=(
                DROP_INPUT_LAYER
            ),
            segment_length=(
                SEGMENT_LENGTH
            ),
            sampling_rate=(
                SAMPLE_RATE
            ),
            overlap_rate=(
                OVERLAP_RATE
            ),
        )

        if len(
            segment_features
        ) == 0:
            raise ValueError(
                "No audio generated segment:"
                f"{wav_path}"
            )

        feature_list.extend(
            segment_features
        )
        indices_list.extend(
            segment_indices
        )
        labels_list.extend(
            [
                int(
                    target_label
                )
            ]
            * len(
                segment_features
            )
        )

        kept = copy.deepcopy(
            row
        )
        kept[
            "audio_index_in_subset"
        ] = int(audio_id)
        kept[
            "official_audio_id"
        ] = int(audio_id)
        kept[
            "audio_id"
        ] = int(audio_id)
        kept[
            "participant_id"
        ] = participant_id
        kept[
            "diagnosis"
        ] = diagnosis
        kept[
            "target_label"
        ] = int(target_label)
        kept[
            "num_segments"
        ] = int(
            len(
                segment_features
            )
        )

        manifest.append(
            kept
        )

    return (
        feature_list,
        labels_list,
        indices_list,
        manifest,
    )


def to_index_array(
    indices: Sequence[Tuple[int, int]],
    split_name: str,
) -> np.ndarray:
    """Converts the segment index to a array of [N,2] int64."""
    index_array = np.asarray(
        indices,
        dtype=np.int64,
    )

    if index_array.size == 0:
        raise ValueError(
            f"{split_name} The indices are empty."
        )

    return index_array.reshape(-1, 2)


def count_audio_classes(
    labels: torch.Tensor,
    indices: np.ndarray,
    split_name: str,
) -> Dict[int, int]:
    """
    Number of audio-level classifications,
    and check all segment labels for the same audio_id.
    """
    labels_np = (
        labels
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.int64
        )
        .reshape(-1)
    )
    indices_np = np.asarray(
        indices,
        dtype=np.int64,
    ).reshape(
        -1,
        2,
    )

    counts = {
        0: 0,
        1: 0,
        2: 0,
    }

    for audio_id in sorted(
        np.unique(
            indices_np[
                :,
                0,
            ]
        ).tolist()
    ):
        mask = (
            indices_np[
                :,
                0,
            ]
            == int(
                audio_id
            )
        )

        unique_labels = np.unique(
            labels_np[
                mask
            ]
        )

        if len(
            unique_labels
        ) != 1:
            raise ValueError(
                f"{split_name} Middle audio_id={audio_id} "
                "The segment label is inconsistent:"
                f"{unique_labels.tolist()}"
            )

        class_id = int(
            unique_labels[0]
        )
        if class_id not in counts:
            raise ValueError(
                f"{split_name} Illegal categories appear in this category:"
                f"{class_id}"
            )

        counts[
            class_id
        ] += 1

    return counts


def validate_three_class_feature_bundle(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    split_name: str,
) -> Dict[str, int]:
    """
    Strictly authenticate the three categories before saving feature / label / index.

    Request:
        features : [N,T,C] torch.float32
        labels   : [N] torch.long,And includes 0/1/2
        indices  : [N,2] np.int64
    """
    if features.ndim != 3:
        raise ValueError(
            f"{split_name} feature It is necessary to be [N,T,C],"
            f"Current ={tuple(features.shape)}"
        )

    if features.dtype != torch.float32:
        raise TypeError(
            f"{split_name} features It is necessary to be torch.float32,"
            f"Current ={features.dtype}"
        )

    labels = (
        labels
        .detach()
        .cpu()
        .reshape(-1)
    )
    if labels.dtype != torch.long:
        raise TypeError(
            f"{split_name} labels It is necessary to be torch.long,"
            f"Current ={labels.dtype}"
        )

    indices = np.asarray(
        indices,
        dtype=np.int64,
    ).reshape(
        -1,
        2,
    )
    if indices.dtype != np.int64:
        raise TypeError(
            f"{split_name} indices It is necessary to be np.int64,"
            f"Current ={indices.dtype}"
        )

    if not (
        len(features)
        == len(labels)
        == len(indices)
    ):
        raise ValueError(
            f"{split_name} feature/label/index Numbers are inconsistent:"
            f"{len(features)}, "
            f"{len(labels)}, "
            f"{len(indices)}"
        )

    unique_labels = sorted(
        int(value)
        for value in (
            torch.unique(
                labels
            )
            .tolist()
        )
    )
    if unique_labels != [
        0,
        1,
        2,
    ]:
        raise ValueError(
            f"{split_name} It is important to have a zero-one-two component at the same time."
            f"Current ={unique_labels}"
        )

    segment_counts = {
        0:
            int(
                torch.sum(
                    labels == 0
                ).item()
            ),
        1:
            int(
                torch.sum(
                    labels == 1
                ).item()
            ),
        2:
            int(
                torch.sum(
                    labels == 2
                ).item()
            ),
    }

    audio_counts = (
        count_audio_classes(
            labels=labels,
            indices=indices,
            split_name=(
                split_name
            ),
        )
    )

    result = {
        "segment_total":
            int(
                len(
                    labels
                )
            ),

        "segment_class0":
            int(
                segment_counts[0]
            ),
        "segment_class1":
            int(
                segment_counts[1]
            ),
        "segment_class2":
            int(
                segment_counts[2]
            ),

        "segment_hc":
            int(
                segment_counts[0]
            ),
        "segment_mci":
            int(
                segment_counts[1]
            ),
        "segment_dementia":
            int(
                segment_counts[2]
            ),

        "audio_total":
            int(
                sum(
                    audio_counts.values()
                )
            ),

        "audio_class0":
            int(
                audio_counts[0]
            ),
        "audio_class1":
            int(
                audio_counts[1]
            ),
        "audio_class2":
            int(
                audio_counts[2]
            ),

        "audio_hc":
            int(
                audio_counts[0]
            ),
        "audio_mci":
            int(
                audio_counts[1]
            ),
        "audio_dementia":
            int(
                audio_counts[2]
            ),
    }

    print(
        f"[LABEL CHECK] {split_name} segment | "
        f"HC={result['segment_hc']} | "
        f"MCI={result['segment_mci']} | "
        f"Dementia={result['segment_dementia']} | "
        f"total={result['segment_total']}"
    )
    print(
        f"[LABEL CHECK] {split_name} audio | "
        f"HC={result['audio_hc']} | "
        f"MCI={result['audio_mci']} | "
        f"Dementia={result['audio_dementia']} | "
        f"total={result['audio_total']}"
    )

    return result


# =============================================================================
# 8) Save PPROCES-2 Class III features
# =============================================================================
def save_feature_bundle(
    save_root: str,
    single_layer: int,
    mask_offset: int,
    stage3_ckpt_path: str,
    checkpoint_load_report: Dict[str, Any],
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
    train_distribution: Dict[str, int],
    test_distribution: Dict[str, int],
) -> Dict[str, str]:
    """
    Saves as LRU_3class.py directly readable official train/test
    Six document formats.

    Consistent with the classification of frozen in the six main documents basename,
    Not changed by CPFT or by category III.
    """
    ensure_dir(
        save_root
    )

    feature_model_tag = (
        get_feature_model_tag(
            mask_offset
        )
    )
    feature_selection_tag = (
        get_feature_selection_tag(
            single_layer
        )
    )

    layer_name = build_layer_name(
        single_layer=int(
            single_layer
        ),
        unfreeze_layer_count=int(
            UNFREEZE_ENCODER_LAYERS
        ),
        path_param_tags=(
            PATH_PARAM_TAGS
        ),
    )

    stage3_subdir = (
        build_stage3_subdir(
            mask_offset
        )
    )

    file_tag = (
        f"{FEATURE_PREFIX}_"
        f"single{int(single_layer)}_"
        f"{OV_STR}_{SEC_STR}"
    )

    output_paths = {
        "feature_train":
            os.path.join(
                save_root,
                f"feature_{file_tag}_train.pt",
            ),
        "feature_test":
            os.path.join(
                save_root,
                f"feature_{file_tag}_test.pt",
            ),
        "label_train":
            os.path.join(
                save_root,
                f"labels1_{file_tag}_train.pt",
            ),
        "label_test":
            os.path.join(
                save_root,
                f"labels2_{file_tag}_test.pt",
            ),
        "index_train":
            os.path.join(
                save_root,
                f"indices_{file_tag}_train.pt",
            ),
        "index_test":
            os.path.join(
                save_root,
                f"indices_{file_tag}_test.pt",
            ),
    }

    torch.save(
        train_features.to(
            torch.float32
        ),
        output_paths[
            "feature_train"
        ],
    )
    torch.save(
        test_features.to(
            torch.float32
        ),
        output_paths[
            "feature_test"
        ],
    )

    torch.save(
        train_labels.to(
            torch.long
        ),
        output_paths[
            "label_train"
        ],
    )
    torch.save(
        test_labels.to(
            torch.long
        ),
        output_paths[
            "label_test"
        ],
    )

    torch.save(
        np.asarray(
            train_indices,
            dtype=np.int64,
        ),
        output_paths[
            "index_train"
        ],
    )
    torch.save(
        np.asarray(
            test_indices,
            dtype=np.int64,
        ),
        output_paths[
            "index_test"
        ],
    )

    if SAVE_META:
        meta = {
            "task":
                "process2_hc_mci_dementia_3class",

            "class_mapping": {
                "0":
                    CLASS_NAMES[0],
                "1":
                    CLASS_NAMES[1],
                "2":
                    CLASS_NAMES[2],
            },

            "label_source":
                "PROCESS-2/meta-info.csv diagnosis",
            "split_source":
                "PROCESS-2/meta-info.csv Split",
            "process2_meta_path":
                str(
                    PROCESS2_META_PATH
                ),
            "mmse_used":
                False,
            "audio_task":
                "CTD",

            "dataset_option":
                str(
                    DATASET_OPTION
                ),
            "run_protocol":
                str(
                    RUN_PROTOCOL_TAG
                ),

            "official_train_used_in_cpft_training":
                True,
            "official_test_used_in_cpft_training":
                False,
            "official_test_usage":
                "final_feature_export_only",

            "single_layer":
                int(
                    single_layer
                ),
            "unfreeze_encoder_layers":
                int(
                    UNFREEZE_ENCODER_LAYERS
                ),
            "layer_name":
                str(
                    layer_name
                ),

            "mask_offset":
                int(
                    mask_offset
                ),
            "stage3_subdir":
                str(
                    stage3_subdir
                ),

            "feature_model_tag":
                str(
                    feature_model_tag
                ),
            "feature_selection_tag":
                str(
                    feature_selection_tag
                ),
            "feature_file_tag":
                str(
                    file_tag
                ),

            "stage3_ckpt_path":
                str(
                    stage3_ckpt_path
                ),
            "checkpoint_load_report":
                checkpoint_load_report,
            "whisper_base_model_name":
                str(
                    WHISPER_BASE_MODEL_NAME
                ),

            "sampling_rate":
                int(
                    SAMPLE_RATE
                ),
            "segment_sec":
                int(
                    SEGMENT_SEC
                ),
            "overlap_rate":
                float(
                    OVERLAP_RATE
                ),

            "norm_for_single":
                (
                    None
                    if NORM_FOR_SINGLE
                    is None
                    else str(
                        NORM_FOR_SINGLE
                    )
                ),
            "drop_input_layer":
                bool(
                    DROP_INPUT_LAYER
                ),
            "audio_normalize":
                bool(
                    ENABLE_AUDIO_NORMALIZE
                ),

            "feature_dtype":
                "torch.float32",
            "label_dtype":
                "torch.int64",
            "index_dtype":
                "numpy.int64",
            "index_format":
                (
                    "[N,2], "
                    "column0=audio_id, "
                    "column1=segment_id"
                ),

            "feature_train_semantics":
                (
                    "official_train_all_"
                    "process2_ctd_3class"
                ),
            "feature_test_semantics":
                (
                    "official_test_all_"
                    "process2_ctd_3class"
                ),

            "audio_id_policy":
                (
                    "process2_participant_number_"
                    "shared_by_frozen_and_cpft"
                ),

            "train_feature_shape":
                list(
                    train_features.shape
                ),
            "test_feature_shape":
                list(
                    test_features.shape
                ),

            "train_distribution":
                train_distribution,
            "test_distribution":
                test_distribution,

            "saved_files":
                output_paths,

            "downstream_recommended_config": {
                "SCRIPT":
                    "LRU_3class.py",
                "FEATURE_MODEL_NAME":
                    str(
                        feature_model_tag
                    ),
                "SINGLE_LAYERS": [
                    int(
                        single_layer
                    )
                ],
                "FEATURE_SELECTION":
                    str(
                        feature_selection_tag
                    ),
                "FEATURE_FILE_TAG":
                    str(
                        file_tag
                    ),
            },

            "cpft_method":
                str(
                    CPFT_METHOD_TAG
                ),

            "path_param_tags":
                list(
                    PATH_PARAM_TAGS
                ),
            "path_param_suffix":
                build_path_param_suffix(
                    PATH_PARAM_TAGS
                ),
            "stage1_source_path_param_tags":
                list(
                    STAGE1_SOURCE_PATH_PARAM_TAGS
                ),

            "cpft_ks": [
                int(value)
                for value in (
                    CPFT_KS
                )
            ],
            "cpft_expected_num_clusters": [
                int(value)
                for value in (
                    CPFT_EXPECTED_NUM_CLUSTERS
                )
            ],

            "cpft_stage3_lambda_clus":
                float(
                    CPFT_STAGE3_LAMBDA_CLUS
                ),
            "cpft_stage3_lambda_diag":
                float(
                    CPFT_STAGE3_LAMBDA_DIAG
                ),

            "stage3_mask_inject_after_layers": [
                int(value)
                for value in (
                    STAGE3_MASK_INJECT_AFTER_LAYERS
                )
            ],
        }

        save_json(
            meta,
            os.path.join(
                save_root,
                "feature_meta.json",
            ),
        )

        save_json(
            {
                "task":
                    "process2_hc_mci_dementia_3class",
                "split":
                    "train",
                "class_mapping": {
                    "0":
                        CLASS_NAMES[0],
                    "1":
                        CLASS_NAMES[1],
                    "2":
                        CLASS_NAMES[2],
                },
                "entries":
                    train_manifest,
            },
            os.path.join(
                save_root,
                "train_manifest.json",
            ),
        )

        save_json(
            {
                "task":
                    "process2_hc_mci_dementia_3class",
                "split":
                    "test",
                "class_mapping": {
                    "0":
                        CLASS_NAMES[0],
                    "1":
                        CLASS_NAMES[1],
                    "2":
                        CLASS_NAMES[2],
                },
                "entries":
                    test_manifest,
            },
            os.path.join(
                save_root,
                "test_manifest.json",
            ),
        )

        save_text(
            str(
                stage3_ckpt_path
            ),
            os.path.join(
                save_root,
                "stage3_ckpt_path.txt",
            ),
        )

        save_text(
            (
                "PROCESS-2 HC/MCI/Dementia 3-class "
                "CPFT feature export.\n"
                "Labels: HC=0, MCI=1, Dementia=2.\n"
                "Label source: meta-info.csv diagnosis.\n"
                "Split source: meta-info.csv Split.\n"
                "Stage1/Stage2/Stage3 use official TRAIN only.\n"
                "Official TEST is loaded only here for final feature export.\n"
                "MMSE is not used and missing MMSE never filters samples.\n"
                "Audio task: CTD only.\n"
                "audio_id uses PROCESS-2 participant number and is shared "
                "with the frozen three-class baseline.\n"
            ),
            os.path.join(
                save_root,
                "protocol_note.txt",
            ),
        )

    print(
        f"[SAVE] root={save_root}"
    )
    print(
        "  semantics: "
        "train=official_train_PROCESS2_CTD_3class, "
        "test=official_test_PROCESS2_CTD_3class"
    )
    for name, path in (
        output_paths.items()
    ):
        print(
            f"  - {name}: {path}"
        )

    return output_paths


# =============================================================================
# 9) Export Tasks Single dataset / layer / unfreeze / maskoff
# =============================================================================
def run_one_configuration(
    single_layer: int,
    mask_offset: int,
) -> Dict[str, Any]:
    """
    Full export process for the individual PROCESS-2 three classification configuration.

    1. Read official train/test;
    2. Generate HC/MCI/Dementia 0/1/2 labels with meta-info.csv;
    3. Use the number participant as stable audio_id;
    4. Find last.ckpt for CPFT_3class_stage3.py;
    5. Load only Whisper parameters in Stage3 back to HuggingFace WhisperModel;
    6. Extracting official train/test segment features;
    7. Saves six documents that are directly readable at LRU_3class.py bundle.
    """
    single_layer = int(
        single_layer
    )
    mask_offset = int(
        mask_offset
    )

    save_root = get_save_root(
        DATASET_OPTION,
        single_layer,
        mask_offset,
    )

    layer_name = build_layer_name(
        single_layer=(
            single_layer
        ),
        unfreeze_layer_count=int(
            UNFREEZE_ENCODER_LAYERS
        ),
        path_param_tags=(
            PATH_PARAM_TAGS
        ),
    )

    stage3_run_root = (
        build_run_root(
            dataset_option=(
                DATASET_OPTION
            ),
            layer_id=(
                single_layer
            ),
        )
    )

    stage3_dir = os.path.join(
        stage3_run_root,
        build_stage3_subdir(
            mask_offset
        ),
    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "[3CLASS EXTRACT]"
    )
    print(
        "  DATASET_OPTION =",
        DATASET_OPTION,
    )
    print(
        "  LABELS         = "
        "HC:0 | MCI:1 | Dementia:2"
    )
    print(
        "  LABEL_SOURCE   =",
        PROCESS2_META_PATH,
    )
    print(
        "  RUN_PROTOCOL   =",
        RUN_PROTOCOL_TAG,
    )
    print(
        "  single_layer   =",
        single_layer,
    )
    print(
        "  unfreeze       =",
        UNFREEZE_ENCODER_LAYERS,
    )
    print(
        "  layer_name     =",
        layer_name,
    )
    print(
        "  mask_offset    =",
        mask_offset,
    )
    print(
        "  stage3_dir     =",
        stage3_dir,
    )
    print(
        "  save_root      =",
        save_root,
    )
    print(
        "  extract target = "
        "official train + official test"
    )
    print(
        "=" * 100
    )

    (
        train_entries,
        test_entries,
    ) = (
        load_official_train_test_entries(
            DATASET_OPTION
        )
    )

    train_audio_distribution = (
        summarize_audio_entries(
            train_entries,
            "official_train",
        )
    )
    test_audio_distribution = (
        summarize_audio_entries(
            test_entries,
            "official_test",
        )
    )

    stage3_ckpt_path = (
        find_stage3_checkpoint(
            dataset_option=(
                DATASET_OPTION
            ),
            layer_id=(
                single_layer
            ),
            mask_offset=(
                mask_offset
            ),
            override_ckpt=(
                STAGE3_CKPT_OVERRIDE
            ),
        )
    )

    print(
        "[CKPT] Stage3 checkpoint = "
        f"{stage3_ckpt_path}"
    )

    processor = (
        WhisperProcessor
        .from_pretrained(
            WHISPER_BASE_MODEL_NAME,
            local_files_only=(
                LOCAL_FILES_ONLY
            ),
        )
    )
    model = (
        WhisperModel
        .from_pretrained(
            WHISPER_BASE_MODEL_NAME,
            local_files_only=(
                LOCAL_FILES_ONLY
            ),
        )
    )

    checkpoint_load_report = (
        load_stage3_ckpt_into_whisper_model(
            model,
            stage3_ckpt_path,
        )
    )

    model.eval()
    model.to(
        DEVICE
    )

    ln_post = (
        model.encoder.layer_norm
    )
    top_layer_index = int(
        model.config.encoder_layers
    )

    train_result = (
        extract_entries_features(
            entries=(
                train_entries
            ),
            subset_name=(
                "official_train"
            ),
            processor=(
                processor
            ),
            model=model,
            single_layer=(
                single_layer
            ),
            ln_post=(
                ln_post
            ),
            top_layer_index=(
                top_layer_index
            ),
        )
    )
    (
        train_feature_list,
        train_label_list,
        train_index_list,
        train_manifest,
    ) = train_result

    test_result = (
        extract_entries_features(
            entries=(
                test_entries
            ),
            subset_name=(
                "official_test"
            ),
            processor=(
                processor
            ),
            model=model,
            single_layer=(
                single_layer
            ),
            ln_post=(
                ln_post
            ),
            top_layer_index=(
                top_layer_index
            ),
        )
    )
    (
        test_feature_list,
        test_label_list,
        test_index_list,
        test_manifest,
    ) = test_result

    global_time_steps = max(
        max(
            int(
                feature.shape[0]
            )
            for feature
            in train_feature_list
        ),
        max(
            int(
                feature.shape[0]
            )
            for feature
            in test_feature_list
        ),
    )

    print(
        "[PAD] global_time_steps="
        f"{global_time_steps}"
    )

    train_features = (
        stack_with_time_padding(
            train_feature_list,
            target_t=(
                global_time_steps
            ),
        )
        .to(
            torch.float32
        )
    )
    test_features = (
        stack_with_time_padding(
            test_feature_list,
            target_t=(
                global_time_steps
            ),
        )
        .to(
            torch.float32
        )
    )

    train_labels = torch.tensor(
        train_label_list,
        dtype=torch.long,
    )
    test_labels = torch.tensor(
        test_label_list,
        dtype=torch.long,
    )

    train_indices = (
        to_index_array(
            train_index_list,
            "official_train",
        )
    )
    test_indices = (
        to_index_array(
            test_index_list,
            "official_test",
        )
    )

    train_distribution = (
        validate_three_class_feature_bundle(
            features=(
                train_features
            ),
            labels=(
                train_labels
            ),
            indices=(
                train_indices
            ),
            split_name=(
                "official_train"
            ),
        )
    )
    test_distribution = (
        validate_three_class_feature_bundle(
            features=(
                test_features
            ),
            labels=(
                test_labels
            ),
            indices=(
                test_indices
            ),
            split_name=(
                "official_test"
            ),
        )
    )

    # Check again:
    # Original audio enterry and final session bindle
    # HC/MCI/Dementia The numbers must be identical.
    for (
        split_name,
        original_dist,
        feature_dist,
    ) in [
        (
            "official_train",
            train_audio_distribution,
            train_distribution,
        ),
        (
            "official_test",
            test_audio_distribution,
            test_distribution,
        ),
    ]:
        for key in [
            "audio_total",
            "audio_class0",
            "audio_class1",
            "audio_class2",
        ]:
            if int(
                original_dist[
                    key
                ]
            ) != int(
                feature_dist[
                    key
                ]
            ):
                raise ValueError(
                    f"{split_name} Original audio and export characteristics "
                    "audio-level Discrepancies in category statistics: \n"
                    f"key={key}\n"
                    f"original={original_dist}\n"
                    f"feature={feature_dist}"
                )

    print(
        "[SHAPE] train_features="
        f"{tuple(train_features.shape)}"
    )
    print(
        "[SHAPE] test_features="
        f"{tuple(test_features.shape)}"
    )
    print(
        "[SHAPE] train_labels="
        f"{tuple(train_labels.shape)}"
    )
    print(
        "[SHAPE] test_labels="
        f"{tuple(test_labels.shape)}"
    )
    print(
        "[SHAPE] train_indices="
        f"{tuple(train_indices.shape)}"
    )
    print(
        "[SHAPE] test_indices="
        f"{tuple(test_indices.shape)}"
    )

    output_paths = (
        save_feature_bundle(
            save_root=(
                save_root
            ),
            single_layer=(
                single_layer
            ),
            mask_offset=(
                mask_offset
            ),
            stage3_ckpt_path=(
                stage3_ckpt_path
            ),
            checkpoint_load_report=(
                checkpoint_load_report
            ),
            train_features=(
                train_features
            ),
            test_features=(
                test_features
            ),
            train_labels=(
                train_labels
            ),
            test_labels=(
                test_labels
            ),
            train_indices=(
                train_indices
            ),
            test_indices=(
                test_indices
            ),
            train_manifest=(
                train_manifest
            ),
            test_manifest=(
                test_manifest
            ),
            train_distribution=(
                train_distribution
            ),
            test_distribution=(
                test_distribution
            ),
        )
    )

    result = {
        "dataset":
            str(
                DATASET_OPTION
            ),
        "single_layer":
            int(
                single_layer
            ),
        "unfreeze_encoder_layers":
            int(
                UNFREEZE_ENCODER_LAYERS
            ),
        "mask_offset":
            int(
                mask_offset
            ),
        "feature_model_name":
            str(
                get_feature_model_tag(
                    mask_offset
                )
            ),
        "save_root":
            str(
                save_root
            ),
        "stage3_ckpt_path":
            str(
                stage3_ckpt_path
            ),
        "output_paths":
            output_paths,
    }

    del (
        processor,
        model,
        train_feature_list,
        test_feature_list,
        train_features,
        test_features,
        train_labels,
        test_labels,
        train_indices,
        test_indices,
    )

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        f"[DONE] {DATASET_OPTION} | "
        f"single{single_layer} | "
        f"unfree{UNFREEZE_ENCODER_LAYERS} | "
        f"maskoff{mask_offset} | "
        f"feature_model="
        f"{get_feature_model_tag(mask_offset)}"
    )

    return result


# =============================================================================
# 10) Main entrance
# =============================================================================
def validate_global_configuration() -> None:
    """
    Checks cross- script parameters and directory coverage risks for PROCESS-2 classification extract before startup.
    """
    if DATASET_OPTIONS != [
        "PROCESS-2"
    ]:
        raise ValueError(
            "CPFT_3class_extract.py Allow PROCESS-2 only."
            f"Current DATASET_OPTIONS={DATASET_OPTIONS}"
        )

    if len(
        LAYERS_TO_RUN
    ) == 0:
        raise ValueError(
            "LAYERS_TO_RUN It's empty."
        )

    if len(
        UNFREEZE_ENCODER_LAYERS_LIST
    ) == 0:
        raise ValueError(
            "UNFREEZE_ENCODER_LAYERS_LIST It's empty."
        )

    if len(
        STAGE3_MASK_INJECT_AFTER_LAYERS
    ) == 0:
        raise ValueError(
            "STAGE3_MASK_INJECT_AFTER_LAYERS It's empty."
        )

    if (
        len(
            STAGE3_MASK_INJECT_AFTER_LAYERS
        ) > 1
        and not (
            INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG
        )
    ):
        raise ValueError(
            "The current configuration will export more than one mask_offset at the same time."
            "But the feature model catalogue doesn't contain maskoff,"
            ". \n"
            "Please. "
            "INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG=True,"
            "Or just one STAGE3_MASK_INJECT_AFTER_LAYERS."
        )

    if [
        int(value)
        for value in CPFT_KS
    ] != [
        int(value)
        for value in (
            CPFT_EXPECTED_NUM_CLUSTERS
        )
    ]:
        raise ValueError(
            "CPFT_KS and "
            "CPFT_EXPECTED_NUM_CLUSTERS Inconsistent:"
            f"{CPFT_KS} != "
            f"{CPFT_EXPECTED_NUM_CLUSTERS}"
        )

    if MODEL_ROOT_TAG != (
        "CPFT_lr5e-5_epo5_3class"
    ):
        raise ValueError(
            "MODEL_ROOT_TAG Classification with III "
            "Stage1/Stage2/Stage3 Inconsistent:"
            f"{MODEL_ROOT_TAG}"
        )

    if NUM_CLASSES != 3:
        raise ValueError(
            f"NUM_CLASSES Must be 3, current ={NUM_CLASSES}"
        )

    if not os.path.isfile(
        PROCESS2_META_PATH
    ):
        raise FileNotFoundError(
            "PROCESS2_META_PATH Cannot initialise Evolution's mail component."
            f"{PROCESS2_META_PATH}"
        )

    if str(
        CPFT_METHOD_TAG
    ) != "diagnosis_aware_cpft":
        raise ValueError(
            "CPFT_METHOD_TAG The sentence should read "
            "diagnosis_aware_cpft,"
            f"Current ={CPFT_METHOD_TAG}"
        )

    expected_feature_model = (
        "ft_CPFT_lr5e-5_epo5_3class"
    )
    if not (
        INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG
    ):
        actual_feature_model = (
            get_feature_model_tag(
                STAGE3_MASK_INJECT_AFTER_LAYERS[
                    0
                ]
            )
        )
        if (
            actual_feature_model
            != expected_feature_model
        ):
            raise ValueError(
                "Final FEATURE_MODEL_NAME with "
                "LRU_3class.py Inconsistent agreement:"
                f"actual={actual_feature_model}, "
                f"expected={expected_feature_model}"
            )


def main() -> None:
    """
    Through PROCESS-2 the number of unfrozen layers, expression layers and mask_offset,
    Export features for the three categories Stage3 checkpoint that have been trained for each group.
    """
    global DATASET_OPTION
    global UNFREEZE_ENCODER_LAYERS

    validate_global_configuration()

    print(
        "\n"
        + "=" * 100
    )
    print(
        "[3CLASS EXTRACT CONFIG]"
    )
    print(
        "  TASK               = "
        "PROCESS-2 HC/MCI/Dementia 3-class"
    )
    print(
        "  CLASS_PROJECT_ROOT =",
        CLASS_PROJECT_ROOT,
    )
    print(
        "  CPFT_ROOT         =",
        CPFT_ROOT,
    )
    print(
        "  FEATURE_ROOT       =",
        FEATURE_ROOT,
    )
    print(
        "  PROCESS2_META_PATH =",
        PROCESS2_META_PATH,
    )
    print(
        "  DATASET_OPTIONS    =",
        DATASET_OPTIONS,
    )
    print(
        "  CLASS_NAMES        =",
        CLASS_NAMES,
    )
    print(
        "  MODEL_ROOT_TAG     =",
        MODEL_ROOT_TAG,
    )
    print(
        "  CPFT_METHOD_TAG   =",
        CPFT_METHOD_TAG,
    )
    print(
        "  UNFREEZE_ENCODER_LAYERS_LIST =",
        UNFREEZE_ENCODER_LAYERS_LIST,
    )
    print(
        "  LAYERS_TO_RUN      =",
        LAYERS_TO_RUN,
    )
    print(
        "  STAGE3_MASK_INJECT_AFTER_LAYERS =",
        STAGE3_MASK_INJECT_AFTER_LAYERS,
    )
    print(
        "  WHISPER_BASE_MODEL_NAME =",
        WHISPER_BASE_MODEL_NAME,
    )
    print(
        "  SEGMENT_SEC        =",
        SEGMENT_SEC,
    )
    print(
        "  OVERLAP_RATE       =",
        OVERLAP_RATE,
    )
    print(
        "  NORM_FOR_SINGLE    =",
        NORM_FOR_SINGLE,
    )
    print(
        "  DEVICE             =",
        DEVICE,
    )
    print(
        "  downstream FEATURE_MODEL_NAME =",
        get_feature_model_tag(
            STAGE3_MASK_INJECT_AFTER_LAYERS[
                0
            ]
        ),
    )
    print(
        "=" * 100
    )

    completed: List[
        Dict[str, Any]
    ] = []

    for dataset_option in [
        str(value)
        for value in (
            DATASET_OPTIONS
        )
    ]:
        for unfreeze_count in [
            int(value)
            for value in (
                UNFREEZE_ENCODER_LAYERS_LIST
            )
        ]:
            DATASET_OPTION = str(
                dataset_option
            )
            UNFREEZE_ENCODER_LAYERS = int(
                unfreeze_count
            )

            for layer_id in [
                int(value)
                for value in (
                    LAYERS_TO_RUN
                )
            ]:
                for mask_offset in [
                    int(value)
                    for value in (
                        STAGE3_MASK_INJECT_AFTER_LAYERS
                    )
                ]:
                    completed.append(
                        run_one_configuration(
                            single_layer=(
                                layer_id
                            ),
                            mask_offset=(
                                mask_offset
                            ),
                        )
                    )

    print(
        "\n"
        + "=" * 100
    )
    print(
        "[ALL DONE]"
    )

    for item in completed:
        print(
            f"  dataset={item['dataset']} | "
            f"single={item['single_layer']} | "
            f"unfree={item['unfreeze_encoder_layers']} | "
            f"maskoff={item['mask_offset']} | "
            f"save_root={item['save_root']}"
        )

    print(
        "\nWhen the same LRU_3class.py reads the CPFT III classification,"
        "Just set up:"
    )

    for mask_offset in [
        int(value)
        for value in (
            STAGE3_MASK_INJECT_AFTER_LAYERS
        )
    ]:
        print(
            "  FEATURE_MODEL_NAME = "
            f'"{get_feature_model_tag(mask_offset)}"'
        )

    print(
        "=" * 100
    )


if __name__ == "__main__":
    main()
