# -*- coding: utf-8 -*-
"""
CPFT_3class_stage1.py

Cluster Pseudo-Label Fine-Tuning (CPFT).
============================================================

PROCESS-2 HC / MCI / Dementia Third classification CPFT Stage1.

Definition of tasks
--------
    0 = HC
    1 = MCI
    2 = Dementia

Unique Label Source
------------
    /home/user/Downloads/PROCESS-2/meta-info.csv

Use fields:
    IDs
    diagnosis
    Split

MMSE / age / gender Do not participate in this subcategory Stage1.

Data protocol
--------
- Run PROCESS-2 only;
- Read only *__CTD.wav;
- Stage1 Use official TRAIN only;
- official TEST Do not enter Stage1;
- cc/cd The directory is only used for consistency verification and no longer directly generates classification labels;
- stable audio_id Use PROCESS-2 participant numbers.

Stage1 Two-headed target.
---------------
1. utterance head:
   For the entire paragraph after attentive statistics pooling
   HC / MCI / Dementia Three categories.

2. frame head:
   Copy the current audio diagnosis tab to all valid Whisper token,
   As a sub-category three target for weak oversight.

Total loss:
    total_loss
        = LAMBDA_UTT * utterance_cross_entropy
        + LAMBDA_FRAME * frame_cross_entropy

training protocol
--------
- official TRAIN Full training;
- Do not use official TEST;
- Do not do validation;
- Do not do early stopping;
- Fixed STAGE1_MAX_EPOCHS=10;
- Only at the end of the training last.ckpt;
- class weight is not used for the first version of the Third Classification baseline.

Save Path
--------
/home/user/Alzheimer_code_jax_class/
└── finetune/ft_CPFT/
    └── PROCESS-2/
        └── CPFT_lr5e-5_epo5_3class/
            └── singleX_unfreeZ/
                ├── stage1_class/
                │   ├── last.ckpt
                │   └── train_manifest.json
                └── meta/

Run Example
--------
conda activate whisper_cpft_faiss
python CPFT_3class_stage1.py

Cluster Pseudo-Label Fine-Tuning (CPFT).
"""


try:
    from .paper_names import paper_dataset_name
except ImportError:
    from paper_names import paper_dataset_name

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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import librosa

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel


# ============================================================
# 2) Parameter Settings: Change this priority
# ============================================================

# Current PROCOESS-2 Category III Project Root Directory.
CLASS_PROJECT_ROOT = "/home/user/Alzheimer_code_jax_class"

# Upstream CPFT scripts with a root directory of intermediate results.
PROJECT_ROOT = os.path.join(CLASS_PROJECT_ROOT, "finetune")
CPFT_ROOT = os.path.join(PROJECT_ROOT, "ft_CPFT_3class")

# Only PROCESS-2 is running the current three classification experiments.
DATASET_OPTIONS = [
    "PROCESS-2",
]

# Keeps a single value variable, main() is dynamically overwrite in the batch cycle.
DATASET_OPTION = DATASET_OPTIONS[0]

# Currently priority is given to the 12 layer.
LAYERS_TO_RUN = [12]
STAGE1_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE3_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE4_OUTPUT_LAYERS_TO_RUN = LAYERS_TO_RUN

# How many layers are unfrozen continuously relative to the current level, including the layer itself.
# single12 + unfree7 -> Unfreezing the encoder 6-12 layer.
UNFREEZE_ENCODER_LAYERS_LIST = [2,3,4,5,6,7,8,9,10,11,12]
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# official_train only Protocol label, enter meta only, without adding an additional level of path.
RUN_PROTOCOL_TAG = "official_train_only"

# Stage1/2/3/extract There must be a uniform lab root label.
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5_3class"

# Stage1 Do not rely on KMeans parameters in itself, default does not add additional path suffixes.
STAGE1_SOURCE_PATH_PARAM_TAGS: List[str] = []
PATH_PARAM_TAGS: List[str] = []

# Whisper Local model.
WHISPER_MODEL_NAME = "/home/user/whisper_lru_3/whisper_small"

# means layer integration: mean is usually used for single layers.
STAGE1_REPR_POOL = "mean"  # "mean" / "scalar_mix"

# Each audio in Stage1 forms a 30second view:
# Long audio centre is cropped and short audio is repeated.
TARGET_SEC = 30.0

# PROCESS-2 HC/MCI/Dementia Definition of the three categories.
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

# Double head loss weight.
LAMBDA_UTT = 1.0
LAMBDA_FRAME = 0.5

# Heading structure of the classification.
HEAD_HIDDEN = 128
HEAD_DROPOUT = 0.30
LABEL_SMOOTHING = 0.05

# Optimizing relevance: Retain the CPFT Stage1 existing AdamW group learning rate agreement.
LR_HEAD = 1e-4
LR_WHISPER = 1e-5
WEIGHT_DECAY = 5e-2
WARMUP_RATIO = 0.10
FLAT_RATIO = 0.20
MIN_LR_RATIO = 0.05

# Training control.
BATCH_SIZE = 8
NUM_WORKERS = 4
STAGE1_MAX_EPOCHS = 10
TRAINING_STEP = 2000  # It is updated at the true DataLoader step in each experiment.
PRECISION = 16        # 16 / 32
ACCUMULATE_GRAD_BATCHES = 1
GRAD_CLIP = 1.0

# Other.
SEED = 42
ENABLE_CSV_LOGGER = True
RESUME_CHECKPOINT = None
SAVE_MANIFESTS = True


# ============================================================
# 3) Constants and data set paths
# ============================================================
os.environ.pop("PL_TRAINER_AMP_BACKEND", None)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000

# PROCESS-2 official Contents.
DATASETS: Dict[str, Dict[str, str]] = {
    "PROCESS-2": {
        "train_cc": "/home/user/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/user/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/user/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/user/Downloads/PROCESS-2/test/cd",
    },
}


# ============================================================
# 4) Random Feeds and Common Tools
# ============================================================
def seed_everything_for_repro(seed: int) -> None:
    """Standardize the setting of random seeds to ensure, to the extent possible, that experiments are re-emerging."""
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
    """Can not create folder lock on %s: %s"""
    os.makedirs(path, exist_ok=True)


def natural_key(x: str):
    """Sort naturally: let S002 be placed before S010, and single2 before single10."""
    s = Path(str(x)).name
    return [
        int(t) if t.isdigit() else t.lower()
        for t in re.findall(r"\d+|\D+", s)
    ]


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


def compute_valid_token_length(wav_len_samples: int) -> int:
    """
    Maps the true validity of waveform length to Whisper encoder effective token number.

    Approximate relationship:
        mel_frames ~= ceil(samples / HOP_LENGTH)
        token_len  ~= (mel_frames + 1) // 2
    """
    if int(wav_len_samples) <= 0:
        return 1

    n_mel_valid = int(math.ceil(int(wav_len_samples) / float(HOP_LENGTH)))
    n_mel_valid = min(MAX_MEL_FRAMES, max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)


def canonical_wav_path(path: str) -> str:
    """Regularizes the wav path to avoid matching between the relative path and the symbol link."""
    return str(Path(str(path)).expanduser().resolve())


def normalize_process2_participant_id(value: str) -> str:
    """
    Harmonization PROCESS-2 participant ID.

    meta-info.csv:
        PROCESS-2_rec__003

    CTD wav:
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

    MMSE / age / gender Neither of them was involved in the generation of the three labels;
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
        required_columns - set(frame.columns)
    )
    if missing_columns:
        raise ValueError(
            "meta-info.csv Synchronising folder failed: %s: %s"
            f"{sorted(missing_columns)}"
        )

    meta_map: Dict[str, Dict[str, Any]] = {}

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
        split_name = normalize_process2_split(
            row["Split"]
        )

        if participant_id in meta_map:
            raise ValueError(
                "meta-info.csv participant ID Repeat:"
                f"{participant_id}"
            )

        meta_map[participant_id] = {
            "participant_id": participant_id,
            "raw_id": str(row["IDs"]),
            "diagnosis": diagnosis_name,
            "target_label": int(target_label),
            "split": split_name,
            "meta_row_index": int(row_index),
        }

    if len(meta_map) == 0:
        raise ValueError(
            "meta-info.csv No valid participant."
        )

    labels = [
        int(row["target_label"])
        for row in meta_map.values()
    ]
    unique_labels = sorted(set(labels))
    if unique_labels != [0, 1, 2]:
        raise ValueError(
            "meta-info.csv Must contain both HC/MCI/Dementia "
            f"Category III, current labels={unique_labels}"
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
    PROCESS-2 Only __CTD.wav is retained; all other data sets are kept.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None


def list_wav_files(
    audio_dir: str,
    dataset_key: Optional[str] = None,
) -> List[str]:
    """Read wav in official subdirectories and perform natural sorting."""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    audio_dir = str(audio_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"official audio dir Cannot initialise Evolution's mail component.{audio_dir}")

    files: List[str] = []
    for name in os.listdir(audio_dir):
        if not str(name).lower().endswith(".wav"):
            continue

        wav_path = str(Path(audio_dir) / name)
        if not _is_process2_ctd_wav_path(
            wav_path,
            dataset_key=dataset_key,
        ):
            continue

        files.append(wav_path)

    files = sorted(files, key=natural_key)

    if len(files) == 0:
        raise RuntimeError(
            "official audio dir No eligible wav:"
            f"{audio_dir}, dataset_key={dataset_key}"
        )

    if str(dataset_key).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] {audio_dir}: "
            f"Read only CTD wav, number of documents ={len(files)}"
        )

    return files


def build_entries_from_official_dir(
    audio_dir: str,
    split_name: str,
    group_name: str,
    meta_map: Dict[str, Dict[str, Any]],
    dataset_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convert a PROCESS-2 official directory to three categories entry.

    Important:
    - cc/cd Folders no longer decide the final target_label;
    - Final label from meta-info.csv diagnosis;
    - cc/cd For data consistency checks only.

    Rules:
        cc -> diagnosis Must be HC
        cd -> diagnosis Must be MCI or Dementia
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        raise ValueError(
            "CPFT_3class_stage1.py For the time being, only PROCESS-2 is supported."
            f"Got it. Dataset_key={dataset_key}"
        )

    split_name = normalize_process2_split(
        split_name
    )
    group_name = str(
        group_name
    ).strip().lower()

    if group_name not in {"cc", "cd"}:
        raise ValueError(
            "group_name It is necessary to be cc/cd,"
            f"Current ={group_name}"
        )

    entries: List[Dict[str, Any]] = []

    for wav_path in list_wav_files(
        audio_dir,
        dataset_key=dataset_key,
    ):
        wav_path = canonical_wav_path(
            wav_path
        )
        participant_id = (
            normalize_process2_participant_id(
                wav_path
            )
        )

        if participant_id not in meta_map:
            raise KeyError(
                "official CTD wav Unable to find in meta-info.csv: \n"
                f"wav_path={wav_path}\n"
                f"participant_id={participant_id}"
            )

        meta = meta_map[participant_id]

        if str(meta["split"]) != split_name:
            raise ValueError(
                "official The list of contents is inconsistent with meta Split: \n"
                f"participant={participant_id}\n"
                f"directory_split={split_name}\n"
                f"meta_split={meta['split']}"
            )

        diagnosis = str(meta["diagnosis"])
        target_label = int(
            meta["target_label"]
        )

        if group_name == "cc":
            if diagnosis != "HC":
                raise ValueError(
                    "PROCESS-2 cc Folders must be HC:\n"
                    f"participant={participant_id}\n"
                    f"diagnosis={diagnosis}\n"
                    f"wav={wav_path}"
                )
        else:
            if diagnosis not in {
                "MCI",
                "Dementia",
            }:
                raise ValueError(
                    "PROCESS-2 cd Folders must respond "
                    "MCI/Dementia:\n"
                    f"participant={participant_id}\n"
                    f"diagnosis={diagnosis}\n"
                    f"wav={wav_path}"
                )

        audio_id = process2_audio_id(
            participant_id
        )

        entries.append(
            {
                "wav_path": wav_path,
                "wav_name": Path(wav_path).name,
                "participant_id": participant_id,
                "raw_participant_id": str(
                    meta["raw_id"]
                ),
                "split": split_name,

                # Keeping group and source_group,
                # Compatible to follow Stage2 to read old fields.
                "group": group_name,
                "source_group": group_name,
                "source_subset": (
                    f"{split_name}_{group_name}"
                ),

                "diagnosis": diagnosis,
                "target_label": target_label,

                # stable participant-level audio id
                "official_audio_id": int(
                    audio_id
                ),
                "audio_id": int(
                    audio_id
                ),

                "process2_task_filter": "CTD",
                "label_source": (
                    "meta-info.csv:diagnosis"
                ),
            }
        )

    return entries


def validate_classification_entries(
    entries: List[Dict[str, Any]],
    split_name: str,
    require_all_classes: bool = True,
) -> Dict[str, Any]:
    """
    Checking for PROCESS-2 Class III entry:
    - wav Existence;
    - target_label For 0/1/2 only;
    - diagnosis Consistency with target_label;
    - audio_id (a) One;
    - Formally split must comprise three categories simultaneously.
    """
    if len(entries) == 0:
        raise ValueError(
            f"{split_name} entries It's empty."
        )

    labels: List[int] = []
    audio_ids: List[int] = []

    expected_diagnosis = {
        0: "HC",
        1: "MCI",
        2: "Dementia",
    }

    for row_index, row in enumerate(entries):
        wav_path = row.get(
            "wav_path",
            None,
        )
        if wav_path in [None, ""]:
            raise ValueError(
                f"{split_name} Article {row_index} Article "
                "entry Missing wav_path."
            )

        if not os.path.isfile(
            str(wav_path)
        ):
            raise FileNotFoundError(
                f"{split_name} wav Cannot initialise Evolution's mail component."
                f"{wav_path}"
            )

        if "target_label" not in row:
            raise ValueError(
                f"{split_name} Article {row_index} Article "
                "entry Missing target_label."
            )

        label = int(
            row["target_label"]
        )
        if label not in {0, 1, 2}:
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
        if diagnosis != expected_diagnosis[label]:
            raise ValueError(
                f"{split_name} diagnosis Inconsistent with label:"
                f"diagnosis={diagnosis}, "
                f"label={label}"
            )

        labels.append(label)

        if "audio_id" not in row:
            raise ValueError(
                f"{split_name} Article {row_index} Article "
                "entry Missing audio_id."
            )
        audio_ids.append(
            int(row["audio_id"])
        )

    unique_labels = sorted(
        set(labels)
    )
    if (
        require_all_classes
        and unique_labels != [0, 1, 2]
    ):
        raise ValueError(
            f"{split_name} Must be included at the same time. "
            "HC/MCI/Dementia Three groups,"
            f"Current ={unique_labels}"
        )

    if len(audio_ids) != len(
        set(audio_ids)
    ):
        raise ValueError(
            f"{split_name} There is a repetition of audio_id."
        )

    labels_array = np.asarray(
        labels,
        dtype=np.int64,
    )

    class0 = int(
        np.sum(labels_array == 0)
    )
    class1 = int(
        np.sum(labels_array == 1)
    )
    class2 = int(
        np.sum(labels_array == 2)
    )
    total = int(
        len(labels_array)
    )

    stats = {
        "total": total,

        # Numerical field: Compatible for follow-up script reading.
        "class0": class0,
        "class1": class1,
        "class2": class2,

        # semantic fields: facilitate manual checking.
        "hc": class0,
        "mci": class1,
        "dementia": class2,

        "class0_ratio": float(
            class0 / max(total, 1)
        ),
        "class1_ratio": float(
            class1 / max(total, 1)
        ),
        "class2_ratio": float(
            class2 / max(total, 1)
        ),
        "unique_labels": unique_labels,
    }

    print(
        f"[{split_name}] audio | "
        f"total={total} | "
        f"HC={class0} | "
        f"MCI={class1} | "
        f"Dementia={class2}"
    )

    return stats


def load_official_train_entries(
    dataset_option: str,
) -> List[Dict[str, Any]]:
    """
    Construct PROCESS-2 official_train, category III entries.

    Stage1/2/3 Use official_train only;
    official_test Use only in the final CPFT_3class_extract.py term.
    """
    if dataset_option != "PROCESS-2":
        raise ValueError(
            "CPFT_3class_stage1.py For the time being, only PROCESS-2 is supported."
            f"Got it.daset_option={dataset_option}"
        )

    if dataset_option not in DATASETS:
        raise ValueError(
            "DATASETS Cannot initialise Evolution's mail component."
            f"{dataset_option}"
        )

    paths = DATASETS[
        dataset_option
    ]

    meta_map = load_process2_meta(
        PROCESS2_META_PATH
    )

    train_entries = (
        build_entries_from_official_dir(
            paths["train_cc"],
            split_name="train",
            group_name="cc",
            meta_map=meta_map,
            dataset_key=dataset_option,
        )
        + build_entries_from_official_dir(
            paths["train_cd"],
            split_name="train",
            group_name="cd",
            meta_map=meta_map,
            dataset_key=dataset_option,
        )
    )

    if len(train_entries) == 0:
        raise ValueError(
            "official_train entries Empty:"
            f"dataset={dataset_option}"
        )

    # Harmonized order to stable participant-level audio_id.
    train_entries = sorted(
        train_entries,
        key=lambda row: (
            int(row["audio_id"]),
            natural_key(
                row.get(
                    "wav_path",
                    "",
                )
            ),
        ),
    )

    # Check for actual scans TRAIN participant
    # Consistent with meta-info.csv TRAIN participant.
    expected_train_participants = {
        participant_id
        for participant_id, row
        in meta_map.items()
        if row["split"] == "train"
    }
    actual_train_participants = {
        str(row["participant_id"])
        for row in train_entries
    }

    missing_audio = sorted(
        expected_train_participants
        - actual_train_participants
    )
    unexpected_audio = sorted(
        actual_train_participants
        - expected_train_participants
    )

    if unexpected_audio:
        raise ValueError(
            "official_train It doesn't belong. "
            "meta TRAIN Participant:"
            f"{unexpected_audio[:20]}"
        )

    if (
        STRICT_META_AUDIO_COVERAGE
        and missing_audio
    ):
        raise ValueError(
            "meta-info.csv TRAIN participant "
            "No matches found for CTD wav:"
            f"missing_count={len(missing_audio)}, "
            f"examples={missing_audio[:20]}"
        )

    stats = validate_classification_entries(
        train_entries,
        split_name="official_train",
        require_all_classes=True,
    )

    if STRICT_META_AUDIO_COVERAGE:
        expected = {
            "total": 320,
            "class0": 160,
            "class1": 120,
            "class2": 40,
        }
        for key, expected_value in expected.items():
            if int(stats[key]) != int(
                expected_value
            ):
                raise ValueError(
                    "PROCESS-2 official_train "
                    f"{key}={stats[key]},"
                    f"Expected ={expected_value}"
                )

    return train_entries


# ============================================================
# 5) Stage1 Three Classification Datasets and collate
# ============================================================
class ADClassDataset(data.Dataset):
    """
    Stage1 PROCESS-2 HC/MCI/Dementia Three classifications of data sets.

    Audio read rules:
        - Do not require original WAV to be 16 kHz;
        - librosa.load Automatic monochannels;
        - Automatic re-sampling to sample_rate (default 16000 Hz);
        - wav_len Indicates the number of valid sample points after re-sampling, before 30 seconds is processed.

    Return each sample:
        wav          : Float32 waveform
        target_label : 0/1/2
        wav_len      : Valid sample points
        name         : Filename (returnname=True)
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        returnname: bool = False,
        sample_rate: int = SAMPLE_RATE,
        target_sec: float = TARGET_SEC,
    ):
        self.entries = [
            copy.deepcopy(x)
            for x in entries
        ]
        self.returnname = bool(
            returnname
        )
        self.sample_rate = int(
            sample_rate
        )
        self.target_seqlen = int(
            float(target_sec)
            * self.sample_rate
        )

        if len(self.entries) == 0:
            raise ValueError(
                "ADClassDataset Roger that, entries."
            )

        self.labels = [
            int(x["target_label"])
            for x in self.entries
        ]
        invalid = sorted(
            set(self.labels)
            - {0, 1, 2}
        )
        if invalid:
            raise ValueError(
                "ADClassDataset Include illegal three classification labels:"
                f"{invalid}"
            )

    def __len__(self) -> int:
        return len(
            self.entries
        )

    def __getitem__(
        self,
        index: int,
    ):
        """
        Reads a official train audio and harmonizes the format required for Whisper.

        Data may contain 16 kHz / 44.1 kHz / 48 kHz.
        librosa.load(..., sr=16000, mono=True)
        Responsible for the harmonization of sampling rates and channels.
        """
        row = self.entries[index]

        wav_path = str(
            row["wav_path"]
        )
        target_label = int(
            row["target_label"]
        )
        name = Path(
            wav_path
        ).name

        if not os.path.isfile(
            wav_path
        ):
            raise FileNotFoundError(
                f"Could not close temporary folder: %s{wav_path}"
            )

        if target_label not in {
            0,
            1,
            2,
        }:
            raise ValueError(
                "Audio tags are not legal. "
                "HC/MCI/Dementia Third Classification Label:"
                f"path={wav_path}, "
                f"label={target_label}"
            )

        try:
            wav, loaded_sample_rate = (
                librosa.load(
                    wav_path,
                    sr=int(
                        self.sample_rate
                    ),
                    mono=True,
                    dtype=np.float32,
                )
            )
        except Exception as exc:
            raise RuntimeError(
                "Audio reading or resampling failed: \n"
                f"path={wav_path}\n"
                "target_sample_rate="
                f"{self.sample_rate}\n"
                f"error={exc}"
            ) from exc

        if int(
            loaded_sample_rate
        ) != int(
            self.sample_rate
        ):
            raise RuntimeError(
                "The sampling rate after audio sampling was abnormal:"
                f"path={wav_path}, "
                f"expected_sr={self.sample_rate}, "
                f"actual_sr={loaded_sample_rate}"
            )

        wav = np.asarray(
            wav,
            dtype=np.float32,
        ).reshape(-1)

        if wav.size == 0:
            raise ValueError(
                f"Read empty audio:{wav_path}"
            )

        if not np.all(
            np.isfinite(wav)
        ):
            raise ValueError(
                "Audio contains NaN or Inf:"
                f"{wav_path}"
            )

        wav_len = int(
            wav.shape[0]
        )

        if not self.returnname:
            return (
                wav,
                target_label,
                wav_len,
            )

        return (
            wav,
            target_label,
            wav_len,
            name,
        )

    @staticmethod
    def _repeat_or_center_crop_to_length(
        x: np.ndarray,
        target_len: int,
    ) -> Tuple[np.ndarray, int]:
        """
        Consistency with original Stage1 data view:
        - Longer than 30 seconds: centre tailoring;
        - Less than 30 seconds: Duplicate filling;
        - valid_len The original valid length is still recorded.
        """
        x = np.asarray(
            x,
            dtype=np.float32,
        ).reshape(-1)
        length = int(
            x.shape[0]
        )

        if length <= 0:
            return (
                np.zeros(
                    (target_len,),
                    dtype=np.float32,
                ),
                0,
            )

        if length >= target_len:
            start = (
                length - target_len
            ) // 2
            segment = x[
                start : start + target_len
            ]
            return (
                segment.astype(
                    np.float32,
                    copy=False,
                ),
                target_len,
            )

        repeat_count = (
            target_len // length
        )
        remainder = (
            target_len
            - repeat_count * length
        )

        output = np.tile(
            x,
            repeat_count,
        )
        if remainder > 0:
            output = np.concatenate(
                [
                    output,
                    x[:remainder],
                ],
                axis=0,
            )

        output = output[
            :target_len
        ]
        return (
            output.astype(
                np.float32,
                copy=False,
            ),
            length,
        )

    def repeat30s_collate(
        self,
        batch,
    ):
        """
        Set batch to 30 seconds waveform.

        Return:
            wav_30       : [B, 480000]
            target_label : [B], 0/1/2
            valid_len    : [B]
            names        : List[str](returnname=True)
        """
        target_len = (
            self.target_seqlen
        )

        def process(item):
            if self.returnname:
                (
                    wav,
                    target_label,
                    wav_len,
                    name,
                ) = item
            else:
                (
                    wav,
                    target_label,
                    wav_len,
                ) = item

            wav = np.asarray(
                wav,
                dtype=np.float32,
            ).reshape(-1)
            wav_len = max(
                0,
                min(
                    int(wav_len),
                    int(
                        wav.shape[0]
                    ),
                ),
            )
            base = (
                wav[:wav_len]
                if wav_len > 0
                else wav
            )

            (
                wav_30,
                valid_len,
            ) = (
                self
                ._repeat_or_center_crop_to_length(
                    base,
                    target_len=target_len,
                )
            )

            if self.returnname:
                return (
                    wav_30,
                    int(target_label),
                    int(valid_len),
                    name,
                )

            return (
                wav_30,
                int(target_label),
                int(valid_len),
            )

        processed = list(
            map(
                process,
                batch,
            )
        )
        return default_collate(
            processed
        )


# ============================================================
# 6) Whisper With class header components
# ============================================================
class LearnableScalarMix(nn.Module):
    """Learning multilayer integration: [L,B,T,C] -> [B,T,C]."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = int(n_layers)
        self.w = nn.Parameter(torch.zeros(self.n_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        if int(stack.size(0)) != int(self.n_layers):
            raise RuntimeError(
                f"ScalarMix Expecting L={self.n_layers},"
                f"Current ={stack.size(0)}"
            )

        weights = torch.softmax(self.w, dim=0).view(-1, 1, 1, 1)
        mixed = torch.sum(weights * stack, dim=0)
        return self.gamma * mixed


class WhisperEncoderWrapper(nn.Module):
    """
    Whisper encoder Encapsulation:
    1. After the full freeze, only the specified encoder layer will be unfrozen;
    2. Supports single or multi-layer expressions;
    3. Support for mean/scalar_mix;
    4. Construct token mask using valid_wav_lens.
    """

    def __init__(
        self,
        model_name: str,
        unfreeze_encoder_layers: Optional[Sequence[int]] = None,
        scalar_mix_n_layers: Optional[int] = None,
    ):
        super().__init__()

        self.model = WhisperModel.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_name
        )
        self.hidden_size = int(self.model.config.d_model)

        if unfreeze_encoder_layers is None:
            unfreeze_encoder_layers = []

        requested_layers = sorted(
            {
                int(x)
                for x in list(unfreeze_encoder_layers)
                if int(x) > 0
            }
        )

        self.scalar_mix = (
            LearnableScalarMix(int(scalar_mix_n_layers))
            if scalar_mix_n_layers is not None
            else None
        )

        # Freeze the whole Whisper first.
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        encoder_layers = self.model.encoder.layers
        total_layers = len(encoder_layers)

        valid_layers: List[int] = []
        for layer_index_1based in requested_layers:
            if not (1 <= int(layer_index_1based) <= int(total_layers)):
                continue

            layer = encoder_layers[int(layer_index_1based) - 1]
            for parameter in layer.parameters():
                parameter.requires_grad = True
            valid_layers.append(int(layer_index_1based))

        self.unfreeze_encoder_layers = valid_layers

        # When unfrozen at encoder level, unfrozen at the same time encoder final layer norm.
        if len(self.unfreeze_encoder_layers) > 0:
            for parameter in self.model.encoder.layer_norm.parameters():
                parameter.requires_grad = True

        print(
            "[UNFREEZE] Stage1 Whisper encoder layers = "
            f"{self.unfreeze_encoder_layers}"
        )

    def trainable_params(self) -> List[torch.Tensor]:
        """Returns all current training parameters."""
        return [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

    @staticmethod
    def _parse_repr_layers(
        repr_layers: Union[str, List[int], None],
    ) -> Optional[List[int]]:
        if repr_layers is None:
            return None

        if isinstance(repr_layers, str):
            text = repr_layers.strip()
            if text == "":
                return None
            return [
                int(part.strip())
                for part in text.split(",")
                if part.strip() != ""
            ]

        if isinstance(repr_layers, (list, tuple)):
            return [int(x) for x in repr_layers]

        return None

    def _build_attn_mask(
        self,
        input_features: torch.Tensor,
        valid_wav_lens: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if valid_wav_lens is None:
            return None

        if not torch.is_tensor(valid_wav_lens):
            valid_wav_lens = torch.tensor(valid_wav_lens)

        valid_wav_lens = valid_wav_lens.detach().cpu().long().view(-1)
        batch_size = int(input_features.size(0))

        if int(valid_wav_lens.numel()) != batch_size:
            raise RuntimeError(
                "valid_wav_lens Numbers do not match batch:"
                f"expect={batch_size}, got={valid_wav_lens.numel()}"
            )

        mel_length = int(input_features.size(-1))
        token_length = (mel_length + 1) // 2

        valid_token_lengths: List[int] = []
        for item_index in range(batch_size):
            valid_tokens = compute_valid_token_length(
                int(valid_wav_lens[item_index].item())
            )
            valid_token_lengths.append(
                min(valid_tokens, token_length)
            )

        valid_token_lengths_tensor = torch.tensor(
            valid_token_lengths,
            device=input_features.device,
            dtype=torch.long,
        )

        token_ids = torch.arange(
            token_length,
            device=input_features.device,
        ).view(1, token_length).expand(batch_size, token_length)

        return (
            token_ids < valid_token_lengths_tensor.view(batch_size, 1)
        ).bool()

    def forward(
        self,
        wav: torch.Tensor,
        sampling_rate: int = SAMPLE_RATE,
        repr_layers: Union[str, List[int], None] = None,
        repr_pool: str = "mean",
        valid_wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enter waveform to output Whisper token for [B,T,C]."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)

        wav_cpu = wav.detach().cpu().numpy()
        target_length = int(self.feature_extractor.n_samples)

        feature_batch = self.feature_extractor(
            wav_cpu,
            sampling_rate=int(sampling_rate),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=target_length,
        )

        device = next(self.model.parameters()).device
        input_features = feature_batch.input_features.to(device)

        attention_mask = None
        if valid_wav_lens is not None:
            attention_mask = self._build_attn_mask(
                input_features,
                valid_wav_lens,
            )

        parsed_layers = self._parse_repr_layers(repr_layers)

        if parsed_layers is None:
            try:
                encoder_output = self.model.encoder(
                    input_features=input_features,
                    attention_mask=attention_mask,
                    return_dict=True,
                    output_hidden_states=False,
                )
            except TypeError:
                encoder_output = self.model.encoder(
                    input_features=input_features,
                    return_dict=True,
                    output_hidden_states=False,
                )
            return encoder_output.last_hidden_state

        try:
            encoder_output = self.model.encoder(
                input_features=input_features,
                attention_mask=attention_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        except TypeError:
            encoder_output = self.model.encoder(
                input_features=input_features,
                return_dict=True,
                output_hidden_states=True,
            )

        hidden_states = encoder_output.hidden_states
        selected_layers: List[torch.Tensor] = []

        for layer_id in parsed_layers:
            if int(layer_id) < 1 or int(layer_id) >= len(hidden_states):
                raise ValueError(
                    f"repr layer Illegal: layer={layer_id}, "
                    f"hidden_states_len={len(hidden_states)}"
                )
            selected_layers.append(hidden_states[int(layer_id)])

        stacked = torch.stack(selected_layers, dim=0)

        if str(repr_pool) == "mean":
            return stacked.mean(dim=0)

        if str(repr_pool) == "scalar_mix":
            if self.scalar_mix is None:
                raise RuntimeError(
                    "repr_pool='scalar_mix',However, scalar_mix was not initialized."
                )
            return self.scalar_mix(stacked)

        raise ValueError(f"Unknown repr_pool={repr_pool}")


class LinearHead(nn.Module):
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(int(in_dim))
        self.fc1 = nn.Linear(int(in_dim), int(hidden_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.fc2 = nn.Linear(int(hidden_dim), int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.fc2(x)


class AttentiveStatsPool(nn.Module):
    """attentive mean + std pooling with mask, with an output dimension 2*in_dim."""

    def __init__(
        self,
        in_dim: int,
        attn_hidden: int = 128,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = float(eps)
        self.attention = nn.Sequential(
            nn.Linear(int(in_dim), int(attn_hidden)),
            nn.Tanh(),
            nn.Linear(int(attn_hidden), 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention_score = self.attention(x).squeeze(-1)

        if mask is not None:
            mask = mask.bool()
            attention_score = attention_score.masked_fill(~mask, -1e4)

        attention_weight = F.softmax(attention_score, dim=1)

        if mask is not None:
            attention_weight = attention_weight * mask.float()
            attention_weight = attention_weight / attention_weight.sum(
                dim=1,
                keepdim=True,
            ).clamp_min(self.eps)

        mean = torch.sum(
            x * attention_weight.unsqueeze(-1),
            dim=1,
        )
        variance = torch.sum(
            ((x - mean.unsqueeze(1)) ** 2)
            * attention_weight.unsqueeze(-1),
            dim=1,
        )
        std = torch.sqrt(variance.clamp_min(self.eps))
        return torch.cat([mean, std], dim=-1)


# ============================================================
# 7) LightningModule:PROCESS-2 Category III Stage1
# ============================================================
class WhisperStage1Class(pl.LightningModule):
    """
    PROCESS-2 HC/MCI/Dementia-aware Stage1 Double head 3 classification model.

    utterance head:The entire section 30 sec view classification;
    frame head:The 0/1/2 diagnostic label for copying the same audio on valid token is used as an auxiliary target.
    """

    def __init__(
        self,
        train_entries: List[Dict[str, Any]],
        saving_path: str,
        run_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_entries"])

        self.saving_path = str(saving_path)
        self.run_config = copy.deepcopy(run_config)

        self.batch_size = int(run_config["BATCH_SIZE"])
        self.num_workers = int(run_config["NUM_WORKERS"])
        self.maxstep = int(run_config["TRAINING_STEP"])

        self.lambda_utt = float(run_config["LAMBDA_UTT"])
        self.lambda_frame = float(run_config["LAMBDA_FRAME"])
        self.num_classes = int(run_config["NUM_CLASSES"])

        self.repr_layers = run_config["CURRENT_STAGE1_REPR_LAYERS"]
        self.repr_pool = str(run_config["STAGE1_REPR_POOL"])

        self.train_dataset = ADClassDataset(
            entries=train_entries,
            returnname=True,
            sample_rate=SAMPLE_RATE,
            target_sec=float(run_config["TARGET_SEC"]),
        )

        try:
            self.utt_criterion = nn.CrossEntropyLoss(
                label_smoothing=float(run_config["LABEL_SMOOTHING"])
            )
            self.frame_criterion = nn.CrossEntropyLoss(
                label_smoothing=float(run_config["LABEL_SMOOTHING"])
            )
        except TypeError:
            # Compatibility with old version label_smoothing PyTorch does not support it.
            self.utt_criterion = nn.CrossEntropyLoss()
            self.frame_criterion = nn.CrossEntropyLoss()

        parsed_layers = WhisperEncoderWrapper._parse_repr_layers(
            self.repr_layers
        )
        scalar_mix_n_layers = (
            None
            if parsed_layers is None
            else len(parsed_layers)
        )

        self.whisper = WhisperEncoderWrapper(
            model_name=str(run_config["WHISPER_MODEL_NAME"]),
            unfreeze_encoder_layers=run_config[
                "STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS"
            ],
            scalar_mix_n_layers=(
                scalar_mix_n_layers
                if self.repr_pool == "scalar_mix"
                else None
            ),
        )

        model_dim = int(self.whisper.hidden_size)

        # frame head:Each Whisper token produces three categories logits.
        self.frame_head = LinearHead(
            in_dim=model_dim,
            hidden_dim=int(run_config["HEAD_HIDDEN"]),
            out_dim=self.num_classes,
            dropout=float(run_config["HEAD_DROPOUT"]),
        )

        # utterance head:attention mean+std pooling , and then output three categories logits.
        self.utt_pool = AttentiveStatsPool(
            model_dim,
            attn_hidden=int(run_config["HEAD_HIDDEN"]),
        )
        self.utt_head = LinearHead(
            in_dim=model_dim * 2,
            hidden_dim=int(run_config["HEAD_HIDDEN"]),
            out_dim=self.num_classes,
            dropout=float(run_config["HEAD_DROPOUT"]),
        )

        # utterance indicators used to calculate the accurate classification epoch-level III.
        # Line = true class, column = predicted class.
        self.register_buffer(
            "_train_confusion",
            torch.zeros(
                (
                    self.num_classes,
                    self.num_classes,
                ),
                dtype=torch.long,
            ),
            persistent=False,
        )

    @staticmethod
    def _build_token_mask(
        valid_wav_lens: torch.Tensor,
        token_length: int,
        device,
    ) -> torch.Tensor:
        valid_wav_lens = valid_wav_lens.detach().cpu().long().view(-1)
        batch_size = int(valid_wav_lens.numel())

        valid_token_lengths: List[int] = []
        for item_index in range(batch_size):
            valid_tokens = compute_valid_token_length(
                int(valid_wav_lens[item_index].item())
            )
            valid_token_lengths.append(
                min(valid_tokens, int(token_length))
            )

        valid_token_lengths_tensor = torch.tensor(
            valid_token_lengths,
            device=device,
            dtype=torch.long,
        )

        token_ids = torch.arange(
            int(token_length),
            device=device,
        ).view(1, int(token_length)).expand(
            batch_size,
            int(token_length),
        )

        return token_ids < valid_token_lengths_tensor.view(batch_size, 1)

    def forward(
        self,
        wav_batch: torch.Tensor,
        valid_len: Optional[torch.Tensor] = None,
    ):
        """
        Return:
            frame_logits : [B,T,3]
            utt_logits   : [B,3]
            token_mask   : [B,T]
        """
        representations = self.whisper(
            wav_batch,
            sampling_rate=SAMPLE_RATE,
            repr_layers=self.repr_layers,
            repr_pool=self.repr_pool,
            valid_wav_lens=valid_len,
        )

        token_mask = None
        if valid_len is not None:
            token_mask = self._build_token_mask(
                valid_len,
                token_length=int(representations.size(1)),
                device=representations.device,
            )

        utterance_representation = self.utt_pool(
            representations,
            mask=token_mask,
        )
        utterance_logits = self.utt_head(utterance_representation)
        frame_logits = self.frame_head(representations)

        return frame_logits, utterance_logits, token_mask

    def train_dataloader(self):
        """official_train DataLoader."""
        num_workers = int(self.num_workers)
        return data.DataLoader(
            self.train_dataset,
            batch_size=int(self.batch_size),
            num_workers=num_workers,
            drop_last=False,
            shuffle=True,
            collate_fn=self.train_dataset.repeat30s_collate,
            persistent_workers=(num_workers > 0),
            pin_memory=torch.cuda.is_available(),
        )

    def configure_optimizers(self):
        """
        AdamW Grouping learning rate:
        - (a) Three subheadings and pooling: LR_HEAD;
        - Whisper Defrozen layer: LR_WHISPER;
        - scalar mix:LR_HEAD.
        """
        head_parameters: List[torch.Tensor] = []
        head_parameters += list(self.frame_head.parameters())
        head_parameters += list(self.utt_head.parameters())
        head_parameters += list(self.utt_pool.parameters())

        all_whisper_trainables = list(self.whisper.trainable_params())

        scalar_mix_parameters: List[torch.Tensor] = []
        if getattr(self.whisper, "scalar_mix", None) is not None:
            scalar_mix_parameters = list(
                self.whisper.scalar_mix.parameters()
            )

        scalar_mix_ids = {
            id(parameter)
            for parameter in scalar_mix_parameters
        }
        whisper_parameters = [
            parameter
            for parameter in all_whisper_trainables
            if id(parameter) not in scalar_mix_ids
        ]

        parameter_groups: List[Dict[str, Any]] = [
            {
                "name": "head",
                "params": head_parameters,
                "lr": float(self.run_config["LR_HEAD"]),
            }
        ]

        if len(whisper_parameters) > 0:
            parameter_groups.append(
                {
                    "name": "whisper",
                    "params": whisper_parameters,
                    "lr": float(self.run_config["LR_WHISPER"]),
                }
            )

        if len(scalar_mix_parameters) > 0:
            parameter_groups.append(
                {
                    "name": "scalar_mix",
                    "params": scalar_mix_parameters,
                    "lr": float(self.run_config["LR_HEAD"]),
                }
            )

        optimizer = optim.AdamW(
            parameter_groups,
            weight_decay=float(self.run_config["WEIGHT_DECAY"]),
        )

        total_training_steps = max(1, int(self.maxstep))
        warmup_steps = int(
            float(self.run_config["WARMUP_RATIO"])
            * total_training_steps
        )
        flat_steps = int(
            float(self.run_config["FLAT_RATIO"])
            * total_training_steps
        )
        minimum_lr_ratio = float(self.run_config["MIN_LR_RATIO"])

        def lr_multiplier(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step + 1) / float(max(1, warmup_steps))

            if current_step < warmup_steps + flat_steps:
                return 1.0

            decay_steps = max(
                1,
                total_training_steps - warmup_steps - flat_steps,
            )
            decay_step = min(
                decay_steps,
                max(0, current_step - warmup_steps - flat_steps),
            )
            progress = float(decay_step) / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (
                minimum_lr_ratio
                + (1.0 - minimum_lr_ratio) * cosine
            )

        scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_multiplier,
            ),
            "interval": "step",
        }
        return [optimizer], [scheduler]

    @staticmethod
    def _optimizer_learning_rates(optimizer) -> Dict[str, float]:
        result: Dict[str, float] = {}
        if optimizer is None:
            return result

        for group_index, group in enumerate(optimizer.param_groups):
            group_name = str(group.get("name", f"group{group_index}"))
            result[f"lr_{group_name}"] = float(group["lr"])
        return result

    def _frame_loss_on_valid_tokens(
        self,
        frame_logits: torch.Tensor,
        target_labels: torch.Tensor,
        valid_wav_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Copy each audio 0/1/2 tab to valid token and calculate the help CE.
        """
        batch_size, token_length, num_classes = frame_logits.shape

        valid_token_lengths: List[int] = []
        for item_index in range(batch_size):
            valid_tokens = compute_valid_token_length(
                int(valid_wav_lens[item_index].item())
            )
            valid_token_lengths.append(
                min(valid_tokens, int(token_length))
            )

        valid_token_lengths_tensor = torch.tensor(
            valid_token_lengths,
            device=frame_logits.device,
            dtype=torch.long,
        )
        token_ids = torch.arange(
            token_length,
            device=frame_logits.device,
        ).view(1, token_length).expand(batch_size, token_length)
        valid_mask = token_ids < valid_token_lengths_tensor.view(
            batch_size,
            1,
        )

        flattened_logits = frame_logits.reshape(-1, num_classes)
        flattened_labels = target_labels.unsqueeze(1).expand(
            -1,
            token_length,
        ).reshape(-1)
        flattened_mask = valid_mask.reshape(-1)

        if int(flattened_mask.long().sum().item()) == 0:
            zero_loss = flattened_logits.sum() * 0.0
            return zero_loss, zero_loss.detach()

        valid_logits = flattened_logits[flattened_mask]
        valid_labels = flattened_labels[flattened_mask]

        frame_loss = self.frame_criterion(
            valid_logits,
            valid_labels,
        )
        frame_predictions = torch.argmax(valid_logits, dim=-1)
        frame_accuracy = (
            frame_predictions == valid_labels
        ).float().mean()

        return frame_loss, frame_accuracy

    def on_train_epoch_start(self) -> None:
        """Each epoch begins with the 3x3 utterance Confusion Matrix."""
        self._train_confusion.zero_()

    @torch.no_grad()
    def _update_train_confusion(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """
        The matrix is confused by the cumulative three categories.

        matrix[true_class, predicted_class]
        """
        predictions = (
            predictions
            .detach()
            .long()
            .reshape(-1)
        )
        targets = (
            targets
            .detach()
            .long()
            .reshape(-1)
        )

        if predictions.numel() != targets.numel():
            raise ValueError(
                "predictions Unmatched with targets quantity."
            )

        if predictions.numel() == 0:
            return

        invalid_predictions = (
            (predictions < 0)
            | (
                predictions
                >= self.num_classes
            )
        )
        invalid_targets = (
            (targets < 0)
            | (
                targets
                >= self.num_classes
            )
        )

        if bool(
            invalid_predictions.any()
            or invalid_targets.any()
        ):
            raise ValueError(
                "Stage1 Category III prediction/target "
                "A value exceeding 0/1/2 appears."
            )

        encoded = (
            targets * self.num_classes
            + predictions
        )
        counts = torch.bincount(
            encoded,
            minlength=(
                self.num_classes
                * self.num_classes
            ),
        ).reshape(
            self.num_classes,
            self.num_classes,
        )

        self._train_confusion += counts.to(
            self._train_confusion.device
        )
    def training_step(self, batch, batch_idx):
        """
        total_loss = LAMBDA_UTT * utterance CE
                   + LAMBDA_FRAME * frame CE
        """
        wav_30, target_labels, valid_len, _names = batch

        frame_logits, utterance_logits, _token_mask = self(
            wav_30,
            valid_len,
        )

        target_labels = target_labels.to(
            utterance_logits.device
        ).long()
        valid_len = valid_len.to(frame_logits.device).long()

        utterance_loss = self.utt_criterion(
            utterance_logits,
            target_labels,
        )
        frame_loss, frame_accuracy = self._frame_loss_on_valid_tokens(
            frame_logits,
            target_labels,
            valid_len,
        )

        total_loss = (
            self.lambda_utt * utterance_loss
            + self.lambda_frame * frame_loss
        )

        utterance_predictions = torch.argmax(
            utterance_logits,
            dim=-1,
        )
        utterance_accuracy = (
            utterance_predictions == target_labels
        ).float().mean()

        self._update_train_confusion(
            utterance_predictions,
            target_labels,
        )

        batch_size = int(target_labels.numel())
        log_values: Dict[str, Any] = {
            "train_loss": total_loss,
            "train_loss_utt": utterance_loss.detach(),
            "train_loss_frame": frame_loss.detach(),
            "train_utt_accuracy": utterance_accuracy.detach(),
            "train_frame_accuracy": frame_accuracy.detach(),
        }

        if (
            self.trainer is not None
            and hasattr(self.trainer, "optimizers")
            and len(self.trainer.optimizers) > 0
        ):
            log_values.update(
                self._optimizer_learning_rates(
                    self.trainer.optimizers[0]
                )
            )

        self.log_dict(
            log_values,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )
        return total_loss

    def on_train_epoch_end(self) -> None:
        """
        Calculated from the entire round of 3x3 utterance Confusion Matrix:
        - Accuracy
        - Macro Precision / Recall / F1
        - HC / MCI / Dementia Each F1

        Macro-F1 is the unweighted arithmetic mean of the HC, MCI, and
        Dementia class-specific F1 scores.

        These indicators are used only for observation of Stage1 training and not for early stopping
        or checkpoint selection.
        """
        matrix = self._train_confusion.float()

        total = matrix.sum().clamp_min(1.0)
        true_positive = torch.diag(matrix)

        predicted_count = matrix.sum(dim=0)
        true_count = matrix.sum(dim=1)

        precision_per_class = (
            true_positive
            / predicted_count.clamp_min(1.0)
        )
        recall_per_class = (
            true_positive
            / true_count.clamp_min(1.0)
        )
        f1_per_class = (
            2.0
            * precision_per_class
            * recall_per_class
            / (
                precision_per_class
                + recall_per_class
            ).clamp_min(1e-12)
        )

        accuracy = true_positive.sum() / total
        precision_macro = precision_per_class.mean()
        recall_macro = recall_per_class.mean()
        f1_macro = f1_per_class.mean()

        self.log_dict(
            {
                "train_utt_accuracy_epoch_exact":
                    accuracy,
                "train_utt_precision_macro_epoch":
                    precision_macro,
                "train_utt_recall_macro_epoch":
                    recall_macro,
                "train_utt_f1_macro_epoch":
                    f1_macro,
                "train_utt_f1_hc_epoch":
                    f1_per_class[0],
                "train_utt_f1_mci_epoch":
                    f1_per_class[1],
                "train_utt_f1_dementia_epoch":
                    f1_per_class[2],
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
# ============================================================
# 8) Layer Resolution and Path Construction
# ============================================================
def get_whisper_encoder_layer_count(
    whisper_model_name_or_path: str,
) -> int:
    """Read Whisper config and return to encoder layers."""
    config = WhisperConfig.from_pretrained(whisper_model_name_or_path)
    num_layers = getattr(config, "encoder_layers", None)
    if num_layers is None:
        raise AttributeError(
            "WhisperConfig Missing encoder_layers."
        )
    return int(num_layers)


def resolve_unfreeze_layers_for_single_layer(
    single_layer: int,
    total_layers: int,
    unfreeze_layer_count: int,
) -> List[int]:
    """
    The actual 1-based encoder layer number is interpreted according to the number indicating the layer of defrozen layers.

    single12 + unfree7 -> [6,7,8,9,10,11,12]
    """
    single_layer = int(single_layer)
    total_layers = int(total_layers)
    unfreeze_layer_count = int(unfreeze_layer_count)

    if not (1 <= single_layer <= total_layers):
        raise ValueError(
            f"single_layer={single_layer} Beyond [1,{total_layers}]"
        )
    if unfreeze_layer_count <= 0:
        raise ValueError(
            "UNFREEZE_ENCODER_LAYERS Must be >= 1."
        )

    start_layer = max(
        1,
        single_layer - unfreeze_layer_count + 1,
    )
    end_layer = min(single_layer, total_layers)
    resolved = list(range(start_layer, end_layer + 1))

    if len(resolved) == 0:
        raise ValueError(
            "Could not close temporary folder: %s"
            f"single_layer={single_layer}, "
            f"unfreeze_layer_count={unfreeze_layer_count}"
        )
    return resolved


def _clean_path_tag(value) -> str:
    """Clears the path parameter labels."""
    if value is None:
        return ""

    text = str(value).strip()
    if text in {"", "None", "none", "null", "Null"}:
        return ""

    text = text.replace("=", "").replace(" ", "")
    text = re.sub(r"[^0-9A-Za-z_.-]+", "", text)
    return text.strip("_-. ")


def build_path_param_suffix(
    path_param_tags=None,
) -> str:
    """Converts the list of path parameters to the suffix of the underlined connection."""
    if path_param_tags is None:
        path_param_tags = PATH_PARAM_TAGS

    if isinstance(path_param_tags, (str, int, float)):
        tags = [path_param_tags]
    else:
        tags = list(path_param_tags)

    cleaned = [_clean_path_tag(x) for x in tags]
    cleaned = [x for x in cleaned if x != ""]
    return "_".join(cleaned)


def build_layer_name(
    single_layer: int,
    unfreeze_layer_count: int,
    path_param_tags=None,
) -> str:
    """Construct singleX_unfreeZ[_suffix]."""
    base = (
        f"single{int(single_layer)}_"
        f"unfree{int(unfreeze_layer_count)}"
    )
    suffix = build_path_param_suffix(path_param_tags)
    return base if suffix == "" else f"{base}_{suffix}"


def build_run_config(
    dataset_option: str,
    layer_id: int,
) -> Dict[str, Any]:
    """
    Construct the configuration of the Stage1 PROCESS-2 classification.

    Path:
        CPFT_ROOT/
        PROCESS-2/
        CPFT_lr5e-5_epo5_3class/
        singleX_unfreeZ/
        stage1_class/
    """
    if dataset_option != "PROCESS-2":
        raise ValueError(
            "CPFT_3class_stage1.py For the time being, only PROCESS-2 is supported."
            f"Got it.daset_option={dataset_option}"
        )

    layer_id = int(
        layer_id
    )

    total_layers = (
        get_whisper_encoder_layer_count(
            WHISPER_MODEL_NAME
        )
    )

    resolved_unfreeze_layers = (
        resolve_unfreeze_layers_for_single_layer(
            single_layer=layer_id,
            total_layers=total_layers,
            unfreeze_layer_count=(
                UNFREEZE_ENCODER_LAYERS
            ),
        )
    )

    layer_name = build_layer_name(
        single_layer=layer_id,
        unfreeze_layer_count=(
            UNFREEZE_ENCODER_LAYERS
        ),
        path_param_tags=PATH_PARAM_TAGS,
    )

    run_root = os.path.join(
        CPFT_ROOT,
        str(dataset_option),
        MODEL_ROOT_TAG,
        layer_name,
    )
    stage1_save_dir = os.path.join(
        run_root,
        "stage1_class",
    )
    meta_save_dir = os.path.join(
        run_root,
        "meta",
    )

    return {
        "task":
            "process2_hc_mci_dementia_3class",
        "class_mapping": {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
            "2": CLASS_NAMES[2],
        },

        "label_source":
            "PROCESS-2/meta-info.csv diagnosis",
        "split_source":
            "PROCESS-2/meta-info.csv Split",
        "PROCESS2_META_PATH":
            PROCESS2_META_PATH,
        "mmse_used":
            False,
        "audio_task":
            "CTD",

        "CLASS_PROJECT_ROOT":
            CLASS_PROJECT_ROOT,
        "PROJECT_ROOT":
            PROJECT_ROOT,
        "CPFT_ROOT":
            CPFT_ROOT,

        "DATASET_OPTION":
            str(dataset_option),
        "LAYER_ID":
            layer_id,
        "RUN_PROTOCOL_TAG":
            str(RUN_PROTOCOL_TAG),

        "official_train_used":
            True,
        "official_test_used":
            False,

        "LAYER_TAG":
            layer_name,
        "LAYER_NAME":
            layer_name,

        "PATH_PARAM_TAGS":
            list(PATH_PARAM_TAGS),
        "PATH_PARAM_SUFFIX":
            build_path_param_suffix(
                PATH_PARAM_TAGS
            ),
        "STAGE1_SOURCE_PATH_PARAM_TAGS":
            list(
                STAGE1_SOURCE_PATH_PARAM_TAGS
            ),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX":
            build_path_param_suffix(
                STAGE1_SOURCE_PATH_PARAM_TAGS
            ),

        "MODEL_ROOT_TAG":
            MODEL_ROOT_TAG,
        "RUN_ROOT":
            run_root,
        "STAGE1_SAVE_DIR":
            stage1_save_dir,
        "META_SAVE_DIR":
            meta_save_dir,

        "CURRENT_STAGE1_REPR_LAYERS":
            [layer_id],
        "STAGE1_REPR_POOL":
            STAGE1_REPR_POOL,

        "UNFREEZE_ENCODER_LAYERS":
            int(
                UNFREEZE_ENCODER_LAYERS
            ),
        "STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS":
            list(
                resolved_unfreeze_layers
            ),
        "TOTAL_ENCODER_LAYERS":
            int(total_layers),

        "STAGE1_WHISPER_TRAIN_LAYER_START":
            int(
                min(
                    resolved_unfreeze_layers
                )
            ),
        "STAGE1_WHISPER_TRAIN_LAYER_END":
            int(
                max(
                    resolved_unfreeze_layers
                )
            ),
        "STAGE1_WHISPER_TRAIN_LAST_N":
            None,

        "WHISPER_MODEL_NAME":
            WHISPER_MODEL_NAME,
        "TARGET_SEC":
            float(TARGET_SEC),

        "NUM_CLASSES":
            int(NUM_CLASSES),
        "LAMBDA_UTT":
            float(LAMBDA_UTT),
        "LAMBDA_FRAME":
            float(LAMBDA_FRAME),

        "HEAD_HIDDEN":
            int(HEAD_HIDDEN),
        "HEAD_DROPOUT":
            float(HEAD_DROPOUT),
        "LABEL_SMOOTHING":
            float(LABEL_SMOOTHING),

        "LR_HEAD":
            float(LR_HEAD),
        "LR_WHISPER":
            float(LR_WHISPER),
        "WEIGHT_DECAY":
            float(WEIGHT_DECAY),
        "WARMUP_RATIO":
            float(WARMUP_RATIO),
        "FLAT_RATIO":
            float(FLAT_RATIO),
        "MIN_LR_RATIO":
            float(MIN_LR_RATIO),

        "BATCH_SIZE":
            int(BATCH_SIZE),
        "NUM_WORKERS":
            int(NUM_WORKERS),

        "STAGE1_MAX_EPOCHS":
            int(
                STAGE1_MAX_EPOCHS
            ),
        "STAGE1_CHECKPOINT_SELECTION":
            "last",

        "TRAINING_STEP":
            int(TRAINING_STEP),
        "PRECISION":
            int(PRECISION),
        "ACCUMULATE_GRAD_BATCHES":
            int(
                ACCUMULATE_GRAD_BATCHES
            ),
        "GRAD_CLIP":
            float(GRAD_CLIP),

        "SEED":
            int(SEED),
        "ENABLE_CSV_LOGGER":
            bool(
                ENABLE_CSV_LOGGER
            ),
        "RESUME_CHECKPOINT":
            RESUME_CHECKPOINT,
        "SAVE_MANIFESTS":
            bool(
                SAVE_MANIFESTS
            ),

        "STAGE1_REPR_LAYERS_TO_RUN":
            list(
                STAGE1_REPR_LAYERS_TO_RUN
            ),
        "STAGE2_REP_LAYERS_TO_RUN":
            list(
                STAGE2_REP_LAYERS_TO_RUN
            ),
        "STAGE3_REPR_LAYERS_TO_RUN":
            list(
                STAGE3_REPR_LAYERS_TO_RUN
            ),
        "STAGE4_OUTPUT_LAYERS_TO_RUN":
            list(
                STAGE4_OUTPUT_LAYERS_TO_RUN
            ),
    }


# ============================================================
# 9) Trainer,List with Single Runs
# ============================================================
def build_trainer(
    stage1_save_dir: str,
    run_config: Dict[str, Any],
) -> Tuple[Trainer, Optional[CSVLogger]]:
    """
    Stage1 no-val-last Trainer:
    - Do not do validation;
    - Do not do early stopping;
    - Fixed epoch;
    - Save last.ckpt only.
    """
    ensure_dir(stage1_save_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=stage1_save_dir,
        filename="whisper-s1-class-last",
        save_top_k=0,
        save_last=True,
        monitor=None,
        verbose=True,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    precision = (
        "16-mixed"
        if int(run_config["PRECISION"]) == 16
        and torch.cuda.is_available()
        else "32-true"
    )

    logger: Union[CSVLogger, bool] = False
    enable_progress_bar = True
    if bool(run_config["ENABLE_CSV_LOGGER"]):
        logger = CSVLogger(
            save_dir=stage1_save_dir,
            name="csv_logs",
        )
        enable_progress_bar = False

    trainer = Trainer(
        precision=precision,
        callbacks=[checkpoint_callback],
        max_epochs=int(run_config["STAGE1_MAX_EPOCHS"]),
        max_steps=-1,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=10,
        accumulate_grad_batches=int(
            run_config["ACCUMULATE_GRAD_BATCHES"]
        ),
        gradient_clip_val=float(run_config["GRAD_CLIP"]),
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )

    return trainer, logger if isinstance(logger, CSVLogger) else None


def save_stage1_manifests(
    stage1_save_dir: str,
    meta_save_dir: str,
    run_config: Dict[str, Any],
    train_entries: List[Dict[str, Any]],
) -> None:
    """
    Keeps statistics for the Stage1 three categories, manifest and audio-level.

    train_manifest.json is only an official-train
    Audio / diagnostic label entry.
    """
    ensure_dir(
        stage1_save_dir
    )
    ensure_dir(
        meta_save_dir
    )

    save_json(
        run_config,
        os.path.join(
            stage1_save_dir,
            "config.json",
        ),
    )
    save_json(
        run_config,
        os.path.join(
            meta_save_dir,
            "stage1_class_config.json",
        ),
    )

    save_text(
        "PROCESS-2 HC/MCI/Dementia 3-class protocol.\n"
        "Stage1 uses official_train only.\n"
        "official_test is not used.\n"
        "label_source=meta-info.csv diagnosis\n"
        "MMSE is not used.\n",
        os.path.join(
            meta_save_dir,
            "source_protocol.txt",
        ),
    )

    labels = np.asarray(
        [
            int(row["target_label"])
            for row in train_entries
        ],
        dtype=np.int64,
    )

    class_stats = {
        "task":
            "process2_hc_mci_dementia_3class",
        "class_mapping": {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
            "2": CLASS_NAMES[2],
        },

        "label_source":
            "PROCESS-2/meta-info.csv diagnosis",
        "split_source":
            "PROCESS-2/meta-info.csv Split",
        "mmse_used":
            False,
        "audio_task":
            "CTD",

        "train_audio_total":
            int(len(labels)),

        "train_class0":
            int(
                np.sum(labels == 0)
            ),
        "train_class1":
            int(
                np.sum(labels == 1)
            ),
        "train_class2":
            int(
                np.sum(labels == 2)
            ),

        "train_hc":
            int(
                np.sum(labels == 0)
            ),
        "train_mci":
            int(
                np.sum(labels == 1)
            ),
        "train_dementia":
            int(
                np.sum(labels == 2)
            ),

        "official_train_used":
            True,
        "official_test_used":
            False,
    }

    save_json(
        class_stats,
        os.path.join(
            meta_save_dir,
            "dataset_class_stats.json",
        ),
    )

    run_summary = {
        "task":
            "process2_hc_mci_dementia_3class",
        "dataset_option":
            run_config["DATASET_OPTION"],
        "layer_id":
            int(
                run_config["LAYER_ID"]
            ),
        "layer_name":
            str(
                run_config["LAYER_NAME"]
            ),
        "run_protocol":
            str(
                run_config["RUN_PROTOCOL_TAG"]
            ),
        "num_train_entries":
            int(
                len(train_entries)
            ),
        "repr_layers":
            run_config[
                "CURRENT_STAGE1_REPR_LAYERS"
            ],
        "repr_pool":
            run_config[
                "STAGE1_REPR_POOL"
            ],
        "unfreeze_encoder_layers":
            int(
                run_config[
                    "UNFREEZE_ENCODER_LAYERS"
                ]
            ),
        "resolved_unfreeze_encoder_layers":
            run_config[
                "STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS"
            ],
        "total_encoder_layers":
            int(
                run_config[
                    "TOTAL_ENCODER_LAYERS"
                ]
            ),
        "stage1_checkpoint_selection":
            "last",
    }

    save_json(
        run_summary,
        os.path.join(
            meta_save_dir,
            "stage1_run_summary.json",
        ),
    )

    if bool(
        run_config["SAVE_MANIFESTS"]
    ):
        save_json(
            {
                "task":
                    "process2_hc_mci_dementia_3class",
                "split":
                    "train",
                "official_train_used":
                    True,
                "official_test_used":
                    False,
                "class_mapping": {
                    "0": CLASS_NAMES[0],
                    "1": CLASS_NAMES[1],
                    "2": CLASS_NAMES[2],
                },
                "label_source":
                    "PROCESS-2/meta-info.csv diagnosis",
                "num_entries":
                    int(
                        len(train_entries)
                    ),
                "entries":
                    train_entries,
            },
            os.path.join(
                stage1_save_dir,
                "train_manifest.json",
            ),
        )


def run_one_stage1_experiment(
    dataset_option: str,
    layer_id: int,
) -> None:
    """Runs a Stage1 classification experiment for PROCESS-2 layer/unfree."""
    run_config = build_run_config(
        dataset_option=dataset_option,
        layer_id=layer_id,
    )

    stage1_save_dir = str(
        run_config["STAGE1_SAVE_DIR"]
    )
    meta_save_dir = str(
        run_config["META_SAVE_DIR"]
    )
    ensure_dir(
        stage1_save_dir
    )
    ensure_dir(
        meta_save_dir
    )

    print("=" * 100)
    print(
        "[Stage1-3CLASS] "
        f"DATASET={dataset_option} | "
        f"LAYER={layer_id} | "
        f"PROTOCOL={RUN_PROTOCOL_TAG}"
    )
    print(
        "[Stage1-3CLASS] "
        "labels: HC=0 | MCI=1 | Dementia=2"
    )
    print(
        "[Stage1-3CLASS] "
        "label source = meta-info.csv diagnosis"
    )
    print(
        "[Stage1-3CLASS] "
        "official_test used = False"
    )
    print(
        "[Stage1-3CLASS] "
        f"STAGE1_SAVE_DIR={stage1_save_dir}"
    )

    train_entries = (
        load_official_train_entries(
            dataset_option
        )
    )
    class_stats = (
        validate_classification_entries(
            train_entries,
            split_name="official_train",
            require_all_classes=True,
        )
    )

    # drop_last=False:
    # Each epoch uses all of the official train audio.
    steps_per_epoch = max(
        1,
        int(
            math.ceil(
                len(train_entries)
                / float(
                    max(
                        1,
                        BATCH_SIZE,
                    )
                )
            )
        ),
    )
    effective_training_steps = max(
        1,
        steps_per_epoch
        * int(
            STAGE1_MAX_EPOCHS
        ),
    )

    run_config["TRAINING_STEP"] = int(
        effective_training_steps
    )
    run_config[
        "STAGE1_CHECKPOINT_SELECTION"
    ] = "last"
    run_config["TRAIN_CLASS_STATS"] = (
        class_stats
    )

    print(
        "[Stage1-3CLASS][OFFICIAL-TRAIN] "
        f"train_wavs={len(train_entries)} | "
        f"HC={class_stats['class0']} | "
        f"MCI={class_stats['class1']} | "
        f"Dementia={class_stats['class2']} | "
        f"steps_per_epoch={steps_per_epoch} | "
        f"epochs={STAGE1_MAX_EPOCHS} | "
        "effective_training_steps="
        f"{effective_training_steps}"
    )

    save_stage1_manifests(
        stage1_save_dir=stage1_save_dir,
        meta_save_dir=meta_save_dir,
        run_config=run_config,
        train_entries=train_entries,
    )

    trainer, _logger = build_trainer(
        stage1_save_dir,
        run_config,
    )
    model = WhisperStage1Class(
        train_entries=train_entries,
        saving_path=stage1_save_dir,
        run_config=run_config,
    )

    resume_checkpoint = (
        run_config[
            "RESUME_CHECKPOINT"
        ]
    )
    if resume_checkpoint in [
        None,
        "",
        "None",
    ]:
        resume_checkpoint = None
    elif not os.path.isfile(
        str(resume_checkpoint)
    ):
        raise FileNotFoundError(
            "RESUME_CHECKPOINT Cannot initialise Evolution's mail component."
            f"{resume_checkpoint}"
        )

    trainer.fit(
        model,
        ckpt_path=resume_checkpoint,
    )

    last_checkpoint_path = os.path.join(
        stage1_save_dir,
        "last.ckpt",
    )

    if not os.path.isfile(
        last_checkpoint_path
    ):
        raise FileNotFoundError(
            "Stage1 Not found after the third class training. "
            f"last.ckpt:{last_checkpoint_path}"
        )

    save_text(
        "Stage1 PROCESS-2 HC/MCI/Dementia "
        "3-class classification finished successfully.\n"
        f"last_checkpoint={last_checkpoint_path}\n"
        "labels=HC:0,MCI:1,Dementia:2\n"
        "label_source=meta-info.csv diagnosis\n"
        "official_test_used=False\n"
        "checkpoint_selection=last\n",
        os.path.join(
            stage1_save_dir,
            "run_notes.txt",
        ),
    )

    print(
        "[Stage1-3CLASS] finished: "
        f"{last_checkpoint_path}"
    )

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# 10) Main entrance
# ============================================================
def main() -> None:
    """
    Through:
        PROCESS-2
        -> UNFREEZE_ENCODER_LAYERS_LIST
        -> LAYERS_TO_RUN
    """
    global DATASET_OPTION
    global UNFREEZE_ENCODER_LAYERS

    seed_everything_for_repro(
        SEED
    )

    if DATASET_OPTIONS != [
        "PROCESS-2"
    ]:
        raise ValueError(
            "CPFT_3class_stage1.py Allow Only "
            f"PROCESS-2,Current ={DATASET_OPTIONS}"
        )

    if len(
        UNFREEZE_ENCODER_LAYERS_LIST
    ) == 0:
        raise ValueError(
            "UNFREEZE_ENCODER_LAYERS_LIST It's empty."
        )

    if len(
        LAYERS_TO_RUN
    ) == 0:
        raise ValueError(
            "LAYERS_TO_RUN It's empty."
        )

    if not os.path.isfile(
        PROCESS2_META_PATH
    ):
        raise FileNotFoundError(
            "PROCESS2_META_PATH Cannot initialise Evolution's mail component."
            f"{PROCESS2_META_PATH}"
        )

    print("[CONFIG]")
    print(
        "  TASK = "
        "PROCESS-2 HC/MCI/Dementia 3-class"
    )
    print(
        "  CLASS_PROJECT_ROOT =",
        CLASS_PROJECT_ROOT,
    )
    print(
        "  PROJECT_ROOT =",
        PROJECT_ROOT,
    )
    print(
        "  CPFT_ROOT =",
        CPFT_ROOT,
    )
    print(
        "  PROCESS2_META_PATH =",
        PROCESS2_META_PATH,
    )
    print(
        "  DATASET_OPTIONS =",
        DATASET_OPTIONS,
    )
    print(
        "  CLASS_NAMES =",
        CLASS_NAMES,
    )
    print(
        "  RUN_PROTOCOL_TAG =",
        RUN_PROTOCOL_TAG,
    )
    print(
        "  LAYERS_TO_RUN =",
        LAYERS_TO_RUN,
    )
    print(
        "  MODEL_ROOT_TAG =",
        MODEL_ROOT_TAG,
    )
    print(
        "  UNFREEZE_ENCODER_LAYERS_LIST =",
        UNFREEZE_ENCODER_LAYERS_LIST,
    )
    print(
        "  WHISPER_MODEL_NAME =",
        WHISPER_MODEL_NAME,
    )
    print(
        "  STAGE1_MAX_EPOCHS =",
        STAGE1_MAX_EPOCHS,
    )
    print(
        "  LAMBDA_UTT =",
        LAMBDA_UTT,
    )
    print(
        "  LAMBDA_FRAME =",
        LAMBDA_FRAME,
    )
    print(
        "  official_test used = False"
    )
    print(
        "  class weighting = False"
    )

    for dataset_option in [
        str(x)
        for x in DATASET_OPTIONS
    ]:
        if dataset_option not in DATASETS:
            raise ValueError(
                "Unsupported DATASET_OPTION="
                f"{dataset_option}"
            )

        for unfreeze_count in [
            int(x)
            for x in (
                UNFREEZE_ENCODER_LAYERS_LIST
            )
        ]:
            DATASET_OPTION = (
                dataset_option
            )
            UNFREEZE_ENCODER_LAYERS = int(
                unfreeze_count
            )

            print(
                "\n"
                + "=" * 100
            )
            print(
                "[Stage1 3CLASS BATCH ITEM]"
            )
            print(
                "  DATASET_OPTION =",
                DATASET_OPTION,
            )
            print(
                "  UNFREEZE_ENCODER_LAYERS =",
                UNFREEZE_ENCODER_LAYERS,
            )
            print(
                "=" * 100
            )

            for layer_id in [
                int(x)
                for x in LAYERS_TO_RUN
            ]:
                run_one_stage1_experiment(
                    dataset_option=(
                        DATASET_OPTION
                    ),
                    layer_id=layer_id,
                )


if __name__ == "__main__":
    main()
