# -*- coding: utf-8 -*-
"""
Whisper ADD 2 training script for freezing/ fine-tuning features + LRU
============================================================

Definition of tasks
--------
0:Health/Creation Group (cc)
1:Disease Group (cd)

Datasets
------
- ADReSS-IS2020
- ADReSSo21
- Pitt
- PROCESS-2

Data protocol
--------
- official train:(a) All for training;
- official test:Each epoch calculates Accuracy, F1-score, ROC-AUC,
  For monitoring and mapping only;
- official test Do not participate in early stopping, epoch selection, checkpoint selection,
  checkpoint Recovery, adaptation of learning rates or any training decisions;
- Do not implement official train internal division by 50%.

Training protocol (programme B)
-----------------
- Maximum training 10 epoch;
- early stopping Monitor only the epoch train_loses of the financial train;
- patience=3,min_delta=1e-4;
- restore_best=False;
- Final use of the final round of models at the time of actual cessation;
- Adam,No weight decay, no gradient clipping;
- warmup + cosine decay;
- Officially multiple random seeds: 401, 402, 403, 404, 405.

Run Example
--------
conda activate audiojax_gpu
python LRU_2class.py

Attention.
----
1. This script is responsible only for the LRU classification and does not include other downstream models.
2. Input features can be generated either by frozen.py or by CPFT_2class_extract.py.
3. The index to be kept in the two export scripts is usually [N, 2]:
   Column 0 is audio_id, column 1 is text_id;
   Only column 0 will be extracted from this version as the index to the polymer downstream audio-level.
4. The Python process should be reactivated after the amendment of CUDA_VISIBLE_DEVICES.
"""

# ============================================================
# 0. Light import and operation environment
# ============================================================
# The parent process does not import JAX/PyTorch in advance when it discovers a task and gives birth to a child.
# A parent process can be avoided from taking advantage of GPU.
import os
import sys
import re
import json
import time
import pickle
import random
import subprocess
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List, Optional

# Default use of number 5 can be found at GPU; the server can also be overwritten in advance of the command line when running.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "5")

# Closes one-time pre-allocation of JAX to full storage, allowing for the continuous operation of multiple data sets.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Limiting the number of CPU threads and reducing the number of collusive thread competitions at multiple subprocesses.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# ============================================================
# 1. Global configuration: usually only needs to modify here
# ============================================================
PROJECT_ROOT = "/home/user/Alzheimer_code_jax_class"
FEATURE_ROOT = os.path.join(PROJECT_ROOT, "feature_tensor")
OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "outputs")
DATA_DIR = os.path.join(PROJECT_ROOT, "data_dir")

# Classified version presplit Cache is completely isolated from old items.
PRESPLIT_CACHE_NAME = "UEA"
FORCE_REBUILD_PRESPLIT_CACHE = False

# Clear four data sets that run.
DATASET_OPTIONS = ["ADReSS-IS2020"]


def _canonical_dataset_option(value: str) -> str:
    """Map the public ADReSS name to the existing internal dataset key."""
    value = str(value).strip()
    return "ADReSS-IS2020" if value == "ADReSS" else value

# ============================================================
# Directory of upstream characteristics
# ============================================================
# CPFT Features:
#     feature_tensor/<DATASET>/ft_CPFT_lr5e-5_epo5/singleX_unfreeZ/
FEATURE_MODEL_NAME = "ft_CPFT_lr5e-5_epo5"

# For Freezing Whisper read:
# FEATURE_MODEL_NAME = "whisper_small"


# ============================================================
# Whisper Show layer filters
# ============================================================
# For example:
#     [12]      -> single12 or single12_unfreeZ remain;
#     [11, 12]  -> At the same time, the levels 11 and 12 shall be maintained.
SINGLE_LAYERS = [12]


# ============================================================
# Add: Filter unfree layers only for "modified features"
# ============================================================
# 1. Leave space []:
#       Do not filter on unfree layers;
#       single12_unfree1,single12_unfree2 We'll keep it.
#
# 2. Set to [1]:
#       Only *_unfree1 will be retained.
#
# 3. Joint use with SINGLE_LAYERS:
#       SINGLE_LAYERS = [12]
#       UNFREE_LAYER_NUM = [1]
#
#    Final matches only:
#       single12_unfree1
#
# 4. The Frozen Identity Directory single12 does not contain "_unfreeZ",
#    This variable will not be removed by the filter even if it is not empty.
UNFREE_LAYER_NUM = [1,2,3,4,5,6,7,8,9,10,11,12]

# For the current script, only LRU will be retained.
MODEL_NAME = "lru"

# The parent process creates an independent subprocess for each "Dataset x Feature Layer ".
# This will allow the full release of JAX compile caches and GPUs after the end of the mission.
ENABLE_SUBPROCESS_BATCH = True

# Single random torrent debug switch.
RUN_SINGLE_SEED_400 = False
SINGLE_SEED = 400

# Formally multi-random seeds.
RUN_MULTI_SEEDS = True
MULTI_SEEDS = [401, 402, 403, 404, 405]

# Training protocol.
BATCH_SIZE = 16
TARGET_EPOCHS = 10
LEARNING_RATE = 1e-4
WARMUP_EPOCHS = 2
COSINE_ALPHA = 0.05

USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_RESTORE_BEST = False
EARLY_STOPPING_MONITOR = "train_loss"

# Classification and aggregation configuration.
NUM_CLASSES = 2
POSITIVE_CLASS_ID = 1
CLASSIFICATION_THRESHOLD = 0.5
AUDIO_AGG_METHOD = "mean_positive_probability"

# Whether each segment sequence feature is standardized along a time axis.
# To maintain the current formal baseline, the default is closed.
FEATURE_STANDARDIZE = False

# LRU + Classify the header configuration.
READOUT_POOL = "mean"
LRU_CONFIG = {
    "num_blocks": 2,
    "hidden_dim": 128,
    "ssm_dim": 64,
    "repr_dim": 128,
    "mlp_hidden_dim": 128,
    "pool": READOUT_POOL,
}

# The fixed semantic field for option B is included in all outcome documents.
SELECTION_POLICY = "last_epoch_or_train_loss_early_stop"
SELECTED_BY = "last_epoch_or_train_loss_early_stop_no_test_selection"
OFFICIAL_TEST_SEMANTICS = "monitor_only_not_used_for_selection"


# ============================================================
# 2. Generic small tool
# ============================================================
def _is_blank(value) -> bool:
    """Determines whether the configuration value is empty."""
    return value is None or str(value).strip().lower() in {"", "none", "null"}


def _natural_key(path_value: str):
    """Sort naturally, ensure that single2 is located before single10."""
    name = Path(path_value).name
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.findall(r"\d+|\D+", name)
    ]


def _extract_single_layer(selection_name: str) -> Optional[int]:
    """Parsing layer numbers from a directory name like single12."""
    match = re.match(r"^single(\d+)(?:_|$)", str(selection_name), flags=re.IGNORECASE)
    return None if match is None else int(match.group(1))


def _extract_unfree_layer_num(selection_name: str) -> Optional[int]:
    """
    Parsing the number of frozen layers from a fine-tuned feature selection directory name.

    Example:
        single12_unfree1            -> 1
        single12_unfree7            -> 7
        single12_unfree7_ks32_64    -> 7
        single12                    -> None
    """
    match = re.search(
        r"(?:^|_)unfree(\d+)(?:_|$)",
        str(selection_name),
        flags=re.IGNORECASE,
    )
    return None if match is None else int(match.group(1))


def _selection_matches_unfree_nums(
    selection_name: str,
    allowed_unfree_nums,
) -> bool:
    """
    selection is judged to be consistent with UNFREE_LAYER_NUM.

    Rules:
    1. UNFREE_LAYER_NUM=[]:
       Close unfree filter, all selection remain.

    2. selection Does not include "_unfreeZ":
       is considered to be a freezing feature, e.g. single12, independent of this filter.

    3. selection Includes "_unfreeZ":
       Only if Z is located in UNFREE_LAYER_NUM.
    """
    allowed_values = {
        int(value)
        for value in allowed_unfree_nums
    }

    if len(allowed_values) == 0:
        return True

    unfree_num = _extract_unfree_layer_num(selection_name)

    # Freezing features do not have unfree fields and are not filtered by fine-tuning layers.
    if unfree_num is None:
        return True

    return int(unfree_num) in allowed_values


def _parse_feature_file_tag(feature_train_path: str) -> Optional[str]:
    """Parsing <tag> from feature_<tag>_train.pt."""
    match = re.match(r"^feature_(.+)_train\.pt$", Path(feature_train_path).name)
    return None if match is None else str(match.group(1))


def _feature_bundle_paths(feature_dir: str, file_tag: str) -> Dict[str, str]:
    """Generates a complete path to a group of taxonomic features files."""
    return {
        "feature_train": os.path.join(feature_dir, f"feature_{file_tag}_train.pt"),
        "feature_test": os.path.join(feature_dir, f"feature_{file_tag}_test.pt"),
        "label_train": os.path.join(feature_dir, f"labels1_{file_tag}_train.pt"),
        "label_test": os.path.join(feature_dir, f"labels2_{file_tag}_test.pt"),
        "index_train": os.path.join(feature_dir, f"indices_{file_tag}_train.pt"),
        "index_test": os.path.join(feature_dir, f"indices_{file_tag}_test.pt"),
    }


def _has_complete_feature_bundle(feature_dir: str, file_tag: str) -> bool:
    """Checks the completeness of the six.pt input files."""
    return all(os.path.exists(path) for path in _feature_bundle_paths(feature_dir, file_tag).values())


def _json_safe(value):
    """Convert objects such as NumPy/Path to a normal type that can be saved to JSON."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return float(value) if value == value and abs(value) != float("inf") else None
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def save_json(obj, path: str):
    """Save JSON with UTF-8."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(_json_safe(obj), file, ensure_ascii=False, indent=2)


# ============================================================
# 3. Classified Characteristic Mission Discovery with Father / Subprocess Control
# ============================================================
def discover_feature_contexts() -> List[Dict[str, str]]:
    """
    Classification features are found in the following directories:

        feature_tensor/<dataset>/<FEATURE_MODEL_NAME>/<selection>/

    selection These could include:
        single12
        single12_unfree1
        single12_unfree7_ks32_64

    Filter rules:
    1. SINGLE_LAYERS Control singleX;
    2. UNFREE_LAYER_NUM Control only fine-tuned features containing _unfreeZ;
    3. Freeze table singleX does not contain _unfreeZ and is not filtered under unfree;
    4. Six.pt documents must be complete.
    """
    allowed_layers = {
        int(layer)
        for layer in SINGLE_LAYERS
    }
    allowed_unfree_nums = {
        int(value)
        for value in UNFREE_LAYER_NUM
    }

    print(
        "[DISCOVER CONFIG] "
        f"FEATURE_MODEL_NAME={FEATURE_MODEL_NAME} | "
        f"SINGLE_LAYERS={sorted(allowed_layers)} | "
        f"UNFREE_LAYER_NUM={sorted(allowed_unfree_nums)}"
    )

    contexts: List[Dict[str, str]] = []

    for dataset_name in [_canonical_dataset_option(value) for value in DATASET_OPTIONS]:
        dataset_root = os.path.join(
            FEATURE_ROOT,
            dataset_name,
            FEATURE_MODEL_NAME,
        )

        if not os.path.isdir(dataset_root):
            print(
                "[DISCOVER] The feature root directory does not exist. Skip:"
                f"{dataset_root}"
            )
            continue

        feature_train_files = sorted(
            Path(dataset_root).rglob(
                "feature_*_train.pt"
            ),
            key=lambda path: _natural_key(str(path)),
        )

        for feature_train_path in feature_train_files:
            feature_dir = str(
                feature_train_path.parent
            )
            selection_name = (
                feature_train_path.parent.name
            )

            # ----------------------------------------------------
            # 1. Filter on single Layer
            # ----------------------------------------------------
            layer_id = _extract_single_layer(
                selection_name
            )

            if (
                layer_id is None
                or layer_id not in allowed_layers
            ):
                continue

            # ----------------------------------------------------
            # 2. Filter on unfree layers only for fine-tuning features
            # ----------------------------------------------------
            if not _selection_matches_unfree_nums(
                selection_name=selection_name,
                allowed_unfree_nums=UNFREE_LAYER_NUM,
            ):
                print(
                    "[DISCOVER] unfree Not in list allowed, skip:"
                    f"{selection_name}"
                )
                continue

            unfree_num = _extract_unfree_layer_num(
                selection_name
            )

            file_tag = _parse_feature_file_tag(
                str(feature_train_path)
            )
            if _is_blank(file_tag):
                continue

            if not _has_complete_feature_bundle(
                feature_dir,
                str(file_tag),
            ):
                print(
                    "[DISCOVER] Six files are incomplete, skipping:"
                    f"{feature_dir} | tag={file_tag}"
                )
                continue

            contexts.append({
                "dataset": dataset_name,
                "feature_model": FEATURE_MODEL_NAME,
                "selection": selection_name,
                "layer": str(layer_id),
                "unfree": (
                    ""
                    if unfree_num is None
                    else str(unfree_num)
                ),
                "feature_dir": feature_dir,
                "file_tag": str(file_tag),
            })

    contexts.sort(
        key=lambda item: (
            [_canonical_dataset_option(value) for value in DATASET_OPTIONS].index(
                item["dataset"]
            ),
            int(item["layer"]),
            (
                -1
                if _is_blank(item.get("unfree"))
                else int(item["unfree"])
            ),
            item["selection"],
            item["file_tag"],
        )
    )
    return contexts

def launch_batch_children_if_needed() -> bool:
    """
    The parent process entrance.

    Returning to True indicates that the current process is a parent process and that all sub-tasks have been completed,
    The caller should withdraw immediately; return to False indicates that individual tasks should continue for the time being.
    """
    is_child = os.getenv("LRU_CLASS_CHILD", "0") == "1"
    if is_child or not ENABLE_SUBPROCESS_BATCH:
        return False

    contexts = discover_feature_contexts()
    if len(contexts) == 0:
        raise FileNotFoundError(
            "No functional classification features have been found. Please run frozen.py or CPFT_2class_extract.py and check: \n"
            f"{FEATURE_ROOT}/<DATASET>/{FEATURE_MODEL_NAME}/<selection>/"
        )

    print("=" * 100)
    print(f"[BATCH] Discovered {len(contexts)} One mission.")
    for index, context in enumerate(contexts, start=1):
        print(
            f"[{index:02d}] dataset={context['dataset']} | "
            f"selection={context['selection']} | "
            f"unfree={context.get('unfree', '') or '-'} | "
            f"tag={context['file_tag']}"
        )
    print("=" * 100)

    script_path = str(Path(__file__).resolve())

    for index, context in enumerate(contexts, start=1):
        env = os.environ.copy()
        env.update({
            "LRU_CLASS_CHILD": "1",
            "LRU_CLASS_DATASET": context["dataset"],
            "LRU_CLASS_FEATURE_MODEL": context["feature_model"],
            "LRU_CLASS_SELECTION": context["selection"],
            "LRU_CLASS_UNFREE_LAYER": context.get("unfree", ""),
            "LRU_CLASS_FEATURE_DIR": context["feature_dir"],
            "LRU_CLASS_FILE_TAG": context["file_tag"],
        })

        print("\n" + "#" * 100)
        print(
            f"[BATCH {index}/{len(contexts)}] Start:"
            f"{context['dataset']} / {context['selection']}"
        )
        print("#" * 100)

        completed = subprocess.run(
            [sys.executable, script_path],
            env=env,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Submission failed, returncode={completed.returncode}:{context}"
            )

    print("\n[BATCH] All data sets and layers have been configured.")
    return True


if launch_batch_children_if_needed():
    sys.exit(0)


# ============================================================
# 4. Current sub-task configuration
# ============================================================
# In sub-process mode, these values are passed by parent processes via environmental variables.
# When the batch mode is closed, the first data set and the first layer are used as a fallback.
ACTIVE_DATASET = _canonical_dataset_option(os.getenv("LRU_CLASS_DATASET", DATASET_OPTIONS[0]))
ACTIVE_FEATURE_MODEL = os.getenv("LRU_CLASS_FEATURE_MODEL", FEATURE_MODEL_NAME)

# Default selection when closing batch mode:
# - UNFREE_LAYER_NUM Non-hours, use by default the first unfree in the list;
# - Saves space by using singleX by default.
DEFAULT_SELECTION = (
    f"single{SINGLE_LAYERS[0]}_unfree{UNFREE_LAYER_NUM[0]}"
    if len(UNFREE_LAYER_NUM) > 0
    else f"single{SINGLE_LAYERS[0]}"
)

ACTIVE_SELECTION = os.getenv(
    "LRU_CLASS_SELECTION",
    DEFAULT_SELECTION,
)
_PARSED_ACTIVE_UNFREE_LAYER = _extract_unfree_layer_num(
    ACTIVE_SELECTION
)
_ACTIVE_UNFREE_ENV = os.getenv(
    "LRU_CLASS_UNFREE_LAYER",
    "",
).strip()

if _is_blank(_ACTIVE_UNFREE_ENV):
    ACTIVE_UNFREE_LAYER = _PARSED_ACTIVE_UNFREE_LAYER
else:
    ACTIVE_UNFREE_LAYER = int(
        _ACTIVE_UNFREE_ENV
    )

    if (
        _PARSED_ACTIVE_UNFREE_LAYER is not None
        and int(ACTIVE_UNFREE_LAYER)
        != int(_PARSED_ACTIVE_UNFREE_LAYER)
    ):
        raise ValueError(
            "unfree layers that the parent process transmits do not match the selection directory name:"
            f"env={ACTIVE_UNFREE_LAYER}, "
            f"selection={ACTIVE_SELECTION}, "
            f"parsed={_PARSED_ACTIVE_UNFREE_LAYER}"
        )

ACTIVE_FEATURE_DIR = os.getenv(
    "LRU_CLASS_FEATURE_DIR",
    os.path.join(FEATURE_ROOT, ACTIVE_DATASET, ACTIVE_FEATURE_MODEL, ACTIVE_SELECTION),
)
ACTIVE_FILE_TAG = os.getenv(
    "LRU_CLASS_FILE_TAG",
    f"whisper_single{SINGLE_LAYERS[0]}_0.3_30s",
)

OUTPUT_SELECTION_ROOT = os.path.join(
    OUTPUT_ROOT,
    ACTIVE_DATASET,
    ACTIVE_FEATURE_MODEL,
    MODEL_NAME,
    ACTIVE_SELECTION,
)
os.makedirs(OUTPUT_SELECTION_ROOT, exist_ok=True)

INPUT_PATHS = _feature_bundle_paths(ACTIVE_FEATURE_DIR, ACTIVE_FILE_TAG)

print("\n" + "=" * 100)
print("Current Category II Tasks")
print("=" * 100)
print("ACTIVE_DATASET       =", ACTIVE_DATASET)
print("ACTIVE_FEATURE_MODEL =", ACTIVE_FEATURE_MODEL)
print("ACTIVE_SELECTION     =", ACTIVE_SELECTION)
print("ACTIVE_UNFREE_LAYER  =", ACTIVE_UNFREE_LAYER)
print("ACTIVE_FEATURE_DIR   =", ACTIVE_FEATURE_DIR)
print("ACTIVE_FILE_TAG      =", ACTIVE_FILE_TAG)
print("OUTPUT_ROOT          =", OUTPUT_SELECTION_ROOT)
for name, path in INPUT_PATHS.items():
    print(f"[{name}] {path} | exists={os.path.exists(path)}")

missing_paths = [path for path in INPUT_PATHS.values() if not os.path.exists(path)]
if missing_paths:
    raise FileNotFoundError("The following input file does not exist: \n" + "\n".join(missing_paths))


# ============================================================
# 5. Weight level dependency: import only in actual sub-task
# ============================================================
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch

import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from data_dir.dataloaders import Dataloader
from models.LRU import LRU

print("JAX devices =", jax.devices())


# ============================================================
# 6. Random Source and Data Read
# ============================================================
def set_global_seed(seed: int):
    """
    Sets random seeds for Python, NumPy, PyTorch.

    JAX , the randomity is controlled by the visible PRNGKey; other libraries are kept synchronized here.
    Prevents the introduction of unfixed random behaviour in subsequent assistive codes.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_numpy(value, dtype=None):
    """Converts Tensor/list/ndarray to NumPy."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    if dtype is not None:
        value = value.astype(dtype)
    return value


def standardize_sequence_features(features: np.ndarray) -> np.ndarray:
    """
    Standardized for each segment axis.

    Enter shape: [N, T, C]
    Statistical dimensions: axis=1, i.e., independent statistics for each sample, for each signature channel.
    """
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return ((features - mean) / (std + 1e-8)).astype(np.float32)


def ensure_binary_labels(labels, split_name: str) -> np.ndarray:
    """Mark the label into int64 one-dimensional arrays and verify strictly that only 0 and 1 are present."""
    labels = np.asarray(labels).reshape(-1)

    if not np.all(np.isfinite(labels)):
        raise ValueError(f"{split_name} The label contains NaN or Inf.")

    integer_labels = labels.astype(np.int64)
    if not np.allclose(labels, integer_labels):
        raise ValueError(f"{split_name} Label contains integer values:{np.unique(labels).tolist()}")

    unique_values = sorted(np.unique(integer_labels).tolist())
    if not set(unique_values).issubset({0, 1}):
        raise ValueError(f"{split_name} Label is not 0/1:{unique_values}")

    if set(unique_values) != {0, 1}:
        raise ValueError(
            f"{split_name} There are no two categories at the same time, the current label ={unique_values}."
        )

    return integer_labels


def validate_audio_label_consistency(audio_ids, labels, split_name: str):
    """Confirms that all of the text of the same audio_id has exactly the same real category."""
    audio_ids = np.asarray(audio_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    if len(audio_ids) != len(labels):
        raise ValueError(
            f"{split_name}: audio_ids Inconsistent with labels length:"
            f"{len(audio_ids)} != {len(labels)}"
        )

    for audio_id in np.unique(audio_ids):
        values = np.unique(labels[audio_ids == int(audio_id)])
        if len(values) != 1:
            raise ValueError(
                f"{split_name}: audio_id={audio_id} The segment label is inconsistent:"
                f"{values.tolist()}"
            )


def summarize_class_distribution(audio_ids, labels, split_name: str) -> Dict[str, int]:
    """Counts the number of categories at the level of subment and audio."""
    audio_ids = np.asarray(audio_ids, dtype=np.int64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    audio_labels = []
    for audio_id in sorted(np.unique(audio_ids).tolist()):
        audio_labels.append(int(labels[audio_ids == int(audio_id)][0]))
    audio_labels = np.asarray(audio_labels, dtype=np.int64)

    result = {
        "segment_total": int(len(labels)),
        "segment_class0": int(np.sum(labels == 0)),
        "segment_class1": int(np.sum(labels == 1)),
        "audio_total": int(len(audio_labels)),
        "audio_class0": int(np.sum(audio_labels == 0)),
        "audio_class1": int(np.sum(audio_labels == 1)),
    }

    print(
        f"[{split_name}] segment: total={result['segment_total']} | "
        f"class0={result['segment_class0']} | class1={result['segment_class1']}"
    )
    print(
        f"[{split_name}] audio:   total={result['audio_total']} | "
        f"class0={result['audio_class0']} | class1={result['audio_class1']}"
    )
    return result


def get_presplit_cache_dir() -> str:
    """
    Constructs the only presplit cache directory for the current task.

    Path structure:
        data_dir/processed/UEA/<dataset>/<feature_model>/lru/<selection>/<file_tag>/

    Downstream model names are written to the cache path, which avoids other experiments and overlays the same batch of pkl.
    """
    return os.path.join(
        DATA_DIR,
        "processed",
        PRESPLIT_CACHE_NAME,
        ACTIVE_DATASET,
        ACTIVE_FEATURE_MODEL,
        MODEL_NAME,
        ACTIVE_SELECTION,
        ACTIVE_FILE_TAG,
    )


def save_pickle(value, path: str):
    """Saves pkl and automatically creates a parent directory."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(value, file)


def load_pickle(path: str):
    """Read pkl."""
    with open(path, "rb") as file:
        return pickle.load(file)

def trusted_torch_load(file_path: str):
    """
    Loads.pt documents generated by frozen.py or CPFT_2class_extract.py for the current item.

    PyTorch 2.6 Start, toch.load()'s weights_only parameters by default from False
    becomes True. weights_only=True applies mainly to purely model weights and Tensor,
    However, the item indices_*.pt contains NumPy ndarray and would therefore:

        Unsupported global:
        numpy._core.multiarray._reconstruct

    The current document is all derived from the project ' s own features, and the source is credible.
    This allows for a visible setting weights_only=False.

    Also retain compatibility with the old version of PyTorch:
    The old version torch.load() returns to its original formulation if the weights_only parameter is not supported.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Could not find.pt file to load: \n{file_path}"
        )

    try:
        return torch.load(
            file_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        # Compatibility with old version PyTorch of the weights_only parameter.
        return torch.load(
            file_path,
            map_location="cpu",
        )


def extract_audio_ids(index_object, split_name: str) -> np.ndarray:
    """
    Draws a dimension audio_id from the segment index saved by the export script.

    Standard index format
    ------------
    frozen.py Save with CPFT_2class_extract.py for each segment:

        (audio_id, segment_id)

    Thus, index objects usually have the following shapes:

        [N, 2]

    Of which:
        Column 0: audio_id, which original audio is the current segment;
        Column 1: segment_id for the current segment segment number in the audio.

    Downstream Use
    --------
    LRU Projection of each segment, followed by audio_id for the same audio
    segment The probability of disease is averaged. Therefore, only column 0 is retained here.

    Significant notes
    --------
    reshape(-1) cannot be used directly for the [N, 2] index. Otherwise the length of the index will be from N
    Become 2N and mix audio_id and segment_id into the same one-dimensional array.
    This resulted in inconsistent numbers of Feature/label/index.

    Compatibility Format
    --------
    - [N, 2] or [N, K]: extract column 0 as audio_id;
    - [N]:Considers a dimension of history that has been saved directly audio_id;
    - Other dimensions: reporting errors directly and avoiding silently producing erroneous aggregate results.
    """
    index_array = to_numpy(
        index_object,
        np.int64,
    )

    if index_array.ndim == 2:
        if index_array.shape[1] < 1:
            raise ValueError(
                f"{split_name} _Other Organiser"
                f"shape={index_array.shape}"
            )

        # Column 0 in the standard format is audio_id and column 1 is segment_id.
        audio_ids = index_array[:, 0]

    elif index_array.ndim == 1:
        # Compatible historical version: Indexed files have been saved directly in one dimension audio_id.
        audio_ids = index_array

    else:
        raise ValueError(
            f"{split_name} index dimensions are invalid: shape={index_array.shape}."
            "The expected format is [N] or [N, 2]."
        )

    audio_ids = np.asarray(
        audio_ids,
        dtype=np.int64,
    ).reshape(-1)

    if len(audio_ids) == 0:
        raise ValueError(
            f"{split_name} The audio_id is empty."
        )

    if not np.all(np.isfinite(audio_ids)):
        raise ValueError(
            f"{split_name} audio_id contains NaN or Inf."
        )

    print(
        f"[INDEX] {split_name} | "
        f"Original index shape={index_array.shape} | "
        f"After extraction audio_id shape={audio_ids.shape} | "
        f"Audio only ={len(np.unique(audio_ids))}"
    )

    return audio_ids


def convert_feature_pt_to_presplit_cache() -> str:
    """
    Converts six.pt files generated by the export script to presplit pkl.

    Map rules:
    - X_train/y_train/idx_train = official train;
    - X_val/y_val/idx_val       = official test Monitoring aliases;
    - X_test/y_test/idx_test    = official test The official test alias.

    val Same data content as test, but official test never participates in the early stop or model selection.
    """
    cache_dir = get_presplit_cache_dir()
    expected_names = [
        "X_train.pkl", "X_val.pkl", "X_test.pkl",
        "y_train.pkl", "y_val.pkl", "y_test.pkl",
        "idx_train.pkl", "idx_val.pkl", "idx_test.pkl",
        "classification_presplit_summary.json",
    ]

    cache_complete = all(os.path.exists(os.path.join(cache_dir, name)) for name in expected_names)
    if cache_complete and not FORCE_REBUILD_PRESPLIT_CACHE:
        print(f"[CACHE] Reuse existing category presplit cache:{cache_dir}")
        return cache_dir

    print(f"[CACHE] Build category presplit cache:{cache_dir}")
    os.makedirs(cache_dir, exist_ok=True)
########################################################################################
    # ============================================================
    # Reading official train/test feature files generated by frozen.py or CPFT_2class_extract.py
    # ============================================================
    # Do not use default torch.load():
    # PyTorch 2.6 Default to weights_only=True and possible saving in indices files
    # NumPy ndarray,The default mode cannot be counter-sequenced.
    #
    # trusted_torch_load() Prominent use of self-generated, source-credible documents for the project
    # weights_only=False,At the same time, it is compatible with the old version PyTorch.

    # official train/test The Whisper feature.
    X_train = to_numpy(
        trusted_torch_load(
            INPUT_PATHS["feature_train"]
        ),
        np.float32,
    )

    X_test = to_numpy(
        trusted_torch_load(
            INPUT_PATHS["feature_test"]
        ),
        np.float32,
    )

    # official train/test The second label.
    # ensure_binary_labels() The labels will be further checked for 0 and 1 only.
    y_train = ensure_binary_labels(
        to_numpy(
            trusted_torch_load(
                INPUT_PATHS["label_train"]
            )
        ),
        "official_train",
    )

    y_test = ensure_binary_labels(
        to_numpy(
            trusted_torch_load(
                INPUT_PATHS["label_test"]
            )
        ),
        "official_test",
    )

    # ============================================================
    # Extract audio_id from the 2-D segment index saved by the export feature script
    # ============================================================
    # Standard index shape [N, 2]:
    #
    #     indices[:, 0] = audio_id
    #     indices[:, 1] = segment_id
    #
    # When audio-level probability aggregation is performed downstream, only audio_id is required.
    # Not directly reshape(-1), otherwise [N, 2] will be wrongly expanded to 2N,
    # And mix audio_id with a second_id.
    idx_train = extract_audio_ids(
        trusted_torch_load(
            INPUT_PATHS["index_train"]
        ),
        "official_train",
    )

    idx_test = extract_audio_ids(
        trusted_torch_load(
            INPUT_PATHS["index_test"]
        ),
        "official_test",
    )
##############################################################################
    if X_train.ndim != 3 or X_test.ndim != 3:
        raise ValueError(
            "LRU The signature must be [N, T, C]:"
            f"train={X_train.shape}, test={X_test.shape}"
        )

    for split_name, X, y, idx in [
        ("official_train", X_train, y_train, idx_train),
        ("official_test", X_test, y_test, idx_test),
    ]:
        if not (len(X) == len(y) == len(idx)):
            raise ValueError(
                f"{split_name} _Other Organiser"
                f"{len(X)}, {len(y)}, {len(idx)}"
            )
        validate_audio_label_consistency(idx, y, split_name)

    # official test Saves compatible aliases val monitor and ultimately test.
    save_pickle(X_train.astype(np.float32), os.path.join(cache_dir, "X_train.pkl"))
    save_pickle(X_test.astype(np.float32), os.path.join(cache_dir, "X_val.pkl"))
    save_pickle(X_test.astype(np.float32), os.path.join(cache_dir, "X_test.pkl"))

    save_pickle(y_train.astype(np.int64), os.path.join(cache_dir, "y_train.pkl"))
    save_pickle(y_test.astype(np.int64), os.path.join(cache_dir, "y_val.pkl"))
    save_pickle(y_test.astype(np.int64), os.path.join(cache_dir, "y_test.pkl"))

    save_pickle(idx_train.astype(np.int64), os.path.join(cache_dir, "idx_train.pkl"))
    save_pickle(idx_test.astype(np.int64), os.path.join(cache_dir, "idx_val.pkl"))
    save_pickle(idx_test.astype(np.int64), os.path.join(cache_dir, "idx_test.pkl"))

    train_distribution = summarize_class_distribution(idx_train, y_train, "official_train")
    test_distribution = summarize_class_distribution(idx_test, y_test, "official_test")

    summary = {
        "task": "ad_binary_classification",
        "class_mapping": {"0": "healthy_control", "1": "ad_or_disease"},
        "source": "classification_feature_bundle_pt",
        "dataset": ACTIVE_DATASET,
        "feature_model": ACTIVE_FEATURE_MODEL,
        "feature_selection": ACTIVE_SELECTION,
        "unfree_layer_num": ACTIVE_UNFREE_LAYER,
        "unfree_layer_filter": list(UNFREE_LAYER_NUM),
        "feature_file_tag": ACTIVE_FILE_TAG,
        "official_train_shape": list(X_train.shape),
        "official_test_shape": list(X_test.shape),
        "class_distribution": {
            "train": train_distribution,
            "test": test_distribution,
        },
        "split_protocol": {
            "train": "official_train",
            "val": "official_test_monitor_alias",
            "test": "official_test_final_alias",
        },
        "index_semantics": {
            "source_format": "[N,2] -> (audio_id, segment_id), or legacy [N]",
            "used_by_downstream": "first_column_audio_id_only",
            "segment_id_used_for_audio_aggregation": False,
        },
        "input_paths": INPUT_PATHS,
    }
    save_json(summary, os.path.join(cache_dir, "classification_presplit_summary.json"))
    return cache_dir


def load_classification_dataset() -> SimpleNamespace:
    """Read from category presplit pkl official train/test and construct Dataloader."""
    cache_dir = convert_feature_pt_to_presplit_cache()

    X_train = np.asarray(load_pickle(os.path.join(cache_dir, "X_train.pkl")), dtype=np.float32)
    X_test = np.asarray(load_pickle(os.path.join(cache_dir, "X_test.pkl")), dtype=np.float32)

    y_train = ensure_binary_labels(
        load_pickle(os.path.join(cache_dir, "y_train.pkl")),
        "official_train",
    )
    y_test = ensure_binary_labels(
        load_pickle(os.path.join(cache_dir, "y_test.pkl")),
        "official_test",
    )

    idx_train = np.asarray(
        load_pickle(os.path.join(cache_dir, "idx_train.pkl")),
        dtype=np.int64,
    ).reshape(-1)
    idx_test = np.asarray(
        load_pickle(os.path.join(cache_dir, "idx_test.pkl")),
        dtype=np.int64,
    ).reshape(-1)

    if FEATURE_STANDARDIZE:
        X_train = standardize_sequence_features(X_train)
        X_test = standardize_sequence_features(X_test)

    train_distribution = summarize_class_distribution(idx_train, y_train, "official_train")
    test_distribution = summarize_class_distribution(idx_test, y_test, "official_test")

    dataloaders = {
        "train": Dataloader(X_train, y_train.astype(np.int64), inmemory=False),
        "test": Dataloader(X_test, y_test.astype(np.int64), inmemory=False),
    }

    return SimpleNamespace(
        name=ACTIVE_DATASET,
        data_dim=int(X_train.shape[-1]),
        label_dim=NUM_CLASSES,
        dataloaders=dataloaders,
        presplit_cache_dir=cache_dir,
        arrays={
            "train": {"X": X_train, "y": y_train, "audio_ids": idx_train},
            "test": {"X": X_test, "y": y_test, "audio_ids": idx_test},
        },
        class_distribution={
            "train": train_distribution,
            "test": test_distribution,
        },
    )


# ============================================================
# 7. LRU Sequence Bones and Second Class Read Head
# ============================================================
def ensure_sequence_representation(sequence):
    """Collapse the output of LRU into [T, C]."""
    sequence = jnp.asarray(sequence)
    if sequence.ndim == 1:
        return sequence[None, :]
    if sequence.ndim == 2:
        return sequence
    if sequence.ndim >= 3:
        return sequence.reshape((-1, sequence.shape[-1]))
    raise ValueError(f"Unidentified sequence means shape={sequence.shape}")


def pool_sequence_representation(sequence, mode="mean"):
    """The time series hides the state as a single segment expression."""
    sequence = ensure_sequence_representation(sequence)
    mode = str(mode).strip().lower()

    if mode == "mean":
        return jnp.mean(sequence, axis=0)
    if mode == "max":
        return jnp.max(sequence, axis=0)
    if mode == "last":
        return sequence[-1]
    if mode == "meanmax":
        return jnp.concatenate(
            [jnp.mean(sequence, axis=0), jnp.max(sequence, axis=0)],
            axis=0,
        )
    raise ValueError(f"Unsupported READOUT_POOL={mode}")


class HiddenStatePooledClassifier(eqx.Module):
    """
    LRU Hides the second classification of the sequence.

    Structure:
        LRU hidden sequence
        -> Time pool
        -> LayerNorm
        -> Linear
        -> GELU
        -> Linear(2)

    The output is two logitss without softmax:
        logits[0]:Health/comparison scores
        logits[1]:Classes of diseases
    """
    backbone: object
    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    pool_mode: str

    def __init__(
        self,
        backbone,
        repr_dim: int,
        mlp_hidden_dim: int,
        num_classes: int = 2,
        pool_mode: str = "mean",
        *,
        key,
    ):
        self.backbone = backbone
        self.pool_mode = str(pool_mode)

        pooled_dim = int(repr_dim) * (2 if self.pool_mode == "meanmax" else 1)
        key_fc1, key_fc2 = jr.split(key, 2)

        self.norm = eqx.nn.LayerNorm((pooled_dim,))
        self.fc1 = eqx.nn.Linear(pooled_dim, int(mlp_hidden_dim), key=key_fc1)
        self.fc2 = eqx.nn.Linear(int(mlp_hidden_dim), int(num_classes), key=key_fc2)

    def __call__(self, x, state, key):
        sequence_repr, new_state = self.backbone(x, state, key)
        pooled = pool_sequence_representation(sequence_repr, mode=self.pool_mode)
        pooled = self.norm(pooled)
        hidden = jax.nn.gelu(self.fc1(pooled))
        logits = self.fc2(hidden)
        return logits, new_state


def create_lru_classifier(data_dim: int, seed: int):
    """Creates LRU backbone with two categories to read out."""
    key = jr.PRNGKey(int(seed))
    key_backbone, key_head = jr.split(key, 2)

    repr_dim = int(LRU_CONFIG["repr_dim"])
    backbone = LRU(
        num_blocks=int(LRU_CONFIG["num_blocks"]),
        data_dim=int(data_dim),
        N=int(LRU_CONFIG["ssm_dim"]),
        H=int(LRU_CONFIG["hidden_dim"]),
        output_dim=repr_dim,
        # The False here means that LRU returns the hidden sequence.
        # The classification of the overall task is done by the external two-dimensional header.
        classification=False,
        output_step=1,
        key=key_backbone,
    )

    model = HiddenStatePooledClassifier(
        backbone=backbone,
        repr_dim=repr_dim,
        mlp_hidden_dim=int(LRU_CONFIG["mlp_hidden_dim"]),
        num_classes=NUM_CLASSES,
        pool_mode=str(LRU_CONFIG["pool"]),
        key=key_head,
    )
    state = eqx.nn.State(model)
    return model, state


# ============================================================
# 8. Cross Entropy,Optimizers and reasoning
# ============================================================
@eqx.filter_jit
def calc_output(model, X, state, key):
    """Implementation of parallel forward communication for all segment of a batch."""
    logits, new_state = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, None),
        out_axes=(0, None),
    )(X, state, key)
    return logits, new_state


@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def classification_loss(diff_model, static_model, X, y, state, key):
    """General II classification Cross Entropy does not use category weighting or focal loss."""
    model = eqx.combine(diff_model, static_model)
    logits, new_state = calc_output(model, X, state, key)

    logits = jnp.asarray(logits)
    labels = jnp.asarray(y).reshape(-1).astype(jnp.int32)

    if logits.ndim != 2 or logits.shape[-1] != NUM_CLASSES:
        raise ValueError(
            f"Category logits should read [B,] {NUM_CLASSES}],Current shape={logits.shape}"
        )

    loss_per_sample = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits,
        labels=labels,
    )
    return jnp.mean(loss_per_sample), new_state


@eqx.filter_jit
def make_step(model, X, y, state, optimizer, optimizer_state, key):
    """Completes a forward, reverse and Adam update."""
    diff_model, static_model = eqx.partition(model, eqx.is_inexact_array)

    (loss_value, new_state), gradients = classification_loss(
        diff_model,
        static_model,
        X,
        y,
        state,
        key,
    )

    parameters = eqx.filter(model, eqx.is_inexact_array)
    updates, new_optimizer_state = optimizer.update(
        gradients,
        optimizer_state,
        params=parameters,
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, new_optimizer_state, loss_value


def build_warmup_cosine_schedule(total_steps: int, warmup_steps: int):
    """The tectonic scheme B uses the warmup + cosine learning rate."""
    total_steps = max(int(total_steps), 1)
    warmup_steps = int(min(max(warmup_steps, 1), max(total_steps - 1, 1)))

    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=float(LEARNING_RATE),
        transition_steps=warmup_steps,
    )
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=float(LEARNING_RATE),
        decay_steps=max(total_steps - warmup_steps, 1),
        alpha=float(COSINE_ALPHA),
    )
    return optax.join_schedules(
        [warmup_schedule, cosine_schedule],
        boundaries=[warmup_steps],
    )


def predict_segments(model, state, dataloader, batch_size: int, key) -> Dict[str, np.ndarray]:
    """
    Inference to a split return:
    - Two categories logits;
    - (a) The probability of the type of illness;
    - Type of projection;
    - Real category.
    """
    inference_model = eqx.tree_inference(model, value=True)

    logits_all = []
    positive_prob_all = []
    pred_class_all = []
    true_class_all = []

    for X_batch, y_batch in dataloader.loop_epoch(int(batch_size)):
        key, subkey = jr.split(key)
        logits, _ = calc_output(inference_model, X_batch, state, subkey)
        probabilities = jax.nn.softmax(logits, axis=-1)

        logits_np = np.asarray(logits, dtype=np.float32)
        probabilities_np = np.asarray(probabilities, dtype=np.float32)

        logits_all.append(logits_np)
        positive_prob_all.append(probabilities_np[:, POSITIVE_CLASS_ID])
        pred_class_all.append(np.argmax(probabilities_np, axis=-1).astype(np.int64))
        true_class_all.append(np.asarray(y_batch).reshape(-1).astype(np.int64))

    return {
        "logits": np.concatenate(logits_all, axis=0),
        "positive_probability": np.concatenate(positive_prob_all, axis=0),
        "predicted_label": np.concatenate(pred_class_all, axis=0),
        "true_label": np.concatenate(true_class_all, axis=0),
    }


# ============================================================
# 9. Segment Level and audio indicators
# ============================================================
def compute_binary_metrics(y_true, y_pred, y_prob) -> Dict[str, object]:
    """Compute Accuracy, Precision, Recall, F1, ROC-AUC and confuse matrices."""
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    y_prob = np.asarray(y_prob, dtype=np.float64).reshape(-1)

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("y_true,y_pred,y_prob Length is inconsistent.")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    if len(np.unique(y_true)) == 2:
        auc_value = float(roc_auc_score(y_true, y_prob))
    else:
        auc_value = float("nan")

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    return {
        "count": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": auc_value,
        "specificity": specificity,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "confusion_matrix": matrix.tolist(),
    }


def aggregate_audio_predictions(audio_ids, true_labels, positive_probabilities):
    """
    An arithmetic average of the disease probability of the same audio segment gets the probability of audio-level.

    audio_pred = 1,When the average probability of illness is >= 0.5; otherwise 0.
    """
    audio_ids = np.asarray(audio_ids, dtype=np.int64).reshape(-1)
    true_labels = np.asarray(true_labels, dtype=np.int64).reshape(-1)
    positive_probabilities = np.asarray(positive_probabilities, dtype=np.float64).reshape(-1)

    if not (len(audio_ids) == len(true_labels) == len(positive_probabilities)):
        raise ValueError("audio_ids,true_labels,positive_probabilities Length is inconsistent.")

    rows = []
    for audio_id in sorted(np.unique(audio_ids).tolist()):
        mask = audio_ids == int(audio_id)
        unique_true = np.unique(true_labels[mask])
        if len(unique_true) != 1:
            raise ValueError(
                f"audio_id={audio_id} segment The real label is inconsistent:{unique_true.tolist()}"
            )

        audio_probability = float(np.mean(positive_probabilities[mask]))
        rows.append({
            "audio_id": int(audio_id),
            "true_label": int(unique_true[0]),
            "positive_probability": audio_probability,
            "predicted_label": int(audio_probability >= CLASSIFICATION_THRESHOLD),
            "num_segments": int(np.sum(mask)),
        })

    audio_df = pd.DataFrame(rows)
    metrics = compute_binary_metrics(
        y_true=audio_df["true_label"].to_numpy(),
        y_pred=audio_df["predicted_label"].to_numpy(),
        y_prob=audio_df["positive_probability"].to_numpy(),
    )
    return {"prediction_df": audio_df, "metrics": metrics}


def evaluate_split(model, state, dataloader, audio_ids, batch_size: int, key):
    """One-time completion of segment reasoning, segment indicators and audio aggregation."""
    segment_output = predict_segments(model, state, dataloader, batch_size, key)
    segment_metrics = compute_binary_metrics(
        y_true=segment_output["true_label"],
        y_pred=segment_output["predicted_label"],
        y_prob=segment_output["positive_probability"],
    )
    audio_output = aggregate_audio_predictions(
        audio_ids=audio_ids,
        true_labels=segment_output["true_label"],
        positive_probabilities=segment_output["positive_probability"],
    )
    return {
        "segment_output": segment_output,
        "segment_metrics": segment_metrics,
        "audio_output": audio_output,
    }


# ============================================================
# 10. Training process for each random torrent
# ============================================================
def save_confusion_matrix_csv(metrics: Dict[str, object], path: str):
    """Save two times two confusion matrices as CSV with the same name."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.int64)
    frame = pd.DataFrame(
        matrix,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    frame.to_csv(path, encoding="utf-8-sig")


def build_segment_prediction_df(audio_ids, output: Dict[str, np.ndarray]) -> pd.DataFrame:
    """segment forecasts are collated."""
    audio_ids = np.asarray(audio_ids, dtype=np.int64).reshape(-1)
    return pd.DataFrame({
        "segment_index": np.arange(len(audio_ids), dtype=np.int64),
        "audio_id": audio_ids,
        "true_label": output["true_label"],
        "predicted_label": output["predicted_label"],
        "positive_probability": output["positive_probability"],
        "logit_class0": output["logits"][:, 0],
        "logit_class1": output["logits"][:, 1],
    })


def flatten_metrics(prefix: str, metrics: Dict[str, object]) -> Dict[str, object]:
    """Converts the indicator dictionary to a flat field suitable for CSV."""
    keys = [
        "count", "accuracy", "precision", "recall", "f1", "roc_auc",
        "specificity", "tn", "fp", "fn", "tp",
    ]
    return {f"{prefix}_{key}": metrics.get(key, np.nan) for key in keys}


def train_one_seed(dataset: SimpleNamespace, seed: int) -> Dict[str, object]:
    """
    Train a random seed.

    Every epoch:
    1. Completion of a epoch update on official train;
    2. Records epoch train_loss;
    3. Recording on official test/audio Accuracy, F1, ROC-AUC;
    4. The determination of whether to stop early is based only on train_loss.
    """
    set_global_seed(seed)

    model, state = create_lru_classifier(dataset.data_dim, seed=seed)

    steps_per_epoch = int(np.ceil(dataset.dataloaders["train"].size / BATCH_SIZE))
    total_steps = int(TARGET_EPOCHS * steps_per_epoch)
    warmup_steps = int(WARMUP_EPOCHS * steps_per_epoch)

    schedule = build_warmup_cosine_schedule(total_steps, warmup_steps)
    optimizer = optax.adam(learning_rate=schedule)
    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    key = jr.PRNGKey(int(seed))
    best_train_loss_for_stop = float("inf")
    no_improve_epochs = 0
    stopped_early = False
    stop_reason = "max_epochs_reached"

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(OUTPUT_SELECTION_ROOT, f"seed{int(seed)}_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    epoch_rows = []
    last_epoch = 0
    last_step = 0

    print("\n" + "-" * 100)
    print(f"[seed={seed}] Start training.")
    print(f"steps_per_epoch={steps_per_epoch} | total_steps={total_steps}")
    print("optimizer=Adam | weight_decay=0 | gradient_clip=off")
    print(
        f"early_stopping: monitor=train_loss | patience={EARLY_STOPPING_PATIENCE} | "
        f"min_delta={EARLY_STOPPING_MIN_DELTA} | restore_best=False"
    )
    print("official_test: Monitoring per epoch but not participating in any model selection")
    print("-" * 100)

    global_step = 0
    start_time = time.time()

    for epoch in range(1, TARGET_EPOCHS + 1):
        key, shuffle_key = jr.split(key)
        train_iterator = dataset.dataloaders["train"].loop(BATCH_SIZE, key=shuffle_key)
        epoch_losses = []

        for _ in range(steps_per_epoch):
            X_batch, y_batch = next(train_iterator)
            key, step_key = jr.split(key)

            model, state, optimizer_state, loss_value = make_step(
                model,
                X_batch,
                y_batch,
                state,
                optimizer,
                optimizer_state,
                step_key,
            )

            global_step += 1
            epoch_losses.append(float(loss_value))

        epoch_train_loss = float(np.mean(epoch_losses))
        last_epoch = int(epoch)
        last_step = int(global_step)

        # official test Each epoch evaluation, but only monitor-only.
        key, test_key = jr.split(key)
        test_monitor = evaluate_split(
            model=model,
            state=state,
            dataloader=dataset.dataloaders["test"],
            audio_ids=dataset.arrays["test"]["audio_ids"],
            batch_size=BATCH_SIZE,
            key=test_key,
        )

        test_segment_metrics = test_monitor["segment_metrics"]
        test_audio_metrics = test_monitor["audio_output"]["metrics"]

        improved = epoch_train_loss < (
            best_train_loss_for_stop - float(EARLY_STOPPING_MIN_DELTA)
        )
        if improved:
            best_train_loss_for_stop = epoch_train_loss
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        current_lr = float(schedule(max(global_step - 1, 0)))

        epoch_row = {
            "dataset": ACTIVE_DATASET,
            "feature_model": ACTIVE_FEATURE_MODEL,
            "feature_selection": ACTIVE_SELECTION,
            "model": MODEL_NAME,
            "seed": int(seed),
            "epoch": int(epoch),
            "step": int(global_step),
            "epoch_train_loss": epoch_train_loss,
            "best_train_loss_for_stop": float(best_train_loss_for_stop),
            "train_loss_improved": int(improved),
            "no_improve_epochs": int(no_improve_epochs),
            "learning_rate": current_lr,

            # The following official_test fields are used to monitor curves only.
            "official_test_segment_accuracy_monitor_only": float(test_segment_metrics["accuracy"]),
            "official_test_segment_f1_monitor_only": float(test_segment_metrics["f1"]),
            "official_test_segment_roc_auc_monitor_only": float(test_segment_metrics["roc_auc"]),
            "official_test_audio_accuracy_monitor_only": float(test_audio_metrics["accuracy"]),
            "official_test_audio_f1_monitor_only": float(test_audio_metrics["f1"]),
            "official_test_audio_roc_auc_monitor_only": float(test_audio_metrics["roc_auc"]),

            "selection_policy": SELECTION_POLICY,
            "selected_by": SELECTED_BY,
            "official_test_semantics": OFFICIAL_TEST_SEMANTICS,
            "test_monitor_used_for_selection": 0,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopping_restore_best": 0,
            "is_selected_epoch": 0,
        }
        epoch_rows.append(epoch_row)

        print(
            f"[seed={seed}] epoch={epoch}/{TARGET_EPOCHS} | "
            f"train_loss={epoch_train_loss:.6f} | "
            f"best_train_loss={best_train_loss_for_stop:.6f} | "
            f"no_improve={no_improve_epochs}/{EARLY_STOPPING_PATIENCE} | "
            f"test_seg_acc={test_segment_metrics['accuracy']:.4f} | "
            f"test_seg_f1={test_segment_metrics['f1']:.4f} | "
            f"test_seg_auc={test_segment_metrics['roc_auc']:.4f} | "
            f"test_audio_acc={test_audio_metrics['accuracy']:.4f} | "
            f"test_audio_f1={test_audio_metrics['f1']:.4f} | "
            f"test_audio_auc={test_audio_metrics['roc_auc']:.4f} | "
            f"time={time.time() - start_time:.1f}s"
        )

        if USE_EARLY_STOPPING and no_improve_epochs >= EARLY_STOPPING_PATIENCE:
            stopped_early = True
            stop_reason = (
                f"train_loss_no_improve_for_{EARLY_STOPPING_PATIENCE}_epochs"
            )
            print(
                f"[seed={seed}] early stopping at epoch={epoch} | "
                "reported_model=last_epoch_at_stop | restore_best=False"
            )
            break

    # Finally, epoch, which was effectively discontinued, was marked as the official report epoch.
    for row in epoch_rows:
        row["is_selected_epoch"] = int(int(row["epoch"]) == int(last_epoch))
        row["stopped_early"] = int(stopped_early)
        row["stop_reason"] = stop_reason

    epoch_trace_df = pd.DataFrame(epoch_rows)
    epoch_trace_path = os.path.join(run_dir, "epoch_trace.csv")
    epoch_trace_df.to_csv(epoch_trace_path, index=False, encoding="utf-8-sig")

    # It also keeps the usual epoch arrays for subsequent training and monitoring curves.
    np.save(
        os.path.join(run_dir, "epoch_train_loss.npy"),
        epoch_trace_df["epoch_train_loss"].to_numpy(dtype=np.float32),
    )
    np.save(
        os.path.join(run_dir, "official_test_audio_accuracy_monitor_only.npy"),
        epoch_trace_df["official_test_audio_accuracy_monitor_only"].to_numpy(dtype=np.float32),
    )
    np.save(
        os.path.join(run_dir, "official_test_audio_f1_monitor_only.npy"),
        epoch_trace_df["official_test_audio_f1_monitor_only"].to_numpy(dtype=np.float32),
    )
    np.save(
        os.path.join(run_dir, "official_test_audio_roc_auc_monitor_only.npy"),
        epoch_trace_df["official_test_audio_roc_auc_monitor_only"].to_numpy(dtype=np.float32),
    )

    # Saves the final model and corresponds to Equinox State when stopping.
    # best_model It's just the old alias, the semantic remains last model.
    eqx.tree_serialise_leaves(os.path.join(run_dir, "last_model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(run_dir, "last_state.eqx"), state)
    eqx.tree_serialise_leaves(os.path.join(run_dir, "selected_last_model.eqx"), model)
    eqx.tree_serialise_leaves(os.path.join(run_dir, "selected_last_state.eqx"), state)
    eqx.tree_serialise_leaves(os.path.join(run_dir, "best_model.eqx"), model)

    with open(os.path.join(run_dir, "selection_policy_note.txt"), "w", encoding="utf-8") as file:
        file.write(
            "selection_policy=last_epoch_or_train_loss_early_stop\n"
            "selected_by=last_epoch_or_train_loss_early_stop_no_test_selection\n"
            "official_test metrics are recorded every epoch for monitoring only.\n"
            "official_test is not used for early stopping or checkpoint selection.\n"
            "early_stopping_monitor=train_loss\n"
            "early_stopping_restore_best=False\n"
            "reported_model=last model at training stop\n"
        )

    # The final model is used for a full formal train/test evaluation at the time of cessation.
    key, train_eval_key, test_eval_key = jr.split(key, 3)
    final_train = evaluate_split(
        model=model,
        state=state,
        dataloader=dataset.dataloaders["train"],
        audio_ids=dataset.arrays["train"]["audio_ids"],
        batch_size=BATCH_SIZE,
        key=train_eval_key,
    )
    final_test = evaluate_split(
        model=model,
        state=state,
        dataloader=dataset.dataloaders["test"],
        audio_ids=dataset.arrays["test"]["audio_ids"],
        batch_size=BATCH_SIZE,
        key=test_eval_key,
    )

    # Saves the forecast details.
    train_segment_df = build_segment_prediction_df(
        dataset.arrays["train"]["audio_ids"],
        final_train["segment_output"],
    )
    test_segment_df = build_segment_prediction_df(
        dataset.arrays["test"]["audio_ids"],
        final_test["segment_output"],
    )

    train_segment_df.to_csv(
        os.path.join(run_dir, "train_segment_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    test_segment_df.to_csv(
        os.path.join(run_dir, "test_segment_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    final_train["audio_output"]["prediction_df"].to_csv(
        os.path.join(run_dir, "train_audio_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )
    final_test["audio_output"]["prediction_df"].to_csv(
        os.path.join(run_dir, "test_audio_predictions.csv"),
        index=False,
        encoding="utf-8-sig",
    )

    save_confusion_matrix_csv(
        final_test["audio_output"]["metrics"],
        os.path.join(run_dir, "test_audio_confusion_matrix.csv"),
    )
    save_confusion_matrix_csv(
        final_test["segment_metrics"],
        os.path.join(run_dir, "test_segment_confusion_matrix.csv"),
    )

    final_metrics_json = {
        "train_segment": final_train["segment_metrics"],
        "train_audio": final_train["audio_output"]["metrics"],
        "test_segment": final_test["segment_metrics"],
        "test_audio": final_test["audio_output"]["metrics"],
    }
    save_json(final_metrics_json, os.path.join(run_dir, "final_metrics.json"))

    result_row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": ACTIVE_DATASET,
        "feature_model": ACTIVE_FEATURE_MODEL,
        "feature_selection": ACTIVE_SELECTION,
        "unfree_layer_num": ACTIVE_UNFREE_LAYER,
        "feature_file_tag": ACTIVE_FILE_TAG,
        "model": MODEL_NAME,
        "seed": int(seed),
        "selected_epoch": int(last_epoch),
        "selected_step": int(last_step),
        "selection_policy": SELECTION_POLICY,
        "selected_by": SELECTED_BY,
        "test_monitor_used_for_selection": 0,
        "early_stopping_monitor": EARLY_STOPPING_MONITOR,
        "early_stopping_patience": int(EARLY_STOPPING_PATIENCE),
        "early_stopping_min_delta": float(EARLY_STOPPING_MIN_DELTA),
        "early_stopping_restore_best": 0,
        "stopped_early": int(stopped_early),
        "stop_reason": stop_reason,
        "learning_rate": float(LEARNING_RATE),
        "max_epochs": int(TARGET_EPOCHS),
        "batch_size": int(BATCH_SIZE),
        "audio_aggregation_method": AUDIO_AGG_METHOD,
        "classification_threshold": float(CLASSIFICATION_THRESHOLD),
        "run_dir": run_dir,
    }
    result_row.update(flatten_metrics("train_segment", final_train["segment_metrics"]))
    result_row.update(flatten_metrics("train_audio", final_train["audio_output"]["metrics"]))
    result_row.update(flatten_metrics("test_segment", final_test["segment_metrics"]))
    result_row.update(flatten_metrics("test_audio", final_test["audio_output"]["metrics"]))

    print(
        f"[seed={seed}] FINAL test audio | "
        f"Accuracy={result_row['test_audio_accuracy']:.4f} | "
        f"Precision={result_row['test_audio_precision']:.4f} | "
        f"Recall={result_row['test_audio_recall']:.4f} | "
        f"F1={result_row['test_audio_f1']:.4f} | "
        f"AUC={result_row['test_audio_roc_auc']:.4f}"
    )

    return {
        "result_row": result_row,
        "epoch_trace": epoch_trace_df,
        "test_audio_metrics": final_test["audio_output"]["metrics"],
        "run_dir": run_dir,
    }


# ============================================================
# 11. Multi-random feed aggregation
# ============================================================
def build_seed_list() -> List[int]:
    """A list of random torrents that are re-established by running switches."""
    seeds: List[int] = []
    if RUN_SINGLE_SEED_400:
        seeds.append(int(SINGLE_SEED))
    if RUN_MULTI_SEEDS:
        seeds.extend(int(seed) for seed in MULTI_SEEDS)

    deduplicated = []
    seen = set()
    for seed in seeds:
        if seed not in seen:
            deduplicated.append(seed)
            seen.add(seed)

    if len(deduplicated) == 0:
        raise ValueError("RUN_SINGLE_SEED_400 and RUN_MULTI_SEEDS both False, no running torrent.")
    return deduplicated


def build_mean_std_summary(seed_metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the number of random seeds mean and sample std for the main disaggregated indicators."""
    metric_columns = [
        "test_segment_accuracy",
        "test_segment_precision",
        "test_segment_recall",
        "test_segment_f1",
        "test_segment_roc_auc",
        "test_segment_specificity",
        "test_audio_accuracy",
        "test_audio_precision",
        "test_audio_recall",
        "test_audio_f1",
        "test_audio_roc_auc",
        "test_audio_specificity",
    ]

    rows = []
    for column in metric_columns:
        values = pd.to_numeric(seed_metrics_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        rows.append({
            "metric": column,
            "count": int(len(values)),
            "mean": float(np.mean(values)) if len(values) > 0 else np.nan,
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "min": float(np.min(values)) if len(values) > 0 else np.nan,
            "max": float(np.max(values)) if len(values) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def save_training_config(dataset: SimpleNamespace, seeds: List[int]):
    """Saves current data, models and training protocol snapshots."""
    config = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_name": "LRU_2class.py",
        "task": "ad_binary_classification",
        "class_mapping": {
            "0": "healthy_control_cc",
            "1": "disease_cd",
        },
        "dataset": {
            "active_dataset": ACTIVE_DATASET,
            "protocol": "official_train_as_train_official_test_as_monitor_and_final_test",
            "class_distribution": dataset.class_distribution,
            "process2_positive_definition": "MCI and Dementia are both class 1 through cd directories",
            "pitt_label_source": "cc_enhence/cd_enhence directory split from the feature export pipeline",
        },
        "feature": {
            "feature_model": ACTIVE_FEATURE_MODEL,
            "selection": ACTIVE_SELECTION,
            "single_layers_filter": list(SINGLE_LAYERS),
            "unfree_layer_num": ACTIVE_UNFREE_LAYER,
            "unfree_layer_filter": list(UNFREE_LAYER_NUM),
            "feature_file_tag": ACTIVE_FILE_TAG,
            "feature_dir": ACTIVE_FEATURE_DIR,
            "input_paths": INPUT_PATHS,
            "feature_standardize": FEATURE_STANDARDIZE,
            "data_dim": dataset.data_dim,
            "presplit_cache_dir": dataset.presplit_cache_dir,
            "force_rebuild_presplit_cache": FORCE_REBUILD_PRESPLIT_CACHE,
        },
        "model": {
            "name": MODEL_NAME,
            "config": LRU_CONFIG,
            "num_classes": NUM_CLASSES,
            "readout": "hidden_state_pooling_layernorm_mlp_linear2",
        },
        "training": {
            "seeds": seeds,
            "batch_size": BATCH_SIZE,
            "target_epochs": TARGET_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "optimizer": "Adam",
            "weight_decay": 0.0,
            "gradient_clipping": False,
            "warmup_epochs": WARMUP_EPOCHS,
            "cosine_alpha": COSINE_ALPHA,
            "early_stopping_enabled": USE_EARLY_STOPPING,
            "early_stopping_monitor": EARLY_STOPPING_MONITOR,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
            "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
            "early_stopping_restore_best": EARLY_STOPPING_RESTORE_BEST,
            "selection_policy": SELECTION_POLICY,
            "selected_by": SELECTED_BY,
            "official_test_semantics": OFFICIAL_TEST_SEMANTICS,
        },
        "evaluation": {
            "segment_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"],
            "audio_metrics": ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix"],
            "audio_aggregation": AUDIO_AGG_METHOD,
            "classification_threshold": CLASSIFICATION_THRESHOLD,
        },
    }
    save_json(config, os.path.join(OUTPUT_SELECTION_ROOT, "training_config.json"))


def run_current_context():
    """Runs the current data set and all random torrents of the feature layer."""
    dataset = load_classification_dataset()
    seeds = build_seed_list()
    save_training_config(dataset, seeds)

    seed_results = []
    all_epoch_traces = []
    confusion_sum = np.zeros((2, 2), dtype=np.int64)

    for seed in seeds:
        result = train_one_seed(dataset, seed)
        seed_results.append(result["result_row"])

        trace = result["epoch_trace"].copy()
        trace["run_dir"] = result["run_dir"]
        all_epoch_traces.append(trace)

        confusion_sum += np.asarray(
            result["test_audio_metrics"]["confusion_matrix"],
            dtype=np.int64,
        )

    seed_metrics_df = pd.DataFrame(seed_results)
    seed_metrics_path = os.path.join(
        OUTPUT_SELECTION_ROOT,
        f"{ACTIVE_DATASET}_lru_2class_seed_metrics.csv",
    )
    seed_metrics_df.to_csv(seed_metrics_path, index=False, encoding="utf-8-sig")

    if all_epoch_traces:
        combined_epoch_trace = pd.concat(all_epoch_traces, ignore_index=True)
        combined_epoch_trace.to_csv(
            os.path.join(
                OUTPUT_SELECTION_ROOT,
                f"{ACTIVE_DATASET}_lru_2class_epoch_trace.csv",
            ),
            index=False,
            encoding="utf-8-sig",
        )

    summary_df = build_mean_std_summary(seed_metrics_df)
    summary_df.to_csv(
        os.path.join(
            OUTPUT_SELECTION_ROOT,
            f"{ACTIVE_DATASET}_lru_2class_metrics_mean_std.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    confusion_sum_df = pd.DataFrame(
        confusion_sum,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    )
    confusion_sum_df.to_csv(
        os.path.join(
            OUTPUT_SELECTION_ROOT,
            f"{ACTIVE_DATASET}_lru_2class_confusion_matrix_sum.csv",
        ),
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 100)
    print("Current job complete")
    print("=" * 100)
    print("seed metrics:", seed_metrics_path)
    print(summary_df.to_string(index=False))


# ============================================================
# 12. Main entrance
# ============================================================
if __name__ == "__main__":
    run_current_context()
