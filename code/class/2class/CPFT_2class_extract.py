#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPFT_2class_extract.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.
============================================================

Training received by CPFT_2class_stage3.py AD-aware CPFT Stage3 checkpoint
Load back to HuggingFace WhisperModel and export official downstream for category AD
train / official test The freezing feature.

Definitions of the second classification
----------
    0 = Health/Creation Group (cc)
    1 = Disease Group (cd)

Enter checkpoint
---------------
/home/user/Alzheimer_code_jax_class/
    finetune/ft_CPFT/
    <DATASET>/
    CPFT_lr5e-5_epo5/
    singleX_unfreeZ/
    stage3_CPFT_maskoffK/
    last.ckpt

Output Character
--------
/home/user/Alzheimer_code_jax_class/
    feature_tensor/
    <DATASET>/
    ft_CPFT_lr5e-5_epo5/
    singleX_unfreeZ/

For example:

/home/user/Alzheimer_code_jax_class/
    feature_tensor/
    ADReSS-IS2020/
    ft_CPFT_lr5e-5_epo5/
    single12_unfree1/

Save each directory:

    feature_whisper_single12_0.3_30s_train.pt
    feature_whisper_single12_0.3_30s_test.pt
    labels1_whisper_single12_0.3_30s_train.pt
    labels2_whisper_single12_0.3_30s_test.pt
    indices_whisper_single12_0.3_30s_train.pt
    indices_whisper_single12_0.3_30s_test.pt

Data protocol
--------
1. Stage1 / Stage2 / Stage3 Use official train only.
2. official test Read only when this script finally exports features.
3. This script does not train models, select checkpoint using an official test.
5. Do not read MMSE metadata, do not perform missing MMSE filters.
6. The tags are available directly from official cc/cd directory:
       cc -> 0
       cd -> 1
7. Pitt Strict use of cc_enhence/cd_enhence available in DATASETS["Pitt"]
8. PROCESS-2 Read only the name of the document at the end of __CTD wav.

Characteristic protocols
--------
1. librosa.load(sr=16000, mono=True)
2. 30 seconds
3. 30% overlap
4. Assign single layer
5. Treatment of ln_post consistent with the freezing of Whisper baselines
6. Save tab as torch.long
7. The index is maintained as NumPy int64, shape [N,2]:
       Column 0 = audio_id
       Column 1 = second_id

Run Example
--------
conda activate whisper_cpft_faiss
python CPFT_2class_extract.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.

Annotations
----
Script file places do not affect the result directory. All paths are specified by the following parameter area.
"""

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

import librosa
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import WhisperModel, WhisperProcessor


# =============================================================================
# 2) Parameters Settings: must be aligned to class_stage1_v2 / class_stage2 / class_stage3
# =============================================================================

# The current AD Second Classification Project Root Catalogue.
CLASS_PROJECT_ROOT = "/home/user/Alzheimer_code_jax_class"

# CPFT Stage1/2/3 An intermediate result root directory.
PROJECT_ROOT = os.path.join(CLASS_PROJECT_ROOT, "finetune")
CPFT_ROOT = os.path.join(PROJECT_ROOT, "ft_CPFT")

# Final feature root directory.
FEATURE_ROOT = os.path.join(CLASS_PROJECT_ROOT, "feature_tensor")

# Four official data sets.
DATASET_OPTIONS = ["ADReSS-IS2020"]


def _canonical_dataset_option(value: str) -> str:
    """Map the public ADReSS name to the existing internal dataset key."""
    value = str(value).strip()
    return "ADReSS-IS2020" if value == "ADReSS" else value

# Keeps a single value variable for internal recycling.
DATASET_OPTION = _canonical_dataset_option(DATASET_OPTIONS[0])

# The expression layer that needs to be extracted must be consistent with the training level Stage1/2/3.
LAYERS_TO_RUN = [12]

# How many layers of encoder have been unfrozen continuously relative to the current one.
# Formal batches may be used [7,8,9,10,11,12];
# A group may provisionally be [1] or [7] at the time of the first debug.
UNFREEZE_ENCODER_LAYERS_LIST = [7,8,9,10,11,12]
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# Stage3 mask Injection deviations must be consistent with CPFT_2class_stage3.py.
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]

# Upstream protocol tags are only written in meta and do not add to the directory hierarchy.
RUN_PROTOCOL_TAG = "official_train_test"


# Stage1 / Stage2 / Stage3 / extract There must be unity.
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5"

# Path experiment label.
STAGE1_SOURCE_PATH_PARAM_TAGS: List[str] = []
PATH_PARAM_TAGS: List[str] = []

# Stage3 Configure, for meta records only, to be consistent with class_stage2/class_stage3.
CPFT_KS = [32, 64]
CPFT_EXPECTED_NUM_CLUSTERS = [32, 64]
CPFT_STAGE3_LAMBDA_CLUS = 1.0
CPFT_STAGE3_LAMBDA_AD = 0.05
CPFT_METHOD_TAG = "ad_aware_cpft"

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
#     ft_CPFT_lr5e-5_epo5/
#
# If more than one mask_offset is exported at a later time, it may be changed to True to prevent cross-covering,
# The directory will then become ft_CPFT_lr5e-5_epo5_maskoffK/.
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
        ft_CPFT_lr5e-5_epo5/
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
# 4) official train / official test Audio Directory
# =============================================================================
DATASETS: Dict[str, Dict[str, str]] = {
    "PROCESS-2": {
        "train_cc": "/home/user/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/user/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/user/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/user/Downloads/PROCESS-2/test/cd",
    },
    "ADReSS-IS2020": {
        "train_cc": (
            "/home/user/Downloads/dementiabank/"
            "ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc"
        ),
        "train_cd": (
            "/home/user/Downloads/dementiabank/"
            "ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd"
        ),
        "test_cc": (
            "/home/user/Downloads/dementiabank/"
            "ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio"
        ),
        "test_cd": (
            "/home/user/Downloads/dementiabank/"
            "ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio"
        ),
    },
    "ADReSSo21": {
        "train_cc": (
            "/home/user/Downloads/dementiabank/"
            "ADReSSo21/diagnosis/train/audio/cn"
        ),
        "train_cd": (
            "/home/user/Downloads/dementiabank/"
            "ADReSSo21/diagnosis/train/audio/ad"
        ),
        "test_cc": (
            "/home/user/Downloads/dementiabank/"
            "ADReSSo21/diagnosis/test-dist/Control"
        ),
        "test_cd": (
            "/home/user/Downloads/dementiabank/"
            "ADReSSo21/diagnosis/test-dist/ProbableAD"
        ),
    },
    "Pitt": {
        "train_cc": (
            "/home/user/Downloads/dementiabank/"
            "Pitt_new/norm/train_test_split/train/cc_enhence"
        ),
        "train_cd": (
            "/home/user/Downloads/dementiabank/"
            "Pitt_new/norm/train_test_split/train/cd_enhence"
        ),
        "test_cc": (
            "/home/user/Downloads/dementiabank/"
            "Pitt_new/norm/train_test_split/test/cc_enhence"
        ),
        "test_cd": (
            "/home/user/Downloads/dementiabank/"
            "Pitt_new/norm/train_test_split/test/cd_enhence"
        ),
    },
}


def diagnosis_label_from_group(group_name: str) -> int:
    """Maps the cc/cd directory semanticly to 0/1 tab."""
    group = str(group_name).strip().lower()

    if group == "cc":
        return 0
    if group == "cd":
        return 1

    raise ValueError(
        f"group_name must be cc/cd, current ={group_name}"
    )


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


def build_entries_from_official_dir(
    audio_dir: str,
    split_name: str,
    group_name: str,
    dataset_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    For the list of official train_cc/train_cd/test_cc/test_cd to entries

    Each article entry carries directly target_label:
        cc -> 0
        cd -> 1
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    split_name = str(split_name).strip().lower()
    group_name = str(group_name).strip().lower()

    if split_name not in {"train", "test"}:
        raise ValueError(
            f"split_name Must be train/test, current ={split_name}"
        )

    target_label = diagnosis_label_from_group(group_name)

    entries: List[Dict[str, Any]] = []

    for wav_path in list_wav_files(
        audio_dir,
        dataset_key=dataset_key,
    ):
        entries.append({
            "wav_path": canonical_wav_path(wav_path),
            "wav_name": Path(wav_path).name,
            "split": split_name,
            "group": group_name,
            "target_label": int(target_label),
            "source_subset": f"{split_name}_{group_name}",
            "process2_task_filter": (
                "CTD"
                if str(dataset_key).strip().lower() == "process-2"
                else ""
            ),
        })

    return entries


def load_official_train_test_entries(
    dataset_option: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Construct a certain train/ official test inductions.

    The order is fixed as:
        train_cc -> train_cd
        test_cc  -> test_cd

    train/test Separately allocated stable official_audio_id.
    """
    if dataset_option not in DATASETS:
        raise ValueError(
            f"DATASETS No Configuration {dataset_option}"
        )

    paths = DATASETS[dataset_option]

    train_entries = (
        build_entries_from_official_dir(
            paths["train_cc"],
            split_name="train",
            group_name="cc",
            dataset_key=dataset_option,
        )
        +
        build_entries_from_official_dir(
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
        +
        build_entries_from_official_dir(
            paths["test_cd"],
            split_name="test",
            group_name="cd",
            dataset_key=dataset_option,
        )
    )

    if len(train_entries) == 0 or len(test_entries) == 0:
        raise ValueError(
            "official train/test entries Empty:"
            f"train={len(train_entries)}, "
            f"test={len(test_entries)}"
        )

    # train/test Individual numbering.
    for audio_id, row in enumerate(train_entries):
        row["official_audio_id"] = int(audio_id)
        row["audio_id"] = int(audio_id)

    for audio_id, row in enumerate(test_entries):
        row["official_audio_id"] = int(audio_id)
        row["audio_id"] = int(audio_id)

    return train_entries, test_entries


def summarize_audio_entries(
    entries: Sequence[Dict[str, Any]],
    split_name: str,
) -> Dict[str, int]:
    """Statistically available audio class distribution."""
    labels = np.asarray(
        [int(row["target_label"]) for row in entries],
        dtype=np.int64,
    )

    if not set(np.unique(labels).tolist()).issubset({0, 1}):
        raise ValueError(
            f"{split_name} A non-0/1 tab appears:"
            f"{np.unique(labels).tolist()}"
        )

    result = {
        "audio_total": int(len(labels)),
        "audio_class0": int(np.sum(labels == 0)),
        "audio_class1": int(np.sum(labels == 1)),
    }

    print(
        f"[{split_name}] audio_total={result['audio_total']} | "
        f"class0={result['audio_class0']} | "
        f"class1={result['audio_class1']}"
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
    Find checkpoint saved at CPFT_2class_stage3.py.

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
            "STAGE3_MASK_INJECT_AFTER_LAYERS Consistency with CPFT_2class_stage3.py."
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
    Loads only WhisperModel from Lighting checkpoint for CPFT_2class_stage3.py.

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

    # CPFT_2class_stage3.py WhisperModel attribute is called self.model,
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

    The label is reproduced directly from each audio target_label.
    audio_id Use pre-allocated official_audio_id.
    """
    feature_list: List[torch.Tensor] = []
    labels_list: List[int] = []
    indices_list: List[Tuple[int, int]] = []
    manifest: List[Dict[str, Any]] = []

    for entry_index, row in enumerate(
        tqdm(
            entries,
            desc=f"[{subset_name}] single{single_layer}",
            ncols=100,
        )
    ):
        wav_path = canonical_wav_path(
            row["wav_path"]
        )
        target_label = int(
            row["target_label"]
        )

        if target_label not in {0, 1}:
            raise ValueError(
                f"{subset_name} Illegal labels appear:"
                f"path={wav_path}, label={target_label}"
            )

        if not os.path.isfile(wav_path):
            raise FileNotFoundError(
                f"Could not close temporary folder: %s{wav_path}"
            )

        if row.get("official_audio_id") is not None:
            audio_id = int(
                row["official_audio_id"]
            )
        elif row.get("audio_id") is not None:
            audio_id = int(
                row["audio_id"]
            )
        else:
            audio_id = int(entry_index)

        segment_features, segment_indices = (
            get_segment_features(
                audio_file=wav_path,
                audio_file_index=audio_id,
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
        )

        if len(segment_features) == 0:
            raise ValueError(
                f"No audio generated segment:{wav_path}"
            )

        feature_list.extend(segment_features)
        indices_list.extend(segment_indices)
        labels_list.extend(
            [int(target_label)]
            * len(segment_features)
        )

        kept = copy.deepcopy(row)
        kept["audio_index_in_subset"] = int(audio_id)
        kept["official_audio_id"] = int(audio_id)
        kept["audio_id"] = int(audio_id)
        kept["target_label"] = int(target_label)
        kept["num_segments"] = int(
            len(segment_features)
        )
        manifest.append(kept)

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


def validate_binary_feature_bundle(
    features: torch.Tensor,
    labels: torch.Tensor,
    indices: np.ndarray,
    split_name: str,
) -> Dict[str, int]:
    """
    Strictly check feature/label/index numbers and audio-level label consistency before saving.
    """
    if features.ndim != 3:
        raise ValueError(
            f"{split_name} feature It is necessary to be [N,T,C],"
            f"Current ={tuple(features.shape)}"
        )

    labels = labels.detach().cpu().to(torch.long).reshape(-1)
    indices = np.asarray(
        indices,
        dtype=np.int64,
    ).reshape(-1, 2)

    if not (
        len(features)
        == len(labels)
        == len(indices)
    ):
        raise ValueError(
            f"{split_name} feature/label/index Numbers are inconsistent:"
            f"{len(features)}, {len(labels)}, {len(indices)}"
        )

    unique_labels = sorted(
        torch.unique(labels).tolist()
    )
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(
            f"{split_name} Label is not 0/1:{unique_labels}"
        )

    audio_class_counts = {
        0: 0,
        1: 0,
    }

    for audio_id in sorted(
        np.unique(indices[:, 0]).tolist()
    ):
        mask = indices[:, 0] == int(audio_id)
        audio_labels = np.unique(
            labels.numpy()[mask]
        )

        if len(audio_labels) != 1:
            raise ValueError(
                f"{split_name} Middle audio_id={audio_id} "
                f"The segment label is inconsistent:{audio_labels.tolist()}"
            )

        class_id = int(audio_labels[0])
        audio_class_counts[class_id] += 1

    result = {
        "segment_total": int(len(labels)),
        "segment_class0": int(
            torch.sum(labels == 0).item()
        ),
        "segment_class1": int(
            torch.sum(labels == 1).item()
        ),
        "audio_total": int(
            len(np.unique(indices[:, 0]))
        ),
        "audio_class0": int(
            audio_class_counts[0]
        ),
        "audio_class1": int(
            audio_class_counts[1]
        ),
    }

    print(
        f"[{split_name}] segment_total={result['segment_total']} | "
        f"class0={result['segment_class0']} | "
        f"class1={result['segment_class1']}"
    )
    print(
        f"[{split_name}] audio_total={result['audio_total']} | "
        f"class0={result['audio_class0']} | "
        f"class1={result['audio_class1']}"
    )

    return result


# =============================================================================
# 8) Save Second Class Identity
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
    Saves the six-file format for official train/test readable at LRU_2class.py.
    """
    ensure_dir(save_root)

    feature_model_tag = get_feature_model_tag(
        mask_offset
    )
    feature_selection_tag = (
        get_feature_selection_tag(single_layer)
    )

    layer_name = build_layer_name(
        single_layer=int(single_layer),
        unfreeze_layer_count=int(
            UNFREEZE_ENCODER_LAYERS
        ),
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage3_subdir = build_stage3_subdir(
        mask_offset
    )
    file_tag = (
        f"{FEATURE_PREFIX}_"
        f"single{int(single_layer)}_"
        f"{OV_STR}_{SEC_STR}"
    )

    output_paths = {
        "feature_train": os.path.join(
            save_root,
            f"feature_{file_tag}_train.pt",
        ),
        "feature_test": os.path.join(
            save_root,
            f"feature_{file_tag}_test.pt",
        ),
        "label_train": os.path.join(
            save_root,
            f"labels1_{file_tag}_train.pt",
        ),
        "label_test": os.path.join(
            save_root,
            f"labels2_{file_tag}_test.pt",
        ),
        "index_train": os.path.join(
            save_root,
            f"indices_{file_tag}_train.pt",
        ),
        "index_test": os.path.join(
            save_root,
            f"indices_{file_tag}_test.pt",
        ),
    }

    torch.save(
        train_features.to(torch.float32),
        output_paths["feature_train"],
    )
    torch.save(
        test_features.to(torch.float32),
        output_paths["feature_test"],
    )
    torch.save(
        train_labels.to(torch.long),
        output_paths["label_train"],
    )
    torch.save(
        test_labels.to(torch.long),
        output_paths["label_test"],
    )
    torch.save(
        np.asarray(train_indices, dtype=np.int64),
        output_paths["index_train"],
    )
    torch.save(
        np.asarray(test_indices, dtype=np.int64),
        output_paths["index_test"],
    )

    if SAVE_META:
        meta = {
            "task": "ad_binary_classification",
            "class_mapping": {
                "0": "healthy_control",
                "1": "ad_or_disease",
            },
            "dataset_option": str(DATASET_OPTION),
            "run_protocol": str(RUN_PROTOCOL_TAG),
            "official_test_used_in_cpft_training": False,

            "single_layer": int(single_layer),
            "unfreeze_encoder_layers": int(
                UNFREEZE_ENCODER_LAYERS
            ),
            "layer_name": str(layer_name),

            "mask_offset": int(mask_offset),
            "stage3_subdir": str(stage3_subdir),
            "feature_model_tag": str(
                feature_model_tag
            ),
            "feature_selection_tag": str(
                feature_selection_tag
            ),
            "feature_file_tag": str(file_tag),

            "stage3_ckpt_path": str(
                stage3_ckpt_path
            ),
            "checkpoint_load_report": (
                checkpoint_load_report
            ),
            "whisper_base_model_name": str(
                WHISPER_BASE_MODEL_NAME
            ),

            "sampling_rate": int(SAMPLE_RATE),
            "segment_sec": int(SEGMENT_SEC),
            "overlap_rate": float(
                OVERLAP_RATE
            ),
            "norm_for_single": (
                None
                if NORM_FOR_SINGLE is None
                else str(NORM_FOR_SINGLE)
            ),
            "drop_input_layer": bool(
                DROP_INPUT_LAYER
            ),
            "audio_normalize": bool(
                ENABLE_AUDIO_NORMALIZE
            ),

            "label_dtype": "torch.int64",
            "index_dtype": "numpy.int64",
            "index_format": (
                "[N,2], column0=audio_id, "
                "column1=segment_id"
            ),

            "feature_train_semantics": (
                "official_train_all_no_mmse_filter"
            ),
            "feature_test_semantics": (
                "official_test_all_no_mmse_filter"
            ),
            "audio_id_policy": (
                "stable_official_audio_id_"
                "train_test_independent_"
                "cc_then_cd"
            ),

            "train_feature_shape": list(
                train_features.shape
            ),
            "test_feature_shape": list(
                test_features.shape
            ),
            "train_distribution": (
                train_distribution
            ),
            "test_distribution": (
                test_distribution
            ),
            "saved_files": output_paths,

            "downstream_recommended_config": {
                "FEATURE_MODEL_NAME": str(
                    feature_model_tag
                ),
                "SINGLE_LAYERS": [
                    int(single_layer)
                ],
                "FEATURE_SELECTION": str(
                    feature_selection_tag
                ),
                "FEATURE_FILE_TAG": str(
                    file_tag
                ),
            },

            "cpft_method": str(
                CPFT_METHOD_TAG
            ),
            "path_param_tags": list(
                PATH_PARAM_TAGS
            ),
            "path_param_suffix": (
                build_path_param_suffix(
                    PATH_PARAM_TAGS
                )
            ),
            "stage1_source_path_param_tags": list(
                STAGE1_SOURCE_PATH_PARAM_TAGS
            ),
            "cpft_ks": [
                int(value)
                for value in CPFT_KS
            ],
            "cpft_expected_num_clusters": [
                int(value)
                for value in (
                    CPFT_EXPECTED_NUM_CLUSTERS
                )
            ],
            "cpft_stage3_lambda_clus": float(
                CPFT_STAGE3_LAMBDA_CLUS
            ),
            "cpft_stage3_lambda_ad": float(
                CPFT_STAGE3_LAMBDA_AD
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
                "train_manifest": train_manifest
            },
            os.path.join(
                save_root,
                "train_manifest.json",
            ),
        )
        save_json(
            {
                "test_manifest": test_manifest
            },
            os.path.join(
                save_root,
                "test_manifest.json",
            ),
        )
        save_text(
            str(stage3_ckpt_path),
            os.path.join(
                save_root,
                "stage3_ckpt_path.txt",
            ),
        )
        save_text(
            "AD binary classification feature export.\n"
            "Stage1/Stage2/Stage3 use official train only.\n"
            "Official test is loaded only here for final feature export.\n"
            "No MMSE metadata or missing-MMSE filtering is used.\n"
            "Labels: cc=0, cd=1.\n",
            os.path.join(
                save_root,
                "protocol_note.txt",
            ),
        )

    print(f"[SAVE] root={save_root}")
    print(
        "  semantics: "
        "train=official_train_all_no_mmse_filter, "
        "test=official_test_all_no_mmse_filter"
    )
    for name, path in output_paths.items():
        print(f"  - {name}: {path}")

    return output_paths


# =============================================================================
# 9) Export Tasks Single dataset / layer / unfreeze / maskoff
# =============================================================================
def run_one_configuration(
    single_layer: int,
    mask_offset: int,
) -> Dict[str, Any]:
    """
    A single configured full export process.

    1. Read official train/test;
    2. Generate 0/1 tags directly;
    3. Find last.ckpt for CPFT_2class_stage3.py;
    4. Load the Whisper parameter back to HuggingFace WhisperModel in Stage3;
    5. (a) To extract train/test features;
    6. Saves the second category six file bundle.
    """
    single_layer = int(single_layer)
    mask_offset = int(mask_offset)

    save_root = get_save_root(
        DATASET_OPTION,
        single_layer,
        mask_offset,
    )

    layer_name = build_layer_name(
        single_layer=single_layer,
        unfreeze_layer_count=int(
            UNFREEZE_ENCODER_LAYERS
        ),
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage3_run_root = build_run_root(
        dataset_option=DATASET_OPTION,
        layer_id=single_layer,
    )
    stage3_dir = os.path.join(
        stage3_run_root,
        build_stage3_subdir(mask_offset),
    )

    print("\n" + "=" * 100)
    print("[CLASS EXTRACT]")
    print("  DATASET_OPTION =", DATASET_OPTION)
    print("  RUN_PROTOCOL   =", RUN_PROTOCOL_TAG)
    print("  single_layer   =", single_layer)
    print("  unfreeze       =", UNFREEZE_ENCODER_LAYERS)
    print("  layer_name     =", layer_name)
    print("  mask_offset    =", mask_offset)
    print("  stage3_dir     =", stage3_dir)
    print("  save_root      =", save_root)
    print("  extract target = official train + official test")
    print("=" * 100)

    train_entries, test_entries = (
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

    stage3_ckpt_path = find_stage3_checkpoint(
        dataset_option=DATASET_OPTION,
        layer_id=single_layer,
        mask_offset=mask_offset,
        override_ckpt=STAGE3_CKPT_OVERRIDE,
    )

    print(
        f"[CKPT] Stage3 checkpoint = "
        f"{stage3_ckpt_path}"
    )

    processor = WhisperProcessor.from_pretrained(
        WHISPER_BASE_MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = WhisperModel.from_pretrained(
        WHISPER_BASE_MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

    checkpoint_load_report = (
        load_stage3_ckpt_into_whisper_model(
            model,
            stage3_ckpt_path,
        )
    )

    model.eval()
    model.to(DEVICE)

    ln_post = model.encoder.layer_norm
    top_layer_index = int(
        model.config.encoder_layers
    )

    train_result = extract_entries_features(
        entries=train_entries,
        subset_name="official_train",
        processor=processor,
        model=model,
        single_layer=single_layer,
        ln_post=ln_post,
        top_layer_index=top_layer_index,
    )
    (
        train_feature_list,
        train_label_list,
        train_index_list,
        train_manifest,
    ) = train_result

    test_result = extract_entries_features(
        entries=test_entries,
        subset_name="official_test",
        processor=processor,
        model=model,
        single_layer=single_layer,
        ln_post=ln_post,
        top_layer_index=top_layer_index,
    )
    (
        test_feature_list,
        test_label_list,
        test_index_list,
        test_manifest,
    ) = test_result

    global_time_steps = max(
        max(
            int(feature.shape[0])
            for feature in train_feature_list
        ),
        max(
            int(feature.shape[0])
            for feature in test_feature_list
        ),
    )

    print(
        f"[PAD] global_time_steps="
        f"{global_time_steps}"
    )

    train_features = stack_with_time_padding(
        train_feature_list,
        target_t=global_time_steps,
    )
    test_features = stack_with_time_padding(
        test_feature_list,
        target_t=global_time_steps,
    )

    train_labels = torch.tensor(
        train_label_list,
        dtype=torch.long,
    )
    test_labels = torch.tensor(
        test_label_list,
        dtype=torch.long,
    )

    train_indices = to_index_array(
        train_index_list,
        "official_train",
    )
    test_indices = to_index_array(
        test_index_list,
        "official_test",
    )

    train_distribution = (
        validate_binary_feature_bundle(
            features=train_features,
            labels=train_labels,
            indices=train_indices,
            split_name="official_train",
        )
    )
    test_distribution = (
        validate_binary_feature_bundle(
            features=test_features,
            labels=test_labels,
            indices=test_indices,
            split_name="official_test",
        )
    )

    # Rechecking for characteristics statistics is consistent with the original audio entry statistics.
    for split_name, original_dist, feature_dist in [
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
        if (
            int(original_dist["audio_total"])
            != int(feature_dist["audio_total"])
            or int(original_dist["audio_class0"])
            != int(feature_dist["audio_class0"])
            or int(original_dist["audio_class1"])
            != int(feature_dist["audio_class1"])
        ):
            raise ValueError(
                f"{split_name} Original audio and export characteristics "
                "audio-level Discrepancies in category statistics: \n"
                f"original={original_dist}\n"
                f"feature={feature_dist}"
            )

    print(
        f"[SHAPE] train_features="
        f"{tuple(train_features.shape)}"
    )
    print(
        f"[SHAPE] test_features="
        f"{tuple(test_features.shape)}"
    )
    print(
        f"[SHAPE] train_labels="
        f"{tuple(train_labels.shape)}"
    )
    print(
        f"[SHAPE] test_labels="
        f"{tuple(test_labels.shape)}"
    )
    print(
        f"[SHAPE] train_indices="
        f"{tuple(train_indices.shape)}"
    )
    print(
        f"[SHAPE] test_indices="
        f"{tuple(test_indices.shape)}"
    )

    output_paths = save_feature_bundle(
        save_root=save_root,
        single_layer=single_layer,
        mask_offset=mask_offset,
        stage3_ckpt_path=stage3_ckpt_path,
        checkpoint_load_report=checkpoint_load_report,
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        test_labels=test_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        train_distribution=train_distribution,
        test_distribution=test_distribution,
    )

    result = {
        "dataset": str(DATASET_OPTION),
        "single_layer": int(single_layer),
        "unfreeze_encoder_layers": int(
            UNFREEZE_ENCODER_LAYERS
        ),
        "mask_offset": int(mask_offset),
        "save_root": str(save_root),
        "stage3_ckpt_path": str(
            stage3_ckpt_path
        ),
        "output_paths": output_paths,
    }

    # Timely release of visible and high volumes.
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
        f"maskoff{mask_offset}"
    )

    return result


# =============================================================================
# 10) Main entrance
# =============================================================================
def validate_global_configuration() -> None:
    """Checks the cross script parameters and directory overlay risk before starting."""
    if len(DATASET_OPTIONS) == 0:
        raise ValueError(
            "DATASET_OPTIONS It's empty."
        )

    if len(LAYERS_TO_RUN) == 0:
        raise ValueError(
            "LAYERS_TO_RUN It's empty."
        )

    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError(
            "UNFREEZE_ENCODER_LAYERS_LIST It's empty."
        )

    if len(STAGE3_MASK_INJECT_AFTER_LAYERS) == 0:
        raise ValueError(
            "STAGE3_MASK_INJECT_AFTER_LAYERS It's empty."
        )

    if (
        len(STAGE3_MASK_INJECT_AFTER_LAYERS) > 1
        and not INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG
    ):
        raise ValueError(
            "The current configuration will export several mask_offsets simultaneously, but a feature model directory "
            "Do not include maskoff and then run tasks will cover first run tasks. \n"
            "Please put INCLUDE_MASK_OFFSET_IN_FEATURE_MODEL_TAG=True,"
            "Or just one STAGE3_MASK_INJECT_AFTER_LAYERS."
        )

    if [
        int(value)
        for value in CPFT_KS
    ] != [
        int(value)
        for value in CPFT_EXPECTED_NUM_CLUSTERS
    ]:
        raise ValueError(
            "CPFT_KS Inconsistent with CPFT_EXPECTED_NUM_CLUSTERS:"
            f"{CPFT_KS} != {CPFT_EXPECTED_NUM_CLUSTERS}"
        )


def main() -> None:
    """
    Through the data sets, the number of unfrozen layers, the expression layers and mask_offset,
    Export features for each group Stage3 checkpoint that has been trained.
    """
    global DATASET_OPTION
    global UNFREEZE_ENCODER_LAYERS

    validate_global_configuration()

    print("\n" + "=" * 100)
    print("[CONFIG]")
    print("  CLASS_PROJECT_ROOT =", CLASS_PROJECT_ROOT)
    print("  CPFT_ROOT         =", CPFT_ROOT)
    print("  FEATURE_ROOT       =", FEATURE_ROOT)
    print("  DATASET_OPTIONS    =", DATASET_OPTIONS)
    print("  MODEL_ROOT_TAG     =", MODEL_ROOT_TAG)
    print(
        "  UNFREEZE_ENCODER_LAYERS_LIST =",
        UNFREEZE_ENCODER_LAYERS_LIST,
    )
    print("  LAYERS_TO_RUN      =", LAYERS_TO_RUN)
    print(
        "  STAGE3_MASK_INJECT_AFTER_LAYERS =",
        STAGE3_MASK_INJECT_AFTER_LAYERS,
    )
    print(
        "  WHISPER_BASE_MODEL_NAME =",
        WHISPER_BASE_MODEL_NAME,
    )
    print("  SEGMENT_SEC        =", SEGMENT_SEC)
    print("  OVERLAP_RATE       =", OVERLAP_RATE)
    print("  NORM_FOR_SINGLE    =", NORM_FOR_SINGLE)
    print("  DEVICE             =", DEVICE)
    print("=" * 100)

    completed: List[Dict[str, Any]] = []

    for dataset_option in [_canonical_dataset_option(value) for value in DATASET_OPTIONS]:
        for unfreeze_count in [
            int(value)
            for value in UNFREEZE_ENCODER_LAYERS_LIST
        ]:
            DATASET_OPTION = str(dataset_option)
            UNFREEZE_ENCODER_LAYERS = int(
                unfreeze_count
            )

            for layer_id in [
                int(value)
                for value in LAYERS_TO_RUN
            ]:
                for mask_offset in [
                    int(value)
                    for value in (
                        STAGE3_MASK_INJECT_AFTER_LAYERS
                    )
                ]:
                    completed.append(
                        run_one_configuration(
                            single_layer=layer_id,
                            mask_offset=mask_offset,
                        )
                    )

    print("\n" + "=" * 100)
    print("[ALL DONE]")
    for item in completed:
        print(
            f"  dataset={item['dataset']} | "
            f"single={item['single_layer']} | "
            f"unfree={item['unfreeze_encoder_layers']} | "
            f"maskoff={item['mask_offset']} | "
            f"save_root={item['save_root']}"
        )

    print("\nDownstream LRU_2class.py to find feature model directory:")
    for mask_offset in [
        int(value)
        for value in STAGE3_MASK_INJECT_AFTER_LAYERS
    ]:
        print(
            "  FEATURE_MODEL_NAME = "
            f'"{get_feature_model_tag(mask_offset)}"'
        )

    print("=" * 100)


if __name__ == "__main__":
    main()
