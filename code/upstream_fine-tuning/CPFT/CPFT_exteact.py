#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPFT_exteact.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.
===================================

Use
----
The Stage3 checkpoint of CPFT_stage3.py training will be brought back to the current official downstream mainline.
And align the identity saving path to a new path specification at Stage3 v3.

Stage3 v3 The path code for checkpoint is:

    mmse_cpft_runs/<DATASET>/<MODEL_ROOT_TAG>/singleX_unfreeZ/foldY/stage3_CPFT_maskoffK/

This script is read for:

    mmse_cpft_runs/<DATASET>/whisper_small_cpft/single12_unfree1/fold1/stage3_CPFT_maskoff1/*.ckpt

and save as readable finetune feature directory: lru+transformer+lstm.py

    feature_tensor/<DATASET>/whisper_small_cpft_s3_maskoff1_ft/single12_unfree1/fold1/
    feature_tensor/<DATASET>/whisper_small_cpft_s3_maskoff1_ft/single12_unfree1/fold2/
    ...

The reason why you did it.
------------
1. CPFT_stage3.py The old layerX/foldY/stage3_CPFT/ path is no longer used.
2. Stage3 v3 Core variables include:
   - Form: singleX
   - Number of unfrozen layers: unfreeZ
   - mask Injection deviation: maskoffK
3. singleX_unfreeZ is more suitable for downstream LRU as a selection catalogue;
   maskoffK Better suited as part of FT_MODEL_NAME to avoid mask_offset-exposed features overlaying each other.

Pitt subject-disjoint Amendments:
- Pitt Final export no longer follows old raw train/test membership;
- Rebuild final train/test using the same original_test_subjects_holdout strategy as the current Pitt subject-disjoint protocol;
- final train=184 subjects / 276 visits,final test=68 subjects / 122 visits,subject overlap=0;
- Write train_val_split_<file_tag>.json extra to ensure that local audio_id downstream of LRU corresponds to the order of the new feature.

This script is reproduced using the core features of frozen.py to extract logic:
1. librosa.load(sr=16000, mono=True)
2. 30s segment
3. 0.3 overlap
4. single layer extraction
5. labels / indices Save Format

Attention.
----
1. This script will be written in feature_tensor/ and should be rerun after pre-training in Stage3 is confirmed as available.
2. Does not cover original baseline:
       feature_tensor/<DATASET>/whisper_small/...
       feature_tensor/<DATASET>/whisper_small_ft/...
3. The recommended read profile for lru+transformer+lstm.py is written in feature_meta.json.
"""

# =============================================================================
# 0) Environment and random seeds: must be set before import torch
# =============================================================================
import os
import random
import numpy as np

# A visible binding card; it can be set at None if it is desired to be controlled from outside CUDA_VISIBLE_DEVICES.
GPU_ID = 4
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
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import librosa
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import WhisperProcessor, WhisperModel

"""
conda activate whisper_cpft_faiss
python CPFT_exteact.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.
"""

# =============================================================================
# 2) Parameter Settings: Change here as a priority
# =============================================================================
PROJECT_ROOT = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune"
MMSE_CPFT_ROOT = os.path.join(PROJECT_ROOT, "ft_CPFT")

PROJECT_ROOT2 = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT = os.path.join(PROJECT_ROOT2, "feature_tensor")
SPLIT_ROOT = os.path.join(PROJECT_ROOT, "split")

# Datasets: optional "ADReSS-IS2020", "ADReSSo21", "Pitt", "PROCESS-2"
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

# ---------------- Pitt subject-disjoint final split(PITT only) - - - -
# The same certainty strategy as the current Pitt subject-disjoint protocol:
# 1) Merge all the candidates in the original four lists of Pitt train/test visit;
# 2) Only visit which matches the valid MMSE with (subject_id, visit_index);
# 3) (a) All valid visites entering final test for > =1 valid visit subject in original test;
# 4) All remaining subjects enter final train;
# 5) final feature train/test Use the new local audio_id, without the old table of contents.
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

# Which layers need to be drawn. It must be consistent with the training level at Stage1/2/3.
LAYERS_TO_RUN = [12]

# Harmonised model root label with Stage1/2/3
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5"

# ===== Path variable aligned to CPFT_stage3.py =====
# UNFREEZE_ENCODER_LAYERS Indicates how many layers are unfrozen over the current expression single_layer.
# e.g. single12 + UNFREEZE_ENCODER_LAYERS=1 -> selection single12_unfree1;
#      single12 + UNFREEZE_ENCODER_LAYERS=4 -> selection Contents single12_unfree4.
# The batch runs different amounts of unfrozen layers.
UNFREEZE_ENCODER_LAYERS_LIST = [1,2,3,4,5,6,7,8,9,10,11,12]

# Keep single-value variables for old logic such as build_layer_name / get_feature_selection_tag / int(...).
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])



# Here's the variable name for CPFT_stage3.py.
# Note: The values in the list are not actual encoder tiles, but rather offsets to the mask relative to the " first trainingable layer ".
#   [1] -> checkpoint Subdirectories stage3_CPFT_maskoff1
#   [2] -> checkpoint Subdirectories stage3_CPFT_maskoff2
# If you go through more than one offset, this script will be saved to a different FT_MODEL_NAME, avoiding cross-over.
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]

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

# Stage3 Downstream feature FT_MODEL_NAME will no longer be written dead, and will be constructed automatically by mask_offset:
#   FT_MODEL_NAME       = whisper_small_mmse_cpft_s3_maskoffK
#   FEATURE_MODEL_TAG   = whisper_small_mmse_cpft_s3_maskoffK_ft
#   feature selection   = singleX_unfreeZ
"""
When running ks64_128, replace with:
CPFT_KS = [64, 128]
CPFT_EXPECTED_NUM_CLUSTERS = [64, 128]
CPFT_STAGE3_LAMBDA_MMSE = 0.0
CPFT_METHOD_TAG = "clean_cpft"
Concurrent settings:
PATH_PARAM_TAGS = ["ks64_128"]
STAGE1_SOURCE_PATH_PARAM_TAGS = []
"""
# ============================================================
# Path Parameter Label
# ============================================================
# Stage2 / Stage3 Reads the path label used for Stage1 checkpoint.
# It is recommended that [] be maintained to indicate Stage1 path to single12_unfree1/foldY/stage1_mmse/.
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

# For export meta only, it must be consistent with the true configuration of Stage2/Stage3.
CPFT_KS = [32, 64]
CPFT_EXPECTED_NUM_CLUSTERS = [32, 64]
CPFT_STAGE3_LAMBDA_MMSE = 0.05
CPFT_METHOD_TAG = ""


# Whisper Original local directory HuggingFace /.
# Stage3 checkpoint is Lightning ckpt, not HuggingFace catalogue;
# Therefore, the original Whisper structure is loaded and the Whisper weight in Stage3 ckpt is injected.
WHISPER_BASE_MODEL_NAME = "/home/user/whisper_lru_3/whisper_small"
LOCAL_FILES_ONLY = True



# An absolute path can be filled if you want to specify a Stage3 checkpoint manually.
# If None, search automatically under a new path Stage3 v3:
#   mmse_cpft_runs/<DATASET>/<MODEL_ROOT_TAG>/singleX_unfreeZ/foldY/stage3_CPFT_maskoffK/
# Priority is given to the most recent best ckpt, and back to last.ckpt if not found.
STAGE3_CKPT_OVERRIDE = None

# Feature filename still in frozen.py style:
#   feature_whisper_single12_0.3_30s_train.pt
FEATURE_PREFIX = "whisper"

# 30s + 0.3 overlap,Consistency with frozen.py.
SAMPLE_RATE = 16000
SEGMENT_SEC = 30
SEGMENT_LENGTH = int(SEGMENT_SEC * SAMPLE_RATE)
OVERLAP_RATE = 0.3
OV_STR = "0.3"
SEC_STR = f"{SEGMENT_SEC}s"

# Single-storey features are regularized. Keeps the same as your original Whisper baseline used settings.
# Optional: None / "layernorm" / "std" / "ln_post"
NORM_FOR_SINGLE = "ln_post"
DROP_INPUT_LAYER = True

# Quantified volume: consistent with frozen.py.
ENABLE_AUDIO_NORMALIZE = True

# Whether to save missing MMSE records and meta files.
SAVE_META = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# 3) Random Feeds and Common Tools
# =============================================================================
torch.manual_seed(400)
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


def natural_key(p: str):
    """
    Filename naturally sorting:
    - S002 Line up before S010
    - fold2 Line up before fold10
    """
    s = Path(str(p)).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


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


def build_stage3_subdir(mask_offset: int) -> str:
    """
    Subdirectorate name for Stage3 checkpoint consistent with CPFT_stage3.py.
    mask_offset=1 -> stage3_CPFT_maskoff1
    mask_offset=2 -> stage3_CPFT_maskoff2
    """
    return f"stage3_CPFT_maskoff{int(mask_offset)}"


def get_ft_model_name(mask_offset: int) -> str:
    """
    Construct FT_MODEL_NAME for lru+transformer+lstm.py.

    Put maskoffK in FT_MODEL_NAME instead of selection catalogue,
    This way selection remains singleX_unfreeZ, which is more naturally aligned to the read logic of UNFREE_LAYER_NUM.
    """
    return f"{MODEL_ROOT_TAG}"
    #return f"{MODEL_ROOT_TAG}_maskoff{int(mask_offset)}"


def get_feature_model_tag(mask_offset: int) -> str:
    """feature_tensor The real model directory name: FT_MODEL_NAME + _ft."""
    return f"ft_{get_ft_model_name(mask_offset)}"


def get_feature_selection_tag(single_layer: int) -> str:
    """feature_tensor The directory name under selection aligns with layer_name on Stage3 v3."""
    return build_layer_name(
        single_layer=single_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )


def canonical_wav_path(path: str) -> str:
    """Regularizes the wav path to avoid matching with the relative path / symbol link."""
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


def torch_load_compatible(path: str, map_location="cpu"):
    """
    torch.load compatible with PyTorch 2.6+.
    Some environments default to weights_only=True and result in Lightning ckpt reading failure;
    Therefore, priority is given to display weights_only=False, and the old version automatically returns when this parameter is not supported.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =============================================================================
# 4) MMSE Metadata reading: Consistency with frozen.py / Stage1
# =============================================================================
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
# 4.1) official train / official test Audio Directory
# =============================================================================
# Note:
# frozen.py The semantics of feature_*_train.pt / feature_*_test.pt are:
#   train.pt = Official collection of train
#   test.pt  = Official collection of test
#
# Therefore, although this script is loaded under fold_id fold-specific Stage3 checkpoint,
# However, the drawing object must be restored to official train + official test,
# No further preservation of fold train / fold val as train/test.
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

def _pick_first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> str:
    """A real line of DataFrame is found in the list of candidates."""
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"No candidate fields found {candidates},Current list:{list(df.columns)}")


def _safe_to_float(x):
    """Safely turns float; NA/ empty returns None."""
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
    """
    Harmonize MMSE map keys:
    - ADReSS Series: Use lower case stem
    - Pitt:Extracting number subject id
    - PROCESS-2:Remove _SFT / _PFT / _CTD postfix, return to partyid
    """
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
    """
    Parsing subject_id and visit_index from Pitt file names.
    For example:
        002-0c-en.wav -> ("2", 0)
        013-4c-en.wav -> ("13", 4)
    """
    if raw_name is None or pd.isna(raw_name):
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    return str(int(m.group(1))), int(m.group(2))


def _pitt_visit_to_mmse_col(visit_index: int) -> str:
    """
    Pitt visit Map to MMSE for listing:
        0 -> mms
        1 -> mmse2
        2 -> mmse3
        ...
    """
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

def _read_process2_mmse(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Parsing meta-info.csv of PROCESS-2, building MMSE maps of four subsets.
    """
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


def load_mmse_maps_for_dataset(dataset_key: str) -> Dict[str, Dict[Any, float]]:
    """Loads the MMSE map of the specified data set."""
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"Dataset not configured yet {dataset_key} MSME_META_PATHS")

    cfg = MMSE_META_PATHS[dataset_key]
    if dataset_key == "ADReSS-IS2020":
        return _read_adress_is2020_mmse(cfg)
    if dataset_key == "ADReSSo21":
        return _read_adresso21_mmse(cfg)
    if dataset_key == "Pitt":
        return _read_pitt_mmse(cfg)
    if dataset_key == "PROCESS-2":
        return _read_process2_mmse(cfg)

    raise ValueError(f"Unrealized datasets {dataset_key} read on MMSE.")



def resolve_mmse_for_entry(row: Dict[str, Any], dataset_key: str, mmse_maps: Dict[str, Dict[Any, float]]) -> Tuple[Optional[float], str]:
    """
    For official train / official test entry, get MMSE.

    Key amendments:
    - The old version only handles the fold train/ fold value, so the MMSE is taken by default from the train_cc/ train_cd;
    - This script will now be drawn from official train + official test;
    - Therefore MMSE must be taken from test_cc/ test_cd when split = "test".

    entry Need to include:
        wav_path
        group: "cc" Or "cd"
        split: "train" Or "test"
    """
    # fold json or use hand-held entry if you already have a target_mse.
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

    # official test test_cc / test_cd must be used;
    # fold val Remains within the official train division and therefore remains train_* if split_name == "val".
    is_official_test = split_name in ["test", "official_test", "test_cc", "test_cd"]

    if is_official_test:
        bucket = "test_cc" if group == "cc" else "test_cd"
    else:
        bucket = "train_cc" if group == "cc" else "train_cd"

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

# =============================================================================
# 5) fold json Read
# =============================================================================
def load_fold_entries(fold_json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read fold{i}.json generated by generate_five_folds_from_split.py.
    Return:
        train_entries, val_entries, fold_meta
    """
    if not os.path.isfile(fold_json_path):
        raise FileNotFoundError(f"Old json:{fold_json_path}")

    with open(fold_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    all_entries = meta.get("train_entries", [])
    if not isinstance(all_entries, list):
        raise ValueError("fold json is missing from train_entries list.")

    train_entries = []
    val_entries = []

    for row in all_entries:
        row = copy.deepcopy(row)
        if row.get("wav_path", None) in [None, ""]:
            continue
        row["wav_path"] = canonical_wav_path(row["wav_path"])
        row["wav_name"] = str(row.get("wav_name", Path(row["wav_path"]).name))
        split = str(row.get("split", "")).strip().lower()
        if split == "train":
            train_entries.append(row)
        elif split == "val":
            val_entries.append(row)

    train_entries = sorted(train_entries, key=lambda x: natural_key(x.get("wav_path", "")))
    val_entries = sorted(val_entries, key=lambda x: natural_key(x.get("wav_path", "")))

    if len(train_entries) == 0 or len(val_entries) == 0:
        raise ValueError(f"fold json Parse empty: train={len(train_entries)}, val={len(val_entries)}")

    return train_entries, val_entries, meta


def attach_mmse_and_filter(entries: List[Dict[str, Any]], dataset_key: str, mmse_maps: Dict[str, Dict[Any, float]], subset_name: str):
    """
    Enter entries back to target_mmse.
    wav without MMSE will skip while recording missing information.
    """
    valid_entries = []
    missing_records = []

    for row in entries:
        mmse, source = resolve_mmse_for_entry(row, dataset_key, mmse_maps)
        if mmse is None:
            rec = {
                "dataset": dataset_key,
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

def _is_process2_ctd_wav_path(wav_path: str, dataset_key: str = None) -> bool:
    """
    PROCESS-2 Filter for wav:
        Only the name of the document stem will be retained at the end of __CTD wav.

    Example:
        PROCESS-2_rec__001__CTD.wav -> Reservations
        PROCESS-2_rec__001__SFT.wav -> Skip
        PROCESS-2_rec__001__PFT.wav -> Skip

    Other data sets are not filtered.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None


def list_wav_files(audio_dir: str, dataset_key: str = None) -> List[str]:
    """
    Reads a wav file under a official subdirectorate andsorts it to natural_key.

    PROCESS-2:
        Read __CTD.wav only.

    Other data sets:
        Reads all.wav under the directory.

    This sorting is consistent with the logic of downstream local audio_id:
        train_cc Internal Sort Natural
        train_cd Internal Sort Natural
        test_cc Internal Sort Natural
        test_cd Internal Sort Natural
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

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
            f"official audio dir No eligible wav document:{audio_dir}, "
            f"dataset_key={dataset_key}"
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
    """
    Renumber official train_cc/train_cd/test_cc/test_cd as a entry list.

    The returned entry sequence must be identical to the downstream feature index:
    - train When saving: all before and all behind;
    - test When saving: test_cc all before, test_cd all behind.

    PROCESS-2:
        Only __CTD.wav will be retained.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

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
            "process2_task_filter": "CTD" if str(dataset_key).strip().lower() == "process-2" else "",
        })

    return entries



def _pitt_make_missing_visit_record(source_subset: str, wav_path: str, subject_id=None, visit_index=None) -> Dict[str, Any]:
    """Pitt subject-disjoint Unable to access MSME review visit at reconstruction."""
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


def _save_pitt_subject_disjoint_audit_extract(
    visit_df: pd.DataFrame,
    subject_df: pd.DataFrame,
    missing_records: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Saves Pitt split audit files with the calibre of the current Pitt subject-disjoint protocol / SCPA Stage1."""
    split_root = os.path.join(FEATURE_ROOT, "Pitt", "subject_disjoint_split")
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
    """
    Rebuilding in harmony with the current Pitt subject-disjoint protocol Pitt final train/test.

    Return:
      final train = 184 subjects / 276 valid visits
      final test  =  68 subjects / 122 valid visits

    train/test Inner audio_id is renumbered from 0 and sorted by final train_cc+training_cd/ test_cc+test_cd,
    To ensure that indices_*.pt is identical to the local audio_id in sidecar.
    """
    global _PITT_SUBJECT_DISJOINT_CACHE

    if _PITT_SUBJECT_DISJOINT_CACHE is not None:
        cached = _PITT_SUBJECT_DISJOINT_CACHE
        return copy.deepcopy(cached["train_entries"]), copy.deepcopy(cached["test_entries"])

    if PITT_SPLIT_POLICY != "original_test_subjects_holdout":
        raise ValueError(f"Unsupported PITT_SPLIT_POLICY={PITT_SPLIT_POLICY}")

    dataset_paths = DATASETS["Pitt"]
    mmse_maps = load_mmse_maps_for_dataset("Pitt")
    all_mmse_map = mmse_maps["train_cc"]

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
    if final_train_subjects & final_test_subjects:
        raise RuntimeError(f"[Pitt split] subject-disjoint Check failed: {sorted(final_train_subjects & final_test_subjects)}")

    final_records: Dict[str, List[Dict[str, Any]]] = {
        "train_cc": [], "train_cd": [], "test_cc": [], "test_cd": []
    }
    for rec in records:
        rec["final_split"] = "test" if rec["subject_id"] in final_test_subjects else "train"
        rec["final_subset"] = f"{rec['final_split']}_{rec['group']}"
        final_records[rec["final_subset"]].append(rec)

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
        ["final_split", "subject_id", "visit_index", "wav_name"], kind="stable"
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
    audit_paths = _save_pitt_subject_disjoint_audit_extract(visit_df, subject_df, missing_records)

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
    print("[Pitt subject-disjoint final split | SCPA Extract]")
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


def load_official_train_test_entries(dataset_option: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Construct a certain train/ official test inductions.

    Key amendments:
    - Distribution of stable official_audio_id before MMSE filters;
    - Even if S001.wav is filtered, it should not be allowed to move ahead of audio_id;
    - indices_*.pt id must be used to align with the wave lookup of split json/ LRU.
    """
    if dataset_option not in DATASETS:
        raise ValueError(f"DATASETS No configuration DATASET_OPTION={dataset_option}")

    # Pitt-MMSE Earmarked: The final feature must be exported to the new subject-disjoint split,
    # The old directory membership cannot continue to be considered as final train/test.
    if str(dataset_option) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return _build_pitt_subject_disjoint_train_test_entries()

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

    test_entries = (
        build_entries_from_official_dir(
            paths["test_cc"],
            split_name="test",
            group_name="cc",
            dataset_key=dataset_option,
        )
        + build_entries_from_official_dir(
            paths["test_cd"],
            split_name="test",
            group_name="cd",
            dataset_key=dataset_option,
        )
    )

    if str(dataset_option).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] official entries: "
            f"train={len(train_entries)}, test={len(test_entries)}"
        )

    if len(train_entries) == 0 or len(test_entries) == 0:
        raise ValueError(
            f"official train/test entries : Train={len(train_entries)}, test={len(test_entries)}"
        )

    # Filter to MMSE and fix official audio id.
    # train/test Separately numbered, corresponding to feature_*_train.pt / feature_*_test.pt.
    for i, row in enumerate(train_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    for i, row in enumerate(test_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    return train_entries, test_entries


# =============================================================================
# 6) Stage3 checkpoint Find and Load
# =============================================================================
def build_run_root(dataset_option: str, layer_id: int, fold_id: int = PROTOCOL_FOLD_ID) -> str:
    """
    Construct Stage3 checkpoint corresponding to run_root.

    Current path:
        mmse_cpft_runs/<DATASET>/<MODEL_ROOT_TAG>/<singleX_unfreeZ>

    _Other Organiser
        fold1
        official_train_test
    """
    layer_name = build_layer_name(
        single_layer=int(layer_id),
        unfreeze_layer_count=int(UNFREEZE_ENCODER_LAYERS),
        path_param_tags=PATH_PARAM_TAGS,
    )

    return os.path.join(
        MMSE_CPFT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        layer_name,
    )


def find_stage3_checkpoint(
    dataset_option: str,
    layer_id: int,
    fold_id: int,
    mask_offset: int,
    override_ckpt: Optional[str] = None,
) -> str:
    """
    Automatically search for Stage3 v3 checkpoint.

    Priority:
    1. override_ckpt (a) Non-empty direct use;
    2. stage3_CPFT_maskoffK Last.ckpt;
    3. stage3_CPFT_maskoffK/last.ckpt.
    """
    if override_ckpt not in [None, "", "None"]:
        ckpt = str(override_ckpt)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"STAGE3_CKPT_OVERRIDE Cannot initialise Evolution's mail component.{ckpt}")
        return ckpt

    run_root = build_run_root(dataset_option, layer_id, fold_id)
    stage3_dir = Path(run_root) / build_stage3_subdir(mask_offset)
    if not stage3_dir.exists():
        raise FileNotFoundError(
            "Stage3 v3 Directory does not exist: {}\n"
            "Please confirm DATANET_OPTION/ LAYERS_TO_RUN/ "
            "UNFREEZE_ENCODER_LAYERS / STAGE3_MASK_INJECT_AFTER_LAYERS / FOLDS_TO_RUN "
            "Consistent with this script.".format(stage3_dir)
        )

    all_ckpts = sorted(stage3_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    non_last = [p for p in all_ckpts if p.name != "last.ckpt"]

    if len(non_last) > 0:
        return str(non_last[0].resolve())

    last_ckpt = stage3_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt.resolve())

    raise FileNotFoundError(f"Yes. {stage3_dir} Not found below Stage3 checkpoint.")


def load_stage3_ckpt_into_whisper_model(model: WhisperModel, ckpt_path: str) -> None:
    """
    Load WhisperModel weights from Stage3 Lightning checkpoint to the current HuggingFace WhisperModel.

    Stage3 checkpoint Usually includes:
        model.encoder...
        model.decoder...
        cluster_heads...
        mmse_head...
        mask_emb...

    Currently, just loads the WhisperModel parameters themselves, so that cluster_heads/ mmse_head automatically ignores them.
    """
    ckpt = torch_load_compatible(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    target_sd = model.state_dict()
    candidate_prefixes = ["model.", "whisper.model.", "whisper.", ""]

    best_loaded = {}
    best_prefix = None

    for pref in candidate_prefixes:
        loaded = {}
        for k, v in state.items():
            if pref != "" and not str(k).startswith(pref):
                continue
            new_k = str(k)[len(pref):] if pref != "" else str(k)
            if new_k in target_sd and tuple(target_sd[new_k].shape) == tuple(v.shape):
                loaded[new_k] = v

        if len(loaded) > len(best_loaded):
            best_loaded = loaded
            best_prefix = pref

    if len(best_loaded) == 0:
        raise RuntimeError(f"No WhisperModel parameters matched from Stage3 checkpoint:{ckpt_path}")

    missing, unexpected = model.load_state_dict(best_loaded, strict=False)
    print(
        f"[Stage3-CKPT] loaded checkpoint: {ckpt_path}\n"
        f"[Stage3-CKPT] prefix={best_prefix!r}, loaded_params={len(best_loaded)}, "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


# =============================================================================
# 7) Normalize feature with single layer extraction Whisper
# =============================================================================
def apply_norm(h: torch.Tensor, mode: Optional[str], ln_post=None, skip_ln_post: bool = False):
    """
    Normalize [1,T,C] features.
    Consistency with frozen.py:
        None / "layernorm" / "std" / "ln_post"
    """
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
    sampling_rate: int = SAMPLE_RATE,
    overlap_rate: float = OVERLAP_RATE,
):
    """
    To press 30s + overlap for individual wav and to draw a specified single layer feature.

    Return:
        seg_features: List[Tensor[T,C]]
        seg_indices:  List[(audio_file_index, segment_index)]
    """
    y, _ = librosa.load(audio_file, sr=sampling_rate, mono=True, dtype=np.float32)

    if ENABLE_AUDIO_NORMALIZE:
        peak = float(np.max(np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-8:
            y = y / peak

    step_size = int(segment_length * (1 - overlap_rate))
    if step_size <= 0:
        raise ValueError(f"overlap_rate={overlap_rate} Caused step_size<=0")

    total_segments = int(len(y) / step_size) + (
        1 if len(y) % step_size > overlap_rate * segment_length else 0
    )
    total_segments = max(1, total_segments)

    seg_features = []
    seg_indices = []
    warned_double_ln_top = False

    for i in range(total_segments):
        start = i * step_size
        end = i * step_size + segment_length
        segment = y[start:end]

        # Consistent with frozen.py: Only the non-first paragraphs are filled forward.
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


def stack_with_time_padding(feature_list: List[torch.Tensor], target_t: Optional[int] = None, pad_value: float = 0.0):
    """
    Fill a set of [T,C] features to a uniform time length, and then stop into [N,T,C].
    Consistency with the frozen.py preservation format.
    """
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
            pad = torch.full((target_t - t, feat_dim), float(pad_value), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)
        elif t > target_t:
            feat = feat[:target_t]
        padded.append(feat)

    return torch.stack(padded, dim=0)


# =============================================================================
# 8) Individual fold / layer feature extraction and preservation
# =============================================================================

def extract_entries_features(entries: List[Dict[str, Any]], subset_name: str, processor, model, single_layer: int, ln_post, top_layer_index: int):
    """
    Draw segment-level features from official train / official test entries.

    Key amendments:
    - indices audio_id in which official_audio_id is allocated before filtering;
    - Could not use filtered enumerate(entries) number;
    - Otherwise S001.Wav will be filtered and the audio id will move forward as a whole, and LRU will be mismatched.
    """
    feature_list = []
    labels_list = []
    indices_list = []
    kept_entries = []

    for compact_idx, row in enumerate(tqdm(entries, desc=f"[{subset_name}] single{single_layer}", ncols=100)):
        wav_path = canonical_wav_path(row["wav_path"])
        mmse = float(row["target_mmse"])

        if not os.path.isfile(wav_path):
            raise FileNotFoundError(f"Could not close temporary folder: %s{wav_path}")

        if row.get("official_audio_id", None) is not None:
            audio_idx = int(row["official_audio_id"])
        elif row.get("audio_id", None) is not None:
            audio_idx = int(row["audio_id"])
        else:
            audio_idx = int(compact_idx)

        seg_feats, seg_idx = get_segment_features(
            audio_file=wav_path,
            audio_file_index=int(audio_idx),
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

        feature_list.extend(seg_feats)
        indices_list.extend(seg_idx)
        labels_list.extend([float(mmse)] * len(seg_feats))

        kept = copy.deepcopy(row)
        kept["audio_index_in_subset"] = int(audio_idx)
        kept["official_audio_id"] = int(audio_idx)
        kept["audio_id"] = int(audio_idx)
        kept["compact_audio_index_after_mmse_filter"] = int(compact_idx)
        kept["num_segments"] = int(len(seg_feats))
        kept_entries.append(kept)

    return feature_list, labels_list, indices_list, kept_entries


def get_save_root(dataset_option: str, single_layer: int, fold_id: int, mask_offset: int) -> str:
    """
    Construct the current lru+transformer+lstm.py readable non-old finetune feature directory.

    Output:
        feature_tensor/<DATASET>/<FT_MODEL_NAME>_ft/<selection>/

    No additional:
        fold1
        official_train_test
    """
    return os.path.join(
        FEATURE_ROOT,
        dataset_option,
        get_feature_model_tag(mask_offset),
        get_feature_selection_tag(single_layer),
    )

def _entry_match_keys(row: Dict[str, Any]) -> List[str]:
    """
    Generate multiple matches for key for wav entry to match the actual exported train_manifest back to fold json.
    """
    keys = []

    wav_path = row.get("wav_path", "")
    wav_name = row.get("wav_name", "")

    if wav_path not in [None, ""]:
        p = str(wav_path)
        try:
            keys.append(canonical_wav_path(p))
        except Exception:
            pass
        keys.append(Path(p).name.lower())
        keys.append(Path(p).stem.lower())

    if wav_name not in [None, ""]:
        n = str(wav_name)
        keys.append(Path(n).name.lower())
        keys.append(Path(n).stem.lower())

    out = []
    seen = set()
    for k in keys:
        k = str(k).strip()
        if k == "":
            continue
        if k not in seen:
            out.append(k)
            seen.add(k)

    return out


def build_fold_split_lookup_for_audio_cut(fold_json_path: str) -> Dict[str, str]:
    """
    Construct wav->train/val lookup from authority old json.
    Only used to generate local train_val_split_{file_tag}.json in the current feature directory.
    """
    if not os.path.isfile(fold_json_path):
        raise FileNotFoundError(f"fold json Cannot initialise Evolution's mail component.{fold_json_path}")

    with open(fold_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    rows = meta.get("train_entries", [])
    if not isinstance(rows, list):
        raise ValueError(f"fold json Missing train_entries: {fold_json_path}")

    lookup: Dict[str, str] = {}

    for row in rows:
        split = str(row.get("split", "")).strip().lower()
        if split not in ["train", "val"]:
            continue

        for key in _entry_match_keys(row):
            lookup[key] = split

    if len(lookup) == 0:
        raise ValueError(f"Failed to construct split lookup from fold json:{fold_json_path}")

    return lookup


def resolve_split_from_lookup(row: Dict[str, Any], lookup: Dict[str, str]) -> Optional[str]:
    """
    Try to explain whether the wav is a train or an val from a number of keys.
    """
    for key in _entry_match_keys(row):
        if key in lookup:
            return lookup[key]
    return None


def write_local_train_val_split_json(
    save_root: str,
    file_tag: str,
    fold_json_path: str,
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
) -> str:
    """
    Save LRU priority local read split metadata under the current feature directory:
        train_val_split_{file_tag}.json

    Role:
    - Use the actual export train_manifest;
    - audio_id Consistency with audio_id in indices_train.pt;
    - fold_val Use only at LRU/S5 downstream.
    """
    split_lookup = build_fold_split_lookup_for_audio_cut(fold_json_path)

    local_train_entries = []
    train_audio_ids = []
    val_audio_ids = []
    uncovered = []

    for row in train_manifest:
        split = resolve_split_from_lookup(row, split_lookup)

        audio_id = row.get("audio_index_in_subset", row.get("official_audio_id", row.get("audio_id", None)))
        if audio_id is None:
            raise ValueError(f"train_manifest entry Missing audio_id: {row}")

        audio_id = int(audio_id)

        if split not in ["train", "val"]:
            uncovered.append({
                "wav_name": row.get("wav_name", ""),
                "wav_path": row.get("wav_path", ""),
                "audio_id": int(audio_id),
            })
            continue

        kept = copy.deepcopy(row)
        kept["audio_id"] = int(audio_id)
        kept["official_audio_id"] = int(audio_id)
        kept["split"] = split
        local_train_entries.append(kept)

        if split == "train":
            train_audio_ids.append(int(audio_id))
        elif split == "val":
            val_audio_ids.append(int(audio_id))

    if len(uncovered) > 0:
        raise ValueError(
            f"Local split metadata generation failed: fold json does not overwrite actual export {len(uncovered)} A train audio."
            f"Example:{uncovered[:5]}"
        )

    local_test_entries = []
    test_audio_ids = []

    for row in test_manifest:
        audio_id = row.get("audio_index_in_subset", row.get("official_audio_id", row.get("audio_id", None)))
        if audio_id is None:
            raise ValueError(f"test_manifest entry Missing audio_id: {row}")

        audio_id = int(audio_id)

        kept = copy.deepcopy(row)
        kept["audio_id"] = int(audio_id)
        kept["official_audio_id"] = int(audio_id)
        kept["split"] = "test"
        local_test_entries.append(kept)
        test_audio_ids.append(int(audio_id))

    split_metadata = {
        "format_version": 1,
        "task_type": "mmse_regression",
        "dataset_key": DATASET_OPTION,
        "file_tag": str(file_tag),
        "feature_save_path": str(Path(save_root).resolve()),
        "source_fold_json_path": str(Path(fold_json_path).resolve()),

        "train_audio_ids": sorted(set(int(x) for x in train_audio_ids)),
        "val_audio_ids": sorted(set(int(x) for x in val_audio_ids)),
        "test_audio_ids": sorted(set(int(x) for x in test_audio_ids)),

        "train_entries": local_train_entries,
        "test_entries": local_test_entries,

        "feature_train_semantics": "official_train_all_after_missing_mmse_filter",
        "feature_test_semantics": "official_test_all_after_missing_mmse_filter",
        "audio_id_policy": "stable_official_audio_id_before_mmse_filter",
    }

    out_path = os.path.join(save_root, f"train_val_split_{file_tag}.json")
    save_json(split_metadata, out_path)

    print(f"[SAVE] local train/val split metadata -> {out_path}")
    print(
        f"       train_audio_ids={len(split_metadata['train_audio_ids'])}, "
        f"val_audio_ids={len(split_metadata['val_audio_ids'])}, "
        f"test_audio_ids={len(split_metadata['test_audio_ids'])}"
    )

    return out_path

def write_pitt_lru_audio_lookup_sidecar(
    save_root: str,
    file_tag: str,
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
) -> Optional[str]:
    """
    For Pitt SCPA features, write the same agreement as the current Pitt subject-disjoint protocol:
        train_val_split_<file_tag>.json

    This is necessary: subject-disjoint re-established local audio_id order no longer equals the old raw train/test directory order.
    Downstream LRU must give priority to reading this sidecar in order to restore the audio_id in indices to the current wav.
    """
    if str(DATASET_OPTION) != "Pitt" or not bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return None

    def _row(row: Dict[str, Any], split_name: str) -> Dict[str, Any]:
        audio_id = row.get("audio_index_in_subset", row.get("official_audio_id", row.get("audio_id", None)))
        if audio_id is None:
            raise RuntimeError(f"[Pitt sidecar] Missing audio_id: {row}")
        subject_id = row.get("subject_id")
        visit_index = row.get("visit_index")
        if subject_id is None or visit_index is None:
            subject_id, visit_index = _extract_pitt_subject_visit(Path(row["wav_path"]).stem)
        if subject_id is None or visit_index is None:
            raise RuntimeError(f"[Pitt sidecar] Unable to parse subject/visit: {row['wav_path']}")
        return {
            "audio_id": int(audio_id),
            "split": str(split_name),
            "wav_path": canonical_wav_path(row["wav_path"]),
            "wav_name": str(row.get("wav_name", Path(row["wav_path"]).name)),
            "subject_id": str(subject_id),
            "visit_index": int(visit_index),
            "group": str(row.get("group", "")),
            "target_mmse": float(row["target_mmse"]),
        }

    train_rows = [_row(row, "train") for row in train_manifest]
    test_rows = [_row(row, "test") for row in test_manifest]

    train_subjects = {r["subject_id"] for r in train_rows}
    test_subjects = {r["subject_id"] for r in test_rows}
    overlap = train_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"[Pitt sidecar] train/test subject overlap != 0: {sorted(overlap)}")

    # audio_id Must start continuously from 0 in their own split.
    train_ids = sorted({int(r["audio_id"]) for r in train_rows})
    test_ids = sorted({int(r["audio_id"]) for r in test_rows})
    if train_ids != list(range(len(train_rows))):
        raise RuntimeError(f"[Pitt sidecar] train audio_id Uncontinuous 0..N-1: {train_ids[:10]} ...")
    if test_ids != list(range(len(test_rows))):
        raise RuntimeError(f"[Pitt sidecar] test audio_id Uncontinuous 0..N-1: {test_ids[:10]} ...")

    payload = {
        "format_version": 3,
        "task_type": "mmse_regression",
        "dataset": "Pitt",
        "split_strategy": PITT_SPLIT_POLICY,
        "subject_disjoint": True,
        "train_test_subject_overlap": 0,
        "train_audio_ids": train_ids,
        "val_audio_ids": [],
        "train_entries": train_rows,
        "test_entries": test_rows,
    }

    sidecar_path = os.path.join(save_root, f"train_val_split_{file_tag}.json")
    save_json(payload, sidecar_path)
    print(f"[Pitt split] feature audio-id sidecar -> {sidecar_path}")
    return sidecar_path


def save_feature_bundle(
    save_root: str,
    single_layer: int,
    fold_id: int,
    mask_offset: int,
    fold_json_path: str,
    stage3_ckpt_path: str,
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_labels: torch.Tensor,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    train_manifest: List[Dict[str, Any]],
    test_manifest: List[Dict[str, Any]],
    missing_train: List[Dict[str, Any]],
    missing_test: List[Dict[str, Any]],
):
    """
    Saves official_train and official_test features in the format consumed by lru+transformer+lstm.py.

    Semantic:
        feature_*_train.pt = official_train All segments with MMSE tags
        feature_*_test.pt  = official_test All segments with MMSE tags
    """
    ensure_dir(save_root)

    mask_offset = int(mask_offset)
    ft_model_name = get_ft_model_name(mask_offset)
    feature_model_tag = get_feature_model_tag(mask_offset)
    feature_selection_tag = get_feature_selection_tag(single_layer)

    layer_name = build_layer_name(
        single_layer=single_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage3_subdir = build_stage3_subdir(mask_offset)

    file_tag = f"{FEATURE_PREFIX}_single{int(single_layer)}_{OV_STR}_{SEC_STR}"

    paths = {
        "feature_train": os.path.join(save_root, f"feature_{file_tag}_train.pt"),
        "feature_test": os.path.join(save_root, f"feature_{file_tag}_test.pt"),
        "labels_train": os.path.join(save_root, f"labels1_{file_tag}_train.pt"),
        "labels_test": os.path.join(save_root, f"labels2_{file_tag}_test.pt"),
        "indices_train": os.path.join(save_root, f"indices_{file_tag}_train.pt"),
        "indices_test": os.path.join(save_root, f"indices_{file_tag}_test.pt"),
    }

    torch.save(train_features, paths["feature_train"])
    torch.save(test_features, paths["feature_test"])
    torch.save(train_labels, paths["labels_train"])
    torch.save(test_labels, paths["labels_test"])
    torch.save(train_indices, paths["indices_train"])
    torch.save(test_indices, paths["indices_test"])

    pitt_sidecar_path = write_pitt_lru_audio_lookup_sidecar(
        save_root=save_root,
        file_tag=file_tag,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
    )

    if SAVE_META:
        meta = {
            "dataset_option": DATASET_OPTION,
            "run_protocol": str(RUN_PROTOCOL_TAG),
            "fold_id": None,
            "fold_disabled": True,
            "pitt_subject_disjoint": bool(str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
            "pitt_split_policy": (PITT_SPLIT_POLICY if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_train_subjects": (PITT_EXPECTED_FINAL_TRAIN_SUBJECTS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_test_subjects": (PITT_EXPECTED_FINAL_TEST_SUBJECTS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_train_visits": (PITT_EXPECTED_FINAL_TRAIN_VISITS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_expected_final_test_visits": (PITT_EXPECTED_FINAL_TEST_VISITS if str(DATASET_OPTION) == "Pitt" else None),
            "pitt_lru_audio_lookup_sidecar": pitt_sidecar_path,

            "single_layer": int(single_layer),
            "unfreeze_encoder_layers": int(UNFREEZE_ENCODER_LAYERS),
            "layer_name": str(layer_name),

            "mask_offset": int(mask_offset),
            "stage3_subdir": str(stage3_subdir),
            "feature_model_tag": str(feature_model_tag),
            "ft_model_name_for_ssm": str(ft_model_name),
            "feature_selection_tag_for_ssm": str(feature_selection_tag),
            "feature_file_tag_for_ssm": file_tag,

            "stage3_ckpt_path": str(stage3_ckpt_path),
            "whisper_base_model_name": str(WHISPER_BASE_MODEL_NAME),

            "sampling_rate": int(SAMPLE_RATE),
            "segment_sec": int(SEGMENT_SEC),
            "overlap_rate": float(OVERLAP_RATE),
            "norm_for_single": str(NORM_FOR_SINGLE),
            "drop_input_layer": bool(DROP_INPUT_LAYER),

            "train_num_wavs": int(len(train_manifest)),
            "test_num_wavs": int(len(test_manifest)),
            "train_num_segments": int(train_features.shape[0]),
            "test_num_segments": int(test_features.shape[0]),
            "train_feature_shape": list(train_features.shape),
            "test_feature_shape": list(test_features.shape),
            "missing_train": int(len(missing_train)),
            "missing_test": int(len(missing_test)),

            "feature_train_semantics": (
                "pitt_subject_disjoint_final_train"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_train_all_after_missing_mmse_filter"
            ),
            "feature_test_semantics": (
                "pitt_subject_disjoint_final_test_holdout"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_test_all_after_missing_mmse_filter"
            ),
            "audio_id_policy": (
                "local_audio_id_rebuilt_after_pitt_subject_disjoint_split"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "stable_official_audio_id_before_mmse_filter"
            ),
            "saved_files": paths,

            "ssm_recommended_config": {
                "FEATURE_SOURCE_MODES": ["finetune"],
                "FT_MODEL_NAME": str(ft_model_name),
                "FEATURE_MODEL_TAG": str(feature_model_tag),
                "FEATURE_SELECTION_TAG": str(feature_selection_tag),
                "FEATURE_FILE_TAG": file_tag,
                "SINGLE_LAYERS": [int(single_layer)],
                "UNFREE_LAYER_NUM": [int(UNFREEZE_ENCODER_LAYERS)],
                "STAGE3_MASK_INJECT_AFTER_LAYERS": [int(mask_offset)],
            },

            "CPFT_METHOD": str(CPFT_METHOD_TAG),
            "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
            "PATH_PARAM_SUFFIX": build_path_param_suffix(PATH_PARAM_TAGS),
            "STAGE1_SOURCE_PATH_PARAM_TAGS": list(STAGE1_SOURCE_PATH_PARAM_TAGS),
            "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(STAGE1_SOURCE_PATH_PARAM_TAGS),
            "CPFT_KS": [int(k) for k in CPFT_KS],
            "CPFT_EXPECTED_NUM_CLUSTERS": [int(k) for k in CPFT_EXPECTED_NUM_CLUSTERS],
            "CPFT_STAGE3_LAMBDA_MMSE": float(CPFT_STAGE3_LAMBDA_MMSE),
            "STAGE3_MASK_INJECT_AFTER_LAYERS": [int(x) for x in STAGE3_MASK_INJECT_AFTER_LAYERS],
        }

        save_json(meta, os.path.join(save_root, "feature_meta.json"))
        save_json({"train_manifest": train_manifest}, os.path.join(save_root, "train_manifest.json"))
        save_json({"test_manifest": test_manifest}, os.path.join(save_root, "test_manifest.json"))

        missing = {
            "missing_train": missing_train,
            "missing_test": missing_test,
        }
        save_json(missing, os.path.join(save_root, "missing_mmse_records.json"))

        save_text(str(stage3_ckpt_path), os.path.join(save_root, "stage3_ckpt_path.txt"))

        save_text(
            (
                "Pitt subject-disjoint protocol: final train/test are rebuilt by subject; no fold split json is used by lru+transformer+lstm.py.\n"
                if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
                else "official_train_test protocol: no fold split json is used by lru+transformer+lstm.py.\n"
            ),
            os.path.join(save_root, "protocol_note.txt"),
        )

    print(f"[SAVE] {save_root}")
    if str(DATASET_OPTION) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT:
        print("  semantics: train=Pitt subject-disjoint final train, test=Pitt subject-disjoint final test holdout")
    else:
        print("  semantics: train=official_train_all_after_missing_mmse_filter, test=official_test_all_after_missing_mmse_filter")
    for k, p in paths.items():
        print(f"  - {k}: {p}")



def run_one_fold_layer(single_layer: int, fold_id: int, mask_offset: int) -> None:
    """
    Full export process for individual layer + mask_offset.

    Current protocol:
    - Read no fold Stage3 checkpoint;
    - (a) The extraction of official_train + official_test features;
    - train/test Separate filters are missing MMSE;
    - Saves to lru+transformer+lstm.py readable no-old feed_tensor directory.
    """
    mask_offset = int(mask_offset)
    fold_id = int(PROTOCOL_FOLD_ID)

    fold_json_path = ""

    save_root = get_save_root(DATASET_OPTION, single_layer, fold_id, mask_offset)

    layer_name = build_layer_name(
        single_layer=single_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage3_dir = os.path.join(
        build_run_root(
            dataset_option=DATASET_OPTION,
            layer_id=int(single_layer),
            fold_id=int(PROTOCOL_FOLD_ID),
        ),
        build_stage3_subdir(mask_offset),
    )

    stage3_subdir = build_stage3_subdir(mask_offset)
    ft_model_name = get_ft_model_name(mask_offset)
    feature_model_tag = get_feature_model_tag(mask_offset)

    print("\n" + "=" * 100)
    print(f"[RUN] DATASET_OPTION = {DATASET_OPTION}")
    print(f"[RUN] RUN_PROTOCOL   = {RUN_PROTOCOL_TAG}")
    print(f"[RUN] single_layer   = {single_layer}")
    print(f"[RUN] layer_name     = {layer_name}")
    print(f"[RUN] mask_offset    = {mask_offset}")
    print(f"[RUN] stage3_subdir  = {stage3_subdir}")
    print(f"[RUN] stage3_dir     = {stage3_dir}")
    print(f"[RUN] fold disabled  = True")
    print(f"[RUN] FT_MODEL_NAME  = {ft_model_name}")
    print(f"[RUN] FEATURE_MODEL  = {feature_model_tag}")
    print(f"[RUN] save_root      = {save_root}")
    print(
        "[RUN] extract target = Pitt subject-disjoint final train + final test"
        if DATASET_OPTION == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
        else "[RUN] extract target = official train + official test"
    )
    print("=" * 100)

    train_entries_raw, test_entries_raw = load_official_train_test_entries(DATASET_OPTION)
    mmse_maps = load_mmse_maps_for_dataset(DATASET_OPTION)

    train_entries, missing_train = attach_mmse_and_filter(
        train_entries_raw,
        DATASET_OPTION,
        mmse_maps,
        subset_name="train",
    )
    test_entries, missing_test = attach_mmse_and_filter(
        test_entries_raw,
        DATASET_OPTION,
        mmse_maps,
        subset_name="test",
    )

    if len(train_entries) == 0 or len(test_entries) == 0:
        raise ValueError(
            f"MMSE Filter official train/test empty:"
            f"train={len(train_entries)}, test={len(test_entries)}"
        )

    stage3_ckpt_path = find_stage3_checkpoint(
        dataset_option=DATASET_OPTION,
        layer_id=int(single_layer),
        fold_id=int(PROTOCOL_FOLD_ID),
        mask_offset=int(mask_offset),
        override_ckpt=STAGE3_CKPT_OVERRIDE,
    )

    print(f"[CKPT] Stage3 checkpoint = {stage3_ckpt_path}")
    print(f"[DATA] official train wavs raw = {len(train_entries_raw)}")
    print(f"[DATA] official test wavs raw  = {len(test_entries_raw)}")
    print(f"[DATA] official train valid MMSE = {len(train_entries)}")
    print(f"[DATA] official test valid MMSE  = {len(test_entries)}")
    print(f"[DATA] missing train = {len(missing_train)}, missing test = {len(missing_test)}")

    processor = WhisperProcessor.from_pretrained(
        WHISPER_BASE_MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    model = WhisperModel.from_pretrained(
        WHISPER_BASE_MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )
    load_stage3_ckpt_into_whisper_model(model, stage3_ckpt_path)

    model.eval().to(DEVICE)
    ln_post = model.encoder.layer_norm
    top_layer_index = int(model.config.encoder_layers)

    with torch.no_grad():
        train_feat_list, train_labels_list, train_indices_list, train_manifest = extract_entries_features(
            entries=train_entries,
            subset_name="official_train",
            processor=processor,
            model=model,
            single_layer=int(single_layer),
            ln_post=ln_post,
            top_layer_index=top_layer_index,
        )

        test_feat_list, test_labels_list, test_indices_list, test_manifest = extract_entries_features(
            entries=test_entries,
            subset_name="official_test",
            processor=processor,
            model=model,
            single_layer=int(single_layer),
            ln_post=ln_post,
            top_layer_index=top_layer_index,
        )

    if len(train_feat_list) == 0 or len(test_feat_list) == 0:
        raise ValueError(
            f"When the feature is drawn, official train/test segment shall be empty:"
            f"train={len(train_feat_list)}, test={len(test_feat_list)}"
        )

    global_t = max(
        max(int(x.shape[0]) for x in train_feat_list),
        max(int(x.shape[0]) for x in test_feat_list),
    )
    print(f"[PAD] global target_t = {global_t}")

    train_features = stack_with_time_padding(train_feat_list, target_t=global_t)
    test_features = stack_with_time_padding(test_feat_list, target_t=global_t)

    train_labels = torch.tensor(train_labels_list, dtype=torch.float32)
    test_labels = torch.tensor(test_labels_list, dtype=torch.float32)

    train_indices = np.asarray(train_indices_list, dtype=np.int64)
    test_indices = np.asarray(test_indices_list, dtype=np.int64)

    print(f"[SHAPE] train_features = {tuple(train_features.shape)}")
    print(f"[SHAPE] test_features  = {tuple(test_features.shape)}")
    print(f"[SHAPE] train_labels   = {tuple(train_labels.shape)}")
    print(f"[SHAPE] test_labels    = {tuple(test_labels.shape)}")
    print(f"[SHAPE] train_indices  = {tuple(train_indices.shape)}")
    print(f"[SHAPE] test_indices   = {tuple(test_indices.shape)}")

    save_feature_bundle(
        save_root=save_root,
        single_layer=int(single_layer),
        fold_id=int(PROTOCOL_FOLD_ID),
        mask_offset=int(mask_offset),
        fold_json_path=fold_json_path,
        stage3_ckpt_path=stage3_ckpt_path,
        train_features=train_features,
        test_features=test_features,
        train_labels=train_labels,
        test_labels=test_labels,
        train_indices=train_indices,
        test_indices=test_indices,
        train_manifest=train_manifest,
        test_manifest=test_manifest,
        missing_train=missing_train,
        missing_test=missing_test,
    )

    print(f"[DONE] official_train_test single{single_layer} maskoff{mask_offset}")

# =============================================================================
# 9) Main entrance
# =============================================================================
def main() -> None:
    """
    DATANET_OPTIONS, UNFREEZE_ENCODER_LEYERS_LIST, LAYERS_TO_RUN
    And STAGE3_MASK_INJECT_AFTER_LAYERS.
    No more folding.
    """
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS There is a minimum need for a data set to be empty.")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST To be empty, a minimum number of unfrozen layers is required.")

    print("[CONFIG]")
    print("  PROJECT_ROOT                    =", PROJECT_ROOT)
    print("  DATASET_OPTIONS                 =", DATASET_OPTIONS)
    print("  RUN_PROTOCOL_TAG                =", RUN_PROTOCOL_TAG)
    print("  MODEL_ROOT_TAG                  =", MODEL_ROOT_TAG)
    print("  UNFREEZE_ENCODER_LAYERS_LIST    =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  STAGE3_MASK_INJECT_AFTER_LAYERS =", STAGE3_MASK_INJECT_AFTER_LAYERS)
    print("  WHISPER_BASE_MODEL              =", WHISPER_BASE_MODEL_NAME)
    print("  LAYERS_TO_RUN                   =", LAYERS_TO_RUN)
    print("  SEGMENT_SEC                     =", SEGMENT_SEC)
    print("  OVERLAP_RATE                    =", OVERLAP_RATE)
    print("  NORM_FOR_SINGLE                 =", NORM_FOR_SINGLE)
    print("  DEVICE                          =", DEVICE)
    print("  fold disabled                   = True")
    print("  protocol subdir disabled        = True")

    completed_items = []

    for dataset_option in [_canonical_dataset_option(x) for x in DATASET_OPTIONS]:
        for unfreeze_count in [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "=" * 100)
            print("[EXTRACT BATCH ITEM]")
            print("  DATASET_OPTION =", DATASET_OPTION)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("=" * 100)

            for layer_id in [int(x) for x in LAYERS_TO_RUN]:
                for mask_offset in [int(x) for x in STAGE3_MASK_INJECT_AFTER_LAYERS]:
                    run_one_fold_layer(
                        single_layer=layer_id,
                        fold_id=PROTOCOL_FOLD_ID,
                        mask_offset=mask_offset,
                    )
                    completed_items.append({
                        "dataset": DATASET_OPTION,
                        "unfreeze": int(UNFREEZE_ENCODER_LAYERS),
                        "layer": int(layer_id),
                        "mask_offset": int(mask_offset),
                        "feature_root": os.path.join(
                            FEATURE_ROOT,
                            DATASET_OPTION,
                            get_feature_model_tag(mask_offset),
                            get_feature_selection_tag(layer_id),
                        ),
                    })

    print("\n" + "=" * 100)
    print("[ALL DONE]")
    print("Output directory:")
    for item in completed_items:
        print(
            f"  - dataset={item['dataset']} "
            f"unfree={item['unfreeze']} "
            f"single={item['layer']} "
            f"maskoff={item['mask_offset']} "
            f"-> {item['feature_root']}"
        )

    print()
    print("lru+transformer+lstm.py Suggested settings:")
    print('  FEATURE_SOURCE_MODES = ["finetune"]')
    print(f"  SINGLE_LAYERS = {LAYERS_TO_RUN}")
    print(f"  UNFREE_LAYER_NUM = {UNFREEZE_ENCODER_LAYERS_LIST}")
    for mask_offset in [int(x) for x in STAGE3_MASK_INJECT_AFTER_LAYERS]:
        print(f'  # maskoff{mask_offset}')
        print(f'  FT_MODEL_NAME = "{get_ft_model_name(mask_offset)}"')
    print("=" * 100)


if __name__ == "__main__":
    main()