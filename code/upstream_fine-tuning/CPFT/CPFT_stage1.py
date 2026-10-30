
# -*- coding: utf-8 -*-
"""
CPFT_stage1.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.

MMSE-aware CPFT Stage 1.
Pitt subject-disjoint configuration:
- Exclude the original Pitt training directory from Stage 1 input;
- Use the same original_test_subjects_holdout strategy as the current Pitt subject-disjoint protocol;
- Final train: 184 subjects / 276 valid visits; final test: 68 subjects / 122 valid visits;
- Train Stage 1 only on the final training split; keep the final test split fully excluded.

Core changes:
1. Stop using original Stage1 valid_split+random_split, but read it directly
   generate_five_folds_from_split.py Generate authoritative fold{i}.json.
2. The task was changed from emotional classification to MMSE-aware learning:
   - utt_head: MMSE (retrieval) after the prediction has been standardized
   - frame_head: Projection of coarse-grained MMSE bin (classification)
3. The top provides a clear " parameter setting area ", which can be changed directly like frozen.py / lru+transformer+lstm.py.
4. Supported the passage of the past layers / folds and wrote the results to:
   mmse_cpft_runs/<DATASET>/whisper_small_mmse_cpft/layerX/foldY/stage1_mmse/

Note:
- The current implementation is guided by the principle that " variables are opened first, default is linked first".
- Stage1 is controlled by the current layer_id by default.
- If a more detailed layer search is undertaken in the future, the stage-specific configuration can be expanded in the parameters area without rewrite the main frame.
"""


import os
import json
import math
import copy
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
GPU_ID = 3
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data.dataloader import default_collate

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from transformers import WhisperConfig, WhisperModel, WhisperFeatureExtractor

"""
conda activate whisper_cpft_faiss
python CPFT_stage1.py

CPFT stands for Cluster Pseudo-label Fine-tuning with a Task objective.
"""
# ============================================================
# 0) Parameter Settings (Priority direct change here, no command line required)
# ============================================================

PROJECT_ROOT = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune"
MMSE_CPFT_ROOT = os.path.join(PROJECT_ROOT, "ft_CPFT")

# Pitt subject-disjoint The audit documents are centrally placed under the main project feature_tensor/Pitt.
# Consistent with the current Pitt subject-disjoint protocol / SSA.py downstream sidecar calibre.
PROJECT_ROOT_MAIN = "/home/user/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT_MAIN, "feature_tensor")

# ---------------- Pitt subject-disjoint final split(PITT only) - - - -
# The same certainty strategy as the current Pitt subject-disjoint protocol:
# 1) Merge all the candidates in the original four lists of Pitt train/test visit;
# 2) Only visit which matches the valid MMSE with (subject_id, visit_index);
# 3) As long as a subject appears in the original test > =1 valid visit, all valid observations of the subject enter final test;
# 4) All remaining subjects enter final train;
# 5) Not physically moving/deleting the original wav, re-establishing logic only in memory.
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

# Stage1 There will be several trips to unfreeze/layer; Pitt final split in the same process will be constructed only once.
_PITT_SUBJECT_DISJOINT_CACHE = None


# Optional: "ADReSS-IS2020","ADReSSo21", "Pitt", "PROCESS-2"
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

# Public Controls
# If only to maintain the original scope of CPFT, it can be changed back to [1,2,3,4,5,6,7,8,9,10,11];
# But CPFT_exteact.py must be consistent here.
LAYERS_TO_RUN = [12]

# Stage1:Indicates the extent of the unfrozen layer / Whisper
# Stage1:Indicates the extent of the unfrozen layer / Whisper
STAGE1_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN

# Refer to VanillaFT.py:
# UNFREEZE_ENCODER_LAYERS This means "how many layers of unfrozen continue to include the current player_id itself relative to the current player".
#
# For example:
#   UNFREEZE_ENCODER_LAYERS = 1
#       layer12 -> Unfreeze [12]
#       Save Directory -> single12_unfree1/fold1/stage1_mmse/
#
#   UNFREEZE_ENCODER_LAYERS = 2
#       layer12 -> Unfreeze [11, 12]
#       Save Directory -> single12_unfree2/fold1/stage1_mmse/
#
#   UNFREEZE_ENCODER_LAYERS = 4
#       layer12 -> Unfreezing [9, 10, 11, 12]
#       Save Directory - > single12_unfree4/fold1/stage1_mmse/
#
# The first version is proposed to be 1 if it is to be consistent with <freezing layer12 only.
# If you want to repeat your previous conduct STAGE1_WHISPER_TRAIN_LAYER_START=9, END=12, set 4.
# The batch runs different amounts of unfrozen layers.
UNFREEZE_ENCODER_LAYERS_LIST = [8]

# Keep single-value variables for old logic such as build_layer_name / run_config / int(...).
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# The list of specific defrozen layers that are actually being desolated is automatically generated by layer_id in build_run_config(...).
RESOLVED_UNFREEZE_ENCODER_LAYERS = []

# Current semantic label: protocol
# - For meta / manifest / log;
# - Do not enter the mmse_cpft_runs path;
# - Do not enter the feature_tensor path;
# - CPFT The intermediate output path no longer contains /official_train_test.
RUN_PROTOCOL_TAG = "official_train_test"

# Compatibility with the place value of the old field; do not participate in path and training divisions.
PROTOCOL_FOLD_ID = 1
FOLDS_TO_RUN = [PROTOCOL_FOLD_ID]

# Pre-training experimental root labels
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5"


# ============================================================
# Path Parameter Label
# ============================================================
# Stage2 / Stage3 Reads the path label used for Stage1 checkpoint.
# It is recommended that [] be maintained to indicate Stage1 path to single12_unfree1/foldY/stage1_mmse/.
# If Stage1 itself carries ks suffix, the same label is added here.
# ["ks32_64","clus1.0","mmse0.05","last","s1epo10","s3epo15","uttcap128"]
STAGE1_SOURCE_PATH_PARAM_TAGS = []

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


# The layer variables that will be used in the future phase will be retained only for the "expression layer" to allow for subsequent meta-information records.
# Note: The position of the mask input for Stage3 is not defined in Stage1.
# Stage3 v3 Automatically interprets it according to UNFREEZE_ENCODER_LAYERS and STAGE3_MASK_INJECT_AFTER_LAYERS.
STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE3_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN
STAGE4_OUTPUT_LAYERS_TO_RUN = LAYERS_TO_RUN

# ---------------- Models and trainings super-intended ----------------
WHISPER_MODEL_NAME = "/home/user/whisper_lru_3/whisper_small"

# For extraction: current Stage1 default single layer; if multiple layers are to be integrated in the future, repr_layers can be changed to a list / string
STAGE1_REPR_POOL = "mean"        # "mean" / "scalar_mix"
TARGET_SEC = 30.0                # Align the 30s view with the same script

# MMSE Crude particle fraction drum boundary (left closed right open), default 5 slot:
# [0,11) [11,16) [16,21) [21,26) [26,31)
MMSE_BIN_EDGES = [0.0, 11.0, 16.0, 21.0, 26.0, 31.0]

# Losses and structure
LAMBDA_UTT = 1.0
LAMBDA_FRAME = 0.5
HEAD_HIDDEN = 128
HEAD_DROPOUT = 0.30
HUBER_DELTA = 1.0
LABEL_SMOOTHING = 0.05

# Optimize relevance
LR_HEAD = 1e-4
LR_WHISPER = 1e-5
WEIGHT_DECAY = 5e-2
WARMUP_RATIO = 0.10
FLAT_RATIO = 0.20
MIN_LR_RATIO = 0.05

#++++++++++++++++++++++++++++++++++++++++++++++
# Training control
BATCH_SIZE = 8
NUM_WORKERS = 4

# no-val-last:
# Stage1 fold_val will not be used upstream, early stopping will not be done, and valid_loss will not be selected best.
# Fixed training STAGE1_MAX_EPOCHS epoch and preservation last.ckpt.
STAGE1_USE_FOLD_VAL = False
STAGE1_MAX_EPOCHS = 10
STAGE1_CHECKPOINT_SELECTION = "last"

# TRAINING_STEP Estimates of the number of steps to be used only for LR scheduler;
# run_one_experiment will be automatically updated according to train_entries and STAGE1_MAX_EPOCHS.
TRAINING_STEP = 2000

PRECISION = 16                   # 16 / 32
CHECK_VAL_EVERY_N_EPOCH = 1
EARLY_STOP_PATIENCE = 0
SAVE_TOP_K = 0
ACCUMULATE_GRAD_BATCHES = 1
GRAD_CLIP = 1.0
#++++++++++++++++++++++++++++++++++++++++++++++

# Other
SEED = 42
ENABLE_CSV_LOGGER = True
RESUME_CHECKPOINT = None

# Whether to make the original MMSE list json / to facilitate review
SAVE_MANIFESTS = True

# ============================================================
# Stage1 Quality evaluation output
# ============================================================
# Role:
# 1. Automatically reading CSVLogger metrics.csv;
# 2. Stag1 valid_los / valid_utt_mae_raw /rain-valid Gap;
# 3. Output quality_report.json, quality_summary.csv, quality_decision.txt;
# 4. To determine whether the current Stage1 checkpoint is worth entering Stage2.
ENABLE_STAGE_QUALITY_EVAL = False
QUALITY_REPORT_FILENAME = "quality_report.json"
QUALITY_SUMMARY_FILENAME = "quality_summary.csv"
QUALITY_DECISION_FILENAME = "quality_decision.txt"

# ============================================================
# 1) Constant
# ============================================================
os.environ.pop("PL_TRAINER_AMP_BACKEND", None)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
TARGET_30S_SAMPLES = int(TARGET_SEC * SAMPLE_RATE)


# ============================================================
# 2) Random Feeds and Common Tools
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
    pl.seed_everything(seed, workers=True)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def ensure_dir(path: str) -> None:
    """
    Creates a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def natural_key(x: str):
    """
    Natural ranking key: Placing fold2 before fold10 and S002 before S010.
    """
    s = Path(str(x)).name
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]


def save_json(obj: Dict[str, Any], path: str) -> None:
    """
    Saves json, automatically creates a parent directory.
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
        f.write(text)


def compute_valid_token_length(wav_len_samples: int) -> int:
    """
    Maps the true and valid length of waveform (specify points) to token length of Whisper encoder.

    Whisper:
        mel frames ~= ceil(len / hop)
        token len  ~= (mel_frames + 1) // 2
    """
    if wav_len_samples <= 0:
        return 1
    n_mel_valid = int(math.ceil(wav_len_samples / float(HOP_LENGTH)))
    n_mel_valid = min(MAX_MEL_FRAMES, max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)


def bin_mmse(mmse_value: float, bin_edges: List[float]) -> int:
    """
    Maps successive MMSE into thick particles bin.
    For example, default drums:
        [0,10], [11,15], [16,20], [21,25], [26,30]
    returns to 0-based bin id.
    """
    x = float(mmse_value)
    edges = [float(e) for e in bin_edges]
    if len(edges) < 2:
        raise ValueError("MMSE_BIN_EDGES At least two borders are needed.")
    # np.digitize returns 1..N less 1 to 0..N-1
    bid = int(np.digitize([x], edges[1:-1], right=False)[0])
    bid = max(0, min(len(edges) - 2, bid))
    return bid


# ============================================================
# 3) Read logic for MMSE metadata aligned to frozen.py
# ============================================================
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


def _pick_first_existing_column(df: pd.DataFrame, candidates: List[str]) -> str:
    """
    In the list of candidates, find the column that really exists in DataFrame.
    """
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"No candidate fields found {candidates},Current list:{list(df.columns)}")


def _safe_to_float(x):
    """
    Securely converts the text / Excel cells to float.
    Empty, NA, None returns None.
    """
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


def _normalize_mmse_key(dataset_key: str, raw_id) -> Optional[str]:
    """
    Harmonizes the MMSE index keys for different data sets.

    PROCESS-2:
        meta-info.csv It's participant-level ID:.
            PROCESS-2_rec__001
        wav stem It's task-level ID:.
            PROCESS-2_rec__001__SFT / __PFT / __CTD
        The end of the mandate would therefore need to be removed.
    """
    if raw_id is None or pd.isna(raw_id):
        return None

    s = str(raw_id).strip()
    s = Path(s).stem

    dataset_key_norm = str(dataset_key).strip().lower()

    if dataset_key_norm == "pitt":
        import re
        m = re.search(r"\d+", s)
        if m is None:
            return None
        return str(int(m.group(0)))

    if dataset_key_norm == "process-2":
        import re
        s = re.sub(r"__(SFT|PFT|CTD)$", "", s, flags=re.IGNORECASE)
        return s.lower()

    return s.lower()


def _extract_pitt_subject_visit(raw_name: str) -> Tuple[Optional[str], Optional[int]]:
    """
    Remove (subject_id, visit_index) from the name of the document Pitt.
    For example:
        002-0c-en.wav -> ("2", 0)
        013-4c-en.wav -> ("13", 4)
    """
    import re
    if raw_name is None:
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    subject_key = str(int(m.group(1)))
    visit_index = int(m.group(2))
    return subject_key, visit_index


def _pitt_visit_to_mmse_col(visit_index: int) -> str:
    """
    Pitt readme Reciprocal:
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
    """
    Construct { norms ID: mmse} from DataFrame columns.
    """
    out = {}
    for _, row in df.iterrows():
        key = _normalize_mmse_key(dataset_key, row[id_col])
        val = _safe_to_float(row[mmse_col])
        if key is None or val is None:
            continue
        out[key] = float(val)
    return out


def _read_adress_is2020_mmse(cfg: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """
    Read txt MMSE metadata from ADReSS-IS2020.
    """
    def _read_txt(txt_path: str) -> Dict[str, float]:
        out = {}
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if (not line) or line.startswith("ID"):
                    continue
                parts = [x.strip() for x in line.split(";")]
                sid = _normalize_mmse_key("ADReSS-IS2020", parts[0])
                mmse_val = _safe_to_float(parts[-1])
                if sid is None or mmse_val is None:
                    continue
                out[sid] = float(mmse_val)
        return out

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

    meta-info.csv Need to include:
        IDs
        diagnosis
        MMSE
        Split

    Directory map:
        HC -> cc
        MCI / Dementia -> cd
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
    """
    Read MMSE csv at ADReSSo21.
    """
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
    """
    Read visit-level MMSE at Pitt.
    Return:
        {(subject_id, visit_index): mmse}
    """
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
            import re
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
    """
    Loads MMSE maps by data set.
    """
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"Current unconfigurated dataset {dataset_key} is the MMSE metadata path.")

    cfg = MMSE_META_PATHS[dataset_key]
    if dataset_key == "ADReSS-IS2020":
        return _read_adress_is2020_mmse(cfg)
    if dataset_key == "ADReSSo21":
        return _read_adresso21_mmse(cfg)
    if dataset_key == "Pitt":
        return _read_pitt_mmse(cfg)
    if dataset_key == "PROCESS-2":
        return _read_process2_mmse(cfg)

    raise ValueError(f"Unrealized datasets {dataset_key} The MMSE parsing.")

# ============================================================
# 3.5) official train/test Audio Directory Read
# ============================================================
DATASETS: Dict[str, Dict[str, str]] = {
    "PROCESS-2": {
        "train_cc": "/home/user/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/user/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/user/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/user/Downloads/PROCESS-2/test/cd",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "ADReSSo21": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "Pitt": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence",
        "test_cd": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence",
    },
}


def canonical_wav_path(path: str) -> str:
    """Regularizes the wav path to avoid matching with the relative path / symbol link."""
    return str(Path(str(path)).expanduser().resolve())


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
    Reads the wav file in official sub-directories and sorts it in natural_key.

    PROCESS-2:
        Read __CTD.wav only.

    Other data sets:
        Reads all.wav under the directory.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    audio_dir = str(audio_dir)
    if not os.path.isdir(audio_dir):
        raise FileNotFoundError(f"official audio dir Cannot initialise Evolution's mail component.{audio_dir}")

    files = []
    for name in os.listdir(audio_dir):
        if not str(name).lower().endswith(".wav"):
            continue

        wav_path = str(Path(audio_dir) / name)

        if not _is_process2_ctd_wav_path(wav_path, dataset_key=dataset_key):
            continue

        files.append(wav_path)

    files = sorted(files, key=natural_key)

    if len(files) == 0:
        raise RuntimeError(
            f"official audio dir No eligible wav document:{audio_dir}, "
            f"dataset_key={dataset_key}"
        )

    if str(dataset_key).strip().lower() == "process-2":
        print(f"[PROCESS-2][CTD] {audio_dir}: Read only CTD wav, number of documents = {len(files)}")

    return files


def build_entries_from_official_dir(
    audio_dir: str,
    split_name: str,
    group_name: str,
    dataset_key: str = None,
) -> List[Dict[str, Any]]:
    """
    Renumber official train_cc/train_cd/test_cc/test_cd as a entry list.

    PROCESS-2:
        Only __CTD.wav will be retained.

    Stage1 I'll just use the split_name=train'.
    """
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    split_name = str(split_name).strip().lower()
    group_name = str(group_name).strip().lower()

    if split_name not in {"train", "test"}:
        raise ValueError(f"split_name Must be train/test, current ={split_name}")
    if group_name not in {"cc", "cd"}:
        raise ValueError(f"group_name must be cc/cd, current ={group_name}")

    entries: List[Dict[str, Any]] = []
    for wav_path in list_wav_files(audio_dir, dataset_key=dataset_key):
        wav_path = canonical_wav_path(wav_path)
        entries.append({
            "wav_path": wav_path,
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


def _build_pitt_subject_disjoint_train_test_entries() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Rebuilding in harmony with the current Pitt subject-disjoint protocol Pitt final train/test.

    Stage1 Only the returned final train will be used, but it's still fully constructed final test,
    To test 398 visits/ 252 subjects / 184 tain subjects / 68 test subprojects.
    """
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

    # Maintain certainty with the current Pitt subject-disjoint protocol natural sort.
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
    print("[Pitt subject-disjoint final split | SCPA Stage1]")
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


def load_official_train_entries(dataset_option: str) -> List[Dict[str, Any]]:
    """
    Construct official_train entries.

    Stage1 / Stage2 / Stage3 Use official_train only.
    official_test Used only at CPFT_exteact.py final export stage.
    """
    if dataset_option not in DATASETS:
        raise ValueError(f"DATASETS No configuration DATASET_OPTION={dataset_option}")

    # Pitt-MMSE Specialized: Stage1 can no longer use the original raw train directory directly,
    # Otherwise, the former train-test subjects follow-up visits will be involved in upstream supervisory training.
    if str(dataset_option) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        train_entries, _test_entries = _build_pitt_subject_disjoint_train_test_entries()
        return train_entries

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

    if str(dataset_option).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] official_train entries: "
            f"train_cc + train_cd = {len(train_entries)}"
        )

    if len(train_entries) == 0:
        raise ValueError(f"official_train entries Air:daset={dataset_option}")

    train_entries = sorted(train_entries, key=lambda x: natural_key(x.get("wav_path", "")))

    # Filter to MMSE and fix official audio id.
    for i, row in enumerate(train_entries):
        row["official_audio_id"] = int(i)
        row["audio_id"] = int(i)

    return train_entries


# ============================================================
# 4) fold json Read & & List Structures
# ============================================================
def load_fold_entries(fold_json_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read authoritative fold json, returns:
    - train_entries
    - val_entries
    - Original meta
    """
    if not os.path.exists(fold_json_path):
        raise FileNotFoundError(f"Old json:{fold_json_path}")

    with open(fold_json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    all_entries = meta.get("train_entries", [])
    train_entries = [copy.deepcopy(x) for x in all_entries if str(x.get("split", "")).lower() == "train"]
    val_entries = [copy.deepcopy(x) for x in all_entries if str(x.get("split", "")).lower() == "val"]

    if len(train_entries) == 0 or len(val_entries) == 0:
        raise ValueError(
            f"fold json Medium train/val is empty: train={len(train_entries)}, val={len(val_entries)}."
        )

    return train_entries, val_entries, meta


def enrich_entries_with_mmse(
    entries: List[Dict[str, Any]],
    dataset_option: str,
    mmse_maps: Dict[str, Dict[Any, float]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Refill MMSE with fold entries.
    Return:
    - List of valid entries
    - Missing list of entries for MMSE
    """
    valid_entries: List[Dict[str, Any]] = []
    missing_entries: List[Dict[str, Any]] = []

    for row in entries:
        row = copy.deepcopy(row)
        wav_path = row.get("wav_path", None)
        group = str(row.get("group", "")).strip().lower()
        split_name = str(row.get("split", "")).strip().lower()

        if wav_path in [None, ""]:
            row["missing_reason"] = "missing_wav_path"
            missing_entries.append(row)
            continue

        # If old json already has target_mmse (commonly Pitt), priority direct
        if row.get("target_mmse", None) not in [None, ""]:
            try:
                row["target_mmse"] = float(row["target_mmse"])
                valid_entries.append(row)
                continue
            except Exception:
                pass

        wav_stem = Path(str(wav_path)).stem

        if dataset_option == "Pitt":
            subject_key, visit_index = _extract_pitt_subject_visit(wav_stem)
            if subject_key is None or visit_index is None:
                row["missing_reason"] = "cannot_parse_pitt_subject_visit"
                missing_entries.append(row)
                continue
            key = (subject_key, int(visit_index))
            split_bucket = "train_cc" if group == "cc" else "train_cd"
            mmse_map = mmse_maps.get(split_bucket, {})
            if key not in mmse_map:
                row["missing_reason"] = f"missing_{_pitt_visit_to_mmse_col(visit_index)}"
                missing_entries.append(row)
                continue
            row["target_mmse"] = float(mmse_map[key])
            valid_entries.append(row)
            continue

        # ADReSS-IS2020 / ADReSSo21:Choose train_cc / train_cd under group
        key = _normalize_mmse_key(dataset_option, wav_stem)
        split_bucket = "train_cc" if group == "cc" else "train_cd"
        mmse_map = mmse_maps.get(split_bucket, {})
        if key not in mmse_map:
            row["missing_reason"] = f"missing_mmse_{split_name}_{group}"
            missing_entries.append(row)
            continue
        row["target_mmse"] = float(mmse_map[key])
        valid_entries.append(row)

    return valid_entries, missing_entries


# ============================================================
# 5) Data set with collate
# ============================================================
class MMSEFoldDataset(data.Dataset):
    """
    Stage1-MMSE Specialized data sets.

    Enter:
        entries: Each element needs to contain at least wav_path, target_mmse
    Output:
        wav: np.float32 1D
        mmse_z: Standardized MMSE (for utterance returns)
        mmse_raw: Original MMSE (for logs, inventories, etc.)
        mmse_bin: Crude particle size MMSE brackets (for frame classification)
        wav_len: Original true length (for valid token mask)
        name: wav File name (for returnname=True only)
    """
    def __init__(
        self,
        entries: List[Dict[str, Any]],
        target_mean: float,
        target_std: float,
        mmse_bin_edges: List[float],
        returnname: bool = False,
        sample_rate: int = 16000,
        target_sec: float = 30.0,
    ):
        self.entries = [copy.deepcopy(x) for x in entries]
        self.returnname = bool(returnname)
        self.sample_rate = int(sample_rate)
        self.target_seqlen = int(float(target_sec) * self.sample_rate)
        self.target_mean = float(target_mean)
        self.target_std = float(max(1e-6, target_std))
        self.mmse_bin_edges = [float(x) for x in mmse_bin_edges]

        if len(self.entries) == 0:
            raise ValueError("MMSEFoldDataset Roger that, entries.")

        self.mmse_values = [float(x["target_mmse"]) for x in self.entries]
        self.bin_ids = [bin_mmse(x, self.mmse_bin_edges) for x in self.mmse_values]
        self.num_bins = len(self.mmse_bin_edges) - 1

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int):
        row = self.entries[index]
        wav_path = row["wav_path"]
        wav, sr = sf.read(wav_path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)

        wav = wav.astype(np.float32)
        wav_len = int(wav.shape[0])

        mmse_raw = float(row["target_mmse"])
        mmse_z = (mmse_raw - self.target_mean) / self.target_std
        mmse_bin = int(bin_mmse(mmse_raw, self.mmse_bin_edges))
        name = Path(wav_path).name

        if not self.returnname:
            return wav, float(mmse_z), float(mmse_raw), int(mmse_bin), int(wav_len)
        return wav, float(mmse_z), float(mmse_raw), int(mmse_bin), int(wav_len), name

    @staticmethod
    def _repeat_or_center_crop_to_length(x: np.ndarray, target_len: int) -> Tuple[np.ndarray, int]:
        """
        Consistency with the same door Stage1:
        - Longer than 30s: Center tailor
        - Less than 30s: Duplicate fill
        Return:
            wav_30s, valid_len
        Where valid_len is the original active length, followed by the construction of valid token mask.
        """
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        L = int(x.shape[0])
        if L <= 0:
            return np.zeros((target_len,), dtype=np.float32), 0

        if L >= target_len:
            start = (L - target_len) // 2
            seg = x[start:start + target_len]
            return seg.astype(np.float32, copy=False), target_len

        n_rep = target_len // L
        rem = target_len - n_rep * L
        out = np.tile(x, n_rep)
        if rem > 0:
            out = np.concatenate([out, x[:rem]], axis=0)
        if out.shape[0] != target_len:
            out = out[:target_len]
        return out.astype(np.float32, copy=False), L

    def repeat30s_collate(self, batch):
        """
        collate_fn:
        Collapse all wavs in the battling to 30s, and returns the weight of the Whisper directly.
        """
        target_len = self.target_seqlen

        def process(x):
            if self.returnname:
                wav, mmse_z, mmse_raw, mmse_bin, wav_len, name = x
            else:
                wav, mmse_z, mmse_raw, mmse_bin, wav_len = x

            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            wav_len = int(wav_len)
            wav_len = max(0, min(wav_len, wav.shape[0]))
            base = wav[:wav_len] if wav_len > 0 else wav

            wav_30, valid_len = self._repeat_or_center_crop_to_length(base, target_len=target_len)

            if self.returnname:
                return wav_30, float(mmse_z), float(mmse_raw), int(mmse_bin), int(valid_len), name
            return wav_30, float(mmse_z), float(mmse_raw), int(mmse_bin), int(valid_len)

        batch = list(map(process, batch))
        return default_collate(batch)


# ============================================================
# 6) Model Widgets
# ============================================================
class LearnableScalarMix(nn.Module):
    """
    Learning multilayer integration.
    Enter:
        [L, B, T, C]
    Output:
        [B, T, C]
    """
    def __init__(self, n_layers: int):
        super().__init__()
        self.n_layers = int(n_layers)
        self.w = nn.Parameter(torch.zeros(self.n_layers))
        self.gamma = nn.Parameter(torch.tensor(1.0))

    def forward(self, stack: torch.Tensor) -> torch.Tensor:
        if stack.size(0) != self.n_layers:
            raise RuntimeError(f"ScalarMix Expecting L={self.n_layers},Current Get {stack.size(0)}.")
        weights = torch.softmax(self.w, dim=0).view(-1, 1, 1, 1)
        mixed = torch.sum(weights * stack, dim=0)
        return self.gamma * mixed


class WhisperEncoderWrapper(nn.Module):
    """
    Whisper encoder Sealed.

    Function:
    1. Freezing all parameters after supporting the freeze, only the specified layer range is unfrozen
    2. Support for reading single or multiple layers hidden states
    3. Support for mean / scalar_mix
    4. Support construction of token-level attention_mask based on valid_wav_lens
    """

    def __init__(
        self,
        model_name: str = "openai/whisper-small",
        unfreeze_encoder_layers: Optional[Sequence[int]] = None,
        scalar_mix_n_layers: Optional[int] = None,
    ):
        """
        Whisper encoder wrapper.

        Unfreezing logic reference VanillaFT.py:
        - Externally first dissects the list of specific layer numbers based on the current single_layer and UNFREEZE_ENCODER_LAYERS;
        - This is for unfreezing the encoder layers by tiered list;
        - Tier 1-based, for example, [12] for unfrozen encoder.layers[11].
        """
        super().__init__()
        self.model = WhisperModel.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.hidden_size = self.model.config.d_model

        # Record the current actual unfrozen layers for follow-up manifest / debug.
        if unfreeze_encoder_layers is None:
            unfreeze_encoder_layers = []
        self.unfreeze_encoder_layers = sorted({
            int(x) for x in list(unfreeze_encoder_layers)
            if int(x) > 0
        })

        self.scalar_mix = LearnableScalarMix(int(scalar_mix_n_layers)) if scalar_mix_n_layers is not None else None

        # 1) Freeze all parameters Whisper first
        for p in self.model.parameters():
            p.requires_grad = False

        encoder_layers = self.model.encoder.layers
        total_layers = len(encoder_layers)

        # 2) Unfreeze the encoder layer by 1-based
        valid_layers = []
        for layer_idx_1based in self.unfreeze_encoder_layers:
            if not (1 <= int(layer_idx_1based) <= total_layers):
                continue

            layer = encoder_layers[int(layer_idx_1based) - 1]
            for p in layer.parameters():
                p.requires_grad = True
            valid_layers.append(int(layer_idx_1based))

        self.unfreeze_encoder_layers = valid_layers

        # 3) If you unfrozen at least one layer, then at the same time unfrozen the last player_norm
        if len(self.unfreeze_encoder_layers) > 0:
            for p in self.model.encoder.layer_norm.parameters():
                p.requires_grad = True

        print(f"[UNFREEZE] Stage1 Whisper encoder layers = {self.unfreeze_encoder_layers}")

    def trainable_params(self) -> List[torch.Tensor]:
        """
        Returns all parameters of the current requires_grad=True.
        """
        return [p for p in self.parameters() if p.requires_grad]

    @staticmethod
    def _parse_repr_layers(repr_layers: Union[str, List[int], None]) -> Optional[List[int]]:
        """
        Parsing repr_layers:
        - "9,10,11,12" -> [9,10,11,12]
        - [12] -> [12]
        - None / "" -> None
        """
        if repr_layers is None:
            return None
        if isinstance(repr_layers, str):
            s = repr_layers.strip()
            if s == "":
                return None
            parts = [p.strip() for p in s.split(",") if p.strip() != ""]
            return [int(p) for p in parts]
        if isinstance(repr_layers, (list, tuple)):
            return [int(x) for x in repr_layers]
        return None

    def _build_attn_mask(self, input_features: torch.Tensor, valid_wav_lens: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Construct token level attachment_mask: [B, tok_T], True= Visible, False= invalid position.
        """
        if valid_wav_lens is None:
            return None
        if not torch.is_tensor(valid_wav_lens):
            valid_wav_lens = torch.tensor(valid_wav_lens)
        vwl = valid_wav_lens.detach().cpu().long().view(-1)
        B = input_features.size(0)
        if vwl.numel() != B:
            raise RuntimeError(f"valid_wav_lens Number does not match batch: expect={B}, got={vwl.numel()}")

        mel_T = int(input_features.size(-1))
        tok_T = (mel_T + 1) // 2

        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(vwl[i].item()))
            valid_Ts.append(min(vt, tok_T))
        valid_Ts = torch.tensor(valid_Ts, device=input_features.device, dtype=torch.long)

        t = torch.arange(tok_T, device=input_features.device).view(1, tok_T).expand(B, tok_T)
        return (t < valid_Ts.view(B, 1)).bool()

    def forward(
        self,
        wav: torch.Tensor,
        sampling_rate: int = 16000,
        repr_layers: Union[str, List[int], None] = None,
        repr_pool: str = "mean",
        valid_wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Enter waveform to output [B, T, C] for token expression.
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

        layers = self._parse_repr_layers(repr_layers)

        # If no explicit specified layer, then return directly to the last level
        if layers is None:
            try:
                encoder_out = self.model.encoder(
                    input_features=input_features,
                    attention_mask=attn_mask,
                    return_dict=True,
                    output_hidden_states=False,
                )
            except TypeError:
                encoder_out = self.model.encoder(
                    input_features=input_features,
                    return_dict=True,
                    output_hidden_states=False,
                )
            return encoder_out.last_hidden_state

        # Open hidden states when specifying a layer
        try:
            encoder_out = self.model.encoder(
                input_features=input_features,
                attention_mask=attn_mask,
                return_dict=True,
                output_hidden_states=True,
            )
        except TypeError:
            encoder_out = self.model.encoder(
                input_features=input_features,
                return_dict=True,
                output_hidden_states=True,
            )

        hs = encoder_out.hidden_states
        selected = []
        for l in layers:
            if l < 1 or l >= len(hs):
                raise ValueError(f"repr_layers The middle stage is illegal: layer={l}, hidden_states_len={len(hs)}")
            selected.append(hs[l])

        stack = torch.stack(selected, dim=0)

        if repr_pool == "mean":
            return stack.mean(dim=0)

        if repr_pool == "scalar_mix":
            if self.scalar_mix is None:
                raise RuntimeError("repr_pool='scalar_mix',However, scalar_mix has not been initialized.")
            if self.scalar_mix.w.device != stack.device:
                self.scalar_mix = self.scalar_mix.to(stack.device)
            return self.scalar_mix(stack)

        raise ValueError(f"Unknown repr_pool={repr_pool}")


class LinearHead(nn.Module):
    """
    General MLP head:
        LayerNorm -> Linear -> GELU -> Dropout -> Linear
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.layer_norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class AttentiveStatsPool(nn.Module):
    """
    With mask attentive mean + std pooling.
    Output dimension = 2 * in_dim.
    """
    def __init__(self, in_dim: int, attn_hidden: int = 128, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.attn = nn.Sequential(
            nn.Linear(in_dim, attn_hidden),
            nn.Tanh(),
            nn.Linear(attn_hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.attn(x).squeeze(-1)
        if mask is not None:
            if mask.dtype != torch.bool:
                mask = mask.bool()
            a = a.masked_fill(~mask, -1e4)

        a = F.softmax(a, dim=1)

        if mask is not None:
            a = a * mask.float()
            a = a / a.sum(dim=1, keepdim=True).clamp_min(self.eps)

        mean = torch.sum(x * a.unsqueeze(-1), dim=1)
        var = torch.sum(((x - mean.unsqueeze(1)) ** 2) * a.unsqueeze(-1), dim=1)
        std = torch.sqrt(var.clamp_min(self.eps))
        return torch.cat([mean, std], dim=-1)


# ============================================================
# 7) LightningModule:MMSE-aware Stage1
# ============================================================
class WhisperStage1MMSE(pl.LightningModule):
    """
    MMSE-aware Stage1.

    Training objectives:
    - utterance Level: Return to standardized MMSE
    - frame Level: All valid token projections are the same coarse MMSE bin
    """
    def __init__(
        self,
        train_entries: List[Dict[str, Any]],
        val_entries: List[Dict[str, Any]],
        saving_path: str,
        run_config: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["train_entries", "val_entries"])

        self.saving_path = str(saving_path)
        self.run_config = copy.deepcopy(run_config)

        self.batch_size = int(run_config["BATCH_SIZE"])
        self.num_workers = int(run_config["NUM_WORKERS"])
        self.maxstep = int(run_config["TRAINING_STEP"])

        self.lambda_utt = float(run_config["LAMBDA_UTT"])
        self.lambda_frame = float(run_config["LAMBDA_FRAME"])
        self.huber_delta = float(run_config["HUBER_DELTA"])

        self.repr_layers = run_config["CURRENT_STAGE1_REPR_LAYERS"]
        self.repr_pool = str(run_config["STAGE1_REPR_POOL"])
        self.mmse_bin_edges = [float(x) for x in run_config["MMSE_BIN_EDGES"]]

        # Use only train integration target standardizer
        train_mmse = np.asarray([float(x["target_mmse"]) for x in train_entries], dtype=np.float32)
        self.target_mean = float(np.mean(train_mmse))
        self.target_std = float(max(1e-6, np.std(train_mmse)))

        self.train_dataset = MMSEFoldDataset(
            entries=train_entries,
            target_mean=self.target_mean,
            target_std=self.target_std,
            mmse_bin_edges=self.mmse_bin_edges,
            returnname=True,
            sample_rate=SAMPLE_RATE,
            target_sec=float(run_config["TARGET_SEC"]),
        )
        self.val_dataset = MMSEFoldDataset(
            entries=val_entries,
            target_mean=self.target_mean,
            target_std=self.target_std,
            mmse_bin_edges=self.mmse_bin_edges,
            returnname=True,
            sample_rate=SAMPLE_RATE,
            target_sec=float(run_config["TARGET_SEC"]),
        )

        self.num_bins = int(self.train_dataset.num_bins)

        # The weight of the class for the frame loss only is based on the bin distribution of the training set
        bin_counter = {}
        for bid in self.train_dataset.bin_ids:
            bin_counter[bid] = bin_counter.get(bid, 0) + 1
        weight_counts = [bin_counter.get(i, 1) for i in range(self.num_bins)]
        weights = torch.tensor(weight_counts, dtype=torch.float)
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        self.register_buffer("frame_class_weights", weights)

        try:
            self.frame_criterion = nn.CrossEntropyLoss(
                weight=self.frame_class_weights,
                label_smoothing=float(run_config["LABEL_SMOOTHING"]),
            )
        except TypeError:
            self.frame_criterion = nn.CrossEntropyLoss(weight=self.frame_class_weights)

        # Huber / SmoothL1:Revert to MMSE after standardization is more stable
        try:
            self.utt_criterion = nn.HuberLoss(delta=self.huber_delta)
        except Exception:
            self.utt_criterion = nn.SmoothL1Loss(beta=self.huber_delta)

        parsed_layers = WhisperEncoderWrapper._parse_repr_layers(self.repr_layers)
        scalar_mix_n_layers = None if parsed_layers is None else len(parsed_layers)


        self.whisper = WhisperEncoderWrapper(
            model_name=str(run_config["WHISPER_MODEL_NAME"]),
            unfreeze_encoder_layers=run_config["STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS"],
            scalar_mix_n_layers=scalar_mix_n_layers if self.repr_pool == "scalar_mix" else None,
        )
        d_model = int(self.whisper.hidden_size)

        self.frame_head = LinearHead(
            in_dim=d_model,
            hidden_dim=int(run_config["HEAD_HIDDEN"]),
            out_dim=self.num_bins,
            dropout=float(run_config["HEAD_DROPOUT"]),
        )
        self.utt_pool = AttentiveStatsPool(d_model, attn_hidden=int(run_config["HEAD_HIDDEN"]))
        self.utt_head = LinearHead(
            in_dim=d_model * 2,
            hidden_dim=int(run_config["HEAD_HIDDEN"]),
            out_dim=1,
            dropout=float(run_config["HEAD_DROPOUT"]),
        )

    @staticmethod
    def _build_token_mask(valid_wav_lens: torch.Tensor, T: int, device) -> torch.Tensor:
        """
        Construct token-level valid position mask based on the original active wavelength length.
        """
        vwl = valid_wav_lens.detach().to("cpu").long().view(-1)
        B = vwl.numel()
        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(vwl[i].item()))
            valid_Ts.append(min(vt, T))
        valid_Ts = torch.tensor(valid_Ts, device=device, dtype=torch.long)
        t = torch.arange(T, device=device).view(1, T).expand(B, T)
        return (t < valid_Ts.view(B, 1))

    def forward(self, wav_batch: torch.Tensor, valid_len: Optional[torch.Tensor] = None):
        """
        Forward output:
        - frame_logits: [B, T, num_bins]
        - utt_pred_z:   [B]
        - token_mask:   [B, T]
        """
        reps = self.whisper(
            wav_batch,
            sampling_rate=SAMPLE_RATE,
            repr_layers=self.repr_layers,
            repr_pool=self.repr_pool,
            valid_wav_lens=valid_len,
        )

        token_mask = None
        if valid_len is not None:
            token_mask = self._build_token_mask(valid_len.to(reps.device), reps.size(1), reps.device)

        utt_repr = self.utt_pool(reps, mask=token_mask)
        utt_pred_z = self.utt_head(utt_repr).squeeze(-1)
        frame_logits = self.frame_head(reps)
        return frame_logits, utt_pred_z, token_mask

    def train_dataloader(self):
        """
        Training Series DataLoader.
        """
        nw = int(self.num_workers)
        pin = torch.cuda.is_available()
        return data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            drop_last=True,
            shuffle=True,
            collate_fn=self.train_dataset.repeat30s_collate,
            persistent_workers=(nw > 0),
            pin_memory=pin,
        )

    def val_dataloader(self):
        """
        Validation collection DataLoader.
        """
        nw = int(self.num_workers)
        pin = torch.cuda.is_available()
        return data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=nw,
            drop_last=False,
            shuffle=False,
            collate_fn=self.val_dataset.repeat30s_collate,
            persistent_workers=(nw > 0),
            pin_memory=pin,
        )

    def configure_optimizers(self):
        """
        Optimizer and learning ratio schedule:
        - head params: lr_head
        - whisper params: lr_whisper
        - scalar_mix params: Same head Ir
        """
        head_params: List[torch.Tensor] = []
        head_params += list(self.frame_head.parameters())
        head_params += list(self.utt_head.parameters())
        head_params += list(self.utt_pool.parameters())

        all_whisper_trainables = list(self.whisper.trainable_params())

        mix_params = []
        if getattr(self.whisper, "scalar_mix", None) is not None:
            mix_params = list(self.whisper.scalar_mix.parameters())

        mix_ids = {id(p) for p in mix_params}
        whisper_params = [p for p in all_whisper_trainables if id(p) not in mix_ids]

        optimizer = optim.AdamW(
            [
                {"params": head_params, "lr": float(self.run_config["LR_HEAD"])},
                {"params": whisper_params, "lr": float(self.run_config["LR_WHISPER"])},
                {"params": mix_params, "lr": float(self.run_config["LR_HEAD"])},
            ],
            weight_decay=float(self.run_config["WEIGHT_DECAY"]),
        )

        num_training_steps = int(self.maxstep)
        num_warmup_steps = int(float(self.run_config["WARMUP_RATIO"]) * num_training_steps)
        num_flat_steps = int(float(self.run_config["FLAT_RATIO"]) * num_training_steps)
        min_lr_ratio = float(self.run_config["MIN_LR_RATIO"])

        def lambda_lr(current_step: int):
            if num_training_steps <= 0:
                return 1.0
            if current_step < num_warmup_steps:
                return float(current_step + 1) / float(max(1, num_warmup_steps))
            if current_step < (num_warmup_steps + num_flat_steps):
                return 1.0
            decay_steps = max(1, num_training_steps - (num_warmup_steps + num_flat_steps))
            s = min(decay_steps, max(0, current_step - (num_warmup_steps + num_flat_steps)))
            progress = float(s) / float(decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        scheduler = {"scheduler": optim.lr_scheduler.LambdaLR(optimizer, lambda_lr), "interval": "step"}
        return [optimizer], [scheduler]

    @staticmethod
    def _get_lrs_from_optimizer(opt):
        """
        param_groups from optimizer to interpret the current learning rate to facilitate log entry.
        """
        if opt is None or len(opt.param_groups) < 3:
            return None, None, None
        lr_head = float(opt.param_groups[0]["lr"])
        lr_whisper = float(opt.param_groups[1]["lr"])
        lr_mix = float(opt.param_groups[2]["lr"])
        return lr_head, lr_whisper, lr_mix

    def _frame_loss_on_valid_tokens(
        self,
        frame_logits: torch.Tensor,
        mmse_bin: torch.Tensor,
        valid_wav_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the loss and accuracy of the frame classification only on the valid token.
        """
        B, T, C = frame_logits.shape
        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(valid_wav_lens[i].item()))
            valid_Ts.append(min(vt, T))
        valid_Ts = torch.tensor(valid_Ts, device=frame_logits.device, dtype=torch.long)

        t_ids = torch.arange(T, device=frame_logits.device).view(1, T).expand(B, T)
        mask = t_ids < valid_Ts.view(B, 1)

        logits_flat = frame_logits.reshape(-1, C)
        labels_flat = mmse_bin.unsqueeze(1).expand(-1, T).reshape(-1)
        mask_flat = mask.reshape(-1)

        if mask_flat.sum().item() == 0:
            loss_frame = logits_flat.sum() * 0.0
            acc = loss_frame.detach() * 0.0
            return loss_frame, acc

        loss_frame = self.frame_criterion(logits_flat[mask_flat], labels_flat[mask_flat])
        pred = torch.argmax(logits_flat[mask_flat], dim=-1)
        acc = (pred == labels_flat[mask_flat]).float().mean()
        return loss_frame, acc

    def training_step(self, batch, batch_idx):
        """
        Training steps:
        total_loss = lambda_utt * utterance_huber + lambda_frame * frame_ce
        """
        wav_30, mmse_z, mmse_raw, mmse_bin, valid_len, _names = batch

        frame_logits, utt_pred_z, _token_mask = self(wav_30, valid_len)

        mmse_z = mmse_z.to(utt_pred_z.device).float()
        mmse_bin = mmse_bin.to(frame_logits.device).long()
        valid_len = valid_len.to(frame_logits.device).long()

        loss_utt = self.utt_criterion(utt_pred_z, mmse_z)
        loss_frame, acc_frame = self._frame_loss_on_valid_tokens(frame_logits, mmse_bin, valid_len)
        loss = self.lambda_utt * loss_utt + self.lambda_frame * loss_frame

        utt_mae_z = torch.mean(torch.abs(utt_pred_z.detach() - mmse_z.detach()))

        lr_head, lr_whisper, lr_mix = None, None, None
        if self.trainer is not None and hasattr(self.trainer, "optimizers") and len(self.trainer.optimizers) > 0:
            lr_head, lr_whisper, lr_mix = self._get_lrs_from_optimizer(self.trainer.optimizers[0])

        log_dict = {
            "train_loss": loss,
            "train_loss_utt": loss_utt.detach(),
            "train_loss_frame": loss_frame.detach(),
            "train_frame_acc": acc_frame.detach(),
            "train_utt_mae_z": utt_mae_z.detach(),
        }
        if lr_head is not None:
            log_dict["lr_head"] = lr_head
        if lr_whisper is not None:
            log_dict["lr_whisper"] = lr_whisper
        if lr_mix is not None:
            log_dict["lr_mix"] = lr_mix

        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Authentication step:
        Recording valid_loss as a monitoring indicator for checkpoint / early stop.
        """
        wav_30, mmse_z, mmse_raw, mmse_bin, valid_len, _names = batch

        frame_logits, utt_pred_z, _token_mask = self(wav_30, valid_len)

        mmse_z = mmse_z.to(utt_pred_z.device).float()
        mmse_raw = mmse_raw.to(utt_pred_z.device).float()
        mmse_bin = mmse_bin.to(frame_logits.device).long()
        valid_len = valid_len.to(frame_logits.device).long()

        loss_utt = self.utt_criterion(utt_pred_z, mmse_z)
        loss_frame, acc_frame = self._frame_loss_on_valid_tokens(frame_logits, mmse_bin, valid_len)
        loss = self.lambda_utt * loss_utt + self.lambda_frame * loss_frame

        # Recording of MAE against standardization to facilitate direct understanding
        utt_pred_raw = utt_pred_z * self.target_std + self.target_mean
        utt_mae_raw = torch.mean(torch.abs(utt_pred_raw.detach() - mmse_raw.detach()))

        self.log_dict(
            {
                "valid_loss": loss,
                "valid_loss_utt": loss_utt.detach(),
                "valid_loss_frame": loss_frame.detach(),
                "valid_frame_acc": acc_frame.detach(),
                "valid_utt_mae_raw": utt_mae_raw.detach(),
            },
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        return loss

def get_whisper_encoder_layer_count(whisper_model_name_or_path: str) -> int:
    """
    Read Whisper config and return to encoder the total layer.
    For example, whisper-small is 12 layer.
    """
    cfg = WhisperConfig.from_pretrained(whisper_model_name_or_path)
    num_layers = getattr(cfg, "encoder_layers", None)
    if num_layers is None:
        raise AttributeError("WhisperConfig is missing encoder_layers and cannot automatically resolve layers.")
    return int(num_layers)


def resolve_unfreeze_layers_for_single_layer(
    single_layer: int,
    total_layers: int,
    unfreeze_layer_count: int,
) -> List[int]:
    """
    Explains which layers are actually unfrozen, based on the current single_layer and UNFREEZE_ENCODER_LAYERS.

    Rules:
    - UNFREEZE_ENCODER_LAYERS = 1 -> [single_layer]
    - UNFREEZE_ENCODER_LAYERS = 2 -> [single_layer-1, single_layer]
    - UNFREEZE_ENCODER_LAYERS = 3 -> [single_layer-2, single_layer-1, single_layer]
    - If single_layer leads to a smaller crossing, the crop is automatically cut to [1, total_layers].
    """
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

# ============================================================
# 8) run config And Save Path Construct
# ============================================================

def build_run_config(dataset_option: str, layer_id: int, fold_id: int = PROTOCOL_FOLD_ID) -> Dict[str, Any]:
    """
    Construct the current Stage1 run configuration.

    Current protocol:
    - official_train All for Stage1;
    - official_test (b) Not participate in Stage1;
    - Do not read fold json;
    - Path does not contain fold1 or official_train_test.

    Path:
        mmse_cpft_runs/<DATASET>/<MODEL_ROOT_TAG>/<singleX_unfreeZ>/stage1_mmse/
    """
    layer_id = int(layer_id)
    fold_id = int(PROTOCOL_FOLD_ID)

    total_layers = get_whisper_encoder_layer_count(WHISPER_MODEL_NAME)

    resolved_unfreeze_layers = resolve_unfreeze_layers_for_single_layer(
        single_layer=layer_id,
        total_layers=total_layers,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
    )

    layer_name = build_layer_name(
        single_layer=layer_id,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    protocol_tag = str(RUN_PROTOCOL_TAG)

    run_root = os.path.join(
        MMSE_CPFT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        layer_name,
    )

    stage1_save_dir = os.path.join(run_root, "stage1_mmse")
    meta_save_dir = os.path.join(run_root, "meta")

    current_stage1_repr_layers = [layer_id]

    legacy_train_layer_start = min(resolved_unfreeze_layers)
    legacy_train_layer_end = max(resolved_unfreeze_layers)

    return {
        "PROJECT_ROOT": PROJECT_ROOT,
        "MMSE_CPFT_ROOT": MMSE_CPFT_ROOT,
        "DATASET_OPTION": dataset_option,
        "LAYER_ID": int(layer_id),

        # Compatibility with old fields; no longer true fold.
        "FOLD_ID": int(fold_id),
        "FOLD_TAG": "not_used",
        "FOLD_JSON_PATH": "",
        "RUN_PROTOCOL_TAG": protocol_tag,

        "LAYER_TAG": layer_name,
        "LAYER_NAME": layer_name,
        "PATH_PARAM_TAGS": list(globals().get("PATH_PARAM_TAGS", [])),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(),
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(STAGE1_SOURCE_PATH_PARAM_TAGS),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(STAGE1_SOURCE_PATH_PARAM_TAGS),

        "MODEL_ROOT_TAG": MODEL_ROOT_TAG,
        "RUN_ROOT": run_root,
        "STAGE1_SAVE_DIR": stage1_save_dir,
        "META_SAVE_DIR": meta_save_dir,

        "CURRENT_STAGE1_REPR_LAYERS": current_stage1_repr_layers,
        "STAGE1_REPR_POOL": STAGE1_REPR_POOL,

        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),
        "STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS": list(resolved_unfreeze_layers),
        "TOTAL_ENCODER_LAYERS": int(total_layers),

        "STAGE1_WHISPER_TRAIN_LAYER_START": int(legacy_train_layer_start),
        "STAGE1_WHISPER_TRAIN_LAYER_END": int(legacy_train_layer_end),
        "STAGE1_WHISPER_TRAIN_LAST_N": None,

        "WHISPER_MODEL_NAME": WHISPER_MODEL_NAME,
        "TARGET_SEC": TARGET_SEC,
        "MMSE_BIN_EDGES": MMSE_BIN_EDGES,
        "LAMBDA_UTT": LAMBDA_UTT,
        "LAMBDA_FRAME": LAMBDA_FRAME,
        "HUBER_DELTA": HUBER_DELTA,
        "HEAD_HIDDEN": HEAD_HIDDEN,
        "HEAD_DROPOUT": HEAD_DROPOUT,
        "LABEL_SMOOTHING": LABEL_SMOOTHING,
        "LR_HEAD": LR_HEAD,
        "LR_WHISPER": LR_WHISPER,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "WARMUP_RATIO": WARMUP_RATIO,
        "FLAT_RATIO": FLAT_RATIO,
        "MIN_LR_RATIO": MIN_LR_RATIO,
        "BATCH_SIZE": BATCH_SIZE,
        "NUM_WORKERS": NUM_WORKERS,

        "STAGE1_USE_FOLD_VAL": False,
        "STAGE1_MAX_EPOCHS": int(STAGE1_MAX_EPOCHS),
        "STAGE1_CHECKPOINT_SELECTION": "last",
        "TRAINING_STEP": TRAINING_STEP,
        "PRECISION": PRECISION,
        "CHECK_VAL_EVERY_N_EPOCH": CHECK_VAL_EVERY_N_EPOCH,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "SAVE_TOP_K": SAVE_TOP_K,
        "ACCUMULATE_GRAD_BATCHES": ACCUMULATE_GRAD_BATCHES,
        "GRAD_CLIP": GRAD_CLIP,
        "SEED": SEED,
        "ENABLE_CSV_LOGGER": ENABLE_CSV_LOGGER,
        "RESUME_CHECKPOINT": RESUME_CHECKPOINT,
        "SAVE_MANIFESTS": SAVE_MANIFESTS,

        "STAGE1_REPR_LAYERS_TO_RUN": STAGE1_REPR_LAYERS_TO_RUN,
        "STAGE2_REP_LAYERS_TO_RUN": STAGE2_REP_LAYERS_TO_RUN,
        "STAGE3_REPR_LAYERS_TO_RUN": STAGE3_REPR_LAYERS_TO_RUN,
        "STAGE4_OUTPUT_LAYERS_TO_RUN": STAGE4_OUTPUT_LAYERS_TO_RUN,
    }


def build_trainer(stage1_save_dir: str, run_config: Dict[str, Any]) -> Tuple[Trainer, Optional[CSVLogger]]:
    """
    Stage1 no-val-last Trainer.

    - Do not use fold_val;
    - Do not do EarlyStopping;
    - Do not save best on valid_loss;
    - Fixed max_epochs;
    - Save last.ckpt only.
    """
    ensure_dir(stage1_save_dir)

    callbacks = [
        ModelCheckpoint(
            dirpath=stage1_save_dir,
            filename="whisper-s1-mmse-last",
            save_top_k=0,
            save_last=True,
            monitor=None,
            verbose=True,
        )
    ]

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    pl_precision = "16-mixed" if (int(run_config["PRECISION"]) == 16 and torch.cuda.is_available()) else "32-true"

    logger = False
    enable_progress_bar = True
    if bool(run_config["ENABLE_CSV_LOGGER"]):
        logger = CSVLogger(save_dir=stage1_save_dir, name="csv_logs")
        enable_progress_bar = False

    trainer = Trainer(
        precision=pl_precision,
        callbacks=callbacks,
        max_epochs=int(run_config.get("STAGE1_MAX_EPOCHS", 15)),
        max_steps=-1,
        accelerator=accelerator,
        devices=1,
        logger=logger,
        enable_progress_bar=enable_progress_bar,
        log_every_n_steps=10,
        accumulate_grad_batches=int(run_config["ACCUMULATE_GRAD_BATCHES"]),
        gradient_clip_val=float(run_config["GRAD_CLIP"]),
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )
    return trainer, logger


def save_stage1_manifests(
    stage1_save_dir: str,
    meta_save_dir: str,
    run_config: Dict[str, Any],
    fold_meta: Dict[str, Any],
    train_entries: List[Dict[str, Any]],
    val_entries: List[Dict[str, Any]],
    missing_train: List[Dict[str, Any]],
    missing_val: List[Dict[str, Any]],
) -> None:
    """
    Maintains an important list and configuration for this Stage1 operation to facilitate follow-up checks.
    """
    ensure_dir(stage1_save_dir)
    ensure_dir(meta_save_dir)

    # Configure
    save_json(run_config, os.path.join(stage1_save_dir, "config.json"))
    save_text(
        "official_train_test protocol: no fold json is used. Stage1 uses official_train only.\n",
        os.path.join(meta_save_dir, "source_protocol.txt"),
    )
    # fold Summary
    fold_summary = {
        "dataset_option": run_config["DATASET_OPTION"],
        "layer_id": run_config["LAYER_ID"],
        "layer_name": run_config["LAYER_NAME"],
        "fold_id": run_config["FOLD_ID"],
        "run_protocol": run_config.get("RUN_PROTOCOL_TAG", ""),
        "fold_disabled": True,
        "num_train_valid_entries": len(train_entries),
        "num_val_valid_entries": len(val_entries),
        "num_train_missing_mmse": len(missing_train),
        "num_val_missing_mmse": len(missing_val),
        "repr_layers": run_config["CURRENT_STAGE1_REPR_LAYERS"],
        "repr_pool": run_config["STAGE1_REPR_POOL"],

        # New unfrozen logic
        "unfreeze_encoder_layers": run_config["UNFREEZE_ENCODER_LAYERS"],
        "resolved_unfreeze_encoder_layers": run_config["STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS"],
        "total_encoder_layers": run_config["TOTAL_ENCODER_LAYERS"],

        # Compatibility with old fields to facilitate comparison of pre-books
        "whisper_train_layer_start": run_config["STAGE1_WHISPER_TRAIN_LAYER_START"],
        "whisper_train_layer_end": run_config["STAGE1_WHISPER_TRAIN_LAYER_END"],
    }
    save_json(fold_summary, os.path.join(meta_save_dir, "fold_summary.json"))

    if bool(run_config["SAVE_MANIFESTS"]):
        save_json({"entries": train_entries}, os.path.join(stage1_save_dir, "train_manifest.json"))
        save_json({"entries": val_entries}, os.path.join(stage1_save_dir, "val_manifest.json"))
        save_json({"entries": missing_train}, os.path.join(meta_save_dir, "missing_train_mmse_entries.json"))
        save_json({"entries": missing_val}, os.path.join(meta_save_dir, "missing_val_mmse_entries.json"))

    # Brief statistics
    def _stats(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(entries) == 0:
            return {"count": 0}
        mmse = np.asarray([float(x["target_mmse"]) for x in entries], dtype=np.float32)
        bins = np.asarray([bin_mmse(float(x["target_mmse"]), run_config["MMSE_BIN_EDGES"]) for x in entries], dtype=np.int64)
        return {
            "count": int(len(entries)),
            "mmse_min": float(np.min(mmse)),
            "mmse_max": float(np.max(mmse)),
            "mmse_mean": float(np.mean(mmse)),
            "mmse_std": float(np.std(mmse)),
            "bin_counts": {str(i): int((bins == i).sum()) for i in range(len(run_config["MMSE_BIN_EDGES"]) - 1)},
        }

    dataset_stats = {
        "train": _stats(train_entries),
        "val": _stats(val_entries),
    }
    save_json(dataset_stats, os.path.join(meta_save_dir, "dataset_stats.json"))


# ============================================================
# 8.5) Stage1 Quality evaluation: auto-output report after training
# ============================================================
def find_latest_metrics_csv(log_root: str) -> Optional[str]:
    """
    Finds metrics.csv for output Lightning CSVLogger.

    The current CSVLogger(...) saving in the build_traininger is:
        CSVLogger(save_dir=stage1_save_dir, name="csv_logs")

    Therefore, metrics.csv is usually located in:
        stage1_save_dir/csv_logs/version_*/metrics.csv
    """
    root = Path(str(log_root))
    candidates = list(root.glob("csv_logs/version_*/metrics.csv"))
    if len(candidates) == 0:
        return None
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0].resolve())


def _metric_series(df: pd.DataFrame, metric_name: str) -> pd.Series:
    """
    Securely remove a non-empty sequence from metric.
    """
    if df is None or metric_name not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[metric_name], errors="coerce").dropna()


def _last_metric(df: pd.DataFrame, metric_name: str) -> Optional[float]:
    """
    Take a last non-empty value from metric.
    """
    s = _metric_series(df, metric_name)
    if len(s) == 0:
        return None
    return float(s.iloc[-1])


def _best_row_by_min_metric(df: pd.DataFrame, metric_name: str) -> Optional[pd.Series]:
    """
    Press metric_name to take the smallest value in the row.
    """
    if df is None or metric_name not in df.columns:
        return None
    tmp = df.copy()
    tmp[metric_name] = pd.to_numeric(tmp[metric_name], errors="coerce")
    tmp = tmp.dropna(subset=[metric_name])
    if len(tmp) == 0:
        return None
    return tmp.loc[tmp[metric_name].idxmin()]


def _metric_at_epoch(df: pd.DataFrame, metric_name: str, epoch: Optional[int]) -> Optional[float]:
    """
    Try to read a metric that specifies the epoch.
    returns the last non-empty value of metric if epoch does not exist.
    """
    if df is None or metric_name not in df.columns:
        return None

    tmp = df.copy()
    tmp[metric_name] = pd.to_numeric(tmp[metric_name], errors="coerce")
    tmp = tmp.dropna(subset=[metric_name])
    if len(tmp) == 0:
        return None

    if epoch is not None and "epoch" in tmp.columns:
        ep = pd.to_numeric(tmp["epoch"], errors="coerce")
        hit = tmp[ep == int(epoch)]
        if len(hit) > 0:
            return float(hit[metric_name].iloc[-1])

    return float(tmp[metric_name].iloc[-1])


def _find_latest_non_last_ckpt(stage1_save_dir: str) -> Optional[str]:
    """
    Finds the latest non-last.ckpt checkpoint under stage1_save_dir.
    """
    p = Path(str(stage1_save_dir))
    ckpts = sorted(p.glob("*.ckpt"), key=lambda x: x.stat().st_mtime, reverse=True)
    non_last = [x for x in ckpts if x.name != "last.ckpt"]
    if len(non_last) > 0:
        return str(non_last[0].resolve())
    last = p / "last.ckpt"
    if last.exists():
        return str(last.resolve())
    return None


def write_stage1_quality_report(
    stage1_save_dir: str,
    run_config: Dict[str, Any],
    train_entries: List[Dict[str, Any]],
    val_entries: List[Dict[str, Any]],
    missing_train: List[Dict[str, Any]],
    missing_val: List[Dict[str, Any]],
) -> None:
    """
    Stage1 Quality evaluation after training.

    Main judgements:
    1. Success in saving checkpoint;
    2. valid_loss Whether there has been a decrease;
    3. valid_utt_mae_raw Readable;
    4. Whether there is a clear train-valid gap;
    5. best epoch It's too early.
    """
    ensure_dir(stage1_save_dir)

    metrics_csv = find_latest_metrics_csv(stage1_save_dir)
    df = pd.read_csv(metrics_csv) if metrics_csv is not None else None

    best_row = _best_row_by_min_metric(df, "valid_loss") if df is not None else None

    best_valid_loss = None
    best_epoch = None
    if best_row is not None:
        best_valid_loss = float(best_row["valid_loss"])
        if "epoch" in best_row.index and not pd.isna(best_row["epoch"]):
            best_epoch = int(best_row["epoch"])

    initial_valid_loss = None
    valid_loss_series = _metric_series(df, "valid_loss") if df is not None else pd.Series(dtype=float)
    if len(valid_loss_series) > 0:
        initial_valid_loss = float(valid_loss_series.iloc[0])

    last_valid_loss = _last_metric(df, "valid_loss") if df is not None else None
    best_valid_utt_mae_raw = _metric_at_epoch(df, "valid_utt_mae_raw", best_epoch) if df is not None else None
    last_valid_utt_mae_raw = _last_metric(df, "valid_utt_mae_raw") if df is not None else None

    # Lightning Train_loss on _epoch=True writes train_los_epoch.
    train_loss_at_best = None
    for cand in ["train_loss_epoch", "train_loss"]:
        train_loss_at_best = _metric_at_epoch(df, cand, best_epoch) if df is not None else None
        if train_loss_at_best is not None:
            break

    train_valid_gap = None
    if train_loss_at_best is not None and best_valid_loss is not None:
        train_valid_gap = float(best_valid_loss - train_loss_at_best)

    ckpt_path = _find_latest_non_last_ckpt(stage1_save_dir)

    warnings = []
    decision = "go_stage2"

    if ckpt_path is None:
        warnings.append("no_checkpoint_found")
        decision = "reject_stage1"

    if best_valid_loss is None:
        warnings.append("missing_valid_loss")
        decision = "reject_stage1"

    if initial_valid_loss is not None and best_valid_loss is not None:
        improvement = float(initial_valid_loss - best_valid_loss)
        if improvement <= 1e-6:
            warnings.append("valid_loss_no_improvement")
            if decision != "reject_stage1":
                decision = "warn_stage1"
    else:
        improvement = None

    if best_valid_utt_mae_raw is None:
        warnings.append("missing_valid_utt_mae_raw")
        if decision != "reject_stage1":
            decision = "warn_stage1"

    if best_epoch is not None and best_epoch <= 1:
        warnings.append("best_epoch_too_early")
        if decision != "reject_stage1":
            decision = "warn_stage1"

    if train_valid_gap is not None:
        # Empirical threshold: Stage1 loss scale is not fixed, so only mild warming is done here.
        if train_valid_gap > max(0.5, 0.3 * abs(float(best_valid_loss))):
            warnings.append("large_train_valid_gap")
            if decision != "reject_stage1":
                decision = "warn_stage1"

    report = {
        "stage": "stage1",
        "dataset_option": run_config["DATASET_OPTION"],
        "layer_id": int(run_config["LAYER_ID"]),
        "layer_name": str(run_config["LAYER_NAME"]),
        "fold_id": int(run_config["FOLD_ID"]),
        "repr_layers": run_config["CURRENT_STAGE1_REPR_LAYERS"],
        "repr_pool": run_config["STAGE1_REPR_POOL"],
        "unfreeze_encoder_layers": int(run_config["UNFREEZE_ENCODER_LAYERS"]),
        "resolved_unfreeze_encoder_layers": list(run_config["STAGE1_RESOLVED_UNFREEZE_ENCODER_LAYERS"]),
        "total_encoder_layers": int(run_config["TOTAL_ENCODER_LAYERS"]),

        "num_train_entries": int(len(train_entries)),
        "num_val_entries": int(len(val_entries)),
        "num_missing_train_mmse": int(len(missing_train)),
        "num_missing_val_mmse": int(len(missing_val)),

        "metrics_csv": metrics_csv,
        "checkpoint_path": ckpt_path,
        "initial_valid_loss": initial_valid_loss,
        "best_valid_loss": best_valid_loss,
        "last_valid_loss": last_valid_loss,
        "valid_loss_improvement": improvement,
        "best_valid_utt_mae_raw": best_valid_utt_mae_raw,
        "last_valid_utt_mae_raw": last_valid_utt_mae_raw,
        "best_epoch": best_epoch,
        "train_loss_at_best": train_loss_at_best,
        "train_valid_gap": train_valid_gap,

        "decision": decision,
        "warnings": warnings,
    }

    report_path = os.path.join(stage1_save_dir, QUALITY_REPORT_FILENAME)
    summary_path = os.path.join(stage1_save_dir, QUALITY_SUMMARY_FILENAME)
    decision_path = os.path.join(stage1_save_dir, QUALITY_DECISION_FILENAME)

    save_json(report, report_path)
    pd.DataFrame([report]).to_csv(summary_path, index=False)
    save_text(f"{decision}\nwarnings={warnings}\n", decision_path)

    print(f"[Stage1-Quality] report saved -> {report_path}")
    print(f"[Stage1-Quality] decision = {decision}, warnings = {warnings}")

# ============================================================
# 9) Single Test Run
# ============================================================
def run_one_stage1_experiment(dataset_option: str, layer_id: int, fold_id: int = PROTOCOL_FOLD_ID) -> None:
    """
    Runs a single official_train_test Stage1-MMSE experiment.

    Missing MMSE strategy:
    - Stage1 Filter missing MMSE;
    - missing entries Writing meta/missing_train_mmse_entries.json;
    - official_test Do not participate in Stage1.
    """
    run_config = build_run_config(dataset_option=dataset_option, layer_id=layer_id, fold_id=fold_id)
    stage1_save_dir = run_config["STAGE1_SAVE_DIR"]
    meta_save_dir = run_config["META_SAVE_DIR"]

    ensure_dir(stage1_save_dir)
    ensure_dir(meta_save_dir)

    print("=" * 80)
    print(f"[Stage1-MMSE] DATASET={dataset_option} | LAYER={layer_id} | PROTOCOL={RUN_PROTOCOL_TAG}")
    print("[Stage1-MMSE] fold disabled = True")
    print(f"[Stage1-MMSE] STAGE1_SAVE_DIR = {stage1_save_dir}")
    if dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT:
        print(f"[Stage1-MMSE][Pitt] subject-disjoint policy = {PITT_SPLIT_POLICY}")
        print("[Stage1-MMSE][Pitt] Stage1 trains only on final train: expected 184 subjects / 276 visits.")

    train_entries_raw = load_official_train_entries(dataset_option)

    mmse_maps = load_mmse_maps_for_dataset(dataset_option)
    train_entries, missing_train = enrich_entries_with_mmse(
        train_entries_raw,
        dataset_option,
        mmse_maps,
    )

    if len(train_entries) == 0:
        raise ValueError(
            f"official_train Empty after refilling MMSE: dataset={dataset_option}, layer={layer_id}"
        )

    # Do not use validation. To be compatible with the WhisperStage1MMSE internal val_dataset construction, a train sample of 1 is used for the position.
    # build_trainer() Middle limit_val_batches=0 so won't really run validation.
    val_entries_placeholder = train_entries[:1]
    missing_val = []

    steps_per_epoch = max(1, len(train_entries) // max(1, int(BATCH_SIZE)))
    effective_training_steps = max(1, steps_per_epoch * int(STAGE1_MAX_EPOCHS))
    run_config["TRAINING_STEP"] = int(effective_training_steps)
    run_config["STAGE1_USE_FOLD_VAL"] = False
    run_config["STAGE1_CHECKPOINT_SELECTION"] = "last"

    print(
        f"[Stage1-MMSE][OFFICIAL-TRAIN] "
        f"raw_train_wavs={len(train_entries_raw)}, "
        f"valid_mmse_wavs={len(train_entries)}, "
        f"missing_mmse_wavs={len(missing_train)}, "
        f"steps_per_epoch={steps_per_epoch}, "
        f"epochs={STAGE1_MAX_EPOCHS}, "
        f"effective_training_steps={effective_training_steps}"
    )

    protocol_meta = {
        "protocol": str(RUN_PROTOCOL_TAG),
        "fold_id": None,
        "fold_disabled": True,
        "stage1_use_fold_val": False,
        "official_train_used": True,
        "official_test_used": False,
        "pitt_subject_disjoint": bool(dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
        "pitt_split_policy": (PITT_SPLIT_POLICY if dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT else None),
        "pitt_expected_final_train_subjects": (PITT_EXPECTED_FINAL_TRAIN_SUBJECTS if dataset_option == "Pitt" else None),
        "pitt_expected_final_train_visits": (PITT_EXPECTED_FINAL_TRAIN_VISITS if dataset_option == "Pitt" else None),
        "note": (
            "Stage1 Pitt subject-disjoint final-train only; final test is never used by Stage1."
            if dataset_option == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT
            else "Stage1 official_train_test: official_train is used; official_test is not used."
        ),
    }

    save_stage1_manifests(
        stage1_save_dir=stage1_save_dir,
        meta_save_dir=meta_save_dir,
        run_config=run_config,
        fold_meta=protocol_meta,
        train_entries=train_entries,
        val_entries=[],
        missing_train=missing_train,
        missing_val=missing_val,
    )

    trainer, _logger = build_trainer(stage1_save_dir, run_config)
    model = WhisperStage1MMSE(
        train_entries=train_entries,
        val_entries=val_entries_placeholder,
        saving_path=stage1_save_dir,
        run_config=run_config,
    )

    ckpt_path = run_config["RESUME_CHECKPOINT"]
    if ckpt_path in [None, "None", ""]:
        ckpt_path = None

    trainer.fit(model, ckpt_path=ckpt_path)

    if bool(ENABLE_STAGE_QUALITY_EVAL):
        raise RuntimeError(
            "official_train_test no-val-last The settings should be maintained at ENABLE_STAGE_QUALITY_EVAL=False."
        )

    save_text(
        "Stage1-MMSE official_train_test run finished successfully.\n",
        os.path.join(stage1_save_dir, "run_notes.txt"),
    )

    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


# ============================================================
# 10) Main entrance: past layers / folds
# ============================================================
def main():
    """
    Main entrance:
    - Traveling through DATASET_OPTIONS;
    - Traveling through UNFREEZE_ENCODER_LAYERS_LIST;
    - Traveling through LAYERS_TO_RUN;
    - Do not travel to fold;
    - official_train All for Stage1.
    """
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    seed_everything_for_repro(SEED)

    if len(LAYERS_TO_RUN) == 0:
        raise ValueError("LAYERS_TO_RUN To be empty, at least one layer number.")
    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS There is a minimum need for a data set to be empty.")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST To be empty, a minimum number of unfrozen layers is required.")

    print("[CONFIG]")
    print("  PROJECT_ROOT =", PROJECT_ROOT)
    print("  MMSE_CPFT_ROOT =", MMSE_CPFT_ROOT)
    print("  DATASET_OPTIONS =", DATASET_OPTIONS)
    print("  RUN_PROTOCOL_TAG =", RUN_PROTOCOL_TAG)
    print("  LAYERS_TO_RUN =", LAYERS_TO_RUN)
    print("  MODEL_ROOT_TAG =", MODEL_ROOT_TAG)
    print("  UNFREEZE_ENCODER_LAYERS_LIST =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  WHISPER_MODEL_NAME =", WHISPER_MODEL_NAME)
    print("  STAGE1_MAX_EPOCHS =", STAGE1_MAX_EPOCHS)
    print("  fold disabled = True")
    print("  protocol subdir disabled = True")

    for dataset_option in [_canonical_dataset_option(x) for x in DATASET_OPTIONS]:
        if dataset_option not in MMSE_META_PATHS:
            raise ValueError(f"Unsupported DATASET_OPTION={dataset_option}")

        for unfreeze_count in [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "=" * 100)
            print("[Stage1 BATCH ITEM]")
            print("  DATASET_OPTION =", DATASET_OPTION)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("=" * 100)

            for layer_id in LAYERS_TO_RUN:
                run_one_stage1_experiment(
                    dataset_option=DATASET_OPTION,
                    layer_id=int(layer_id),
                    fold_id=PROTOCOL_FOLD_ID,
                )


if __name__ == "__main__":
    main()
