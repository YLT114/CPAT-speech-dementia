# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-


"""MMSE regression: Support only mamba3.
Retain existing data pre-processing, training budget, random seed, monitoring and evaluation agreements."""
"""
conda activate audiojax_gpu_mamba3
python mamba3.py
"""

# ===== 0. Global Configuration Module (first here) =====


import os


import sys


import re


import subprocess


from pathlib import Path


from typing import Dict, List, Optional, Sequence


CONFIG_ROOT_DIR = "/home/user/Alzheimer_code_jax_mmse_speech_v2"


CONFIG_FEATURE_ROOT = os.path.join(CONFIG_ROOT_DIR, "feature_tensor")


CONFIG_OUTPUT_ROOT = os.path.join(CONFIG_ROOT_DIR, "outputs_repeat")


# UEA Cache.


CONFIG_PRESPLIT_CACHE_NAME = "UEA"


# ====== Basic data set / model configuration ======
# Optional: "ADReSS-IS2020", "ADReSSo21", "Pitt", "PROCESS-2"
#
# Modify here when the batch runs.


DATASET_OPTIONS = ["ADReSS-IS2020"]


def _canonical_dataset_option(value: str) -> str:
    """Map the public ADReSS name to the existing internal dataset key."""
    value = str(value).strip()
    return "ADReSS-IS2020" if value == "ADReSS" else value


# Keeps a single value variable for internal use in old functions;
# The parent process will overwrite it individually in _launch_batch_runs_if_ned() using DATASET_OPTIONS;
# child The process will cover it through LRU_ACTIVE_DATASET_OPTION.


DATASET_OPTION = _canonical_dataset_option(DATASET_OPTIONS[0])


# This script supports two returns to PyTorchSSM:
# - MODEL_NAME="mamba3" :Mamba3 SISO template, output directory to outputs/<DATASET>/<FEATURE>/mamba3/<SELECTION>/.
#
# Compatibility description:
#


MODEL_NAMES = ["mamba3"]


MODEL_NAME = MODEL_NAMES[0]


# ====== Characteristic source configuration (freezing feature + fine-tuning) ======
# FEAT_MODEL_NAMES Use only when FEATURE_SOURCE_MODES contains "freeze".
# The parent process scans each of these frozen upstream profiles.


FEAT_MODEL_NAMES = ["whisper_small"]


# Keeps a single value variable for internal use in old functions;
# The parent process will overwrite it by pressing FEAT_MODEL_NAMES in _launch_batch_runs_if_ned();
# child The process will cover it through LRU_ACTIVE_FEAT_MODEL_NAME.


FEAT_MODEL_NAME = FEAT_MODEL_NAMES[0]


# Fine-tuning the list of feature names.
# Note: Do not insert a prefix for ft_; the script will automatically create ft_{FT_MODEL_NAME}.


FT_MODEL_NAMES = [
    "CPFT_lr5e-5_epo5",
    "vanilla_lr1e-5_epo3_segmentloss",
    "SSA_lr3e-6_epo5_segmentloss",
]


# Keeps a single value variable for internal use in old functions;
# The parent process will overwrite FT_MODEL_NAMES in _launch_batch_runs_if_ned()
# child The process will cover it through LRU_ACTIVE_FT_MODEL_NAME.


FT_MODEL_NAME = FT_MODEL_NAMES[0]


# Currently FEATURE_SOURCE_MODES=["freeze"], the freeze feature is read by default.


FEATURE_SOURCE_MODES = ["freeze"]  # Option: ["freeze", ["finenetune"), or both.


ENABLE_RECURSIVE_BATCH = True  # Set False if you want to use a custom path,; keep True when a parent process batch is found


# ====== Run Switch ======
# Mamba3 Separately, support seed=400 single-run and multi-seed.


RUN_SINGLE_SEED_400 = False


RUN_MULTI_SEEDS = True


MULTI_SEEDS = [401, 402, 403, 404, 405]


# ====== Add: Layered filter the feature directory to read ======


SINGLE_LAYERS = [12]


# ====== Add: Filter ====== on unfree layers only for "minding features"


UNFREE_LAYER_NUM = []


# ====== Single-run configuration ======


FEATURE_MODEL_TAG = ""


FEATURE_SELECTION_TAG = ""


FEATURE_DATA_DIR_OVERRIDE = None


FEATURE_FILE_TAG = "whisper_single12_0.3_30s"


"""
conda activate audiojax_gpu_mamba3
python mamba3.py
"""


# ====== Save path label ======


OUTPUT_MODEL_TAG = MODEL_NAME


OUTPUT_DATASET_TAG = DATASET_OPTION


OUTPUT_FEATURE_TAG = FEATURE_MODEL_TAG


OUTPUT_SELECTION_TAG = FEATURE_SELECTION_TAG


OUTPUT_EXPORT_PREFIX_OVERRIDE = None


#######################################################
# ====== Training and evaluation configuration ======


RANDOM_SEED = 400


BATCH_SIZE = 16


TARGET_EPOCHS = 10


USE_FIXED_EPOCHS = True


NUM_STEPS = 6000


PRINT_STEPS = None


# ====== no-leak early stopping / checkpoint policy ======
# official_test For monitoring / reporting;
# Not for early stopping, best epoch selection, candidate checkpoint selection.


USE_EARLY_STOPPING = True


EARLY_STOPPING_STEPS = 3


EARLY_STOPPING_MONITOR = "train_loss"


EARLY_STOPPING_MIN_DELTA = 1e-4


EARLY_STOPPING_RESTORE_BEST = False


# Final reporting strategy:
# - No early stop: report concludes epoch;
# - train_loss early stop:Final report upon termination epoch;
# - Do not use official_test to select epoch.


EPOCH_SELECTION_POLICY = "last_epoch_or_train_loss_early_stop"


TEST_MONITOR_ONLY = True


LR_MAMBA3 = 1e-4


WARMUP_EPOCHS = 2


COSINE_ALPHA = 0.05


# PyTorch AdamW / grad clip


GRAD_CLIP_NORM = 1.0


OPT_WEIGHT_DECAY = 1e-4


# no-leak These variables are only kept for log compatibility under the mode and do not participate in model selection.


EPOCH_SELECTION_OBJECTIVE = "audio_rmse"


#########################################################
# ====== audio-level Aggregation Configuration ======


AUDIO_AGG_PRIMARY_METHOD = "mean"


AUDIO_AGG_TRIM_RATIO = 0.10


# ====== PROCESS-2 Earmarked: MMSE barrel weight / mix loss ======
# To construct [target, weight] in build_raw_dataset(), the True must be maintained.


USE_PROCESS2_MMSE_BIN_WEIGHT = False


# PROCESS-2 train The distribution is extremely high MMSE and therefore the combined 3 slot is used for training:
# [0,21), [21,26), [26,31)


PROCESS2_MMSE_BIN_EDGES = [0.0, 21.0, 26.0, 31.0]


PROCESS2_MMSE_BIN_LABELS = ["[0,21)", "[21,26)", "[26,31)"]


# effective-number weighting Parameters


PROCESS2_MMSE_BIN_WEIGHT_BETA = 0.99


PROCESS2_MMSE_BIN_WEIGHT_MAX = 2.0


PROCESS2_WEIGHT_EPS = 1e-8


# Mixed loss:
# alpha = 0.0 -> Normal MSE
# alpha = 1.0 -> Pure nighted MSE


PROCESS2_MIXED_LOSS_ALPHA = 0.3


FEATURE_STANDARDIZE = False  # Default False


INCLUDE_TIME = False


T_MAX = 1.0


# ====== Compatible with old version of the background map ======


FILE_TAG_MAP = {
    "ADReSS-IS2020": "whisper_single12_0.3_30s",
    "ADReSSo21": "whisper_single12_0.3_30s",
    "PROCESS-2": "whisper_single12_0.3_30s",
}


def _is_blank(value) -> bool:
    return value is None or str(value).strip() in ["", "None", "none", "null", "Null"]


def _parse_int_env(env_name: str, default: Optional[int]) -> Optional[int]:
    raw = os.getenv(env_name, None)
    if raw is None or _is_blank(raw):
        return default
    return int(str(raw).strip())


def _parse_int_list_env(env_name: str, default: Sequence[int]) -> List[int]:
    raw = os.getenv(env_name, None)
    if raw is None:
        return [int(x) for x in default]
    raw = str(raw).strip()
    if _is_blank(raw):
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(int(part))
    return out


def _parse_str_list_env(env_name: str, default: Sequence[str]) -> List[str]:
    raw = os.getenv(env_name, None)
    if raw is None:
        return [str(x) for x in default]
    raw = str(raw).strip()
    if _is_blank(raw):
        return []
    out = []
    for part in raw.split(","):
        part = part.strip()
        if part == "":
            continue
        out.append(str(part))
    return out


def _parse_bool_env(env_name: str, default: bool) -> bool:
    raw = os.getenv(env_name, None)
    if raw is None or _is_blank(raw):
        return bool(default)

    raw = str(raw).strip().lower()
    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"Environmental variables {env_name} Could not close temporary folder: %s {raw}")


###############################################################################


def normalize_model_name(name: str) -> str:
    low = str(name).strip().lower()
    if low in ["mamba3", "mamba3_siso", "mamba"]:
        return "mamba3"
    raise ValueError("This script only supports mamba3, copy: " + str(name))


def normalize_model_name_list(names: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for x in names:
        name = normalize_model_name(x)
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out


#########################################################################################


def _extract_single_layer_from_selection(selection_tag: str) -> Optional[int]:
    if _is_blank(selection_tag):
        return None

    first_part = str(selection_tag).strip().replace("\\", "/").split("/")[0]
    m = re.match(r"^single(\d+)(?:_|$)", first_part, flags=re.IGNORECASE)
    if m is None:
        return None

    return int(m.group(1))


def _extract_unfree_num_from_selection(selection_tag: str) -> Optional[int]:
    """
    Parsing the number of unfree from the directory name.
    Example:
        single5_unfree1      -> 1
        single12_unfree12    -> 12
        single5              -> None
        single5/legacy_partition        -> None
    """
    if _is_blank(selection_tag):
        return None

    first_part = str(selection_tag).strip().replace("\\", "/").split("/")[0]
    m = re.search(r"_unfree(\d+)(?:_|$)", first_part, flags=re.IGNORECASE)
    if m is None:
        return None

    return int(m.group(1))


def _selection_matches_single_layers(selection_tag: str, allowed_layers: Sequence[int]) -> bool:
    allowed = sorted({int(x) for x in allowed_layers if int(x) > 0})
    if len(allowed) == 0:
        return True

    layer_id = _extract_single_layer_from_selection(selection_tag)
    if layer_id is None:
        return False

    return int(layer_id) in set(allowed)


def _selection_matches_unfree_nums(selection_tag: str, allowed_unfree_nums: Sequence[int]) -> bool:
    allowed = sorted({int(x) for x in allowed_unfree_nums if int(x) > 0})
    if len(allowed) == 0:
        return True

    unfree_num = _extract_unfree_num_from_selection(selection_tag)
    if unfree_num is None:
        return False

    return int(unfree_num) in set(allowed)


# =============================================


def _parse_feature_file_tag(feature_train_path: str) -> Optional[str]:
    name = Path(feature_train_path).name
    m = re.match(r"feature_(.+)_train\.pt$", name)
    return None if m is None else str(m.group(1))


def _has_complete_feature_bundle(feature_dir: str, file_tag: str) -> bool:
    required = [
        f"feature_{file_tag}_train.pt",
        f"feature_{file_tag}_test.pt",
        f"labels1_{file_tag}_train.pt",
        f"labels2_{file_tag}_test.pt",
        f"indices_{file_tag}_train.pt",
        f"indices_{file_tag}_test.pt",
    ]
    return all(os.path.exists(os.path.join(feature_dir, name)) for name in required)


def _clean_output_selection_tag(selection_tag: str) -> str:
    """
    Cleans up output path selection tag.

    Objectives:
    - Frozen model: single12 -> single12
    - Fine-tuning model: single12_unfree1 -> single12_unfree1
    - If mistransmitted to single12_unfree1/legacy_partition, take single12_unfree1 only

    Note:
    There's no single12_unfree1 compressed to single12.
    """
    if _is_blank(selection_tag):
        return "unknown_selection"

    return str(selection_tag).strip().replace("\\", "/").split("/")[0]


def _build_lru_output_prefix(
    dataset_name: str,
    output_feature_model_tag: str,
    downstream_model_name: str,
    selection_tag: str,
) -> str:
    "_build_lru_output_prefix:Model for mamba3 retained in this document."
    return os.path.join(
        CONFIG_OUTPUT_ROOT,
        str(dataset_name).strip(),
        str(output_feature_model_tag).strip(),
        normalize_model_name(downstream_model_name),
        _clean_output_selection_tag(selection_tag),
    )


def _discover_frozen_contexts() -> List[Dict[str, object]]:
    """
    Frozen feature directory found.

    Input directory:
        feature_tensor/<dataset>/<FEAT_MODEL_NAME>/<selection>/

    Output directory:
        outputs/<dataset>/<FEAT_MODEL_NAME>/<downstream_model>/<selection>

    Each feature configuration corresponds to an output directory.
    """
    if "freeze" not in {str(x).strip().lower() for x in FEATURE_SOURCE_MODES}:
        return []

    root_dir = os.path.join(CONFIG_FEATURE_ROOT, DATASET_OPTION, FEAT_MODEL_NAME)
    if not os.path.isdir(root_dir):
        print(f"[DISCOVER] Freezing feature root directory does not exist; skipping: {root_dir}")
        return []

    contexts = []

    for feature_train_path in sorted(Path(root_dir).rglob("feature_*_train.pt")):
        feature_dir = str(feature_train_path.parent)
        file_tag = _parse_feature_file_tag(str(feature_train_path))
        if _is_blank(file_tag):
            continue
        if not _has_complete_feature_bundle(feature_dir, file_tag):
            continue
        # Identification of history cross-check directory: Automatically discovers skipping, and the visible directory covers the retention of the original behaviour.
        if re.fullmatch(r"fold(\d+)", Path(feature_train_path).parent.name):
            continue

        rel_selection = os.path.relpath(feature_dir, root_dir)
        rel_selection = (
            feature_train_path.parent.name
            if rel_selection in [".", ""]
            else rel_selection.replace("\\", "/")
        )

        if not _selection_matches_single_layers(rel_selection, SINGLE_LAYERS):
            continue

        contexts.append(
            {
                "source_mode": "freeze",
                "dataset": DATASET_OPTION,
                "feature_model_tag": FEAT_MODEL_NAME,
                "feature_selection_tag": _clean_output_selection_tag(rel_selection),
                # Enter Information
                "feature_file_tag": str(file_tag),
                "feature_data_dir_override": feature_dir,
                "output_export_prefix": _build_lru_output_prefix(
                    dataset_name=DATASET_OPTION,
                    output_feature_model_tag=FEAT_MODEL_NAME,
                    downstream_model_name=MODEL_NAME,
                    selection_tag=rel_selection,
                ),
            }
        )

    return contexts


def _discover_finetune_contexts() -> List[Dict[str, object]]:
    """A fine-tuning feature was found under feature_tensor/<dataset>/ft_<FT_MODEL_NAME>/<selection>/.
    Automatically scans past historical cross-check subdirectories; output uses real input feature model names.
    """
    if "finetune" not in {str(x).strip().lower() for x in FEATURE_SOURCE_MODES}:
        return []

    input_feature_model_tag = f"ft_{FT_MODEL_NAME}"
    root_dir = os.path.join(CONFIG_FEATURE_ROOT, DATASET_OPTION, input_feature_model_tag)

    if not os.path.isdir(root_dir):
        print(f"[DISCOVER] Fine-tuning root directory does not exist, skip: {root_dir}")
        return []

    contexts = []

    for feature_train_path in sorted(Path(root_dir).rglob("feature_*_train.pt")):
        feature_dir_path = Path(feature_train_path).parent

        #   <selection>/feature_*.pt
        # Identification of history cross-check directory: Automatically discovers skipping, and the visible directory covers the retention of the original behaviour.
        legacy_partition_match = re.fullmatch(r"fold(\d+)", feature_dir_path.name)

        if legacy_partition_match is not None:
            continue

        feature_dir = str(feature_dir_path)
        selection_dir_path = feature_dir_path

        file_tag = _parse_feature_file_tag(str(feature_train_path))
        if _is_blank(file_tag):
            continue
        if not _has_complete_feature_bundle(feature_dir, file_tag):
            continue

        rel_selection = os.path.relpath(str(selection_dir_path), root_dir)
        rel_selection = (
            selection_dir_path.name
            if rel_selection in [".", ""]
            else rel_selection.replace("\\", "/")
        )

        if not _selection_matches_single_layers(rel_selection, SINGLE_LAYERS):
            continue
        if not _selection_matches_unfree_nums(rel_selection, UNFREE_LAYER_NUM):
            continue

        output_selection_tag = _clean_output_selection_tag(rel_selection)

        contexts.append(
            {
                "source_mode": "finetune",
                "dataset": DATASET_OPTION,
                "feature_model_tag": str(input_feature_model_tag).strip(),
                "feature_selection_tag": output_selection_tag,
                # The true input information is extraly maintained to facilitate log tracking.
                "input_feature_model_tag": input_feature_model_tag,
                "input_feature_selection_tag": rel_selection,
                "feature_file_tag": str(file_tag),
                "feature_data_dir_override": feature_dir,
                "output_export_prefix": _build_lru_output_prefix(
                    dataset_name=DATASET_OPTION,
                    output_feature_model_tag=input_feature_model_tag,
                    downstream_model_name=MODEL_NAME,
                    selection_tag=output_selection_tag,
                ),
            }
        )

    return contexts


# ====== Environmental variable coverage: for use ====== for parent processes to derive from child processes
# Key changes:
# 2) The parent process can discover multiple models at once, and each child is responsible for only one model.
# 3) The output path is naturally separated to outputs/<active_model>/...


DATASET_OPTION = _canonical_dataset_option(os.getenv("LRU_ACTIVE_DATASET_OPTION", DATASET_OPTION))


# child (a) The process is responsible for only one active dataset;
# The parent process retains DATASET_OPTIONS for bulk discovery of multiple data sets.


if os.getenv("LRU_BATCH_CHILD", "0") == "1":
    DATASET_OPTIONS = [str(DATASET_OPTION)]


FEAT_MODEL_NAMES = _parse_str_list_env("LRU_FEAT_MODEL_NAMES", FEAT_MODEL_NAMES)


if len(FEAT_MODEL_NAMES) == 0:
    raise ValueError("FEAT_MODEL_NAMES At least one name of the frozen feature model is needed for the sake of emptiness.")


ACTIVE_FEAT_MODEL_NAME = os.getenv("LRU_ACTIVE_FEAT_MODEL_NAME", None)


if not _is_blank(ACTIVE_FEAT_MODEL_NAME):
    FEAT_MODEL_NAME = str(ACTIVE_FEAT_MODEL_NAME).strip()
else:
    FEAT_MODEL_NAME = str(FEAT_MODEL_NAMES[0]).strip()


FT_MODEL_NAME = os.getenv("LRU_ACTIVE_FT_MODEL_NAME", FT_MODEL_NAME)


# child The process shall be responsible for only one active frozen feature model;
# The parent process keeps FEAT_MODEL_NAMES for bulk discovery of multiple frozen features catalogues.


if os.getenv("LRU_BATCH_CHILD", "0") == "1":
    FEAT_MODEL_NAMES = [str(FEAT_MODEL_NAME)]


# child The process shall be responsible for only one active finetune model;
# The parent process retains FT_MODEL_NAMES for batch discovery of multiple fine-tuned feature directories.


if os.getenv("LRU_BATCH_CHILD", "0") == "1":
    FT_MODEL_NAMES = [str(FT_MODEL_NAME)]


FEATURE_SOURCE_MODES = _parse_str_list_env("LRU_FEATURE_SOURCE_MODES", FEATURE_SOURCE_MODES)


# ++++++++++++++++++++++++++++++++++++++++++++++++++


SINGLE_LAYERS = _parse_int_list_env("LRU_SINGLE_LAYERS", SINGLE_LAYERS)


UNFREE_LAYER_NUM = _parse_int_list_env("LRU_UNFREE_LAYER_NUM", UNFREE_LAYER_NUM)


# ++++++++++++++++++++++++++++++++++++++++++++++++++


MODEL_NAMES = normalize_model_name_list(_parse_str_list_env("LRU_MODEL_NAMES", MODEL_NAMES))


RUN_SINGLE_SEED_400 = _parse_bool_env("LRU_RUN_SINGLE_SEED_400", RUN_SINGLE_SEED_400)


RUN_MULTI_SEEDS = _parse_bool_env("LRU_RUN_MULTI_SEEDS", RUN_MULTI_SEEDS)


MULTI_SEEDS = _parse_int_list_env("LRU_MULTI_SEEDS", MULTI_SEEDS)


USE_EARLY_STOPPING = _parse_bool_env("LRU_USE_EARLY_STOPPING", USE_EARLY_STOPPING)


EARLY_STOPPING_STEPS = _parse_int_env("LRU_EARLY_STOPPING_STEPS", EARLY_STOPPING_STEPS)


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


ACTIVE_MODEL_NAME = os.getenv("LRU_ACTIVE_MODEL_NAME", None)


if not _is_blank(ACTIVE_MODEL_NAME):
    MODEL_NAME = normalize_model_name(ACTIVE_MODEL_NAME)
else:
    if len(MODEL_NAMES) == 0:
        raise ValueError("MODEL_NAMES At least one model name is needed for the purpose of being empty.")
    MODEL_NAME = normalize_model_name(MODEL_NAMES[0])


# child Only the current active model is retained; the parent process MODEL_NAMES can still contain more than one


if not _is_blank(ACTIVE_MODEL_NAME):
    MODEL_NAMES = [MODEL_NAME]


FEATURE_MODEL_TAG = os.getenv("LRU_ACTIVE_FEATURE_MODEL_TAG", FEATURE_MODEL_TAG)


FEATURE_SELECTION_TAG = os.getenv("LRU_ACTIVE_FEATURE_SELECTION_TAG", FEATURE_SELECTION_TAG)


FEATURE_DATA_DIR_OVERRIDE = os.getenv(
    "LRU_ACTIVE_FEATURE_DATA_DIR_OVERRIDE", FEATURE_DATA_DIR_OVERRIDE
)


FEATURE_FILE_TAG = os.getenv("LRU_ACTIVE_FEATURE_FILE_TAG", FEATURE_FILE_TAG)


OUTPUT_EXPORT_PREFIX_OVERRIDE = os.getenv(
    "LRU_ACTIVE_OUTPUT_EXPORT_PREFIX", OUTPUT_EXPORT_PREFIX_OVERRIDE
)


ACTIVE_SOURCE_MODE = os.getenv("LRU_ACTIVE_SOURCE_MODE", None)


# Synchronize these display/dividend labels


OUTPUT_MODEL_TAG = MODEL_NAME


OUTPUT_DATASET_TAG = DATASET_OPTION


OUTPUT_FEATURE_TAG = FEATURE_MODEL_TAG


OUTPUT_SELECTION_TAG = FEATURE_SELECTION_TAG


DATASET_NAME = DATASET_OPTION


FEATURE_SUBDIR = (
    os.path.join(FEATURE_MODEL_TAG, FEATURE_SELECTION_TAG)
    if not _is_blank(FEATURE_SELECTION_TAG)
    else str(FEATURE_MODEL_TAG).strip()
)


FILE_TAG = (
    str(FEATURE_FILE_TAG).strip()
    if not _is_blank(FEATURE_FILE_TAG)
    else FILE_TAG_MAP.get(DATASET_NAME, None)
)


if _is_blank(FILE_TAG):
    raise ValueError("FEATURE_FILE_TAG It's empty, and there's no bottom map found in FILE_TAG_MAP.")


print("[ACTIVE CONFIG]")


print("  DATASET_OPTIONS =", globals().get("DATASET_OPTIONS", [DATASET_OPTION]))


print("  DATASET_OPTION =", DATASET_OPTION)


print("  FEAT_MODEL_NAMES =", globals().get("FEAT_MODEL_NAMES", [FEAT_MODEL_NAME]))


print("  FEAT_MODEL_NAME =", FEAT_MODEL_NAME)


print("  FT_MODEL_NAMES =", globals().get("FT_MODEL_NAMES", [FT_MODEL_NAME]))


print("  FT_MODEL_NAME =", FT_MODEL_NAME)


print("  MODEL_NAMES =", MODEL_NAMES)


print("  ACTIVE_MODEL_NAME =", MODEL_NAME)


print("  RUN_SINGLE_SEED_400 =", RUN_SINGLE_SEED_400)


print("  RUN_MULTI_SEEDS =", RUN_MULTI_SEEDS)


print("  MULTI_SEEDS =", MULTI_SEEDS)


print("  USE_EARLY_STOPPING =", USE_EARLY_STOPPING)


print("  EARLY_STOPPING_STEPS =", EARLY_STOPPING_STEPS)


print("  EARLY_STOPPING_MONITOR =", EARLY_STOPPING_MONITOR)


print("  EARLY_STOPPING_MIN_DELTA =", EARLY_STOPPING_MIN_DELTA)


print("  EARLY_STOPPING_RESTORE_BEST =", EARLY_STOPPING_RESTORE_BEST)


print("  EPOCH_SELECTION_POLICY =", EPOCH_SELECTION_POLICY)


print("  TEST_MONITOR_ONLY =", TEST_MONITOR_ONLY)


print("  FEATURE_MODEL_TAG =", FEATURE_MODEL_TAG)


print("  FEATURE_SELECTION_TAG =", FEATURE_SELECTION_TAG)


print("  ACTIVE_SOURCE_MODE =", ACTIVE_SOURCE_MODE)


print("  FEATURE_DATA_DIR_OVERRIDE =", FEATURE_DATA_DIR_OVERRIDE)


# ==============================================================================


def _build_single_selection_contexts() -> List[Dict[str, object]]:
    """
    Single-run mode constructs an independent run context.

    Read the rules:
    1) Read the directory directly if FEATURE_DATA_DIR_OVERRIDE is not empty;
    2) If you don't, try to freeze/normal directory:
        feature_tensor/<dataset>/<FEATURE_MODEL_TAG>/<FEATURE_SELECTION_TAG>
    3) In the case of finetune, scan:
        feature_tensor/<dataset>/<FT_MODEL_NAME>_ft/<selection>

    Output path no more:
        Additional protocol directory or cross-check directory

    If the actual directory is single12_unfree1, even if FEATURE_SELECTION_TAG writes single12,
    As long as SINGLE_LAYERS/UNFREE_LAYER_NUM matches, the output will also be used as single12_unfree1.
    """

    source_mode = (
        str(ACTIVE_SOURCE_MODE).strip()
        if not _is_blank(ACTIVE_SOURCE_MODE)
        else (
            "finetune"
            if "finetune" in {str(x).strip().lower() for x in FEATURE_SOURCE_MODES}
            else "freeze"
        )
    )

    candidate_items = []

    def _add_candidate(feature_dir: str, selection_tag: str):
        if _is_blank(feature_dir):
            return
        feature_dir = str(feature_dir).strip()
        if not os.path.isdir(feature_dir):
            return

        # Priority for current FILE_TAG
        if _has_complete_feature_bundle(feature_dir, FILE_TAG):
            candidate_items.append((feature_dir, str(FILE_TAG), str(selection_tag)))
            return

        # If FILE_TAG does not match, automatically parsing file_tag from feature_*_train.pt
        for p_train in sorted(Path(feature_dir).glob("feature_*_train.pt")):
            parsed_tag = _parse_feature_file_tag(str(p_train))
            if _is_blank(parsed_tag):
                continue
            if _has_complete_feature_bundle(feature_dir, parsed_tag):
                candidate_items.append((feature_dir, str(parsed_tag), str(selection_tag)))
                return

    if not _is_blank(FEATURE_DATA_DIR_OVERRIDE):
        # When you specify a real feature directory manually, selection_tag uses FEATURE_SELECTION_TAG by default.
        # If the directory name is single12_unfree1, priority is given to the directory name as output selection.
        override_dir = str(FEATURE_DATA_DIR_OVERRIDE).strip()
        override_name = Path(override_dir).name
        # Identification of history cross-check directory: Automatically discovers skipping, and the visible directory covers the retention of the original behaviour.
        if re.fullmatch(r"fold\d+", override_name):
            override_selection = Path(override_dir).parent.name
        else:
            override_selection = override_name

        _add_candidate(override_dir, override_selection)

    else:
        base_dir = os.path.join(
            CONFIG_FEATURE_ROOT,
            DATASET_OPTION,
            FEATURE_MODEL_TAG,
            FEATURE_SELECTION_TAG,
        )
        _add_candidate(base_dir, FEATURE_SELECTION_TAG)

        # 2) Old compatible directory
        old_dir = os.path.join(
            CONFIG_FEATURE_ROOT,
            FEATURE_MODEL_TAG,
            FEATURE_SELECTION_TAG,
            DATASET_OPTION,
        )
        _add_candidate(old_dir, FEATURE_SELECTION_TAG)

        if str(source_mode).strip().lower() == "finetune":
            ft_root = os.path.join(CONFIG_FEATURE_ROOT, DATASET_OPTION, f"ft_{FT_MODEL_NAME}")

            if os.path.isdir(ft_root):
                for feature_train_path in sorted(Path(ft_root).rglob("feature_*_train.pt")):
                    feature_dir_path = Path(feature_train_path).parent

                    # Identification of history cross-check directory: Automatically discovers skipping, and the visible directory covers the retention of the original behaviour.

                    legacy_partition_match = re.fullmatch(r"fold(\d+)", feature_dir_path.name)
                    if legacy_partition_match is not None:
                        feature_dir = str(feature_dir_path)
                        selection_dir_path = feature_dir_path.parent
                    else:
                        feature_dir = str(feature_dir_path)
                        selection_dir_path = feature_dir_path

                    rel_selection = os.path.relpath(str(selection_dir_path), ft_root)
                    rel_selection = (
                        selection_dir_path.name
                        if rel_selection in [".", ""]
                        else rel_selection.replace("\\", "/")
                    )

                    # Layer Filter
                    if not _selection_matches_single_layers(rel_selection, SINGLE_LAYERS):
                        continue

                    # unfree Filter
                    if not _selection_matches_unfree_nums(rel_selection, UNFREE_LAYER_NUM):
                        continue

                    # If FEATURE_SELECTION_TAG wrote single12,
                    # Allow matching single12_unfree1.
                    requested_layer = _extract_single_layer_from_selection(FEATURE_SELECTION_TAG)
                    candidate_layer = _extract_single_layer_from_selection(rel_selection)
                    if requested_layer is not None and candidate_layer is not None:
                        if int(requested_layer) != int(candidate_layer):
                            continue
                    else:
                        if _clean_output_selection_tag(
                            rel_selection
                        ) != _clean_output_selection_tag(FEATURE_SELECTION_TAG):
                            continue

                    file_tag = _parse_feature_file_tag(str(feature_train_path))
                    if _is_blank(file_tag):
                        continue
                    if not _has_complete_feature_bundle(feature_dir, file_tag):
                        continue

                    candidate_items.append((feature_dir, str(file_tag), str(rel_selection)))

    if len(candidate_items) == 0:
        raise FileNotFoundError(
            "No full feature directory found under single-run mode. \n"
            f"DATASET_OPTION={DATASET_OPTION}\n"
            f"FEATURE_MODEL_TAG={FEATURE_MODEL_TAG}\n"
            f"FT_MODEL_NAME={FT_MODEL_NAME}\n"
            f"FEATURE_SELECTION_TAG={FEATURE_SELECTION_TAG}\n"
            f"FILE_TAG={FILE_TAG}\n"
            f"SINGLE_LAYERS={SINGLE_LAYERS}\n"
            f"UNFREE_LAYER_NUM={UNFREE_LAYER_NUM}\n"
            f"FEATURE_DATA_DIR_OVERRIDE={FEATURE_DATA_DIR_OVERRIDE}"
        )

    selected_feature_dir, selected_file_tag, selected_selection_tag = candidate_items[0]
    output_selection_tag = _clean_output_selection_tag(selected_selection_tag)

    # Output level feature model:
    # - finetune Use a real fine-tuning feature directory name, such as whisper_small_tsc_ft;
    # - freeze Use FEAT_MODEL_NAME, e.g. whisper_small.
    if str(source_mode).strip().lower() == "finetune":
        output_feature_model_tag = f"{str(FT_MODEL_NAME).strip()}_ft"
    else:
        output_feature_model_tag = str(FEAT_MODEL_NAME).strip()

    return [
        {
            "source_mode": source_mode,
            "dataset": DATASET_OPTION,
            "feature_model_tag": output_feature_model_tag,
            "feature_selection_tag": output_selection_tag,
            "feature_file_tag": str(selected_file_tag),
            "feature_data_dir_override": selected_feature_dir,
            "output_export_prefix": _build_lru_output_prefix(
                dataset_name=DATASET_OPTION,
                output_feature_model_tag=output_feature_model_tag,
                downstream_model_name=MODEL_NAME,
                selection_tag=output_selection_tag,
            ),
        }
    ]


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def _build_output_export_prefix_for_child_model(ctx: Dict[str, object], model_name: str) -> str:
    "_build_output_export_prefix_for_child_model:Model for mamba3 retained in this document."
    return _build_lru_output_prefix(
        dataset_name=str(ctx["dataset"]).strip(),
        output_feature_model_tag=str(ctx["feature_model_tag"]).strip(),
        downstream_model_name=model_name,
        selection_tag=str(ctx["feature_selection_tag"]).strip(),
    )


def _launch_batch_runs_if_needed():
    """
    After the parent process batch finds context, press:

        dataset × frozen_feature_model / ft_model × context × model_name

    Cartesian starts child.

    Each child shall be responsible only for:
    - One dataset
    - Frozen feature model or a finetune feature model
    - A feature directory
    - A real downstream model
    """
    global DATASET_OPTION
    global FEAT_MODEL_NAME
    global FT_MODEL_NAME
    global OUTPUT_DATASET_TAG
    global DATASET_NAME
    global FILE_TAG
    global ACTIVE_SOURCE_MODE

    if os.getenv("LRU_BATCH_CHILD", "0") == "1":
        return

    script_path = os.path.abspath(__file__) if "__file__" in globals() else None
    if _is_blank(script_path):
        return

    dataset_options = [_canonical_dataset_option(x) for x in globals().get("DATASET_OPTIONS", [DATASET_OPTION])]
    dataset_options = [x for x in dataset_options if x != ""]

    if len(dataset_options) == 0:
        raise ValueError("DATASET_OPTIONS There is a minimum need for a data set to be empty.")

    feature_source_modes_norm = {str(x).strip().lower() for x in FEATURE_SOURCE_MODES}

    if "freeze" in feature_source_modes_norm:
        feat_model_names = [
            str(x).strip() for x in globals().get("FEAT_MODEL_NAMES", [FEAT_MODEL_NAME])
        ]
        feat_model_names = [x for x in feat_model_names if x != ""]
        if len(feat_model_names) == 0:
            raise ValueError("FEATURE_SOURCE_MODES Contains freeze but FEAT_MODEL_NAMES is empty.")
    else:
        # FEAT_MODEL_NAME does not participate in the feature directory discovery when not running freeze.
        feat_model_names = [str(FEAT_MODEL_NAME).strip()]

    if "finetune" in feature_source_modes_norm:
        ft_model_names = [str(x).strip() for x in globals().get("FT_MODEL_NAMES", [FT_MODEL_NAME])]
        ft_model_names = [x for x in ft_model_names if x != ""]
        if len(ft_model_names) == 0:
            raise ValueError("FEATURE_SOURCE_MODES Contains finetune but FT_MODEL_NAMES is empty.")
    else:
        # Only run freeze when FT_MODEL_NAME does not participate in feature directory discovery.
        ft_model_names = [str(FT_MODEL_NAME).strip()]

    all_contexts = []

    original_dataset_option = DATASET_OPTION
    original_feat_model_name = FEAT_MODEL_NAME
    original_ft_model_name = FT_MODEL_NAME
    original_active_source_mode = ACTIVE_SOURCE_MODE
    original_output_dataset_tag = globals().get("OUTPUT_DATASET_TAG", DATASET_OPTION)
    original_dataset_name = globals().get("DATASET_NAME", DATASET_OPTION)
    original_file_tag = globals().get("FILE_TAG", FEATURE_FILE_TAG)

    def _refresh_file_tag_for_dataset():
        global FILE_TAG

        # FEATURE_FILE_TAG (a) remain unchanged at the time of the apparent designation;
        # Click FILE_TAG_MAP to interpret the current dataset when not specified.
        if _is_blank(FEATURE_FILE_TAG):
            FILE_TAG = FILE_TAG_MAP.get(DATASET_NAME, None)
        else:
            FILE_TAG = str(FEATURE_FILE_TAG).strip()

        if _is_blank(FILE_TAG):
            raise ValueError(
                f"FEATURE_FILE_TAG Empty and not found in FILE_TAG_MAP "
                f"dataset={DATASET_NAME} The spin map."
            )

    def _append_contexts(contexts, active_feat_model_name, active_ft_model_name):
        for ctx in contexts:
            ctx = dict(ctx)
            ctx["active_feat_model_name"] = str(active_feat_model_name)
            ctx["active_ft_model_name"] = str(active_ft_model_name)
            all_contexts.append(ctx)

    for dataset_option in dataset_options:
        DATASET_OPTION = str(dataset_option)
        OUTPUT_DATASET_TAG = DATASET_OPTION
        DATASET_NAME = DATASET_OPTION
        _refresh_file_tag_for_dataset()

        if bool(ENABLE_RECURSIVE_BATCH):
            run_mode = "recursive_discovery"

            # 1) Frozen characteristic: Discovered by FEAT_MODEL_NAMES.
            if "freeze" in feature_source_modes_norm:
                ACTIVE_SOURCE_MODE = "freeze"
                for feat_model_name in feat_model_names:
                    FEAT_MODEL_NAME = str(feat_model_name)
                    contexts = _discover_frozen_contexts()

                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FEAT_MODEL={FEAT_MODEL_NAME}] "
                        f"mode = {run_mode}"
                    )
                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FEAT_MODEL={FEAT_MODEL_NAME}] "
                        f"discovered frozen contexts = {len(contexts)}"
                    )

                    _append_contexts(
                        contexts=contexts,
                        active_feat_model_name=FEAT_MODEL_NAME,
                        active_ft_model_name=FT_MODEL_NAME,
                    )

            # 2) Fine-tuning characteristic: Again per case at FT_MODEL_NAMES, do not match FEAT_MODEL_NAMES for repeated cartesian calculus.
            if "finetune" in feature_source_modes_norm:
                ACTIVE_SOURCE_MODE = "finetune"
                for ft_model_name in ft_model_names:
                    FT_MODEL_NAME = str(ft_model_name)
                    contexts = _discover_finetune_contexts()

                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FT_MODEL={FT_MODEL_NAME}] "
                        f"mode = {run_mode}"
                    )
                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FT_MODEL={FT_MODEL_NAME}] "
                        f"discovered finetune contexts = {len(contexts)}"
                    )

                    _append_contexts(
                        contexts=contexts,
                        active_feat_model_name=FEAT_MODEL_NAME,
                        active_ft_model_name=FT_MODEL_NAME,
                    )

        else:
            run_mode = "single_selection"

            # One-way mode also supports freeze / finetune running separately to avoid duplication.
            if "freeze" in feature_source_modes_norm:
                ACTIVE_SOURCE_MODE = "freeze"
                for feat_model_name in feat_model_names:
                    FEAT_MODEL_NAME = str(feat_model_name)
                    contexts = _build_single_selection_contexts()

                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FEAT_MODEL={FEAT_MODEL_NAME}] "
                        f"mode = {run_mode}"
                    )
                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FEAT_MODEL={FEAT_MODEL_NAME}] "
                        f"discovered frozen contexts = {len(contexts)}"
                    )

                    _append_contexts(
                        contexts=contexts,
                        active_feat_model_name=FEAT_MODEL_NAME,
                        active_ft_model_name=FT_MODEL_NAME,
                    )

            if "finetune" in feature_source_modes_norm:
                ACTIVE_SOURCE_MODE = "finetune"
                for ft_model_name in ft_model_names:
                    FT_MODEL_NAME = str(ft_model_name)
                    contexts = _build_single_selection_contexts()

                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FT_MODEL={FT_MODEL_NAME}] "
                        f"mode = {run_mode}"
                    )
                    print(
                        f"[BATCH][DATASET={DATASET_OPTION}][FT_MODEL={FT_MODEL_NAME}] "
                        f"discovered finetune contexts = {len(contexts)}"
                    )

                    _append_contexts(
                        contexts=contexts,
                        active_feat_model_name=FEAT_MODEL_NAME,
                        active_ft_model_name=FT_MODEL_NAME,
                    )

    # Restore parent processes to display variables and avoid subsequent logs being misleading.
    DATASET_OPTION = original_dataset_option
    FEAT_MODEL_NAME = original_feat_model_name
    FT_MODEL_NAME = original_ft_model_name
    ACTIVE_SOURCE_MODE = original_active_source_mode
    OUTPUT_DATASET_TAG = original_output_dataset_tag
    DATASET_NAME = original_dataset_name
    FILE_TAG = original_file_tag

    # This is done again on a global scale to avoid the duplication of the same directory under a special configuration.
    dedup_contexts = []
    seen = set()
    for ctx in all_contexts:
        key = (
            str(ctx.get("dataset", "")),
            str(ctx.get("source_mode", "")),
            str(ctx.get("active_feat_model_name", "")),
            str(ctx.get("active_ft_model_name", "")),
            str(ctx.get("feature_model_tag", "")),
            str(ctx.get("feature_selection_tag", "")),
            str(ctx.get("feature_file_tag", "")),
            str(ctx.get("feature_data_dir_override", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        dedup_contexts.append(ctx)

    contexts = dedup_contexts

    print(f"[BATCH] dataset_options = {dataset_options}")
    print(f"[BATCH] feat_model_names = {feat_model_names}")
    print(f"[BATCH] ft_model_names = {ft_model_names}")
    print(f"[BATCH] total discovered contexts = {len(contexts)}")
    print(f"[BATCH] model_names = {MODEL_NAMES}")

    for ctx in contexts:
        print(
            f"[BATCH] dataset={ctx['dataset']} | "
            f"feat_model={ctx.get('active_feat_model_name', FEAT_MODEL_NAME)} | "
            f"ft_model={ctx.get('active_ft_model_name', FT_MODEL_NAME)} | "
            f"{ctx['source_mode']:9s} | "
            f"feature_model={ctx['feature_model_tag']} | "
            f"selection={ctx['feature_selection_tag']} | "
            f"layer={_extract_single_layer_from_selection(ctx['feature_selection_tag'])} | "
            f"unfree={_extract_unfree_num_from_selection(ctx['feature_selection_tag'])} | "
            f"file_tag={ctx['feature_file_tag']} | "
            f"feature_dir={ctx['feature_data_dir_override']}"
        )

    if len(contexts) == 0:
        raise ValueError(
            "No running feature directory found, please check "
            "DATASET_OPTIONS / FEAT_MODEL_NAMES / FT_MODEL_NAMES / "
            "FEATURE_SELECTION_TAG."
        )

    model_names_to_run = normalize_model_name_list(MODEL_NAMES)
    if len(model_names_to_run) == 0:
        raise ValueError("MODEL_NAMES Empty, unable to start any child.")

    for ctx in contexts:
        for model_name_child in model_names_to_run:
            child_env = os.environ.copy()
            child_env["LRU_BATCH_CHILD"] = "1"

            # ===== Current context dataset /feature configuration = = =
            child_env["LRU_ACTIVE_DATASET_OPTION"] = str(ctx["dataset"])
            child_env["LRU_ACTIVE_FEAT_MODEL_NAME"] = str(
                ctx.get("active_feat_model_name", FEAT_MODEL_NAME)
            )
            child_env["LRU_ACTIVE_FT_MODEL_NAME"] = str(
                ctx.get("active_ft_model_name", FT_MODEL_NAME)
            )

            # ===== Model Level Configuration =====
            child_env["LRU_MODEL_NAMES"] = model_name_child
            child_env["LRU_ACTIVE_MODEL_NAME"] = model_name_child
            child_env["LRU_RUN_SINGLE_SEED_400"] = "1" if bool(RUN_SINGLE_SEED_400) else "0"
            child_env["LRU_RUN_MULTI_SEEDS"] = "1" if bool(RUN_MULTI_SEEDS) else "0"
            child_env["LRU_MULTI_SEEDS"] = ",".join(str(x) for x in MULTI_SEEDS)

            # ===== no-leak early stopping Configure =====
            child_env["LRU_USE_EARLY_STOPPING"] = "1" if bool(USE_EARLY_STOPPING) else "0"
            child_env["LRU_EARLY_STOPPING_STEPS"] = str(int(EARLY_STOPPING_STEPS))

            # ===== Accurate running parameter for the current context =====
            child_env["LRU_ACTIVE_SOURCE_MODE"] = str(ctx["source_mode"])

            child_env["LRU_SINGLE_LAYERS"] = ",".join(str(x) for x in SINGLE_LAYERS)
            child_env["LRU_UNFREE_LAYER_NUM"] = ",".join(str(x) for x in UNFREE_LAYER_NUM)

            child_env["LRU_ACTIVE_FEATURE_MODEL_TAG"] = str(ctx["feature_model_tag"])
            child_env["LRU_ACTIVE_FEATURE_SELECTION_TAG"] = str(ctx["feature_selection_tag"])
            child_env["LRU_ACTIVE_FEATURE_DATA_DIR_OVERRIDE"] = str(
                ctx["feature_data_dir_override"]
            )
            child_env["LRU_ACTIVE_FEATURE_FILE_TAG"] = str(ctx["feature_file_tag"])
            child_env["LRU_ACTIVE_OUTPUT_EXPORT_PREFIX"] = (
                _build_output_export_prefix_for_child_model(
                    ctx=ctx,
                    model_name=model_name_child,
                )
            )

            print(
                "[BATCH->CHILD] "
                f"dataset={child_env['LRU_ACTIVE_DATASET_OPTION']} | "
                f"model={child_env['LRU_ACTIVE_MODEL_NAME']} | "
                f"feat_model={child_env['LRU_ACTIVE_FEAT_MODEL_NAME']} | "
                f"ft_model={child_env['LRU_ACTIVE_FT_MODEL_NAME']} | "
                f"source_mode={child_env['LRU_ACTIVE_SOURCE_MODE']} | "
                f"feature_model_tag={child_env['LRU_ACTIVE_FEATURE_MODEL_TAG']} | "
                f"selection={child_env['LRU_ACTIVE_FEATURE_SELECTION_TAG']} | "
            )

            subprocess.run([sys.executable, script_path], env=child_env, check=True)

    raise SystemExit(0)


##########################################################################


_launch_batch_runs_if_needed()


GPU_ID = 5  # Default to 5


os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)


print("DATASET_NAME =", DATASET_NAME)


print("ACTIVE_SOURCE_MODE =", ACTIVE_SOURCE_MODE)


print("FEATURE_MODEL_TAG =", FEATURE_MODEL_TAG)


print("FEATURE_SELECTION_TAG =", FEATURE_SELECTION_TAG)


print("FEATURE_SUBDIR =", FEATURE_SUBDIR)


print("FEATURE_DATA_DIR_OVERRIDE =", FEATURE_DATA_DIR_OVERRIDE)


print("FILE_TAG =", FILE_TAG)


print("MODEL_NAME =", MODEL_NAME)


print("MODEL_NAMES =", MODEL_NAMES)


print("RUN_SINGLE_SEED_400 =", RUN_SINGLE_SEED_400)


print("RUN_MULTI_SEEDS =", RUN_MULTI_SEEDS)


print("MULTI_SEEDS =", MULTI_SEEDS)


print("OUTPUT_FEATURE_TAG =", OUTPUT_FEATURE_TAG)


print("OUTPUT_SELECTION_TAG =", OUTPUT_SELECTION_TAG)


print("OUTPUT_DATASET_TAG =", OUTPUT_DATASET_TAG)


print("OUTPUT_EXPORT_PREFIX_OVERRIDE =", OUTPUT_EXPORT_PREFIX_OVERRIDE)


print("BATCH_SINGLE_LAYERS =", SINGLE_LAYERS)


print("BATCH_UNFREE_LAYER_NUM =", UNFREE_LAYER_NUM)


print("BATCH_FEATURE_SOURCE_MODES =", FEATURE_SOURCE_MODES)


print(
    "BATCH_SIZE =",
    BATCH_SIZE,
    "TARGET_EPOCHS =",
    TARGET_EPOCHS,
    "USE_FIXED_EPOCHS =",
    USE_FIXED_EPOCHS,
)


print("PRINT_STEPS =", PRINT_STEPS, "(None=auto=steps_per_epoch)")


###############################################################################


print("USE_EARLY_STOPPING =", USE_EARLY_STOPPING)


print("EARLY_STOPPING_STEPS =", EARLY_STOPPING_STEPS)


print("EARLY_STOPPING_MONITOR =", EARLY_STOPPING_MONITOR)


print("EARLY_STOPPING_MIN_DELTA =", EARLY_STOPPING_MIN_DELTA)


print("EARLY_STOPPING_RESTORE_BEST =", EARLY_STOPPING_RESTORE_BEST)


print("EPOCH_SELECTION_POLICY =", EPOCH_SELECTION_POLICY)


print("TEST_MONITOR_ONLY =", TEST_MONITOR_ONLY)


print(
    "EPOCH_SELECTION_OBJECTIVE =",
    EPOCH_SELECTION_OBJECTIVE,
    "(monitor only; not used for checkpoint selection)",
)


print(
    "FEATURE_STANDARDIZE =", FEATURE_STANDARDIZE, "INCLUDE_TIME =", INCLUDE_TIME, "T_MAX =", T_MAX
)


# ===== 1. Path Read cell (Ref NCMMSC2021_LogNCDE.ipynb) =====


ROOT_DIR = CONFIG_ROOT_DIR


FEATURE_ROOT = CONFIG_FEATURE_ROOT


DATA_DIR = os.path.join(ROOT_DIR, "data_dir")


OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")


FEATURE_DATA_DIR = (
    str(FEATURE_DATA_DIR_OVERRIDE).strip()
    if not _is_blank(FEATURE_DATA_DIR_OVERRIDE)
    else os.path.join(FEATURE_ROOT, DATASET_NAME, FEATURE_MODEL_TAG, FEATURE_SELECTION_TAG)
)


def resolve_output_export_prefix():
    """
    Parsing the currently running output directory.

    _Other Organiser
        logncde_official_train_test
        Cross-check Directory
    """
    if not _is_blank(OUTPUT_EXPORT_PREFIX_OVERRIDE):
        return str(OUTPUT_EXPORT_PREFIX_OVERRIDE).strip()

    return _build_lru_output_prefix(
        dataset_name=OUTPUT_DATASET_TAG,
        output_feature_model_tag=OUTPUT_FEATURE_TAG,
        downstream_model_name=MODEL_NAME,
        selection_tag=OUTPUT_SELECTION_TAG,
    )


def get_output_root(dataset_tag, model_name, feature_tag, selection_tag):
    """
    Gets an output root directory for the current training run.

    Use child/context as a priority for OUTPUT_EXPORT_PREFIX.
    fallback Use the output directory of the current feature configuration.
    """
    output_prefix = globals().get("OUTPUT_EXPORT_PREFIX", None)

    if output_prefix is not None and str(output_prefix).strip() not in [
        "",
        "None",
        "none",
        "null",
        "Null",
    ]:
        return str(output_prefix).strip()

    return _build_lru_output_prefix(
        dataset_name=dataset_tag,
        output_feature_model_tag=feature_tag,
        downstream_model_name=model_name,
        selection_tag=selection_tag,
    )


OUTPUT_EXPORT_PREFIX = resolve_output_export_prefix()


OUTPUT_PATH = OUTPUT_EXPORT_PREFIX


print("OUTPUT_EXPORT_PREFIX =", OUTPUT_EXPORT_PREFIX)


def get_presplit_cache_dir(
    data_dir,
    dataset_name,
    feature_model_tag=None,
    feature_selection_tag=None,
    file_tag=None,
    cache_model_name=None,
):
    """Cache Directory: <data_dir>/processed/<CONFIG_PRESPLIT_CACHE_NAME>/<dataset>/
    <feature_model>/<downstream_model>/<selection>/<file_tag>.The models use separate caches.
    """
    cache_root_name = str(globals().get("CONFIG_PRESPLIT_CACHE_NAME", "UEA")).strip()

    parts = [
        str(data_dir).strip(),
        "processed",
        cache_root_name,
        str(dataset_name).strip(),
    ]

    if not _is_blank(feature_model_tag):
        parts.append(str(feature_model_tag).strip())

    # Downstream model names are added to the cache path, avoiding multiple models and sending and writing the same batch of pkl.
    if _is_blank(cache_model_name):
        cache_model_name = globals().get("MODEL_NAME", None)

    if not _is_blank(cache_model_name):
        parts.append(str(cache_model_name).strip())

    if not _is_blank(feature_selection_tag):
        selection_parts = (
            str(feature_selection_tag).strip().replace("\\", "/").strip("/").split("/")
        )
        parts.extend([p for p in selection_parts if str(p).strip() != ""])

    if not _is_blank(file_tag):
        parts.append(str(file_tag).strip())

    return os.path.join(*parts)


# ===== 1. Path Read cell (Ref NCMMSC2021_LogNCDE.ipynb) =====


import os


if FEATURE_DATA_DIR_OVERRIDE is not None and str(FEATURE_DATA_DIR_OVERRIDE).strip() not in [
    "",
    "None",
    "none",
    "null",
    "Null",
]:
    FEATURE_DATA_DIR = str(FEATURE_DATA_DIR_OVERRIDE).strip()
else:
    FEATURE_DATA_DIR = os.path.join(
        FEATURE_ROOT, DATASET_NAME, FEATURE_MODEL_TAG, FEATURE_SELECTION_TAG
    )
    if not os.path.exists(FEATURE_DATA_DIR):
        FEATURE_DATA_DIR_LEGACY = os.path.join(
            FEATURE_ROOT, FEATURE_MODEL_TAG, FEATURE_SELECTION_TAG, DATASET_NAME
        )
        if os.path.exists(FEATURE_DATA_DIR_LEGACY):
            FEATURE_DATA_DIR = FEATURE_DATA_DIR_LEGACY


FEATURE_TRAIN_PATH = os.path.join(FEATURE_DATA_DIR, f"feature_{FILE_TAG}_train.pt")


FEATURE_TEST_PATH = os.path.join(FEATURE_DATA_DIR, f"feature_{FILE_TAG}_test.pt")


LABEL_TRAIN_PATH = os.path.join(FEATURE_DATA_DIR, f"labels1_{FILE_TAG}_train.pt")


LABEL_TEST_PATH = os.path.join(FEATURE_DATA_DIR, f"labels2_{FILE_TAG}_test.pt")


INDEX_TRAIN_PATH = os.path.join(FEATURE_DATA_DIR, f"indices_{FILE_TAG}_train.pt")


INDEX_TEST_PATH = os.path.join(FEATURE_DATA_DIR, f"indices_{FILE_TAG}_test.pt")


# official_train / official_test protocol:
# Read train_val_split_*.json no more.


PATH_DICT = {
    "feature_train": FEATURE_TRAIN_PATH,
    "feature_test": FEATURE_TEST_PATH,
    "label_train": LABEL_TRAIN_PATH,
    "label_test": LABEL_TEST_PATH,
    "index_train": INDEX_TRAIN_PATH,
    "index_test": INDEX_TEST_PATH,
}


print("FEATURE_DATA_DIR =", FEATURE_DATA_DIR)


for k, v in PATH_DICT.items():
    print(f"[{k}] {v} | exists={os.path.exists(v)}")


print("[split_protocol] official_train -> train")


print("[split_protocol] official_test  -> test")


print("[protocol] No additional division is performed; official_test is always evaluated and exported.")


print("[protocol] official_train Full-scale training; official_test monitoring and reporting only, not participating in model selection or early stopping.")


print("[protocol] val_* is the compatible field name for official_test_monitor.")


# ===== 2. Basic import module: Mamba3-only =====


import sys


import time


import pickle


from datetime import datetime


import json


import atexit


import numpy as np


import pandas as pd


import matplotlib.pyplot as plt


from sklearn.metrics import r2_score


import torch


try:
    from IPython.display import display
except Exception:

    def display(x):
        if hasattr(x, "to_string"):
            print(x.to_string())
        else:
            print(x)


os.chdir(ROOT_DIR)


if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


print("cwd =", os.getcwd())


print("torch =", torch.__version__)


print("torch cuda =", torch.version.cuda)


print("torch cuda available =", torch.cuda.is_available())


# ===== 4. Execute data conversion (building directly from.pt presplit pkl, MMSE regression) =====


import os


import json


import pickle


import numpy as np


import torch


def _to_numpy(x, dtype=None):
    """Toch. Tensor / np.ndarray / list to numbery"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def _load_json(path):
    """Read json documents."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_pkl(obj, path):
    """Save the object to pkl."""
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# ===================================================


def _canonical_wav_path(wav_path):
    """The uniform wav path indicates that avoidance of inconsistencies between relative/absolute paths leads to a failure of matching."""
    return str(Path(wav_path).expanduser().resolve())


def _natural_key(path_str):
    """Order of nature: single2 before single10; S002 before S010."""
    name = Path(path_str).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", name)]


def _list_wav_files(audio_dir, dataset_name=None):
    """
    Reads wav in the directory and sorts it in natural order.

    PROCESS-2:
        Read only document name stem at the end of __CTD wav.
        Example:
            PROCESS-2_rec__001__CTD.wav

    Other data sets:
        Keep the same logic and read all.wav under the directory.
    """
    if audio_dir is None or (not os.path.isdir(audio_dir)):
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")

    dataset_name_norm = str(dataset_name).strip().lower()

    files = []
    for f in os.listdir(audio_dir):
        if not str(f).lower().endswith(".wav"):
            continue

        if dataset_name_norm == "process-2":
            stem = Path(f).stem
            if not re.search(r"__CTD$", stem, flags=re.IGNORECASE):
                continue

        files.append(str(Path(audio_dir) / f))

    files = sorted(files, key=_natural_key)
    return files


DATASET_AUDIO_DIRS = {
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
    "ADReSS_m": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSS-M/train1/train_wav/HC",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSS-M/train1/train_wav/AD",
        "test_cc": "/home/user/Downloads/dementiabank/ADReSS-M/train1/test-gr/HC",
        "test_cd": "/home/user/Downloads/dementiabank/ADReSS-M/train1/test-gr/AD",
    },
}


def _parse_file_tag_from_feature_train_path(feature_train_path):
    """
    From:
        .../feature_whisper_single10_0.3_30s_train.pt
    Parsing:
        whisper_single10_0.3_30s
    """
    name = os.path.basename(str(feature_train_path))
    m = re.match(r"^feature_(.+)_train\.pt$", name)
    if m is None:
        return None
    return str(m.group(1))


def _build_audio_entry_lookup_from_split_meta(split_meta):
    """
    Constructed from split metadata for the current feature:
        current_local_audio_id -> {wav_path, wav_name, target_mmse}

    Note:
    This can only be used for the "current identity of its own metadata";
    External cross-checking of split files cannot be used directly for the current local audio_id lookup.
    """
    lookup = {}

    for row in split_meta.get("train_entries", []):
        if not isinstance(row, dict):
            continue

        audio_id = row.get("audio_id", None)
        wav_path = row.get("wav_path", None)
        if audio_id is None or wav_path in [None, ""]:
            continue

        audio_id = int(audio_id)
        wav_path = _canonical_wav_path(wav_path)

        target_mmse = row.get("target_mmse", None)
        try:
            target_mmse = float(target_mmse) if target_mmse is not None else None
        except Exception:
            target_mmse = None

        wav_name = row.get("wav_name", None)
        if wav_name in [None, ""]:
            wav_name = os.path.basename(wav_path)

        lookup[audio_id] = {
            "audio_id": audio_id,
            "wav_path": wav_path,
            "wav_name": str(wav_name),
            "target_mmse": target_mmse,
        }

    return lookup


def _build_current_feature_audio_entry_lookup_from_dataset(dataset_name):
    """
    Rebuild the map of the current feature local audio_id -> wav_path to freeze features.

    This is a strict simulation of frozen.py's train index generation order:
        train_cc Sort Naturally
        + train_cd Sort Naturally
    """
    if dataset_name not in DATASET_AUDIO_DIRS:
        raise ValueError(f"Current unconfigurated dataset {dataset_name} is the original audio directory.")

    dataset_paths = DATASET_AUDIO_DIRS[dataset_name]

    ordered_files = []
    ordered_files.extend(_list_wav_files(dataset_paths["train_cc"], dataset_name=dataset_name))
    ordered_files.extend(_list_wav_files(dataset_paths["train_cd"], dataset_name=dataset_name))

    if str(dataset_name).strip().lower() == "process-2":
        print(f"[PROCESS-2] Rebuild the train audio_id map only with CTD wav, number of documents: {len(ordered_files)}")

    lookup = {}
    for audio_id, wav_path in enumerate(ordered_files):
        wav_path = _canonical_wav_path(wav_path)
        lookup[int(audio_id)] = {
            "audio_id": int(audio_id),
            "wav_path": wav_path,
            "wav_name": os.path.basename(wav_path),
            "target_mmse": None,
        }

    return lookup


def _load_current_feature_audio_entry_lookup(dataset_name, feature_train_path):
    """
    Read the map of the current feature itself local audio_id -> wav_path.
    Data are not reclassified based on the membership relationship train/val in the metadata.

    Priority:
    1) Current train_val_split_<file_tag>.json under feature_dir
       - Applicable to fine-tuning features (local metadata co-source with current indices)
    2) If non-existent, back to the original data directory order
       - Applicable to freezing characteristics
    """
    feature_dir = os.path.dirname(str(feature_train_path))
    file_tag = _parse_file_tag_from_feature_train_path(feature_train_path)

    if file_tag is not None:
        local_split_path = os.path.join(feature_dir, f"train_val_split_{file_tag}.json")
        if os.path.exists(local_split_path):
            split_meta = _load_json(local_split_path)
            lookup = _build_audio_entry_lookup_from_split_meta(split_meta)
            if len(lookup) > 0:
                return lookup, local_split_path

    fallback_lookup = _build_current_feature_audio_entry_lookup_from_dataset(dataset_name)
    return fallback_lookup, "__dataset_train_order_fallback__"


# +++++++++++++++++==============================================
# ==============================================


def convert_stage1_pt_to_mmse_presplit_pkl(
    dataset_name,
    data_dir,
    feature_train_path,
    feature_test_path,
    label_train_path,
    label_test_path,
    index_train_path,
    index_test_path,
    seed=400,
):
    """Converts.pt to downstream cache, official_training all; official_test for monitoring and reporting.
    No additional disaggregation of data; val cache is a compatible alias for official_test.
    Local train_val_split_<file_tag>.json is used only to track the mapping of audio ID and wav_path.
    """

    # ========= 1) Path Check =========
    required_paths = {
        "feature_train_path": feature_train_path,
        "feature_test_path": feature_test_path,
        "label_train_path": label_train_path,
        "label_test_path": label_test_path,
        "index_train_path": index_train_path,
        "index_test_path": index_test_path,
    }
    for name, p in required_paths.items():
        if p is None or not os.path.exists(p):
            raise FileNotFoundError(f"{name} Cannot initialise Evolution's mail component. {p}")

    # ========= 2) Read train/test segment Level =========
    X_train_full = _to_numpy(torch.load(feature_train_path, map_location="cpu"), dtype=np.float32)
    X_test = _to_numpy(torch.load(feature_test_path, map_location="cpu"), dtype=np.float32)

    y_train_full = _to_numpy(
        torch.load(label_train_path, map_location="cpu"), dtype=np.float32
    ).reshape(-1)
    y_test = _to_numpy(torch.load(label_test_path, map_location="cpu"), dtype=np.float32).reshape(
        -1
    )

    idx_train_full = _to_numpy(torch.load(index_train_path, map_location="cpu", weights_only=False))
    idx_test_full = _to_numpy(torch.load(index_test_path, map_location="cpu", weights_only=False))

    if idx_train_full.ndim != 2 or idx_train_full.shape[1] < 2:
        raise ValueError(
            f"index_train For (N,2) or at least 2D >=2, currently shape={idx_train_full.shape}"
        )
    if idx_test_full.ndim != 2 or idx_test_full.shape[1] < 2:
        raise ValueError(f"index_test For (N,2) or at least 2D >=2, currently shape={idx_test_full.shape}")

    idx_train_full = idx_train_full[:, :2].astype(np.int64)
    idx_test_full = idx_test_full[:, :2].astype(np.int64)

    # ========= 2.1) Construct "Current feature owns local audio_id -> wav_path" =========
    current_audio_entry_lookup, current_audio_lookup_source_path = (
        _load_current_feature_audio_entry_lookup(
            dataset_name=dataset_name,
            feature_train_path=feature_train_path,
        )
    )

    ################################################################################################################################
    # ========= 3/4) no-leak protocol: official_train -> train, official_test -> test =========
    # - official_train All segment enter X_train;
    # - official_test All segment into X_test.
    #
    # Compatibility description:
    # - To reuse old dataloaders / metric pipeline, keep X_val/y_val/idx_val;
    # - But X_val/y_val/idx_val is only the legal leegacy alias of the legal_test;
    # - official_test For monitoring / reporting;
    # - Not for early stopping, best epoch selection, candidate checkpoint selection.

    audio_entry_lookup_source_path = current_audio_lookup_source_path
    split_strategy = "noleak_official_train_as_train_official_test_as_test_monitor_only"

    X_train = X_train_full
    y_train = y_train_full
    idx_train = idx_train_full

    X_val = X_test
    y_val = y_test
    idx_val = idx_test_full

    idx_test = idx_test_full

    train_audio_ids = set(idx_train[:, 0].astype(np.int64).tolist())
    val_audio_ids = set(idx_val[:, 0].astype(np.int64).tolist())
    test_audio_ids = set(idx_test[:, 0].astype(np.int64).tolist())

    print("[split] LogNCDE-style official train/test protocol enabled.")
    print("[split] official_train -> train")
    print("[split] official_test -> val Cache aliases, monitoring and reporting only")
    print("[split] official_test  -> test")
    print(
        "[split] audio counts | "
        f"train={len(train_audio_ids)}, "
        f"val={len(val_audio_ids)}, "
        f"test={len(test_audio_ids)}"
    )
    ###################################################################################################################
    # ========= 5) Fundamental legality check =========
    if len(X_train) == 0:
        raise ValueError("official_train Characteristics are empty. Please check the training profile.")
    if len(X_val) == 0:
        raise ValueError("official_test Monitor features are empty. Please check the test features file.")
    if len(X_test) == 0:
        raise ValueError("X_test Please check the test feature file.")

    if not (len(X_train) == len(y_train) == len(idx_train)):
        raise ValueError("train The length of the X / y / idx does not match.")
    if not (len(X_val) == len(y_val) == len(idx_val)):
        raise ValueError("val The length of the X / y / idx does not match.")
    if not (len(X_test) == len(y_test) == len(idx_test)):
        raise ValueError("test The length of the X / y / idx does not match.")

    ds_dir = get_presplit_cache_dir(
        data_dir=data_dir,
        dataset_name=dataset_name,
        feature_model_tag=globals().get("FEATURE_MODEL_TAG", None),
        feature_selection_tag=globals().get("FEATURE_SELECTION_TAG", None),
        file_tag=globals().get("FILE_TAG", None),
    )
    os.makedirs(ds_dir, exist_ok=True)

    print("[presplit_cache_dir]", ds_dir)

    # ========= 7) Save as presplit pkl =========
    _save_pkl(X_train.astype(np.float32), os.path.join(ds_dir, "X_train.pkl"))
    _save_pkl(X_val.astype(np.float32), os.path.join(ds_dir, "X_val.pkl"))
    _save_pkl(X_test.astype(np.float32), os.path.join(ds_dir, "X_test.pkl"))

    _save_pkl(y_train.astype(np.float32).reshape(-1), os.path.join(ds_dir, "y_train.pkl"))
    _save_pkl(y_val.astype(np.float32).reshape(-1), os.path.join(ds_dir, "y_val.pkl"))
    _save_pkl(y_test.astype(np.float32).reshape(-1), os.path.join(ds_dir, "y_test.pkl"))

    _save_pkl(idx_train[:, 0].astype(np.int64), os.path.join(ds_dir, "idx_train.pkl"))
    _save_pkl(idx_val[:, 0].astype(np.int64), os.path.join(ds_dir, "idx_val.pkl"))
    _save_pkl(idx_test[:, 0].astype(np.int64), os.path.join(ds_dir, "idx_test.pkl"))

    _save_pkl(idx_train.astype(np.int64), os.path.join(ds_dir, "pairidx_train.pkl"))
    _save_pkl(idx_val.astype(np.int64), os.path.join(ds_dir, "pairidx_val.pkl"))
    _save_pkl(idx_test.astype(np.int64), os.path.join(ds_dir, "pairidx_test.pkl"))

    # ========= 8) Save Summary =========

    summary = {
        "dataset_name": dataset_name,
        "task": "mmse_regression",
        "source": "direct_from_audio_cut_jax_mmse_pt",
        #####################
        "split_protocol": "official_train_as_train_official_test_as_test",
        "train_semantics": "official_train",
        "test_semantics": "official_test",
        "compatibility_aliases": {
            "X_val.pkl": "alias_of_X_test.pkl",
            "y_val.pkl": "alias_of_y_test.pkl",
            "idx_val.pkl": "alias_of_idx_test.pkl",
            "val_*": "legacy_internal_alias_of_test_*",
        },
        #########################
        "feature_train_path": feature_train_path,
        "feature_test_path": feature_test_path,
        "label_train_path": label_train_path,
        "label_test_path": label_test_path,
        "index_train_path": index_train_path,
        "index_test_path": index_test_path,
        "audio_entry_lookup_source_path": audio_entry_lookup_source_path,
        "split_strategy": split_strategy,
        "seed": int(seed),
        "note": (
            "official_train is used completely for training; "
            "official_test is used as test and for monitoring/reporting only. "
            "official_test is not used for early stopping, best epoch selection, or checkpoint selection. "
            "No fold_train/fold_val split is used. Legacy val files are compatibility aliases of test files."
        ),
        "output_dir": ds_dir,
        "shapes": {
            "X_train": list(X_train.shape),
            "X_val": list(X_val.shape),
            "X_test": list(X_test.shape),
            "y_train": list(y_train.reshape(-1).shape),
            "y_val": list(y_val.reshape(-1).shape),
            "y_test": list(y_test.reshape(-1).shape),
            "idx_train_audio": list(idx_train[:, 0].reshape(-1).shape),
            "idx_val_audio": list(idx_val[:, 0].reshape(-1).shape),
            "idx_test_audio": list(idx_test[:, 0].reshape(-1).shape),
        },
        ##########################
        "audio_counts": {
            "train_audio": int(len(np.unique(idx_train[:, 0]))),
            "test_audio": int(len(np.unique(idx_test[:, 0]))),
            "legacy_val_alias_audio": int(len(np.unique(idx_val[:, 0]))),
        },
        "segment_counts": {
            "train_segments": int(len(idx_train)),
            "test_segments": int(len(idx_test)),
            "legacy_val_alias_segments": int(len(idx_val)),
        },
        "target_stats": {
            "train_min": float(np.min(y_train)),
            "train_max": float(np.max(y_train)),
            "train_mean": float(np.mean(y_train)),
            "train_std": float(np.std(y_train)),
            "test_min": float(np.min(y_test)),
            "test_max": float(np.max(y_test)),
            "test_mean": float(np.mean(y_test)),
            "test_std": float(np.std(y_test)),
        },
        ################################
    }

    summary_path = os.path.join(ds_dir, "stage1_pt_to_mmse_presplit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary


# ========================================
# ===== Execute conversion =====


convert_info = convert_stage1_pt_to_mmse_presplit_pkl(
    dataset_name=DATASET_NAME,
    data_dir=DATA_DIR,
    feature_train_path=FEATURE_TRAIN_PATH,
    feature_test_path=FEATURE_TEST_PATH,
    label_train_path=LABEL_TRAIN_PATH,
    label_test_path=LABEL_TEST_PATH,
    index_train_path=INDEX_TRAIN_PATH,
    index_test_path=INDEX_TEST_PATH,
    seed=RANDOM_SEED,
)


print(json.dumps(convert_info, ensure_ascii=False, indent=2))


print("\n[OK] MMSE regression presplit pkl has been generated directly from.pt at audio_cut_jax.")


print("[NEXT] Caches will continue to be carried out and training and evaluation carried out.")


# ===== 5A. Characteristic standardization (reference preprocess_data to NCMMSC2021_LogNCDE.ipynb) =====
# Defines the characteristic standardization function (the formula and dimension of the strict re-entry reference achieved).


def preprocess_data(features):
    """Each sample is standardized along time dimensions, resulting in a zero-average and unit variation of sequence characteristics."""
    # Calculates the mean value of each sample on the time dimension (axis=1), maintaining the dimension for easy broadcasting.
    mean = features.mean(axis=1, keepdims=True)
    # Calculate the standard deviation of each sample at the time dimension (axis=1) and keep the dimension easy to broadcast.
    std = features.std(axis=1, keepdims=True)
    # Standardize by formula and add 1e-8 to prevent the elimination of zero.
    standardized_features = (features - mean) / (std + 1e-8)
    print(mean)
    print(std)
    # Returns the standardized feature.
    return standardized_features


# Prints a standardized strategy note.


print("preprocess_data Ready: press axis=1 to do (x-mean)/(std+1e-8)")


# ===== 5. Build Dataset (MMSE regression version, add train-only y standardization) =====
# ===== 5. Build Dataset (MMSE regression version, add train-only y standardization) =====


from types import SimpleNamespace


class Dataloader:
    """
    Mamba3-only Pure NumPy dataloader.

    Two y formats are supported:
      1) Normal MMSE regression tag:
            y.shape = [N]

      2) PROCESS-2 weighted loss Label:
            y.shape = [N, 2]
            y[:, 0] = target
            y[:, 1] = sample_weight

    Key points:
      - Not directly to y reshape(-1), otherwise [N,2] will become [2N],
        This resulted in an inconsistency in the number of samples X/y.
      - Sample number checks must be y.shape[0].
    """

    def __init__(self, X, y, inmemory=False):
        self.X = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        if y_arr.ndim == 0:
            y_arr = y_arr.reshape(1)

        # Normal tab keeps [N]
        if y_arr.ndim == 1:
            self.y = y_arr.astype(np.float32)

        # PROCESS-2 weighted label Keep [N, 2]
        elif y_arr.ndim == 2:
            self.y = y_arr.astype(np.float32)

        # Other abnormal shapes: only 1D is reserved as sample dimensions and the rest of the dimensions are displayed
        else:
            self.y = y_arr.reshape(y_arr.shape[0], -1).astype(np.float32)

        self.inmemory = bool(inmemory)

        if self.X.shape[0] != self.y.shape[0]:
            raise ValueError(
                f"Dataloader X/y Samples are inconsistent: X.shape={self.X.shape}, y.shape={self.y.shape}"
            )

        self.size = int(self.X.shape[0])

    def loop_epoch(self, batch_size, shuffle=False, rng=None):
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size Must be > 0, now ={batch_size}")

        indices = np.arange(self.size)

        if shuffle:
            if rng is None:
                rng = np.random.default_rng()
            rng.shuffle(indices)

        for start in range(0, self.size, batch_size):
            batch_idx = indices[start : start + batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def loop(self, batch_size, key=None):
        """
        Compatible with the old interface. Not using JAX key here.
        Infinite loop epoch and default shuffle.
        """
        rng = np.random.default_rng()
        while True:
            for X_batch, y_batch in self.loop_epoch(batch_size, shuffle=True, rng=rng):
                yield X_batch, y_batch


os.makedirs(os.path.join(DATA_DIR, "processed", "toy"), exist_ok=True)


os.makedirs(os.path.join(DATA_DIR, "processed", "PPG"), exist_ok=True)


os.makedirs(
    os.path.join(
        DATA_DIR,
        "processed",
        str(globals().get("CONFIG_PRESPLIT_CACHE_NAME", "UEA")).strip(),
    ),
    exist_ok=True,
)


# ===== Target standardized configuration =====


if "TARGET_STANDARDIZE" not in globals():
    TARGET_STANDARDIZE = True


if "TARGET_STANDARDIZE_EPS" not in globals():
    TARGET_STANDARDIZE_EPS = 1e-6


ACTIVE_TARGET_NORMALIZER = None


def ensure_regression_targets(y):
    """Form a regression label of any shape into a float32 array."""
    y_np = np.asarray(y, dtype=np.float32)
    if y_np.ndim == 0:
        y_np = y_np.reshape(1)
    if y_np.ndim > 1:
        if y_np.shape[-1] == 1:
            y_np = y_np.reshape(-1)
        else:
            raise ValueError(f"The regression label should be 1 dimension or last dimension 1 and currently shape={y_np.shape}")
    return y_np.astype(np.float32)


def fit_target_normalizer_train_only(y_train, eps=1e-6):
    """Only labeling standardizer based on train."""
    y_train = ensure_regression_targets(y_train)
    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    std = max(std, float(eps))
    return {
        "enabled": True,
        "mean": mean,
        "std": std,
        "eps": float(eps),
    }


def normalize_targets(y, normalizer):
    """Press train-only normalizer to standardize."""
    y = ensure_regression_targets(y)
    if normalizer is None or not bool(normalizer.get("enabled", False)):
        return y.astype(np.float32)
    mean = float(normalizer["mean"])
    std = float(normalizer["std"])
    return ((y - mean) / std).astype(np.float32)


def denormalize_targets(y, normalizer=None):
    """Reverse the prediction/labelling of standardized space back to the original MMSE."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if normalizer is None:
        normalizer = globals().get("ACTIVE_TARGET_NORMALIZER", None)
    if normalizer is None or not bool(normalizer.get("enabled", False)):
        return y.astype(np.float32)
    mean = float(normalizer["mean"])
    std = float(normalizer["std"])
    return (y * std + mean).astype(np.float32)


def _maybe_denorm_np(y, normalizer=None):
    """
    If target normalizer is enabled, return standardized space from y to the original MMSE.
    If not enabled, return to float32 one-dimensional array.
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if normalizer is None:
        normalizer = globals().get("ACTIVE_TARGET_NORMALIZER", None)
    return denormalize_targets(y, normalizer)


def compute_regression_metrics_np(y_true, y_pred, already_denormalized=False):
    """
    Calculation of regression indicators.

    Parameters
    ----
    y_true, y_pred:
        One-dimensional labels and forecasts.
    already_denormalized:
        False: Inputs are still in standardized space, which is then counter-standardized to the original MMSE and then calculates the indicator.
        True : Enter already the original MMSE to calculate the indicator directly.
    """
    if already_denormalized:
        y_true = np.asarray(y_true, dtype=np.float32).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float32).reshape(-1)
    else:
        y_true = _maybe_denorm_np(y_true).reshape(-1)
        y_pred = _maybe_denorm_np(y_pred).reshape(-1)

    if len(y_true) == 0:
        return {
            "count": 0,
            "rmse": float("nan"),
            "mae": float("nan"),
            "r2": float("nan"),
            "pearson": float("nan"),
        }

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true Unmatched with y_print length: {y_true.shape[0]} vs {y_pred.shape[0]}")

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    if len(y_true) >= 2 and float(np.std(y_true)) > 1e-12:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = float("nan")

    if len(y_true) >= 2 and float(np.std(y_true)) > 1e-12 and float(np.std(y_pred)) > 1e-12:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    return {
        "count": int(len(y_true)),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "pearson": pearson,
    }


def summarize_target_distribution(labels, split_name, normalizer=None, already_normalized=False):
    """
    Prints the tab distribution.
    already_normalized=False: Print directly on the original MMSE
    already_normalized=True : Prints both standardized space and counter-standardized summaries of original MMSE
    """
    y = ensure_regression_targets(labels)

    if already_normalized:
        y_norm = y
        y_raw = denormalize_targets(y_norm, normalizer)
        print(
            f"{split_name} normalized_target_count={len(y_norm)} "
            f"min={float(np.min(y_norm)):.4f} max={float(np.max(y_norm)):.4f} "
            f"mean={float(np.mean(y_norm)):.4f} std={float(np.std(y_norm)):.4f}"
        )
        print(
            f"{split_name} raw_target_count={len(y_raw)} "
            f"min={float(np.min(y_raw)):.4f} max={float(np.max(y_raw)):.4f} "
            f"mean={float(np.mean(y_raw)):.4f} std={float(np.std(y_raw)):.4f}"
        )
    else:
        print(
            f"{split_name} raw_target_count={len(y)} "
            f"min={float(np.min(y)):.4f} max={float(np.max(y)):.4f} "
            f"mean={float(np.mean(y)):.4f} std={float(np.std(y)):.4f}"
        )


def _is_process2_dataset(dataset_name=None) -> bool:
    """
    Current PROCESS-2 dedicated judgement.
    Enable MMSE-bin weighted loss only on PROCESS-2.
    """
    if dataset_name is None:
        dataset_name = globals().get("DATASET_NAME", globals().get("DATASET_OPTION", ""))
    return str(dataset_name).strip().lower() == "process-2"


def assign_process2_merged_mmse_bin(mmse_value) -> str:
    """
    PROCESS-2 Specialized merge MMSE bin:
        [0,21), [21,26), [26,31)

    Note:
    - train [0,11) no samples, [11,16) very few samples;
      Training weights are therefore not used in the original 5 brackets and are replaced by the merged 3 brackets.
    """
    val = float(mmse_value)

    if 0.0 <= val < 21.0:
        return "[0,21)"
    if 21.0 <= val < 26.0:
        return "[21,26)"
    if 26.0 <= val < 31.0:
        return "[26,31)"

    return "out_of_range"


def compute_effective_number_bin_weights(bin_counts, beta=0.99, max_weight=4.0):
    """
    Calculated on the basis of the quantity of audio per MMSE bin effective-number weight.

    bin_counts:
        {"[0,21)": 9, "[21,26)": 28, "[26,31)": 100}

    Return:
        Every bin audio-level weight.
    """
    beta = float(beta)
    max_weight = float(max_weight)

    raw_weights = {}
    for bin_name, n in bin_counts.items():
        n = int(n)
        if n <= 0:
            continue

        effective_num = 1.0 - (beta**n)
        w = (1.0 - beta) / max(effective_num, float(PROCESS2_WEIGHT_EPS))
        raw_weights[bin_name] = float(w)

    total_audio = sum(int(v) for v in bin_counts.values() if int(v) > 0)
    if total_audio <= 0 or len(raw_weights) == 0:
        return {k: 1.0 for k in bin_counts.keys()}

    weighted_sum = 0.0
    for bin_name, w in raw_weights.items():
        weighted_sum += float(bin_counts[bin_name]) * float(w)

    mean_weight = weighted_sum / max(float(total_audio), float(PROCESS2_WEIGHT_EPS))

    bin_weights = {}
    for bin_name in bin_counts.keys():
        if bin_name not in raw_weights:
            bin_weights[bin_name] = 0.0
            continue

        w = float(raw_weights[bin_name]) / max(float(mean_weight), float(PROCESS2_WEIGHT_EPS))
        w = min(w, max_weight)
        bin_weights[bin_name] = float(w)

    return bin_weights


def build_process2_segment_weights(y_train_raw, idx_train_audio, beta=0.99, max_weight=4.0):
    """
    PROCESS-2 Specialized for segment weight.

    Logical:
    1) First, on the basis of audio, MMSE bin;
    2) At bin audio-level effective-number weight;
    3) The number of segment weights per segment = audio_bin_weight / current audio;
    4) The average weight of the final re-allocation to segment would be approximately 1, stabilizing the loss numerical scale.

    Parameters:
        y_train_raw:
            train segment Original MMSE Label, shape [N]
        idx_train_audio:
            train segment Corresponding to audio id, shape [N]
            From presplit cache idx_train.pkl

    Return:
        segment_weights:
            shape [N],Every loss weight on train segment
        info:
            Statistical information for printing and preservation
    """
    y_raw = ensure_regression_targets(y_train_raw).astype(np.float32)
    audio_ids = np.asarray(idx_train_audio).astype(np.int64).reshape(-1)

    if len(y_raw) != len(audio_ids):
        raise ValueError(
            "[PROCESS-2 weight] y_train_raw Inconsistent with idx_train_audio length:"
            f"len(y_train_raw)={len(y_raw)}, len(idx_train_audio)={len(audio_ids)}"
        )

    unique_audio_ids = sorted(set(audio_ids.tolist()))

    audio_mmse = {}
    audio_num_segments = {}

    for aid in unique_audio_ids:
        mask = audio_ids == int(aid)
        audio_mmse[int(aid)] = float(y_raw[mask][0])
        audio_num_segments[int(aid)] = int(mask.sum())

    bin_counts = {label: 0 for label in PROCESS2_MMSE_BIN_LABELS}
    out_of_range_count = 0

    audio_bin = {}
    for aid in unique_audio_ids:
        b = assign_process2_merged_mmse_bin(audio_mmse[int(aid)])
        audio_bin[int(aid)] = b

        if b in bin_counts:
            bin_counts[b] += 1
        else:
            out_of_range_count += 1

    bin_weights = compute_effective_number_bin_weights(
        bin_counts=bin_counts,
        beta=beta,
        max_weight=max_weight,
    )

    segment_weights = np.zeros(len(y_raw), dtype=np.float32)

    for i, aid in enumerate(audio_ids):
        aid = int(aid)
        b = audio_bin[aid]

        if b in bin_weights and float(bin_weights[b]) > 0:
            audio_weight = float(bin_weights[b])
        else:
            audio_weight = 1.0

        n_seg = max(int(audio_num_segments[aid]), 1)
        segment_weights[i] = float(audio_weight) / float(n_seg)

    # Normalize to segment average weight approximately 1
    segment_weights = segment_weights / max(
        float(segment_weights.mean()), float(PROCESS2_WEIGHT_EPS)
    )
    segment_weights = segment_weights.astype(np.float32)

    info = {
        "enabled": True,
        "method": "process2_merged_mmse_bin_effective_number_audio_level",
        "bin_edges": [float(x) for x in PROCESS2_MMSE_BIN_EDGES],
        "bin_labels": list(PROCESS2_MMSE_BIN_LABELS),
        "beta": float(beta),
        "max_weight": float(max_weight),
        "bin_counts_audio": {str(k): int(v) for k, v in bin_counts.items()},
        "bin_weights_audio": {str(k): float(v) for k, v in bin_weights.items()},
        "out_of_range_audio_count": int(out_of_range_count),
        "num_train_segments": int(len(y_raw)),
        "num_train_audio": int(len(unique_audio_ids)),
        "segment_weight_min": float(np.min(segment_weights)),
        "segment_weight_max": float(np.max(segment_weights)),
        "segment_weight_mean": float(np.mean(segment_weights)),
        "segment_weight_std": float(np.std(segment_weights)),
    }

    return segment_weights, info


def pack_regression_target_with_weight(y_target, segment_weights):
    """
    Pack training targets to [target, weight].
    - Column 0: The original regression label, possibly standardized y
    - Column 1: loss weight

    val/test No need to pack. Keep a dimensional tag.
    """
    y_target = ensure_regression_targets(y_target).astype(np.float32)
    segment_weights = np.asarray(segment_weights, dtype=np.float32).reshape(-1)

    if len(y_target) != len(segment_weights):
        raise ValueError(
            "y_target Inconsistent with segment_weights length:" f"{len(y_target)} != {len(segment_weights)}"
        )

    return np.stack([y_target, segment_weights], axis=1).astype(np.float32)


def compute_process2_audio_bin_metrics(y_true_audio, y_pred_audio):
    """
    PROCESS-2 Indicators specific to audio-level per-bin / macro-bin.
    Enter the original MMSE space.
    """
    y_true_audio = np.asarray(y_true_audio, dtype=np.float32).reshape(-1)
    y_pred_audio = np.asarray(y_pred_audio, dtype=np.float32).reshape(-1)

    rows = []
    rmse_values = []
    mae_values = []

    for bin_label in PROCESS2_MMSE_BIN_LABELS:
        left = float(bin_label.split(",")[0].replace("[", ""))
        right = float(bin_label.split(",")[1].replace(")", ""))
        mask = (y_true_audio >= left) & (y_true_audio < right)

        if int(mask.sum()) == 0:
            row = {
                "mmse_bin": bin_label,
                "count": 0,
                "rmse": float("nan"),
                "mae": float("nan"),
            }
        else:
            yt = y_true_audio[mask]
            yp = y_pred_audio[mask]
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            mae = float(np.mean(np.abs(yp - yt)))
            row = {
                "mmse_bin": bin_label,
                "count": int(mask.sum()),
                "rmse": rmse,
                "mae": mae,
            }
            rmse_values.append(rmse)
            mae_values.append(mae)

        rows.append(row)

    macro = {
        "macro_bin_rmse": float(np.mean(rmse_values)) if len(rmse_values) > 0 else float("nan"),
        "macro_bin_mae": float(np.mean(mae_values)) if len(mae_values) > 0 else float("nan"),
        "macro_bin_nonempty_count": int(len(rmse_values)),
    }

    return {
        "per_bin": rows,
        "macro": macro,
    }


def build_raw_dataset(dataset_name, include_time=False, T=1.0, seed=400):
    """
    Read train/val/test features and labels from presplit pkl and build dataset.
    Key changes:
    1) train-only y Standardization
    2) dataset Save target_normalizer
    3) Global ACTIVE_TARGET_NORMALIZER synchronized update for follow-up evaluation counter-standardisation
    """
    global ACTIVE_TARGET_NORMALIZER

    ds_dir = get_presplit_cache_dir(
        data_dir=DATA_DIR,
        dataset_name=dataset_name,
        feature_model_tag=FEATURE_MODEL_TAG,
        feature_selection_tag=FEATURE_SELECTION_TAG,
        file_tag=FILE_TAG,
    )
    if not os.path.exists(ds_dir):
        raise FileNotFoundError(
            f"No presplit cache directory found: {ds_dir}\n"
            f"Please confirm that convert_stage1_pt_to_mmse_presplit_pkl has been successfully implemented."
        )

    print("[build_raw_dataset] ds_dir =", ds_dir)
    # ========================================
    with open(os.path.join(ds_dir, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(ds_dir, "X_val.pkl"), "rb") as f:
        X_val = pickle.load(f)
    with open(os.path.join(ds_dir, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)

    with open(os.path.join(ds_dir, "y_train.pkl"), "rb") as f:
        y_train_raw = pickle.load(f)
    with open(os.path.join(ds_dir, "y_val.pkl"), "rb") as f:
        y_val_raw = pickle.load(f)
    with open(os.path.join(ds_dir, "y_test.pkl"), "rb") as f:
        y_test_raw = pickle.load(f)

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train_raw = ensure_regression_targets(y_train_raw)
    y_val_raw = ensure_regression_targets(y_val_raw)
    y_test_raw = ensure_regression_targets(y_test_raw)

    if FEATURE_STANDARDIZE:
        X_train = preprocess_data(X_train)
        X_val = preprocess_data(X_val)
        X_test = preprocess_data(X_test)

    if TARGET_STANDARDIZE:
        target_normalizer = fit_target_normalizer_train_only(
            y_train_raw,
            eps=TARGET_STANDARDIZE_EPS,
        )
        y_train = normalize_targets(y_train_raw, target_normalizer)
        y_val = normalize_targets(y_val_raw, target_normalizer)
        y_test = normalize_targets(y_test_raw, target_normalizer)
    else:
        target_normalizer = {
            "enabled": False,
            "mean": 0.0,
            "std": 1.0,
            "eps": float(TARGET_STANDARDIZE_EPS),
        }
        y_train = y_train_raw.astype(np.float32)
        y_val = y_val_raw.astype(np.float32)
        y_test = y_test_raw.astype(np.float32)

    if include_time:
        seq_len = int(X_train.shape[1])
        t = (float(T) / float(seq_len)) * np.arange(seq_len, dtype=np.float32)

        ts_train = np.repeat(t[None, :], X_train.shape[0], axis=0)
        ts_val = np.repeat(t[None, :], X_val.shape[0], axis=0)
        ts_test = np.repeat(t[None, :], X_test.shape[0], axis=0)

        X_train = np.concatenate([ts_train[:, :, None], X_train], axis=2)
        X_val = np.concatenate([ts_val[:, :, None], X_val], axis=2)
        X_test = np.concatenate([ts_test[:, :, None], X_test], axis=2)

    # ===================== PROCESS-2 Earmarked: Constructed train segment weights =====================
    train_segment_weights = np.ones(len(y_train), dtype=np.float32)
    process2_weight_info = {
        "enabled": False,
        "reason": "not_PROCESS-2_or_disabled",
    }

    y_train_for_loader = y_train.astype(np.float32)

    if bool(USE_PROCESS2_MMSE_BIN_WEIGHT) and _is_process2_dataset(dataset_name):
        idx_train_pkl_path = os.path.join(ds_dir, "idx_train.pkl")
        if not os.path.exists(idx_train_pkl_path):
            raise FileNotFoundError(
                "[PROCESS-2 weight] idx_train.pkl is required to amend segment weights by audio_id,"
                f"However, not found: {idx_train_pkl_path}"
            )

        with open(idx_train_pkl_path, "rb") as f:
            idx_train_audio = pickle.load(f)

        train_segment_weights, process2_weight_info = build_process2_segment_weights(
            y_train_raw=y_train_raw,
            idx_train_audio=idx_train_audio,
            beta=float(PROCESS2_MMSE_BIN_WEIGHT_BETA),
            max_weight=float(PROCESS2_MMSE_BIN_WEIGHT_MAX),
        )

        # Training loader with [target, weight]; target remains standardized with y_train.
        y_train_for_loader = pack_regression_target_with_weight(
            y_target=y_train,
            segment_weights=train_segment_weights,
        )

        print("[PROCESS-2 weight] enabled")
        print("[PROCESS-2 weight] bin_counts_audio =", process2_weight_info["bin_counts_audio"])
        print("[PROCESS-2 weight] bin_weights_audio =", process2_weight_info["bin_weights_audio"])
        print(
            "[PROCESS-2 weight] segment_weight "
            f"min={process2_weight_info['segment_weight_min']:.6f} "
            f"max={process2_weight_info['segment_weight_max']:.6f} "
            f"mean={process2_weight_info['segment_weight_mean']:.6f} "
            f"std={process2_weight_info['segment_weight_std']:.6f}"
        )

        process2_weight_json = os.path.join(ds_dir, "process2_mmse_bin_weight_info.json")
        with open(process2_weight_json, "w", encoding="utf-8") as f:
            json.dump(process2_weight_info, f, ensure_ascii=False, indent=2)
        print("[PROCESS-2 weight] saved:", process2_weight_json)

    raw_dataloaders = {
        "train": Dataloader(X_train, y_train_for_loader, inmemory=False),
        "val": Dataloader(X_val, y_val, inmemory=False),
        "test": Dataloader(X_test, y_test, inmemory=False),
    }

    data_dim = int(X_train.shape[-1])
    label_dim = 1

    dataset = SimpleNamespace(
        name=dataset_name,
        raw_dataloaders=raw_dataloaders,
        data_dim=data_dim,
        label_dim=label_dim,
        target_normalizer=target_normalizer,
        raw_targets={
            "train": y_train_raw,
            "val": y_val_raw,
            "test": y_test_raw,
        },
        normalized_targets={
            "train": y_train,
            "val": y_val,
            "test": y_test,
        },
        train_segment_weights=train_segment_weights,
        process2_weight_info=process2_weight_info,
    )

    ACTIVE_TARGET_NORMALIZER = target_normalizer
    return dataset


dataset = build_raw_dataset(DATASET_NAME, include_time=INCLUDE_TIME, T=T_MAX, seed=RANDOM_SEED)


print("data_dim =", dataset.data_dim)


print("label_dim =", dataset.label_dim)


print("train size =", dataset.raw_dataloaders["train"].size)


print("val size =", dataset.raw_dataloaders["val"].size)


print("test size =", dataset.raw_dataloaders["test"].size)


print("FEATURE_STANDARDIZE =", FEATURE_STANDARDIZE)


print("TARGET_STANDARDIZE  =", TARGET_STANDARDIZE)


print("target_normalizer   =", dataset.target_normalizer)


print("\n[Raw MMSE distribution]")


summarize_target_distribution(dataset.raw_targets["train"], "train", already_normalized=False)


summarize_target_distribution(dataset.raw_targets["val"], "val", already_normalized=False)


summarize_target_distribution(dataset.raw_targets["test"], "test", already_normalized=False)


print("\n[Normalized target distribution]")


summarize_target_distribution(
    dataset.normalized_targets["train"],
    "train",
    normalizer=dataset.target_normalizer,
    already_normalized=True,
)


summarize_target_distribution(
    dataset.normalized_targets["val"],
    "val",
    normalizer=dataset.target_normalizer,
    already_normalized=True,
)


summarize_target_distribution(
    dataset.normalized_targets["test"],
    "test",
    normalizer=dataset.target_normalizer,
    already_normalized=True,
)


#############################################################################################################


READOUT_POOL = "mean"


DEFAULT_MODEL_CONFIG = {
    "mamba3": {
        "backend": "mamba3",
        "num_blocks": 2,
        "hidden_dim": 128,
        "state_dim": 128,
        "headdim": 64,
        "expansion": 2,
        "dropout": 0.1,
        "pool": READOUT_POOL,
        "mlp_hidden_dim": 128,
        "is_mimo": False,
        "mimo_rank": 1,
        "chunk_size": 64,
        "is_outproj_norm": False,
        "rope_fraction": 0.5,
    }
}


def _get_active_learning_rate_for_model(model_name: str) -> float:
    model_name = normalize_model_name(model_name)
    return float(LR_MAMBA3)


def _get_model_display_name(model_name: str) -> str:
    """Log display with name."""
    model_name = normalize_model_name(model_name)
    return "Mamba3SISORegressor"


#########################################################################################################
# Note:
# - The two external layers remain the same for fair comparison.


try:
    from mamba_ssm import Mamba3 as Mamba3Layer

    MAMBA3_SSM_AVAILABLE = True
    MAMBA3_IMPORT_ERROR = ""
except Exception as e:
    MAMBA3_SSM_AVAILABLE = False
    MAMBA3_IMPORT_ERROR = str(e)
    print("[] mamba_ssm.Mamba3 was not imported successfully:", MAMBA3_IMPORT_ERROR)


class Mamba3SISOBlock(torch.nn.Module):
    "Mamba3SISOBlock:Model for mamba3 retained in this document."

    def __init__(
        self,
        hidden_dim: int,
        state_dim: int = 128,
        headdim: int = 64,
        expansion: int = 2,
        dropout: float = 0.10,
        chunk_size: int = 64,
        is_outproj_norm: bool = False,
        rope_fraction: float = 0.5,
    ):
        super().__init__()

        hidden_dim = int(hidden_dim)
        state_dim = int(state_dim)
        headdim = int(headdim)
        expansion = int(expansion)

        if (hidden_dim * expansion) % headdim != 0:
            raise ValueError(
                f"Mamba3 I'm asking you to ask for a hidden_dim * expansion to be removed by the headdim."
                f"Current hidden_dim={hidden_dim}, expansion={expansion}, headdim={headdim}."
            )

        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.mamba = Mamba3Layer(
            d_model=hidden_dim,
            d_state=state_dim,
            expand=expansion,
            headdim=headdim,
            is_mimo=False,
            mimo_rank=1,
            chunk_size=int(chunk_size),
            is_outproj_norm=bool(is_outproj_norm),
            rope_fraction=float(rope_fraction),
            dtype=None,
        )
        self.dropout = torch.nn.Dropout(float(dropout))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class MambaSequenceRegressor(torch.nn.Module):
    "MambaSequenceRegressor:Model for mamba3 retained in this document."

    def __init__(
        self,
        input_dim: int,
        num_blocks: int = 2,
        hidden_dim: int = 128,
        state_dim: int = 128,
        headdim: int = 64,
        expansion: int = 2,
        dropout: float = 0.1,
        pool: str = "mean",
        mlp_hidden_dim: int = 128,
        chunk_size: int = 64,
        is_outproj_norm: bool = False,
        rope_fraction: float = 0.5,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.pool = str(pool).strip().lower()
        self.block_type = "mamba3"
        if self.pool not in {"mean", "last", "meanmax"}:
            raise ValueError(f"Unsupported pooling: {pool}")
        self.input_proj = torch.nn.Linear(self.input_dim, self.hidden_dim)
        self.blocks = torch.nn.ModuleList(
            [
                Mamba3SISOBlock(
                    hidden_dim=self.hidden_dim,
                    state_dim=int(state_dim),
                    headdim=int(headdim),
                    expansion=int(expansion),
                    dropout=float(dropout),
                    chunk_size=int(chunk_size),
                    is_outproj_norm=bool(is_outproj_norm),
                    rope_fraction=float(rope_fraction),
                )
                for _ in range(int(num_blocks))
            ]
        )
        readout_dim = self.hidden_dim * 2 if self.pool == "meanmax" else self.hidden_dim
        self.readout_norm = torch.nn.LayerNorm(readout_dim)
        self.regressor = torch.nn.Sequential(
            torch.nn.Linear(readout_dim, int(mlp_hidden_dim)),
            torch.nn.GELU(),
            torch.nn.Dropout(float(dropout)),
            torch.nn.Linear(int(mlp_hidden_dim), 1),
        )

    def _pool_sequence(self, x):
        if self.pool == "mean":
            return torch.mean(x, dim=1)
        if self.pool == "last":
            return x[:, -1, :]
        if self.pool == "meanmax":
            return torch.cat([torch.mean(x, dim=1), torch.amax(x, dim=1)], dim=-1)
        raise RuntimeError(f"Unknown pooling: {self.pool}")

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self._pool_sequence(x)
        x = self.readout_norm(x)
        pred = self.regressor(x).squeeze(-1)
        return pred


class Mamba3SISORegressor(MambaSequenceRegressor):
    "Mamba3SISORegressor:Model for mamba3 retained in this document."

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def create_mamba_model_torch(data_dim, label_dim, config, seed=400, model_name=None):
    "create_mamba_model_torch:Model for mamba3 retained in this document."
    model_name = normalize_model_name(
        model_name if model_name is not None else config.get("model_name", MODEL_NAME)
    )
    if not MAMBA3_SSM_AVAILABLE:
        raise ImportError(
            "mamba_ssm.Mamba3 Not available. Please install state-spaces/mamba from the source code. Original error:"
            + str(MAMBA3_IMPORT_ERROR)
        )
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    common_kwargs = dict(
        input_dim=int(data_dim),
        num_blocks=int(config.get("num_blocks", 2)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        state_dim=int(config.get("state_dim", 128)),
        headdim=int(config.get("headdim", 64)),
        expansion=int(config.get("expansion", 2)),
        dropout=float(config.get("dropout", 0.1)),
        pool=str(config.get("pool", READOUT_POOL)),
        mlp_hidden_dim=int(config.get("mlp_hidden_dim", 128)),
        chunk_size=int(config.get("chunk_size", 64)),
    )
    model = Mamba3SISORegressor(
        **common_kwargs,
        is_outproj_norm=bool(config.get("is_outproj_norm", False)),
        rope_fraction=float(config.get("rope_fraction", 0.5)),
    )
    return (model, None)


################################################################################################################
# ===== 8. Return indicators and candidatures checkpoint tool =====


def evaluate_segment_metrics(segment_preds, segment_true_labels):
    """The segment level projection directly calculates the regression indicator and is a semantic envelope for `compute_regression_metrics_np`."""
    return compute_regression_metrics_np(segment_true_labels, segment_preds)


def _resolve_audio_agg_method(agg_method=None):
    method = (
        str(
            agg_method
            if agg_method is not None
            else globals().get("AUDIO_AGG_PRIMARY_METHOD", "mean")
        )
        .strip()
        .lower()
    )

    alias_map = {
        "mean": "mean",
        "avg": "mean",
        "average": "mean",
        "median": "median",
        "trimmed": "trimmed_mean",
        "trimmed_mean": "trimmed_mean",
        "trimmean": "trimmed_mean",
    }
    if method not in alias_map:
        raise ValueError(f"Unknown audio polymeric mode: {agg_method}")
    return alias_map[method]


def _resolve_audio_trim_ratio(trim_ratio=None):
    ratio = float(
        trim_ratio if trim_ratio is not None else globals().get("AUDIO_AGG_TRIM_RATIO", 0.10)
    )
    # Avoid extremes leading to all being emptied
    ratio = max(0.0, min(0.49, ratio))
    return ratio


def _trimmed_mean_1d(values, trim_ratio=0.10):
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return float("nan")
    if arr.size == 1:
        return float(arr[0])

    ratio = _resolve_audio_trim_ratio(trim_ratio)
    k = int(np.floor(arr.size * ratio))

    # Degradation to normal mean when too short
    if k <= 0 or (2 * k) >= arr.size:
        return float(np.mean(arr))

    arr_sorted = np.sort(arr)
    arr_trimmed = arr_sorted[k : arr.size - k]
    if arr_trimmed.size == 0:
        return float(np.mean(arr))
    return float(np.mean(arr_trimmed))


def _reduce_segment_predictions(values, method="mean", trim_ratio=0.10):
    method = _resolve_audio_agg_method(method)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)

    if arr.size == 0:
        return float("nan")

    if method == "mean":
        return float(np.mean(arr))
    if method == "median":
        return float(np.median(arr))
    if method == "trimmed_mean":
        return _trimmed_mean_1d(arr, trim_ratio=trim_ratio)

    raise ValueError(f"Unknown Aggregation: {method}")


def aggregate_audio_predictions(
    audio_ids,
    segment_preds,
    segment_true,
    agg_method=None,
    trim_ratio=None,
):
    """
    segment_preds / segment_true Inputs are usually still standardized.
    This is a counter-standardization to the original MMSE and then to aggregate at audio_id.

    Return:
      - metrics: Results of the current Main Aggregation Method (AUDIO_AGG_PRIMARY_METHOD)
      - all_methods: mean / median / trimmed_mean Three sets of results.
    """
    agg_method = _resolve_audio_agg_method(agg_method)
    trim_ratio = _resolve_audio_trim_ratio(trim_ratio)

    audio_ids = np.asarray(audio_ids).astype(int).reshape(-1)
    segment_preds = _maybe_denorm_np(segment_preds).reshape(-1)
    segment_true = _maybe_denorm_np(segment_true).reshape(-1)

    if not (len(audio_ids) == len(segment_preds) == len(segment_true)):
        raise ValueError("audio_ids / segment_preds / segment_true Length is inconsistent.")

    if len(audio_ids) == 0:
        empty = {
            "sorted_audio_ids": np.asarray([], dtype=int),
            "y_true_audio": np.asarray([], dtype=np.float32),
            "y_pred_audio": np.asarray([], dtype=np.float32),
            "metrics": compute_regression_metrics_np([], [], already_denormalized=True),
        }
        return {
            "aggregation_method": agg_method,
            "trim_ratio": float(trim_ratio),
            "sorted_audio_ids": empty["sorted_audio_ids"],
            "y_true_audio": empty["y_true_audio"],
            "y_pred_audio": empty["y_pred_audio"],
            "metrics": empty["metrics"],
            "all_methods": {
                "mean": dict(empty),
                "median": dict(empty),
                "trimmed_mean": dict(empty),
            },
        }

    pred_map = {}
    true_map = {}
    for aid, pred, y in zip(audio_ids, segment_preds, segment_true):
        aid = int(aid)
        pred_map.setdefault(aid, []).append(float(pred))
        true_map.setdefault(aid, []).append(float(y))

    sorted_ids = np.asarray(sorted(pred_map.keys()), dtype=int)
    y_true_audio = np.asarray(
        [float(np.mean(true_map[int(aid)])) for aid in sorted_ids],
        dtype=np.float32,
    )

    all_methods = {}
    for method_name in ["mean", "median", "trimmed_mean"]:
        y_pred_audio = np.asarray(
            [
                _reduce_segment_predictions(
                    pred_map[int(aid)],
                    method=method_name,
                    trim_ratio=trim_ratio,
                )
                for aid in sorted_ids
            ],
            dtype=np.float32,
        )

        metrics = compute_regression_metrics_np(
            y_true_audio,
            y_pred_audio,
            already_denormalized=True,  # Key: Avoid duplication of anti-standardization
        )

        per_bin_result = None
        if _is_process2_dataset(globals().get("DATASET_NAME", globals().get("DATASET_OPTION", ""))):
            per_bin_result = compute_process2_audio_bin_metrics(
                y_true_audio=y_true_audio,
                y_pred_audio=y_pred_audio,
            )
            metrics = dict(metrics)
            metrics.update(per_bin_result["macro"])

        all_methods[method_name] = {
            "sorted_audio_ids": sorted_ids.copy(),
            "y_true_audio": y_true_audio.copy(),
            "y_pred_audio": y_pred_audio,
            "metrics": metrics,
            "process2_per_bin_metrics": (
                None if per_bin_result is None else per_bin_result["per_bin"]
            ),
            "process2_macro_bin_metrics": (
                None if per_bin_result is None else per_bin_result["macro"]
            ),
        }

    selected = all_methods[agg_method]
    return {
        "aggregation_method": agg_method,
        "trim_ratio": float(trim_ratio),
        "sorted_audio_ids": selected["sorted_audio_ids"],
        "y_true_audio": selected["y_true_audio"],
        "y_pred_audio": selected["y_pred_audio"],
        "metrics": selected["metrics"],
        "process2_per_bin_metrics": selected.get("process2_per_bin_metrics", None),
        "process2_macro_bin_metrics": selected.get("process2_macro_bin_metrics", None),
        "all_methods": all_methods,
    }


def evaluate_audio_metrics(
    audio_ids, segment_preds, segment_true_labels, agg_method=None, trim_ratio=None
):
    """The full audio-level forecast results and indicators are then returned by audio-consolidation."""
    return aggregate_audio_predictions(
        audio_ids=audio_ids,
        segment_preds=segment_preds,
        segment_true=segment_true_labels,
        agg_method=agg_method,
        trim_ratio=trim_ratio,
    )


def print_audio_aggregation_comparison(tag, audio_metric_result):
    "print_audio_aggregation_comparison:Model for mamba3 retained in this document."
    print(f"\n[{tag}] audio aggregation comparison")

    if audio_metric_result is None:
        print("  <empty>")
        return

    # ===== Case 1: Complete structure, including all_methods =====
    if isinstance(audio_metric_result, dict) and ("all_methods" in audio_metric_result):
        for method_name in ["mean", "median", "trimmed_mean"]:
            if method_name not in audio_metric_result["all_methods"]:
                continue
            row = audio_metric_result["all_methods"][method_name]["metrics"]
            print(
                f"  {method_name:12s} | "
                f"rmse={float(row['rmse']):.4f} "
                f"mae={float(row['mae']):.4f} "
                f"r2={float(row['r2']):.4f} "
                f"pearson={float(row['pearson']):.4f}"
            )
        return

    # ===== Status 2: baseline structure, with only current main polymer result =====
    method_name = str(audio_metric_result.get("aggregation_method", "mean"))
    row = audio_metric_result.get("metrics", None)

    if row is None:
        print("  <missing metrics>")
        return

    print(
        f"  {method_name:12s} | "
        f"rmse={float(row['rmse']):.4f} "
        f"mae={float(row['mae']):.4f} "
        f"r2={float(row['r2']):.4f} "
        f"pearson={float(row['pearson']):.4f}"
    )


# no-leak Version checkpoint will no longer be used to review candidate / test-metric for epoch.
# Old functions select_epoch_candidates_from_snapshots() and choose_best_candidate()
# Deleted to avoid the misuse of official_test / legacy val_* indicators for model selection.
# ===== 8B. MAMBA3 Training and Evaluation Core (PyTorch Returning Branch) =====


def get_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_torch_batch(X, y, device):
    """
    Convert NumPy/JAX batch to PyTorch tensor on the specified device.

    Compatibility:
    1) Normal tag:
        y shape [B]
        weight = ones

    2) PROCESS-2 weighted loss Label:
        y shape [B, 2]
        y[:, 0] = target
        y[:, 1] = sample_weight
    """
    X_t = torch.as_tensor(np.asarray(X), dtype=torch.float32, device=device)

    y_np = np.asarray(y, dtype=np.float32)

    if y_np.ndim == 2 and y_np.shape[1] >= 2:
        target_np = y_np[:, 0]
        weight_np = y_np[:, 1]
    else:
        target_np = y_np.reshape(-1)
        weight_np = np.ones_like(target_np, dtype=np.float32)

    y_t = torch.as_tensor(target_np, dtype=torch.float32, device=device).reshape(-1)
    w_t = torch.as_tensor(weight_np, dtype=torch.float32, device=device).reshape(-1)
    w_t = torch.clamp(w_t, min=float(PROCESS2_WEIGHT_EPS))

    return X_t, y_t, w_t


def mixed_mse_loss_torch(preds, target, sample_weight=None):
    """
    PyTorch mixed loss,Align with JAX loss of lru+transformer+lstm.py.

    General data set:
        sample_weight For None or 1, mixed loss the equivalent is normal MSE.

    PROCESS-2:
        y (a) is packaged to [target, weight];
        Use mixed loss:
            ordinary_mse = mean(error^2)
            weighted_mse = sum(weight * error^2) / sum(weight)
            loss = (1 - alpha) * ordinary_mse + alpha * weighted_mse
    """
    preds = preds.reshape(-1)
    target = target.reshape(-1)

    sq_error = (preds - target) ** 2
    ordinary_mse = torch.mean(sq_error)

    if sample_weight is None:
        weighted_mse = ordinary_mse
    else:
        sample_weight = sample_weight.reshape(-1)
        sample_weight = torch.clamp(sample_weight, min=float(PROCESS2_WEIGHT_EPS))
        denom = torch.clamp(sample_weight.sum(), min=float(PROCESS2_WEIGHT_EPS))
        weighted_mse = (sample_weight * sq_error).sum() / denom

    alpha = torch.as_tensor(
        float(PROCESS2_MIXED_LOSS_ALPHA),
        dtype=ordinary_mse.dtype,
        device=ordinary_mse.device,
    )
    alpha = torch.clamp(alpha, min=0.0, max=1.0)

    loss = (1.0 - alpha) * ordinary_mse + alpha * weighted_mse
    return loss


def predict_segment_values_torch(model, dataloader, batch_size, device):
    """Implements a round of reasoning on the PyTorch branch and returns the forecast and label at the level of the subment."""
    model.eval()
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for X, y in dataloader.loop_epoch(batch_size):
            X_t, y_t, _ = to_torch_batch(X, y, device)
            preds = model(X_t)
            preds_all.append(preds.detach().cpu().numpy().reshape(-1))
            labels_all.append(y_t.detach().cpu().numpy().reshape(-1))

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return preds_all, labels_all


def _clone_torch_state_dict_cpu(model):
    """Copy the current PyTorch model parameter to CPU and avoid candidate checkpoint holding GPU visible."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_torch_state_dict_snapshot(model, state_dict_snapshot, device):
    """Load the current model with a snapshot of CPU/GPU state_dict."""
    model.load_state_dict(state_dict_snapshot, strict=True)
    model.to(device)
    return model


def _torch_warmup_cosine_lr(base_lr, step, num_steps, warmup_steps, cosine_alpha):
    """warmup + cosine decay, which corresponds to the sub-semantics of JAX. step starts with 1."""
    base_lr = float(base_lr)
    step = int(step)
    num_steps = int(max(num_steps, 1))
    warmup_steps = int(max(warmup_steps, 1))
    cosine_alpha = float(cosine_alpha)

    if step <= warmup_steps:
        return base_lr * float(step) / float(warmup_steps)

    denom = max(num_steps - warmup_steps, 1)
    progress = float(step - warmup_steps) / float(denom)
    progress = min(max(progress, 0.0), 1.0)

    cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
    decayed = cosine_alpha + (1.0 - cosine_alpha) * cosine_decay
    return base_lr * float(decayed)


def _set_torch_optimizer_lr(optimizer, lr_value):
    for group in optimizer.param_groups:
        group["lr"] = float(lr_value)


def evaluate_checkpoint_regression_pipeline_torch(
    model,
    dataloaders,
    batch_size,
    device,
    audio_ids_train,
    audio_ids_val=None,
    audio_ids_test=None,
    run_test=False,
    audio_agg_method=None,
    audio_trim_ratio=None,
):
    """
    PyTorch checkpoint segment/audio level of evaluation.

    Current agreement:
    - train = official_train
    - test  = official_test
    - dataloaders["val"] Just historical compatibility alias, real equivalent to official_test
    - Unified External Filling test_*; val_* is reserved only as an old code compatibility field
    """
    train_preds, train_labels = predict_segment_values_torch(
        model, dataloaders["train"], batch_size, device
    )

    # Priority read dataloaders["test"]; if non-existent, return to legacy dataloaders["val"]
    if "test" in dataloaders:
        test_loader = dataloaders["test"]
    elif "val" in dataloaders:
        test_loader = dataloaders["val"]
    else:
        raise ValueError("dataloaders['test'] or legacy dataloaders['val'] is required.")

    test_preds, test_labels = predict_segment_values_torch(model, test_loader, batch_size, device)

    train_segment_metrics = evaluate_segment_metrics(train_preds, train_labels)
    test_segment_metrics = evaluate_segment_metrics(test_preds, test_labels)

    train_audio_metrics = evaluate_audio_metrics(
        audio_ids_train,
        train_preds,
        train_labels,
        agg_method=audio_agg_method,
        trim_ratio=audio_trim_ratio,
    )

    if audio_ids_test is None:
        audio_ids_test = audio_ids_val
    if audio_ids_test is None:
        raise ValueError("audio_ids_test; legacy may be faxed to audio_ids_val if required.")

    test_audio_metrics = evaluate_audio_metrics(
        audio_ids_test,
        test_preds,
        test_labels,
        agg_method=audio_agg_method,
        trim_ratio=audio_trim_ratio,
    )

    return {
        "train_preds": train_preds,
        "train_labels": train_labels,
        "test_preds": test_preds,
        "test_labels": test_labels,
        "train_segment_metrics": train_segment_metrics,
        "test_segment_metrics": test_segment_metrics,
        "train_audio_metrics": train_audio_metrics,
        "test_audio_metrics": test_audio_metrics,
        # legacy compatibility aliases
        "val_preds": test_preds,
        "val_labels": test_labels,
        "val_segment_metrics": test_segment_metrics,
        "val_audio_metrics": test_audio_metrics,
    }


def infinite_epoch_batches_torch(dataloader, batch_size):
    """
    An unlimited catch iterative for a pure PyTorch branch.
    Not dependent on jax.random / dataloader.loop(key=...).
    Each epoch with a sample of NumPy shuffle training.
    """
    rng = np.random.default_rng()
    while True:
        for X, y in dataloader.loop_epoch(batch_size, shuffle=True, rng=rng):
            yield X, y


def train_mamba_model_torch(
    model_name,
    model,
    dataloaders,
    lr,
    batch_size,
    num_steps,
    print_steps,
    early_stopping_steps,
    warmup_steps,
    cosine_alpha,
    seed,
    dataset_name,
    epoch_selection_objective,
    audio_ids_val,
    use_early_stopping=True,
):
    """
    Mamba3SISORegressor A no-leak PyTorch training cycle.

    no-leak Rules:
    - official_test For monitoring / reporting;
    - best epoch without official_test;
    - checkpoint is not selected for official_test;
    - early stopping Based only on train_loss_epoch;
    - Eventually selected checkpoint is the last training epoch / early stop to stop epoch.
    """

    if audio_ids_val is None:
        raise ValueError(
            f"{model_name} Training needs audio_ids_val for the calculation of official_test monitor indicators."
        )

    device = get_torch_device()

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model = model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(lr),
        weight_decay=float(OPT_WEIGHT_DECAY),
    )

    steps_per_epoch = int(np.ceil(dataloaders["train"].size / batch_size))
    total_epochs = int(np.ceil(num_steps / max(steps_per_epoch, 1)))

    if (print_steps is None) or (str(print_steps).strip().lower() in ["", "none", "auto"]):
        effective_print_steps = int(max(steps_per_epoch, 1))
    else:
        effective_print_steps = int(max(int(print_steps), 1))

    history = {
        "step": [],
        "loss": [],
        "lr": [],
        "train_loss_epoch": [],
        "train_segment_rmse": [],
        "test_segment_rmse": [],
        "test_audio_rmse": [],
    }
    epoch_trace = []

    candidate_snapshots = []
    selected_state_dict_snapshot = _clone_torch_state_dict_cpu(model)
    selected_epoch = 0

    best_train_loss = float("inf")
    no_improve = 0
    stopped_early = False
    stop_reason = "completed_all_steps"

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_base_dir = get_output_root(
        OUTPUT_DATASET_TAG, model_name, OUTPUT_FEATURE_TAG, OUTPUT_SELECTION_TAG
    )
    run_dir = os.path.join(run_base_dir, run_time)
    os.makedirs(run_dir, exist_ok=True)

    train_loop = infinite_epoch_batches_torch(dataloaders["train"], batch_size)
    epoch_losses = []

    print(f"[{model_name}] torch_device = {device}")
    print(
        f"[{model_name}] effective_print_steps={effective_print_steps} (config_print_steps={print_steps})"
    )
    print(f"[{model_name}] readout = {_get_model_display_name(model_name)}")
    print(f"[{model_name}] target_standardize = {bool(globals().get('TARGET_STANDARDIZE', False))}")
    print(
        f"[{model_name}] active_target_normalizer = {globals().get('ACTIVE_TARGET_NORMALIZER', None)}"
    )
    print(
        f"[{model_name}] no-leak policy = {EPOCH_SELECTION_POLICY}; official_test monitor only = {TEST_MONITOR_ONLY}"
    )
    print(f"[{model_name}] early_stopping_monitor = {EARLY_STOPPING_MONITOR}")

    print(f"[{model_name}] USE_PROCESS2_MMSE_BIN_WEIGHT = {bool(USE_PROCESS2_MMSE_BIN_WEIGHT)}")
    print(f"[{model_name}] PROCESS2_MMSE_BIN_WEIGHT_BETA = {float(PROCESS2_MMSE_BIN_WEIGHT_BETA)}")
    print(f"[{model_name}] PROCESS2_MMSE_BIN_WEIGHT_MAX = {float(PROCESS2_MMSE_BIN_WEIGHT_MAX)}")
    print(f"[{model_name}] PROCESS2_MIXED_LOSS_ALPHA = {float(PROCESS2_MIXED_LOSS_ALPHA)}")
    t0 = time.time()

    for step in range(1, int(num_steps) + 1):
        X, y = next(train_loop)
        X_t, y_t, w_t = to_torch_batch(X, y, device)

        lr_value = _torch_warmup_cosine_lr(
            base_lr=lr,
            step=step,
            num_steps=num_steps,
            warmup_steps=warmup_steps,
            cosine_alpha=cosine_alpha,
        )
        _set_torch_optimizer_lr(optimizer, lr_value)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        preds = model(X_t)
        loss = mixed_mse_loss_torch(preds, y_t, w_t)

        if not torch.isfinite(loss):
            raise FloatingPointError(f"{model_name} loss Non-limited value: step={step}, loss={loss.item()}")

        loss.backward()

        if float(GRAD_CLIP_NORM) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(GRAD_CLIP_NORM))

        optimizer.step()

        loss_value = float(loss.detach().cpu().item())
        epoch_losses.append(loss_value)

        is_epoch_end = step % steps_per_epoch == 0
        is_step_log = step % effective_print_steps == 0
        need_eval = is_step_log or is_epoch_end

        if need_eval:
            train_preds, train_labels = predict_segment_values_torch(
                model, dataloaders["train"], batch_size, device
            )

            if "test" in dataloaders:
                test_loader = dataloaders["test"]
            else:
                test_loader = dataloaders["val"]

            test_preds, test_labels = predict_segment_values_torch(
                model, test_loader, batch_size, device
            )

            train_segment_metrics = evaluate_segment_metrics(train_preds, train_labels)
            test_segment_metrics = evaluate_segment_metrics(test_preds, test_labels)
            test_audio_metrics = evaluate_audio_metrics(
                audio_ids_val,  # legacy Parameters name;actually corresponds to official_test audio ids
                test_preds,
                test_labels,
                agg_method=AUDIO_AGG_PRIMARY_METHOD,
                trim_ratio=AUDIO_AGG_TRIM_RATIO,
            )["metrics"]

            history["step"].append(int(step))
            history["loss"].append(float(loss_value))
            history["lr"].append(float(lr_value))
            history["train_segment_rmse"].append(float(train_segment_metrics["rmse"]))
            history["test_segment_rmse"].append(float(test_segment_metrics["rmse"]))
            history["test_audio_rmse"].append(float(test_audio_metrics["rmse"]))

            print(
                f"[{model_name}] step={step}/{int(num_steps)} "
                f"loss={loss_value:.6f} "
                f"test_segment_rmse={float(test_segment_metrics['rmse']):.4f} "
                f"test_audio_rmse={float(test_audio_metrics['rmse']):.4f} "
                f"elapsed={(time.time() - t0):.1f}s "
                f"[monitor only]"
            )

            if is_epoch_end:
                epoch_id = int(step // steps_per_epoch)
                train_loss_epoch = (
                    float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else float(loss_value)
                )
                epoch_losses = []

                history["train_loss_epoch"].append(float(train_loss_epoch))

                selected_epoch = int(epoch_id)
                selected_state_dict_snapshot = _clone_torch_state_dict_cpu(model)

                # Note: The current checkpoint of epoch/ to stop epoch is kept here, not the best official_test for checkpoint.
                torch.save(selected_state_dict_snapshot, os.path.join(run_dir, "selected_model.pt"))
                torch.save(selected_state_dict_snapshot, os.path.join(run_dir, "last_model.pt"))

                improved_train_loss = train_loss_epoch < (
                    best_train_loss - float(EARLY_STOPPING_MIN_DELTA)
                )
                if improved_train_loss:
                    best_train_loss = float(train_loss_epoch)
                    no_improve = 0
                else:
                    no_improve += 1

                candidate_snapshots.append(
                    {
                        "epoch": int(epoch_id),
                        "step": int(step),
                        "train_loss_epoch": float(train_loss_epoch),
                        "test_segment_rmse": float(test_segment_metrics["rmse"]),
                        "test_audio_rmse": float(test_audio_metrics["rmse"]),
                        # no-leak Tag
                        "selection_policy": str(EPOCH_SELECTION_POLICY),
                        "selected_by": "last_epoch_or_train_loss_early_stop",
                        "test_monitor_used_for_selection": 0,
                        "early_stopping_monitor": str(EARLY_STOPPING_MONITOR),
                        # legacy aliases:Compatibility only for follow-up forms, not for selection
                        "val_segment_rmse": float(test_segment_metrics["rmse"]),
                        "val_audio_rmse": float(test_audio_metrics["rmse"]),
                        "val_selection_score": float(train_loss_epoch),
                        "test_selection_score": float(train_loss_epoch),
                        "state_dict_snapshot": selected_state_dict_snapshot,
                    }
                )

                epoch_trace.append(
                    {
                        "epoch": int(epoch_id),
                        "step": int(step),
                        "loss": float(loss_value),
                        "lr": float(lr_value),
                        "train_loss_epoch": float(train_loss_epoch),
                        "train_segment_rmse": float(train_segment_metrics["rmse"]),
                        "train_segment_mae": float(train_segment_metrics["mae"]),
                        "test_segment_rmse": float(test_segment_metrics["rmse"]),
                        "test_segment_mae": float(test_segment_metrics["mae"]),
                        "test_segment_r2": float(test_segment_metrics["r2"]),
                        "test_audio_rmse": float(test_audio_metrics["rmse"]),
                        "test_audio_mae": float(test_audio_metrics["mae"]),
                        "test_audio_r2": float(test_audio_metrics["r2"]),
                        "epoch_selection_objective": str(epoch_selection_objective),
                        "test_selection_score": float(train_loss_epoch),
                        "selection_policy": str(EPOCH_SELECTION_POLICY),
                        "selected_by": "last_epoch_or_train_loss_early_stop",
                        "test_monitor_used_for_selection": 0,
                        "early_stopping_monitor": str(EARLY_STOPPING_MONITOR),
                        "early_stopping_used": int(bool(use_early_stopping)),
                        "no_improve": int(no_improve),
                        # legacy compatibility aliases
                        "val_segment_rmse": float(test_segment_metrics["rmse"]),
                        "val_segment_mae": float(test_segment_metrics["mae"]),
                        "val_segment_r2": float(test_segment_metrics["r2"]),
                        "val_audio_rmse": float(test_audio_metrics["rmse"]),
                        "val_audio_mae": float(test_audio_metrics["mae"]),
                        "val_audio_r2": float(test_audio_metrics["r2"]),
                        "val_selection_score": float(train_loss_epoch),
                        "is_best_epoch": 0,
                    }
                )

                print(
                    f"[{model_name}] epoch={epoch_id}/{total_epochs} "
                    f"train_loss_epoch={train_loss_epoch:.6f} "
                    f"test_segment_rmse={float(test_segment_metrics['rmse']):.4f} "
                    f"test_audio_rmse={float(test_audio_metrics['rmse']):.4f} "
                    f"[official_test monitor only]"
                )

                if use_early_stopping and (no_improve >= int(early_stopping_steps)):
                    stopped_early = True
                    stop_reason = f"train_loss_no_improve_{int(early_stopping_steps)}_epochs"
                    print(
                        f"[{model_name}] early stopping at epoch={epoch_id}; monitor=train_loss; no test metric used."
                    )
                    break

    torch.save(selected_state_dict_snapshot, os.path.join(run_dir, "selected_model.pt"))
    torch.save(_clone_torch_state_dict_cpu(model), os.path.join(run_dir, "last_model.pt"))

    np.save(os.path.join(run_dir, "steps.npy"), np.array(history["step"]))
    np.save(os.path.join(run_dir, "loss.npy"), np.array(history["loss"]))
    np.save(os.path.join(run_dir, "lr.npy"), np.array(history["lr"]))
    np.save(os.path.join(run_dir, "train_loss_epoch.npy"), np.array(history["train_loss_epoch"]))
    np.save(
        os.path.join(run_dir, "train_segment_rmse.npy"), np.array(history["train_segment_rmse"])
    )
    np.save(os.path.join(run_dir, "test_segment_rmse.npy"), np.array(history["test_segment_rmse"]))
    np.save(os.path.join(run_dir, "test_audio_rmse.npy"), np.array(history["test_audio_rmse"]))

    # legacy compatibility aliases
    np.save(os.path.join(run_dir, "val_segment_rmse.npy"), np.array(history["test_segment_rmse"]))
    np.save(os.path.join(run_dir, "val_audio_rmse.npy"), np.array(history["test_audio_rmse"]))

    epoch_decision_info = {
        "selection_policy": str(EPOCH_SELECTION_POLICY),
        "selected_epoch": int(selected_epoch),
        "selected_by": "last_epoch_or_train_loss_early_stop",
        "test_monitor_used_for_selection": 0,
        "early_stopping_monitor": str(EARLY_STOPPING_MONITOR),
        "early_stopping_used": int(bool(use_early_stopping)),
        "stopped_early": int(bool(stopped_early)),
        "stop_reason": str(stop_reason),
        "best_train_loss": float(best_train_loss),
    }

    return {
        "model": model,
        "best_state_dict": selected_state_dict_snapshot,
        "selected_state_dict": selected_state_dict_snapshot,
        "best_epoch": int(selected_epoch),
        "selected_epoch": int(selected_epoch),
        "best_test_selection_score": float(best_train_loss),
        "best_val_selection_score": float(best_train_loss),
        "candidate_snapshots": candidate_snapshots,
        "run_dir": run_dir,
        "history": history,
        "epoch_trace": epoch_trace,
        "epoch_decision_info": epoch_decision_info,
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": total_epochs,
        "num_steps_run": int(num_steps),
        "device": device,
    }


# ===== 9. Retrodiagnosis mapping (repair version: avoid re-standardization in drawing) =====


def plot_regression_diagnostics(
    labels,
    predictions,
    plot_title_prefix="",
    metrics=None,
    already_denormalized=True,
    bins=20,
):
    """
    Production of regression diagnostic maps, including:
      1) True vs Pred scatter
      2) Residual histogram

    Parameters
    ----
    labels / predictions:
        If already_denormalized=True, they should have been "originally MMSE"
        If already_denormalized=False, they should also be "standard space" values
    metrics:
        Selectable. If you enter, use metrics directly as a title presentation, without recalculating.
        Fits for val_audio_metrics / test_audio_metrics["metrics"].
    already_denormalized:
        True  -> Input is original MMSE and no longer anti-standardized
        False -> Enter still standardized space, the function will first counter-standardize
    bins:
        Number of debris Histogram Boxes
    """
    labels_np = np.asarray(labels, dtype=np.float32).reshape(-1)
    pred_np = np.asarray(predictions, dtype=np.float32).reshape(-1)

    if labels_np.shape[0] != pred_np.shape[0]:
        raise ValueError(
            f"labels Unmatched with length of prections: {labels_np.shape[0]} vs {pred_np.shape[0]}"
        )

    # ===== Critical repair point =====
    # If metrics is not passed from outside, it is recalculated here;
    # For audio plot, labels/predictions is already the original MMSE,
    # It is therefore necessary to set up already_denormalized=True to avoid a repetition of anti-standardization.
    if metrics is None:
        metrics = compute_regression_metrics_np(
            labels_np,
            pred_np,
            already_denormalized=already_denormalized,
        )

    # Coordinates used for drawings. The unifying guarantee is the original MMSE
    if already_denormalized:
        labels_plot = labels_np
        pred_plot = pred_np
    else:
        labels_plot = _maybe_denorm_np(labels_np)
        pred_plot = _maybe_denorm_np(pred_np)

    labels_plot = np.asarray(labels_plot, dtype=np.float32).reshape(-1)
    pred_plot = np.asarray(pred_plot, dtype=np.float32).reshape(-1)

    lo = float(min(labels_plot.min(), pred_plot.min()))
    hi = float(max(labels_plot.max(), pred_plot.max()))
    pad = max((hi - lo) * 0.05, 0.5)

    r2_text = "nan" if np.isnan(metrics["r2"]) else f"{metrics['r2']:.4f}"

    # ===== Scatter =====
    plt.figure(figsize=(6, 5))
    plt.scatter(labels_plot, pred_plot, alpha=0.75)
    plt.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        linestyle="--",
        linewidth=1,
    )
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.xlabel("True MMSE")
    plt.ylabel("Predicted MMSE")
    plt.title(
        f"{plot_title_prefix} Scatter\n"
        f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, R²={r2_text}"
    )
    plt.grid(alpha=0.3)
    plt.show()

    # ===== Residual Histogram =====
    residuals = pred_plot - labels_plot
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=bins)
    plt.xlabel("Residual (Pred - True)")
    plt.ylabel("Count")
    plt.title(f"{plot_title_prefix} Residual Histogram")
    plt.grid(alpha=0.3)
    plt.show()

    return metrics


import os


import sys


import io


import atexit


import pprint


from datetime import datetime


import pandas as pd


# ============================================================
# This cell is based on the original append_result_to_csv tool function,
# Add a new "notebook output log system":
#
# Objectives:
# 1) From the training stage, output the follow-up print/ misreporting / display form
#    (b) Harmonize to the same log document;
# 2) Do not destroy the display of the notebook on the page;
# 3) As much as possible to keep repeatable: if you repeat the current phase, you automatically unload the old logger
#    Reload a new logger to avoid repeating the same article output.
#
# Note:
# - The "log" here is a pure text log;
# - matplotlib The image cannot be embedded directly into the txt log, so here it is:
#   a) Write in the log a description of "the map is shown";
#   b) Save the map as png file and write the saving path to log;
# - pandas.DataFrame / Series / dict / list , if display()
#   The presentation will also be translated into readable text and written to the same log, to the extent possible.
# ============================================================


def append_result_to_csv(row, csv_path):
    """
    Adds a line to CSV and automatically compats inconsistencies.

    Processing logic:
    - If the target file does not exist, create it directly;
    - If already exists, read the old table and then add new and old columns to the list;
    - Eventually returned to DataFrame after writing back, so that notebook could continue.
    """
    # The current line should be changed to DataFrame
    new_df = pd.DataFrame([row])

    # If file does not exist, write directly and return
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        new_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return new_df

    # Read Old File
    old_df = pd.read_csv(csv_path)

    # Compatibility columns are inconsistent: merge
    all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))

    # Alignment
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    # Append
    out_df = pd.concat([old_df, new_df], ignore_index=True)

    # Write back
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return out_df


# =========================
# 1) Log Path Configuration
# =========================
# Logs are centrally placed under the current experimental output directory, so that the results of the CSV / checkpoint / projections are managed together.


NOTEBOOK_LOG_DIR = os.path.join(OUTPUT_EXPORT_PREFIX, "notebook_logs")


os.makedirs(NOTEBOOK_LOG_DIR, exist_ok=True)


# Use the time stamp to generate the log filename for this run.
# Benefits:
# - New files are generated every time a log is restarted, and no logs from the previous experiment are covered;
# - This path will be shared from cell17 to the end of cell.


NOTEBOOK_LOG_TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")


NOTEBOOK_OUTPUT_LOG_PATH = os.path.join(
    NOTEBOOK_LOG_DIR, f"cell17_and_after_output_{NOTEBOOK_LOG_TIME_TAG}.log"
)


# Saves the image separately to figures subdirectories and writes the image path to the same txt log.


NOTEBOOK_LOG_FIGURE_DIR = os.path.join(NOTEBOOK_LOG_DIR, f"figures_{NOTEBOOK_LOG_TIME_TAG}")


os.makedirs(NOTEBOOK_LOG_FIGURE_DIR, exist_ok=True)


# =========================
# 2) Some basic tool functions
# =========================


def _safe_close_notebook_log_handle():
    """
    Securely closes the handle of the old log file.
    Repeats the current phase and attempts to close the old handle to prevent the leak of the file handle.
    """
    handle = globals().get("_NOTEBOOK_LOG_FILE_HANDLE", None)
    if handle is not None:
        try:
            handle.flush()
            handle.close()
        except Exception:
            pass
    globals()["_NOTEBOOK_LOG_FILE_HANDLE"] = None


def _open_notebook_log_handle(log_path):
    """
    Opens the log file handle (additional mode).
    This is a separate envelope that allows the current phase to be repeated.
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handle = open(log_path, "a", encoding="utf-8")
    globals()["_NOTEBOOK_LOG_FILE_HANDLE"] = handle
    return handle


def _get_notebook_log_handle():
    """
    Take the current global log file handle.
    If the handle does not exist or is closed, press the current NOTEBOOK_OUTPUT_LOG_PATH to reopen.
    """
    handle = globals().get("_NOTEBOOK_LOG_FILE_HANDLE", None)
    if handle is None or getattr(handle, "closed", False):
        handle = _open_notebook_log_handle(NOTEBOOK_OUTPUT_LOG_PATH)
    return handle


def _write_notebook_log_raw(text):
    """
    Original text to txt log without additional formatting.

    Note:
    - It's not going to be print to notebook;
    - Responsibility for writing documents only.
    """
    if text is None:
        return
    handle = _get_notebook_log_handle()
    handle.write(str(text))
    handle.flush()


def write_notebook_log_line(text=""):
    """
    Writes a whole line of text to the log.
    An automatic line break will be added to facilitate the follow-up to grep / search for / manually.
    """
    _write_notebook_log_raw(f"{text}\n")


def notebook_log_section(title):
    """
    Insert a distinct "subsection heading" in the same log.

    Purpose:
    - Mark " Which implementation to start now cell ";
    - To make it easier to locate follow-up training logs, DataFrame output, and misdirected information.
    """
    title = str(title)
    bar = "=" * 100
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    write_notebook_log_line("")
    write_notebook_log_line(bar)
    write_notebook_log_line(f"[{now_str}] {title}")
    write_notebook_log_line(bar)


def _format_object_for_log(obj):
    """
    Converts the object displayed at display() to a string that is suitable for writing the pure text at log as far as possible.

    Dealing with priority:
    1) DataFrame / Series:Converted to table text;
    2) dict / list / tuple / set:Use of pprint beauty;
    3) matplotlib Figure:Recording summary information such as map titles, axes quantities;
    4) Other object: Return to repr(obj).
    """
    try:
        import matplotlib.figure as mpl_figure
    except Exception:
        mpl_figure = None

    if obj is None:
        return "None"

    if isinstance(obj, pd.DataFrame):
        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.width",
            200,
            "display.max_colwidth",
            200,
        ):
            return obj.to_string(index=True)

    if isinstance(obj, pd.Series):
        with pd.option_context(
            "display.max_rows",
            None,
            "display.width",
            200,
            "display.max_colwidth",
            200,
        ):
            return obj.to_string()

    if isinstance(obj, (dict, list, tuple, set)):
        return pprint.pformat(obj, width=140, compact=False, sort_dicts=False)

    if (mpl_figure is not None) and isinstance(obj, mpl_figure.Figure):
        fig_title = ""
        try:
            if getattr(obj, "_suptitle", None) is not None:
                fig_title = obj._suptitle.get_text()
        except Exception:
            fig_title = ""
        return f"<matplotlib.figure.Figure axes={len(obj.axes)} suptitle={fig_title!r}>"

    try:
        return repr(obj)
    except Exception as e:
        return f"<unreprable object: {type(obj).__name__}, error={e}>"


# =========================
# 3) stdout / stderr Double (Tee)
# =========================


class NotebookLogTee(io.TextIOBase):
    """
    A simple tee lieam:
    - One is still written at notebook, formerly stdout / stderr;
    - An additional entry is included in NOTEBOOK_OUTPUT_LOG_PATH.

    The benefits of doing so are:
    - You can still see print in real time in a terminal or interface;
    - All print / traceback will be kept in the unified log.
    """

    def __init__(self, original_stream, stream_name="STDOUT"):
        super().__init__()
        self.original_stream = original_stream
        self.stream_name = stream_name
        self._is_notebook_log_tee = True

    def write(self, s):
        # Retain the original display behaviour notebook
        if self.original_stream is not None:
            self.original_stream.write(s)

        # Write back to the uniform log file
        _write_notebook_log_raw(s)
        return len(s)

    def flush(self):
        if self.original_stream is not None:
            self.original_stream.flush()
        handle = globals().get("_NOTEBOOK_LOG_FILE_HANDLE", None)
        if handle is not None and not getattr(handle, "closed", False):
            handle.flush()

    def isatty(self):
        return False


def _restore_previous_notebook_logger_if_needed():
    """
    If the last logger has been hung, the original stdout / stderr / display / plt.show will be restored here.
    Re-installation of new logger to avoid "same output being repeated several times".
    """
    # Restore stdout
    if getattr(sys.stdout, "_is_notebook_log_tee", False):
        try:
            sys.stdout = sys.stdout.original_stream
        except Exception:
            pass

    # Restore stderr
    if getattr(sys.stderr, "_is_notebook_log_tee", False):
        try:
            sys.stderr = sys.stderr.original_stream
        except Exception:
            pass

    # Restore display
    if globals().get("_NOTEBOOK_LOGGER_DISPLAY_PATCHED", False):
        if "_NOTEBOOK_LOG_ORIGINAL_DISPLAY" in globals():
            globals()["display"] = globals()["_NOTEBOOK_LOG_ORIGINAL_DISPLAY"]
        globals()["_NOTEBOOK_LOGGER_DISPLAY_PATCHED"] = False

    # Restore plt.show
    if globals().get("_NOTEBOOK_LOGGER_PLT_SHOW_PATCHED", False):
        try:
            import matplotlib.pyplot as plt

            if "_NOTEBOOK_LOG_ORIGINAL_PLT_SHOW" in globals():
                plt.show = globals()["_NOTEBOOK_LOG_ORIGINAL_PLT_SHOW"]
        except Exception:
            pass
        globals()["_NOTEBOOK_LOGGER_PLT_SHOW_PATCHED"] = False

    # Close old file handles
    _safe_close_notebook_log_handle()


# =========================
# 4) display() Supplementary log with plt.show()
# =========================


def _patch_display_for_logging():
    """
    Packaging display():
    - The page is still displayed as normal;
    - The text of an object display is also written into the same log.
    """
    try:
        from IPython.display import display as ipy_display
    except Exception:
        return

    original_display = globals().get("display", None)
    if original_display is None:
        original_display = ipy_display

    globals()["_NOTEBOOK_LOG_ORIGINAL_DISPLAY"] = original_display

    def logged_display(*objs, **kwargs):
        # Maintain normal rendering behavior for notebook
        original_display(*objs, **kwargs)

        # And then the text of the summary of objects / will be added to log
        if len(objs) == 0:
            write_notebook_log_line("[display] <no object>")
            return

        write_notebook_log_line("[display] Output objects are as follows:")
        for idx, obj in enumerate(objs, start=1):
            write_notebook_log_line(f"[display][obj_{idx}] type={type(obj).__name__}")
            write_notebook_log_line(_format_object_for_log(obj))
            write_notebook_log_line("-" * 80)

    globals()["display"] = logged_display
    globals()["_NOTEBOOK_LOGGER_DISPLAY_PATCHED"] = True


def _patch_matplotlib_show_for_logging():
    """
    Packaging plt.show():
    - Still normally displays the graph in notebook;
    - Saves the current to be displayed image to png;
    - and write the image save path to txt log.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    original_show = plt.show
    globals()["_NOTEBOOK_LOG_ORIGINAL_PLT_SHOW"] = original_show

    def logged_show(*args, **kwargs):
        fig_nums = list(plt.get_fignums())

        # Keep all current figure drives and write logs before the true show.
        for local_idx, fig_num in enumerate(fig_nums, start=1):
            try:
                fig = plt.figure(fig_num)
                fig_title = ""
                if getattr(fig, "_suptitle", None) is not None:
                    fig_title = fig._suptitle.get_text()

                fig_path = os.path.join(
                    NOTEBOOK_LOG_FIGURE_DIR,
                    f"figure_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{local_idx:02d}.png",
                )
                fig.savefig(fig_path, dpi=200, bbox_inches="tight")
                write_notebook_log_line(
                    f"[figure] Saved Image | fig_num={fig_num} | title={fig_title!r} | path={fig_path}"
                )
            except Exception as e:
                write_notebook_log_line(
                    f"[figure] Failed to save image | fig_num={fig_num} | error={repr(e)}"
                )

        return original_show(*args, **kwargs)

    plt.show = logged_show
    globals()["_NOTEBOOK_LOGGER_PLT_SHOW_PATCHED"] = True


# =========================
# 5) Activate log system
# =========================


def enable_notebook_output_logging():
    """
    Launch of the notebook Harmonized Log system, which will be effective after the current phase.

    Scope of entry into force:
    - print(...) -> Write terminal or interactive interface + to the same txt log
    - Error reporting traceback -> Write terminal or interactive interface + Add to the same txt log
    - display(df / dict / list / series / figure Summary ) -> to the same txt log
    - plt.show() -> Save extra png and write the path to the same txt log
    """
    # Unmount old logger and avoid repeating patch
    _restore_previous_notebook_logger_if_needed()

    # Open a new log file
    _open_notebook_log_handle(NOTEBOOK_OUTPUT_LOG_PATH)

    # Record start
    write_notebook_log_line("#" * 100)
    write_notebook_log_line("# Training output log")
    write_notebook_log_line(f"# start_time = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_notebook_log_line(f"# log_path   = {NOTEBOOK_OUTPUT_LOG_PATH}")
    write_notebook_log_line(f"# figure_dir = {NOTEBOOK_LOG_FIGURE_DIR}")
    write_notebook_log_line("#" * 100)

    # Save original stdout / stderr, followed by tee
    globals()["_NOTEBOOK_LOG_ORIGINAL_STDOUT"] = sys.stdout
    globals()["_NOTEBOOK_LOG_ORIGINAL_STDERR"] = sys.stderr

    sys.stdout = NotebookLogTee(globals()["_NOTEBOOK_LOG_ORIGINAL_STDOUT"], "STDOUT")
    sys.stderr = NotebookLogTee(globals()["_NOTEBOOK_LOG_ORIGINAL_STDERR"], "STDERR")

    # Patch display / plt.show
    _patch_display_for_logging()
    _patch_matplotlib_show_for_logging()

    # Close file handle automatically when registered to exit
    atexit.register(_safe_close_notebook_log_handle)

    # Give this log a general title
    notebook_log_section("The log system has been activated: from the training phase, the same log has been integrated.")

    # Also point the log position in the terminal or interactive interface so that you can go straight to it.
    print("[RunLog] A uniform output log has been enabled.")
    print("[RunLog] log_path   =", NOTEBOOK_OUTPUT_LOG_PATH)
    print("[RunLog] figure_dir =", NOTEBOOK_LOG_FIGURE_DIR)


# Start immediately.
# The output of the cell 17/ 18/19/20/ 21 will automatically fall to the same log if the current phase is run.


enable_notebook_output_logging()


# ===== Mamba3-only Single Result Saver =====


def _metric_or_nan(metric_block, key):
    if metric_block is None:
        return np.nan
    return float(metric_block.get(key, np.nan))


def _get_official_test_segment_metrics(selected_pipeline):
    return (
        selected_pipeline.get("test_segment_metrics")
        if selected_pipeline.get("test_segment_metrics") is not None
        else selected_pipeline.get("val_segment_metrics")
    )


def _get_official_test_audio_metrics(selected_pipeline):
    return (
        selected_pipeline.get("test_audio_metrics")
        if selected_pipeline.get("test_audio_metrics") is not None
        else selected_pipeline.get("val_audio_metrics")
    )


def save_single_run_outputs(
    model_name,
    seed,
    selected_epoch,
    selected_pipeline,
    run_dir=None,
):
    """Saves Mamba3 single run results; common external use of test_* names."""
    os.makedirs(OUTPUT_EXPORT_PREFIX, exist_ok=True)

    model_name = normalize_model_name(model_name)
    export_prefix = os.path.join(
        OUTPUT_EXPORT_PREFIX, f"{DATASET_NAME}_{model_name}_seed{int(seed)}"
    )

    train_segment_metrics = selected_pipeline.get("train_segment_metrics")
    train_audio_metrics = (
        None
        if selected_pipeline.get("train_audio_metrics") is None
        else selected_pipeline["train_audio_metrics"]["metrics"]
    )

    test_segment_metrics = _get_official_test_segment_metrics(selected_pipeline)
    test_audio_result = _get_official_test_audio_metrics(selected_pipeline)
    test_audio_metrics = None if test_audio_result is None else test_audio_result["metrics"]

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": DATASET_NAME,
        "model": model_name,
        "split_protocol": "official_train_as_train_official_test_as_test",
        "train_semantics": "official_train",
        "test_semantics": "official_test",
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "run_dir": "" if run_dir is None else str(run_dir),
        "train_segment_rmse": _metric_or_nan(train_segment_metrics, "rmse"),
        "train_segment_mae": _metric_or_nan(train_segment_metrics, "mae"),
        "train_segment_r2": _metric_or_nan(train_segment_metrics, "r2"),
        "train_segment_pearson": _metric_or_nan(train_segment_metrics, "pearson"),
        "train_audio_rmse": _metric_or_nan(train_audio_metrics, "rmse"),
        "train_audio_mae": _metric_or_nan(train_audio_metrics, "mae"),
        "train_audio_r2": _metric_or_nan(train_audio_metrics, "r2"),
        "train_audio_pearson": _metric_or_nan(train_audio_metrics, "pearson"),
        "test_segment_rmse": _metric_or_nan(test_segment_metrics, "rmse"),
        "test_segment_mae": _metric_or_nan(test_segment_metrics, "mae"),
        "test_segment_r2": _metric_or_nan(test_segment_metrics, "r2"),
        "test_segment_pearson": _metric_or_nan(test_segment_metrics, "pearson"),
        "test_audio_rmse": _metric_or_nan(test_audio_metrics, "rmse"),
        "test_audio_mae": _metric_or_nan(test_audio_metrics, "mae"),
        "test_audio_r2": _metric_or_nan(test_audio_metrics, "r2"),
        "test_audio_pearson": _metric_or_nan(test_audio_metrics, "pearson"),
    }

    single_csv_path = export_prefix + "_single_run_metrics.csv"
    single_df = append_result_to_csv(row, single_csv_path)
    print("single run CSV saved to:", single_csv_path)

    test_audio_pred_path = None
    if test_audio_result is not None:
        test_audio_pred_df = pd.DataFrame(
            {
                "audio_id": test_audio_result["sorted_audio_ids"],
                "true_mmse": test_audio_result["y_true_audio"],
                "pred_mmse": test_audio_result["y_pred_audio"],
            }
        )
        test_audio_pred_path = export_prefix + "_test_audio_predictions.csv"
        test_audio_pred_df.to_csv(test_audio_pred_path, index=False, encoding="utf-8-sig")
        print("test audio prediction CSV saved to:", test_audio_pred_path)

    context = {
        "dataset": DATASET_NAME,
        "model": model_name,
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "selected_pipeline": selected_pipeline,
        "run_dir": run_dir,
        "test_audio_predictions_path": test_audio_pred_path,
    }

    return {
        "single_row": row,
        "single_df": single_df,
        "single_csv_path": single_csv_path,
        "test_audio_pred_path": test_audio_pred_path,
        "context": context,
    }


notebook_log_section(f"{MODEL_NAME} Single model running (PyTorch)")


if not RUN_SINGLE_SEED_400:
    print(f"[SKIP] RUN_SINGLE_SEED_400=False,Skip {MODEL_NAME} seed=400 Single run.")
else:
    dataset = build_raw_dataset(DATASET_NAME, include_time=INCLUDE_TIME, T=T_MAX, seed=RANDOM_SEED)
    dataloaders = dataset.raw_dataloaders
    single_run_ds_dir = get_presplit_cache_dir(
        data_dir=DATA_DIR,
        dataset_name=DATASET_NAME,
        feature_model_tag=FEATURE_MODEL_TAG,
        feature_selection_tag=FEATURE_SELECTION_TAG,
        file_tag=FILE_TAG,
    )
    idx_train_pkl = os.path.join(single_run_ds_dir, "idx_train.pkl")
    idx_val_pkl = os.path.join(single_run_ds_dir, "idx_val.pkl")
    idx_test_pkl = os.path.join(single_run_ds_dir, "idx_test.pkl")
    print(f"[{MODEL_NAME}_single_run] presplit ds_dir =", single_run_ds_dir)
    audio_ids_train = np.asarray(pickle.load(open(idx_train_pkl, "rb"))).astype(int)
    audio_ids_val = np.asarray(pickle.load(open(idx_val_pkl, "rb"))).astype(int)
    audio_ids_test = np.asarray(pickle.load(open(idx_test_pkl, "rb"))).astype(int)
    steps_per_epoch = int(np.ceil(dataloaders["train"].size / BATCH_SIZE))
    if USE_FIXED_EPOCHS:
        num_steps_run = int(TARGET_EPOCHS * steps_per_epoch)
    else:
        num_steps_run = int(NUM_STEPS)
    effective_warmup_steps = int(max(1, WARMUP_EPOCHS * steps_per_epoch))
    total_epochs_run = int(np.ceil(num_steps_run / max(steps_per_epoch, 1)))
    print(
        f"{MODEL_NAME} train plan => steps_per_epoch={steps_per_epoch}, warmup_steps={effective_warmup_steps}, num_steps_run={num_steps_run}, total_epochs={total_epochs_run}, use_early_stopping={USE_EARLY_STOPPING}"
    )
    model_cfg = dict(DEFAULT_MODEL_CONFIG[MODEL_NAME])
    model, _ = create_mamba_model_torch(
        data_dim=dataset.data_dim, label_dim=dataset.label_dim, config=model_cfg, seed=RANDOM_SEED
    )
    train_result = train_mamba_model_torch(
        model_name=MODEL_NAME,
        model=model,
        dataloaders=dataloaders,
        lr=_get_active_learning_rate_for_model(MODEL_NAME),
        batch_size=BATCH_SIZE,
        num_steps=num_steps_run,
        print_steps=PRINT_STEPS,
        early_stopping_steps=EARLY_STOPPING_STEPS,
        warmup_steps=effective_warmup_steps,
        cosine_alpha=COSINE_ALPHA,
        seed=RANDOM_SEED,
        dataset_name=DATASET_NAME,
        epoch_selection_objective=EPOCH_SELECTION_OBJECTIVE,
        audio_ids_val=audio_ids_val,
        use_early_stopping=USE_EARLY_STOPPING,
    )
    device = train_result["device"]
    selected_epoch = int(train_result["selected_epoch"])
    epoch_decision_info = dict(train_result["epoch_decision_info"])
    _load_torch_state_dict_snapshot(model, train_result["selected_state_dict"], device)
    selected_pipeline = evaluate_checkpoint_regression_pipeline_torch(
        model=model,
        dataloaders=dataloaders,
        batch_size=BATCH_SIZE,
        device=device,
        audio_ids_train=audio_ids_train,
        audio_ids_val=audio_ids_val,
        audio_ids_test=audio_ids_test,
        run_test=True,
        audio_agg_method=AUDIO_AGG_PRIMARY_METHOD,
        audio_trim_ratio=AUDIO_AGG_TRIM_RATIO,
    )
    train_segment_metrics = selected_pipeline["train_segment_metrics"]
    test_segment_metrics = selected_pipeline["test_segment_metrics"]
    train_audio_metrics = selected_pipeline["train_audio_metrics"]["metrics"]
    test_audio_metrics = selected_pipeline["test_audio_metrics"]["metrics"]
    print(f"\n[{MODEL_NAME} Selected checkpoint metrics | no-leak]")
    print("selection_policy =", epoch_decision_info["selection_policy"])
    print("selected_epoch =", selected_epoch)
    print("selected_by =", epoch_decision_info["selected_by"])
    print(
        "test_monitor_used_for_selection =", epoch_decision_info["test_monitor_used_for_selection"]
    )
    print("early_stopping_monitor =", epoch_decision_info["early_stopping_monitor"])
    print("stopped_early =", epoch_decision_info["stopped_early"])
    print("stop_reason =", epoch_decision_info["stop_reason"])
    print("train_segment_metrics =", train_segment_metrics)
    print("test_segment_metrics  =", test_segment_metrics)
    print("train_audio_metrics   =", train_audio_metrics)
    print("test_audio_metrics    =", test_audio_metrics)
    print("\n[Plots] Test audio diagnostics")
    plot_regression_diagnostics(
        labels=selected_pipeline["test_audio_metrics"]["y_true_audio"],
        predictions=selected_pipeline["test_audio_metrics"]["y_pred_audio"],
        plot_title_prefix=f"{DATASET_NAME} | {MODEL_NAME} | Test Audio",
        metrics=test_audio_metrics,
        already_denormalized=True,
    )
    torch.save(
        train_result["selected_state_dict"],
        os.path.join(train_result["run_dir"], "selected_model.pt"),
    )
    save_info = save_single_run_outputs(
        model_name=MODEL_NAME,
        seed=RANDOM_SEED,
        selected_epoch=selected_epoch,
        selected_pipeline=selected_pipeline,
        run_dir=train_result["run_dir"],
    )
    LAST_SINGLE_RUN_CONTEXT = save_info["context"]
    display(save_info["single_df"].tail(5))
    print_audio_aggregation_comparison("Test", selected_pipeline["test_audio_metrics"])


#############################################################################################


notebook_log_section(f"{MODEL_NAME} multi-seed Run (PyTorch)")


if not RUN_MULTI_SEEDS:
    print(f"[SKIP] RUN_MULTI_SEEDS=False,Skip {MODEL_NAME} multi-seed.")
else:
    SEEDS_TO_RUN = list(MULTI_SEEDS)
    dataset_ms = build_raw_dataset(
        DATASET_NAME, include_time=INCLUDE_TIME, T=T_MAX, seed=RANDOM_SEED
    )
    dataloaders_ms = dataset_ms.raw_dataloaders
    steps_per_epoch_ms = int(np.ceil(dataloaders_ms["train"].size / BATCH_SIZE))
    if USE_FIXED_EPOCHS:
        num_steps_run_ms = int(TARGET_EPOCHS * steps_per_epoch_ms)
    else:
        num_steps_run_ms = int(NUM_STEPS)
    effective_warmup_steps_ms = int(max(1, WARMUP_EPOCHS * steps_per_epoch_ms))
    total_epochs_run_ms = int(np.ceil(num_steps_run_ms / max(steps_per_epoch_ms, 1)))
    print(
        f"{MODEL_NAME} multi-seed train plan => steps_per_epoch={steps_per_epoch_ms}, warmup_steps={effective_warmup_steps_ms}, num_steps_run={num_steps_run_ms}, total_epochs={total_epochs_run_ms}, use_early_stopping={USE_EARLY_STOPPING}"
    )
    multi_seed_ds_dir = get_presplit_cache_dir(
        data_dir=DATA_DIR,
        dataset_name=DATASET_NAME,
        feature_model_tag=FEATURE_MODEL_TAG,
        feature_selection_tag=FEATURE_SELECTION_TAG,
        file_tag=FILE_TAG,
    )
    idx_train_pkl_ms = os.path.join(multi_seed_ds_dir, "idx_train.pkl")
    idx_val_pkl_ms = os.path.join(multi_seed_ds_dir, "idx_val.pkl")
    idx_test_pkl_ms = os.path.join(multi_seed_ds_dir, "idx_test.pkl")
    print(f"[{MODEL_NAME}_multi_seed] presplit ds_dir =", multi_seed_ds_dir)
    audio_ids_train_ms = np.asarray(pickle.load(open(idx_train_pkl_ms, "rb"))).astype(int)
    audio_ids_val_ms = np.asarray(pickle.load(open(idx_val_pkl_ms, "rb"))).astype(int)
    audio_ids_test_ms = np.asarray(pickle.load(open(idx_test_pkl_ms, "rb"))).astype(int)
    seed_epoch_rows = []
    seed_best_rows = []
    multi_run_contexts = []
    model_cfg_ms = dict(DEFAULT_MODEL_CONFIG[MODEL_NAME])
    for seed_ms in SEEDS_TO_RUN:
        print(f"\n[MultiSeed-{MODEL_NAME}] start seed={seed_ms}")
        model_ms, _ = create_mamba_model_torch(
            data_dim=dataset_ms.data_dim,
            label_dim=dataset_ms.label_dim,
            config=model_cfg_ms,
            seed=seed_ms,
            model_name=MODEL_NAME,
        )
        train_result_ms = train_mamba_model_torch(
            model_name=MODEL_NAME,
            model=model_ms,
            dataloaders=dataloaders_ms,
            lr=_get_active_learning_rate_for_model(MODEL_NAME),
            batch_size=BATCH_SIZE,
            num_steps=num_steps_run_ms,
            print_steps=PRINT_STEPS,
            early_stopping_steps=EARLY_STOPPING_STEPS,
            warmup_steps=effective_warmup_steps_ms,
            cosine_alpha=COSINE_ALPHA,
            seed=seed_ms,
            dataset_name=DATASET_NAME,
            epoch_selection_objective=EPOCH_SELECTION_OBJECTIVE,
            audio_ids_val=audio_ids_val_ms,
            use_early_stopping=USE_EARLY_STOPPING,
        )
        device_ms = train_result_ms["device"]
        for em in train_result_ms["epoch_trace"]:
            seed_epoch_rows.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": DATASET_NAME,
                    "model": str(MODEL_NAME),
                    "seed": int(seed_ms),
                    "epoch": int(em["epoch"]),
                    "step": int(em["step"]),
                    "loss": float(em["loss"]),
                    "lr": float(em["lr"]),
                    "train_segment_rmse": float(em["train_segment_rmse"]),
                    "train_segment_mae": float(em["train_segment_mae"]),
                    "test_segment_rmse": float(em["test_segment_rmse"]),
                    "test_segment_mae": float(em["test_segment_mae"]),
                    "test_segment_r2": float(em["test_segment_r2"]),
                    "test_audio_rmse": float(em["test_audio_rmse"]),
                    "test_audio_mae": float(em["test_audio_mae"]),
                    "test_audio_r2": float(em["test_audio_r2"]),
                    "epoch_selection_objective": str(em["epoch_selection_objective"]),
                    "train_loss_epoch": float(em.get("train_loss_epoch", np.nan)),
                    "test_selection_score": float(em["test_selection_score"]),
                    "selection_policy": str(em.get("selection_policy", EPOCH_SELECTION_POLICY)),
                    "selected_by": str(
                        em.get("selected_by", "last_epoch_or_train_loss_early_stop")
                    ),
                    "test_monitor_used_for_selection": int(
                        em.get("test_monitor_used_for_selection", 0)
                    ),
                    "early_stopping_monitor": str(
                        em.get("early_stopping_monitor", EARLY_STOPPING_MONITOR)
                    ),
                    "val_segment_rmse": float(em["test_segment_rmse"]),
                    "val_segment_mae": float(em["test_segment_mae"]),
                    "val_segment_r2": float(em["test_segment_r2"]),
                    "val_audio_rmse": float(em["test_audio_rmse"]),
                    "val_audio_mae": float(em["test_audio_mae"]),
                    "val_audio_r2": float(em["test_audio_r2"]),
                    "val_selection_score": float(em["test_selection_score"]),
                    "is_best_epoch": int(em["is_best_epoch"]),
                    "run_dir": train_result_ms["run_dir"],
                }
            )
        selected_epoch_ms = int(train_result_ms["selected_epoch"])
        epoch_decision_info_ms = dict(train_result_ms["epoch_decision_info"])
        _load_torch_state_dict_snapshot(model_ms, train_result_ms["selected_state_dict"], device_ms)
        selected_pipeline_ms = evaluate_checkpoint_regression_pipeline_torch(
            model=model_ms,
            dataloaders=dataloaders_ms,
            batch_size=BATCH_SIZE,
            device=device_ms,
            audio_ids_train=audio_ids_train_ms,
            audio_ids_val=audio_ids_val_ms,
            audio_ids_test=audio_ids_test_ms,
            run_test=True,
            audio_agg_method=AUDIO_AGG_PRIMARY_METHOD,
            audio_trim_ratio=AUDIO_AGG_TRIM_RATIO,
        )
        torch.save(
            train_result_ms["selected_state_dict"],
            os.path.join(train_result_ms["run_dir"], "selected_model.pt"),
        )
        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": DATASET_NAME,
            "model": str(MODEL_NAME),
            "split_protocol": "official_train_as_train_official_test_as_test",
            "train_semantics": "official_train",
            "test_semantics": "official_test",
            "seed": int(seed_ms),
            "selected_epoch": int(selected_epoch_ms),
            "selected_by": str(epoch_decision_info_ms["selected_by"]),
            "selection_policy": str(epoch_decision_info_ms["selection_policy"]),
            "test_monitor_used_for_selection": int(
                epoch_decision_info_ms["test_monitor_used_for_selection"]
            ),
            "early_stopping_monitor": str(epoch_decision_info_ms["early_stopping_monitor"]),
            "early_stopping_used": int(epoch_decision_info_ms["early_stopping_used"]),
            "stopped_early": int(epoch_decision_info_ms["stopped_early"]),
            "stop_reason": str(epoch_decision_info_ms["stop_reason"]),
            "best_train_loss": float(epoch_decision_info_ms["best_train_loss"]),
            "test_segment_rmse": float(selected_pipeline_ms["test_segment_metrics"]["rmse"]),
            "test_segment_mae": float(selected_pipeline_ms["test_segment_metrics"]["mae"]),
            "test_segment_r2": float(selected_pipeline_ms["test_segment_metrics"]["r2"]),
            "test_segment_pearson": float(selected_pipeline_ms["test_segment_metrics"]["pearson"]),
            "test_audio_rmse": float(selected_pipeline_ms["test_audio_metrics"]["metrics"]["rmse"]),
            "test_audio_mae": float(selected_pipeline_ms["test_audio_metrics"]["metrics"]["mae"]),
            "test_audio_r2": float(selected_pipeline_ms["test_audio_metrics"]["metrics"]["r2"]),
            "test_audio_pearson": float(
                selected_pipeline_ms["test_audio_metrics"]["metrics"]["pearson"]
            ),
            "val_segment_rmse": float(selected_pipeline_ms["test_segment_metrics"]["rmse"]),
            "val_segment_mae": float(selected_pipeline_ms["test_segment_metrics"]["mae"]),
            "val_segment_r2": float(selected_pipeline_ms["test_segment_metrics"]["r2"]),
            "val_segment_pearson": float(selected_pipeline_ms["test_segment_metrics"]["pearson"]),
            "val_audio_rmse": float(selected_pipeline_ms["test_audio_metrics"]["metrics"]["rmse"]),
            "val_audio_mae": float(selected_pipeline_ms["test_audio_metrics"]["metrics"]["mae"]),
            "val_audio_r2": float(selected_pipeline_ms["test_audio_metrics"]["metrics"]["r2"]),
            "val_audio_pearson": float(
                selected_pipeline_ms["test_audio_metrics"]["metrics"]["pearson"]
            ),
        }
        seed_best_rows.append(row)
        multi_run_contexts.append(
            {
                "dataset": DATASET_NAME,
                "model": str(MODEL_NAME),
                "seed": int(seed_ms),
                "selected_epoch": int(selected_epoch_ms),
                "selection_policy": str(epoch_decision_info_ms["selection_policy"]),
                "selected_by": str(epoch_decision_info_ms["selected_by"]),
                "test_monitor_used_for_selection": int(
                    epoch_decision_info_ms["test_monitor_used_for_selection"]
                ),
                "selected_pipeline": selected_pipeline_ms,
                "audio_ids_test": audio_ids_test_ms,
                "run_dir": train_result_ms["run_dir"],
            }
        )
    seed_epoch_df = pd.DataFrame(seed_epoch_rows)
    seed_best_df = pd.DataFrame(seed_best_rows)
    export_prefix_ms = os.path.join(OUTPUT_EXPORT_PREFIX, f"{DATASET_NAME}_multiseed")
    os.makedirs(OUTPUT_EXPORT_PREFIX, exist_ok=True)
    epoch_csv_ms = export_prefix_ms + "_epoch_trace.csv"
    best_csv_ms = export_prefix_ms + "_best_metrics.csv"
    if len(seed_epoch_df) > 0:
        seed_epoch_df.to_csv(epoch_csv_ms, index=False, encoding="utf-8-sig")
        print(f"{MODEL_NAME} epoch trace CSV saved to:", epoch_csv_ms)
    if len(seed_best_df) > 0:
        numeric_cols = [
            "best_train_loss",
            "test_segment_rmse",
            "test_segment_mae",
            "test_segment_r2",
            "test_segment_pearson",
            "test_audio_rmse",
            "test_audio_mae",
            "test_audio_r2",
            "test_audio_pearson",
        ]
        summary_rows = []
        for col in numeric_cols:
            if col not in seed_best_df.columns:
                continue
            summary_rows.append(
                {
                    "metric": col,
                    "mean": float(seed_best_df[col].mean()),
                    "std": float(seed_best_df[col].std(ddof=1)) if len(seed_best_df) > 1 else 0.0,
                    "n": int(seed_best_df[col].notna().sum()),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        seed_best_df.to_csv(best_csv_ms, index=False, encoding="utf-8-sig")
        summary_csv_ms = export_prefix_ms + "_summary_test_focus.csv"
        summary_df.to_csv(summary_csv_ms, index=False, encoding="utf-8-sig")
        print(f"{MODEL_NAME} best metrics CSV saved to:", best_csv_ms)
        print(f"{MODEL_NAME} test-focused summary CSV saved to:", summary_csv_ms)
        display(seed_best_df)
        display(summary_df)
    LAST_MULTI_SEED_CONTEXTS = multi_run_contexts


# Inserts a subsection heading in the unification log to allow for the subsequent location of the output at the current stage.


notebook_log_section("Export audio-level Forecast and Summary Results")


# ===== 12. Export summary results: test / official_test =====


contexts = []


last_multi_seed_contexts = globals().get("LAST_MULTI_SEED_CONTEXTS", [])


last_single_run_context = globals().get("LAST_SINGLE_RUN_CONTEXT", None)


if isinstance(last_multi_seed_contexts, list) and len(last_multi_seed_contexts) > 0:
    contexts = last_multi_seed_contexts
    print(f"[PostExport] Use {MODEL_NAME} Multiple seed context, number ={len(contexts)}")
elif last_single_run_context is not None:
    contexts = [last_single_run_context]
    print(f"[PostExport] Use {MODEL_NAME} Context of Single seed")
else:
    print(f"[PostExport] No available context detected, run first {MODEL_NAME} Single seed or multi-seed processes.")


def _safe_get_audio_method_block(audio_metric_result, method_name):
    """
    Compatibility with old and new structures:
    - New structure: audio_metric_result["all_methods"][method_name]
    - Old structure: only current main aggregate results
    """
    if audio_metric_result is None:
        return None

    if isinstance(audio_metric_result, dict) and ("all_methods" in audio_metric_result):
        return audio_metric_result["all_methods"].get(method_name)

    # Compatibility with old structure: No all_methods, only return to the group primary
    primary_method = str(audio_metric_result.get("aggregation_method", "mean")).strip().lower()
    if method_name == primary_method:
        return {
            "sorted_audio_ids": np.asarray(audio_metric_result.get("sorted_audio_ids", [])),
            "y_true_audio": np.asarray(audio_metric_result.get("y_true_audio", [])),
            "y_pred_audio": np.asarray(audio_metric_result.get("y_pred_audio", [])),
            "metrics": dict(audio_metric_result.get("metrics", {})),
        }
    return None


def _empty_metric_dict():
    return {
        "rmse": np.nan,
        "mae": np.nan,
        "r2": np.nan,
        "pearson": np.nan,
    }


def _flatten_metric_block(prefix, metric_block):
    """
    A metrics display of method in a single-store dictionary, easy to write summary CSV.
    prefix For example:
      - "val_audio_mean"
      - "test_audio_trimmed_mean"
    """
    metrics = _empty_metric_dict()
    if metric_block is not None and isinstance(metric_block, dict):
        metrics_raw = metric_block.get("metrics", {})
        if isinstance(metrics_raw, dict):
            metrics.update(metrics_raw)

    return {
        f"{prefix}_rmse": float(metrics["rmse"]) if pd.notna(metrics["rmse"]) else np.nan,
        f"{prefix}_mae": float(metrics["mae"]) if pd.notna(metrics["mae"]) else np.nan,
        f"{prefix}_r2": float(metrics["r2"]) if pd.notna(metrics["r2"]) else np.nan,
        f"{prefix}_pearson": float(metrics["pearson"]) if pd.notna(metrics["pearson"]) else np.nan,
    }


def build_audio_prediction_df(contexts, split="test"):
    """
    Export audio-level forecast details.
    Each audio corresponds to three lines (mean/ median/trimmed_mean), which facilitates reprocessing comparisons.
    """
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"split Must be val or test, currently received: {split}")

    rows = []
    for ctx in contexts:
        ds_name = str(ctx["dataset"])
        md_name = str(ctx["model"])
        sd = int(ctx["seed"])
        selected_epoch = int(ctx["selected_epoch"])
        pipe = ctx["selected_pipeline"]

        audio_metric_result = pipe.get(f"{split}_audio_metrics")
        if audio_metric_result is None:
            continue

        primary_method = str(audio_metric_result.get("aggregation_method", "mean"))
        trim_ratio = float(
            audio_metric_result.get("trim_ratio", globals().get("AUDIO_AGG_TRIM_RATIO", 0.10))
        )

        for method_name in ["mean", "median", "trimmed_mean"]:
            block = _safe_get_audio_method_block(audio_metric_result, method_name)
            if block is None:
                continue

            sorted_audio_ids = np.asarray(block.get("sorted_audio_ids", []))
            y_true_audio = np.asarray(block.get("y_true_audio", []))
            y_pred_audio = np.asarray(block.get("y_pred_audio", []))

            for aid, y_true, y_pred in zip(sorted_audio_ids, y_true_audio, y_pred_audio):
                rows.append(
                    {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "dataset": ds_name,
                        "model": md_name,
                        "seed": sd,
                        "selected_epoch": selected_epoch,
                        "split": split,
                        "agg_method": method_name,
                        "is_primary_agg": int(method_name == primary_method),
                        "trim_ratio": float(trim_ratio),
                        "audio_id": int(aid),
                        "true_mmse": float(y_true),
                        "pred_mmse": float(y_pred),
                        "squared_error": float((float(y_pred) - float(y_true)) ** 2),
                        "abs_error": float(abs(float(y_pred) - float(y_true))),
                    }
                )

    return pd.DataFrame(rows)


def build_context_summary_df(contexts, include_test=False):
    """
    Scale the core indicators for each context (each seed / model) into one row.

    Default:
    - Only summary of results (historical column *)
    - include_test=True , and then the test column will be rolled out together
    """
    rows = []
    for ctx in contexts:
        pipe = ctx["selected_pipeline"]

        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": str(ctx["dataset"]),
            "model": str(ctx["model"]),
            "seed": int(ctx["seed"]),
            "selected_epoch": int(ctx["selected_epoch"]),
        }

        # ===== val segment-level =====
        if pipe.get("val_segment_metrics") is not None:
            row.update(
                {
                    "val_segment_rmse": float(pipe["val_segment_metrics"]["rmse"]),
                    "val_segment_mae": float(pipe["val_segment_metrics"]["mae"]),
                    "val_segment_r2": float(pipe["val_segment_metrics"]["r2"]),
                    "val_segment_pearson": float(pipe["val_segment_metrics"]["pearson"]),
                }
            )
        else:
            row.update(
                {
                    "val_segment_rmse": np.nan,
                    "val_segment_mae": np.nan,
                    "val_segment_r2": np.nan,
                    "val_segment_pearson": np.nan,
                }
            )

        # ===== val audio-level =====
        val_audio_metric_result = pipe.get("val_audio_metrics")
        if val_audio_metric_result is None:
            row.update(
                {
                    "val_audio_primary_method": np.nan,
                    "val_audio_trim_ratio": np.nan,
                    "val_audio_rmse_primary": np.nan,
                    "val_audio_mae_primary": np.nan,
                    "val_audio_r2_primary": np.nan,
                    "val_audio_pearson_primary": np.nan,
                }
            )
            for method_name in ["mean", "median", "trimmed_mean"]:
                row.update(_flatten_metric_block(f"val_audio_{method_name}", None))
        else:
            primary_method = str(val_audio_metric_result.get("aggregation_method", "mean"))
            trim_ratio = float(
                val_audio_metric_result.get(
                    "trim_ratio", globals().get("AUDIO_AGG_TRIM_RATIO", 0.10)
                )
            )
            primary_metrics = val_audio_metric_result.get("metrics", _empty_metric_dict())

            row.update(
                {
                    "val_audio_primary_method": primary_method,
                    "val_audio_trim_ratio": float(trim_ratio),
                    "val_audio_rmse_primary": float(primary_metrics.get("rmse", np.nan)),
                    "val_audio_mae_primary": float(primary_metrics.get("mae", np.nan)),
                    "val_audio_r2_primary": float(primary_metrics.get("r2", np.nan)),
                    "val_audio_pearson_primary": float(primary_metrics.get("pearson", np.nan)),
                }
            )

            for method_name in ["mean", "median", "trimmed_mean"]:
                block = _safe_get_audio_method_block(val_audio_metric_result, method_name)
                row.update(_flatten_metric_block(f"val_audio_{method_name}", block))

        # ===== test(== sync, corrected by elderman ==
        if include_test:
            if pipe.get("test_segment_metrics") is not None:
                row.update(
                    {
                        "test_segment_rmse": float(pipe["test_segment_metrics"]["rmse"]),
                        "test_segment_mae": float(pipe["test_segment_metrics"]["mae"]),
                        "test_segment_r2": float(pipe["test_segment_metrics"]["r2"]),
                        "test_segment_pearson": float(pipe["test_segment_metrics"]["pearson"]),
                    }
                )
            else:
                row.update(
                    {
                        "test_segment_rmse": np.nan,
                        "test_segment_mae": np.nan,
                        "test_segment_r2": np.nan,
                        "test_segment_pearson": np.nan,
                    }
                )

            test_audio_metric_result = pipe.get("test_audio_metrics")
            if test_audio_metric_result is None:
                row.update(
                    {
                        "test_audio_primary_method": np.nan,
                        "test_audio_trim_ratio": np.nan,
                        "test_audio_rmse_primary": np.nan,
                        "test_audio_mae_primary": np.nan,
                        "test_audio_r2_primary": np.nan,
                        "test_audio_pearson_primary": np.nan,
                    }
                )
                for method_name in ["mean", "median", "trimmed_mean"]:
                    row.update(_flatten_metric_block(f"test_audio_{method_name}", None))
            else:
                primary_method = str(test_audio_metric_result.get("aggregation_method", "mean"))
                trim_ratio = float(
                    test_audio_metric_result.get(
                        "trim_ratio", globals().get("AUDIO_AGG_TRIM_RATIO", 0.10)
                    )
                )
                primary_metrics = test_audio_metric_result.get("metrics", _empty_metric_dict())

                row.update(
                    {
                        "test_audio_primary_method": primary_method,
                        "test_audio_trim_ratio": float(trim_ratio),
                        "test_audio_rmse_primary": float(primary_metrics.get("rmse", np.nan)),
                        "test_audio_mae_primary": float(primary_metrics.get("mae", np.nan)),
                        "test_audio_r2_primary": float(primary_metrics.get("r2", np.nan)),
                        "test_audio_pearson_primary": float(primary_metrics.get("pearson", np.nan)),
                    }
                )

                for method_name in ["mean", "median", "trimmed_mean"]:
                    block = _safe_get_audio_method_block(test_audio_metric_result, method_name)
                    row.update(_flatten_metric_block(f"test_audio_{method_name}", block))

        rows.append(row)

    return pd.DataFrame(rows)


def build_seed_metric_df(contexts, split="test"):
    """
    Export audio-level indicators for each seed x each polymer method.
    Focused on rmse to facilitate follow-up to the 5-seed average RMSE summary.
    """
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"split Must be val or test, currently received: {split}")

    rows = []
    for ctx in contexts:
        ds_name = str(ctx["dataset"])
        md_name = str(ctx["model"])
        sd = int(ctx["seed"])
        selected_epoch = int(ctx["selected_epoch"])
        pipe = ctx["selected_pipeline"]

        audio_metric_result = pipe.get(f"{split}_audio_metrics")
        if audio_metric_result is None:
            continue

        primary_method = str(audio_metric_result.get("aggregation_method", "mean"))
        trim_ratio = float(
            audio_metric_result.get("trim_ratio", globals().get("AUDIO_AGG_TRIM_RATIO", 0.10))
        )

        for method_name in ["mean", "median", "trimmed_mean"]:
            block = _safe_get_audio_method_block(audio_metric_result, method_name)
            if block is None:
                continue

            metrics = _empty_metric_dict()
            metrics_raw = block.get("metrics", {})
            if isinstance(metrics_raw, dict):
                metrics.update(metrics_raw)

            rows.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": ds_name,
                    "model": md_name,
                    "seed": sd,
                    "selected_epoch": selected_epoch,
                    "split": split,
                    "agg_method": method_name,
                    "is_primary_agg": int(method_name == primary_method),
                    "trim_ratio": float(trim_ratio),
                    "rmse": float(metrics["rmse"]) if pd.notna(metrics["rmse"]) else np.nan,
                    "mae": float(metrics["mae"]) if pd.notna(metrics["mae"]) else np.nan,
                    "r2": float(metrics["r2"]) if pd.notna(metrics["r2"]) else np.nan,
                    "pearson": (
                        float(metrics["pearson"]) if pd.notna(metrics["pearson"]) else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_seed_mean_rmse_df(seed_metric_df):
    """
    By dataset / model / agg_method, do multi-seed average RMSE.
    If you're currently five random seeds, here n_seeds should show five.
    """
    if seed_metric_df is None or len(seed_metric_df) == 0:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model",
                "split",
                "agg_method",
                "n_seeds",
                "mean_rmse",
                "std_rmse",
                "min_rmse",
                "max_rmse",
            ]
        )

    out_df = (
        seed_metric_df.groupby(["dataset", "model", "split", "agg_method"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_rmse=("rmse", "mean"),
            std_rmse=("rmse", "std"),
            min_rmse=("rmse", "min"),
            max_rmse=("rmse", "max"),
            mean_mae=("mae", "mean"),
            mean_r2=("r2", "mean"),
            mean_pearson=("pearson", "mean"),
        )
        .sort_values(["dataset", "model", "split", "agg_method"])
        .reset_index(drop=True)
    )

    return out_df


if len(contexts) > 0:
    post_prefix = os.path.join(OUTPUT_EXPORT_PREFIX, f"{DATASET_NAME}_posthoc")
    os.makedirs(OUTPUT_EXPORT_PREFIX, exist_ok=True)

    # ===== Default lead: test / official_test =====
    test_audio_pred_df = build_audio_prediction_df(contexts, split="test")
    summary_df = build_context_summary_df(contexts, include_test=True)
    test_seed_metric_df = build_seed_metric_df(contexts, split="test")
    test_seed_mean_rmse_df = build_seed_mean_rmse_df(test_seed_metric_df)

    test_audio_pred_csv = post_prefix + "_test_audio_predictions_all_aggs.csv"
    summary_csv = post_prefix + "_summary_test_focus.csv"
    test_seed_metric_csv = post_prefix + "_test_seed_rmse_all_aggs.csv"
    test_seed_mean_rmse_csv = post_prefix + "_test_seed_rmse_mean_all_aggs.csv"

    test_audio_pred_df.to_csv(test_audio_pred_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    test_seed_metric_df.to_csv(test_seed_metric_csv, index=False, encoding="utf-8-sig")
    test_seed_mean_rmse_df.to_csv(test_seed_mean_rmse_csv, index=False, encoding="utf-8-sig")

    print("[PostExport] test audio prediction CSV saved to:", test_audio_pred_csv)
    print("[PostExport] summary CSV saved to:", summary_csv)
    print("[PostExport] test seed RMSE CSV saved to:", test_seed_metric_csv)
    print("[PostExport] test seed mean RMSE CSV saved to:", test_seed_mean_rmse_csv)

    print("\n[PostExport] summary_df")
    display(summary_df)

    print("\n[PostExport] test seed RMSE by aggregation method")
    display(
        test_seed_metric_df[
            ["dataset", "model", "seed", "agg_method", "rmse", "mae", "r2", "pearson"]
        ]
        .sort_values(["dataset", "model", "seed", "agg_method"])
        .reset_index(drop=True)
    )

    print("\n[PostExport] test aggregation overview (multi-seed mean RMSE)")
    display(
        test_seed_mean_rmse_df[
            [
                "dataset",
                "model",
                "split",
                "agg_method",
                "n_seeds",
                "mean_rmse",
                "std_rmse",
                "min_rmse",
                "max_rmse",
            ]
        ]
    )
