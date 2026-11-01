# -*- coding: utf-8 -*-
"""
CPFT_2class_stage3.py

Cluster Pseudo-Label Fine-Tuning (CPFT).
============================================================

AD-aware CPFT Stage3 Subject Training Script.

This script is aligned to the following documents:
    CPFT_2class_stage1.py
    CPFT_2class_stage2.py

Definitions of the second classification
----------
    0 = Health/Creation Group (cc)
    1 = Disease Group (cd)

Stage3 Joint training objectives
------------------
1. masked cluster prediction
   Use a cluster pseudo labels generated on the official train,
   Calculate cluster Cross Entropy only on Whisper token, which is validly labelled mask.

2. audio-level AD auxiliary classification
   Each 30 second chunk generates two categories logits;
   All original audio chunk logits press audio_id to equalize;
   Calculating Cross Entropy with the audio-level 0/1 label.

Total loss:
    total_loss = LAMBDA_CLUS * cluster_loss
               + LAMBDA_AD * ad_classification_loss

Data protocol
--------
- Stage1,Stage2,Stage3 Use official train only;
- official test Do not enter Stage3;
- Not implement validation;
- Not to implement early stopping;
- Fixed training STAGE3_MAX_EPOCHS;
- Only last.ckpt at the end of the actual training;
- (a) Do not read any continuous scoring metadata;
- Non-implementation of standardized regression targets, continuous value drums or missing scoring filters.

Enter
----
<CPFT_ROOT>/<DATASET>/<MODEL_ROOT_TAG>/<singleX_unfreeZ>/
    stage1_class/last.ckpt
    stage2_clusters/train/all-clus.json
    stage2_clusters/train/chunks_manifest.json

Output
----
<CPFT_ROOT>/<DATASET>/<MODEL_ROOT_TAG>/<singleX_unfreeZ>/
    stage3_CPFT_maskoffK/
        last.ckpt
        config.json
        train_manifest.json
        csv_logs/
    meta/
        stage3_class_meta.json

Run Example
--------
conda activate whisper_cpft_faiss
python CPFT_2class_stage3.py

Cluster Pseudo-Label Fine-Tuning (CPFT).

Note:
The location of the script file does not affect the middle result directory;
The result directory is fixed from CPFT_ROOT to:
    /home/user/Alzheimer_code_jax_class/finetune/ft_CPFT
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from transformers import WhisperConfig, WhisperFeatureExtractor, WhisperModel


# ============================================================
# 2) Parameters Settings: must be aligned to Stage1 / Stage2 / extract
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

# That's a layer.
LAYERS_TO_RUN = [12]
STAGE3_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN

# Relative to how many layers of encoder have been unfrozen continuously.
# single12 + unfree7 -> Unfreeze the 6-12 layer.
UNFREEZE_ENCODER_LAYERS_LIST = [7,8,9,10,11,12]
UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# mask_offset=1 Indicates mask to be inserted after the first trainingable level.
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]

# official_train only protocol. Write meta only, without adding a path layer.
RUN_PROTOCOL_TAG = "official_train_only"

# Stage1 / Stage2 / Stage3 / extract There must be unity.
MODEL_ROOT_TAG = "CPFT_lr5e-5_epo5"

# Stage1 The source directory is managed separately from the parameter suffix of the current Stage2/Stage3 directory.
STAGE1_SOURCE_PATH_PARAM_TAGS: List[str] = []
PATH_PARAM_TAGS: List[str] = []

# Whisper Local model.
WHISPER_MODEL_NAME = "/home/user/whisper_lru_3/whisper_small"

# Manually assign Stage1 checkpoint; None indicates automatic search for stage1_class/last.ckpt.
STAGE1_CKPT_OVERRIDE: Optional[str] = None

# Continue training from Stage3 checkpoint; None indicates initialization from Stage1.
RESUME_STAGE3_CKPT: Optional[str] = None

# Stage2 The words cluster heads must be consistent with the words KS in CPFT_2class_stage2.py.
EXPECTED_NUM_CLUSTERS = [32, 64]

# 30 Seconds, 30% overlap, must match Stage2 and final features.
SEGMENT_SEC = 30
OVERLAP_RATE = 0.3
CHUNK_KEY_VERSION = "wav_path_chunk_start_end_v1"

# masked cluster prediction.
MASK_PROB = 0.06
MASK_SPAN = 10
LAMBDA_CLUS = 1.0

# audio-level AD Subsidiarity classification.
LAMBDA_AD = 0.05
STAGE3_AUX_CLASSIFICATION_LEVEL = "segment_logit_mean"
NUM_CLASSES = 2
CLASS_NAMES = {
    0: "healthy_control",
    1: "ad_or_disease",
}

# BATCH_SIZE The synonym is "a collection of original audio."
# All chinks inside each audio are spread in Collate.
STAGE3_AUDIO_BATCH_SIZE = 2
NUM_WORKERS = 4

# Model structure.
HEAD_DROPOUT = 0.2
L2_NORMALIZE_REPS = False

# CPFT Stage3 The original optimization agreement.
LR_ENC = 5e-5
LR_HEAD = 7e-5
WEIGHT_DECAY = 0.02
PRECISION = 16
GRAD_CLIP = 1.0
ACCUMULATE_GRAD_BATCHES = 1

# no-val-last:
STAGE3_MAX_EPOCHS = 5
STAGE3_CHECKPOINT_SELECTION = "last"

# TRAINING_STEP It is updated at the true DataLoader step in each experiment.

# Other.
SEED = 42
SAVE_MANIFESTS = True
ENABLE_CSV_LOGGER = True


# ============================================================
# 3) Constant
# ============================================================
os.environ.pop("PL_TRAINER_AMP_BACKEND", None)

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
TOK_T_FULL = (MAX_MEL_FRAMES + 1) // 2
IGNORE_INDEX = -100
SEGMENT_LENGTH = int(float(SEGMENT_SEC) * SAMPLE_RATE)


# ============================================================
# 4) Common tools
# ============================================================
def seed_everything_for_repro(seed: int) -> None:
    """Unified random torrents."""
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
    """Creates a directory."""
    os.makedirs(path, exist_ok=True)


def natural_key(value: str):
    """The natural sorting of file name and audio ID."""
    name = Path(str(value)).name
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.findall(r"\d+|\D+", name)
    ]


def canonical_wav_path(path: str) -> str:
    """Unified wav path to absolute path."""
    return str(Path(str(path)).expanduser().resolve())


def save_json(obj: Any, path: str) -> None:
    """Save JSON."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, indent=2)


def save_text(text: str, path: str) -> None:
    """Saves the text."""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as file:
        file.write(str(text))


def compute_valid_token_length(wav_len_samples: int) -> int:
    """
    Maps the number of valid samples for 16 kHz waveform into the number of Whisper encoder token.
    """
    wav_len_samples = int(wav_len_samples)
    if wav_len_samples <= 0:
        return 1

    mel_frames = int(
        math.ceil(wav_len_samples / float(HOP_LENGTH))
    )
    mel_frames = min(
        MAX_MEL_FRAMES,
        max(1, mel_frames),
    )
    token_length = (mel_frames + 1) // 2
    return max(1, int(token_length))


def _clean_path_tag(value: Any) -> str:
    """Clears the path parameter labels."""
    if value is None:
        return ""

    text = str(value).strip()
    if text in {"", "None", "none", "null", "Null"}:
        return ""

    text = text.replace("=", "").replace(" ", "")
    text = re.sub(r"[^0-9A-Za-z_.-]+", "", text)
    return text.strip("_-.")


def build_path_param_suffix(
    path_param_tags: Optional[Sequence[Any]] = None,
) -> str:
    """Spells the list of path parameters into a folder."""
    if path_param_tags is None:
        path_param_tags = PATH_PARAM_TAGS

    if isinstance(path_param_tags, (str, int, float)):
        values = [path_param_tags]
    else:
        values = list(path_param_tags)

    tags = [_clean_path_tag(value) for value in values]
    tags = [tag for tag in tags if tag != ""]
    return "_".join(tags)


def build_layer_name(
    single_layer: int,
    unfreeze_layer_count: int,
    path_param_tags: Optional[Sequence[Any]] = None,
) -> str:
    """
    Construct a unified selection name.

    For example:
        single12_unfree7
        single12_unfree7_ks64_128
    """
    base = (
        f"single{int(single_layer)}_"
        f"unfree{int(unfreeze_layer_count)}"
    )
    suffix = build_path_param_suffix(path_param_tags)
    return base if suffix == "" else f"{base}_{suffix}"


def get_whisper_encoder_layer_count(
    whisper_model_name_or_path: str,
) -> int:
    """Read Whisper encoder total layer."""
    config = WhisperConfig.from_pretrained(
        whisper_model_name_or_path
    )
    return int(config.encoder_layers)


def resolve_unfreeze_layers_for_single_layer(
    single_layer: int,
    total_layers: int,
    unfreeze_layer_count: int,
) -> List[int]:
    """
    Parsing the actual unfreezing layer.

    single12 + unfree7 -> [6, 7, 8, 9, 10, 11, 12]
    """
    single_layer = int(single_layer)
    total_layers = int(total_layers)
    unfreeze_layer_count = int(unfreeze_layer_count)

    if not 1 <= single_layer <= total_layers:
        raise ValueError(
            f"single_layer={single_layer} Beyond range [1,{total_layers}]"
        )
    if unfreeze_layer_count <= 0:
        raise ValueError(
            "UNFREEZE_ENCODER_LAYERS It must be greater than or equal to 1."
        )

    start_layer = max(
        1,
        single_layer - unfreeze_layer_count + 1,
    )
    end_layer = min(single_layer, total_layers)
    return list(range(start_layer, end_layer + 1))


def resolve_mask_inject_after_layer(
    resolved_unfreeze_layers: Sequence[int],
    mask_offset: int,
) -> int:
    """
    mask injection position is calculated according to the first trainingable layer.

    For example:
        Trainable level [6,...,12], mask_offset=1
        -> Inject after 5 layer.
    """
    if len(resolved_unfreeze_layers) == 0:
        raise ValueError(
            "resolved_unfreeze_layers Can't be empty."
        )

    mask_offset = int(mask_offset)
    if mask_offset <= 0:
        raise ValueError(
            "mask_offset It must be greater than or equal to 1."
        )

    first_trainable = min(
        int(layer) for layer in resolved_unfreeze_layers
    )
    return max(0, first_trainable - mask_offset)


def build_stage3_subdir(mask_offset: int) -> str:
    """Construct subdirectorate name Stage3 checkpoint."""
    return f"stage3_CPFT_maskoff{int(mask_offset)}"


def canonical_label_key(key: str) -> str:
    """
    chunk key for Stage2/Stage3.

    chunk key Format:
        <absolute_wav_path>::chunkXXXX::start...::end...
    """
    key = str(key)
    if "::chunk" in key:
        wav_part, suffix = key.split("::chunk", 1)
        return (
            canonical_wav_path(wav_part)
            + "::chunk"
            + suffix
        )
    return canonical_wav_path(key)


def load_audio_16k(wav_path: str) -> np.ndarray:
    """
    Reads any original sample rate audio and integrates it to 16 kHz, mono-channel, float32.
    """
    wav_path = canonical_wav_path(wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(
            f"[Stage3-CLASS] Could not close temporary folder: %s{wav_path}"
        )

    try:
        waveform, loaded_sr = librosa.load(
            wav_path,
            sr=SAMPLE_RATE,
            mono=True,
            dtype=np.float32,
        )
    except Exception as exc:
        raise RuntimeError(
            "Stage3 Audio reading or resampling failed: \n"
            f"path={wav_path}\n"
            f"target_sr={SAMPLE_RATE}\n"
            f"error={exc}"
        ) from exc

    if int(loaded_sr) != SAMPLE_RATE:
        raise RuntimeError(
            "Stage3 The sampling rate after re-sampling was abnormal:"
            f"path={wav_path}, actual_sr={loaded_sr}"
        )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    if waveform.size == 0:
        raise ValueError(
            f"Stage3 Read empty audio:{wav_path}"
        )
    if not np.all(np.isfinite(waveform)):
        raise ValueError(
            f"Stage3 Audio contains NaN/Inf:{wav_path}"
        )

    return waveform


def load_chunk_with_padding_from_wav(
    waveform: np.ndarray,
    start: int,
    end: int,
    chunk_id: int,
) -> Tuple[np.ndarray, int]:
    """
    Read a 30second chunk from waveform, which has been harmonized to 16 kHz.

    End paragraph completed logically aligned to CPFT_2class_stage2.py.
    """
    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    ).reshape(-1)

    start = int(start)
    end = int(end)
    chunk_id = int(chunk_id)

    segment = waveform[start:end]

    if len(segment) < SEGMENT_LENGTH and chunk_id > 0:
        padding_needed = SEGMENT_LENGTH - len(segment)
        step_size = int(
            SEGMENT_LENGTH * (1.0 - float(OVERLAP_RATE))
        )
        previous_end = start + SEGMENT_LENGTH - step_size
        padding_start = max(
            0,
            previous_end - padding_needed,
        )
        padding_values = waveform[
            padding_start:previous_end
        ]
        segment = np.concatenate(
            (segment, padding_values),
            axis=0,
        )

    if len(segment) > SEGMENT_LENGTH:
        segment = segment[:SEGMENT_LENGTH]

    valid_len = int(min(len(segment), SEGMENT_LENGTH))

    if len(segment) < SEGMENT_LENGTH:
        padding = np.zeros(
            (SEGMENT_LENGTH - len(segment),),
            dtype=np.float32,
        )
        segment = np.concatenate(
            [segment.astype(np.float32, copy=False), padding],
            axis=0,
        )

    return (
        segment.astype(np.float32, copy=False),
        valid_len,
    )


def pad_or_truncate_labels(
    label_tensor: torch.Tensor,
    target_t: int = TOK_T_FULL,
) -> torch.Tensor:
    """
    Replace the words Stage2 cluster label with the words [H, target_t].
    """
    if label_tensor.dim() == 1:
        label_tensor = label_tensor.unsqueeze(0)

    if label_tensor.dim() != 2:
        raise ValueError(
            "cluster label It must be [H,T] or [T],"
            f"Current ={tuple(label_tensor.shape)}"
        )

    head_count, time_length = label_tensor.shape

    if time_length == target_t:
        return label_tensor.long()
    if time_length > target_t:
        return label_tensor[:, :target_t].long()

    padding = torch.full(
        (head_count, target_t - time_length),
        int(IGNORE_INDEX),
        dtype=torch.long,
    )
    return torch.cat(
        [label_tensor.long(), padding],
        dim=1,
    )


def make_time_mask_tok_valid(
    valid_mask_tok: torch.Tensor,
    mask_prob: float,
    mask_span: int,
) -> torch.Tensor:
    """
    Randomly generated within valid token span mask.
    """
    if valid_mask_tok.dim() != 2:
        raise ValueError(
            "valid_mask_tok Must be [B,T]."
        )

    batch_size, time_length = valid_mask_tok.shape
    device = valid_mask_tok.device

    output_mask = torch.zeros(
        batch_size,
        time_length,
        dtype=torch.bool,
        device=device,
    )

    mask_prob = float(mask_prob)
    mask_span = int(mask_span)

    for batch_index in range(batch_size):
        valid_length = int(
            valid_mask_tok[batch_index].long().sum().item()
        )
        valid_length = max(
            1,
            min(valid_length, time_length),
        )

        if valid_length <= 1:
            continue

        target_mask_count = int(
            math.ceil(valid_length * mask_prob)
        )
        if target_mask_count <= 0:
            continue

        masked_count = 0
        attempts = 0
        span_low = max(1, mask_span // 2)
        span_high = max(
            span_low + 1,
            int(mask_span * 1.5),
        )

        while (
            masked_count < target_mask_count
            and attempts < 10 * max(1, target_mask_count)
        ):
            current_span = int(
                torch.randint(
                    span_low,
                    span_high,
                    (1,),
                    device=device,
                ).item()
            )
            current_span = min(
                current_span,
                valid_length,
            )
            start_max = max(
                1,
                valid_length - current_span + 1,
            )
            start = int(
                torch.randint(
                    0,
                    start_max,
                    (1,),
                    device=device,
                ).item()
            )
            end = min(
                valid_length,
                start + current_span,
            )

            if output_mask[batch_index, start:end].any():
                attempts += 1
                continue

            output_mask[batch_index, start:end] = True
            masked_count += end - start

    return output_mask & valid_mask_tok


def torch_load_compatible(
    path: str,
    map_location: str = "cpu",
):
    """
    weights_only Default act compatible with PyTorch 2.6.

    The current checkpoint is generated by the project and the source is credible, and therefore is visible.
    weights_only=False;The old version PyTorch returns automatically when this parameter is not supported.
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


# ============================================================
# 5) Path construction with Stage1 checkpoint
# ============================================================
def build_stage3_run_config(
    dataset_option: str,
    repr_layer: int,
    mask_offset: int,
) -> Dict[str, Any]:
    """
    Construct all paths of the current Stage3 classification.

    """
    repr_layer = int(repr_layer)
    mask_offset = int(mask_offset)

    total_layers = get_whisper_encoder_layer_count(
        WHISPER_MODEL_NAME
    )
    resolved_unfreeze_layers = (
        resolve_unfreeze_layers_for_single_layer(
            single_layer=repr_layer,
            total_layers=total_layers,
            unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        )
    )
    mask_inject_after_layer = (
        resolve_mask_inject_after_layer(
            resolved_unfreeze_layers,
            mask_offset,
        )
    )

    layer_name = build_layer_name(
        single_layer=repr_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )
    stage1_layer_name = build_layer_name(
        single_layer=repr_layer,
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

    stage3_subdir = build_stage3_subdir(
        mask_offset
    )
    stage3_save_dir = os.path.join(
        run_root,
        stage3_subdir,
    )
    meta_save_dir = os.path.join(
        run_root,
        "meta",
    )

    return {
        "task": "ad_binary_classification",
        "stage": "stage3_ad_aware_cpft",
        "class_mapping": {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
        },
        "CLASS_PROJECT_ROOT": CLASS_PROJECT_ROOT,
        "PROJECT_ROOT": PROJECT_ROOT,
        "CPFT_ROOT": CPFT_ROOT,
        "DATASET_OPTION": str(dataset_option),
        "REPR_LAYER": repr_layer,
        "UNFREEZE_ENCODER_LAYERS": int(
            UNFREEZE_ENCODER_LAYERS
        ),
        "RESOLVED_UNFREEZE_ENCODER_LAYERS": list(
            resolved_unfreeze_layers
        ),
        "STAGE3_TRAIN_LAYER_START": int(
            min(resolved_unfreeze_layers)
        ),
        "STAGE3_TRAIN_LAYER_END": int(
            max(resolved_unfreeze_layers)
        ),
        "TOTAL_ENCODER_LAYERS": int(total_layers),
        "MASK_OFFSET": mask_offset,
        "MASK_INJECT_AFTER_LAYER": int(
            mask_inject_after_layer
        ),
        "RUN_PROTOCOL_TAG": str(RUN_PROTOCOL_TAG),
        "official_train_used": True,
        "official_test_used": False,
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
        "STAGE2_SAVE_DIR": stage2_save_dir,
        "STAGE2_TRAIN_DIR": stage2_train_dir,
        "STAGE2_TRAIN_LABELPATH": os.path.join(
            stage2_train_dir,
            "all-clus.json",
        ),
        "STAGE2_TRAIN_MANIFEST": os.path.join(
            stage2_train_dir,
            "chunks_manifest.json",
        ),
        "STAGE3_SAVE_DIR": stage3_save_dir,
        "STAGE3_SUBDIR": stage3_subdir,
        "META_SAVE_DIR": meta_save_dir,
    }


def find_stage1_checkpoint(
    stage1_save_dir: str,
    override_ckpt: Optional[str] = None,
) -> str:
    """
    Read stage1_class/last.ckpt on a priority basis.
    """
    if override_ckpt not in [None, "", "None"]:
        checkpoint_path = str(override_ckpt)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                "STAGE1_CKPT_OVERRIDE Cannot initialise Evolution's mail component."
                f"{checkpoint_path}"
            )
        return str(Path(checkpoint_path).resolve())

    stage1_dir = Path(stage1_save_dir)
    if not stage1_dir.is_dir():
        raise FileNotFoundError(
            "Stage1 class Directory does not exist:"
            f"{stage1_save_dir}"
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
    Only the WhisperModel parameters in Stage1 classification checkpoint are loaded.

    Stage1 The utterance head, the frame head and the attendtive pooling will be ignored.
    """
    checkpoint = torch_load_compatible(
        checkpoint_path,
        map_location="cpu",
    )
    state = checkpoint.get("state_dict", checkpoint)

    if not isinstance(state, dict):
        raise ValueError(
            "Stage1 checkpoint state_dict Not dict:"
            f"{checkpoint_path}"
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

        for raw_key, value in state.items():
            raw_key = str(raw_key)
            if prefix != "" and not raw_key.startswith(prefix):
                continue

            new_key = (
                raw_key[len(prefix):]
                if prefix != ""
                else raw_key
            )

            if (
                new_key in target_state
                and hasattr(value, "shape")
                and tuple(target_state[new_key].shape)
                == tuple(value.shape)
            ):
                loaded[new_key] = value

        if len(loaded) > len(best_loaded):
            best_loaded = loaded
            best_prefix = prefix

    if len(best_loaded) == 0:
        raise RuntimeError(
            "Not from Stage1 classification checkpoint "
            "Match to Whisper Arguments: \n"
            f"{checkpoint_path}"
        )

    missing, unexpected = whisper_model.load_state_dict(
        best_loaded,
        strict=False,
    )

    encoder_loaded = sum(
        key.startswith("encoder.")
        for key in best_loaded
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
        f"missing={len(missing)} | "
        f"unexpected={len(unexpected)}"
    )


# ============================================================
# 6) Stage2 cluster label With manifest
# ============================================================
def load_all_clus_json(
    cluster_json_path: str,
    expected_k_list: Optional[Sequence[int]] = None,
) -> Tuple[Dict[str, torch.Tensor], List[int]]:
    """
    Read all-clus.json generated by CPFT_2class_stage2.py.

    Return:
        cluster_label_dict:
            chunk_key -> LongTensor[H,T]

        k_list:
            The king of each cluster head.
    """
    if not os.path.isfile(cluster_json_path):
        raise FileNotFoundError(
            f"all-clus.json Cannot initialise Evolution's mail component.{cluster_json_path}"
        )

    with open(
        cluster_json_path,
        "r",
        encoding="utf-8",
    ) as file:
        raw = json.load(file)

    if not isinstance(raw, dict):
        raise ValueError(
            "all-clus.json The top level must be dict."
        )

    valid_keys = [
        key for key in raw
        if not str(key).startswith("__")
    ]
    if len(valid_keys) == 0:
        raise ValueError(
            f"{cluster_json_path} No valid chunk key."
        )

    cluster_label_dict: Dict[str, torch.Tensor] = {}
    maximum_ids: Optional[List[int]] = None

    for raw_key, value in raw.items():
        if str(raw_key).startswith("__"):
            continue

        chunk_key = canonical_label_key(raw_key)

        if (
            isinstance(value, list)
            and len(value) > 0
            and isinstance(value[0], list)
        ):
            sequences = value
        elif isinstance(value, list):
            sequences = [value]
        else:
            raise ValueError(
                f"{raw_key} The cluster label is not the list."
            )

        if maximum_ids is None:
            maximum_ids = [0 for _ in sequences]

        if len(sequences) != len(maximum_ids):
            raise ValueError(
                f"{raw_key} and the number of cluster head is inconsistent."
            )

        tensors: List[torch.Tensor] = []
        for head_index, sequence in enumerate(sequences):
            array = torch.tensor(
                sequence,
                dtype=torch.long,
            )
            array = pad_or_truncate_labels(
                array,
                target_t=TOK_T_FULL,
            ).squeeze(0)

            valid_values = array[
                array != IGNORE_INDEX
            ]
            if valid_values.numel() > 0:
                maximum_ids[head_index] = max(
                    maximum_ids[head_index],
                    int(valid_values.max().item()),
                )

            tensors.append(array)

        if chunk_key in cluster_label_dict:
            raise ValueError(
                f"all-clus.json Repeating:{chunk_key}"
            )

        cluster_label_dict[chunk_key] = torch.stack(
            tensors,
            dim=0,
        )

    inferred_k_list = [
        int(value) + 1
        for value in (maximum_ids or [])
    ]

    if expected_k_list is not None:
        expected_k_list = [
            int(value) for value in expected_k_list
        ]
        if len(expected_k_list) != len(inferred_k_list):
            raise ValueError(
                "EXPECTED_NUM_CLUSTERS Inconsistent with all-clus head:"
                f"expected={expected_k_list}, "
                f"inferred={inferred_k_list}"
            )
        k_list = expected_k_list
    else:
        k_list = inferred_k_list

    for chunk_key, labels in cluster_label_dict.items():
        for head_index in range(labels.size(0)):
            valid_values = labels[head_index][
                labels[head_index] != IGNORE_INDEX
            ]
            if (
                valid_values.numel() > 0
                and int(valid_values.max().item())
                >= int(k_list[head_index])
            ):
                raise ValueError(
                    "cluster label Cross-border:"
                    f"key={chunk_key}, "
                    f"head={head_index}, "
                    f"max={int(valid_values.max().item())}, "
                    f"K={k_list[head_index]}"
                )

    return cluster_label_dict, k_list


def check_stage3_cluster_config(
    loaded_k_list: Sequence[int],
    expected_num_clusters: Sequence[int],
    train_clus_path: str,
) -> None:
    """Checks if Stage2 matches the K/head configuration of Stage3."""
    loaded = [int(value) for value in loaded_k_list]
    expected = [
        int(value) for value in expected_num_clusters
    ]

    if loaded != expected:
        raise RuntimeError(
            "Stage3 cluster Cannot initialise Evolution's mail component."
            f"loaded={loaded}, expected={expected}, "
            f"path={train_clus_path}"
        )


def _is_process2_ctd_chunk_entry(
    row: Dict[str, Any],
    dataset_key: Optional[str] = None,
) -> bool:
    """PROCESS-2 Only __CTD audio remains."""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    wav_path = row.get(
        "wav_path",
        row.get("wav_name", ""),
    )
    return (
        re.search(
            r"__CTD$",
            Path(str(wav_path)).stem,
            flags=re.IGNORECASE,
        )
        is not None
    )


def load_stage2_chunks_manifest(
    manifest_path: str,
    dataset_option: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Read CPFT_2class_stage2.py export official train chunks_manifest.json.

    Stage3 Direct consumption of chunk border and chunk_key of Stage2
    Do not regenerate slices to avoid Stage2/Stage3 boundary inconsistencies.
    """
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Stage2 chunks_manifest Cannot initialise Evolution's mail component.{manifest_path}"
        )

    with open(
        manifest_path,
        "r",
        encoding="utf-8",
    ) as file:
        obj = json.load(file)

    if isinstance(obj, dict) and isinstance(
        obj.get("chunks"),
        list,
    ):
        raw_chunks = obj["chunks"]
    elif isinstance(obj, list):
        raw_chunks = obj
    else:
        raise ValueError(
            f"Could not close temporary folder: %s{manifest_path}"
        )

    entries: List[Dict[str, Any]] = []
    dropped_non_ctd = 0
    chunk_keys = set()

    audio_to_label: Dict[str, int] = {}
    audio_to_path: Dict[str, str] = {}

    for raw_row in raw_chunks:
        if not isinstance(raw_row, dict):
            continue

        row = copy.deepcopy(raw_row)

        required_fields = [
            "wav_path",
            "chunk_key",
            "chunk_id",
            "chunk_start",
            "chunk_end",
            "target_label",
        ]
        missing_fields = [
            field for field in required_fields
            if row.get(field, None) in [None, ""]
        ]
        if len(missing_fields) > 0:
            raise ValueError(
                "Stage2 chunk entry Synchronising folder"
                f"{missing_fields},entry={row}"
            )

        row["wav_path"] = canonical_wav_path(
            row["wav_path"]
        )
        if not os.path.isfile(row["wav_path"]):
            raise FileNotFoundError(
                f"Stage3 Audio does not exist:{row['wav_path']}"
            )

        row["wav_name"] = str(
            row.get(
                "wav_name",
                Path(row["wav_path"]).name,
            )
        )
        row["chunk_key"] = canonical_label_key(
            row["chunk_key"]
        )
        row["chunk_id"] = int(row["chunk_id"])
        row["chunk_start"] = int(row["chunk_start"])
        row["chunk_end"] = int(row["chunk_end"])
        row["chunk_valid_len"] = int(
            row.get("chunk_valid_len", SEGMENT_LENGTH)
        )
        row["target_label"] = int(
            row["target_label"]
        )

        if row["target_label"] not in {0, 1}:
            raise ValueError(
                "Stage3 target_label Not 0/1:"
                f"{row['target_label']}"
            )

        audio_id_raw = row.get(
            "official_audio_id",
            row.get("audio_id", None),
        )
        if audio_id_raw in [None, ""]:
            raise ValueError(
                f"Stage2 chunk entry Missing audio_id:{row}"
            )

        row["official_audio_id"] = int(audio_id_raw)
        row["audio_id"] = int(audio_id_raw)
        row["split"] = "train"

        if not _is_process2_ctd_chunk_entry(
            row,
            dataset_key=dataset_option,
        ):
            dropped_non_ctd += 1
            continue

        if row["chunk_key"] in chunk_keys:
            raise ValueError(
                "Stage2 manifest chunk_key Repeat:"
                f"{row['chunk_key']}"
            )
        chunk_keys.add(row["chunk_key"])

        audio_key = str(row["audio_id"])
        label = int(row["target_label"])

        if audio_key in audio_to_label:
            if audio_to_label[audio_key] != label:
                raise ValueError(
                    "The same audio_id chunks have a different label:"
                    f"audio_id={audio_key}"
                )
            if audio_to_path[audio_key] != row["wav_path"]:
                raise ValueError(
                    "The same audio_id corresponds to several wav_path:"
                    f"audio_id={audio_key}"
                )
        else:
            audio_to_label[audio_key] = label
            audio_to_path[audio_key] = row["wav_path"]

        entries.append(row)

    if len(entries) == 0:
        raise ValueError(
            "Stage2 chunks_manifest Read empty."
        )

    unique_labels = sorted(set(audio_to_label.values()))
    if unique_labels != [0, 1]:
        raise ValueError(
            "Stage3 official train It is important to include both categories."
            f"Current ={unique_labels}"
        )

    entries = sorted(
        entries,
        key=lambda row: (
            int(row["audio_id"]),
            int(row["chunk_id"]),
            int(row["chunk_start"]),
        ),
    )

    audio_labels = np.asarray(
        list(audio_to_label.values()),
        dtype=np.int64,
    )
    chunk_labels = np.asarray(
        [int(row["target_label"]) for row in entries],
        dtype=np.int64,
    )

    stats = {
        "manifest_path": manifest_path,
        "raw_chunks": int(len(raw_chunks)),
        "kept_chunks": int(len(entries)),
        "dropped_non_ctd": int(dropped_non_ctd),
        "audio_total": int(len(audio_to_label)),
        "audio_class0": int(np.sum(audio_labels == 0)),
        "audio_class1": int(np.sum(audio_labels == 1)),
        "chunk_class0": int(np.sum(chunk_labels == 0)),
        "chunk_class1": int(np.sum(chunk_labels == 1)),
    }

    print(
        "[Stage3-CLASS][MANIFEST] "
        f"chunks={stats['kept_chunks']} | "
        f"audio={stats['audio_total']} | "
        f"audio_class0={stats['audio_class0']} | "
        f"audio_class1={stats['audio_class1']}"
    )

    return entries, stats


def filter_entries_by_cluster_labels(
    entries: List[Dict[str, Any]],
    cluster_label_dict: Dict[str, torch.Tensor],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Check whether each Stage2 chunk contains cluster pseudo label.
    """
    valid_entries: List[Dict[str, Any]] = []
    missing_entries: List[Dict[str, Any]] = []

    for raw_row in entries:
        row = copy.deepcopy(raw_row)
        chunk_key = canonical_label_key(
            row["chunk_key"]
        )
        row["chunk_key"] = chunk_key

        if chunk_key in cluster_label_dict:
            valid_entries.append(row)
        else:
            row["missing_reason"] = (
                "missing_stage2_cluster_label"
            )
            missing_entries.append(row)

    return valid_entries, missing_entries


# ============================================================
# 7) Stage3 audio-level Datasets
# ============================================================
def get_stage3_audio_id_from_entry(
    row: Dict[str, Any],
) -> str:
    """Parsing audio_id for the same audio chunks polymer."""
    for field in [
        "official_audio_id",
        "audio_id",
        "utt_id",
        "recording_id",
    ]:
        value = row.get(field, None)
        if value not in [None, ""]:
            return str(value)

    wav_path = row.get(
        "wav_path",
        row.get("wav_name", ""),
    )
    if wav_path not in [None, ""]:
        return canonical_wav_path(wav_path)

    raise ValueError(
        f"Could not parse audio_id from entry:{row}"
    )


class Stage3ADPAPTDataset(data.Dataset):
    """
    A item corresponds to the entire original audio 30second chunks.

    collate After:
        flat_wav_30:    [N_chunks, 480000]
        flat_valid_len: [N_chunks]
        flat_labels:    [N_chunks, H, T]
        audio_labels:   [B_audio]
        chunk_to_audio: [N_chunks]
        chunk_counts:   [B_audio]
    """

    def __init__(
        self,
        entries: List[Dict[str, Any]],
        cluster_label_dict: Dict[str, torch.Tensor],
    ):
        self.cluster_label_dict = cluster_label_dict

        if len(entries) == 0:
            raise ValueError(
                "Stage3ADPAPTDataset Roger that, entries."
            )

        audio_groups: Dict[str, Dict[str, Any]] = {}

        for raw_row in entries:
            row = copy.deepcopy(raw_row)
            row["wav_path"] = canonical_wav_path(
                row["wav_path"]
            )
            row["chunk_key"] = canonical_label_key(
                row["chunk_key"]
            )
            row["chunk_id"] = int(row["chunk_id"])
            row["chunk_start"] = int(
                row["chunk_start"]
            )
            row["chunk_end"] = int(row["chunk_end"])
            row["chunk_valid_len"] = int(
                row.get(
                    "chunk_valid_len",
                    SEGMENT_LENGTH,
                )
            )
            row["target_label"] = int(
                row["target_label"]
            )

            chunk_key = row["chunk_key"]
            if chunk_key not in self.cluster_label_dict:
                raise ValueError(
                    "Stage3 dataset chunk Without cluster label:"
                    f"{chunk_key}"
                )

            audio_id = get_stage3_audio_id_from_entry(
                row
            )
            target_label = int(row["target_label"])

            if audio_id not in audio_groups:
                audio_groups[audio_id] = {
                    "audio_id": audio_id,
                    "wav_path": row["wav_path"],
                    "wav_name": str(
                        row.get(
                            "wav_name",
                            Path(row["wav_path"]).name,
                        )
                    ),
                    "target_label": target_label,
                    "chunks": [],
                }
            else:
                previous_label = int(
                    audio_groups[audio_id]["target_label"]
                )
                if previous_label != target_label:
                    raise ValueError(
                        "The chanks label of the same audio is inconsistent:"
                        f"audio_id={audio_id}, "
                        f"previous={previous_label}, "
                        f"current={target_label}"
                    )
                if (
                    audio_groups[audio_id]["wav_path"]
                    != row["wav_path"]
                ):
                    raise ValueError(
                        "The same audio_id corresponds to several wav_path:"
                        f"audio_id={audio_id}"
                    )

            audio_groups[audio_id]["chunks"].append(
                row
            )

        self.records: List[Dict[str, Any]] = []

        for record in audio_groups.values():
            record["chunks"] = sorted(
                record["chunks"],
                key=lambda row: (
                    int(row["chunk_id"]),
                    int(row["chunk_start"]),
                ),
            )
            self.records.append(record)

        self.records = sorted(
            self.records,
            key=lambda record: natural_key(
                record["wav_path"]
            ),
        )

        if len(self.records) == 0:
            raise ValueError(
                "Stage3 The data is empty by audio grouping."
            )

        labels = np.asarray(
            [
                int(record["target_label"])
                for record in self.records
            ],
            dtype=np.int64,
        )
        if sorted(np.unique(labels).tolist()) != [0, 1]:
            raise ValueError(
                "Stage3 dataset It is important to include both categories."
            )

        chunk_counts = [
            len(record["chunks"])
            for record in self.records
        ]

        print(
            "[Stage3-CLASS][DATASET] "
            f"audio={len(self.records)} | "
            f"chunks={sum(chunk_counts)} | "
            f"class0={int(np.sum(labels == 0))} | "
            f"class1={int(np.sum(labels == 1))} | "
            f"min_chunks={min(chunk_counts)} | "
            f"max_chunks={max(chunk_counts)} | "
            f"mean_chunks={float(np.mean(chunk_counts)):.3f}"
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]

        wav_path = canonical_wav_path(
            record["wav_path"]
        )
        waveform = load_audio_16k(wav_path)

        wav_chunks: List[np.ndarray] = []
        valid_lens: List[int] = []
        labels_list: List[torch.Tensor] = []
        chunk_keys: List[str] = []

        for row in record["chunks"]:
            chunk_key = canonical_label_key(
                row["chunk_key"]
            )

            wav_30, valid_len = (
                load_chunk_with_padding_from_wav(
                    waveform,
                    start=int(row["chunk_start"]),
                    end=int(row["chunk_end"]),
                    chunk_id=int(row["chunk_id"]),
                )
            )
            cluster_labels = (
                self.cluster_label_dict[chunk_key]
                .clone()
                .long()
            )

            wav_chunks.append(wav_30)
            valid_lens.append(int(valid_len))
            labels_list.append(cluster_labels)
            chunk_keys.append(chunk_key)

        if len(wav_chunks) == 0:
            raise ValueError(
                "Stage3 audio No valid chunks:"
                f"{record['audio_id']}"
            )

        return {
            "wav_chunks": wav_chunks,
            "valid_lens": valid_lens,
            "labels_list": labels_list,
            "target_label": int(
                record["target_label"]
            ),
            "chunk_keys": chunk_keys,
            "wav_name": str(record["wav_name"]),
            "audio_id": str(record["audio_id"]),
        }

    def repeat30s_collate(
        self,
        batch: List[Dict[str, Any]],
    ):
        """
        Several chunks internal audio exhibits, while retaining chunk_to_audio.
        """
        flat_wav_chunks: List[np.ndarray] = []
        flat_valid_lens: List[int] = []
        flat_labels: List[torch.Tensor] = []
        flat_chunk_keys: List[str] = []

        audio_labels: List[int] = []
        audio_ids: List[str] = []
        wav_names: List[str] = []
        chunk_to_audio: List[int] = []
        chunk_counts: List[int] = []

        for audio_index, item in enumerate(batch):
            wav_chunks = item["wav_chunks"]
            valid_lens = item["valid_lens"]
            labels_list = item["labels_list"]
            chunk_keys = item["chunk_keys"]

            chunk_count = len(wav_chunks)
            if chunk_count <= 0:
                raise ValueError(
                    "collate Quit audio:"
                    f"{item.get('audio_id', '')}"
                )

            audio_labels.append(
                int(item["target_label"])
            )
            audio_ids.append(str(item["audio_id"]))
            wav_names.append(str(item["wav_name"]))
            chunk_counts.append(int(chunk_count))

            for (
                waveform,
                valid_len,
                cluster_labels,
                chunk_key,
            ) in zip(
                wav_chunks,
                valid_lens,
                labels_list,
                chunk_keys,
            ):
                waveform = np.asarray(
                    waveform,
                    dtype=np.float32,
                ).reshape(-1)
                cluster_labels = pad_or_truncate_labels(
                    cluster_labels,
                    target_t=TOK_T_FULL,
                )

                flat_wav_chunks.append(waveform)
                flat_valid_lens.append(int(valid_len))
                flat_labels.append(cluster_labels)
                flat_chunk_keys.append(str(chunk_key))
                chunk_to_audio.append(int(audio_index))

        flat_wav_30 = torch.from_numpy(
            np.stack(
                flat_wav_chunks,
                axis=0,
            ).astype(np.float32, copy=False)
        )
        flat_valid_len = torch.from_numpy(
            np.asarray(
                flat_valid_lens,
                dtype=np.int64,
            )
        )
        flat_cluster_labels = torch.stack(
            flat_labels,
            dim=0,
        ).long()
        audio_labels_tensor = torch.tensor(
            audio_labels,
            dtype=torch.long,
        )
        chunk_to_audio_tensor = torch.tensor(
            chunk_to_audio,
            dtype=torch.long,
        )
        chunk_counts_tensor = torch.tensor(
            chunk_counts,
            dtype=torch.long,
        )

        return (
            flat_wav_30,
            flat_valid_len,
            flat_cluster_labels,
            audio_labels_tensor,
            chunk_to_audio_tensor,
            chunk_counts_tensor,
            flat_chunk_keys,
            wav_names,
            audio_ids,
        )


# ============================================================
# 8) Whisper Stage3 Model
# ============================================================
class Stage3WhisperADPAPTModule(pl.LightningModule):
    """
    Stage3 Joint training model:

        masked cluster prediction
        +
        audio-level AD auxiliary classification
    """

    def __init__(
        self,
        whisper_model_name: str,
        stage1_ckpt_path: str,
        num_clusters: List[int],
        repr_layer: int,
        mask_inject_after_layer: int,
        train_layer_start: int,
        train_layer_end: int,
        train_last_n: Optional[int] = None,
        lr_enc: float = 5e-5,
        lr_head: float = 7e-5,
        weight_decay: float = 0.02,
        mask_prob: float = 0.06,
        mask_span: int = 10,
        lambda_clus: float = 1.0,
        lambda_ad: float = 0.05,
        head_dropout: float = 0.2,
        l2_normalize_reps: bool = False,
        training_step: int = 2000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = WhisperModel.from_pretrained(
            whisper_model_name
        )
        self.feature_extractor = (
            WhisperFeatureExtractor.from_pretrained(
                whisper_model_name
            )
        )
        self.hidden_size = int(
            self.model.config.d_model
        )

        self.num_clusters = [
            int(value) for value in num_clusters
        ]
        self.repr_layer = int(repr_layer)
        self.mask_inject_after_layer = int(
            mask_inject_after_layer
        )

        self.lr_enc = float(lr_enc)
        self.lr_head = float(lr_head)
        self.weight_decay = float(weight_decay)
        self.mask_prob = float(mask_prob)
        self.mask_span = int(mask_span)
        self.lambda_clus = float(lambda_clus)
        self.lambda_ad = float(lambda_ad)
        self.l2_normalize_reps = bool(
            l2_normalize_reps
        )
        self.total_training_steps = int(
            training_step
        )

        # For token mask replacement.
        self.mask_emb = nn.Parameter(
            torch.zeros(self.hidden_size)
        )
        nn.init.normal_(
            self.mask_emb,
            mean=0.0,
            std=0.02,
        )

        # Each K corresponds to one cluster prediction head.
        self.cluster_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(float(head_dropout)),
                    nn.Linear(
                        self.hidden_size,
                        int(cluster_count),
                    ),
                )
                for cluster_count in self.num_clusters
            ]
        )

        # AD auxiliary head:Outputs two logits without softmax.
        self.ad_head = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(float(head_dropout)),
            nn.Linear(
                self.hidden_size,
                NUM_CLASSES,
            ),
        )

        # Only Whisper backbone on Stage1 classification.
        load_stage1_ckpt_into_whisper_model(
            self.model,
            stage1_ckpt_path,
        )

        self._set_trainable_layers(
            train_layer_start=int(train_layer_start),
            train_layer_end=int(train_layer_end),
            train_last_n=train_last_n,
        )

        # A summary of all audio projections under GPU alone at the end of epoch.
        self._train_epoch_targets: List[torch.Tensor] = []
        self._train_epoch_predictions: List[torch.Tensor] = []

    def _set_trainable_layers(
        self,
        train_layer_start: int,
        train_layer_end: int,
        train_last_n: Optional[int],
    ) -> None:
        """
        Freezing most of the parameters of Whisper, only the specified encoder layer.

        cluster heads,ad_head (a) Always trainable with mask_emb.
        """
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        layer_count = int(
            self.model.config.encoder_layers
        )

        if train_last_n not in [None, "", "None"]:
            last_n = int(train_last_n)
            train_layer_start = max(
                1,
                layer_count - last_n + 1,
            )
            train_layer_end = layer_count

        train_layer_start = max(
            1,
            int(train_layer_start),
        )
        train_layer_end = min(
            layer_count,
            int(train_layer_end),
        )

        if train_layer_start > train_layer_end:
            raise ValueError(
                "Stage3 Training level is illegal:"
                f"start={train_layer_start}, "
                f"end={train_layer_end}, "
                f"total={layer_count}"
            )

        for layer_id in range(
            train_layer_start,
            train_layer_end + 1,
        ):
            encoder_layer = (
                self.model.encoder.layers[
                    layer_id - 1
                ]
            )
            for parameter in encoder_layer.parameters():
                parameter.requires_grad = True

        if (
            train_layer_end == layer_count
            and hasattr(
                self.model.encoder,
                "layer_norm",
            )
        ):
            for parameter in (
                self.model.encoder.layer_norm.parameters()
            ):
                parameter.requires_grad = True

        print(
            "[Stage3-CLASS] Whisper trainable encoder layers: "
            f"{train_layer_start}..{train_layer_end} "
            f"/ total={layer_count}"
        )

    def _build_valid_mask(
        self,
        input_features: torch.Tensor,
        valid_wav_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Constructs token level active area mask: [B,T]."""
        batch_size = int(input_features.size(0))
        mel_length = int(input_features.size(-1))
        token_length = (mel_length + 1) // 2

        if valid_wav_lens is None:
            return torch.ones(
                (batch_size, token_length),
                dtype=torch.bool,
                device=input_features.device,
            )

        valid_wav_lens = (
            valid_wav_lens
            .detach()
            .cpu()
            .long()
            .view(-1)
        )

        if valid_wav_lens.numel() != batch_size:
            raise RuntimeError(
                "valid_wav_lens Numbers do not match batch:"
                f"expect={batch_size}, "
                f"got={valid_wav_lens.numel()}"
            )

        valid_token_lengths = []
        for batch_index in range(batch_size):
            token_count = compute_valid_token_length(
                int(
                    valid_wav_lens[
                        batch_index
                    ].item()
                )
            )
            valid_token_lengths.append(
                min(token_count, token_length)
            )

        valid_token_lengths_tensor = torch.tensor(
            valid_token_lengths,
            device=input_features.device,
            dtype=torch.long,
        )
        token_indices = torch.arange(
            token_length,
            device=input_features.device,
        ).view(1, token_length).expand(
            batch_size,
            token_length,
        )

        return (
            token_indices
            < valid_token_lengths_tensor.view(-1, 1)
        ).bool()

    @staticmethod
    def _encoder_layer_forward(
        layer: nn.Module,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compatibility with a different version of transformers WhisperEncoderLayer interface.
        """
        try:
            output = layer(
                hidden_states,
                attention_mask=None,
                layer_head_mask=None,
                output_attentions=False,
            )
        except TypeError:
            try:
                output = layer(
                    hidden_states,
                    attention_mask=None,
                    output_attentions=False,
                )
            except TypeError:
                output = layer(hidden_states)

        return output[0] if isinstance(output, tuple) else output

    def manual_encoder_forward(
        self,
        input_features: torch.Tensor,
        valid_wav_lens: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Manually execute Whisper encoder to inject mask embedding after the specified layer.
        """
        encoder = self.model.encoder

        hidden_states = F.gelu(
            encoder.conv1(input_features)
        )
        hidden_states = F.gelu(
            encoder.conv2(hidden_states)
        )
        hidden_states = hidden_states.permute(
            0,
            2,
            1,
        )

        sequence_length = int(
            hidden_states.size(1)
        )
        position_embedding = (
            encoder.embed_positions.weight[
                :sequence_length,
                :
            ]
            .to(hidden_states.device)
        )
        hidden_states = (
            hidden_states
            + position_embedding.unsqueeze(0)
        )

        dropout_probability = float(
            getattr(encoder, "dropout", 0.0)
        )
        hidden_states = F.dropout(
            hidden_states,
            p=dropout_probability,
            training=self.training,
        )

        valid_mask = self._build_valid_mask(
            input_features,
            valid_wav_lens,
        )

        if mask_positions is not None:
            mask_positions = (
                mask_positions
                .to(hidden_states.device)
                .bool()
            )
            if mask_positions.shape != valid_mask.shape:
                raise RuntimeError(
                    "mask_positions Inconsistent shape with valid_mask:"
                    f"{tuple(mask_positions.shape)} vs "
                    f"{tuple(valid_mask.shape)}"
                )
            mask_positions = (
                mask_positions & valid_mask
            )

        if (
            self.mask_inject_after_layer == 0
            and mask_positions is not None
        ):
            mask_vector = (
                self.mask_emb
                .to(hidden_states.dtype)
                .view(1, 1, -1)
            )
            hidden_states = torch.where(
                mask_positions.unsqueeze(-1),
                mask_vector,
                hidden_states,
            )

        selected_representation = None
        encoder_layer_count = len(encoder.layers)

        if not (
            1
            <= self.repr_layer
            <= encoder_layer_count
        ):
            raise ValueError(
                f"repr_layer={self.repr_layer} It's illegal."
                f"encoder layers={encoder_layer_count}"
            )

        for layer_index, layer in enumerate(
            encoder.layers,
            start=1,
        ):
            hidden_states = self._encoder_layer_forward(
                layer,
                hidden_states,
            )

            if (
                layer_index
                == self.mask_inject_after_layer
                and mask_positions is not None
            ):
                mask_vector = (
                    self.mask_emb
                    .to(hidden_states.dtype)
                    .view(1, 1, -1)
                )
                hidden_states = torch.where(
                    mask_positions.unsqueeze(-1),
                    mask_vector,
                    hidden_states,
                )

            if layer_index == self.repr_layer:
                selected_representation = (
                    hidden_states
                )

        hidden_states = encoder.layer_norm(
            hidden_states
        )

        if self.repr_layer == encoder_layer_count:
            selected_representation = hidden_states

        if selected_representation is None:
            raise RuntimeError(
                "Not extracted to specified expression:"
                f"repr_layer={self.repr_layer}"
            )

        return selected_representation, valid_mask

    def extract_features(
        self,
        wav_30: torch.Tensor,
        valid_len: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """extracts the Whisper token sign from 30 seconds waveform."""
        if wav_30.dim() == 1:
            wav_30 = wav_30.unsqueeze(0)

        waveform_numpy = (
            wav_30.detach().cpu().numpy()
        )
        target_length = int(
            self.feature_extractor.n_samples
        )

        features = self.feature_extractor(
            waveform_numpy,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=target_length,
        )

        input_features = (
            features.input_features.to(self.device)
        )

        return self.manual_encoder_forward(
            input_features=input_features,
            valid_wav_lens=valid_len,
            mask_positions=mask_positions,
        )

    def _pool_utterance(
        self,
        representations: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """To do mean pooling within effective token."""
        mask = (
            valid_mask
            .to(representations.device)
            .float()
            .unsqueeze(-1)
        )
        denominator = mask.sum(
            dim=1
        ).clamp_min(1.0)
        pooled = (
            representations * mask
        ).sum(dim=1) / denominator

        if self.l2_normalize_reps:
            pooled = F.normalize(
                pooled,
                dim=-1,
            )

        return pooled

    @staticmethod
    def _aggregate_chunk_logits_to_audio(
        chunk_logits: torch.Tensor,
        chunk_to_audio: torch.Tensor,
        audio_count: int,
    ) -> torch.Tensor:
        """
        The average value will be [N_chunks,2] logits at audio_id, to be obtained at [B_audio,2].
        """
        if chunk_logits.dim() != 2:
            raise RuntimeError(
                "chunk_logits Must be two-dimensional:"
                f"{tuple(chunk_logits.shape)}"
            )
        if chunk_logits.size(1) != NUM_CLASSES:
            raise RuntimeError(
                "chunk_logits The last dimension must be 2:"
                f"{tuple(chunk_logits.shape)}"
            )

        chunk_count = int(chunk_logits.size(0))
        audio_count = int(audio_count)

        logit_sum = torch.zeros(
            (audio_count, NUM_CLASSES),
            dtype=chunk_logits.dtype,
            device=chunk_logits.device,
        )
        logit_sum.index_add_(
            0,
            chunk_to_audio,
            chunk_logits,
        )

        count = torch.zeros(
            (audio_count, 1),
            dtype=chunk_logits.dtype,
            device=chunk_logits.device,
        )
        count.index_add_(
            0,
            chunk_to_audio,
            torch.ones(
                (chunk_count, 1),
                dtype=chunk_logits.dtype,
                device=chunk_logits.device,
            ),
        )

        return logit_sum / count.clamp_min(1.0)

    def _shared_step(
        self,
        batch,
        stage: str,
    ) -> torch.Tensor:
        """
        Calculates masked bluer loss and auto-level AD loss.
        """
        (
            wav_30,
            valid_len,
            cluster_labels,
            audio_labels,
            chunk_to_audio,
            chunk_counts,
            _chunk_keys,
            _wav_names,
            _audio_ids,
        ) = batch

        wav_30 = wav_30.to(self.device)
        valid_len = (
            valid_len.to(self.device).long()
        )
        cluster_labels = (
            cluster_labels.to(self.device).long()
        )
        audio_labels = (
            audio_labels
            .to(self.device)
            .long()
            .view(-1)
        )
        chunk_to_audio = (
            chunk_to_audio
            .to(self.device)
            .long()
            .view(-1)
        )
        chunk_counts = (
            chunk_counts
            .to(self.device)
            .long()
            .view(-1)
        )

        label_valid = (
            cluster_labels[:, 0, :]
            != IGNORE_INDEX
        )
        mask_positions = make_time_mask_tok_valid(
            valid_mask_tok=label_valid,
            mask_prob=self.mask_prob,
            mask_span=self.mask_span,
        )

        representations, valid_mask = (
            self.extract_features(
                wav_30=wav_30,
                valid_len=valid_len,
                mask_positions=mask_positions,
            )
        )

        number_of_chunks, time_length, _ = (
            representations.shape
        )
        head_count = int(cluster_labels.size(1))
        number_of_audios = int(
            audio_labels.numel()
        )

        if number_of_audios <= 0:
            raise RuntimeError(
                "Stage3 audio batch It's empty."
            )
        if chunk_to_audio.numel() != number_of_chunks:
            raise RuntimeError(
                "chunk_to_audio Numbers do not match chunks."
            )
        if chunk_counts.numel() != number_of_audios:
            raise RuntimeError(
                "chunk_counts Numbers do not match audio batch."
            )
        if int(chunk_counts.sum().item()) != number_of_chunks:
            raise RuntimeError(
                "chunk_counts.sum Unmatched with chunks quantity."
            )
        if int(chunk_to_audio.min().item()) < 0:
            raise RuntimeError(
                "chunk_to_audio Includes negative numbers."
            )
        if (
            int(chunk_to_audio.max().item())
            >= number_of_audios
        ):
            raise RuntimeError(
                "chunk_to_audio Beyond audio batch."
            )
        if head_count != len(self.cluster_heads):
            raise RuntimeError(
                "cluster label heads Inconsistent with model heads:"
                f"labels={head_count}, "
                f"heads={len(self.cluster_heads)}"
            )
        if not torch.all(
            (audio_labels == 0)
            | (audio_labels == 1)
        ):
            raise RuntimeError(
                "audio_labels , the non-0/1 label appears."
            )

        # ----------------------------------------------------
        # masked cluster prediction
        # ----------------------------------------------------
        cluster_losses: List[torch.Tensor] = []
        log_k_values: List[float] = []
        total_masked_tokens = 0
        number_of_heads = len(self.cluster_heads)

        for head_index, head in enumerate(
            self.cluster_heads
        ):
            target = cluster_labels[
                :,
                head_index,
                :,
            ]
            target_valid = target != IGNORE_INDEX
            loss_mask = (
                mask_positions[:, :time_length]
                & target_valid[:, :time_length]
                & valid_mask[:, :time_length]
            )

            masked_count = int(
                loss_mask.long().sum().item()
            )
            total_masked_tokens += masked_count

            logits = head(representations)

            if masked_count > 0:
                head_loss = F.cross_entropy(
                    logits[loss_mask],
                    target[:, :time_length][
                        loss_mask
                    ],
                    reduction="mean",
                )

                if number_of_heads > 1:
                    cluster_count = int(
                        logits.size(-1)
                    )
                    log_k = math.log(
                        float(cluster_count)
                    )
                    head_loss = head_loss / log_k
                    log_k_values.append(log_k)
            else:
                head_loss = logits.sum() * 0.0

            cluster_losses.append(head_loss)

        cluster_loss = torch.stack(
            cluster_losses
        ).mean()

        if (
            number_of_heads > 1
            and len(log_k_values) > 0
        ):
            cluster_loss = cluster_loss * (
                sum(log_k_values)
                / len(log_k_values)
            )

        # ----------------------------------------------------
        # audio-level AD auxiliary classification
        # ----------------------------------------------------
        pooled_chunks = self._pool_utterance(
            representations,
            valid_mask,
        )
        chunk_logits = self.ad_head(
            pooled_chunks
        )
        audio_logits = (
            self._aggregate_chunk_logits_to_audio(
                chunk_logits=chunk_logits,
                chunk_to_audio=chunk_to_audio,
                audio_count=number_of_audios,
            )
        )

        ad_loss = F.cross_entropy(
            audio_logits,
            audio_labels,
            reduction="mean",
        )

        total_loss = (
            self.lambda_clus * cluster_loss
            + self.lambda_ad * ad_loss
        )

        predictions = torch.argmax(
            audio_logits.detach(),
            dim=-1,
        )
        batch_accuracy = (
            predictions == audio_labels
        ).float().mean()

        self.log(
            f"{stage}_loss",
            total_loss,
            prog_bar=True,
            on_step=(stage == "train"),
            on_epoch=True,
            batch_size=number_of_audios,
        )
        self.log(
            f"{stage}_clus_loss",
            cluster_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=number_of_audios,
        )
        self.log(
            f"{stage}_ad_loss",
            ad_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=number_of_audios,
        )
        self.log(
            f"{stage}_audio_accuracy_batch",
            batch_accuracy,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=number_of_audios,
        )

        if stage == "train":
            self._train_epoch_targets.append(
                audio_labels.detach().cpu()
            )
            self._train_epoch_predictions.append(
                predictions.detach().cpu()
            )

            self.log(
                "train_masked_tokens",
                float(total_masked_tokens),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=number_of_audios,
            )
            self.log(
                "train_flat_chunks",
                float(number_of_chunks),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=number_of_audios,
            )
            self.log(
                "train_audio_batch_size",
                float(number_of_audios),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=number_of_audios,
            )
            self.log(
                "train_mean_chunks_per_audio",
                float(number_of_chunks)
                / float(max(number_of_audios, 1)),
                prog_bar=False,
                on_step=True,
                on_epoch=False,
                batch_size=number_of_audios,
            )

        return total_loss

    def on_train_epoch_start(self) -> None:
        """Emptys the current audio-level forecast cache for epoch."""
        self._train_epoch_targets = []
        self._train_epoch_predictions = []

    def on_train_epoch_end(self) -> None:
        """
        audio projections for the entire round of official train are used to calculate the classification indicators.

        Precision, Recall, and F1 use label 1 (Disease Group) as the
        positive class. This F1 is positive-class F1, not Macro-F1.

        These indicators are only used to observe the AD support tasks of Stage3 and are used to monitor the impact of the project.
        Not for early stopping or checkpoint selection.
        """
        if (
            len(self._train_epoch_targets) == 0
            or len(self._train_epoch_predictions) == 0
        ):
            return

        targets = torch.cat(
            self._train_epoch_targets,
            dim=0,
        ).long()
        predictions = torch.cat(
            self._train_epoch_predictions,
            dim=0,
        ).long()

        true_positive = int(
            ((targets == 1) & (predictions == 1))
            .sum()
            .item()
        )
        true_negative = int(
            ((targets == 0) & (predictions == 0))
            .sum()
            .item()
        )
        false_positive = int(
            ((targets == 0) & (predictions == 1))
            .sum()
            .item()
        )
        false_negative = int(
            ((targets == 1) & (predictions == 0))
            .sum()
            .item()
        )

        total = max(1, int(targets.numel()))
        accuracy = (
            true_positive + true_negative
        ) / float(total)

        precision_denominator = (
            true_positive + false_positive
        )
        recall_denominator = (
            true_positive + false_negative
        )

        precision = (
            true_positive
            / float(precision_denominator)
            if precision_denominator > 0
            else 0.0
        )
        recall = (
            true_positive
            / float(recall_denominator)
            if recall_denominator > 0
            else 0.0
        )
        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        self.log(
            "train_audio_accuracy_epoch",
            float(accuracy),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_audio_precision_epoch",
            float(precision),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_audio_recall_epoch",
            float(recall),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train_audio_f1_epoch",
            float(f1),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

    def training_step(
        self,
        batch,
        batch_index: int,
    ) -> torch.Tensor:
        return self._shared_step(
            batch,
            stage="train",
        )

    def configure_optimizers(self):
        """
        Grouping learning rate:
        - Whisper Use of LR_ENC for unfrozen layers;
        - mask_emb,cluster_heads,ad_head Use LR_HEAD.
        """
        encoder_parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        head_parameters = (
            list(self.cluster_heads.parameters())
            + list(self.ad_head.parameters())
            + [self.mask_emb]
        )

        parameter_groups = []

        if len(encoder_parameters) > 0:
            parameter_groups.append(
                {
                    "params": encoder_parameters,
                    "lr": self.lr_enc,
                    "weight_decay": self.weight_decay,
                }
            )

        parameter_groups.append(
            {
                "params": head_parameters,
                "lr": self.lr_head,
                "weight_decay": self.weight_decay,
            }
        )

        optimizer = torch.optim.AdamW(
            parameter_groups
        )

        total_steps = max(
            1,
            int(self.total_training_steps),
        )

        def lr_lambda(step: int) -> float:
            step = int(step)
            if step >= total_steps:
                return 0.0

            return 0.5 * (
                1.0
                + math.cos(
                    math.pi
                    * step
                    / float(total_steps)
                )
            )

        scheduler = (
            torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda,
            )
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


# ============================================================
# 9) Save configuration, Trainer with single run
# ============================================================
def write_stage3_run_meta(
    run_config: Dict[str, Any],
    stage1_checkpoint_path: str,
    train_cluster_path: str,
    manifest_stats: Dict[str, Any],
    loaded_k_list: Sequence[int],
    training_steps: int,
) -> None:
    """Save meta-information for Stage3 classification."""
    method_name = (
        "clean_cpft"
        if float(LAMBDA_AD) == 0.0
        else "ad_aware_cpft"
    )

    metadata = {
        "task": "ad_binary_classification",
        "stage": "stage3_ad_aware_cpft",
        "method": method_name,
        "class_mapping": {
            "0": CLASS_NAMES[0],
            "1": CLASS_NAMES[1],
        },
        "dataset_option": str(
            run_config["DATASET_OPTION"]
        ),
        "official_train_used": True,
        "official_test_used": False,
        "run_protocol": str(
            run_config["RUN_PROTOCOL_TAG"]
        ),
        "model_root_tag": str(MODEL_ROOT_TAG),
        "repr_layer": int(
            run_config["REPR_LAYER"]
        ),
        "layer_name": str(
            run_config["LAYER_NAME"]
        ),
        "unfreeze_encoder_layers": int(
            run_config[
                "UNFREEZE_ENCODER_LAYERS"
            ]
        ),
        "resolved_unfreeze_encoder_layers": list(
            run_config[
                "RESOLVED_UNFREEZE_ENCODER_LAYERS"
            ]
        ),
        "stage3_train_layer_start": int(
            run_config[
                "STAGE3_TRAIN_LAYER_START"
            ]
        ),
        "stage3_train_layer_end": int(
            run_config[
                "STAGE3_TRAIN_LAYER_END"
            ]
        ),
        "mask_offset": int(
            run_config["MASK_OFFSET"]
        ),
        "mask_inject_after_layer": int(
            run_config[
                "MASK_INJECT_AFTER_LAYER"
            ]
        ),
        "stage1_checkpoint_path": str(
            stage1_checkpoint_path
        ),
        "stage2_train_cluster_path": str(
            train_cluster_path
        ),
        "stage2_train_manifest_path": str(
            run_config[
                "STAGE2_TRAIN_MANIFEST"
            ]
        ),
        "stage3_save_dir": str(
            run_config["STAGE3_SAVE_DIR"]
        ),
        "num_clusters": [
            int(value) for value in loaded_k_list
        ],
        "segment_sec": float(SEGMENT_SEC),
        "overlap_rate": float(OVERLAP_RATE),
        "chunk_key_version": str(
            CHUNK_KEY_VERSION
        ),
        "mask_prob": float(MASK_PROB),
        "mask_span": int(MASK_SPAN),
        "lambda_clus": float(LAMBDA_CLUS),
        "lambda_ad": float(LAMBDA_AD),
        "aux_classification_level": str(
            STAGE3_AUX_CLASSIFICATION_LEVEL
        ),
        "audio_logit_aggregation": "mean",
        "ad_head_output_dim": int(NUM_CLASSES),
        "stage3_audio_batch_size": int(
            STAGE3_AUDIO_BATCH_SIZE
        ),
        "stage3_max_epochs": int(
            STAGE3_MAX_EPOCHS
        ),
        "stage3_checkpoint_selection": str(
            STAGE3_CHECKPOINT_SELECTION
        ),
        "effective_training_steps": int(
            training_steps
        ),
        "manifest_stats": manifest_stats,
        "path_param_tags": list(
            PATH_PARAM_TAGS
        ),
        "stage1_source_path_param_tags": list(
            STAGE1_SOURCE_PATH_PARAM_TAGS
        ),
    }

    save_json(
        metadata,
        os.path.join(
            run_config["STAGE3_SAVE_DIR"],
            "config.json",
        ),
    )
    save_json(
        metadata,
        os.path.join(
            run_config["META_SAVE_DIR"],
            "stage3_class_meta.json",
        ),
    )


def build_stage3_trainer(
    stage3_save_dir: str,
) -> Trainer:
    """
    Construct no-val-last Trainer.

    Do not use validation, do not execute early stopping,
    Save last.ckpt only.
    """
    ensure_dir(stage3_save_dir)

    checkpoint_callback = ModelCheckpoint(
        dirpath=stage3_save_dir,
        filename="whisper-s3-class-last",
        monitor=None,
        save_top_k=0,
        save_last=True,
        auto_insert_metric_name=False,
        verbose=True,
    )

    logger = False
    if ENABLE_CSV_LOGGER:
        logger = CSVLogger(
            save_dir=stage3_save_dir,
            name="csv_logs",
        )

    accelerator = (
        "gpu"
        if torch.cuda.is_available()
        else "cpu"
    )
    precision = (
        "16-mixed"
        if PRECISION == 16
        and torch.cuda.is_available()
        else "32-true"
    )

    return Trainer(
        max_epochs=int(STAGE3_MAX_EPOCHS),
        max_steps=-1,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=[checkpoint_callback],
        logger=logger,
        gradient_clip_val=float(GRAD_CLIP),
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=int(
            ACCUMULATE_GRAD_BATCHES
        ),
        log_every_n_steps=1,
        deterministic=False,
        enable_checkpointing=True,
        num_sanity_val_steps=0,
        limit_val_batches=0,
    )


def run_one_experiment(
    run_config: Dict[str, Any],
) -> None:
    """
    Execute a dataset + player + unfreeze + mask_offset experiment.
    """
    seed_everything_for_repro(SEED)

    ensure_dir(run_config["STAGE3_SAVE_DIR"])
    ensure_dir(run_config["META_SAVE_DIR"])

    print("\n" + "=" * 90)
    print(
        "[Stage3-CLASS] DATASET =",
        run_config["DATASET_OPTION"],
    )
    print(
        "[Stage3-CLASS] LAYER_NAME =",
        run_config["LAYER_NAME"],
    )
    print(
        "[Stage3-CLASS] RESOLVED_UNFREEZE =",
        run_config[
            "RESOLVED_UNFREEZE_ENCODER_LAYERS"
        ],
    )
    print(
        "[Stage3-CLASS] MASK_OFFSET =",
        run_config["MASK_OFFSET"],
    )
    print(
        "[Stage3-CLASS] MASK_INJECT_AFTER_LAYER =",
        run_config[
            "MASK_INJECT_AFTER_LAYER"
        ],
    )
    print(
        "[Stage3-CLASS] STAGE1_SAVE_DIR =",
        run_config["STAGE1_SAVE_DIR"],
    )
    print(
        "[Stage3-CLASS] STAGE2_TRAIN_LABELPATH =",
        run_config[
            "STAGE2_TRAIN_LABELPATH"
        ],
    )
    print(
        "[Stage3-CLASS] STAGE2_TRAIN_MANIFEST =",
        run_config[
            "STAGE2_TRAIN_MANIFEST"
        ],
    )
    print(
        "[Stage3-CLASS] STAGE3_SAVE_DIR =",
        run_config["STAGE3_SAVE_DIR"],
    )
    print("=" * 90)

    # 1. Stage2 official train manifest.
    train_entries, manifest_stats = (
        load_stage2_chunks_manifest(
            run_config[
                "STAGE2_TRAIN_MANIFEST"
            ],
            dataset_option=run_config[
                "DATASET_OPTION"
            ],
        )
    )

    # 2. Stage2 cluster pseudo labels.
    train_cluster_dict, loaded_k_list = (
        load_all_clus_json(
            run_config[
                "STAGE2_TRAIN_LABELPATH"
            ],
            expected_k_list=(
                EXPECTED_NUM_CLUSTERS
            ),
        )
    )
    check_stage3_cluster_config(
        loaded_k_list=loaded_k_list,
        expected_num_clusters=(
            EXPECTED_NUM_CLUSTERS
        ),
        train_clus_path=run_config[
            "STAGE2_TRAIN_LABELPATH"
        ],
    )

    train_entries, missing_cluster = (
        filter_entries_by_cluster_labels(
            train_entries,
            train_cluster_dict,
        )
    )
    if len(missing_cluster) > 0:
        save_json(
            {
                "missing_cluster_entries": (
                    missing_cluster
                )
            },
            os.path.join(
                run_config["META_SAVE_DIR"],
                "stage3_missing_cluster_entries.json",
            ),
        )
        raise ValueError(
            "Stage2 manifest Match to all-clus.json incomplete:"
            f"missing={len(missing_cluster)}"
        )

    if len(train_entries) == 0:
        raise ValueError(
            "Stage3 cluster label Match after train is empty."
        )

    # 3. Stage1 class checkpoint.
    stage1_checkpoint_path = (
        find_stage1_checkpoint(
            stage1_save_dir=run_config[
                "STAGE1_SAVE_DIR"
            ],
            override_ckpt=STAGE1_CKPT_OVERRIDE,
        )
    )
    print(
        "[Stage3-CLASS] Stage1 checkpoint =",
        stage1_checkpoint_path,
    )

    # 4. Dataset / DataLoader.
    train_dataset = Stage3ADPAPTDataset(
        entries=train_entries,
        cluster_label_dict=train_cluster_dict,
    )
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=int(
            STAGE3_AUDIO_BATCH_SIZE
        ),
        shuffle=True,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=(
            train_dataset.repeat30s_collate
        ),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(
            int(NUM_WORKERS) > 0
        ),
    )

    steps_per_epoch = int(
        math.ceil(
            len(train_loader)
            / max(
                1,
                int(
                    ACCUMULATE_GRAD_BATCHES
                ),
            )
        )
    )
    effective_training_steps = max(
        1,
        steps_per_epoch
        * int(STAGE3_MAX_EPOCHS),
    )

    print(
        "[Stage3-CLASS] "
        f"audio={len(train_dataset)} | "
        f"steps_per_epoch={steps_per_epoch} | "
        f"epochs={STAGE3_MAX_EPOCHS} | "
        f"effective_steps={effective_training_steps}"
    )

    # 5. Model.
    model = Stage3WhisperADPAPTModule(
        whisper_model_name=WHISPER_MODEL_NAME,
        stage1_ckpt_path=(
            stage1_checkpoint_path
        ),
        num_clusters=loaded_k_list,
        repr_layer=int(
            run_config["REPR_LAYER"]
        ),
        mask_inject_after_layer=int(
            run_config[
                "MASK_INJECT_AFTER_LAYER"
            ]
        ),
        train_layer_start=int(
            run_config[
                "STAGE3_TRAIN_LAYER_START"
            ]
        ),
        train_layer_end=int(
            run_config[
                "STAGE3_TRAIN_LAYER_END"
            ]
        ),
        train_last_n=None,
        lr_enc=float(LR_ENC),
        lr_head=float(LR_HEAD),
        weight_decay=float(WEIGHT_DECAY),
        mask_prob=float(MASK_PROB),
        mask_span=int(MASK_SPAN),
        lambda_clus=float(LAMBDA_CLUS),
        lambda_ad=float(LAMBDA_AD),
        head_dropout=float(HEAD_DROPOUT),
        l2_normalize_reps=bool(
            L2_NORMALIZE_REPS
        ),
        training_step=int(
            effective_training_steps
        ),
    )

    # 6. Meta / manifest.
    write_stage3_run_meta(
        run_config=run_config,
        stage1_checkpoint_path=(
            stage1_checkpoint_path
        ),
        train_cluster_path=run_config[
            "STAGE2_TRAIN_LABELPATH"
        ],
        manifest_stats=manifest_stats,
        loaded_k_list=loaded_k_list,
        training_steps=(
            effective_training_steps
        ),
    )

    save_text(
        stage1_checkpoint_path,
        os.path.join(
            run_config["STAGE3_SAVE_DIR"],
            "stage1_ckpt_path.txt",
        ),
    )
    save_text(
        run_config[
            "STAGE2_TRAIN_LABELPATH"
        ],
        os.path.join(
            run_config["STAGE3_SAVE_DIR"],
            "stage2_train_labelpath.txt",
        ),
    )
    save_text(
        run_config[
            "STAGE2_TRAIN_MANIFEST"
        ],
        os.path.join(
            run_config["STAGE3_SAVE_DIR"],
            "stage2_train_manifest.txt",
        ),
    )

    if SAVE_MANIFESTS:
        save_json(
            {
                "task": (
                    "ad_binary_classification"
                ),
                "split": "train",
                "official_train_used": True,
                "official_test_used": False,
                "class_mapping": {
                    "0": CLASS_NAMES[0],
                    "1": CLASS_NAMES[1],
                },
                "num_entries": int(
                    len(train_entries)
                ),
                "entries": train_entries,
            },
            os.path.join(
                run_config["STAGE3_SAVE_DIR"],
                "train_manifest.json",
            ),
        )

    # 7. Fixed epoch training and preservation last.ckpt.
    trainer = build_stage3_trainer(
        run_config["STAGE3_SAVE_DIR"]
    )
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        ckpt_path=RESUME_STAGE3_CKPT,
    )

    print(
        "\n[Stage3-CLASS] Completed:",
        run_config["STAGE3_SAVE_DIR"],
    )


# ============================================================
# 10) Main entrance
# ============================================================
def main() -> None:
    """
    Walking through:
        DATASET_OPTIONS
        UNFREEZE_ENCODER_LAYERS_LIST
        STAGE3_REPR_LAYERS_TO_RUN
        STAGE3_MASK_INJECT_AFTER_LAYERS
    """
    global DATASET_OPTION
    global UNFREEZE_ENCODER_LAYERS

    if len(DATASET_OPTIONS) == 0:
        raise ValueError(
            "DATASET_OPTIONS Can't be empty."
        )
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError(
            "UNFREEZE_ENCODER_LAYERS_LIST Can't be empty."
        )
    if len(STAGE3_REPR_LAYERS_TO_RUN) == 0:
        raise ValueError(
            "STAGE3_REPR_LAYERS_TO_RUN Can't be empty."
        )
    if len(STAGE3_MASK_INJECT_AFTER_LAYERS) == 0:
        raise ValueError(
            "STAGE3_MASK_INJECT_AFTER_LAYERS Can't be empty."
        )

    print("[CONFIG]")
    print("  CLASS_PROJECT_ROOT =", CLASS_PROJECT_ROOT)
    print("  PROJECT_ROOT =", PROJECT_ROOT)
    print("  CPFT_ROOT =", CPFT_ROOT)
    print("  DATASET_OPTIONS =", DATASET_OPTIONS)
    print(
        "  STAGE3_REPR_LAYERS_TO_RUN =",
        STAGE3_REPR_LAYERS_TO_RUN,
    )
    print(
        "  UNFREEZE_ENCODER_LAYERS_LIST =",
        UNFREEZE_ENCODER_LAYERS_LIST,
    )
    print(
        "  STAGE3_MASK_INJECT_AFTER_LAYERS =",
        STAGE3_MASK_INJECT_AFTER_LAYERS,
    )
    print(
        "  EXPECTED_NUM_CLUSTERS =",
        EXPECTED_NUM_CLUSTERS,
    )
    print("  LAMBDA_CLUS =", LAMBDA_CLUS)
    print("  LAMBDA_AD =", LAMBDA_AD)
    print(
        "  AUX_CLASSIFICATION_LEVEL =",
        STAGE3_AUX_CLASSIFICATION_LEVEL,
    )
    print("  official_test_used = False")

    for dataset_option in [_canonical_dataset_option(value) for value in DATASET_OPTIONS]:
        for unfreeze_count in [
            int(value)
            for value in (
                UNFREEZE_ENCODER_LAYERS_LIST
            )
        ]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = (
                unfreeze_count
            )

            print("\n" + "=" * 100)
            print("[Stage3-CLASS BATCH ITEM]")
            print(
                "  DATASET_OPTION =",
                DATASET_OPTION,
            )
            print(
                "  UNFREEZE_ENCODER_LAYERS =",
                UNFREEZE_ENCODER_LAYERS,
            )
            print("=" * 100)

            for repr_layer in [
                int(value)
                for value in (
                    STAGE3_REPR_LAYERS_TO_RUN
                )
            ]:
                for mask_offset in [
                    int(value)
                    for value in (
                        STAGE3_MASK_INJECT_AFTER_LAYERS
                    )
                ]:
                    run_config = (
                        build_stage3_run_config(
                            dataset_option=(
                                DATASET_OPTION
                            ),
                            repr_layer=repr_layer,
                            mask_offset=(
                                mask_offset
                            ),
                        )
                    )
                    run_one_experiment(
                        run_config
                    )


if __name__ == "__main__":
    main()
