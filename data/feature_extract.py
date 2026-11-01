# -*- coding: utf-8 -*-



try:
    from .paper_names import paper_dataset_name
except ImportError:
    from paper_names import paper_dataset_name
# Converted from Jupyter notebook: frozen.py


# %% cell 1
# ===================== 0) Environment and random seeds (must be set before import torch) =====================
import os, random, numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = os.getenv('CUDA_VISIBLE_DEVICES', '5') 
os.environ["PYTHONHASHSEED"] = "400"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

random.seed(400)
np.random.seed(400)

import torch
torch.manual_seed(400)
torch.cuda.manual_seed_all(400)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ===================== 1) The rest is imported =====================
import re
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel, AutoFeatureExtractor, HubertModel, WavLMModel, Wav2Vec2Model

# ===================== 2) Control area: change parameter ===================== here
# -- One key to switch the feature backbone (only this line) -
# Optional: "hubert" / "wav2vec2"/ "wavlm"/"whisper" /
FEATURE_BACKBONE = "whisper"
FEAT_MODEL_NAME = "whisper_small"

"""
conda activate audiojax_gpu
python frozen.py
"""

# -- Unified configuration table for each bone (model path / model name + output prefix) -
# Note:
# 1) model_name You can write a name HuggingFace or a local model directory.
# 2) feature_prefix Writes the output filename to follow up space_state_models.ipynb automatically.
BACKBONE_CONFIG = {
    "hubert": {
        # The English data suggest using the pre-training model or the corresponding local directory in English HuBERT.
        "model_name": "/home/user/whisper_lru_3/hubert_base_ls960",
        "feature_prefix": "hubert",
    },
    "wav2vec2": {
        "model_name": "/home/user/whisper_lru_3/wav2vec2_base_960h",
        "feature_prefix": "wav2vec2",
    },  
    "wavlm": {
        "model_name": "/home/user/whisper_lru_3/wavlm_base_plus",
        "feature_prefix": "wavlm",
    },
    "whisper": {
        "model_name": "/home/user/whisper_lru_3/whisper_small",# whisper_small crisperwhisper whisper_medium
        "feature_prefix": "whisper",
    },
}

# -- Whether to force local files only (offline environment recommendation True) -
LOCAL_FILES_ONLY = True

# -- Regulate and verify the name of the boner--
FEATURE_BACKBONE = str(FEATURE_BACKBONE).strip().lower()
if FEATURE_BACKBONE not in BACKBONE_CONFIG:
    raise ValueError(
        f"Unsupported FEATURE_BACKBONE: {FEATURE_BACKBONE},Option: {list(BACKBONE_CONFIG.keys())}"
    )

# -- Configure automatic derivatives by bone core "Real Load Model Path / Name" and output prefix --
PRETRAINED_MODEL_NAME = BACKBONE_CONFIG[FEATURE_BACKBONE]["model_name"]
FEATURE_PREFIX = BACKBONE_CONFIG[FEATURE_BACKBONE]["feature_prefix"]

# -- Audio re-entry switch (paper pre-processing contains a re-profiling volume) -
ENABLE_AUDIO_NORMALIZE = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- Datasets - (last level of impact to store catalogues)
DATASET_KEY = "ADReSS-IS2020"            # Optional: "ADReSS-IS2020"/ "ADReSSo21" / "Pitt" / "PROCESS-2"

# -- Sub-paragraph Parameter (comparable to 0.3_30s in file name) -
sampling_rate = 16000 
SEGMENT_SEC   = 30 # Each paragraph 30s = = = = = = = = = > NCMMSC 2021 uses only 30s of long audio?
segment_length1 = SEGMENT_SEC * sampling_rate
overlap_rate1   = 0.3 # 30% Overlap
# -- Select extraction mode -- 
# 1) Single-storey list: One-time multi-storey run (priority, one-storey run if the list is not empty)
SINGLE_LAYER_LIST = [12] # Example: fill in the layer you want to be extracted in bulk; leave empty [] without running single-storey mode
NORM_FOR_SINGLE   = "ln_post"    # None / "layernorm" / "std" / "ln_post" 
"""
# layernorm:
    Call F.layer_norm (h, h.shape [-1:]), no learning parameters are used (weight/bias are default None).
    It's just standardised by its own mean value / for current characteristics.
    "Normalization of pure statistics (no parameter)"
# ln_post:
    It's called from model.encoder.layer_norm,.
    With a trained parameter (weight/bias + eps), is the ultimate post-LN. of the Whisper model
    "Whisper Trained end level LayerNorm( has parameters, fixed distribution )"
# Useful:
    layernorm More common, but not necessarily identical to the top level of Whisper.
    ln_post The standard distribution that pulls the features to the top of Whisper is often used to align different layers.
"""
# 2) After K layer integration list: effective when SINGLE_LAYER_LIST is empty; can run more than K
LAST_K_LIST       = []      # Example: K for batch integration; if integration is not required, insert []
DROP_INPUT_LAYER  = True
LAYER_F1          = None         # Weighted integration if list of layers F1 is provided; waste code
F1_TEMP           = 12           # F1 Temperature parameters for weights, controlling the smoothness of weighted integration (the sharper, the smaller the smoother)
NORM_BEFORE_FUSE  = "ln_post"    # None / "layernorm" / "std" / "ln_post"

# -- Top layer number (None indicates automatic extrapolation from model configuration) -
TOP_LAYER_INDEX = None


# -- Save Root Path (Strictly follow your directory format) -
# whisper_small_1_12,whisper_medium_1_24,whisper_largev2_1_32,hubert_base_1_12
# # ADReSSo21,ADReSS2020,NCMMSC2021,Pitt
# hubert_base/large/xlarge = 1_12:9--10; 1_24:18--21;1_48:38--44

"""
conda activate audiojax_gpu
python frozen.py
"""

# Create directory automatically if it does not exist
# Save root directory:
# The single layer will be further saved to
# /home/user/Alzheimer_code_jax_mmse_speech_v3/feature_tensor/DATASET_KEY/FEAT_MODEL_NAME/{single_layer}/
BASE_SAVE_ROOT = "/home/user/Alzheimer_code_jax_mmse_speech_v3/feature_tensor"

# ===================== 2.1) Oversight Target Configuration =====================
# Optional:
# "diagnosis" -> Keep the original AD Second Classification Label (cc=0, cd=1)
# "mmse"      -> Change to MMSE regression Tag
TARGET_KIND = "mmse"

# MMSE metadata path for each data set
# Note:
# 1) Only the MMSE metadata path is added here, without changing your current audio directory DATASETS path.
# 2) The MMSE file formats of different data sets are different, so the datasets are then interpreted separately.
# 3) Uniform objectives: All are eventually converted to { audioID: mmse} dictionary, which is copied to segment labels.
MMSE_META_PATHS = {
    "PROCESS-2": {
        "meta": "/home/user/Downloads/PROCESS-2/meta-info.csv",
    },
    "ADReSS-IS2020": {
        # Original ADReSS 2020: train split into cc/cd two txt and test shared one outcome document
        "train_cc": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/cc_meta_data.txt",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/cd_meta_data.txt",
        "test_all": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/ADReSS 2020 test result.txt",
    },
    "ADReSSo21": {
        # ADReSSo21:mmse in train csv, test set mmse in groundtruth csv
        "train_all": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/adresso-train-mmse-scores.csv",
        "test_all": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/test_results_task2_groundtruth.csv",
    },
    "Pitt": {
        # Pitt:Use the matrix to map visit-level MMSE
        # readme Description:
        #   In the name of the document 0 -> baseline
        #   1 -> Mission 2
        #   2 -> Mission 3
        #   ...
        # So the following is automatically mapd here at _read_pitt_mmse(...):
        #   0 -> mms
        #   1 -> mmse2
        #   2 -> mmse3
        #   3 -> mmse4
        #   4 -> mmse5
        #   5 -> mmse6
        #   6 -> mmse7
        "all_meta": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/PItt-data(1).xlsx",
        "sheet_name": "data",
        "header_row": 2,
        "id_col": "id",
    },
    "ADReSS_m": {
        # ADReSS-M:Training/test with one roundtruth csv
        "train_all": "/home/user/Downloads/dementiabank/ADReSS-M/train1/training-groundtruth.csv",
        "test_all": "/home/user/Downloads/dementiabank/ADReSS-M/train1/test-gr-groundtruth.csv",
    },
}


# ===================== 3) Dataset path registration form =====================
"""AD_combine It's a combination of train/ AD and train/ MCI. It's only using long audio.
    The test set is for labels.
    cc:Health, cd:"""
DATASETS = {
    "PROCESS-2": {
        "train_cc": "/home/user/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/user/Downloads/PROCESS-2/train/cd",
        "test_cc" : "/home/user/Downloads/PROCESS-2/test/cc",
        "test_cd" : "/home/user/Downloads/PROCESS-2/test/cd",
    },
    "NCMMSC2021": {
        "train_cc": "/home/user/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/HC",
        "train_cd": "/home/user/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/AD_combine",
        "test_cc" : "/home/user/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/HC",
        "test_cd" : "/home/user/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/AD_combine",
    },
    "ADReSSo21": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc" : "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd" : "/home/user/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc" : "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd" : "/home/user/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "Pitt_1": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_test/train/cc",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_test/train/cd",
        "test_cc" : '/home/user/Downloads/dementiabank/Pitt_test/test/cc',
        "test_cd" : '/home/user/Downloads/dementiabank/Pitt_test/test/cd',
    },
    "Pitt": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc" : '/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence',
        "test_cd" : '/home/user/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence',
    },
    "Pitt_val1": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/train/cc",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/train/cd",
        "test_cc" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/test/cc',
        "test_cd" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/test/cd',
    },
    "Pitt_val2": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/train/cc",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/train/cd",
        "test_cc" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/test/cc',
        "test_cd" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/test/cd',
    },
    "Pitt_val3": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/train/cc",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/train/cd",
        "test_cc" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/test/cc',
        "test_cd" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/test/cd',
    },
    "Pitt_val4": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/train/cc",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/train/cd",
        "test_cc" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/test/cc',
        "test_cd" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/test/cd',
    },
    "Pitt_val5": {
        "train_cc": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/train/cc",
        "train_cd": "/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/train/cd",
        "test_cc" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/test/cc',
        "test_cd" : '/home/user/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/test/cd',
    },
    "ADReSS_m": {
    "train_cc": "/home/user/Downloads/dementiabank/ADReSS-M/train1/train_wav/HC",
    "train_cd": "/home/user/Downloads/dementiabank/ADReSS-M/train1/train_wav/AD",
    "test_cc" : '/home/user/Downloads/dementiabank/ADReSS-M/train1/test-gr/HC',
    "test_cd" : '/home/user/Downloads/dementiabank/ADReSS-M/train1/test-gr/AD',
},
}
print("DATASET_KEY =", repr(DATASET_KEY))
print("DATASETS keys =", list(DATASETS.keys()))
assert DATASET_KEY in DATASETS, f"Unknown DATASET_KEY={DATASET_KEY}"



# ===================== 4) Tool Function =====================
def natural_key(p: str):
    """Ensure that filenames are sorted in natural order (number priority)"""
    s = Path(p).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def list_wav_files_for_dataset(audio_path: str, dataset_key: str):
    """
    Harmonizes the wav files that need to be read for the current data set.

    PROCESS-2:
        Read Only Tasks wav with __CTD:
        For example, PROCESS-2_rec__001__CTD.wav

    Other data sets:
        Keep the same logic and read all.wav files under the directory.
    """
    dataset_key_norm = str(dataset_key).strip().lower()

    files = []
    for f in os.listdir(audio_path):
        fp = os.path.join(audio_path, f)

        if not f.lower().endswith(".wav"):
            continue

        if dataset_key_norm == "process-2":
            stem = Path(f).stem
            if not re.search(r"__CTD$", stem, flags=re.IGNORECASE):
                continue

        files.append(fp)

    return sorted(files, key=natural_key)

def weights_from_f1(f1_list, temp=8.0):
    """Change list F1 to softmax weight (temperature temp control distribution smooth)
    Temp Temp controls smooth distribution" means: the bigger the temp, the sharper the weight of the softmax; the smaller the temp, the more even the weight.
    So:
        temp Large: Exp differences are magnified and the weight of the high F1 has risen significantly, and the low F1 has almost been crushed to zero.
        temp Small: Differences are compressed and weights are more even.
    You can use the temp as a "prefer to a high-level stratification knob."
    Consider "distribution smooth" primarily to control the probabilities and generalizations of integration and to avoid overweighting a certain layer. More specifically:
        If not smooth (temp large), almost the highest F1 level is easy to compose (F1 may be just an incidental high score for a single validation set).
        A bit smoother (temp medium or small) would allow multiple layers to contribute, be more robust and less sensitive to changes in data distribution.
        Whisper The layers tend to contain different levels of information (acoustic, semantic, talking person characteristics, etc.), and the appropriate integration is more comprehensive than the single layer.
    The phrase "slipping" is intended to strike a compromise between "the best single layer" and "the smooth integration of multiple layers" and to reduce the risk of incidentality.
    """
    import numpy as _np
    f1 = _np.asarray(f1_list, dtype=_np.float64)
    fmin, fmax = float(f1.min()), float(f1.max())
    if abs(fmax - fmin) < 1e-12:
        return _np.ones_like(f1) / len(f1)
    s = (f1 - fmin) / (fmax - fmin)     # Normalize F1 to [0,1]
    e = _np.exp(temp * s)               
    return e / (e.sum() + 1e-12)

def apply_norm(h: torch.Tensor, mode: str, ln_post=None, skip_ln_post: bool=False):
    """
    Normalize [1,T,C]:
      None / "layernorm" / "std" / "ln_post"
    apply_norm The role is to standardize the hidden features of Whisper so that the different layers or segments are more consistent in their distribution and facilitate subsequent integration and training.
    It supports several models:
        None:Do not do it down to normal, go straight back.
        "layernorm":No learning parameter ) is available for character LayerNorm(.
        "std":Scales only to the overall standard deviation (control measure).
        "ln_post" / "whisper_ln":Characterization is standardized using the last LayerNorm( training parameter ) of the Whisper encoder.
    Why?
        The characteristics of the different layers vary widely and direct integration may be unstable.
        Normalization can bring characteristics closer to each other and enhance integration and training stability.
        ln_post It also allows the intermediate feature to be "segmented" to the top of Whisper distribution, facilitating the use of the same classifier or downstream model.
    One word.
        apply_norm It's about aligning the different layers and segments, stabilizing the training and integration results.
    [1, T, C] is the mass shape of Whisper encoder output:
        1:batch size(Only one audio segment at a time.
        T:Time step length (frame length depending on audio length and Whisper's frame/step length).
        C:The hidden dimensions of feature dimensions (Whisper encoder, such as the small model is 768).
        So each frame has a characteristic vector of C in length, in T frames.
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
            raise ValueError("ln_post is None,But you chose 'lnpt/whosper'.")
        return ln_post(h)
    raise ValueError(f"Unknown norm mode: {mode}")

def stack_with_norm(hs_list, mode, ln_post=None, skip_last_ln_post=False):
    """Reconsolidation of layers before stack
    stack_with_norm The role is to first level the features of multiple layers down to a single volume, in preparation for subsequent "multilayer integration"
    What have you done?
        Enter a list of hs_list( layers [1, T, C]) for each shape.
        If mode is None, directly torch.stack.
        Otherwise, call each layer to apply_norm to be normalized.
        skip_last_ln_post In_post (the top may have post-LN).
    Why would you do that?
        The distribution of the different layers is not uniform, and direct stacking and re-integration allows certain layers to "undertake advantage" because of measurements or statistical differences.
        The first normalization would allow each layer to be integrated on the same scale, more stable and more equitable.
        The top level is jumped to avoid double-diversion (otherwise it would destroy the distribution).
    In one sentence, the stack_with_norm is to align the distribution to make integration more reliable by "multilayer integration".
    """
    if mode is None:
        return torch.stack(hs_list, dim=0)
    out, L = [], len(hs_list)
    for i, h in enumerate(hs_list):
        skip = (skip_last_ln_post and i == L-1 and mode in ("ln_post", "whisper_ln"))
        out.append(apply_norm(h, mode, ln_post=ln_post, skip_ln_post=skip))
    return torch.stack(out, dim=0)

def _model_tag_from_name(model_name: str):
    """Model name relabelled ( such as whisper-small whisper_small) but not used in this script."""
    tail = model_name.split("/")[-1]            # whisper-medium
    return tail.replace("whisper-", "whisper_") # whisper_medium

# ===================== 5) Whisper Models and ln_post =====================
# ===================== 5) Characteristic models and uniform layers =====================
if FEATURE_BACKBONE.lower() == "whisper":
    processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model = WhisperModel.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model.eval().to(DEVICE)
    LN_POST = model.encoder.layer_norm
    if TOP_LAYER_INDEX is None:
        TOP_LAYER_INDEX = int(model.config.encoder_layers)
    # Print ln_post summary of parameters once (optional)
    with torch.no_grad():
        w = LN_POST.weight.detach().cpu().float()
        b = LN_POST.bias.detach().cpu().float()
        print(f"[LN_POST] eps={LN_POST.eps}  elementwise_affine={LN_POST.elementwise_affine}")
        print(f"[LN_POST] weight mean={w.mean():.6f} std={w.std():.6f} min={w.min():.6f} max={w.max():.6f}")
        print(f"[LN_POST] bias   mean={b.mean():.6f} std={b.std():.6f} min={b.min():.6f} max={b.max():.6f}")
#===================================================
else:
    processor = AutoFeatureExtractor.from_pretrained(
        PRETRAINED_MODEL_NAME,
        local_files_only=LOCAL_FILES_ONLY,
    )

    if FEATURE_BACKBONE.lower() == "hubert":
        model = HubertModel.from_pretrained(
            PRETRAINED_MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
        )
    elif FEATURE_BACKBONE.lower() == "wavlm":
        model = WavLMModel.from_pretrained(
            PRETRAINED_MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
        )
    elif FEATURE_BACKBONE.lower() == "wav2vec2":
        model = Wav2Vec2Model.from_pretrained(
            PRETRAINED_MODEL_NAME,
            local_files_only=LOCAL_FILES_ONLY,
        )
    else:
        raise ValueError(f"Unsupported FEATURE_BACKBONE: {FEATURE_BACKBONE}")

    model.eval().to(DEVICE)
    LN_POST = None

    if TOP_LAYER_INDEX is None:
        TOP_LAYER_INDEX = int(model.config.num_hidden_layers)

    # ln_post, which is not a Whisper model, does not support Whisper; if wrong set-up, it returns automatically with a hint.
    if NORM_FOR_SINGLE in ("ln_post", "whisper_ln"):
        print(f"[WARN] {FEATURE_BACKBONE} ln_post, NORM_FOR_SINGLE automatically becomes layernorm.")
        NORM_FOR_SINGLE = "layernorm"
    if NORM_BEFORE_FUSE in ("ln_post", "whisper_ln"):
        print(f"[WARN] {FEATURE_BACKBONE} ln_post, NORM_BEFORE_FUSE automatically becomes layernorm.")
        NORM_BEFORE_FUSE = "layernorm"
#================================================

print(f"[MODEL] FEATURE_BACKBONE={FEATURE_BACKBONE}")
print(f"[MODEL] FEAT_MODEL_NAME={FEAT_MODEL_NAME}")
print(f"[MODEL] PRETRAINED_MODEL_NAME={PRETRAINED_MODEL_NAME}")
print(f"[MODEL] TOP_LAYER_INDEX={TOP_LAYER_INDEX}")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ===================== 6) Single-document sub-section extract =====================
@torch.inference_mode()
def get_segment_features(audio_file: str,
                         audio_file_index: int,
                         single_layer=None,
                         norm_single=None,
                         last_k=None,
                         drop_input_layer=True,  # Throws the input layer and only keeps the output of transformer layers.
                         layer_f1=None,
                         f1_temp=12,
                         norm_mode=None,
                         segment_length=segment_length1,
                         sampling_rate=16000,
                         overlap_rate=overlap_rate1):
    import numpy as _np
    y, _ = librosa.load(audio_file, sr=sampling_rate, mono=True, dtype=_np.float32) # Re-sampling audio to 16k with librosa.load
    # The paper pre-processes include the re-allocation of volume: extracting wave sizes into [-1,1] stable features.
    if ENABLE_AUDIO_NORMALIZE:
        peak = float(_np.max(_np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-8:
            y = y / peak
    step_size = int(segment_length * (1 - overlap_rate))  # For example: 30s, 30%, overlap, ->, step=21s

    # The original logic would lead to a complete skipping of audio below the overlap threshold:
    #   SEGMENT_SEC=30, overlap_rate=0.3 , the threshold is 9 seconds.
    #   If len(y) < 9 seconds, total_segments=0.
    #
    # Part of the CTD mission for PROCESS-2 may be short.
    # In order to avoid the complete loss of short audio, at least 1 segment is generated as long as the audio is not empty.
    if len(y) == 0:
        total_segments = 0
    else:
        total_segments = int(len(y) / step_size) + (
            1 if len(y) % step_size > overlap_rate * segment_length else 0
        )
        total_segments = max(total_segments, 1)
        
    seg_features, seg_indices = [], []
    warned_double_ln_top = False

    for i in range(total_segments):
        start, end = i * step_size, i * step_size + segment_length
        segment = y[start:end]
        if len(segment) < segment_length and i > 0:
            # Aligns with Whisper: Only forward alignment of non-first paragraphs (HubERT/WavLM/Whisper unified policy).
            padding_needed = segment_length - len(segment)
            prev_end = start + segment_length - step_size
            padding_start = max(0, prev_end - padding_needed)
            padding_values = y[padding_start:prev_end]
            segment = _np.concatenate((segment, padding_values))

        if FEATURE_BACKBONE.lower() == "whisper":
            inp = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
            input_features = inp.input_features.to(DEVICE)   # [1,80,Tmels]
            enc_out = model.encoder(input_features=input_features, output_hidden_states=True, return_dict=True)
            hs_all = list(enc_out.hidden_states)
            hs = hs_all[1:] if (drop_input_layer and len(hs_all) >= 2) else hs_all
            # Whether the top level is post-LN (Whisper for specific purposes).
            try:
                already_post = torch.allclose(hs[-1], enc_out.last_hidden_state, atol=1e-5, rtol=1e-5)
            except Exception:
                already_post = True
        else:
            # HuBERT/WavLM Forward: extract hidden_states as a high-level acoustic expression.
            inp = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            enc_out = model(**inp, output_hidden_states=True, return_dict=True)
            hs_all = list(enc_out.hidden_states)
            hs = hs_all[1:] if (drop_input_layer and len(hs_all) >= 2) else hs_all
            # HuBERT/WavLM There is no Whisper post-LN judgement, fixed to False.
            already_post = False



        if single_layer is not None:
            """The code removes the hidden state of the specified layer from the single layer.
            Then press norm_single to normalize."""
            assert 1 <= int(single_layer) <= len(hs), f"SINGLE_LAYER={single_layer} Out of range"
            is_top = (int(single_layer) == len(hs))
            if (not warned_double_ln_top and is_top and already_post and norm_single in ("ln_post", "whisper_ln")):
                print(f"[WARN] {Path(audio_file).name}: The top layer has been post-LN and the external level has been set ln_post - a second ln_post has been skipped.")
                warned_double_ln_top = True
            h = hs[int(single_layer) - 1]
            is_top = (int(single_layer) == len(hs))
            skip_ln = (norm_single in ("ln_post", "whisper_ln")) and (is_top or int(single_layer) == TOP_LAYER_INDEX)
            fused = apply_norm(h, norm_single, ln_post=LN_POST, skip_ln_post=skip_ln)

        else:
            """It means you don't specify a single layer, you go on a multi-layer integration route:
            If layer_f1, is given, the weight is weighted by F1; otherwise the last K layer is taken by last_k for mean integration."""
            if layer_f1 is not None:
                # Weighted integration after statization
                # layer_f1 You are required to evaluate the F1 of each layer offline, and then pass on a list of F1 values of each layer in order of order.
                # The length must be equal to len(hs) (transformer layer). The order corresponds to hs (level 1 to L layer).
                assert len(layer_f1) == len(hs), "layer_f1 Length to match layer number"
                if (not warned_double_ln_top and norm_mode in ("ln_post", "whisper_ln") and already_post):
                    print(f"[WARN] {Path(audio_file).name}: Combining inln_post+top has post-LN -- only for non-tops in in_post.")
                    warned_double_ln_top = True
                H = stack_with_norm(hs, norm_mode, ln_post=LN_POST, skip_last_ln_post=already_post)
                w_np = weights_from_f1(layer_f1, temp=f1_temp)
                w = torch.tensor(w_np, dtype=H.dtype, device=H.device).view(-1,1,1,1)
                fused = (w * H).sum(dim=0)
            else:
                assert last_k is not None and 1 <= last_k <= len(hs), "last_k Invalid"
                selected = hs[-last_k:]
                if (not warned_double_ln_top and norm_mode in ("ln_post", "whisper_ln") and already_post):
                    print(f"[WARN] {Path(audio_file).name}: After K merges ln_post + the top level has been post-LN -- skipping the top level ln_post.")
                    warned_double_ln_top = True
                # Original: skip_last_int_post=already_post
                skip_last = already_post or (len(hs) == TOP_LAYER_INDEX)  # Top = 12 = skips last layer of In_post
                H = stack_with_norm(selected, norm_mode, ln_post=LN_POST, skip_last_ln_post=skip_last)
                fused = H[0] if last_k == 1 else H.mean(dim=0)

        fused = fused.squeeze(0).detach().cpu()  # [T,C],fused The original shape is [1, T, C] (batch=1). squeeze(0) removes the batch dimension [T, C] to facilitate subsequent stacking.
        seg_features.append(fused) # Can not delete folder: %s: No such folder
        seg_indices.append((audio_file_index, i)) # The record "This feature comes from which audio file, the parts of the audio."
    return seg_features, seg_indices

# ===================== 6.1) MMSE Label Tool Function =====================
def _pick_first_existing_column(df: pd.DataFrame, candidates):
    """
    The first real listing from the list of candidates was found in DataFrame.
    Use: The different data sets are not completely identically named (e.g. adressfname / addressfname / ID ),
    So here's a unified setup to avoid having a lot of if-else.
    """
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"No candidate fields found {candidates},is currently listed as:{list(df.columns)}")


def _safe_to_float(x):
    """
    Convert MMSE text security to float.
    - Empty string / NA/ NAN/ Non-> returns None
    - The rest of the parsable numbers -> returns float
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

def _extract_pitt_subject_visit(raw_name):
    """
    Parsing from Pitt filename:
        002-0c-en.wav -> ("2", 0)
        013-4c-en.wav -> ("13", 4)
        001-0.cha     -> ("1", 0)

    Return:
        (subject_key, visit_index)
    If not able to parse, return to (None, None)
    """
    if raw_name is None or pd.isna(raw_name):
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
        0 -> mms      (baseline)
        1 -> mmse2
        2 -> mmse3
        3 -> mmse4
        4 -> mmse5
        5 -> mmse6
        6 -> mmse7
    """
    visit_index = int(visit_index)
    if visit_index < 0:
        raise ValueError(f"Illegal visit_index={visit_index}")
    if visit_index == 0:
        return "mms"
    return f"mmse{visit_index + 1}"

def _normalize_mmse_key(dataset_key: str, raw_id):
    """
    Standardize the ID and stem audio file names in metadata to facilitate dictionary matching.

    PROCESS-2 Features:
    - meta-info.csv The id in the id is the participant level:
        PROCESS-2_rec__001
    - flat Audio stem in the directory is task level ID:
        PROCESS-2_rec__001__SFT
        PROCESS-2_rec__001__PFT
        PROCESS-2_rec__001__CTD
    It is therefore necessary to remove the end of __SFT / __PFT / __CTD and to return to the participant ID.
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
        return str(int(m.group()))

    if dataset_key_norm == "process-2":
        s = re.sub(r"__(SFT|PFT|CTD)$", "", s, flags=re.IGNORECASE)
        return s.lower()

    return s.lower()


def _build_map_from_rows(df: pd.DataFrame, dataset_key: str, id_col: str, mmse_col: str):
    """
    Construct {modify ID: mmse} map from DataFrame (ID, MMSE)
    If MMSE is missing, skip automatically; the following audio matches segment are discarded.
    """
    mmse_map = {}
    for _, row in df.iterrows():
        key = _normalize_mmse_key(dataset_key, row[id_col])
        val = _safe_to_float(row[mmse_col])
        if key is None or val is None:
            continue
        mmse_map[key] = float(val)
    return mmse_map


def _read_adress_is2020_mmse(cfg: dict):
    """
    Parsing txt metadata from ADReSS-IS2020.
    Formatting characteristics:
    - Semicolon Separation
    - train_cc / train_cd One each.
    - test_all One for common use at test_cc / test_cd
    """
    def _read_txt(txt_path: str):
        mmse_map = {}
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

def _read_process2_mmse(cfg: dict):
    """
    Parsing meta-info.csv of PROCESS-2, building MMSE maps of four subsets.

    meta-info.csv Need to include:
        IDs
        diagnosis
        MMSE
        Split

    Second Catalogue Map:
        HC -> cc
        MCI / Dementia -> cd

    Return:
        {
            "train_cc": {participant_id: mmse, ...},
            "train_cd": {participant_id: mmse, ...},
            "test_cc":  {participant_id: mmse, ...},
            "test_cd":  {participant_id: mmse, ...},
        }
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


def _read_adresso21_mmse(cfg: dict):
    """
    Resolve AMSE csv of ADReSSo21.
    The training collection documents also contain:
    - adressfname: Audio ID
    - mmse: MMSE Scores
    - dx: ad / cn
    The test set file contains:
    - ID
    - MMSE
    """
    train_df = pd.read_csv(cfg["train_all"])
    test_df = pd.read_csv(cfg["test_all"])

    train_id_col = _pick_first_existing_column(train_df, ["adressfname", "addressfname", "id"])
    train_mmse_col = _pick_first_existing_column(train_df, ["mmse", "MMSE"])
    train_dx_col = _pick_first_existing_column(train_df, ["dx", "label", "group"])

    test_id_col = _pick_first_existing_column(test_df, ["ID", "id", "adressfname", "addressfname"])
    test_mmse_col = _pick_first_existing_column(test_df, ["MMSE", "mmse"])

    # The training set is divided by dx into health matches/students; test does not need to share anymore, cc/cd shares
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




def _read_pitt_mmse(cfg: dict):
    """
    Parsing the Excel matrix for Pitt, building the visit-level MMSE map:
        (subject_key, visit_index) -> target_mmse

    readme Reciprocal:
        0 -> mms
        1 -> mmse2
        2 -> mmse3
        3 -> mmse4
        4 -> mmse5
        5 -> mmse6
        6 -> mmse7
    """
    df = pd.read_excel(
        cfg["all_meta"],
        sheet_name=cfg.get("sheet_name", "data"),
        header=int(cfg.get("header_row", 2)),
    )

    id_col = _pick_first_existing_column(df, [cfg.get("id_col", "id"), "id"])

    # Unified List Map, Compatible Case Difference
    col_map = {str(c).strip().lower(): c for c in df.columns}

    all_map = {}

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

        # Support 0-6, corresponding to mms / mmse2 ~ mmse7
        for visit_index in range(0, 7):
            col_name = _pitt_visit_to_mmse_col(visit_index).lower()
            if col_name not in col_map:
                continue

            mmse_val = _safe_to_float(row[col_map[col_name]])
            if mmse_val is None:
                continue

            all_map[(subject_key, int(visit_index))] = float(mmse_val)

    # Pitt No separate train/test MMSE file, four subsets shared with the same visit-level map
    return {
        "train_cc": all_map,
        "train_cd": all_map,
        "test_cc": all_map,
        "test_cd": all_map,
    }


def _read_adress_m_mmse(cfg: dict):
    """
    parsing the ADRSS-M roottruth csv.
    The training collection documents include:
    - adressfname
    - dx
    - mmse
    The test set file contains:
    - addressfname(Note that this is usually a "addressfname," one more s)
    - dx
    - mmse
    """
    train_df = pd.read_csv(cfg["train_all"])
    test_df = pd.read_csv(cfg["test_all"])

    train_id_col = _pick_first_existing_column(train_df, ["adressfname", "addressfname", "id", "ID"])
    train_mmse_col = _pick_first_existing_column(train_df, ["mmse", "MMSE"])
    train_dx_col = _pick_first_existing_column(train_df, ["dx", "label", "group"])

    test_id_col = _pick_first_existing_column(test_df, ["addressfname", "adressfname", "id", "ID"])
    test_mmse_col = _pick_first_existing_column(test_df, ["mmse", "MMSE"])

    dx_norm = train_df[train_dx_col].astype(str).str.strip().str.lower()
    train_cc_df = train_df[dx_norm.isin(["control", "cn", "cc"])]
    train_cd_df = train_df[dx_norm.isin(["probablead", "ad", "cd"])]

    train_cc_map = _build_map_from_rows(train_cc_df, "ADReSS_m", train_id_col, train_mmse_col)
    train_cd_map = _build_map_from_rows(train_cd_df, "ADReSS_m", train_id_col, train_mmse_col)
    test_map = _build_map_from_rows(test_df, "ADReSS_m", test_id_col, test_mmse_col)

    return {
        "train_cc": train_cc_map,
        "train_cd": train_cd_map,
        "test_cc": test_map,
        "test_cd": test_map,
    }


def load_mmse_maps_for_dataset(dataset_key: str):
    """
    Reads the MMSE metadata file of the specified data set and returns the dictionary divided by subset.

    returns to a fixed format as:
    {
        "train_cc": {"S002": 30.0, ...},
        "train_cd": {"S079": 11.0, ...},
        "test_cc":  {"S160": 28.0, ...},
        "test_cd":  {"S160": 28.0, ...},
    }

    This makes the logic of generating the next level label need not concern "the bottom file is txt / csv / xlsx".
    """
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"TARGET_KIND='mmse' , the data set is not configured {dataset_key} MSME_META_PATHS")

    cfg = MMSE_META_PATHS[dataset_key]

    if dataset_key == "PROCESS-2":
        mmse_maps = _read_process2_mmse(cfg)
    elif dataset_key == "ADReSS-IS2020":
        mmse_maps = _read_adress_is2020_mmse(cfg)
    elif dataset_key == "ADReSSo21":
        mmse_maps = _read_adresso21_mmse(cfg)
    elif dataset_key == "Pitt":
        mmse_maps = _read_pitt_mmse(cfg)
    elif dataset_key == "ADReSS_m":
        mmse_maps = _read_adress_m_mmse(cfg)
    else:
        raise ValueError(f"Unrealized datasets {dataset_key} The MMSE parsing logic")

    print(
        f"[MMSE] {dataset_key} Map load complete:"
        f"train_cc={len(mmse_maps['train_cc'])}, "
        f"train_cd={len(mmse_maps['train_cd'])}, "
        f"test_cc={len(mmse_maps['test_cc'])}, "
        f"test_cd={len(mmse_maps['test_cd'])}"
    )
    return mmse_maps

def _make_missing_mmse_record(dataset_key: str, subset_name: str, wav_path: str):
    """
    To construct a record for the missing audio of MMSE, the subsequent export of CSV is consolidated.
    In the case of Pitt and has become visit-level MMSE, an additional record will be issued in the form of subject_id / visit_index / expected_mmse_col.
    """
    record = {
        "dataset_key": str(dataset_key),
        "subset_name": str(subset_name),
        "wav_name": Path(wav_path).name,
        "wav_path": str(wav_path),
        "reason": "missing_mmse",
    }

    if str(dataset_key).strip().lower() == "pitt":
        try:
            subject_key, visit_index = _extract_pitt_subject_visit(Path(wav_path).stem)
            record["subject_id"] = subject_key
            record["visit_index"] = visit_index
            if subject_key is not None and visit_index is not None:
                mmse_col = _pitt_visit_to_mmse_col(int(visit_index))
                record["expected_mmse_col"] = mmse_col
                record["reason"] = f"missing_{mmse_col}"
            else:
                record["expected_mmse_col"] = None
        except Exception:
            record["subject_id"] = None
            record["visit_index"] = None
            record["expected_mmse_col"] = None

    return record


def save_missing_mmse_records_csv(
    missing_records,
    save_root: str,
    feature_prefix: str,
    single_layer=None,
    last_k=None,
    ov_str="0.3",
    sec_str="30s",
):
    """
    Export the complete missing MMSE audio list to CSV.
    """
    if missing_records is None or len(missing_records) == 0:
        print("[MMSE] There is no missing MMSE audio for this time. There is no need to export CSV.")
        return None

    os.makedirs(save_root, exist_ok=True)

    if single_layer is not None:
        csv_name = f"missing_mmse_{feature_prefix}_single{single_layer}_{ov_str}_{sec_str}.csv"
    else:
        csv_name = f"missing_mmse_{feature_prefix}_last{last_k}_{ov_str}_{sec_str}.csv"

    csv_path = os.path.join(save_root, csv_name)

    df = pd.DataFrame(missing_records)

    # Decomposition: Only one path remains under the same subset
    dedup_cols = [c for c in ["subset_name", "wav_path"] if c in df.columns]
    if len(dedup_cols) > 0:
        df = df.drop_duplicates(subset=dedup_cols, keep="first")

    # Try to keep the usual lines in front.
    front_cols = [
        "dataset_key",
        "subset_name",
        "wav_name",
        "wav_path",
        "reason",
        "subject_id",
        "visit_index",
        "expected_mmse_col",
    ]
    cols = [c for c in front_cols if c in df.columns] + [c for c in df.columns if c not in front_cols]
    df = df[cols]

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[MMSE] Missing MMSE full list of audio saved to: {csv_path}")
    print(f"[MMSE] Total number of missing audio (after re-entry): {len(df)}")
    return csv_path


def build_mmse_labels_for_segments(audio_path: str, seg_indices, mmse_map: dict, subset_name: str, dataset_key: str):
    """
    In accordance with the sorting rules of extract_dir, the segment level index is remixed to audio ID,
    Copy this audio MMSE score to every message it has.

    Return:
    - labels_tensor
    - keep_mask
    - missing_records: Complete audio records missing from MMSE (for export CSV)
    """
    files = list_wav_files_for_dataset(audio_path, dataset_key)

    audio_paths = [str(fp) for fp in files]
    audio_ids = [Path(fp).stem for fp in files]

    seg_indices = np.asarray(seg_indices, dtype=np.int64)
    keep_mask = np.ones(len(seg_indices), dtype=bool)
    labels = np.zeros(len(seg_indices), dtype=np.float32)

    missing_records = []
    missing_audio_names = []

    for i, (audio_idx, seg_idx) in enumerate(seg_indices):
        audio_idx = int(audio_idx)
        raw_sid = audio_ids[audio_idx]
        wav_path = audio_paths[audio_idx]

        # ------------------------------
        # Pitt:If you have changed tovisit-level MMSE, press (subject, visit)
        # Other data sets: maintaining the logic
        # ------------------------------
        if str(dataset_key).strip().lower() == "pitt" and "_extract_pitt_subject_visit" in globals():
            subject_key, visit_index = _extract_pitt_subject_visit(raw_sid)
            if subject_key is None or visit_index is None:
                keep_mask[i] = False
                rec = _make_missing_mmse_record(dataset_key, subset_name, wav_path)
                missing_records.append(rec)
                missing_audio_names.append(f"{raw_sid} [unparsed_visit]")
                continue

            key = (subject_key, int(visit_index))
            if key not in mmse_map:
                keep_mask[i] = False
                rec = _make_missing_mmse_record(dataset_key, subset_name, wav_path)
                missing_records.append(rec)
                reason = rec.get("reason", "missing_mmse")
                missing_audio_names.append(f"{raw_sid} [{reason}]")
            else:
                labels[i] = float(mmse_map[key])

        else:
            sid = _normalize_mmse_key(dataset_key, raw_sid)

            if sid not in mmse_map:
                keep_mask[i] = False
                rec = _make_missing_mmse_record(dataset_key, subset_name, wav_path)
                missing_records.append(rec)
                missing_audio_names.append(raw_sid)
            else:
                labels[i] = float(mmse_map[sid])

    # The summary is still printed in the log, but CSV saves a full list
    missing_audio_names = sorted(set(missing_audio_names), key=natural_key)
    if len(missing_audio_names) > 0:
        print(
            f"[MMSE] {subset_name} Yes, I do. {len(missing_audio_names)} The audio is missing MMSE and will automatically be discarded:"
            f"{missing_audio_names[:10]}"
        )

    labels_tensor = torch.tensor(labels[keep_mask], dtype=torch.float32)
    return labels_tensor, keep_mask, missing_records


# ===================== 7) Directory through (with progress bar) =====================

def extract_dir(audio_path: str, dataset_key: str = DATASET_KEY, **kwargs):
    files = list_wav_files_for_dataset(audio_path, dataset_key)

    if str(dataset_key).strip().lower() == "process-2":
        print(f"[DIR] {audio_path}")
        print(f"[PROCESS-2] Read only CTD wav, number of documents: {len(files)}")

    features, indices = [], []
    for idx, fp in enumerate(tqdm(files, desc=f"[DIR] {audio_path}", ncols=90)):
        seg_feats, seg_idx = get_segment_features(fp, idx, **kwargs)
        features.extend(seg_feats)
        indices.extend(seg_idx)

    return features, indices

# ===================== 8) Assembly and preservation (strict filename format) =====================
OV_STR  = "0.3"
SEC_STR = f"{SEGMENT_SEC}s"

# Fills a set of [T,C] features to the same time length and then stacks them to avoid torch.stack reporting errors due to T inconsistencies.
def stack_with_time_padding(feature_list, target_t=None, pad_value=0.0):
    # The empty list returns the empty load directly to prevent the subsequent process from collapsing.
    if len(feature_list) == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)
    # Check every element is two-dimensional [T,C].
    for i, feat in enumerate(feature_list):
        if feat.ndim != 2:
            raise ValueError(f"feature_list[{i}] Not a 2-dimensional length. Current shape ={tuple(feat.shape)}")
    # Features must be consistent C.
    feat_dim = int(feature_list[0].shape[1])
    for i, feat in enumerate(feature_list):
        if int(feat.shape[1]) != feat_dim:
            raise ValueError(f"feature_list[{i}] _Other Organiser{feat.shape[1]} != {feat_dim}")
    # Target time length: If not specified, use the current list to T maximum.
    if target_t is None:
        target_t = max(int(feat.shape[0]) for feat in feature_list)
    # Completion (right-side zero) or cut-off to ensure that all become [target_t, C].
    padded = []
    for feat in feature_list:
        t = int(feat.shape[0])
        if t < target_t:
            pad = torch.full((target_t - t, feat_dim), float(pad_value), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)
        elif t > target_t:
            feat = feat[:target_t]
        padded.append(feat)
    # Stack as [N, target_t, C].
    return torch.stack(padded, dim=0)

def run_and_save_one_config(*, dataset_paths, single_layer=None, last_k=None,
                            norm_single=None, norm_mode=None, drop_input_layer=True,
                            layer_f1=None, f1_temp=12):
    # Take four subsets.
    feat_cc_tr, idx_cc_tr = extract_dir(dataset_paths["train_cc"],
                                        single_layer=single_layer, norm_single=norm_single,
                                        last_k=last_k, drop_input_layer=drop_input_layer,
                                        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode)
    feat_cd_tr, idx_cd_tr = extract_dir(dataset_paths["train_cd"],
                                        single_layer=single_layer, norm_single=norm_single,
                                        last_k=last_k, drop_input_layer=drop_input_layer,
                                        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode)
    feat_cc_te, idx_cc_te = extract_dir(dataset_paths["test_cc"],
                                        single_layer=single_layer, norm_single=norm_single,
                                        last_k=last_k, drop_input_layer=drop_input_layer,
                                        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode)
    feat_cd_te, idx_cd_te = extract_dir(dataset_paths["test_cd"],
                                        single_layer=single_layer, norm_single=norm_single,
                                        last_k=last_k, drop_input_layer=drop_input_layer,
                                        layer_f1=layer_f1, f1_temp=f1_temp, norm_mode=norm_mode)

    # First, it is consolidated over the maximum global time frame and then stacked to avoid stack errors resulting from the length of different audio segments.
    all_feats = feat_cc_tr + feat_cd_tr + feat_cc_te + feat_cd_te
    global_t = max(int(feat.shape[0]) for feat in all_feats)
    print(f"[PAD] Unified time length target_t={global_t}")
    features_cc      = stack_with_time_padding(feat_cc_tr, target_t=global_t)
    features_cd      = stack_with_time_padding(feat_cd_tr, target_t=global_t)
    features_cc_test = stack_with_time_padding(feat_cc_te, target_t=global_t)
    features_cd_test = stack_with_time_padding(feat_cd_te, target_t=global_t)

#======================================================================
    # Convert indexes into numbery, so that the filter/ remap later is uniform
    idx_cc_tr = np.array(idx_cc_tr, dtype=np.int64)
    idx_cd_tr = np.array(idx_cd_tr, dtype=np.int64)
    idx_cc_te = np.array(idx_cc_te, dtype=np.int64)
    idx_cd_te = np.array(idx_cd_te, dtype=np.int64)

    # ===================== Tab Generation: Category II / MMSE Second = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    if TARGET_KIND == "mmse":
        mmse_maps = load_mmse_maps_for_dataset(DATASET_KEY)

        # Generates MMSE tags for each segment; if sound does not match MMSE, the segment discards
        labels_cc, keep_cc, missing_cc = build_mmse_labels_for_segments(
            dataset_paths["train_cc"], idx_cc_tr, mmse_maps["train_cc"], "train_cc", DATASET_KEY
        )
        labels_cd, keep_cd, missing_cd = build_mmse_labels_for_segments(
            dataset_paths["train_cd"], idx_cd_tr, mmse_maps["train_cd"], "train_cd", DATASET_KEY
        )
        labels_cc_test, keep_cc_te, missing_cc_test = build_mmse_labels_for_segments(
            dataset_paths["test_cc"], idx_cc_te, mmse_maps["test_cc"], "test_cc", DATASET_KEY
        )
        labels_cd_test, keep_cd_te, missing_cd_test = build_mmse_labels_for_segments(
            dataset_paths["test_cd"], idx_cd_te, mmse_maps["test_cd"], "test_cd", DATASET_KEY
        )

        all_missing_mmse_records = (
            list(missing_cc)
            + list(missing_cd)
            + list(missing_cc_test)
            + list(missing_cd_test)
        )
        # N.B. If some audio does not have MMSE (e.g. S001=NA ), the corresponding features and indexes must be deleted simultaneously
        features_cc      = features_cc[torch.from_numpy(keep_cc)]
        features_cd      = features_cd[torch.from_numpy(keep_cd)]
        features_cc_test = features_cc_test[torch.from_numpy(keep_cc_te)]
        features_cd_test = features_cd_test[torch.from_numpy(keep_cd_te)]

        idx_cc_tr = idx_cc_tr[keep_cc]
        idx_cd_tr = idx_cd_tr[keep_cd]
        idx_cc_te = idx_cc_te[keep_cc_te]
        idx_cd_te = idx_cd_te[keep_cd_te]

    else:
        # Keep the original AD Second Classification Tab
        labels_cc      = torch.zeros(features_cc.shape[0], dtype=torch.float32)
        labels_cd      = torch.ones(features_cd.shape[0], dtype=torch.float32)
        labels_cc_test = torch.zeros(features_cc_test.shape[0], dtype=torch.float32)
        labels_cd_test = torch.ones(features_cd_test.shape[0], dtype=torch.float32)

        all_missing_mmse_records = []

    # train / test Sting
    features1 = torch.cat([features_cc, features_cd], dim=0)
    features2 = torch.cat([features_cc_test, features_cd_test], dim=0)
    labels1   = torch.cat([labels_cc, labels_cd], dim=0)
    labels2   = torch.cat([labels_cc_test, labels_cd_test], dim=0)

    # ===================== indices Merge to ensure that the global audio index of cc/cd does not overlap =====================
    if len(idx_cc_tr) > 0 and len(idx_cd_tr) > 0:
        max_idx = idx_cc_tr[:, 0].max()
        idx_cd_tr[:, 0] += (max_idx + 1)
    indices_train = np.vstack([idx_cc_tr, idx_cd_tr]) if len(idx_cd_tr) > 0 else idx_cc_tr

    if len(idx_cc_te) > 0 and len(idx_cd_te) > 0:
        max_idx = idx_cc_te[:, 0].max()
        idx_cd_te[:, 0] += (max_idx + 1)
    indices_test = np.vstack([idx_cc_te, idx_cd_te]) if len(idx_cd_te) > 0 else idx_cc_te
#===================================================


    # -- Save (strictly conforming to format) -
    # {layer} is a single layer number, {K} is the integrated post-K layer, {overlap} from OV_STR=0.3, {sec} from SEC_STR=30s, 
    # So the title of the document shows at first glance: single layer or integration, parameters, training / tests.
    # 
    # -- Creates a separate save directory for the current configuration --
    if single_layer is not None:
        SINGLE_LAYER = str(single_layer)
        SAVE_ROOT = os.path.join(
            BASE_SAVE_ROOT,
            DATASET_KEY,
            FEAT_MODEL_NAME,
            f"single{SINGLE_LAYER}",
        )
    else:
        # Compatibility of the integration mode after retaining K
        SAVE_ROOT = os.path.join(
            BASE_SAVE_ROOT,
            DATASET_KEY,
            FEAT_MODEL_NAME,
            f"last{last_k}",
        )

    os.makedirs(SAVE_ROOT, exist_ok=True)

    # -- Save (strictly conforming to format) -
    if single_layer is not None:
        file_name  = f"feature_{FEATURE_PREFIX}_single{single_layer}_{OV_STR}_{SEC_STR}_train.pt"
        file_name2 = f"feature_{FEATURE_PREFIX}_single{single_layer}_{OV_STR}_{SEC_STR}_test.pt"
        file_name3 = f"labels1_{FEATURE_PREFIX}_single{single_layer}_{OV_STR}_{SEC_STR}_train.pt"
        file_name4 = f"labels2_{FEATURE_PREFIX}_single{single_layer}_{OV_STR}_{SEC_STR}_test.pt"
        file_name5 = f"indices_{FEATURE_PREFIX}_single{single_layer}_{OV_STR}_{SEC_STR}_train.pt"
        file_name6 = f"indices_{FEATURE_PREFIX}_single{single_layer}_{OV_STR}_{SEC_STR}_test.pt"
    else:
        file_name  = f"feature_{FEATURE_PREFIX}_last{last_k}_{OV_STR}_{SEC_STR}_train.pt"
        file_name2 = f"feature_{FEATURE_PREFIX}_last{last_k}_{OV_STR}_{SEC_STR}_test.pt"
        file_name3 = f"labels1_{FEATURE_PREFIX}_last{last_k}_{OV_STR}_{SEC_STR}_train.pt"
        file_name4 = f"labels2_{FEATURE_PREFIX}_last{last_k}_{OV_STR}_{SEC_STR}_test.pt"
        file_name5 = f"indices_{FEATURE_PREFIX}_last{last_k}_{OV_STR}_{SEC_STR}_train.pt"
        file_name6 = f"indices_{FEATURE_PREFIX}_last{last_k}_{OV_STR}_{SEC_STR}_test.pt"

    full_path  = os.path.join(SAVE_ROOT, file_name)
    full_path2 = os.path.join(SAVE_ROOT, file_name2)
    full_path3 = os.path.join(SAVE_ROOT, file_name3)
    full_path4 = os.path.join(SAVE_ROOT, file_name4)
    full_path5 = os.path.join(SAVE_ROOT, file_name5)
    full_path6 = os.path.join(SAVE_ROOT, file_name6)

    torch.save(features1, full_path)
    torch.save(features2, full_path2)
    torch.save(labels1,   full_path3)
    torch.save(labels2,   full_path4)
    torch.save(indices_train, full_path5)
    torch.save(indices_test,  full_path6)

    # -- Export the missing full audio list of MMSE to CSV if the MMSE task is concerned.
    if TARGET_KIND == "mmse":
        save_missing_mmse_records_csv(
            missing_records=all_missing_mmse_records,
            save_root=SAVE_ROOT,
            feature_prefix=FEATURE_PREFIX,
            single_layer=single_layer,
            last_k=last_k,
            ov_str=OV_STR,
            sec_str=SEC_STR,
        )

    print(f"[SAVE] -> {SAVE_ROOT}")
    print("  -", full_path)
    print("  -", full_path2)
    print("  -", full_path3)
    print("  -", full_path4)
    print("  -", full_path5)
    print("  -", full_path6)

# ===================== 9) Implementation: continuously processing ===================== by list
paths = DATASETS[DATASET_KEY]
print(f"[DATASET] {DATASET_KEY}")
for k,v in paths.items():
    print(f"  {k}: {v}")

if len(SINGLE_LAYER_LIST) > 0:
    print(f"\n[RUN] Single batch:{SINGLE_LAYER_LIST},NORM_FOR_SINGLE={NORM_FOR_SINGLE}")
    for lyr in SINGLE_LAYER_LIST:
        print(f"\n>>> [Single Layer] layer={lyr}")
        run_and_save_one_config(dataset_paths=paths,
                                single_layer=int(lyr),
                                last_k=None,
                                norm_single=NORM_FOR_SINGLE,
                                norm_mode=None,
                                drop_input_layer=DROP_INPUT_LAYER,
                                layer_f1=None,
                                f1_temp=F1_TEMP)
elif len(LAST_K_LIST) > 0:
    print(f"\n[RUN] After K-level integration batch processing:{LAST_K_LIST},NORM_BEFORE_FUSE={NORM_BEFORE_FUSE}")
    for k in LAST_K_LIST:
        print(f"\n>>> [Last-K Fuse] K={k}")
        run_and_save_one_config(dataset_paths=paths,
                                single_layer=None,
                                last_k=int(k),
                                norm_single=None,
                                norm_mode=NORM_BEFORE_FUSE,
                                drop_input_layer=DROP_INPUT_LAYER,
                                layer_f1=LAYER_F1,
                                f1_temp=F1_TEMP)
else:
    raise ValueError("Please provide a list at least in SINGLE_LAYER_LIST or LAST_K_LIST.")


# %% cell 2
import os, sys
print("exe:", sys.executable)
print("pid:", os.getpid())


# %% cell 3

# python -m pip install -U huggingface_hub filelock

# python - <<'PY'
# from huggingface_hub import snapshot_download
# snapshot_download(
#     repo_id="facebook/hubert-base-ls960",
#     local_dir="/home/user/whisper_lru_3/hubert_xlarge_ll60k",
#     local_dir_use_symlinks=False,
# )
# print("done")
# PY

