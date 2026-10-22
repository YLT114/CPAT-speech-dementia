# -*- coding: utf-8 -*-
# Converted from Jupyter notebook: audio_cut_xr(2).ipynb


# %% cell 1
# ===================== 0) 环境与随机源（必须在 import torch 前设置） =====================
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

# ===================== 1) 其余导入 =====================
import re
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
import librosa
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperModel, AutoFeatureExtractor, HubertModel, WavLMModel, Wav2Vec2Model

# ===================== 2) 控制区：在这里改参数 =====================
# —— 一键切换特征骨干（只改这一行）——
# 可选: "hubert" / "wav2vec2"/ "wavlm"/"whisper" /
FEATURE_BACKBONE = "whisper"
FEAT_MODEL_NAME = "whisper_small"

"""
conda activate audiojax_gpu
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v3/frozen.py
"""

# —— 各骨干统一配置表（模型路径/模型名 + 输出前缀）——
# 说明：
# 1) model_name 可以写 HuggingFace 名称，也可以写本地模型目录。
# 2) feature_prefix 会写入输出文件名，用于后续 space_state_models.ipynb 自动匹配。
BACKBONE_CONFIG = {
    "hubert": {
        # 英文数据建议使用英文 HuBERT 预训练模型或对应本地目录。
        "model_name": "/home/juanyu/whisper_lru_3/hubert_base_ls960",
        "feature_prefix": "hubert",
    },
    "wav2vec2": {
        "model_name": "/home/juanyu/whisper_lru_3/wav2vec2_base_960h",
        "feature_prefix": "wav2vec2",
    },  
    "wavlm": {
        "model_name": "/home/juanyu/whisper_lru_3/wavlm_base_plus",
        "feature_prefix": "wavlm",
    },
    "whisper": {
        "model_name": "/home/juanyu/whisper_lru_3/whisper_small",# whisper_small crisperwhisper whisper_medium
        "feature_prefix": "whisper",
    },
}

# —— 是否强制只用本地文件（离线环境建议 True）——
LOCAL_FILES_ONLY = True

# —— 规范化骨干名称并校验 ——
FEATURE_BACKBONE = str(FEATURE_BACKBONE).strip().lower()
if FEATURE_BACKBONE not in BACKBONE_CONFIG:
    raise ValueError(
        f"不支持的 FEATURE_BACKBONE: {FEATURE_BACKBONE}，可选: {list(BACKBONE_CONFIG.keys())}"
    )

# —— 根据骨干配置自动派生“真实加载模型路径/名称”和输出前缀 ——
PRETRAINED_MODEL_NAME = BACKBONE_CONFIG[FEATURE_BACKBONE]["model_name"]
FEATURE_PREFIX = BACKBONE_CONFIG[FEATURE_BACKBONE]["feature_prefix"]

# —— 音频归一化开关（论文预处理包含音量归一化）——
ENABLE_AUDIO_NORMALIZE = True


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# —— 数据集 ——（影响保存目录的最后一级）
DATASET_KEY = "PROCESS-2"             # 可选: "ADReSS-IS2020"/ "ADReSSo21" / "Pitt" / "PROCESS-2"

# —— 分段参数（与文件名中的 0.3_30s 对应） ——
sampling_rate = 16000 
SEGMENT_SEC   = 30 # 每段 30s ==================> NCMMSC2021只利用 30s 的长音频吗？ 是的
segment_length1 = SEGMENT_SEC * sampling_rate
overlap_rate1   = 0.3 # 30% 重叠
# —— 选择提取模式 —— 
# 1) 单层列表：一次性跑多层（优先，如果列表非空就按单层模式跑）
SINGLE_LAYER_LIST = [1,2,3,4,5,6,7,8,9,10,11,12] # 举例：填你要批量提取的层；留空 [] 则不跑单层模式
NORM_FOR_SINGLE   = "ln_post"    # None / "layernorm" / "std" / "ln_post" 
"""
# layernorm:
    调用 F.layer_norm(h, h.shape[-1:]),没有使用任何可学习参数(weight/bias 都是默认 None).
    它只是按当前特征自己的均值/方差做标准化。
    "纯统计归一化（无参数）"
# ln_post:
    调用的是 model.encoder.layer_norm,
    带有训练好的参数 (weight/bias + eps),是 Whisper 模型最终的 post-LN.
    "Whisper 训练出来的终层 LayerNorm(有参数、固定分布)"
# 用途上：
    layernorm 更通用，但分布不一定和 Whisper 顶层一致。
    ln_post 会把特征拉到 Whisper 顶层的标准分布，常用于对齐不同层。
"""
# 2) 后K层融合列表：当 SINGLE_LAYER_LIST 为空时生效；可一次跑多个 K
LAST_K_LIST       = []      # 举例：要批量融合的 K；若不需要融合，置为 []
DROP_INPUT_LAYER  = True
LAYER_F1          = None         # 若提供各层 F1 列表，则做加权融合；废案代码
F1_TEMP           = 12           # F1 权重的温度参数，控制加权融合时的平滑程度（越大越尖锐，越小越平滑）
NORM_BEFORE_FUSE  = "ln_post"    # None / "layernorm" / "std" / "ln_post"

# —— 顶层层号（None 表示从模型配置自动推断）——
TOP_LAYER_INDEX = None


# —— 保存根路径（严格遵从你的目录格式） ——
# whisper_small_1_12、whisper_medium_1_24、whisper_largev2_1_32、hubert_base_1_12
# # ADReSSo21、ADReSS2020、NCMMSC2021、Pitt
# hubert_base/large/xlarge = 1_12:9--10; 1_24:18--21;1_48:38--44

"""

tmux new -s audio_cut_xr_whisper_medium

tmux attach -t audio_cut_xr_small
tmux attach -t audio_cut_xr_medium
tmux attach -t audio_cut_xr_crispser

conda activate audiojax_gpu
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v3/frozen_extract/frozen.py
"""

# 若目录不存在，则自动创建
# 保存根目录：
# 单层模式下会进一步保存到
# /home/juanyu/Alzheimer_code_jax_mmse_speech_v3/feature_tensor/DATASET_KEY/FEAT_MODEL_NAME/{single_layer}/
BASE_SAVE_ROOT = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v3/feature_tensor"

# ===================== 2.1) 监督目标配置 =====================
# 可选：
# "diagnosis" -> 保持原来的 AD 二分类标签（cc=0, cd=1）
# "mmse"      -> 改为 MMSE 回归标签
TARGET_KIND = "mmse"

# 各数据集对应的 MMSE 元数据路径
# 说明：
# 1) 这里只新增 MMSE 元数据路径，不去改动你现有的音频目录 DATASETS 路径。
# 2) 不同数据集的 MMSE 文件格式不同，所以后面会按数据集分别解析。
# 3) 统一目标：最终都转成 {音频ID: mmse} 的字典，再复制到 segment 级标签。
MMSE_META_PATHS = {
    "PROCESS-2": {
        "meta": "/home/juanyu/Downloads/PROCESS-2/meta-info.csv",
    },
    "ADReSS-IS2020": {
        # 原始 ADReSS 2020：train 分成 cc/cd 两个 txt，test 共用一份结果文件
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/cc_meta_data.txt",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/cd_meta_data.txt",
        "test_all": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/ADReSS 2020 test result.txt",
    },
    "ADReSSo21": {
        # ADReSSo21：训练集 mmse 在 train csv 中，测试集 mmse 在 groundtruth csv 中
        "train_all": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/adresso-train-mmse-scores.csv",
        "test_all": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/test_results_task2_groundtruth.csv",
    },
    "Pitt": {
        # Pitt：使用总表做 visit-level MMSE 映射
        # readme 已说明：
        #   文件名里的 0 -> baseline
        #   1 -> 第 2 次访视
        #   2 -> 第 3 次访视
        #   ...
        # 因此这里后续会在 _read_pitt_mmse(...) 里自动映射：
        #   0 -> mms
        #   1 -> mmse2
        #   2 -> mmse3
        #   3 -> mmse4
        #   4 -> mmse5
        #   5 -> mmse6
        #   6 -> mmse7
        "all_meta": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/PItt-data(1).xlsx",
        "sheet_name": "data",
        "header_row": 2,
        "id_col": "id",
    },
    "ADReSS_m": {
        # ADReSS-M：训练/测试各有一份 groundtruth csv
        "train_all": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/training-groundtruth.csv",
        "test_all": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr-groundtruth.csv",
    },
}


# ===================== 3) 数据集路径注册表 =====================
"""AD_combine 就是把 train/AD 和 train/MCI 组合在一起了,也只使用了长音频部分
    测试集是用的带标签的
    cc:健康, cd:患病"""
DATASETS = {
    "PROCESS-2": {
        "train_cc": "/home/juanyu/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/juanyu/Downloads/PROCESS-2/train/cd",
        "test_cc" : "/home/juanyu/Downloads/PROCESS-2/test/cc",
        "test_cd" : "/home/juanyu/Downloads/PROCESS-2/test/cd",
    },
    "NCMMSC2021": {
        "train_cc": "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/HC",
        "train_cd": "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/train/AD_combine",
        "test_cc" : "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/HC",
        "test_cd" : "/home/juanyu/Downloads/dementiabank_extracted/NCMMSC2021_AD/AD_dataset_long/test_have_label/AD_combine",
    },
    "ADReSSo21": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc" : "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd" : "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc" : "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd" : "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "Pitt_1": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_test/train/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_test/train/cd",
        "test_cc" : '/home/juanyu/Downloads/dementiabank/Pitt_test/test/cc',
        "test_cd" : '/home/juanyu/Downloads/dementiabank/Pitt_test/test/cd',
    },
    "Pitt": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc" : '/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence',
        "test_cd" : '/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence',
    },
    "Pitt_val1": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/train/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/train/cd",
        "test_cc" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/test/cc',
        "test_cd" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_1/test/cd',
    },
    "Pitt_val2": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/train/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/train/cd",
        "test_cc" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/test/cc',
        "test_cd" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_2/test/cd',
    },
    "Pitt_val3": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/train/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/train/cd",
        "test_cc" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/test/cc',
        "test_cd" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_3/test/cd',
    },
    "Pitt_val4": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/train/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/train/cd",
        "test_cc" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/test/cc',
        "test_cd" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_4/test/cd',
    },
    "Pitt_val5": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/train/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/train/cd",
        "test_cc" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/test/cc',
        "test_cd" : '/home/juanyu/Downloads/dementiabank/Pitt_new/5_fold_dataset/fold_5/test/cd',
    },
    "ADReSS_m": {
    "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/train_wav/HC",
    "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/train_wav/AD",
    "test_cc" : '/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr/HC',
    "test_cd" : '/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr/AD',
},
}
print("DATASET_KEY =", repr(DATASET_KEY))
print("DATASETS keys =", list(DATASETS.keys()))
assert DATASET_KEY in DATASETS, f"未知 DATASET_KEY={DATASET_KEY}"



# ===================== 4) 工具函数 =====================
def natural_key(p: str):
    """保证文件名按自然顺序排序（数字优先）"""
    s = Path(p).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def list_wav_files_for_dataset(audio_path: str, dataset_key: str):
    """
    统一列出当前数据集需要读取的 wav 文件。

    PROCESS-2:
        只读取任务后缀为 __CTD 的 wav：
        例如 PROCESS-2_rec__001__CTD.wav

    其他数据集:
        保持原逻辑，读取目录下全部 .wav 文件。
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
    """把 F1 列表变成 softmax 权重（温度 temp 控制分布平滑）
    温度 temp 控制分布平滑”指的是: temp 越大,softmax 权重越尖锐;temp 越小,权重越均匀.
    所以：
        temp 大:exp 差异被放大，高 F1 的层权重大幅上升，低 F1 几乎被压成 0。
        temp 小:差异被压缩，权重更平均。
    你可以把 temp 当成“偏好高分层的强度旋钮”。
    考虑“分布平滑”主要是为了控制融合时的鲁棒性和泛化，避免过度押注某一层。更具体点：
        如果不平滑(temp 很大），几乎只靠最高 F1 的那一层,容易过拟合(F1 可能只是某次验证集的偶然高分)。
        平滑一点(temp 中等或偏小）会让多层共同贡献，更稳健，对数据分布变化不那么敏感。
        Whisper 各层往往包含不同层次的信息（声学、语义、说话人特征等），适度融合比单层更全面。
    一句话：“平滑”是为了在“最佳单层”与“多层稳健融合”之间找到折中，减少偶然性带来的风险。
    """
    import numpy as _np
    f1 = _np.asarray(f1_list, dtype=_np.float64)
    fmin, fmax = float(f1.min()), float(f1.max())
    if abs(fmax - fmin) < 1e-12:
        return _np.ones_like(f1) / len(f1)
    s = (f1 - fmin) / (fmax - fmin)     # 把 F1 归一化到 [0,1]
    e = _np.exp(temp * s)               
    return e / (e.sum() + 1e-12)

def apply_norm(h: torch.Tensor, mode: str, ln_post=None, skip_ln_post: bool=False):
    """
    对 [1,T,C] 归一化：
      None / "layernorm" / "std" / "ln_post"
    apply_norm 的作用是：对 Whisper 的隐藏特征做归一化处理，让不同层或不同段的特征分布更一致，便于后续融合与训练。
    它支持几种模式：
        None:不做归一化，直接返回。
        "layernorm"：对特征做标准 LayerNorm(无可学习参数)。
        "std"：只按整体标准差缩放（控制尺度）。
        "ln_post" / "whisper_ln"：用 Whisper 编码器最后的 LayerNorm(带训练参数)对特征做归一化。
    为什么要做：
        不同层的特征分布差异很大，直接融合可能不稳定。
        归一化能把特征拉到相近尺度，提高融合效果与训练稳定性。
        ln_post 还能让中间层特征“对齐”到 Whisper 顶层分布，便于使用同一分类器或下游模型。
    一句话
        apply_norm 是为了在不同层、不同段之间对齐分布、稳定训练和融合结果
    [1, T, C] 是 Whisper 编码器输出的张量形状：
        1:batch size(一次只处理一个音频段).
        T:时间步长度（帧数，取决于音频长度和 Whisper 的帧/步长）.
        C:特征维度(Whisper encoder 的隐藏维度，例如 small 模型是 768).
        所以每一帧都有一个长度为 C 的特征向量，共 T 帧。
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
            raise ValueError("ln_post is None,但选择了 'ln_post/whisper_ln'")
        return ln_post(h)
    raise ValueError(f"Unknown norm mode: {mode}")

def stack_with_norm(hs_list, mode, ln_post=None, skip_last_ln_post=False):
    """对多个层先逐层归一化再 stack
    stack_with_norm 的作用是：对多个层的特征先逐层归一化，再堆叠成一个张量，为后续“多层融合”做准备
    做了什么：
        输入是一组层特征列表 hs_list(每个形状 [1, T, C])。
        如果 mode 为 None,直接 torch.stack。
        否则对每一层调用 apply_norm 做归一化。
        skip_last_ln_post 用来避免顶层重复应用 ln_post(顶层可能已经 post-LN)。
    为什么要这么做：
        不同层的分布不一致，直接堆叠再融合会让某些层因为尺度或统计差异“占便宜”。
        先归一化能让每层在同一尺度上融合，更稳定、更公平。
        对顶层跳过 ln_post 是为了避免重复归一化（否则会破坏原本的分布）。
    一句话:stack_with_norm 是在“多层融合”前做统一的分布对齐，让融合更可靠。
    """
    if mode is None:
        return torch.stack(hs_list, dim=0)
    out, L = [], len(hs_list)
    for i, h in enumerate(hs_list):
        skip = (skip_last_ln_post and i == L-1 and mode in ("ln_post", "whisper_ln"))
        out.append(apply_norm(h, mode, ln_post=ln_post, skip_ln_post=skip))
    return torch.stack(out, dim=0)

def _model_tag_from_name(model_name: str):
    """模型名转标签(比如 whisper-small → whisper_small)但在本脚本里未使用。"""
    tail = model_name.split("/")[-1]            # whisper-medium
    return tail.replace("whisper-", "whisper_") # whisper_medium

# ===================== 5) Whisper 模型与 ln_post =====================
# ===================== 5) 特征模型与归一化层 =====================
if FEATURE_BACKBONE.lower() == "whisper":
    processor = WhisperProcessor.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model = WhisperModel.from_pretrained(PRETRAINED_MODEL_NAME, local_files_only=LOCAL_FILES_ONLY)
    model.eval().to(DEVICE)
    LN_POST = model.encoder.layer_norm
    if TOP_LAYER_INDEX is None:
        TOP_LAYER_INDEX = int(model.config.encoder_layers)
    # 打印一次 ln_post 参数摘要（可选）
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
        raise ValueError(f"不支持的 FEATURE_BACKBONE: {FEATURE_BACKBONE}")

    model.eval().to(DEVICE)
    LN_POST = None

    if TOP_LAYER_INDEX is None:
        TOP_LAYER_INDEX = int(model.config.num_hidden_layers)

    # 非 Whisper 模型不支持 Whisper 的 ln_post 归一化；若误设则自动回退并提示。
    if NORM_FOR_SINGLE in ("ln_post", "whisper_ln"):
        print(f"[WARN] {FEATURE_BACKBONE} 不支持 ln_post，NORM_FOR_SINGLE 自动改为 layernorm。")
        NORM_FOR_SINGLE = "layernorm"
    if NORM_BEFORE_FUSE in ("ln_post", "whisper_ln"):
        print(f"[WARN] {FEATURE_BACKBONE} 不支持 ln_post，NORM_BEFORE_FUSE 自动改为 layernorm。")
        NORM_BEFORE_FUSE = "layernorm"
#================================================

print(f"[MODEL] FEATURE_BACKBONE={FEATURE_BACKBONE}")
print(f"[MODEL] FEAT_MODEL_NAME={FEAT_MODEL_NAME}")
print(f"[MODEL] PRETRAINED_MODEL_NAME={PRETRAINED_MODEL_NAME}")
print(f"[MODEL] TOP_LAYER_INDEX={TOP_LAYER_INDEX}")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ===================== 6) 单文件分段提取 =====================
@torch.inference_mode()
def get_segment_features(audio_file: str,
                         audio_file_index: int,
                         single_layer=None,
                         norm_single=None,
                         last_k=None,
                         drop_input_layer=True,  # 丢掉输入层，只保留 transformer 各层输出。
                         layer_f1=None,
                         f1_temp=12,
                         norm_mode=None,
                         segment_length=segment_length1,
                         sampling_rate=16000,
                         overlap_rate=overlap_rate1):
    import numpy as _np
    y, _ = librosa.load(audio_file, sr=sampling_rate, mono=True, dtype=_np.float32) # 用 librosa.load 将音频重采样到 16k
    # 论文预处理包含音量归一化：将波形缩放到 [-1,1] 稳定特征提取。
    if ENABLE_AUDIO_NORMALIZE:
        peak = float(_np.max(_np.abs(y))) if len(y) > 0 else 0.0
        if peak > 1e-8:
            y = y / peak
    step_size = int(segment_length * (1 - overlap_rate))  # 例如 30s、30% overlap -> step=21s

    # 原逻辑会导致短于 overlap 阈值的音频被完全跳过：
    #   SEGMENT_SEC=30, overlap_rate=0.3 时，阈值是 9 秒。
    #   若 len(y) < 9 秒，则 total_segments=0。
    #
    # 对 PROCESS-2 的 CTD 任务，部分音频可能很短。
    # 为避免短音频完全丢失，只要音频非空，就至少生成 1 个 segment。
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
            # 与 Whisper 对齐：仅对非首段做前向补齐（HubERT/WavLM/Whisper统一策略）。
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
            # 顶层是否已 post-LN（Whisper 专用）。
            try:
                already_post = torch.allclose(hs[-1], enc_out.last_hidden_state, atol=1e-5, rtol=1e-5)
            except Exception:
                already_post = True
        else:
            # HuBERT/WavLM 前向：提取 hidden_states 作为高层声学表示。
            inp = processor(segment, sampling_rate=sampling_rate, return_tensors="pt")
            inp = {k: v.to(DEVICE) for k, v in inp.items()}
            enc_out = model(**inp, output_hidden_states=True, return_dict=True)
            hs_all = list(enc_out.hidden_states)
            hs = hs_all[1:] if (drop_input_layer and len(hs_all) >= 2) else hs_all
            # HuBERT/WavLM 不存在 Whisper 式 post-LN 判断，固定为 False。
            already_post = False



        if single_layer is not None:
            """说明你选择了“单层特征”。代码会从指定层(single_layer)取出该层的隐藏状态,
            再按 norm_single 做归一化。"""
            assert 1 <= int(single_layer) <= len(hs), f"SINGLE_LAYER={single_layer} 超出范围"
            is_top = (int(single_layer) == len(hs))
            if (not warned_double_ln_top and is_top and already_post and norm_single in ("ln_post", "whisper_ln")):
                print(f"[WARN] {Path(audio_file).name}: 顶层已 post-LN，外部又设 ln_post —— 已跳过第二次 ln_post。")
                warned_double_ln_top = True
            h = hs[int(single_layer) - 1]
            is_top = (int(single_layer) == len(hs))
            skip_ln = (norm_single in ("ln_post", "whisper_ln")) and (is_top or int(single_layer) == TOP_LAYER_INDEX)
            fused = apply_norm(h, norm_single, ln_post=LN_POST, skip_ln_post=skip_ln)

        else:
            """说明你没有指定单层，就走“多层融合”路线：
            如果给了 layer_f1,就按 F1 权重加权融合；否则按 last_k 取最后 K 层做均值融合。"""
            if layer_f1 is not None:
                # 做归一化后加权融合
                # layer_f1 需要你自己先离线评估每一层的 F1，然后把每层的 F1 值按层顺序组成一个列表传进去。
                # 长度必须等于 len(hs)（transformer 层数）。顺序要与 hs 对应（第 1 层到第 L 层）。
                assert len(layer_f1) == len(hs), "layer_f1 长度需与层数匹配"
                if (not warned_double_ln_top and norm_mode in ("ln_post", "whisper_ln") and already_post):
                    print(f"[WARN] {Path(audio_file).name}: 融合 ln_post + 顶层已 post-LN —— 仅对非顶层做 ln_post。")
                    warned_double_ln_top = True
                H = stack_with_norm(hs, norm_mode, ln_post=LN_POST, skip_last_ln_post=already_post)
                w_np = weights_from_f1(layer_f1, temp=f1_temp)
                w = torch.tensor(w_np, dtype=H.dtype, device=H.device).view(-1,1,1,1)
                fused = (w * H).sum(dim=0)
            else:
                assert last_k is not None and 1 <= last_k <= len(hs), "last_k 无效"
                selected = hs[-last_k:]
                if (not warned_double_ln_top and norm_mode in ("ln_post", "whisper_ln") and already_post):
                    print(f"[WARN] {Path(audio_file).name}: 后K融合 ln_post + 顶层已 post-LN —— 跳过顶层 ln_post。")
                    warned_double_ln_top = True
                # 原来：skip_last_ln_post=already_post
                skip_last = already_post or (len(hs) == TOP_LAYER_INDEX)  # 顶层=12 时也跳过最后一层的 ln_post
                H = stack_with_norm(selected, norm_mode, ln_post=LN_POST, skip_last_ln_post=skip_last)
                fused = H[0] if last_k == 1 else H.mean(dim=0)

        fused = fused.squeeze(0).detach().cpu()  # [T,C]，fused 原本形状是 [1, T, C]（batch=1）。squeeze(0) 去掉 batch 维度 → [T, C]，便于后续堆叠。
        seg_features.append(fused) # 把这一段的特征存进列表。
        seg_indices.append((audio_file_index, i)) # 记录“这段特征来自哪一个音频文件、该音频的第几段”。
    return seg_features, seg_indices

# ===================== 6.1) MMSE 标签工具函数 =====================
def _pick_first_existing_column(df: pd.DataFrame, candidates):
    """
    从候选列名中找到 DataFrame 里第一个真实存在的列名。
    用途：不同数据集命名不完全一致（例如 adressfname / addressfname / ID），
    因此这里做一个统一兜底，避免后面写很多 if-else。
    """
    norm_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        key = str(cand).strip().lower()
        if key in norm_map:
            return norm_map[key]
    raise KeyError(f"未找到候选列 {candidates}，当前列为：{list(df.columns)}")


def _safe_to_float(x):
    """
    将 MMSE 文本安全转换为 float。
    - 空字符串 / NA / NaN / None -> 返回 None
    - 其余可解析数字 -> 返回 float
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
    从 Pitt 文件名中解析：
        002-0c-en.wav -> ("2", 0)
        013-4c-en.wav -> ("13", 4)
        001-0.cha     -> ("1", 0)

    返回：
        (subject_key, visit_index)
    若无法解析，则返回 (None, None)
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
    Pitt readme 对应关系：
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
        raise ValueError(f"非法 visit_index={visit_index}")
    if visit_index == 0:
        return "mms"
    return f"mmse{visit_index + 1}"

def _normalize_mmse_key(dataset_key: str, raw_id):
    """
    把“元数据里的 ID”和“音频文件名 stem”统一规范化，方便做字典匹配。

    PROCESS-2 特殊点：
    - meta-info.csv 里的 ID 是参与者级 ID：
        PROCESS-2_rec__001
    - flat 目录中的音频 stem 是任务级 ID：
        PROCESS-2_rec__001__SFT
        PROCESS-2_rec__001__PFT
        PROCESS-2_rec__001__CTD
    因此需要去掉末尾的 __SFT / __PFT / __CTD，回到参与者 ID。
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
    从 DataFrame 两列（ID, MMSE）构建 {规范化ID: mmse} 映射。
    若 MMSE 缺失，则自动跳过；后续这些音频对应的 segment 会被丢弃。
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
    解析 ADReSS-IS2020 的 txt 元数据。
    格式特点：
    - 分号分隔
    - train_cc / train_cd 各一份
    - test_all 一份，供 test_cc / test_cd 共用
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
    解析 PROCESS-2 的 meta-info.csv，构建四个子集的 MMSE 映射。

    meta-info.csv 需要包含：
        IDs
        diagnosis
        MMSE
        Split

    二分类目录映射：
        HC -> cc
        MCI / Dementia -> cd

    返回：
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
    解析 ADReSSo21 的 MMSE csv。
    训练集文件里同时包含：
    - adressfname: 音频ID
    - mmse: MMSE 分数
    - dx: ad / cn
    测试集文件里包含：
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

    # 训练集按 dx 分成健康对照/患者两份映射；test 没必要再分，cc/cd 共用同一份
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
    解析 Pitt 的 Excel 总表，构建 visit-level MMSE 映射：
        (subject_key, visit_index) -> target_mmse

    readme 对应关系：
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

    # 统一列名映射，兼容大小写差异
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

        # 支持 0~6，对应 mms / mmse2 ~ mmse7
        for visit_index in range(0, 7):
            col_name = _pitt_visit_to_mmse_col(visit_index).lower()
            if col_name not in col_map:
                continue

            mmse_val = _safe_to_float(row[col_map[col_name]])
            if mmse_val is None:
                continue

            all_map[(subject_key, int(visit_index))] = float(mmse_val)

    # Pitt 没有单独 train/test MMSE 文件，四个子集共享同一份 visit-level 映射
    return {
        "train_cc": all_map,
        "train_cd": all_map,
        "test_cc": all_map,
        "test_cd": all_map,
    }


def _read_adress_m_mmse(cfg: dict):
    """
    解析 ADReSS-M 的 groundtruth csv。
    训练集文件包含：
    - adressfname
    - dx
    - mmse
    测试集文件包含：
    - addressfname（注意这里常见为 addressfname，多了一个 s）
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
    读取指定数据集的 MMSE 元数据文件，返回按子集划分的字典。

    返回格式固定为：
    {
        "train_cc": {"S002": 30.0, ...},
        "train_cd": {"S079": 11.0, ...},
        "test_cc":  {"S160": 28.0, ...},
        "test_cd":  {"S160": 28.0, ...},
    }

    这样后面的 segment 级标签生成逻辑就不需要关心“底层文件到底是 txt / csv / xlsx”。
    """
    if dataset_key not in MMSE_META_PATHS:
        raise ValueError(f"TARGET_KIND='mmse' 时，暂未配置数据集 {dataset_key} 的 MMSE_META_PATHS")

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
        raise ValueError(f"暂未实现数据集 {dataset_key} 的 MMSE 解析逻辑")

    print(
        f"[MMSE] {dataset_key} 映射加载完成："
        f"train_cc={len(mmse_maps['train_cc'])}, "
        f"train_cd={len(mmse_maps['train_cd'])}, "
        f"test_cc={len(mmse_maps['test_cc'])}, "
        f"test_cd={len(mmse_maps['test_cd'])}"
    )
    return mmse_maps

def _make_missing_mmse_record(dataset_key: str, subset_name: str, wav_path: str):
    """
    为缺失 MMSE 的音频构造一条记录，后续统一导出 CSV。
    若是 Pitt 且已启用 visit-level MMSE，则额外记录 subject_id / visit_index / expected_mmse_col。
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
    把完整的缺失 MMSE 音频列表导出成 CSV。
    """
    if missing_records is None or len(missing_records) == 0:
        print("[MMSE] 本次没有缺失 MMSE 的音频，无需导出 CSV。")
        return None

    os.makedirs(save_root, exist_ok=True)

    if single_layer is not None:
        csv_name = f"missing_mmse_{feature_prefix}_single{single_layer}_{ov_str}_{sec_str}.csv"
    else:
        csv_name = f"missing_mmse_{feature_prefix}_last{last_k}_{ov_str}_{sec_str}.csv"

    csv_path = os.path.join(save_root, csv_name)

    df = pd.DataFrame(missing_records)

    # 去重：同一 subset 下同一路径只保留一条
    dedup_cols = [c for c in ["subset_name", "wav_path"] if c in df.columns]
    if len(dedup_cols) > 0:
        df = df.drop_duplicates(subset=dedup_cols, keep="first")

    # 尽量把常用列排前面
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
    print(f"[MMSE] 缺失 MMSE 音频完整列表已保存到: {csv_path}")
    print(f"[MMSE] 缺失音频总数（去重后）: {len(df)}")
    return csv_path


def build_mmse_labels_for_segments(audio_path: str, seg_indices, mmse_map: dict, subset_name: str, dataset_key: str):
    """
    根据 extract_dir 的排序规则，把 segment 级索引映射回音频 ID，
    再把该音频的 MMSE 分数复制给它的每个 segment。

    返回：
    - labels_tensor
    - keep_mask
    - missing_records: 缺失 MMSE 的完整音频记录（用于导出 CSV）
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
        # Pitt：若你已经改成 visit-level MMSE，这里就按 (subject, visit) 查
        # 其他数据集：保持原逻辑
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

    # 日志里仍然打印摘要，但 CSV 会保存完整列表
    missing_audio_names = sorted(set(missing_audio_names), key=natural_key)
    if len(missing_audio_names) > 0:
        print(
            f"[MMSE] {subset_name} 中有 {len(missing_audio_names)} 个音频缺少 MMSE，将自动丢弃："
            f"{missing_audio_names[:10]}"
        )

    labels_tensor = torch.tensor(labels[keep_mask], dtype=torch.float32)
    return labels_tensor, keep_mask, missing_records


# ===================== 7) 目录遍历（带进度条） =====================

def extract_dir(audio_path: str, dataset_key: str = DATASET_KEY, **kwargs):
    files = list_wav_files_for_dataset(audio_path, dataset_key)

    if str(dataset_key).strip().lower() == "process-2":
        print(f"[DIR] {audio_path}")
        print(f"[PROCESS-2] 只读取 CTD wav，文件数: {len(files)}")

    features, indices = [], []
    for idx, fp in enumerate(tqdm(files, desc=f"[DIR] {audio_path}", ncols=90)):
        seg_feats, seg_idx = get_segment_features(fp, idx, **kwargs)
        features.extend(seg_feats)
        indices.extend(seg_idx)

    return features, indices

# ===================== 8) 组装与保存（严格文件名格式） =====================
OV_STR  = "0.3"
SEC_STR = f"{SEGMENT_SEC}s"

# 将一组 [T,C] 特征补齐到同一时间长度后再堆叠，避免 torch.stack 因 T 不一致报错。
def stack_with_time_padding(feature_list, target_t=None, pad_value=0.0):
    # 空列表直接返回空张量，防止后续流程崩溃。
    if len(feature_list) == 0:
        return torch.empty((0, 0, 0), dtype=torch.float32)
    # 检查每个元素都是二维 [T,C]。
    for i, feat in enumerate(feature_list):
        if feat.ndim != 2:
            raise ValueError(f"feature_list[{i}] 不是二维张量，当前形状={tuple(feat.shape)}")
    # 特征维 C 必须一致。
    feat_dim = int(feature_list[0].shape[1])
    for i, feat in enumerate(feature_list):
        if int(feat.shape[1]) != feat_dim:
            raise ValueError(f"feature_list[{i}] 的特征维不一致：{feat.shape[1]} != {feat_dim}")
    # 目标时间长度：若未指定则用当前列表最大 T。
    if target_t is None:
        target_t = max(int(feat.shape[0]) for feat in feature_list)
    # 逐个补齐（右侧补零）或截断，确保都变成 [target_t, C]。
    padded = []
    for feat in feature_list:
        t = int(feat.shape[0])
        if t < target_t:
            pad = torch.full((target_t - t, feat_dim), float(pad_value), dtype=feat.dtype)
            feat = torch.cat([feat, pad], dim=0)
        elif t > target_t:
            feat = feat[:target_t]
        padded.append(feat)
    # 堆叠成 [N, target_t, C]。
    return torch.stack(padded, dim=0)

def run_and_save_one_config(*, dataset_paths, single_layer=None, last_k=None,
                            norm_single=None, norm_mode=None, drop_input_layer=True,
                            layer_f1=None, f1_temp=12):
    # 提取四个子集
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

    # 先按全局最大时间长度统一补齐，再堆叠，避免不同音频段长导致 stack 报错。
    all_feats = feat_cc_tr + feat_cd_tr + feat_cc_te + feat_cd_te
    global_t = max(int(feat.shape[0]) for feat in all_feats)
    print(f"[PAD] 统一时间长度 target_t={global_t}")
    features_cc      = stack_with_time_padding(feat_cc_tr, target_t=global_t)
    features_cd      = stack_with_time_padding(feat_cd_tr, target_t=global_t)
    features_cc_test = stack_with_time_padding(feat_cc_te, target_t=global_t)
    features_cd_test = stack_with_time_padding(feat_cd_te, target_t=global_t)

#======================================================================
    # 先把 indices 转成 numpy，便于后面统一过滤/重映射
    idx_cc_tr = np.array(idx_cc_tr, dtype=np.int64)
    idx_cd_tr = np.array(idx_cd_tr, dtype=np.int64)
    idx_cc_te = np.array(idx_cc_te, dtype=np.int64)
    idx_cd_te = np.array(idx_cd_te, dtype=np.int64)

    # ===================== 标签生成：二分类 / MMSE 二选一 =====================
    if TARGET_KIND == "mmse":
        mmse_maps = load_mmse_maps_for_dataset(DATASET_KEY)

        # 为每个 segment 生成 MMSE 标签；若对应音频无 MMSE，则该 segment 丢弃
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
        # 注意：如果某些音频没有 MMSE（例如 S001=NA），必须同步删掉对应的特征与索引
        features_cc      = features_cc[torch.from_numpy(keep_cc)]
        features_cd      = features_cd[torch.from_numpy(keep_cd)]
        features_cc_test = features_cc_test[torch.from_numpy(keep_cc_te)]
        features_cd_test = features_cd_test[torch.from_numpy(keep_cd_te)]

        idx_cc_tr = idx_cc_tr[keep_cc]
        idx_cd_tr = idx_cd_tr[keep_cd]
        idx_cc_te = idx_cc_te[keep_cc_te]
        idx_cd_te = idx_cd_te[keep_cd_te]

    else:
        # 保持原来的 AD 二分类标签
        labels_cc      = torch.zeros(features_cc.shape[0], dtype=torch.float32)
        labels_cd      = torch.ones(features_cd.shape[0], dtype=torch.float32)
        labels_cc_test = torch.zeros(features_cc_test.shape[0], dtype=torch.float32)
        labels_cd_test = torch.ones(features_cd_test.shape[0], dtype=torch.float32)

        all_missing_mmse_records = []

    # train / test 拼接
    features1 = torch.cat([features_cc, features_cd], dim=0)
    features2 = torch.cat([features_cc_test, features_cd_test], dim=0)
    labels1   = torch.cat([labels_cc, labels_cd], dim=0)
    labels2   = torch.cat([labels_cc_test, labels_cd_test], dim=0)

    # ===================== indices 合并，保证 cc/cd 的全局音频索引不重叠 =====================
    if len(idx_cc_tr) > 0 and len(idx_cd_tr) > 0:
        max_idx = idx_cc_tr[:, 0].max()
        idx_cd_tr[:, 0] += (max_idx + 1)
    indices_train = np.vstack([idx_cc_tr, idx_cd_tr]) if len(idx_cd_tr) > 0 else idx_cc_tr

    if len(idx_cc_te) > 0 and len(idx_cd_te) > 0:
        max_idx = idx_cc_te[:, 0].max()
        idx_cd_te[:, 0] += (max_idx + 1)
    indices_test = np.vstack([idx_cc_te, idx_cd_te]) if len(idx_cd_te) > 0 else idx_cc_te
#===================================================


    # —— 保存（严格遵从格式）——
    # {layer} 是单层编号, {K} 是融合的后 K 层, {overlap} 来自 OV_STR = "0.3", {sec} 来自 SEC_STR = "30s", 
    # 所以文件名一眼就能看出：单层 or 融合、参数、训练/测试。
    # 
    # —— 为当前配置单独创建保存目录 ——
    if single_layer is not None:
        SINGLE_LAYER = str(single_layer)
        SAVE_ROOT = os.path.join(
            BASE_SAVE_ROOT,
            DATASET_KEY,
            FEAT_MODEL_NAME,
            f"single{SINGLE_LAYER}",
        )
    else:
        # 保留后K融合模式的兼容性
        SAVE_ROOT = os.path.join(
            BASE_SAVE_ROOT,
            DATASET_KEY,
            FEAT_MODEL_NAME,
            f"last{last_k}",
        )

    os.makedirs(SAVE_ROOT, exist_ok=True)

    # —— 保存（严格遵从格式）——
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

    # —— 若是 MMSE 任务，则把缺失 MMSE 的完整音频列表导出成 CSV ——
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

# ===================== 9) 执行：按列表连续处理 =====================
paths = DATASETS[DATASET_KEY]
print(f"[DATASET] {DATASET_KEY}")
for k,v in paths.items():
    print(f"  {k}: {v}")

if len(SINGLE_LAYER_LIST) > 0:
    print(f"\n[RUN] 单层批处理：{SINGLE_LAYER_LIST}，NORM_FOR_SINGLE={NORM_FOR_SINGLE}")
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
    print(f"\n[RUN] 后K层融合批处理：{LAST_K_LIST}，NORM_BEFORE_FUSE={NORM_BEFORE_FUSE}")
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
    raise ValueError("请至少在 SINGLE_LAYER_LIST 或 LAST_K_LIST 中提供一个列表。")


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
#     local_dir="/home/juanyu/whisper_lru_3/hubert_xlarge_ll60k",
#     local_dir_use_symlinks=False,
# )
# print("done")
# PY

