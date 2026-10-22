# -*- coding: utf-8 -*-
"""MMSE 回归：仅支持 lru / lstm / transformer。
收敛速度实验精简版；模型结构、数据处理和评估流程对齐 lru+transformer+lstm.py。
可独立运行，无需导入参考脚本；仍需原项目 models/LRU.py、data_dir/dataloaders.py。
先修改 CONFIG_ROOT_DIR、GPU_ID 和数据/特征配置，再在原项目环境运行本文件。
official_train 全量训练；official_test 只用于监测和报告。

conda activate audiojax_gpu
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v2/lru+transformer+lstm_speed.py
"""

# ===== 0. 全局配置单元（先改这里） =====
import os
import sys
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence
CONFIG_ROOT_DIR = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v2"
CONFIG_FEATURE_ROOT = os.path.join(CONFIG_ROOT_DIR, "feature_tensor")
CONFIG_OUTPUT_ROOT = os.path.join(CONFIG_ROOT_DIR, "outputs_repeat")
# UEA 缓存。
CONFIG_PRESPLIT_CACHE_NAME = "UEA"

# ====== 基础数据集 / 模型配置 ======
# 可选: "ADReSS-IS2020" , "ADReSSo21" , "Pitt" , "PROCESS-2"
#
# 批量运行时改这里。
DATASET_OPTIONS = ["ADReSS-IS2020"]
# 保留单值变量，供旧函数内部使用；
# 父进程会在 _launch_batch_runs_if_needed() 中按 DATASET_OPTIONS 逐个覆盖它；
# child 进程会通过 LRU_ACTIVE_DATASET_OPTION 覆盖它。
DATASET_OPTION = DATASET_OPTIONS[0]
# 这里改成“列表控制”：
MODEL_NAMES = ["lru", "lstm", "transformer"]  # 可选: ["lru"], ["lstm"], ["transformer"]

# ====== 特征来源配置（同时兼容冻结特征 + 微调特征） ======
# 1) FEAT_MODEL_NAME 对应 audio_cut_xr.py 中的 FEAT_MODEL_NAME；
#    冻结特征目录形如：
#    feature_tensor/DATASET/FEAT_MODEL_NAME/singleX/
# 2) FT_MODEL_NAME 对应 audio_cut_mmse_regression_ft_mutil.py 中的 MODEL_NAME；
#    微调特征目录形如：
FEAT_MODEL_NAME = (
    "whisper_small"  # 冻结 hubert_base / wav2vec2_base / wavlm_base_plus / whisper_small
)
FT_MODEL_NAME = "ptapt_lr5e-5_epo5"  # 微调不需要 ft_，代码会自动补上
FEATURE_SOURCE_MODES = ["freeze"]  # 可选: ["freeze"], ["finetune"], 或两者都跑
# official_train 全量训练；official_test 仅监测和报告。早停只依据 train_loss。
ENABLE_RECURSIVE_BATCH = True  # 若想使用自定义路径,设为 False；父进程批量发现时保持 True

# ====== 运行开关 ======
#    - RUN_SINGLE_SEED_400=True  -> 额外单独跑一遍 seed=400
#    - RUN_MULTI_SEEDS=True      -> 跑 MULTI_SEEDS
# 默认建议：
#    - 单 seed 关闭
#    - 多 seed 开启
RUN_SINGLE_SEED_400 = False
RUN_MULTI_SEEDS = True
MULTI_SEEDS = [401, 402, 403, 404, 405]  # [401,402,403,404,405]
# val_* 是 official_test 的历史字段名；EVAL_OFFICIAL_TEST 仅控制额外 test_* 别名评估与导出。
EVAL_OFFICIAL_TEST = False
# 兼容旧代码里仍会读取 MODEL_NAME 的位置；
# 真正生效的 active model 会在后面的环境变量覆盖区根据 MODEL_NAMES / ACTIVE_MODEL_NAME 决定。
MODEL_NAME = MODEL_NAMES[0]

# ====== 新增：按层号过滤要递归读取的特征目录 ======
# 用法与 audio_cut_xr.py 类似：
# 1) 留空 []    -> 不按层过滤，维持原来的全量递归
# 2) 设为 [5,6] -> 只读取 single5 / single5_unfree* / single6 / single6_unfree* 等目录
SINGLE_LAYERS = [ 12]

# ====== 新增：仅对“微调特征”按 unfree 层数过滤 ======
# 1) 留空 []  -> 不按 unfree 数过滤，single5_unfree1 / single5_unfree2 都会保留
# 2) 设为 [1] -> 只保留 *_unfree1
# 3) 与 SINGLE_LAYERS 联合使用时：
#    SINGLE_LAYERS=[5], UNFREE_LAYER_NUM=[1]
#    => 只匹配 single5_unfree1
UNFREE_LAYER_NUM = [1]

# ====== 单跑兜底配置（当 ENABLE_RECURSIVE_BATCH=False 或作为 child 进程时使用） ======
FEATURE_MODEL_TAG = ""
FEATURE_SELECTION_TAG = ""
FEATURE_DATA_DIR_OVERRIDE = None
FEATURE_FILE_TAG = "whisper_single12_0.3_30s"

# ====== 保存路径标签 ======
# 按你的要求，最终输出根目录统一为：
OUTPUT_MODEL_TAG = MODEL_NAME
OUTPUT_DATASET_TAG = DATASET_OPTION
OUTPUT_FEATURE_TAG = FEATURE_MODEL_TAG
OUTPUT_SELECTION_TAG = FEATURE_SELECTION_TAG
OUTPUT_EXPORT_PREFIX_OVERRIDE = None

# ====== 训练与评估配置 ======
RANDOM_SEED = 400
BATCH_SIZE = 16

# ====== 收敛曲线实验：所有模型固定训练 TARGET_EPOCHS 轮 ======
TARGET_EPOCHS = 10
# 正式当前训练协议：
# 固定 TARGET_EPOCHS 轮；关闭早停，以相同 epoch 编号汇总多 seed 曲线。
#
# 因此这里必须为 True：
#   num_steps_run = TARGET_EPOCHS * steps_per_epoch
#
# NUM_STEPS 仅作为旧代码兼容兜底；在正式当前训练协议下不会生效。
USE_FIXED_EPOCHS = True
NUM_STEPS = 6000
PRINT_STEPS = None
# 收敛实验关闭早停；以下早停参数仅保留为统一训练接口的配置。
USE_EARLY_STOPPING = False  # 收敛曲线实验：每个 seed 完整训练 TARGET_EPOCHS 轮
EARLY_STOPPING_STEPS = 3
EARLY_STOPPING_MONITOR = "train_loss"
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_RESTORE_BEST = False
# 最终报告策略：
# - 若没有 early stop：报告 TARGET_EPOCHS 的最后 epoch；
# - 若 train_loss early stop：报告停止时的最后 epoch；
# - 不使用 official_test 选择 epoch。
EPOCH_SELECTION_POLICY = "last_epoch_or_train_loss_early_stop"
TEST_MONITOR_ONLY = True
LR_LRU = 1e-4
LR_TRANSFORMER = 1e-4
# LSTM 与 LRU 对齐：相同基础学习率。
LR_LSTM = 1e-4
WARMUP_EPOCHS = 2
COSINE_ALPHA = 0.05

# ====== 时序模型可选优化器配置：默认 Adam，增强开关当前关闭 ======
USE_ADAMW_WITH_GRAD_CLIP = False
OPTIMIZER_ADAMW_CLIP_MODELS = []
GRAD_CLIP_NORM = 1.0
OPT_WEIGHT_DECAY = 1e-4

# ====== 监测指标：仅计算与记录，不参与 checkpoint 选择 ======
EPOCH_SELECTION_OBJECTIVE = "audio_rmse"

# ====== audio-level 聚合配置 ======
AUDIO_AGG_PRIMARY_METHOD = "mean"
AUDIO_AGG_TRIM_RATIO = 0.10

# ====== PROCESS-2 专用：MMSE 分桶加权 / 混合 loss ======
# 必须保持 True，才能在 build_raw_dataset() 中构造 [target, weight]。
USE_PROCESS2_MMSE_BIN_WEIGHT = True
# PROCESS-2 train 分布极端偏向高 MMSE，因此训练时使用合并后的 3 档：
# [0,21), [21,26), [26,31)
PROCESS2_MMSE_BIN_EDGES = [0.0, 21.0, 26.0, 31.0]
PROCESS2_MMSE_BIN_LABELS = ["[0,21)", "[21,26)", "[26,31)"]
# effective-number weighting 参数
PROCESS2_MMSE_BIN_WEIGHT_BETA = 0.99
PROCESS2_MMSE_BIN_WEIGHT_MAX = 2.0
PROCESS2_WEIGHT_EPS = 1e-8
# 混合 loss:
# alpha = 0.0 -> 普通 MSE
# alpha = 1.0 -> 纯 weighted MSE
# alpha = 0.3 -> 推荐优先尝试
PROCESS2_MIXED_LOSS_ALPHA = 0.3  # 0.2\0.3\0.4
FEATURE_STANDARDIZE = (
    False  # 默认False，解冻之后可以对其进行消融实验，看看是否能提升性能。（之前尝试是提升的）
)
INCLUDE_TIME = False
T_MAX = 1.0

# ====== 兼容旧版兜底映射 ======
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

    raise ValueError(f"环境变量 {env_name} 无法解析为 bool: {raw}")

def normalize_model_name(name: str) -> str:
    low = str(name).strip().lower()
    if low in ["lru"]:
        return "lru"
    if low in ["lstm"]:
        return "lstm"
    if low in ["transformer"]:
        return "transformer"
    raise ValueError("本脚本仅支持 lru / lstm / transformer，收到: " + str(name))

def normalize_model_name_list(names: Sequence[str]) -> List[str]:
    out = []
    seen = set()
    for x in names:
        name = normalize_model_name(x)
        if name not in seen:
            out.append(name)
            seen.add(name)
    return out

def should_run_multi_seed_for_model(model_name: str) -> bool:
    "should_run_multi_seed_for_model：用于本文件保留的 lru / lstm / transformer 模型。"
    model_name = normalize_model_name(model_name)
    return bool(RUN_MULTI_SEEDS)

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
    从目录名里解析 unfree 的数字。
    示例：
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
    清理输出路径中的 selection tag。

    目标：
    - 冻结模型：single12 -> single12
    - 微调模型：single12_unfree1 -> single12_unfree1
    - 若误传入 single12_unfree1/legacy_partition，则只取 single12_unfree1

    注意：
    这里不会把 single12_unfree1 压缩成 single12。
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
    """
    LRU 输出路径统一为：

        outputs/<dataset>/<feature_model>/<downstream_model>/<selection>_speed

    示例：
        冻结特征：
            outputs/ADReSS-IS2020/whisper_small/lru/single12_speed

        微调特征：
            outputs/ADReSS-IS2020/whisper_small/lru/single12_unfree1_speed

    不再追加：
        额外的协议目录或交叉验证目录
    """
    clean_selection_tag = _clean_output_selection_tag(selection_tag)
    if not clean_selection_tag.endswith("_speed"):
        clean_selection_tag += "_speed"

    return os.path.join(
        CONFIG_OUTPUT_ROOT,
        str(dataset_name).strip(),
        str(output_feature_model_tag).strip(),
        normalize_model_name(downstream_model_name),
        clean_selection_tag,
    )

def _discover_frozen_contexts() -> List[Dict[str, object]]:
    """
    发现冻结特征目录。

    输入目录：
        feature_tensor/<dataset>/<FEAT_MODEL_NAME>/<selection>/

    输出目录：
        outputs/<dataset>/<FEAT_MODEL_NAME>/<downstream_model>/<selection>_speed

    每个特征配置对应一个输出目录。
    """
    if "freeze" not in {str(x).strip().lower() for x in FEATURE_SOURCE_MODES}:
        return []

    root_dir = os.path.join(CONFIG_FEATURE_ROOT, DATASET_OPTION, FEAT_MODEL_NAME)
    if not os.path.isdir(root_dir):
        print(f"[DISCOVER] 冻结特征根目录不存在，跳过: {root_dir}")
        return []

    contexts = []

    for feature_train_path in sorted(Path(root_dir).rglob("feature_*_train.pt")):
        feature_dir = str(feature_train_path.parent)
        file_tag = _parse_feature_file_tag(str(feature_train_path))
        if _is_blank(file_tag):
            continue
        if not _has_complete_feature_bundle(feature_dir, file_tag):
            continue
        # 跳过历史交叉验证子目录，避免同一特征重复进入批量训练。
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
                # 输出标签
                "feature_model_tag": FEAT_MODEL_NAME,
                "feature_selection_tag": _clean_output_selection_tag(rel_selection),
                # 输入信息
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
    """发现 feature_tensor/<dataset>/ft_<FT_MODEL_NAME>/<selection>/ 下的微调特征。
    自动扫描跳过历史交叉验证子目录；输出使用真实输入特征模型名。
    """
    if "finetune" not in {str(x).strip().lower() for x in FEATURE_SOURCE_MODES}:
        return []

    input_feature_model_tag = f"ft_{FT_MODEL_NAME}"
    root_dir = os.path.join(CONFIG_FEATURE_ROOT, DATASET_OPTION, input_feature_model_tag)

    if not os.path.isdir(root_dir):
        print(f"[DISCOVER] 微调特征根目录不存在，跳过: {root_dir}")
        return []

    contexts = []

    for feature_train_path in sorted(Path(root_dir).rglob("feature_*_train.pt")):
        feature_dir_path = Path(feature_train_path).parent

        #   <selection>/feature_*.pt
        legacy_partition_match = re.fullmatch(r"fold(\d+)", feature_dir_path.name)

        # 自动发现仅使用完整 train/test 特征，跳过历史交叉验证子目录。
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
                # finetune 输出路径使用真实微调特征来源名，例如 whisper_small_tsc_ft。
                "feature_model_tag": str(input_feature_model_tag).strip(),
                "feature_selection_tag": output_selection_tag,
                # 真实输入信息额外保留，方便日志追踪。
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

def _build_run_contexts() -> List[Dict[str, object]]:
    contexts = _discover_frozen_contexts() + _discover_finetune_contexts()

    dedup = []
    seen = set()
    for ctx in contexts:
        key = (
            ctx["source_mode"],
            ctx["feature_model_tag"],
            ctx["feature_selection_tag"],
            ctx["feature_file_tag"],
            ctx["feature_data_dir_override"],
        )
        if key in seen:
            continue
        seen.add(key)
        dedup.append(ctx)

    def _context_sort_key(x):
        selection_tag = str(x["feature_selection_tag"])

        single_layer = _extract_single_layer_from_selection(selection_tag)
        if single_layer is None:
            single_layer = 10**9

        unfree_num = _extract_unfree_num_from_selection(selection_tag)
        if unfree_num is None:
            unfree_num = 10**9

        return (
            str(x["source_mode"]),
            str(x["feature_model_tag"]),
            int(single_layer),
            int(unfree_num),
            str(selection_tag),
            str(x["feature_file_tag"]),
        )

    dedup.sort(key=_context_sort_key)
    return dedup
# 父进程按数据集、特征和模型派生子进程，每个子进程运行一个实验组合。
DATASET_OPTION = os.getenv("LRU_ACTIVE_DATASET_OPTION", DATASET_OPTION)
# child 进程只负责一个 active dataset；
# 父进程保留 DATASET_OPTIONS，用于批量发现多个数据集。
if os.getenv("LRU_BATCH_CHILD", "0") == "1":
    DATASET_OPTIONS = [str(DATASET_OPTION)]
FEAT_MODEL_NAME = os.getenv("LRU_ACTIVE_FEAT_MODEL_NAME", FEAT_MODEL_NAME)
FT_MODEL_NAME = os.getenv("LRU_ACTIVE_FT_MODEL_NAME", FT_MODEL_NAME)
FEATURE_SOURCE_MODES = _parse_str_list_env("LRU_FEATURE_SOURCE_MODES", FEATURE_SOURCE_MODES)
SINGLE_LAYERS = _parse_int_list_env("LRU_SINGLE_LAYERS", SINGLE_LAYERS)
UNFREE_LAYER_NUM = _parse_int_list_env("LRU_UNFREE_LAYER_NUM", UNFREE_LAYER_NUM)
MODEL_NAMES = normalize_model_name_list(_parse_str_list_env("LRU_MODEL_NAMES", MODEL_NAMES))
RUN_SINGLE_SEED_400 = _parse_bool_env("LRU_RUN_SINGLE_SEED_400", RUN_SINGLE_SEED_400)
RUN_MULTI_SEEDS = _parse_bool_env("LRU_RUN_MULTI_SEEDS", RUN_MULTI_SEEDS)
MULTI_SEEDS = _parse_int_list_env("LRU_MULTI_SEEDS", MULTI_SEEDS)
EVAL_OFFICIAL_TEST = _parse_bool_env("LRU_EVAL_OFFICIAL_TEST", EVAL_OFFICIAL_TEST)
USE_EARLY_STOPPING = _parse_bool_env("LRU_USE_EARLY_STOPPING", USE_EARLY_STOPPING)
EARLY_STOPPING_STEPS = _parse_int_env("LRU_EARLY_STOPPING_STEPS", EARLY_STOPPING_STEPS)
ACTIVE_MODEL_NAME = os.getenv("LRU_ACTIVE_MODEL_NAME", None)
if not _is_blank(ACTIVE_MODEL_NAME):
    MODEL_NAME = normalize_model_name(ACTIVE_MODEL_NAME)
else:
    if len(MODEL_NAMES) == 0:
        raise ValueError("MODEL_NAMES 为空，至少需要提供一个模型名。")
    MODEL_NAME = normalize_model_name(MODEL_NAMES[0])
# child 中只保留当前 active model；父进程里 MODEL_NAMES 仍可包含多个
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
# 同步刷新这些“展示/派生”标签
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
    raise ValueError("FEATURE_FILE_TAG 为空，且 FILE_TAG_MAP 中也找不到兜底映射。")
print("[ACTIVE CONFIG]")
print("  DATASET_OPTION =", DATASET_OPTION)
print("  FEAT_MODEL_NAME =", FEAT_MODEL_NAME)
print("  FT_MODEL_NAME =", FT_MODEL_NAME)
print("  MODEL_NAMES =", MODEL_NAMES)
print("  ACTIVE_MODEL_NAME =", MODEL_NAME)
print("  RUN_SINGLE_SEED_400 =", RUN_SINGLE_SEED_400)
print("  RUN_MULTI_SEEDS =", RUN_MULTI_SEEDS)
print("  MULTI_SEEDS =", MULTI_SEEDS)
print("  EVAL_OFFICIAL_TEST =", EVAL_OFFICIAL_TEST)
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

def _build_single_selection_contexts() -> List[Dict[str, object]]:
    """
    单跑模式构造一个独立运行上下文。

    单跑模式的核心规则：
    - 保存路径直接由单跑配置决定；
    - FEATURE_MODEL_TAG 决定 outputs/<dataset>/<feature_model>/...；
    - FEATURE_SELECTION_TAG / 实际匹配到的 selection 决定 outputs/.../<selection>_speed；
    - 不使用 FEATURE_SOURCE_MODES 判断输出路径；
    - 不使用 FT_MODEL_NAME 构造输出路径。

    读取优先级：
    1) 若 FEATURE_DATA_DIR_OVERRIDE 非空，直接读该目录；
    2) 否则读：
        feature_tensor/<dataset>/<FEATURE_MODEL_TAG>/<FEATURE_SELECTION_TAG>
    3) 如果精确目录不存在或文件不完整，则在：
        feature_tensor/<dataset>/<FEATURE_MODEL_TAG>/
       下扫描完整 feature bundle，并按 SINGLE_LAYERS / UNFREE_LAYER_NUM / FEATURE_SELECTION_TAG 过滤。

    输出：
        outputs/<dataset>/<FEATURE_MODEL_TAG>/<MODEL_NAME>/<resolved_selection>
    """

    # 单跑模式不再根据 FEATURE_SOURCE_MODES 推断 freeze / finetune。
    # 这里只作为日志标签使用，不参与路径构造。
    source_mode = "single_run"

    candidate_items = []

    def _add_candidate(feature_dir: str, selection_tag: str):
        """
        candidate_items:
            (feature_dir, file_tag, selection_tag)

        selection_tag 用于输出路径最后一级。
        """
        if _is_blank(feature_dir):
            return

        feature_dir = str(feature_dir).strip()
        if not os.path.isdir(feature_dir):
            return

        # 优先用当前 FILE_TAG。
        if _has_complete_feature_bundle(feature_dir, FILE_TAG):
            candidate_items.append((feature_dir, str(FILE_TAG), str(selection_tag)))
            return

        # 如果 FILE_TAG 不匹配，则自动从 feature_*_train.pt 解析 file_tag。
        for p_train in sorted(Path(feature_dir).glob("feature_*_train.pt")):
            parsed_tag = _parse_feature_file_tag(str(p_train))
            if _is_blank(parsed_tag):
                continue
            if _has_complete_feature_bundle(feature_dir, parsed_tag):
                candidate_items.append((feature_dir, str(parsed_tag), str(selection_tag)))
                return

    if not _is_blank(FEATURE_DATA_DIR_OVERRIDE):
        # 手工指定真实特征目录时，仍然不使用 FT_MODEL_NAME / FEATURE_SOURCE_MODES。
        # 输出 feature model 仍由 FEATURE_MODEL_TAG 决定。
        override_dir = str(FEATURE_DATA_DIR_OVERRIDE).strip()
        override_path = Path(override_dir)

        if re.fullmatch(r"fold\d+", override_path.name, flags=re.IGNORECASE):
            override_selection = override_path.parent.name
        else:
            override_selection = override_path.name

        # 如果你希望严格由 FEATURE_SELECTION_TAG 决定输出 selection，
        # 可把下一行改成：override_selection = FEATURE_SELECTION_TAG
        _add_candidate(override_dir, override_selection)

    else:
        # 1) 精确单跑目录：
        #    feature_tensor/<dataset>/<FEATURE_MODEL_TAG>/<FEATURE_SELECTION_TAG>
        exact_dir = os.path.join(
            CONFIG_FEATURE_ROOT,
            DATASET_OPTION,
            FEATURE_MODEL_TAG,
            FEATURE_SELECTION_TAG,
        )
        _add_candidate(exact_dir, FEATURE_SELECTION_TAG)

        # 2) 旧兼容目录：
        #    feature_tensor/<FEATURE_MODEL_TAG>/<FEATURE_SELECTION_TAG>/<dataset>
        old_dir = os.path.join(
            CONFIG_FEATURE_ROOT,
            FEATURE_MODEL_TAG,
            FEATURE_SELECTION_TAG,
            DATASET_OPTION,
        )
        _add_candidate(old_dir, FEATURE_SELECTION_TAG)

        # 3) 单跑扫描目录：
        #    只扫描 FEATURE_MODEL_TAG 根目录；
        #    不扫描 FT_MODEL_NAME_ft；
        #    不看 FEATURE_SOURCE_MODES。
        single_root = os.path.join(CONFIG_FEATURE_ROOT, DATASET_OPTION, FEATURE_MODEL_TAG)

        if os.path.isdir(single_root):
            requested_layer = _extract_single_layer_from_selection(FEATURE_SELECTION_TAG)
            requested_selection_clean = _clean_output_selection_tag(FEATURE_SELECTION_TAG)

            for feature_train_path in sorted(Path(single_root).rglob("feature_*_train.pt")):
                feature_dir_path = Path(feature_train_path).parent

                if re.fullmatch(r"fold\d+", feature_dir_path.name, flags=re.IGNORECASE):
                    continue

                feature_dir = str(feature_dir_path)
                selection_dir_path = feature_dir_path

                rel_selection = os.path.relpath(str(selection_dir_path), single_root)
                rel_selection = (
                    selection_dir_path.name
                    if rel_selection in [".", ""]
                    else rel_selection.replace("\\", "/")
                )

                # 按层过滤。
                if not _selection_matches_single_layers(rel_selection, SINGLE_LAYERS):
                    continue

                # 按 unfree 过滤。对于 frozen single12，因为没有 unfree，若 UNFREE_LAYER_NUM=[1] 会被过滤掉；
                # 这是你单跑配置中显式要求的行为。
                if not _selection_matches_unfree_nums(rel_selection, UNFREE_LAYER_NUM):
                    continue

                # 若 FEATURE_SELECTION_TAG 写 single12，允许匹配 single12_unfree1；
                # 若写 single12_unfree1，则要求清理后完全一致。
                candidate_layer = _extract_single_layer_from_selection(rel_selection)

                if requested_layer is not None and candidate_layer is not None:
                    if int(requested_layer) != int(candidate_layer):
                        continue
                else:
                    if _clean_output_selection_tag(rel_selection) != requested_selection_clean:
                        continue

                file_tag = _parse_feature_file_tag(str(feature_train_path))
                if _is_blank(file_tag):
                    continue
                if not _has_complete_feature_bundle(feature_dir, file_tag):
                    continue

                candidate_items.append((feature_dir, str(file_tag), str(rel_selection)))

    if len(candidate_items) == 0:
        raise FileNotFoundError(
            "单跑模式下未找到完整特征目录。\n"
            "单跑模式只按 FEATURE_MODEL_TAG / FEATURE_SELECTION_TAG / FEATURE_DATA_DIR_OVERRIDE 查找，"
            "不再使用 FEATURE_SOURCE_MODES 或 FT_MODEL_NAME。\n"
            f"DATASET_OPTION={DATASET_OPTION}\n"
            f"FEATURE_MODEL_TAG={FEATURE_MODEL_TAG}\n"
            f"FEATURE_SELECTION_TAG={FEATURE_SELECTION_TAG}\n"
            f"FILE_TAG={FILE_TAG}\n"
            f"SINGLE_LAYERS={SINGLE_LAYERS}\n"
            f"UNFREE_LAYER_NUM={UNFREE_LAYER_NUM}\n"
            f"FEATURE_DATA_DIR_OVERRIDE={FEATURE_DATA_DIR_OVERRIDE}\n"
            f"expected_exact_dir={os.path.join(CONFIG_FEATURE_ROOT, DATASET_OPTION, FEATURE_MODEL_TAG, FEATURE_SELECTION_TAG)}"
        )

    selected_feature_dir, selected_file_tag, selected_selection_tag = candidate_items[0]
    output_selection_tag = _clean_output_selection_tag(selected_selection_tag)

    # 关键修改：
    # 单跑模式输出 feature model 层级直接使用 FEATURE_MODEL_TAG。
    # 不再根据 FEATURE_SOURCE_MODES / FT_MODEL_NAME / FEAT_MODEL_NAME 推断。
    output_feature_model_tag = str(FEATURE_MODEL_TAG).strip()

    print("[SINGLE-RUN CONTEXT]")
    print("  source_mode =", source_mode)
    print("  FEATURE_MODEL_TAG =", FEATURE_MODEL_TAG)
    print("  FEATURE_SELECTION_TAG =", FEATURE_SELECTION_TAG)
    print("  selected_feature_dir =", selected_feature_dir)
    print("  selected_file_tag =", selected_file_tag)
    print("  selected_selection_tag =", selected_selection_tag)
    print("  output_feature_model_tag =", output_feature_model_tag)
    print("  output_selection_tag =", output_selection_tag)
    print(
        "  output_export_prefix =",
        _build_lru_output_prefix(
            dataset_name=DATASET_OPTION,
            output_feature_model_tag=output_feature_model_tag,
            downstream_model_name=MODEL_NAME,
            selection_tag=output_selection_tag,
        ),
    )

    return [
        {
            "source_mode": source_mode,
            "dataset": DATASET_OPTION,
            # 单跑保存路径直接由 FEATURE_MODEL_TAG 和 resolved selection 决定。
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

def _build_output_export_prefix_for_child_model(ctx: Dict[str, object], model_name: str) -> str:
    """
    为某个 context 下的某个下游模型 child 构造输出目录。

    输出：
        outputs/<dataset>/<feature_model>/<downstream_model>/<selection>_speed

    示例：
        冻结：
            outputs/ADReSS-IS2020/whisper_small/lru/single12_speed

        微调：
            outputs/ADReSS-IS2020/whisper_small/lru/single12_unfree1_speed
    """
    return _build_lru_output_prefix(
        dataset_name=str(ctx["dataset"]).strip(),
        output_feature_model_tag=str(ctx["feature_model_tag"]).strip(),
        downstream_model_name=model_name,
        selection_tag=str(ctx["feature_selection_tag"]).strip(),
    )

def _launch_batch_runs_if_needed():
    """
    父进程批量发现 context 后，按：
        dataset × context × model_name
    的笛卡尔积启动 child。

    每个 child 只负责：
    - 一个 dataset
    - 一个特征目录
    - 一个 active model

    这样 ADReSS-IS2020 / Pitt 的输出目录会天然分开。
    """
    global DATASET_OPTION
    global OUTPUT_DATASET_TAG
    global DATASET_NAME
    global FILE_TAG
    if os.getenv("LRU_BATCH_CHILD", "0") == "1":
        return
    script_path = os.path.abspath(__file__) if "__file__" in globals() else None
    if _is_blank(script_path):
        return
    dataset_options = [str(x).strip() for x in globals().get("DATASET_OPTIONS", [DATASET_OPTION])]
    dataset_options = [x for x in dataset_options if x != ""]
    if len(dataset_options) == 0:
        raise ValueError("DATASET_OPTIONS 为空，至少需要一个数据集。")
    all_contexts = []
    original_dataset_option = DATASET_OPTION
    original_output_dataset_tag = globals().get("OUTPUT_DATASET_TAG", DATASET_OPTION)
    original_dataset_name = globals().get("DATASET_NAME", DATASET_OPTION)
    original_file_tag = globals().get("FILE_TAG", FEATURE_FILE_TAG)
    for dataset_option in dataset_options:
        DATASET_OPTION = str(dataset_option)
        OUTPUT_DATASET_TAG = DATASET_OPTION
        DATASET_NAME = DATASET_OPTION
        if _is_blank(FEATURE_FILE_TAG):
            FILE_TAG = FILE_TAG_MAP.get(DATASET_NAME, None)
        else:
            FILE_TAG = str(FEATURE_FILE_TAG).strip()
        if _is_blank(FILE_TAG):
            raise ValueError(
                f"FEATURE_FILE_TAG 为空，且 FILE_TAG_MAP 中找不到 dataset={DATASET_NAME} 的兜底映射。"
            )
        if bool(ENABLE_RECURSIVE_BATCH):
            run_mode = "recursive_discovery"
            contexts = _build_run_contexts()
        else:
            run_mode = "single_selection"
            contexts = _build_single_selection_contexts()
        print(f"[BATCH][DATASET={DATASET_OPTION}] mode = {run_mode}")
        print(f"[BATCH][DATASET={DATASET_OPTION}] discovered contexts = {len(contexts)}")
        all_contexts.extend(contexts)
    DATASET_OPTION = original_dataset_option
    OUTPUT_DATASET_TAG = original_output_dataset_tag
    DATASET_NAME = original_dataset_name
    FILE_TAG = original_file_tag
    contexts = all_contexts
    print(f"[BATCH] dataset_options = {dataset_options}")
    print(f"[BATCH] total discovered contexts = {len(contexts)}")
    print(f"[BATCH] model_names = {MODEL_NAMES}")
    for ctx in contexts:
        print(
            f"[BATCH] dataset={ctx['dataset']} | {ctx['source_mode']:9s} | feature_model={ctx['feature_model_tag']} | selection={ctx['feature_selection_tag']} | layer={_extract_single_layer_from_selection(ctx['feature_selection_tag'])} | unfree={_extract_unfree_num_from_selection(ctx['feature_selection_tag'])} | file_tag={ctx['feature_file_tag']} | feature_dir={ctx['feature_data_dir_override']}"
        )
    if len(contexts) == 0:
        raise ValueError(
            "未发现任何可运行的特征目录，请检查 DATASET_OPTIONS / FEAT_MODEL_NAME / FT_MODEL_NAME / FEATURE_SELECTION_TAG。"
        )
    model_names_to_run = normalize_model_name_list(MODEL_NAMES)
    if len(model_names_to_run) == 0:
        raise ValueError("MODEL_NAMES 为空，无法启动任何 child。")
    for ctx in contexts:
        for model_name_child in model_names_to_run:
            child_env = os.environ.copy()
            child_env["LRU_BATCH_CHILD"] = "1"
            child_env["LRU_ACTIVE_DATASET_OPTION"] = str(ctx["dataset"])
            child_env["LRU_ACTIVE_FEAT_MODEL_NAME"] = str(FEAT_MODEL_NAME)
            child_env["LRU_ACTIVE_FT_MODEL_NAME"] = str(FT_MODEL_NAME)
            child_env["LRU_MODEL_NAMES"] = model_name_child
            child_env["LRU_ACTIVE_MODEL_NAME"] = model_name_child
            child_env["LRU_RUN_SINGLE_SEED_400"] = "1" if bool(RUN_SINGLE_SEED_400) else "0"
            child_env["LRU_RUN_MULTI_SEEDS"] = "1" if bool(RUN_MULTI_SEEDS) else "0"
            child_env["LRU_MULTI_SEEDS"] = ",".join((str(x) for x in MULTI_SEEDS))
            child_env["LRU_EVAL_OFFICIAL_TEST"] = "1" if bool(EVAL_OFFICIAL_TEST) else "0"
            child_env["LRU_USE_EARLY_STOPPING"] = "1" if bool(USE_EARLY_STOPPING) else "0"
            child_env["LRU_EARLY_STOPPING_STEPS"] = str(int(EARLY_STOPPING_STEPS))
            child_env["LRU_ACTIVE_SOURCE_MODE"] = str(ctx["source_mode"])
            child_env["LRU_SINGLE_LAYERS"] = ",".join((str(x) for x in SINGLE_LAYERS))
            child_env["LRU_UNFREE_LAYER_NUM"] = ",".join((str(x) for x in UNFREE_LAYER_NUM))
            child_env["LRU_ACTIVE_FEATURE_MODEL_TAG"] = str(ctx["feature_model_tag"])
            child_env["LRU_ACTIVE_FEATURE_SELECTION_TAG"] = str(ctx["feature_selection_tag"])
            child_env["LRU_ACTIVE_FEATURE_DATA_DIR_OVERRIDE"] = str(
                ctx["feature_data_dir_override"]
            )
            child_env["LRU_ACTIVE_FEATURE_FILE_TAG"] = str(ctx["feature_file_tag"])
            child_env["LRU_ACTIVE_OUTPUT_EXPORT_PREFIX"] = (
                _build_output_export_prefix_for_child_model(ctx=ctx, model_name=model_name_child)
            )
            print(
                f"[BATCH->CHILD] dataset={child_env['LRU_ACTIVE_DATASET_OPTION']} | model={child_env['LRU_ACTIVE_MODEL_NAME']} | feat_model={child_env['LRU_ACTIVE_FEAT_MODEL_NAME']} | ft_model={child_env['LRU_ACTIVE_FT_MODEL_NAME']} | source_mode={child_env['LRU_ACTIVE_SOURCE_MODE']} | feature_model_tag={child_env['LRU_ACTIVE_FEATURE_MODEL_TAG']} | selection={child_env['LRU_ACTIVE_FEATURE_SELECTION_TAG']} | "
            )
            subprocess.run([sys.executable, script_path], env=child_env, check=True)
    raise SystemExit(0)
_launch_batch_runs_if_needed()
GPU_ID = 5  # 默认为5，所有实验都是在GPU=5上跑的
if "jax" in sys.modules:
    print("[警告] 检测到 jax 已导入；若你修改了 GPU_ID，请重启 kernel。")
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
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
print("EVAL_OFFICIAL_TEST =", EVAL_OFFICIAL_TEST)
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

# ===== 1. 路径读取 cell（参考 NCMMSC2021_LogNCDE.ipynb） =====
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
    解析当前运行的输出目录。

    不再包含：
        logncde_official_train_test
        交叉验证目录
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
    获取当前训练 run 的输出根目录。

    优先使用 child/context 传入的 OUTPUT_EXPORT_PREFIX。
    fallback 同样使用当前特征配置的输出目录。
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
    """缓存目录为 <data_dir>/processed/<CONFIG_PRESPLIT_CACHE_NAME>/<dataset>/
    <feature_model>/<downstream_model>/<selection>/<file_tag>。各下游模型使用独立缓存。
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

    # 下游模型名加入 cache 路径，避免多个模型并发读写同一批 pkl。
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

# ===== 1. 路径读取 cell（参考 NCMMSC2021_LogNCDE.ipynb） =====
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
print("[protocol] official_train 全量训练；official_test 仅监测与报告，不参与选模或早停。")
print("[protocol] val_* 为 official_test_monitor 的历史字段名；test_* 为可选别名。")

# ===== 2. 基础导入单元 =====
import time
import pickle
from datetime import datetime
import copy
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from sklearn.metrics import r2_score
import torch
os.chdir(ROOT_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
print("cwd =", os.getcwd())
print("JAX devices =", jax.devices())

# ===== 4. 执行数据转换（直接从 .pt 构建 presplit pkl，MMSE 回归版） =====

def _to_numpy(x, dtype=None):
    """把 torch.Tensor / np.ndarray / list 统一转成 numpy。"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    else:
        x = np.asarray(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x

def _load_json(path):
    """读取 json 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _save_pkl(obj, path):
    """把对象保存为 pkl。"""
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def _canonical_wav_path(wav_path):
    """统一 wav 路径表示，避免相对/绝对路径不一致导致匹配失败。"""
    return str(Path(wav_path).expanduser().resolve())

def _natural_key(path_str):
    """自然排序：single2 在 single10 前面；S002 在 S010 前面。"""
    name = Path(path_str).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", name)]

def _list_wav_files(audio_dir, dataset_name=None):
    """
    读取目录中的 wav，并按自然顺序排序。

    PROCESS-2:
        只读取文件名 stem 以 __CTD 结尾的 wav。
        示例：
            PROCESS-2_rec__001__CTD.wav

    其他数据集:
        保持原逻辑，读取目录下全部 .wav。
    """
    if audio_dir is None or (not os.path.isdir(audio_dir)):
        raise FileNotFoundError(f"音频目录不存在: {audio_dir}")

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
# 这里只保留当前 LRU 脚本会用到的几个数据集
DATASET_AUDIO_DIRS = {
    "PROCESS-2": {
        "train_cc": "/home/juanyu/Downloads/PROCESS-2/train/cc",
        "train_cd": "/home/juanyu/Downloads/PROCESS-2/train/cd",
        "test_cc": "/home/juanyu/Downloads/PROCESS-2/test/cc",
        "test_cd": "/home/juanyu/Downloads/PROCESS-2/test/cd",
    },
    "ADReSS-IS2020": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cc",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/train/Full_wave_enhanced_audio/cd",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cc_audio",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-IS2020-data/test/Full_wave_enhanced_cd_audio",
    },
    "ADReSSo21": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/cn",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/train/audio/ad",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/Control",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSSo21/diagnosis/test-dist/ProbableAD",
    },
    "Pitt": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cc_enhence",
        "train_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/train/cd_enhence",
        "test_cc": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cc_enhence",
        "test_cd": "/home/juanyu/Downloads/dementiabank/Pitt_new/norm/train_test_split/test/cd_enhence",
    },
    "ADReSS_m": {
        "train_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/train_wav/HC",
        "train_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/train_wav/AD",
        "test_cc": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr/HC",
        "test_cd": "/home/juanyu/Downloads/dementiabank/ADReSS-M/train1/test-gr/AD",
    },
}

def _parse_file_tag_from_feature_train_path(feature_train_path):
    """
    从:
        .../feature_whisper_single10_0.3_30s_train.pt
    解析出:
        whisper_single10_0.3_30s
    """
    name = os.path.basename(str(feature_train_path))
    m = re.match(r"^feature_(.+)_train\.pt$", name)
    if m is None:
        return None
    return str(m.group(1))

def _build_audio_entry_lookup_from_split_meta(split_meta):
    """
    从“当前特征自己的 split metadata”中构造：
        current_local_audio_id -> {wav_path, wav_name, target_mmse}

    注意：
    这里只能用于“当前特征自己的 metadata”；
    不能把外部交叉验证划分文件 当成当前 local audio_id lookup 直接用。
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
    为冻结特征重建“当前特征本地 audio_id -> wav_path”映射。

    这里严格模拟 audio_cut_xr.py 的 train 索引生成顺序：
        train_cc 自然排序
        + train_cd 自然排序
    """
    if dataset_name not in DATASET_AUDIO_DIRS:
        raise ValueError(f"当前未配置数据集 {dataset_name} 的原始音频目录。")

    dataset_paths = DATASET_AUDIO_DIRS[dataset_name]

    ordered_files = []
    ordered_files.extend(_list_wav_files(dataset_paths["train_cc"], dataset_name=dataset_name))
    ordered_files.extend(_list_wav_files(dataset_paths["train_cd"], dataset_name=dataset_name))

    if str(dataset_name).strip().lower() == "process-2":
        print(f"[PROCESS-2] 只使用 CTD wav 重建 train audio_id 映射，文件数: {len(ordered_files)}")

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
    读取“当前这份特征自己的 local audio_id -> wav_path”映射。
    这里只读取音频映射元数据，不使用其中的 train/val 成员关系划分数据。

    优先级：
    1) 当前 feature_dir 下的 train_val_split_<file_tag>.json
       - 适用于微调特征（本地 metadata 与当前 indices 同源）
    2) 若不存在，则回退到按原始数据目录顺序重建
       - 适用于冻结特征
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
    """将上游 .pt 转为下游缓存。official_train 全部训练；official_test 同时保存为
    val 监测集和 test 别名，不进行额外划分。目标标准化在后续加载训练数据时拟合。
    本地 train_val_split_<file_tag>.json 仅用于追踪音频 ID 与 wav_path 的对应关系。
    """

    # ========= 1) 路径检查 =========
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
            raise FileNotFoundError(f"{name} 不存在: {p}")

    # ========= 2) 读取 train/test segment 级张量 =========
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
            f"index_train 应为 (N,2) 或至少第二维>=2，当前 shape={idx_train_full.shape}"
        )
    if idx_test_full.ndim != 2 or idx_test_full.shape[1] < 2:
        raise ValueError(f"index_test 应为 (N,2) 或至少第二维>=2，当前 shape={idx_test_full.shape}")

    idx_train_full = idx_train_full[:, :2].astype(np.int64)
    idx_test_full = idx_test_full[:, :2].astype(np.int64)

    # ========= 2.1) 构造“当前特征自己的 local audio_id -> wav_path” =========
    current_audio_entry_lookup, current_audio_lookup_source_path = (
        _load_current_feature_audio_entry_lookup(
            dataset_name=dataset_name,
            feature_train_path=feature_train_path,
        )
    )

    # ========= 3/4) LogNCDE-style：official_train -> train, official_test -> monitor/report =========
    # - official_train 的全部 segment 都进入 X_train；
    # - official_test 的全部 segment 进入 X_val，用作训练过程 monitoring / reporting；
    # - official_test 不参与梯度更新；
    # - official_test 不用于 early stopping；
    # - official_test 不用于 best epoch / checkpoint selection；
    # - early stopping 若开启，只基于 official_train 的 train_loss；
    # - 最终结果统一取训练停止时的最后模型。
    #
    # 说明：
    # 这里仍然沿用变量名 X_val / y_val / idx_val，是为了最小化下游代码改动。
    # 但该 val 的实验语义是 official_test_monitor，而不是 model-selection validation。

    audio_entry_lookup_source_path = current_audio_lookup_source_path
    split_strategy = (
        "logncde_style_official_train_as_train_official_test_as_monitor_train_loss_stop_last_epoch"
    )

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
    print("[split] official_test  -> val monitor/reporting set")
    print("[split] official_test  -> test alias, only if EVAL_OFFICIAL_TEST=True")
    print("[split] selection policy -> last_epoch_or_train_loss_early_stop")
    print("[split] official_test is NOT used for checkpoint selection or early stopping")
    print("[split] early stopping monitor -> official_train train_loss")
    print(
        "[split] audio counts | "
        f"train={len(train_audio_ids)}, "
        f"val_monitor={len(val_audio_ids)}, "
        f"test_alias={len(test_audio_ids)}"
    )
    # ========= 5) 基本合法性检查 =========
    if len(X_train) == 0:
        raise ValueError("official_train 特征为空，请检查训练特征文件。")
    if len(X_val) == 0:
        raise ValueError("official_test 监测特征为空，请检查测试特征文件。")
    if len(X_test) == 0:
        raise ValueError("X_test 为空，请检查 test 特征文件。")

    if not (len(X_train) == len(y_train) == len(idx_train)):
        raise ValueError("train 的 X / y / idx 长度不一致。")
    if not (len(X_val) == len(y_val) == len(idx_val)):
        raise ValueError("val 的 X / y / idx 长度不一致。")
    if not (len(X_test) == len(y_test) == len(idx_test)):
        raise ValueError("test 的 X / y / idx 长度不一致。")

    ds_dir = get_presplit_cache_dir(
        data_dir=data_dir,
        dataset_name=dataset_name,
        feature_model_tag=globals().get("FEATURE_MODEL_TAG", None),
        feature_selection_tag=globals().get("FEATURE_SELECTION_TAG", None),
        file_tag=globals().get("FILE_TAG", None),
    )
    os.makedirs(ds_dir, exist_ok=True)

    print("[presplit_cache_dir]", ds_dir)

    # ========= 7) 保存为 presplit pkl =========
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

    # ========= 8) 保存摘要 =========

    summary = {
        "dataset_name": dataset_name,
        "task": "mmse_regression",
        "source": "direct_from_audio_cut_jax_mmse_pt",
        "split_protocol": "logncde_style_official_train_as_train_official_test_as_monitor_train_loss_stop_last_epoch",
        "train_semantics": "official_train",
        "val_semantics": "official_test_monitor",
        "test_semantics": "official_test_alias",
        "feature_train_path": feature_train_path,
        "feature_test_path": feature_test_path,
        "label_train_path": label_train_path,
        "label_test_path": label_test_path,
        "index_train_path": index_train_path,
        "index_test_path": index_test_path,
        "audio_entry_lookup_source_path": audio_entry_lookup_source_path,
        "split_strategy": split_strategy,
        "seed": int(seed),
        "epoch_selection_policy": str(globals().get("EPOCH_SELECTION_POLICY", "")),
        "test_monitor_only": bool(globals().get("TEST_MONITOR_ONLY", True)),
        "test_monitor_used_for_selection": 0,
        "early_stopping_monitor": str(globals().get("EARLY_STOPPING_MONITOR", "train_loss")),
        "early_stopping_steps": int(globals().get("EARLY_STOPPING_STEPS", 3)),
        "early_stopping_restore_best": bool(globals().get("EARLY_STOPPING_RESTORE_BEST", False)),
        "note": (
            "LogNCDE-style protocol: official_train is used completely for training; "
            "official_test is mapped to val only for monitoring/reporting. "
            "official_test is not used for early stopping, epoch selection, or checkpoint selection. "
            "Early stopping, if enabled, is based only on official_train train_loss; "
            "the reported model is the last model at training stop."
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
        "audio_counts": {
            "train_audio": int(len(np.unique(idx_train[:, 0]))),
            "val_audio": int(len(np.unique(idx_val[:, 0]))),
            "test_audio": int(len(np.unique(idx_test[:, 0]))),
        },
        "segment_counts": {
            "train_segments": int(len(idx_train)),
            "val_segments": int(len(idx_val)),
            "test_segments": int(len(idx_test)),
        },
        "target_stats": {
            "train_min": float(np.min(y_train)),
            "train_max": float(np.max(y_train)),
            "train_mean": float(np.mean(y_train)),
            "train_std": float(np.std(y_train)),
            "val_min": float(np.min(y_val)),
            "val_max": float(np.max(y_val)),
            "val_mean": float(np.mean(y_val)),
            "val_std": float(np.std(y_val)),
            "test_min": float(np.min(y_test)),
            "test_max": float(np.max(y_test)),
            "test_mean": float(np.mean(y_test)),
            "test_std": float(np.std(y_test)),
        },
    }

    summary_path = os.path.join(ds_dir, "stage1_pt_to_mmse_presplit_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary

# ===== 执行转换 =====
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
print("\n[OK] 已直接从 audio_cut_jax 的 .pt 生成 MMSE 回归版 presplit pkl。")
print("[NEXT] 将继续加载缓存并执行训练与评估。")

# ===== 5A. 特征标准化（参考 NCMMSC2021_LogNCDE.ipynb 的 preprocess_data） =====
# 定义特征标准化函数（严格复用参考实现的公式与维度）。

def preprocess_data(features):
    """沿时间维对每个样本做标准化，使序列特征具有零均值和单位方差。"""
    # 计算每个样本在时间维(axis=1)上的均值，保持维度便于广播。
    mean = features.mean(axis=1, keepdims=True)
    # 计算每个样本在时间维(axis=1)上的标准差，保持维度便于广播。
    std = features.std(axis=1, keepdims=True)
    # 按公式执行标准化，并加 1e-8 防止除零。
    standardized_features = (features - mean) / (std + 1e-8)
    print(mean)
    print(std)
    # 返回标准化后的特征。
    return standardized_features
# 打印标准化策略说明。
print("preprocess_data 已就绪：按 axis=1 做 (x-mean)/(std+1e-8)")

# ===== 5. 构建 Dataset（MMSE 回归版，加入 train-only y 标准化） =====
from types import SimpleNamespace
from data_dir.dataloaders import Dataloader
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

# ===== 目标标准化配置 =====
if "TARGET_STANDARDIZE" not in globals():
    TARGET_STANDARDIZE = True
if "TARGET_STANDARDIZE_EPS" not in globals():
    TARGET_STANDARDIZE_EPS = 1e-6
ACTIVE_TARGET_NORMALIZER = None

def ensure_regression_targets(y):
    """把任意形状的回归标签整理成一维 float32 数组。"""
    y_np = np.asarray(y, dtype=np.float32)
    if y_np.ndim == 0:
        y_np = y_np.reshape(1)
    if y_np.ndim > 1:
        if y_np.shape[-1] == 1:
            y_np = y_np.reshape(-1)
        else:
            raise ValueError(f"回归标签应为 1 维或最后一维为 1，当前 shape={y_np.shape}")
    return y_np.astype(np.float32)

def fit_target_normalizer_train_only(y_train, eps=1e-6):
    """只基于 train 标签拟合标准化器。"""
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
    """按 train-only normalizer 做标准化。"""
    y = ensure_regression_targets(y)
    if normalizer is None or not bool(normalizer.get("enabled", False)):
        return y.astype(np.float32)
    mean = float(normalizer["mean"])
    std = float(normalizer["std"])
    return ((y - mean) / std).astype(np.float32)

def denormalize_targets(y, normalizer=None):
    """把标准化空间的预测/标签反变换回原始 MMSE。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if normalizer is None:
        normalizer = globals().get("ACTIVE_TARGET_NORMALIZER", None)
    if normalizer is None or not bool(normalizer.get("enabled", False)):
        return y.astype(np.float32)
    mean = float(normalizer["mean"])
    std = float(normalizer["std"])
    return (y * std + mean).astype(np.float32)

def summarize_target_distribution(labels, split_name, normalizer=None, already_normalized=False):
    """
    打印标签分布。
    already_normalized=False: 直接按原始 MMSE 打印
    already_normalized=True : 同时打印标准化空间和反标准化回原始 MMSE 的摘要
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
    当前 PROCESS-2 专用判断。
    只在 PROCESS-2 上启用 MMSE-bin weighted loss。
    """
    if dataset_name is None:
        dataset_name = globals().get("DATASET_NAME", globals().get("DATASET_OPTION", ""))
    return str(dataset_name).strip().lower() == "process-2"

def assign_process2_merged_mmse_bin(mmse_value) -> str:
    """
    PROCESS-2 专用合并 MMSE bin：
        [0,21), [21,26), [26,31)

    注意：
    - train 中 [0,11) 没有样本，[11,16) 只有极少样本；
      因此训练加权时不使用原始 5 档，改用合并后的 3 档。
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
    根据每个 MMSE bin 的 audio 数量计算 effective-number weight。

    bin_counts:
        {"[0,21)": 9, "[21,26)": 28, "[26,31)": 100}

    返回：
        每个 bin 的 audio-level weight。
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
    PROCESS-2 专用 segment weight 构造。

    逻辑：
    1) 先按 audio 统计 MMSE bin；
    2) 按 bin 计算 audio-level effective-number weight；
    3) 每个 segment 的权重 = audio_bin_weight / 当前 audio 的 segment 数；
    4) 最后归一化到 segment 平均权重约为 1，稳定 loss 数值尺度。

    参数：
        y_train_raw:
            train segment 的原始 MMSE 标签，shape [N]
        idx_train_audio:
            train segment 对应的 audio id，shape [N]
            来自 presplit 缓存里的 idx_train.pkl

    返回：
        segment_weights:
            shape [N]，每个 train segment 的 loss weight
        info:
            统计信息，用于打印和保存
    """
    y_raw = ensure_regression_targets(y_train_raw).astype(np.float32)
    audio_ids = np.asarray(idx_train_audio).astype(np.int64).reshape(-1)

    if len(y_raw) != len(audio_ids):
        raise ValueError(
            "[PROCESS-2 weight] y_train_raw 与 idx_train_audio 长度不一致："
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

    # 归一化到 segment 平均权重约为 1
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
    把训练目标打包为 [target, weight]。
    - 第 0 列：原来的回归标签，可能是标准化后的 y
    - 第 1 列：loss weight

    val/test 不需要打包，保持一维标签。
    """
    y_target = ensure_regression_targets(y_target).astype(np.float32)
    segment_weights = np.asarray(segment_weights, dtype=np.float32).reshape(-1)

    if len(y_target) != len(segment_weights):
        raise ValueError(
            "y_target 与 segment_weights 长度不一致：" f"{len(y_target)} != {len(segment_weights)}"
        )

    return np.stack([y_target, segment_weights], axis=1).astype(np.float32)

def compute_process2_audio_bin_metrics(y_true_audio, y_pred_audio):
    """
    PROCESS-2 专用 audio-level per-bin / macro-bin 指标。
    输入必须是原始 MMSE 空间。
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
    从 presplit pkl 读取 train/val/test 特征与标签，并构建 dataset。
    关键修改：
    1) train-only y 标准化
    2) dataset 中保存 target_normalizer
    3) 全局 ACTIVE_TARGET_NORMALIZER 同步更新，供后续评估反标准化
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
            f"未找到 presplit 缓存目录: {ds_dir}\n"
            f"请先确认 convert_stage1_pt_to_mmse_presplit_pkl 已成功执行。"
        )

    print("[build_raw_dataset] ds_dir =", ds_dir)
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

    # ===================== PROCESS-2 专用：构造 train segment weights =====================
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
                "[PROCESS-2 weight] 需要 idx_train.pkl 来按 audio_id 修正 segment 权重，"
                f"但未找到: {idx_train_pkl_path}"
            )

        with open(idx_train_pkl_path, "rb") as f:
            idx_train_audio = pickle.load(f)

        train_segment_weights, process2_weight_info = build_process2_segment_weights(
            y_train_raw=y_train_raw,
            idx_train_audio=idx_train_audio,
            beta=float(PROCESS2_MMSE_BIN_WEIGHT_BETA),
            max_weight=float(PROCESS2_MMSE_BIN_WEIGHT_MAX),
        )

        # 训练 loader 使用 [target, weight]；target 仍是标准化后的 y_train。
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
from models.LRU import LRU
# 读出头配置
READOUT_POOL = "mean"
DEFAULT_MODEL_CONFIG = {
    "lru": {
        "num_blocks": 2,
        "hidden_dim": 128,
        "ssm_dim": 64,
        "repr_dim": 128,
        "mlp_hidden_dim": 128,
        "pool": READOUT_POOL,
    },
    "transformer": {
        "d_model": 128,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout": 0.0,
        "repr_dim": 128,
        "mlp_hidden_dim": 128,
        "pool": READOUT_POOL,
    },
    "lstm": {
        "num_layers": 2,
        "hidden_dim": 128,
        "repr_dim": 128,
        "mlp_hidden_dim": 128,
        "pool": READOUT_POOL,
        "bidirectional": False,
    },
}

def _json_safe_value(obj):
    """
    将 Python / NumPy / Path 等对象转换成 json 可安全保存的格式。
    """
    if obj is None:
        return None

    if isinstance(obj, (str, bool, int)):
        return obj

    if isinstance(obj, float):
        try:
            return float(obj) if np.isfinite(obj) else None
        except Exception:
            return None

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if np.isfinite(val) else None

    if isinstance(obj, (np.ndarray,)):
        return _json_safe_value(obj.tolist())

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): _json_safe_value(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe_value(x) for x in list(obj)]

    return str(obj)

def _build_training_config_dir():
    """返回当前模型和特征选择的训练配置保存目录。"""
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
        dataset_name=OUTPUT_DATASET_TAG,
        output_feature_model_tag=OUTPUT_FEATURE_TAG,
        downstream_model_name=MODEL_NAME,
        selection_tag=OUTPUT_SELECTION_TAG,
    )

def build_training_config_snapshot(extra_info=None):
    """汇总当前数据集、模型、特征选择和实际训练协议，生成训练配置。"""
    model_name = normalize_model_name(MODEL_NAME)
    active_model_config = copy.deepcopy(DEFAULT_MODEL_CONFIG[model_name])
    dataset_info = {}
    dataset_obj = globals().get("dataset", None)
    if dataset_obj is not None:
        try:
            dataset_info = {
                "data_dim": int(dataset_obj.data_dim),
                "label_dim": int(dataset_obj.label_dim),
                "train_size_segments": int(dataset_obj.raw_dataloaders["train"].size),
                "val_size_segments": int(dataset_obj.raw_dataloaders["val"].size),
                "test_size_segments": int(dataset_obj.raw_dataloaders["test"].size),
                "target_normalizer": dataset_obj.target_normalizer,
            }
        except Exception as e:
            dataset_info = {"warning": f"dataset 信息读取失败: {repr(e)}"}
    feature_paths = {}
    for key, value in globals().get("PATH_DICT", {}).items():
        feature_paths[key] = {"path": str(value), "exists": bool(os.path.exists(str(value)))}
    try:
        presplit_cache_dir = get_presplit_cache_dir(
            data_dir=DATA_DIR,
            dataset_name=DATASET_NAME,
            feature_model_tag=FEATURE_MODEL_TAG,
            feature_selection_tag=FEATURE_SELECTION_TAG,
            file_tag=FILE_TAG,
            cache_model_name=MODEL_NAME,
        )
    except Exception as e:
        presplit_cache_dir = f"failed_to_resolve: {repr(e)}"
    save_dir = _build_training_config_dir()
    save_path = os.path.join(save_dir, "training_config.json")
    snapshot = {
        "schema_version": 4,
        "EXPERIMENT_KIND": "speed",
        "training": {
            "TARGET_EPOCHS": int(TARGET_EPOCHS),
            "USE_FIXED_EPOCHS": bool(USE_FIXED_EPOCHS),
            "NUM_STEPS": int(NUM_STEPS),
            "BATCH_SIZE": int(BATCH_SIZE),
            "USE_EARLY_STOPPING": bool(USE_EARLY_STOPPING),
            "EARLY_STOPPING_STEPS": int(EARLY_STOPPING_STEPS),
            "EARLY_STOPPING_MIN_DELTA": float(EARLY_STOPPING_MIN_DELTA),
            "EARLY_STOPPING_MONITOR": str(EARLY_STOPPING_MONITOR),
            "EARLY_STOPPING_RESTORE_BEST": bool(EARLY_STOPPING_RESTORE_BEST),
            "LR_LRU": float(LR_LRU),
            "LR_LSTM": float(LR_LSTM),
            "LR_TRANSFORMER": float(LR_TRANSFORMER),
            "WARMUP_EPOCHS": int(WARMUP_EPOCHS),
            "COSINE_ALPHA": float(COSINE_ALPHA),
            "MULTI_SEEDS": list(MULTI_SEEDS),
        },
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "script_name": "lru+transformer+lstm_speed.py",
        "config_save_policy": {
            "level": "selection_result_dir",
            "save_path": save_path,
            "note": "该文件保存当前 official_train/test protocol 的训练参数；保存目录与当前实验结果输出目录一致。",
        },
        "dataset": {
            "DATASET_OPTION": DATASET_OPTION,
            "DATASET_NAME": DATASET_NAME,
            "split_protocol": "official_train_test",
            "train_semantics": "official_train",
            "val_semantics": "official_test_monitor",
            "test_semantics": "official_test_alias",
        },
        "feature": {
            "ACTIVE_SOURCE_MODE": ACTIVE_SOURCE_MODE,
            "FEATURE_SOURCE_MODES": FEATURE_SOURCE_MODES,
            "FEAT_MODEL_NAME": FEAT_MODEL_NAME,
            "FT_MODEL_NAME": FT_MODEL_NAME,
            "FEATURE_MODEL_TAG": FEATURE_MODEL_TAG,
            "FEATURE_SELECTION_TAG": FEATURE_SELECTION_TAG,
            "FEATURE_FILE_TAG": FEATURE_FILE_TAG,
            "FILE_TAG": FILE_TAG,
            "SINGLE_LAYERS": SINGLE_LAYERS,
            "UNFREE_LAYER_NUM": UNFREE_LAYER_NUM,
            "FEATURE_DATA_DIR": globals().get("FEATURE_DATA_DIR", None),
            "FEATURE_DATA_DIR_OVERRIDE": FEATURE_DATA_DIR_OVERRIDE,
            "feature_paths": feature_paths,
        },
        "selection_and_aggregation": {
            "EPOCH_SELECTION_POLICY": str(EPOCH_SELECTION_POLICY),
            "TEST_MONITOR_ONLY": bool(TEST_MONITOR_ONLY),
            "TEST_MONITOR_USED_FOR_SELECTION": 0,
            "EPOCH_SELECTION_OBJECTIVE": EPOCH_SELECTION_OBJECTIVE,
            "AUDIO_AGG_PRIMARY_METHOD": AUDIO_AGG_PRIMARY_METHOD,
            "AUDIO_AGG_TRIM_RATIO": float(AUDIO_AGG_TRIM_RATIO),
            "NOTE": "official_test is used only for monitoring/reporting. Checkpoint selection does not use official_test metrics. The reported model is the last model at training stop.",
        },
        "preprocess": {
            "FEATURE_STANDARDIZE": bool(FEATURE_STANDARDIZE),
            "INCLUDE_TIME": bool(INCLUDE_TIME),
            "T_MAX": float(T_MAX),
            "TARGET_STANDARDIZE": bool(globals().get("TARGET_STANDARDIZE", True)),
            "TARGET_STANDARDIZE_EPS": float(globals().get("TARGET_STANDARDIZE_EPS", 1e-06)),
        },
        "paths": {
            "CONFIG_ROOT_DIR": CONFIG_ROOT_DIR,
            "CONFIG_FEATURE_ROOT": CONFIG_FEATURE_ROOT,
            "CONFIG_OUTPUT_ROOT": CONFIG_OUTPUT_ROOT,
            "OUTPUT_EXPORT_PREFIX": globals().get("OUTPUT_EXPORT_PREFIX", None),
            "OUTPUT_EXPORT_PREFIX_OVERRIDE": OUTPUT_EXPORT_PREFIX_OVERRIDE,
            "training_config_dir": save_dir,
            "training_config_path": save_path,
            "presplit_cache_dir": presplit_cache_dir,
        },
        "dataset_info": dataset_info,
        "extra_info": {} if extra_info is None else dict(extra_info),
    }
    return _json_safe_value(snapshot)

def save_training_config_json(extra_info=None, overwrite=True):
    """将配置写入 OUTPUT_EXPORT_PREFIX/training_config.json。
    overwrite=True 时覆盖同一实验目录的旧配置。
    """
    save_dir = _build_training_config_dir()
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "training_config.json")

    if os.path.exists(save_path) and not bool(overwrite):
        print("[training_config_json] exists, skip:", save_path)
        return save_path

    snapshot = build_training_config_snapshot(extra_info=extra_info)

    tmp_path = save_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)

    os.replace(tmp_path, save_path)

    if not os.path.exists(save_path):
        raise RuntimeError(f"training_config.json 写入失败: {save_path}")

    print("[training_config_json] saved to:", save_path)
    return save_path
TRAINING_CONFIG_JSON_PATH = save_training_config_json(
    extra_info={
        "save_stage": "after_DEFAULT_MODEL_CONFIG_defined_before_training",
        "note": (
            "本文件记录当前数据集、模型和特征选择的训练配置。"
            "适用于 ENABLE_RECURSIVE_BATCH=False 和 ENABLE_RECURSIVE_BATCH=True。"
        ),
    },
    overwrite=True,
)

def _maybe_denorm_np(y, normalizer=None):
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if normalizer is None:
        normalizer = globals().get("ACTIVE_TARGET_NORMALIZER", None)
    return denormalize_targets(y, normalizer)

def _ensure_seq_repr(x):
    """
    统一把 backbone 输出整理成 [T, H]。
    支持:
      []        -> [1,1]
      [H]       -> [1,H]
      [T,H]     -> [T,H]
      [T,H,C]   -> [T, H*C]
    """
    x = jnp.asarray(x)

    if x.ndim == 0:
        return x.reshape(1, 1)

    if x.ndim == 1:
        return x[None, :]

    if x.ndim == 2:
        return x

    t = x.shape[0]
    return x.reshape(t, -1)

def pool_sequence_representation(seq_repr, mode="mean"):
    seq_repr = _ensure_seq_repr(seq_repr)

    if mode == "mean":
        return jnp.mean(seq_repr, axis=0)

    if mode == "last":
        return seq_repr[-1]

    if mode == "meanmax":
        mean_part = jnp.mean(seq_repr, axis=0)
        max_part = jnp.max(seq_repr, axis=0)
        return jnp.concatenate([mean_part, max_part], axis=0)

    raise ValueError(f"未知 pooling 模式: {mode}")

def _apply_layer_norm_over_time(layer_norm, x):
    """
    对 [T, D] 序列的每个时间步分别做 LayerNorm。
    eqx.nn.LayerNorm 默认处理单个向量，因此这里用 vmap 扩展到整段序列。
    """
    return jax.vmap(layer_norm)(x)

def _sinusoidal_positional_encoding(seq_len, d_model, dtype=jnp.float32):
    """
    生成标准 Transformer 正弦位置编码，形状 [T, D]。
    这里使用固定位置编码，不引入额外可学习参数，先保证实现稳妥。
    """
    position = jnp.arange(seq_len, dtype=dtype)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2, dtype=dtype) * (-jnp.log(10000.0) / d_model))

    pe = jnp.zeros((seq_len, d_model), dtype=dtype)
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term[: (d_model // 2)]))
    return pe

class TransformerEncoderBlock(eqx.Module):
    """
    一个轻量 Transformer Encoder block。
    结构：
        x -> LN -> MHA -> Dropout -> Residual
          -> LN -> FFN -> Dropout -> Residual
    """

    q_proj: eqx.nn.Linear
    k_proj: eqx.nn.Linear
    v_proj: eqx.nn.Linear
    out_proj: eqx.nn.Linear

    ff1: eqx.nn.Linear
    ff2: eqx.nn.Linear

    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    num_heads: int
    d_model: int
    head_dim: int

    def __init__(self, d_model, num_heads, ff_dim, dropout, *, key):
        if int(d_model) % int(num_heads) != 0:
            raise ValueError(f"d_model={d_model} 必须能被 num_heads={num_heads} 整除。")

        k1, k2, k3, k4, k5, k6 = jr.split(key, 6)

        self.q_proj = eqx.nn.Linear(int(d_model), int(d_model), key=k1)
        self.k_proj = eqx.nn.Linear(int(d_model), int(d_model), key=k2)
        self.v_proj = eqx.nn.Linear(int(d_model), int(d_model), key=k3)
        self.out_proj = eqx.nn.Linear(int(d_model), int(d_model), key=k4)

        self.ff1 = eqx.nn.Linear(int(d_model), int(ff_dim), key=k5)
        self.ff2 = eqx.nn.Linear(int(ff_dim), int(d_model), key=k6)

        self.norm1 = eqx.nn.LayerNorm((int(d_model),))
        self.norm2 = eqx.nn.LayerNorm((int(d_model),))
        self.dropout = eqx.nn.Dropout(p=float(dropout))

        self.num_heads = int(num_heads)
        self.d_model = int(d_model)
        self.head_dim = int(d_model // num_heads)

    def _self_attention(self, x):
        """
        x: [T, D]
        返回: [T, D]
        """
        q = jax.vmap(self.q_proj)(x)
        k = jax.vmap(self.k_proj)(x)
        v = jax.vmap(self.v_proj)(x)

        T = q.shape[0]
        H = self.num_heads
        Dh = self.head_dim

        q = q.reshape(T, H, Dh)
        k = k.reshape(T, H, Dh)
        v = v.reshape(T, H, Dh)

        attn_logits = jnp.einsum("thd,shd->hts", q, k) / jnp.sqrt(float(Dh))
        attn_weights = jax.nn.softmax(attn_logits, axis=-1)

        attn_out = jnp.einsum("hts,shd->thd", attn_weights, v)
        attn_out = attn_out.reshape(T, H * Dh)

        out = jax.vmap(self.out_proj)(attn_out)
        return out

    def __call__(self, x, *, key):
        """
        x: [T, D]
        """
        _k_attn, k_drop1, k_drop2 = jr.split(key, 3)

        h = _apply_layer_norm_over_time(self.norm1, x)
        h = self._self_attention(h)
        h = self.dropout(h, key=k_drop1)
        x = x + h

        h = _apply_layer_norm_over_time(self.norm2, x)
        h = jax.vmap(self.ff1)(h)
        h = jax.nn.gelu(h)
        h = jax.vmap(self.ff2)(h)
        h = self.dropout(h, key=k_drop2)
        x = x + h

        return x

class TransformerSequenceBackbone(eqx.Module):
    """
    Transformer 序列 backbone。
    输入:
        x: [T, C]
    输出:
        seq_repr: [T, d_model]
    """

    input_proj: eqx.nn.Linear
    blocks: tuple
    d_model: int

    def __init__(self, input_dim, d_model, num_layers, num_heads, ff_dim, dropout, *, key):
        keys = jr.split(key, int(num_layers) + 1)

        self.input_proj = eqx.nn.Linear(int(input_dim), int(d_model), key=keys[0])
        self.blocks = tuple(
            TransformerEncoderBlock(
                d_model=int(d_model),
                num_heads=int(num_heads),
                ff_dim=int(ff_dim),
                dropout=float(dropout),
                key=keys[i + 1],
            )
            for i in range(int(num_layers))
        )
        self.d_model = int(d_model)

    def __call__(self, x, state, key):
        """
        统一成和 LRU backbone 一样的接口：
            seq_repr, new_state = backbone(x, state, key)

        这里 Transformer 不依赖 recurrent state，因此直接把 state 原样传回。
        """
        x = _ensure_seq_repr(x)
        x = jax.vmap(self.input_proj)(x)

        pos = _sinusoidal_positional_encoding(
            seq_len=int(x.shape[0]),
            d_model=int(self.d_model),
            dtype=x.dtype,
        )
        x = x + pos

        if len(self.blocks) > 0:
            block_keys = jr.split(key, len(self.blocks))
            for block, block_key in zip(self.blocks, block_keys):
                x = block(x, key=block_key)

        return x, state

class LSTMSequenceBackbone(eqx.Module):
    "LSTMSequenceBackbone：用于本文件保留的 lru / lstm / transformer 模型。"

    input_proj: eqx.nn.Linear
    cells: tuple
    hidden_dim: int
    num_layers: int

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        bidirectional=False,
        *,
        key,
    ):
        input_dim = int(input_dim)
        hidden_dim = int(hidden_dim)
        num_layers = int(num_layers)

        if input_dim <= 0:
            raise ValueError(f"LSTM input_dim 必须 > 0，收到: {input_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"LSTM hidden_dim 必须 > 0，收到: {hidden_dim}")
        if num_layers <= 0:
            raise ValueError(f"LSTM num_layers 必须 > 0，收到: {num_layers}")

        # 为了和当前 LRU 的单向时序建模条件尽可能一致，
        # 正式主实验不使用 BiLSTM。
        if bool(bidirectional):
            raise ValueError(
                "当前公平比较版 LSTM 只支持 bidirectional=False。"
                "若后续需要 BiLSTM，请单独作为扩展/消融实验。"
            )

        keys = jr.split(key, num_layers + 1)

        # 连续上游特征先投影到统一 hidden_dim。
        self.input_proj = eqx.nn.Linear(
            input_dim,
            hidden_dim,
            key=keys[0],
        )

        # 每一层都保持 hidden_dim -> hidden_dim。
        self.cells = tuple(
            eqx.nn.LSTMCell(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                key=keys[layer_idx + 1],
            )
            for layer_idx in range(num_layers)
        )

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def _run_one_layer(self, cell, xs):
        """
        对单层 LSTM 执行整段序列扫描。

        xs:
            [T, hidden_dim]

        return:
            hs:
            [T, hidden_dim]
        """
        xs = _ensure_seq_repr(xs)

        h0 = jnp.zeros(
            (self.hidden_dim,),
            dtype=xs.dtype,
        )
        c0 = jnp.zeros(
            (self.hidden_dim,),
            dtype=xs.dtype,
        )

        def scan_step(carry, x_t):
            new_carry = cell(
                x_t,
                carry,
            )
            h_t, _c_t = new_carry
            return new_carry, h_t

        (_h_last, _c_last), hs = jax.lax.scan(
            scan_step,
            (h0, c0),
            xs,
        )

        return hs

    def __call__(self, x, state, key):
        """
        与 LRU / Transformer backbone 保持统一接口：

            seq_repr, new_state = backbone(x, state, key)

        输入:
            x: [T, C]

        输出:
            seq_repr: [T, hidden_dim]

        LSTMCell 本身不使用外部 Equinox State；
        为兼容现有训练框架，state 原样返回。
        """
        del key

        x = _ensure_seq_repr(x)

        # [T, C] -> [T, hidden_dim]
        x = jax.vmap(self.input_proj)(x)

        # 堆叠多层单向 LSTM。
        for cell in self.cells:
            x = self._run_one_layer(
                cell,
                x,
            )

        return x, state

class HiddenStatePooledRegressor(eqx.Module):
    backbone: object
    norm: eqx.nn.LayerNorm
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear
    pool_mode: str

    def __init__(self, backbone, repr_dim, mlp_hidden_dim, pool_mode="mean", *, key):
        self.backbone = backbone
        self.pool_mode = str(pool_mode)

        pooled_dim = int(repr_dim) * (2 if str(pool_mode) == "meanmax" else 1)

        k1, k2 = jr.split(key, 2)
        self.norm = eqx.nn.LayerNorm((pooled_dim,))
        self.fc1 = eqx.nn.Linear(pooled_dim, int(mlp_hidden_dim), key=k1)
        self.fc2 = eqx.nn.Linear(int(mlp_hidden_dim), 1, key=k2)

    def __call__(self, x, state, key):
        seq_repr, new_state = self.backbone(x, state, key)
        pooled = pool_sequence_representation(seq_repr, mode=self.pool_mode)
        pooled = self.norm(pooled)
        h = self.fc1(pooled)
        h = jax.nn.gelu(h)
        y = self.fc2(h).reshape(())
        return y, new_state

def create_ssm_model(model_name, data_dim, label_dim, config, seed=400):
    "create_ssm_model：用于本文件保留的 lru / lstm / transformer 模型。"
    model_name = normalize_model_name(model_name)
    key = jr.PRNGKey(seed)
    key_backbone, key_head = jr.split(key, 2)
    repr_dim = int(config.get("repr_dim", config.get("hidden_dim", config.get("d_model", 128))))
    mlp_hidden_dim = int(config.get("mlp_hidden_dim", repr_dim))
    pool_mode = str(config.get("pool", READOUT_POOL))
    if model_name == "lru":
        backbone = LRU(
            num_blocks=int(config["num_blocks"]),
            data_dim=int(data_dim),
            N=int(config["ssm_dim"]),
            H=int(config["hidden_dim"]),
            output_dim=int(repr_dim),
            classification=False,
            output_step=1,
            key=key_backbone,
        )
        model = HiddenStatePooledRegressor(
            backbone=backbone,
            repr_dim=int(repr_dim),
            mlp_hidden_dim=int(mlp_hidden_dim),
            pool_mode=pool_mode,
            key=key_head,
        )
        state = eqx.nn.State(model)
        return (model, state)
    if model_name == "transformer":
        backbone = TransformerSequenceBackbone(
            input_dim=int(data_dim),
            d_model=int(config["d_model"]),
            num_layers=int(config["num_layers"]),
            num_heads=int(config["num_heads"]),
            ff_dim=int(config["ff_dim"]),
            dropout=float(config["dropout"]),
            key=key_backbone,
        )
        model = HiddenStatePooledRegressor(
            backbone=backbone,
            repr_dim=int(config["d_model"]),
            mlp_hidden_dim=int(mlp_hidden_dim),
            pool_mode=pool_mode,
            key=key_head,
        )
        state = eqx.nn.State(model)
        return (model, state)
    if model_name == "lstm":
        hidden_dim = int(config["hidden_dim"])
        repr_dim_lstm = int(config.get("repr_dim", hidden_dim))
        if repr_dim_lstm != hidden_dim:
            raise ValueError(
                f"当前 LSTM 公平比较配置要求 repr_dim == hidden_dim。收到 repr_dim={repr_dim_lstm}, hidden_dim={hidden_dim}"
            )
        backbone = LSTMSequenceBackbone(
            input_dim=int(data_dim),
            hidden_dim=hidden_dim,
            num_layers=int(config["num_layers"]),
            bidirectional=bool(config.get("bidirectional", False)),
            key=key_backbone,
        )
        model = HiddenStatePooledRegressor(
            backbone=backbone,
            repr_dim=repr_dim_lstm,
            mlp_hidden_dim=int(mlp_hidden_dim),
            pool_mode=pool_mode,
            key=key_head,
        )
        state = eqx.nn.State(model)
        return (model, state)
    raise ValueError(f"不支持的 model_name: {model_name}")
# 训练与评估核心

@eqx.filter_jit
def calc_output(model, X, state, key):
    output, state = jax.vmap(
        model,
        axis_name="batch",
        in_axes=(0, None, None),
        out_axes=(0, None),
    )(X, state, key)
    return output, state

def squeeze_regression_output(y_pred):
    y_pred = jnp.asarray(y_pred)

    if y_pred.ndim == 0:
        return y_pred.reshape(1)

    if y_pred.ndim == 1:
        return y_pred

    if y_pred.ndim == 2 and y_pred.shape[1] == 1:
        return y_pred[:, 0]

    # 兜底：保留 batch 维，其余全展平后做均值
    bsz = y_pred.shape[0]
    y_pred = y_pred.reshape(bsz, -1)
    return jnp.mean(y_pred, axis=1)

def squeeze_regression_target(y_true):
    """
    回归标签整理。

    兼容两种格式：
    1) 普通标签：
        y_true shape [B]
    2) PROCESS-2 weighted loss 标签：
        y_true shape [B, 2]
        第 0 列是 target，第 1 列是 sample weight

    评估时只返回第 0 列 target。
    """
    y_true = jnp.asarray(y_true)

    if y_true.ndim == 0:
        return y_true.reshape(1)

    if y_true.ndim == 1:
        return y_true

    if y_true.ndim == 2 and y_true.shape[1] >= 1:
        return y_true[:, 0]

    return y_true.reshape(-1)

def split_regression_target_and_weight_jax(y_true):
    """
    从 y_true 中拆出 target 和 sample_weight。

    普通数据集：
        y_true = [B]
        weight = ones

    PROCESS-2：
        y_true = [B, 2]
        target = y_true[:, 0]
        weight = y_true[:, 1]
    """
    y_true = jnp.asarray(y_true)

    if y_true.ndim == 2 and y_true.shape[1] >= 2:
        target = y_true[:, 0]
        weight = y_true[:, 1]
        weight = jnp.maximum(weight, float(PROCESS2_WEIGHT_EPS))
        return target, weight

    target = squeeze_regression_target(y_true)
    weight = jnp.ones_like(target)
    return target, weight

@eqx.filter_jit
@eqx.filter_value_and_grad(has_aux=True)
def regression_loss(diff_model, static_model, X, y, state, key):
    """
    回归 loss。

    普通数据集:
        sample_weight 全 1，因此 ordinary_mse == weighted_mse，
        mixed loss 自动等价普通 MSE。

    PROCESS-2:
        y 被打包为 [target, weight]；
        使用 mixed loss:
            ordinary_mse = mean(error^2)
            weighted_mse = sum(weight * error^2) / sum(weight)
            loss = (1 - alpha) * ordinary_mse + alpha * weighted_mse
    """
    model = eqx.combine(diff_model, static_model)
    pred_y, new_state = calc_output(model, X, state, key)
    pred_y = squeeze_regression_output(pred_y)

    target, sample_weight = split_regression_target_and_weight_jax(y)

    sq_error = (pred_y - target) ** 2

    ordinary_mse = jnp.mean(sq_error)

    denom = jnp.maximum(jnp.sum(sample_weight), float(PROCESS2_WEIGHT_EPS))
    weighted_mse = jnp.sum(sample_weight * sq_error) / denom

    alpha = jnp.asarray(float(PROCESS2_MIXED_LOSS_ALPHA), dtype=ordinary_mse.dtype)
    alpha = jnp.clip(alpha, 0.0, 1.0)

    loss = (1.0 - alpha) * ordinary_mse + alpha * weighted_mse

    return loss, new_state

@eqx.filter_jit
def make_step(model, filter_spec, X, y, state, opt, opt_state, key):
    """
    单步训练更新。

    修改点：
    - 原来 opt.update(grads, opt_state) 只适用于 Adam 这类不需要 params 的优化器；
    - AdamW 需要当前参数 params 来执行 decoupled weight decay；
    - 因此这里统一传入 params=eqx.filter(model, eqx.is_inexact_array)。

    这样即使当前模型仍使用普通 Adam，也不会受影响；
    如果当前模型使用 AdamW + gradient clipping，则不会再报 params 缺失。
    """
    diff_model, static_model = eqx.partition(model, filter_spec)

    (loss_value, new_state), grads = regression_loss(
        diff_model,
        static_model,
        X,
        y,
        state,
        key,
    )

    params = eqx.filter(model, eqx.is_inexact_array)

    updates, new_opt_state = opt.update(
        grads,
        opt_state,
        params=params,
    )

    new_model = eqx.apply_updates(model, updates)

    return new_model, new_state, new_opt_state, loss_value

def compute_regression_metrics_np(y_true, y_pred, already_denormalized=False):
    """
    若 already_denormalized=False:
        输入被视为“标准化空间”的值，函数内部会先反标准化到原始 MMSE 再计算指标。
    若 already_denormalized=True:
        输入已经是原始 MMSE，不再做反标准化。
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

def predict_segment_values(model, state, dataloader, batch_size, key):
    """
    返回的 preds / labels 仍处在“标准化空间”。
    指标函数内部会反标准化。
    """
    inference_model = eqx.tree_inference(model, value=True)
    preds_all = []
    labels_all = []

    for X, y in dataloader.loop_epoch(batch_size):
        key, subkey = jr.split(key)
        y_hat, _ = calc_output(inference_model, X, state, subkey)
        preds = np.asarray(squeeze_regression_output(y_hat))
        labels = np.asarray(squeeze_regression_target(y))
        preds_all.append(preds)
        labels_all.append(labels)

    preds_all = np.concatenate(preds_all, axis=0)
    labels_all = np.concatenate(labels_all, axis=0)
    return preds_all, labels_all

def evaluate_regression_on_dataloader(model, state, dataloader, batch_size, key):
    preds, labels = predict_segment_values(model, state, dataloader, batch_size, key)
    return compute_regression_metrics_np(labels, preds)

def build_warmup_cosine_schedule(base_lr, num_steps, warmup_steps, cosine_alpha):
    ws = int(min(max(warmup_steps, 1), max(num_steps - 1, 1)))
    warmup_schedule = optax.linear_schedule(
        init_value=0.0,
        end_value=base_lr,
        transition_steps=ws,
    )
    cosine_schedule = optax.cosine_decay_schedule(
        init_value=base_lr,
        decay_steps=max(num_steps - ws, 1),
        alpha=cosine_alpha,
    )
    schedule = optax.join_schedules([warmup_schedule, cosine_schedule], boundaries=[ws])
    return schedule

def build_optimizer_for_model(model_name, schedule):
    "build_optimizer_for_model：用于本文件保留的 lru / lstm / transformer 模型。"
    model_name_norm = normalize_model_name(model_name)
    enabled_models = {
        normalize_model_name(x) for x in globals().get("OPTIMIZER_ADAMW_CLIP_MODELS", ["lru"])
    }

    if bool(globals().get("USE_ADAMW_WITH_GRAD_CLIP", True)) and model_name_norm in enabled_models:
        print(
            f"[optimizer] {model_name_norm}: AdamW + grad_clip | "
            f"clip_norm={float(GRAD_CLIP_NORM)} | weight_decay={float(OPT_WEIGHT_DECAY)}"
        )
        return optax.chain(
            optax.clip_by_global_norm(float(GRAD_CLIP_NORM)),
            optax.adamw(
                learning_rate=schedule,
                weight_decay=float(OPT_WEIGHT_DECAY),
            ),
        )

    print(f"[optimizer] {model_name_norm}: Adam")
    return optax.adam(learning_rate=schedule)

def get_selection_score(metrics_dict, objective):
    objective = str(objective).strip().lower()
    if objective == "audio_rmse":
        return float(metrics_dict["audio"]["rmse"])
    if objective == "segment_rmse":
        return float(metrics_dict["segment"]["rmse"])
    if objective == "audio_mae":
        return float(metrics_dict["audio"]["mae"])
    if objective == "segment_mae":
        return float(metrics_dict["segment"]["mae"])
    raise ValueError(f"不支持的 EPOCH_SELECTION_OBJECTIVE: {objective}")

def train_ssm_model(
    model_name,
    model,
    state,
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
    LogNCDE-style training loop for MMSE regression.

    official_test is mapped to dataloaders["val"] upstream and is evaluated
    repeatedly during training. However, it is monitor/reporting only:

    - official_test does not drive optimizer update;
    - official_test is not used for early stopping;
    - official_test is not used for best epoch selection;
    - official_test is not used for checkpoint selection.

    Early stopping, if enabled, uses official_train train_loss only.
    The reported model is always the last model at training stop:
        - final TARGET_EPOCHS epoch if no early stop;
        - early-stopped epoch if train_loss stops improving.
    It never restores the historical minimum train_loss model.
    """
    key = jr.PRNGKey(seed)
    filter_spec = jax.tree_util.tree_map(lambda _: True, model)
    schedule = build_warmup_cosine_schedule(lr, num_steps, warmup_steps, cosine_alpha)

    opt = build_optimizer_for_model(
        model_name=model_name,
        schedule=schedule,
    )
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

    steps_per_epoch = int(np.ceil(dataloaders["train"].size / batch_size))
    total_epochs = int(np.ceil(num_steps / max(steps_per_epoch, 1)))

    if (print_steps is None) or (str(print_steps).strip().lower() in ["", "none", "auto"]):
        effective_print_steps = int(max(steps_per_epoch, 1))
    else:
        effective_print_steps = int(max(int(print_steps), 1))

    selection_policy = str(
        globals().get("EPOCH_SELECTION_POLICY", "last_epoch_or_train_loss_early_stop")
    )
    selected_by = "last_epoch_or_train_loss_early_stop_no_test_selection"

    early_monitor = str(globals().get("EARLY_STOPPING_MONITOR", "train_loss")).strip().lower()
    if early_monitor != "train_loss":
        raise ValueError(
            f"当前 LogNCDE-style 只允许 EARLY_STOPPING_MONITOR='train_loss'，收到：{early_monitor}"
        )

    early_min_delta = float(globals().get("EARLY_STOPPING_MIN_DELTA", 1e-4))
    restore_best = bool(globals().get("EARLY_STOPPING_RESTORE_BEST", False))
    if restore_best:
        raise ValueError(
            "当前设计要求 EARLY_STOPPING_RESTORE_BEST=False；"
            "不能回滚到历史 train_loss 最优 epoch。"
        )

    history = {
        "step": [],
        "loss": [],
        "lr": [],
        "train_segment_rmse": [],
        "val_segment_rmse": [],
        "val_audio_rmse": [],
        "epoch_train_loss": [],
    }
    epoch_trace = []

    # early stopping 只用于“何时停”，不用于“恢复哪个模型”。
    best_train_loss_for_stop = float("inf")
    no_improve_epochs = 0
    stopped_early = False
    stop_reason = "max_steps_reached"

    run_time = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}"
    run_base_dir = get_output_root(
        OUTPUT_DATASET_TAG, model_name, OUTPUT_FEATURE_TAG, OUTPUT_SELECTION_TAG
    )
    run_dir = os.path.join(run_base_dir, run_time)
    os.makedirs(run_dir, exist_ok=True)

    key, loop_key = jr.split(key)
    train_loop = dataloaders["train"].loop(batch_size, key=loop_key)

    current_epoch_losses = []

    last_epoch = -1
    last_step = -1
    last_model_snapshot, last_state_snapshot = copy.deepcopy((model, state))
    last_epoch_snapshot = None

    t0 = time.time()
    print(
        f"[{model_name}] effective_print_steps={effective_print_steps} (config_print_steps={print_steps})"
    )
    print(f"[{model_name}] readout = hidden-state pooling + MLP")
    print(f"[{model_name}] target_standardize = {bool(globals().get('TARGET_STANDARDIZE', False))}")
    print(
        f"[{model_name}] active_target_normalizer = {globals().get('ACTIVE_TARGET_NORMALIZER', None)}"
    )
    print(f"[{model_name}] selection_policy = {selection_policy}")
    print(
        f"[{model_name}] official_test monitor only = {bool(globals().get('TEST_MONITOR_ONLY', True))}"
    )
    print(f"[{model_name}] early_stopping_effective = {bool(use_early_stopping)}")
    print(f"[{model_name}] early_stopping_monitor = {early_monitor}")
    print(f"[{model_name}] early_stopping_steps = {int(early_stopping_steps)}")
    print(f"[{model_name}] early_stopping_min_delta = {float(early_min_delta)}")
    print(f"[{model_name}] early_stopping_restore_best = {bool(restore_best)}")

    for step in range(1, num_steps + 1):
        X, y = next(train_loop)
        key, step_key = jr.split(key)
        model, state, opt_state, loss_value = make_step(
            model, filter_spec, X, y, state, opt, opt_state, step_key
        )

        loss_float = float(loss_value)
        current_epoch_losses.append(loss_float)

        is_epoch_end = step % steps_per_epoch == 0 or step == num_steps
        is_step_log = step % effective_print_steps == 0
        need_eval = is_step_log or is_epoch_end

        if need_eval:
            key, k1 = jr.split(key)
            train_segment_metrics = evaluate_regression_on_dataloader(
                model, state, dataloaders["train"], batch_size, k1
            )

            key, k2 = jr.split(key)
            val_preds, val_labels = predict_segment_values(
                model, state, dataloaders["val"], batch_size, k2
            )
            val_segment_metrics = compute_regression_metrics_np(val_labels, val_preds)
            val_audio_metrics = aggregate_audio_predictions(audio_ids_val, val_preds, val_labels)[
                "metrics"
            ]

            history["step"].append(step)
            history["loss"].append(loss_float)
            history["lr"].append(float(schedule(step)))
            history["train_segment_rmse"].append(float(train_segment_metrics["rmse"]))
            history["val_segment_rmse"].append(float(val_segment_metrics["rmse"]))
            history["val_audio_rmse"].append(float(val_audio_metrics["rmse"]))

            metrics_pack = {
                "segment": val_segment_metrics,
                "audio": val_audio_metrics,
            }
            monitor_score = get_selection_score(metrics_pack, epoch_selection_objective)

            if is_step_log:
                print(
                    f"[{model_name}] step={step}/{num_steps} "
                    f"epoch={int(np.ceil(step / steps_per_epoch))}/{total_epochs} "
                    f"loss={loss_float:.6f} "
                    f"lr={float(schedule(step)):.6e} "
                    f"train_seg_rmse={float(train_segment_metrics['rmse']):.4f} "
                    f"official_test_monitor_seg_rmse={float(val_segment_metrics['rmse']):.4f} "
                    f"official_test_monitor_audio_rmse={float(val_audio_metrics['rmse']):.4f} "
                    f"monitor_score={float(monitor_score):.4f} "
                    f"time={time.time()-t0:.2f}s"
                )

            if is_epoch_end:
                epoch_id = int(np.ceil(step / steps_per_epoch))
                epoch_train_loss = (
                    float(np.mean(current_epoch_losses))
                    if len(current_epoch_losses) > 0
                    else loss_float
                )
                current_epoch_losses = []

                last_epoch = int(epoch_id)
                last_step = int(step)
                last_model_snapshot, last_state_snapshot = copy.deepcopy((model, state))

                improved_train_loss = epoch_train_loss < (
                    best_train_loss_for_stop - early_min_delta
                )
                if improved_train_loss:
                    best_train_loss_for_stop = float(epoch_train_loss)
                    no_improve_epochs = 0
                else:
                    no_improve_epochs += 1

                last_epoch_snapshot = {
                    "epoch": int(epoch_id),
                    "step": int(step),
                    "epoch_train_loss": float(epoch_train_loss),
                    "val_segment_rmse": float(val_segment_metrics["rmse"]),
                    "val_segment_mae": float(val_segment_metrics["mae"]),
                    "val_segment_r2": float(val_segment_metrics["r2"]),
                    "val_audio_rmse": float(val_audio_metrics["rmse"]),
                    "val_audio_mae": float(val_audio_metrics["mae"]),
                    "val_audio_r2": float(val_audio_metrics["r2"]),
                    "val_selection_score": float(monitor_score),
                    "model_snapshot": last_model_snapshot,
                    "state_snapshot": last_state_snapshot,
                }

                history["epoch_train_loss"].append(float(epoch_train_loss))

                epoch_trace.append(
                    {
                        "epoch": int(epoch_id),
                        "step": int(step),
                        "loss": float(loss_float),
                        "epoch_train_loss": float(epoch_train_loss),
                        "best_train_loss_for_stop": float(best_train_loss_for_stop),
                        "train_loss_improved": int(improved_train_loss),
                        "no_improve_epochs": int(no_improve_epochs),
                        "lr": float(schedule(step)),
                        "train_segment_rmse": float(train_segment_metrics["rmse"]),
                        "train_segment_mae": float(train_segment_metrics["mae"]),
                        "val_segment_rmse": float(val_segment_metrics["rmse"]),
                        "val_segment_mae": float(val_segment_metrics["mae"]),
                        "val_segment_r2": float(val_segment_metrics["r2"]),
                        "val_audio_rmse": float(val_audio_metrics["rmse"]),
                        "val_audio_mae": float(val_audio_metrics["mae"]),
                        "val_audio_r2": float(val_audio_metrics["r2"]),
                        # 兼容旧字段：val_selection_score 只是 monitor score，不参与选择。
                        "epoch_selection_objective": str(epoch_selection_objective),
                        "val_selection_score": float(monitor_score),
                        "is_best_epoch": 0,
                        # 新语义字段。
                        "selection_policy": str(selection_policy),
                        "selected_by": str(selected_by),
                        "val_semantics": "official_test_monitor",
                        "early_stopping_monitor": "train_loss",
                        "early_stopping_used": int(bool(use_early_stopping)),
                        "test_monitor_used_for_selection": 0,
                        "is_selected_epoch": 0,
                    }
                )

                print(
                    f"[{model_name}] epoch={epoch_id}/{total_epochs} "
                    f"epoch_train_loss={float(epoch_train_loss):.6f} "
                    f"best_train_loss_for_stop={float(best_train_loss_for_stop):.6f} "
                    f"no_improve_epochs={int(no_improve_epochs)}/{int(early_stopping_steps)} "
                    f"official_test_monitor_segment_rmse={float(val_segment_metrics['rmse']):.4f} "
                    f"official_test_monitor_audio_rmse={float(val_audio_metrics['rmse']):.4f} "
                    f"monitor_score={float(monitor_score):.4f} "
                    f"selected_by={selected_by}"
                )

                if bool(use_early_stopping) and int(no_improve_epochs) >= int(early_stopping_steps):
                    stopped_early = True
                    stop_reason = f"train_loss_no_improve_for_{int(early_stopping_steps)}_epochs"
                    print(
                        f"[{model_name}] early stopping at epoch={epoch_id} | "
                        f"monitor=train_loss | restore_best=False | "
                        f"reported_model=last_epoch_at_stop"
                    )
                    break

    # 如果 num_steps 不是 epoch 整数倍，仍然把最终模型作为 last model。
    if last_epoch < 0:
        last_epoch = int(np.ceil(num_steps / max(steps_per_epoch, 1)))
        last_step = int(num_steps)
        last_model_snapshot, last_state_snapshot = copy.deepcopy((model, state))
        last_epoch_snapshot = {
            "epoch": int(last_epoch),
            "step": int(last_step),
            "epoch_train_loss": (
                float(np.mean(current_epoch_losses)) if len(current_epoch_losses) > 0 else np.nan
            ),
            "val_selection_score": np.nan,
            "model_snapshot": last_model_snapshot,
            "state_snapshot": last_state_snapshot,
        }

    # 标记“最终报告 epoch”：训练停止时的最后 epoch。
    for row in epoch_trace:
        row["is_selected_epoch"] = int(int(row["epoch"]) == int(last_epoch))

        # 兼容旧字段：is_best_epoch 不再表示 val/test 最优；
        # 这里把最终报告 epoch 标成 1，真实语义见 selected_by / selection_policy。
        row["is_best_epoch"] = int(int(row["epoch"]) == int(last_epoch))

        row["stopped_early"] = int(bool(stopped_early))
        row["stop_reason"] = str(stop_reason)

    # 保存停止时的最后模型。
    eqx.tree_serialise_leaves(os.path.join(run_dir, "last_model.eqx"), last_model_snapshot)
    eqx.tree_serialise_leaves(os.path.join(run_dir, "selected_last_model.eqx"), last_model_snapshot)

    # 兼容旧路径：best_model.eqx 也保存最后 epoch 模型。
    # 注意：它不是 val/test-best，也不是 train-loss-best，只是 last_model 的别名。
    eqx.tree_serialise_leaves(os.path.join(run_dir, "best_model.eqx"), last_model_snapshot)

    with open(os.path.join(run_dir, "selection_policy_note.txt"), "w", encoding="utf-8") as f:
        f.write(
            "EPOCH_SELECTION_POLICY=last_epoch_or_train_loss_early_stop\n"
            "selected_by=last_epoch_or_train_loss_early_stop_no_test_selection\n"
            "official_test is monitor/reporting only and is not used for checkpoint selection.\n"
            "early stopping monitor=train_loss\n"
            "early stopping restore_best=False\n"
            "reported model=last model at training stop\n"
        )

    np.save(os.path.join(run_dir, "steps.npy"), np.array(history["step"]))
    np.save(os.path.join(run_dir, "loss.npy"), np.array(history["loss"]))
    np.save(os.path.join(run_dir, "lr.npy"), np.array(history["lr"]))
    np.save(
        os.path.join(run_dir, "train_segment_rmse.npy"), np.array(history["train_segment_rmse"])
    )
    np.save(os.path.join(run_dir, "val_segment_rmse.npy"), np.array(history["val_segment_rmse"]))
    np.save(os.path.join(run_dir, "val_audio_rmse.npy"), np.array(history["val_audio_rmse"]))
    np.save(os.path.join(run_dir, "epoch_train_loss.npy"), np.array(history["epoch_train_loss"]))

    candidate_snapshots = []
    if last_epoch_snapshot is not None:
        candidate_snapshots = [last_epoch_snapshot]

    return {
        "model": last_model_snapshot,
        "state": last_state_snapshot,
        # 新主字段。
        "selected_model": last_model_snapshot,
        "selected_state": last_state_snapshot,
        "selected_epoch": int(last_epoch),
        "selected_step": int(last_step),
        "selected_by": str(selected_by),
        "selection_policy": str(selection_policy),
        "test_monitor_used_for_selection": 0,
        "early_stopping_monitor": "train_loss",
        "early_stopping_used": int(bool(use_early_stopping)),
        "early_stopping_steps": int(early_stopping_steps),
        "early_stopping_min_delta": float(early_min_delta),
        "early_stopping_restore_best": False,
        "stopped_early": int(bool(stopped_early)),
        "stop_reason": str(stop_reason),
        "best_train_loss_for_stop": float(best_train_loss_for_stop),
        # 兼容旧字段：不再表示 val/test-best，而是 last epoch alias。
        "best_model": last_model_snapshot,
        "best_state": last_state_snapshot,
        "best_epoch": int(last_epoch),
        "best_val_selection_score": np.nan,
        # 兼容旧字段：只保留 last epoch snapshot。
        "candidate_snapshots": candidate_snapshots,
        "last_epoch_snapshot": last_epoch_snapshot,
        "run_dir": run_dir,
        "history": history,
        "epoch_trace": epoch_trace,
        "steps_per_epoch": steps_per_epoch,
        "total_epochs": total_epochs,
        "num_steps_run": int(last_step),
    }

# ===== 8. 回归指标与候选 checkpoint 工具 =====

def evaluate_segment_metrics(segment_preds, segment_true_labels):
    """对 segment 级预测直接计算回归指标，是 `compute_regression_metrics_np` 的语义化封装。"""
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
        raise ValueError(f"未知 audio 聚合方式: {agg_method}")
    return alias_map[method]

def _resolve_audio_trim_ratio(trim_ratio=None):
    ratio = float(
        trim_ratio if trim_ratio is not None else globals().get("AUDIO_AGG_TRIM_RATIO", 0.10)
    )
    # 避免极端值导致全部被裁空
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

    # 太短时退化成普通均值
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

    raise ValueError(f"未知聚合方式: {method}")

def aggregate_audio_predictions(
    audio_ids,
    segment_preds,
    segment_true,
    agg_method=None,
    trim_ratio=None,
):
    """
    segment_preds / segment_true 输入通常仍是标准化空间。
    这里先反标准化到原始 MMSE，再按 audio_id 聚合。

    返回：
      - metrics: 当前主聚合方式（AUDIO_AGG_PRIMARY_METHOD）的结果
      - all_methods: mean / median / trimmed_mean 三组对照结果
    """
    agg_method = _resolve_audio_agg_method(agg_method)
    trim_ratio = _resolve_audio_trim_ratio(trim_ratio)

    audio_ids = np.asarray(audio_ids).astype(int).reshape(-1)
    segment_preds = _maybe_denorm_np(segment_preds).reshape(-1)
    segment_true = _maybe_denorm_np(segment_true).reshape(-1)

    if not (len(audio_ids) == len(segment_preds) == len(segment_true)):
        raise ValueError("audio_ids / segment_preds / segment_true 长度不一致。")

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
            already_denormalized=True,  # 关键：避免重复反标准化
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
    """先按音频聚合 segment 预测，再返回完整的 audio-level 预测结果与指标。"""
    return aggregate_audio_predictions(
        audio_ids=audio_ids,
        segment_preds=segment_preds,
        segment_true=segment_true_labels,
        agg_method=agg_method,
        trim_ratio=trim_ratio,
    )

def print_audio_aggregation_comparison(tag, audio_metric_result):
    "print_audio_aggregation_comparison：用于本文件保留的 lru / lstm / transformer 模型。"
    print(f"\n[{tag}] audio aggregation comparison")

    if audio_metric_result is None:
        print("  <empty>")
        return

    # ===== 情况 1：完整结构，包含 all_methods =====
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

    # ===== 情况 2：baseline 结构，只有当前主聚合结果 =====
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

def evaluate_checkpoint_regression_pipeline(
    model,
    state,
    dataloaders,
    batch_size,
    seed_base,
    audio_ids_train,
    audio_ids_val,
    audio_ids_test=None,
    run_test=False,
    audio_agg_method=None,
    audio_trim_ratio=None,
):
    """
    对某个 checkpoint 进行 segment / audio 两级评估。

    默认 run_test=False：
      - 候选 checkpoint 复算阶段只跑 train / val，不碰 test
    当 run_test=True：
      - 仅对最终选中的 checkpoint 再额外跑一次 test
    """
    train_key = jr.PRNGKey(int(seed_base) + 998)
    val_key = jr.PRNGKey(int(seed_base) + 999)

    train_preds, train_labels = predict_segment_values(
        model, state, dataloaders["train"], batch_size, train_key
    )
    val_preds, val_labels = predict_segment_values(
        model, state, dataloaders["val"], batch_size, val_key
    )

    train_segment_metrics = evaluate_segment_metrics(train_preds, train_labels)
    val_segment_metrics = evaluate_segment_metrics(val_preds, val_labels)

    train_audio_metrics = evaluate_audio_metrics(
        audio_ids_train,
        train_preds,
        train_labels,
        agg_method=audio_agg_method,
        trim_ratio=audio_trim_ratio,
    )
    val_audio_metrics = evaluate_audio_metrics(
        audio_ids_val,
        val_preds,
        val_labels,
        agg_method=audio_agg_method,
        trim_ratio=audio_trim_ratio,
    )

    result = {
        "train_preds": train_preds,
        "train_labels": train_labels,
        "val_preds": val_preds,
        "val_labels": val_labels,
        "test_preds": None,
        "test_labels": None,
        "train_segment_metrics": train_segment_metrics,
        "val_segment_metrics": val_segment_metrics,
        "test_segment_metrics": None,
        "train_audio_metrics": train_audio_metrics,
        "val_audio_metrics": val_audio_metrics,
        "test_audio_metrics": None,
    }

    if run_test:
        if ("test" not in dataloaders) or (audio_ids_test is None):
            raise ValueError("run_test=True 时，必须提供 dataloaders['test'] 和 audio_ids_test。")

        test_key = jr.PRNGKey(int(seed_base) + 1000)
        test_preds, test_labels = predict_segment_values(
            model, state, dataloaders["test"], batch_size, test_key
        )

        test_segment_metrics = evaluate_segment_metrics(test_preds, test_labels)
        test_audio_metrics = evaluate_audio_metrics(
            audio_ids_test,
            test_preds,
            test_labels,
            agg_method=audio_agg_method,
            trim_ratio=audio_trim_ratio,
        )

        result.update(
            {
                "test_preds": test_preds,
                "test_labels": test_labels,
                "test_segment_metrics": test_segment_metrics,
                "test_audio_metrics": test_audio_metrics,
            }
        )

    return result

# ===== 9. 回归诊断绘图（修复版：避免绘图时重复反标准化） =====

def plot_regression_diagnostics(
    labels,
    predictions,
    plot_title_prefix="",
    metrics=None,
    already_denormalized=True,
    bins=20,
):
    """
    绘制回归诊断图，包括：
      1) True vs Pred scatter
      2) Residual histogram

    参数
    ----
    labels / predictions:
        若 already_denormalized=True，则它们应当已经是“原始 MMSE”
        若 already_denormalized=False，则它们应当还是“标准化空间”的值
    metrics:
        可选。若传入，则直接用该 metrics 作为标题展示，不再重算。
        适合传入 val_audio_metrics / test_audio_metrics["metrics"]。
    already_denormalized:
        True  -> 输入已是原始 MMSE，不再反标准化
        False -> 输入仍在标准化空间，函数内部会先反标准化
    bins:
        残差直方图箱数
    """
    labels_np = np.asarray(labels, dtype=np.float32).reshape(-1)
    pred_np = np.asarray(predictions, dtype=np.float32).reshape(-1)

    if labels_np.shape[0] != pred_np.shape[0]:
        raise ValueError(
            f"labels 与 predictions 长度不一致: {labels_np.shape[0]} vs {pred_np.shape[0]}"
        )

    # ===== 关键修复点 =====
    # 如果外部没有传 metrics，则在这里重算；
    # 对 audio plot 而言，传入的 labels/predictions 已经是原始 MMSE，
    # 所以必须设 already_denormalized=True，避免再次反标准化。
    if metrics is None:
        metrics = compute_regression_metrics_np(
            labels_np,
            pred_np,
            already_denormalized=already_denormalized,
        )

    # 用于绘图的坐标，统一保证是原始 MMSE
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
import io
import atexit
import pprint
# 统一记录训练阶段的标准输出、错误、表格和图像。
# 在 CSV 追加工具基础上，
# 提供兼容终端和交互环境的日志系统：
#
# 目标：
# 1) 从训练阶段开始，把后续 print / 报错 / display 表格输出
#    统一追加到同一个 log 文件；
# 2) 保留标准输出和交互环境的显示行为；
# 3) 尽量保持可重复执行：重复启用日志时，会自动先卸载旧 logger，
#    再重新挂载一个新的 logger，避免重复写入同一条输出。
#
# 说明：
# - 这里的“log”是纯文本日志；
# - matplotlib 图像无法直接嵌入 txt 日志，因此这里会：
#   a) 在日志里写入“该图已显示”的说明；
#   b) 同时把图另存为 png 文件，并把保存路径写入 log；
# - pandas.DataFrame / Series / dict / list 等常见对象，如果使用 display()
#   展示，也会尽量转成可读文本后写入同一个 log。

def append_result_to_csv(row, csv_path):
    """
    追加一行结果到 CSV，并自动兼容列不一致的情况。

    处理逻辑：
    - 若目标文件不存在，则直接新建；
    - 若已存在，则先读旧表，再对新旧列取并集后追加；
    - 最终返回写回后的 DataFrame，便于 后续查看。
    """
    # 当前这一行转成 DataFrame
    new_df = pd.DataFrame([row])

    # 若文件不存在，直接写入并返回
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        new_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return new_df

    # 读取旧文件
    old_df = pd.read_csv(csv_path)

    # 兼容列不一致：取并集
    all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))

    # 对齐列
    old_df = old_df.reindex(columns=all_cols)
    new_df = new_df.reindex(columns=all_cols)

    # 追加
    out_df = pd.concat([old_df, new_df], ignore_index=True)

    # 写回
    out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return out_df
# 1) 日志路径配置
# 日志统一放到当前实验输出目录下，便于和 CSV / checkpoint / 预测结果放在一起管理。
NOTEBOOK_LOG_DIR = os.path.join(OUTPUT_EXPORT_PREFIX, "notebook_logs")
os.makedirs(NOTEBOOK_LOG_DIR, exist_ok=True)
# 使用时间戳生成“本次运行”的日志文件名。
# 好处：
# - 每次重新启动日志时都会生成新文件，不会覆盖上一轮实验的日志；
# - 从训练阶段到最终导出共享这一条路径。
NOTEBOOK_LOG_TIME_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
NOTEBOOK_OUTPUT_LOG_PATH = os.path.join(
    NOTEBOOK_LOG_DIR, f"cell17_and_after_output_{NOTEBOOK_LOG_TIME_TAG}.log"
)
# 图像单独保存到 figures 子目录，并把图片路径写入同一个 txt log。
NOTEBOOK_LOG_FIGURE_DIR = os.path.join(NOTEBOOK_LOG_DIR, f"figures_{NOTEBOOK_LOG_TIME_TAG}")
os.makedirs(NOTEBOOK_LOG_FIGURE_DIR, exist_ok=True)
# 2) 一些基础工具函数

def _safe_close_notebook_log_handle():
    """
    安全关闭旧日志文件句柄。
    重复执行此日志阶段 时会先尝试关闭旧句柄，防止文件句柄泄漏。
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
    打开日志文件句柄（追加模式）。
    这里单独封装，便于重复执行此日志阶段 时复用。
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    handle = open(log_path, "a", encoding="utf-8")
    globals()["_NOTEBOOK_LOG_FILE_HANDLE"] = handle
    return handle

def _get_notebook_log_handle():
    """
    取当前全局日志文件句柄。
    若句柄不存在或已关闭，则按当前 NOTEBOOK_OUTPUT_LOG_PATH 重新打开。
    """
    handle = globals().get("_NOTEBOOK_LOG_FILE_HANDLE", None)
    if handle is None or getattr(handle, "closed", False):
        handle = _open_notebook_log_handle(NOTEBOOK_OUTPUT_LOG_PATH)
    return handle

def _write_notebook_log_raw(text):
    """
    往 txt log 里写原始文本，不做额外格式包装。

    注意：
    - 这里不会额外 print 到终端；
    - 仅负责写文件。
    """
    if text is None:
        return
    handle = _get_notebook_log_handle()
    handle.write(str(text))
    handle.flush()

def write_notebook_log_line(text=""):
    """
    往日志里写一整行文本。
    会自动补一个换行，便于后续 grep / 检索 / 人工查看。
    """
    _write_notebook_log_raw(f"{text}\n")

def notebook_log_section(title):
    """
    在同一个 log 中插入一个明显的“分节标题”。

    用途：
    - 标记“当前开始执行哪个 cell”；
    - 让后续训练日志、DataFrame 输出、报错信息更容易定位。
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
    把 display() 展示的对象尽量转成适合写入纯文本 log 的字符串。

    处理优先级：
    1) DataFrame / Series：转成表格文本；
    2) dict / list / tuple / set：使用 pprint 美化；
    3) matplotlib Figure：记录图标题、axes 数量等摘要信息；
    4) 其他对象：退回 repr(obj)。
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
# 3) stdout / stderr 双写（Tee）

class NotebookLogTee(io.TextIOBase):
    """
    一个简单的 tee stream：
    - 一份仍然写到 原始 stdout / stderr；
    - 另一份追加写入 NOTEBOOK_OUTPUT_LOG_PATH。

    这样做的好处是：
    - 你在 终端或交互界面里仍然能实时看到 print；
    - 同时所有 print / traceback 都会保存在统一 log 中。
    """

    def __init__(self, original_stream, stream_name="STDOUT"):
        super().__init__()
        self.original_stream = original_stream
        self.stream_name = stream_name
        self._is_notebook_log_tee = True

    def write(self, s):
        # 先保留 notebook 原始显示行为
        if self.original_stream is not None:
            self.original_stream.write(s)

        # 再写入统一日志文件
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
    如果上一次已经挂过 logger，这里先恢复原始 stdout / stderr / display / plt.show，
    再重新安装新 logger，避免“同一条输出被重复写多次”。
    """
    # 恢复 stdout
    if getattr(sys.stdout, "_is_notebook_log_tee", False):
        try:
            sys.stdout = sys.stdout.original_stream
        except Exception:
            pass

    # 恢复 stderr
    if getattr(sys.stderr, "_is_notebook_log_tee", False):
        try:
            sys.stderr = sys.stderr.original_stream
        except Exception:
            pass

    # 恢复 display
    if globals().get("_NOTEBOOK_LOGGER_DISPLAY_PATCHED", False):
        if "_NOTEBOOK_LOG_ORIGINAL_DISPLAY" in globals():
            globals()["display"] = globals()["_NOTEBOOK_LOG_ORIGINAL_DISPLAY"]
        globals()["_NOTEBOOK_LOGGER_DISPLAY_PATCHED"] = False

    # 恢复 plt.show
    if globals().get("_NOTEBOOK_LOGGER_PLT_SHOW_PATCHED", False):
        try:
            import matplotlib.pyplot as plt

            if "_NOTEBOOK_LOG_ORIGINAL_PLT_SHOW" in globals():
                plt.show = globals()["_NOTEBOOK_LOG_ORIGINAL_PLT_SHOW"]
        except Exception:
            pass
        globals()["_NOTEBOOK_LOGGER_PLT_SHOW_PATCHED"] = False

    # 关闭旧文件句柄
    _safe_close_notebook_log_handle()
# 4) display() 与 plt.show() 的补充日志

def _patch_display_for_logging():
    """
    包装 display()：
    - 页面里仍然正常显示；
    - 同时把 display 对象的文本化结果写入同一个 log。
    """
    try:
        from IPython.display import display as ipy_display
    except Exception:
        # 普通 Python 环境可不安装 IPython。
        globals()["display"] = print
        return

    original_display = globals().get("display", None)
    if original_display is None:
        original_display = ipy_display

    globals()["_NOTEBOOK_LOG_ORIGINAL_DISPLAY"] = original_display

    def logged_display(*objs, **kwargs):
        # 先保持 notebook 的正常渲染行为
        original_display(*objs, **kwargs)

        # 再把对象摘要/文本写入 log
        if len(objs) == 0:
            write_notebook_log_line("[display] <no object>")
            return

        write_notebook_log_line("[display] 输出对象如下：")
        for idx, obj in enumerate(objs, start=1):
            write_notebook_log_line(f"[display][obj_{idx}] type={type(obj).__name__}")
            write_notebook_log_line(_format_object_for_log(obj))
            write_notebook_log_line("-" * 80)

    globals()["display"] = logged_display
    globals()["_NOTEBOOK_LOGGER_DISPLAY_PATCHED"] = True

def _patch_matplotlib_show_for_logging():
    """
    包装 plt.show()：
    - 仍然正常在 当前界面中显示图；
    - 额外把当前待显示图像保存为 png；
    - 并把图像保存路径写入 txt log。
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    original_show = plt.show
    globals()["_NOTEBOOK_LOG_ORIGINAL_PLT_SHOW"] = original_show

    def logged_show(*args, **kwargs):
        fig_nums = list(plt.get_fignums())

        # 在真正 show 之前，把当前所有 figure 存盘并写日志。
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
                    f"[figure] 已保存图像 | fig_num={fig_num} | title={fig_title!r} | path={fig_path}"
                )
            except Exception as e:
                write_notebook_log_line(
                    f"[figure] 图像保存失败 | fig_num={fig_num} | error={repr(e)}"
                )

        return original_show(*args, **kwargs)

    plt.show = logged_show
    globals()["_NOTEBOOK_LOGGER_PLT_SHOW_PATCHED"] = True
# 5) 启动日志系统

def enable_notebook_output_logging():
    """
    启动从“当前阶段 之后”开始生效的 notebook 统一日志系统。

    生效范围：
    - print(...) -> 写 终端或交互界面 + 追加到同一个 txt log
    - 报错 traceback -> 写 终端或交互界面 + 追加到同一个 txt log
    - display(df / dict / list / series / figure 摘要) -> 追加到同一个 txt log
    - plt.show() -> 额外保存 png，并把路径写入同一个 txt log
    """
    # 先卸载旧 logger，避免重复 patch
    _restore_previous_notebook_logger_if_needed()

    # 打开新的日志文件
    _open_notebook_log_handle(NOTEBOOK_OUTPUT_LOG_PATH)

    # 记录启动头
    write_notebook_log_line("#" * 100)
    write_notebook_log_line("# Training output log")
    write_notebook_log_line(f"# start_time = {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    write_notebook_log_line(f"# log_path   = {NOTEBOOK_OUTPUT_LOG_PATH}")
    write_notebook_log_line(f"# figure_dir = {NOTEBOOK_LOG_FIGURE_DIR}")
    write_notebook_log_line("#" * 100)

    # 保存原始 stdout / stderr，随后挂 tee
    globals()["_NOTEBOOK_LOG_ORIGINAL_STDOUT"] = sys.stdout
    globals()["_NOTEBOOK_LOG_ORIGINAL_STDERR"] = sys.stderr

    sys.stdout = NotebookLogTee(globals()["_NOTEBOOK_LOG_ORIGINAL_STDOUT"], "STDOUT")
    sys.stderr = NotebookLogTee(globals()["_NOTEBOOK_LOG_ORIGINAL_STDERR"], "STDERR")

    # 补丁 display / plt.show
    _patch_display_for_logging()
    _patch_matplotlib_show_for_logging()

    # 注册退出时自动关闭文件句柄
    atexit.register(_safe_close_notebook_log_handle)

    # 给本次日志打一个总标题
    notebook_log_section("日志系统已启动：从训练阶段开始统一写入同一个 log")

    # 在当前输出界面提示日志位置，便于直接打开
    print("[RunLog] 已启用统一输出日志。")
    print("[RunLog] log_path   =", NOTEBOOK_OUTPUT_LOG_PATH)
    print("[RunLog] figure_dir =", NOTEBOOK_LOG_FIGURE_DIR)
# 立即启动。
# 这样只要启用日志，后续训练与导出的输出都会自动落到同一个 log。
enable_notebook_output_logging()

def save_epoch_curve_outputs(seed_epoch_df, export_prefix, steps_per_epoch):
    """按相同 epoch 汇总多个 seed 的均值、样本标准差和标准误。"""
    if seed_epoch_df.empty:
        return
    trace = seed_epoch_df.copy()
    metadata = {
        "feature_model_tag": str(FEATURE_MODEL_TAG),
        "feature_selection_tag": str(FEATURE_SELECTION_TAG),
        "feature_source_mode": str(ACTIVE_SOURCE_MODE),
        "feature_file_tag": str(FILE_TAG),
        "steps_per_epoch": int(steps_per_epoch),
        "monitor_semantics": "official_test_monitor",
    }
    for name, value in metadata.items():
        trace[name] = value
    trace["monitor_audio_rmse"] = trace["val_audio_rmse"]
    trace["monitor_segment_rmse"] = trace["val_segment_rmse"]
    trace["epoch_float"] = trace["step"] / float(max(steps_per_epoch, 1))
    trace.to_csv(export_prefix + "_epoch_trace.csv", index=False, encoding="utf-8-sig")

    group_cols = [
        "dataset", "feature_model_tag", "feature_selection_tag",
        "feature_source_mode", "feature_file_tag", "model", "epoch",
    ]
    metrics = [
        "step", "epoch_float", "loss", "lr", "epoch_train_loss",
        "train_segment_rmse", "val_segment_rmse", "val_audio_rmse",
        "monitor_segment_rmse", "monitor_audio_rmse",
    ]
    for col in metrics:
        trace[col] = pd.to_numeric(trace[col], errors="coerce")
    summary = trace.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std", "count"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    for col in metrics:
        counts = summary[f"{col}_count"]
        # 单个有效 seed 的 std / sem 置 0；全部缺失的指标保持 NaN。
        summary.loc[counts == 1, f"{col}_std"] = 0.0
        summary[f"{col}_sem"] = summary[f"{col}_std"] / np.sqrt(counts.replace(0, np.nan))
    summary["n_seeds"] = summary["monitor_audio_rmse_count"].astype(int)
    summary_path = export_prefix + "_epoch_rmse_summary_by_epoch.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    plot_cols = group_cols + ["step_mean", "epoch_float_mean", "n_seeds"]
    for col in ["monitor_audio_rmse", "monitor_segment_rmse", "train_segment_rmse"]:
        plot_cols.extend(f"{col}_{stat}" for stat in ["mean", "std", "sem"])
    plot_path = export_prefix + "_epoch_curve_plot_ready.csv"
    summary[plot_cols].to_csv(plot_path, index=False, encoding="utf-8-sig")
    print("epoch RMSE summary saved to:", summary_path)
    print("epoch curve plot-ready CSV saved to:", plot_path)

def _metric_or_nan(metric_block, key):
    if metric_block is None:
        return np.nan
    return float(metric_block.get(key, np.nan))

def save_single_run_outputs(
    model_name,
    seed,
    selected_epoch,
    selected_pipeline,
    run_dir=None,
    selection_policy="last_epoch_or_train_loss_early_stop",
    selected_by="last_epoch_or_train_loss_early_stop_no_test_selection",
    test_monitor_used_for_selection=0,
    early_stopping_monitor="train_loss",
    early_stopping_used=0,
    stopped_early=0,
    stop_reason="max_steps_reached",
):
    "save_single_run_outputs：用于本文件保留的 lru / lstm / transformer 模型。"
    os.makedirs(OUTPUT_EXPORT_PREFIX, exist_ok=True)

    model_name = normalize_model_name(model_name)
    export_prefix = os.path.join(
        OUTPUT_EXPORT_PREFIX, f"{DATASET_NAME}_{model_name}_seed{int(seed)}"
    )

    train_segment_metrics = selected_pipeline.get("train_segment_metrics")
    val_segment_metrics = selected_pipeline.get("val_segment_metrics")
    test_segment_metrics = selected_pipeline.get("test_segment_metrics")

    train_audio_metrics = (
        None
        if selected_pipeline.get("train_audio_metrics") is None
        else selected_pipeline["train_audio_metrics"]["metrics"]
    )
    val_audio_metrics = (
        None
        if selected_pipeline.get("val_audio_metrics") is None
        else selected_pipeline["val_audio_metrics"]["metrics"]
    )
    test_audio_metrics = (
        None
        if selected_pipeline.get("test_audio_metrics") is None
        else selected_pipeline["test_audio_metrics"]["metrics"]
    )

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": DATASET_NAME,
        "model": model_name,
        "split_protocol": "logncde_style_official_train_as_train_official_test_as_monitor_train_loss_stop_last_epoch",
        "train_semantics": "official_train",
        "val_semantics": "official_test_monitor",
        "test_semantics": "official_test_alias",
        "seed": int(seed),
        "selected_epoch": int(selected_epoch),
        "selection_policy": str(selection_policy),
        "selected_by": str(selected_by),
        "test_monitor_used_for_selection": int(test_monitor_used_for_selection),
        "early_stopping_monitor": str(early_stopping_monitor),
        "early_stopping_used": int(early_stopping_used),
        "stopped_early": int(stopped_early),
        "stop_reason": str(stop_reason),
        "run_dir": "" if run_dir is None else str(run_dir),
        "train_segment_rmse": _metric_or_nan(train_segment_metrics, "rmse"),
        "train_segment_mae": _metric_or_nan(train_segment_metrics, "mae"),
        "train_segment_r2": _metric_or_nan(train_segment_metrics, "r2"),
        "train_segment_pearson": _metric_or_nan(train_segment_metrics, "pearson"),
        "val_segment_rmse": _metric_or_nan(val_segment_metrics, "rmse"),
        "val_segment_mae": _metric_or_nan(val_segment_metrics, "mae"),
        "val_segment_r2": _metric_or_nan(val_segment_metrics, "r2"),
        "val_segment_pearson": _metric_or_nan(val_segment_metrics, "pearson"),
        "train_audio_rmse": _metric_or_nan(train_audio_metrics, "rmse"),
        "train_audio_mae": _metric_or_nan(train_audio_metrics, "mae"),
        "train_audio_r2": _metric_or_nan(train_audio_metrics, "r2"),
        "train_audio_pearson": _metric_or_nan(train_audio_metrics, "pearson"),
        "val_audio_rmse": _metric_or_nan(val_audio_metrics, "rmse"),
        "val_audio_mae": _metric_or_nan(val_audio_metrics, "mae"),
        "val_audio_r2": _metric_or_nan(val_audio_metrics, "r2"),
        "val_audio_pearson": _metric_or_nan(val_audio_metrics, "pearson"),
    }

    if (
        EVAL_OFFICIAL_TEST
        and (test_segment_metrics is not None)
        and (test_audio_metrics is not None)
    ):
        row.update(
            {
                "test_segment_rmse": _metric_or_nan(test_segment_metrics, "rmse"),
                "test_segment_mae": _metric_or_nan(test_segment_metrics, "mae"),
                "test_segment_r2": _metric_or_nan(test_segment_metrics, "r2"),
                "test_segment_pearson": _metric_or_nan(test_segment_metrics, "pearson"),
                "test_audio_rmse": _metric_or_nan(test_audio_metrics, "rmse"),
                "test_audio_mae": _metric_or_nan(test_audio_metrics, "mae"),
                "test_audio_r2": _metric_or_nan(test_audio_metrics, "r2"),
                "test_audio_pearson": _metric_or_nan(test_audio_metrics, "pearson"),
            }
        )

    single_csv_path = export_prefix + "_single_run_metrics.csv"
    single_df = append_result_to_csv(row, single_csv_path)
    print("single run CSV saved to:", single_csv_path)

    val_audio_pred_path = None
    if selected_pipeline.get("val_audio_metrics") is not None:
        val_audio_pred_df = pd.DataFrame(
            {
                "audio_id": selected_pipeline["val_audio_metrics"]["sorted_audio_ids"],
                "true_mmse": selected_pipeline["val_audio_metrics"]["y_true_audio"],
                "pred_mmse": selected_pipeline["val_audio_metrics"]["y_pred_audio"],
            }
        )
        val_audio_pred_path = export_prefix + "_val_audio_predictions.csv"
        val_audio_pred_df.to_csv(val_audio_pred_path, index=False, encoding="utf-8-sig")
        print("official_test monitor audio prediction CSV saved to:", val_audio_pred_path)

    test_audio_pred_path = None
    if EVAL_OFFICIAL_TEST and (selected_pipeline.get("test_audio_metrics") is not None):
        test_audio_pred_df = pd.DataFrame(
            {
                "audio_id": selected_pipeline["test_audio_metrics"]["sorted_audio_ids"],
                "true_mmse": selected_pipeline["test_audio_metrics"]["y_true_audio"],
                "pred_mmse": selected_pipeline["test_audio_metrics"]["y_pred_audio"],
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
        "selection_policy": str(selection_policy),
        "selected_by": str(selected_by),
        "test_monitor_used_for_selection": int(test_monitor_used_for_selection),
        "early_stopping_monitor": str(early_stopping_monitor),
        "early_stopping_used": int(early_stopping_used),
        "stopped_early": int(stopped_early),
        "stop_reason": str(stop_reason),
        "selected_pipeline": selected_pipeline,
        "run_dir": run_dir,
        "val_audio_predictions_path": val_audio_pred_path,
        "test_audio_predictions_path": test_audio_pred_path,
    }

    return {
        "single_row": row,
        "single_df": single_df,
        "single_csv_path": single_csv_path,
        "val_audio_pred_path": val_audio_pred_path,
        "test_audio_pred_path": test_audio_pred_path,
        "context": context,
    }

# ===== 运行 baseline =====
notebook_log_section("单模型运行（MMSE 回归版）")
if not RUN_SINGLE_SEED_400:
    print("[SKIP] RUN_SINGLE_SEED_400=False，跳过 lru / transformer / lstm 的 seed=400 单跑。")
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
    print("[single_run] presplit ds_dir =", single_run_ds_dir)
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
        f"train plan => steps_per_epoch={steps_per_epoch}, warmup_steps={effective_warmup_steps}, num_steps_run={num_steps_run}, total_epochs={total_epochs_run}, use_early_stopping={USE_EARLY_STOPPING}"
    )
    model_cfg = dict(DEFAULT_MODEL_CONFIG[MODEL_NAME])
    model, state = create_ssm_model(
        MODEL_NAME, dataset.data_dim, dataset.label_dim, model_cfg, seed=RANDOM_SEED
    )
    if MODEL_NAME == "lru":
        base_lr = LR_LRU
    elif MODEL_NAME == "transformer":
        base_lr = LR_TRANSFORMER
    elif MODEL_NAME == "lstm":
        base_lr = LR_LSTM
    else:
        raise ValueError(f"单模型运行不支持的 MODEL_NAME: {MODEL_NAME}")
    train_result = train_ssm_model(
        model_name=MODEL_NAME,
        model=model,
        state=state,
        dataloaders=dataloaders,
        lr=base_lr,
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
    selected_model = train_result["selected_model"]
    selected_state = train_result["selected_state"]
    selected_epoch = int(train_result["selected_epoch"])
    epoch_decision_info = {
        "selected_epoch": int(selected_epoch),
        "selected_by": str(
            train_result.get("selected_by", "last_epoch_or_train_loss_early_stop_no_test_selection")
        ),
        "selection_policy": str(
            train_result.get("selection_policy", "last_epoch_or_train_loss_early_stop")
        ),
        "test_monitor_used_for_selection": int(
            train_result.get("test_monitor_used_for_selection", 0)
        ),
        "early_stopping_monitor": str(train_result.get("early_stopping_monitor", "train_loss")),
        "early_stopping_used": int(train_result.get("early_stopping_used", 0)),
        "stopped_early": int(train_result.get("stopped_early", 0)),
        "stop_reason": str(train_result.get("stop_reason", "max_steps_reached")),
    }
    print("\n[Flow] LogNCDE-style last-epoch / train-loss-early-stop reporting")
    print(
        f"[Flow] selected_epoch={epoch_decision_info['selected_epoch']} selected_by={epoch_decision_info['selected_by']} selection_policy={epoch_decision_info['selection_policy']} test_monitor_used_for_selection={epoch_decision_info['test_monitor_used_for_selection']} early_stopping_monitor={epoch_decision_info['early_stopping_monitor']} early_stopping_used={epoch_decision_info['early_stopping_used']} stopped_early={epoch_decision_info['stopped_early']} stop_reason={epoch_decision_info['stop_reason']}"
    )
    selected_pipeline = evaluate_checkpoint_regression_pipeline(
        model=selected_model,
        state=selected_state,
        dataloaders=dataloaders,
        batch_size=BATCH_SIZE,
        seed_base=int(RANDOM_SEED) + 9000 + selected_epoch,
        audio_ids_train=audio_ids_train,
        audio_ids_val=audio_ids_val,
        audio_ids_test=audio_ids_test,
        run_test=EVAL_OFFICIAL_TEST,
        audio_agg_method=AUDIO_AGG_PRIMARY_METHOD,
        audio_trim_ratio=AUDIO_AGG_TRIM_RATIO,
    )
    train_segment_metrics = selected_pipeline["train_segment_metrics"]
    val_segment_metrics = selected_pipeline["val_segment_metrics"]
    train_audio_metrics = selected_pipeline["train_audio_metrics"]["metrics"]
    val_audio_metrics = selected_pipeline["val_audio_metrics"]["metrics"]
    print("\n[Selected checkpoint metrics]")
    print("train_segment_metrics =", train_segment_metrics)
    print("official_test_monitor_segment_metrics   =", val_segment_metrics)
    print("train_audio_metrics   =", train_audio_metrics)
    print("official_test_monitor_audio_metrics     =", val_audio_metrics)
    if EVAL_OFFICIAL_TEST and selected_pipeline.get("test_segment_metrics") is not None:
        print("test_segment_metrics  =", selected_pipeline["test_segment_metrics"])
    if EVAL_OFFICIAL_TEST and selected_pipeline.get("test_audio_metrics") is not None:
        print("test_audio_metrics    =", selected_pipeline["test_audio_metrics"]["metrics"])
    print("\n[Plots] Official test monitor audio diagnostics")
    plot_regression_diagnostics(
        labels=selected_pipeline["val_audio_metrics"]["y_true_audio"],
        predictions=selected_pipeline["val_audio_metrics"]["y_pred_audio"],
        plot_title_prefix=f"{DATASET_NAME} | {MODEL_NAME} | Val Audio",
        metrics=val_audio_metrics,
        already_denormalized=True,
    )
    if EVAL_OFFICIAL_TEST and selected_pipeline.get("test_audio_metrics") is not None:
        print("\n[Plots] Test audio diagnostics")
        plot_regression_diagnostics(
            labels=selected_pipeline["test_audio_metrics"]["y_true_audio"],
            predictions=selected_pipeline["test_audio_metrics"]["y_pred_audio"],
            plot_title_prefix=f"{DATASET_NAME} | {MODEL_NAME} | Test Audio",
            metrics=selected_pipeline["test_audio_metrics"]["metrics"],
            already_denormalized=True,
        )
    save_info = save_single_run_outputs(
        model_name=MODEL_NAME,
        seed=RANDOM_SEED,
        selected_epoch=selected_epoch,
        selected_pipeline=selected_pipeline,
        run_dir=train_result["run_dir"],
        selection_policy=epoch_decision_info["selection_policy"],
        selected_by=epoch_decision_info["selected_by"],
        test_monitor_used_for_selection=epoch_decision_info["test_monitor_used_for_selection"],
        early_stopping_monitor=epoch_decision_info["early_stopping_monitor"],
        early_stopping_used=epoch_decision_info["early_stopping_used"],
        stopped_early=epoch_decision_info["stopped_early"],
        stop_reason=epoch_decision_info["stop_reason"],
    )
    LAST_SINGLE_RUN_CONTEXT = save_info["context"]
    display(save_info["single_df"].tail(5))
    print_audio_aggregation_comparison(
        "Official test monitor", selected_pipeline["val_audio_metrics"]
    )
    if EVAL_OFFICIAL_TEST and selected_pipeline.get("test_audio_metrics") is not None:
        print_audio_aggregation_comparison("Test", selected_pipeline["test_audio_metrics"])
notebook_log_section("跑多 seed（lru / transformer / lstm）")
if not should_run_multi_seed_for_model(MODEL_NAME):
    print(f"[SKIP] 当前 active model = {MODEL_NAME}，不执行多 seed。")
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
        f"multi-seed train plan => steps_per_epoch={steps_per_epoch_ms}, warmup_steps={effective_warmup_steps_ms}, num_steps_run={num_steps_run_ms}, total_epochs={total_epochs_run_ms}, use_early_stopping={USE_EARLY_STOPPING}"
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
    print("[multi_seed] presplit ds_dir =", multi_seed_ds_dir)
    audio_ids_train_ms = np.asarray(pickle.load(open(idx_train_pkl_ms, "rb"))).astype(int)
    audio_ids_val_ms = np.asarray(pickle.load(open(idx_val_pkl_ms, "rb"))).astype(int)
    audio_ids_test_ms = np.asarray(pickle.load(open(idx_test_pkl_ms, "rb"))).astype(int)
    seed_epoch_rows = []
    seed_best_rows = []
    multi_run_contexts = []
    model_name_ms = MODEL_NAME
    model_cfg_ms = dict(DEFAULT_MODEL_CONFIG[model_name_ms])
    if model_name_ms == "lru":
        base_lr_ms = LR_LRU
    elif model_name_ms == "transformer":
        base_lr_ms = LR_TRANSFORMER
    elif model_name_ms == "lstm":
        base_lr_ms = LR_LSTM
    else:
        raise ValueError(f"当前多 seed 主流程不支持模型: {model_name_ms}")
    for seed_ms in SEEDS_TO_RUN:
        print(f"\n[MultiSeed] start model={model_name_ms} seed={seed_ms}")
        model_ms, state_ms = create_ssm_model(
            model_name_ms, dataset_ms.data_dim, dataset_ms.label_dim, model_cfg_ms, seed=seed_ms
        )
        train_result_ms = train_ssm_model(
            model_name=model_name_ms,
            model=model_ms,
            state=state_ms,
            dataloaders=dataloaders_ms,
            lr=base_lr_ms,
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
        for em in train_result_ms["epoch_trace"]:
            seed_epoch_rows.append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "dataset": DATASET_NAME,
                    "model": model_name_ms,
                    "seed": int(seed_ms),
                    "epoch": int(em["epoch"]),
                    "step": int(em["step"]),
                    "loss": float(em["loss"]),
                    "lr": float(em["lr"]),
                    "train_segment_rmse": float(em["train_segment_rmse"]),
                    "train_segment_mae": float(em["train_segment_mae"]),
                    "val_segment_rmse": float(em["val_segment_rmse"]),
                    "val_segment_mae": float(em["val_segment_mae"]),
                    "val_segment_r2": float(em["val_segment_r2"]),
                    "val_audio_rmse": float(em["val_audio_rmse"]),
                    "val_audio_mae": float(em["val_audio_mae"]),
                    "val_audio_r2": float(em["val_audio_r2"]),
                    "epoch_selection_objective": str(
                        em.get("epoch_selection_objective", EPOCH_SELECTION_OBJECTIVE)
                    ),
                    "val_selection_score": float(em.get("val_selection_score", np.nan)),
                    "is_best_epoch": int(em.get("is_best_epoch", 0)),
                    "is_selected_epoch": int(em.get("is_selected_epoch", 0)),
                    "selection_policy": str(
                        em.get("selection_policy", "last_epoch_or_train_loss_early_stop")
                    ),
                    "selected_by": str(
                        em.get(
                            "selected_by", "last_epoch_or_train_loss_early_stop_no_test_selection"
                        )
                    ),
                    "val_semantics": str(em.get("val_semantics", "official_test_monitor")),
                    "early_stopping_monitor": str(em.get("early_stopping_monitor", "train_loss")),
                    "early_stopping_used": int(em.get("early_stopping_used", 0)),
                    "test_monitor_used_for_selection": int(
                        em.get("test_monitor_used_for_selection", 0)
                    ),
                    "epoch_train_loss": float(em.get("epoch_train_loss", np.nan)),
                    "best_train_loss_for_stop": float(em.get("best_train_loss_for_stop", np.nan)),
                    "train_loss_improved": int(em.get("train_loss_improved", 0)),
                    "no_improve_epochs": int(em.get("no_improve_epochs", 0)),
                    "stopped_early": int(em.get("stopped_early", 0)),
                    "stop_reason": str(em.get("stop_reason", "")),
                    "run_dir": train_result_ms["run_dir"],
                }
            )
        selected_epoch_ms = int(train_result_ms["selected_epoch"])
        selected_by_ms = str(
            train_result_ms.get(
                "selected_by", "last_epoch_or_train_loss_early_stop_no_test_selection"
            )
        )
        selection_policy_ms = str(
            train_result_ms.get("selection_policy", "last_epoch_or_train_loss_early_stop")
        )
        test_monitor_used_for_selection_ms = int(
            train_result_ms.get("test_monitor_used_for_selection", 0)
        )
        early_stopping_monitor_ms = str(train_result_ms.get("early_stopping_monitor", "train_loss"))
        early_stopping_used_ms = int(train_result_ms.get("early_stopping_used", 0))
        stopped_early_ms = int(train_result_ms.get("stopped_early", 0))
        stop_reason_ms = str(train_result_ms.get("stop_reason", "max_steps_reached"))
        selected_pipeline_ms = evaluate_checkpoint_regression_pipeline(
            model=train_result_ms["selected_model"],
            state=train_result_ms["selected_state"],
            dataloaders=dataloaders_ms,
            batch_size=BATCH_SIZE,
            seed_base=int(seed_ms) + 9000 + int(selected_epoch_ms),
            audio_ids_train=audio_ids_train_ms,
            audio_ids_val=audio_ids_val_ms,
            audio_ids_test=audio_ids_test_ms,
            run_test=EVAL_OFFICIAL_TEST,
            audio_agg_method=AUDIO_AGG_PRIMARY_METHOD,
            audio_trim_ratio=AUDIO_AGG_TRIM_RATIO,
        )
        print(
            f"[MultiSeed] seed={seed_ms} selected_epoch={selected_epoch_ms} selected_by={selected_by_ms} selection_policy={selection_policy_ms} test_monitor_used_for_selection={test_monitor_used_for_selection_ms} early_stopping_monitor={early_stopping_monitor_ms} early_stopping_used={early_stopping_used_ms} stopped_early={stopped_early_ms} stop_reason={stop_reason_ms}"
        )
        row = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": DATASET_NAME,
            "model": model_name_ms,
            "split_protocol": "logncde_style_official_train_as_train_official_test_as_monitor_train_loss_stop_last_epoch",
            "train_semantics": "official_train",
            "val_semantics": "official_test_monitor",
            "test_semantics": "official_test_alias",
            "seed": int(seed_ms),
            "selected_epoch": int(selected_epoch_ms),
            "selection_policy": str(selection_policy_ms),
            "selected_by": str(selected_by_ms),
            "test_monitor_used_for_selection": int(test_monitor_used_for_selection_ms),
            "early_stopping_monitor": str(early_stopping_monitor_ms),
            "early_stopping_used": int(early_stopping_used_ms),
            "stopped_early": int(stopped_early_ms),
            "stop_reason": str(stop_reason_ms),
            "val_segment_rmse": float(selected_pipeline_ms["val_segment_metrics"]["rmse"]),
            "val_segment_mae": float(selected_pipeline_ms["val_segment_metrics"]["mae"]),
            "val_segment_r2": float(selected_pipeline_ms["val_segment_metrics"]["r2"]),
            "val_segment_pearson": float(selected_pipeline_ms["val_segment_metrics"]["pearson"]),
            "val_audio_rmse": float(selected_pipeline_ms["val_audio_metrics"]["metrics"]["rmse"]),
            "val_audio_mae": float(selected_pipeline_ms["val_audio_metrics"]["metrics"]["mae"]),
            "val_audio_r2": float(selected_pipeline_ms["val_audio_metrics"]["metrics"]["r2"]),
            "val_audio_pearson": float(
                selected_pipeline_ms["val_audio_metrics"]["metrics"]["pearson"]
            ),
        }
        if EVAL_OFFICIAL_TEST and selected_pipeline_ms.get("test_audio_metrics") is not None:
            row.update(
                {
                    "test_segment_rmse": float(
                        selected_pipeline_ms["test_segment_metrics"]["rmse"]
                    ),
                    "test_segment_mae": float(selected_pipeline_ms["test_segment_metrics"]["mae"]),
                    "test_segment_r2": float(selected_pipeline_ms["test_segment_metrics"]["r2"]),
                    "test_segment_pearson": float(
                        selected_pipeline_ms["test_segment_metrics"]["pearson"]
                    ),
                    "test_audio_rmse": float(
                        selected_pipeline_ms["test_audio_metrics"]["metrics"]["rmse"]
                    ),
                    "test_audio_mae": float(
                        selected_pipeline_ms["test_audio_metrics"]["metrics"]["mae"]
                    ),
                    "test_audio_r2": float(
                        selected_pipeline_ms["test_audio_metrics"]["metrics"]["r2"]
                    ),
                    "test_audio_pearson": float(
                        selected_pipeline_ms["test_audio_metrics"]["metrics"]["pearson"]
                    ),
                }
            )
        seed_best_rows.append(row)
        multi_run_contexts.append(
            {
                "dataset": DATASET_NAME,
                "model": model_name_ms,
                "seed": int(seed_ms),
                "selected_epoch": int(selected_epoch_ms),
                "selection_policy": str(selection_policy_ms),
                "selected_by": str(selected_by_ms),
                "test_monitor_used_for_selection": int(test_monitor_used_for_selection_ms),
                "early_stopping_monitor": str(early_stopping_monitor_ms),
                "early_stopping_used": int(early_stopping_used_ms),
                "stopped_early": int(stopped_early_ms),
                "stop_reason": str(stop_reason_ms),
                "selected_pipeline": selected_pipeline_ms,
                "audio_ids_test": None if not EVAL_OFFICIAL_TEST else audio_ids_test_ms,
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
        print("epoch trace CSV saved to:", epoch_csv_ms)
        save_epoch_curve_outputs(seed_epoch_df, export_prefix_ms, steps_per_epoch_ms)
    if len(seed_best_df) > 0:
        numeric_cols = [
            "val_segment_rmse",
            "val_segment_mae",
            "val_segment_r2",
            "val_segment_pearson",
            "val_audio_rmse",
            "val_audio_mae",
            "val_audio_r2",
            "val_audio_pearson",
        ]
        if EVAL_OFFICIAL_TEST:
            numeric_cols.extend(
                [
                    "test_segment_rmse",
                    "test_segment_mae",
                    "test_segment_r2",
                    "test_segment_pearson",
                    "test_audio_rmse",
                    "test_audio_mae",
                    "test_audio_r2",
                    "test_audio_pearson",
                ]
            )
        summary_rows = []
        for model_name_ms in sorted(seed_best_df["model"].unique()):
            sub = seed_best_df[seed_best_df["model"] == model_name_ms].copy()
            mean_row = {col: "" for col in seed_best_df.columns}
            var_row = {col: "" for col in seed_best_df.columns}
            mean_row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mean_row["dataset"] = DATASET_NAME
            mean_row["model"] = model_name_ms
            mean_row["seed"] = "mean"
            var_row["time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            var_row["dataset"] = DATASET_NAME
            var_row["model"] = model_name_ms
            var_row["seed"] = "variance"
            for col in numeric_cols:
                vals = sub[col].astype(float).values
                mean_row[col] = float(np.mean(vals))
                var_row[col] = float(np.var(vals))
            summary_rows.extend([mean_row, var_row])
        out_best_df = pd.concat([seed_best_df, pd.DataFrame(summary_rows)], ignore_index=True)
        out_best_df.to_csv(best_csv_ms, index=False, encoding="utf-8-sig")
        print("best metrics CSV saved to:", best_csv_ms)
        display(out_best_df.tail(10))
    LAST_MULTI_RUN_CONTEXT = multi_run_contexts
notebook_log_section("导出 audio-level 预测与汇总结果")

# ===== 12. 导出汇总结果：默认以 val_audio 为主；test 可选 =====
contexts = []
if (
    ("LAST_MULTI_RUN_CONTEXT" in globals())
    and isinstance(LAST_MULTI_RUN_CONTEXT, list)
    and (len(LAST_MULTI_RUN_CONTEXT) > 0)
):
    contexts = LAST_MULTI_RUN_CONTEXT
    print(f"[PostExport] 使用多 seed 上下文，数量={len(contexts)}")
elif "LAST_SINGLE_RUN_CONTEXT" in globals():
    contexts = [LAST_SINGLE_RUN_CONTEXT]
    print("[PostExport] 使用单 seed 上下文")
else:
    print("[PostExport] 未检测到可用上下文，请先运行单模型或多 seed cell。")

def _safe_get_audio_method_block(audio_metric_result, method_name):
    """
    兼容新旧结构：
    - 新结构：audio_metric_result["all_methods"][method_name]
    - 旧结构：只有当前主聚合结果
    """
    if audio_metric_result is None:
        return None

    if isinstance(audio_metric_result, dict) and ("all_methods" in audio_metric_result):
        return audio_metric_result["all_methods"].get(method_name)

    # 兼容旧结构：没有 all_methods，只能返回 primary 那一组
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
    把一个 method 的 metrics 展平成单层字典列，方便写 summary CSV。
    prefix 例如：
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
    导出 audio-level 预测明细。
    每个 audio 会对应 3 行（mean / median / trimmed_mean），便于后处理比较。
    """
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"split 必须是 val 或 test，当前收到: {split}")

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
    把每个上下文（每个 seed / 每个模型）的核心指标展平成一行。

    默认：
    - 只汇总 official_test_monitor 结果（历史 val_* 列）
    - include_test=True 时，才把 test 相关列一起展开
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

        # ===== test（可选）=====
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
    导出每个 seed × 每个聚合方法的 audio-level 指标。
    重点给出 rmse，便于后续做 5-seed 平均 RMSE 汇总。
    """
    split = str(split).strip().lower()
    if split not in {"val", "test"}:
        raise ValueError(f"split 必须是 val 或 test，当前收到: {split}")

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
    按 dataset / model / agg_method 聚合，得到多 seed 平均 RMSE。
    如果你当前就是 5 个随机种子，这里 n_seeds 应该会显示 5。
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
    # ===== 默认主导出：official_test_monitor（保留 val_* 文件名） =====
    val_audio_pred_df = build_audio_prediction_df(contexts, split="val")
    summary_df = build_context_summary_df(contexts, include_test=EVAL_OFFICIAL_TEST)
    val_seed_metric_df = build_seed_metric_df(contexts, split="val")
    val_seed_mean_rmse_df = build_seed_mean_rmse_df(val_seed_metric_df)
    val_audio_pred_csv = post_prefix + "_val_audio_predictions_all_aggs.csv"
    summary_csv = post_prefix + "_summary_val_focus.csv"
    val_seed_metric_csv = post_prefix + "_val_seed_rmse_all_aggs.csv"
    val_seed_mean_rmse_csv = post_prefix + "_val_seed_rmse_mean_all_aggs.csv"
    val_audio_pred_df.to_csv(val_audio_pred_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    val_seed_metric_df.to_csv(val_seed_metric_csv, index=False, encoding="utf-8-sig")
    val_seed_mean_rmse_df.to_csv(val_seed_mean_rmse_csv, index=False, encoding="utf-8-sig")
    print("[PostExport] official_test monitor audio prediction CSV saved to:", val_audio_pred_csv)
    print("[PostExport] summary CSV saved to:", summary_csv)
    print("[PostExport] val seed RMSE CSV saved to:", val_seed_metric_csv)
    print("[PostExport] val seed mean RMSE CSV saved to:", val_seed_mean_rmse_csv)
    print("\n[PostExport] summary_df")
    display(summary_df)
    print("\n[PostExport] val seed RMSE by aggregation method")
    display(
        val_seed_metric_df[
            ["dataset", "model", "seed", "agg_method", "rmse", "mae", "r2", "pearson"]
        ]
        .sort_values(["dataset", "model", "seed", "agg_method"])
        .reset_index(drop=True)
    )
    print("\n[PostExport] val aggregation overview (multi-seed mean RMSE)")
    display(
        val_seed_mean_rmse_df[
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
    # ===== test 仅在开关打开时导出 =====
    if EVAL_OFFICIAL_TEST:
        test_audio_pred_df = build_audio_prediction_df(contexts, split="test")
        test_seed_metric_df = build_seed_metric_df(contexts, split="test")
        test_seed_mean_rmse_df = build_seed_mean_rmse_df(test_seed_metric_df)
        test_audio_pred_csv = post_prefix + "_test_audio_predictions_all_aggs.csv"
        test_seed_metric_csv = post_prefix + "_test_seed_rmse_all_aggs.csv"
        test_seed_mean_rmse_csv = post_prefix + "_test_seed_rmse_mean_all_aggs.csv"
        test_audio_pred_df.to_csv(test_audio_pred_csv, index=False, encoding="utf-8-sig")
        test_seed_metric_df.to_csv(test_seed_metric_csv, index=False, encoding="utf-8-sig")
        test_seed_mean_rmse_df.to_csv(test_seed_mean_rmse_csv, index=False, encoding="utf-8-sig")
        print("[PostExport] test audio prediction CSV saved to:", test_audio_pred_csv)
        print("[PostExport] test seed RMSE CSV saved to:", test_seed_metric_csv)
        print("[PostExport] test seed mean RMSE CSV saved to:", test_seed_mean_rmse_csv)
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
