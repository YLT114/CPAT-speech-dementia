# -*- coding: utf-8 -*-
"""CPAT Stage2：训练集 token 表征聚类与 chunk 伪标签导出。
精简版保留原参数、FAISS 配置、采样/随机调用顺序、chunk key 和保存路径。
读取 Stage1 train manifest；Pitt 按既有审计校验受试者、visit 与 wav 路径。
默认 KS=[32,64]，per_utt_cap=128；仅使用有效训练 token 拟合聚类并导出 train 标签。

conda activate whisper_ptapt_faiss 
python /home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune/ft_ptapt_repeat/CPAT_stage2.py
"""

import os

# 工程、中间结果与特征根目录；保留原路径。
PROJECT_ROOT = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean/finetune"
MMSE_PTAPT_ROOT = os.path.join(PROJECT_ROOT, "ft_ptapt_repeat")

PROJECT_ROOT_MAIN = "/home/juanyu/Alzheimer_code_jax_mmse_speech_v4_mmse_mean"
FEATURE_ROOT_DIR = os.path.join(PROJECT_ROOT_MAIN, "feature_tensor")

# Pitt Stage1 规范划分审计；Stage2/3 按原规则检查。
PITT_SPLIT_AUDIT_ROOT = os.path.join(FEATURE_ROOT_DIR, "Pitt", "subject_disjoint_split")
PITT_VISIT_MANIFEST_PATH = os.path.join(PITT_SPLIT_AUDIT_ROOT, "pitt_valid_visits_subject_disjoint_manifest.csv")
PITT_SUBJECT_ASSIGNMENT_PATH = os.path.join(PITT_SPLIT_AUDIT_ROOT, "pitt_subject_assignment.csv")

# Pitt：原 test 中有有效 visit 的受试者全部归入 final test，其余归入 final train。
PITT_ENABLE_SUBJECT_DISJOINT_SPLIT = True
PITT_SPLIT_POLICY = "original_test_subjects_holdout"
PITT_EXPECTED_FINAL_TRAIN_SUBJECTS = 184
PITT_EXPECTED_FINAL_TEST_SUBJECTS = 68
PITT_EXPECTED_FINAL_TRAIN_VISITS = 276
PITT_EXPECTED_FINAL_TEST_VISITS = 122

# 批量数据集；默认仅  ["ADReSS-IS2020"]，其他数据集分支仍保留。
DATASET_OPTIONS = ["ADReSS-IS2020"]

DATASET_OPTION = DATASET_OPTIONS[0]

# 各阶段层号保持联动；默认第 12 层。
LAYERS_TO_RUN = [12]

# 相对当前表示层，向前连续解冻的层数；保留遍历顺序。
UNFREEZE_ENCODER_LAYERS_LIST = [8]

UNFREEZE_ENCODER_LAYERS = int(UNFREEZE_ENCODER_LAYERS_LIST[0])

# GPU 设置；保留各文件原值及环境变量生效顺序。
GPU_ID = 3
if GPU_ID is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)

# 协议标签；当前训练不划分 fold，test 仅用于最终导出。
RUN_PROTOCOL_TAG = "official_train_test"

# 旧配置/manifest 的兼容占位；不新增 fold 目录。
PROTOCOL_FOLD_ID = 1
FOLDS_TO_RUN = [PROTOCOL_FOLD_ID]

# 跨阶段模型根标签；各脚本必须一致。
MODEL_ROOT_TAG = "ptapt_lr5e-5_epo5"

# 读取 Stage1 checkpoint 的路径后缀；与实际 Stage1 输出对应。
STAGE1_SOURCE_PATH_PARAM_TAGS =  []

# singleX_unfreeY 的附加路径标签；Stage2、Stage3、导出需一致。
PATH_PARAM_TAGS = []

STAGE1_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN

STAGE2_REP_LAYERS_TO_RUN = LAYERS_TO_RUN

STAGE3_REPR_LAYERS_TO_RUN = LAYERS_TO_RUN

# 相对首个可训练层的 mask 偏移；不是绝对 encoder 层号。
STAGE3_MASK_INJECT_AFTER_LAYERS = [1]  # 1=第一个可训练层的前一层；2=前两层；具体层号由 Stage3 自动解析
STAGE4_OUTPUT_LAYERS_TO_RUN = LAYERS_TO_RUN

CHUNK_LEVEL = True

# 固定切段长度与重叠率；chunk key 格式保持原样。
SEGMENT_SEC = 30
OVERLAP_RATE = 0.3
CHUNK_KEY_VERSION = "wav_path_chunk_start_end_v1"

# Whisper 基模；保持原 checkpoint 与加载配置。
WHISPER_MODEL_NAME = "/home/juanyu/whisper_lru_3/whisper_small"

# 手动指定 Stage1 checkpoint；None 使用原自动查找规则。
STAGE1_CKPT_OVERRIDE = None

# Stage2 聚类头数值；需与 Stage3 EXPECTED_NUM_CLUSTERS 一致。
KS = [32, 64]

# token 采样：ratio / per_chunk / per_utt_cap / all；保留原采样顺序。
SAMPLE_MODE = "per_utt_cap" 
SAMPLE_RATIO = None
SAMPLE_PER_UTT = 128

TARGET_SEC = float(SEGMENT_SEC)
TRAIN_BATCH_SIZE = 4
INFER_BATCH_SIZE = 4
NUM_WORKERS = 4

# FAISS 执行与 KMeans 超参数。
FAISS_GPU = False
FAISS_NREDO = 1
FAISS_MAX_POINTS_PER_CENTROID = 2000
FAISS_VERBOSE = False
KMEANS_MAX_ITER = 200

SEED = 42
PAD_TO_FULL_LENGTH = True
SAVE_EXPORT_METADATA = True

# 当前 Stage2 仅拟合并导出 train 伪标签。
STAGE2_EXPORT_FOLD_VAL_LABELS = False

# 当前 no-val-last 流程要求 False；保留原错误检查。
ENABLE_STAGE_QUALITY_EVAL = False
QUALITY_REPORT_FILENAME = "quality_report.json"
QUALITY_SUMMARY_FILENAME = "quality_summary.csv"
QUALITY_DECISION_FILENAME = "quality_decision.txt"

import copy
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
import pandas as pd
import librosa

import torch
import torch.nn as nn
from torch.utils import data
from torch.utils.data.dataloader import default_collate

from transformers import WhisperModel, WhisperFeatureExtractor

import faiss
from tqdm import tqdm

SAMPLE_RATE = 16000
HOP_LENGTH = 160
MAX_MEL_FRAMES = 3000
TOK_T_FULL = (MAX_MEL_FRAMES + 1) // 2
IGNORE_INDEX = -100
SEGMENT_LENGTH = int(float(SEGMENT_SEC) * SAMPLE_RATE)

# 随机源与通用工具
def seed_everything_for_repro(seed: int) -> None:
    """统一设置随机种子，尽量保证实验可复现。"""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str) -> None:
    """若目录不存在则自动创建。"""
    os.makedirs(path, exist_ok=True)

def natural_key(x: str):
    """自然排序 key。"""
    s = Path(str(x)).name
    return [int(t) if t.isdigit() else t.lower() for t in re.findall(r"\d+|\D+", s)]

def canonical_wav_path(path: str) -> str:
    """统一 wav 路径表示，避免相对/绝对路径不一致导致匹配失败。"""
    return str(Path(str(path)).expanduser().resolve())

def save_json(obj: Dict[str, Any], path: str) -> None:
    """保存 json，并自动创建父目录。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_text(text: str, path: str) -> None:
    """保存纯文本。"""
    ensure_dir(str(Path(path).parent))
    with open(path, "w", encoding="utf-8") as f:
        f.write(str(text))

def compute_valid_token_length(wav_len_samples: int) -> int:
    """将“真实有效 waveform 样本点数”映射为 Whisper encoder 的有效 token 长度。"""
    if wav_len_samples <= 0:
        return 1
    n_mel_valid = int(math.ceil(wav_len_samples / float(HOP_LENGTH)))
    n_mel_valid = min(MAX_MEL_FRAMES, max(1, n_mel_valid))
    n_tok_valid = (n_mel_valid + 1) // 2
    return max(1, n_tok_valid)

def _clean_path_tag(x) -> str:
    """清理路径参数标签，避免空格、斜杠、等号等字符进入目录名。"""
    import re

    if x is None:
        return ""

    s = str(x).strip()
    if s in ["", "None", "none", "null", "Null"]:
        return ""

    s = s.replace("=", "").replace(" ", "")
    s = re.sub(r"[^0-9A-Za-z_.-]+", "", s)
    s = s.strip("_-.")
    return s

def build_path_param_suffix(path_param_tags=None) -> str:
    """根据 PATH_PARAM_TAGS 构造路径后缀。"""
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
    """构造 P-TAPT 统一使用的层目录名。"""
    base = f"single{int(single_layer)}_unfree{int(unfreeze_layer_count)}"
    suffix = build_path_param_suffix(path_param_tags)
    if suffix == "":
        return base
    return f"{base}_{suffix}"

def make_chunk_key(wav_path: str, chunk_id: int, start: int, end: int) -> str:
    """构造 chunk-level 唯一 key。"""
    wav_path = canonical_wav_path(wav_path)
    return f"{wav_path}::chunk{int(chunk_id):04d}::start{int(start)}::end{int(end)}"

def canonical_label_key(key: str) -> str:
    """规范化 Stage2/Stage3 之间用于匹配伪标签的 key。"""
    key = str(key)
    if "::chunk" in key:
        wav_part, suffix = key.split("::chunk", 1)
        return canonical_wav_path(wav_part) + "::chunk" + suffix
    return canonical_wav_path(key)

# chunk 切分、尾段补齐与 key
def split_segment_boundaries(num_samples: int, segment_length: int, overlap_rate: float) -> List[Tuple[int, int, int]]:
    """复用 audio_cut_xr.py 的 30s + overlap 切割思想，生成 chunk 边界。"""
    num_samples = int(num_samples)
    segment_length = int(segment_length)
    overlap_rate = float(overlap_rate)

    if segment_length <= 0:
        raise ValueError(f"segment_length 必须 > 0，当前={segment_length}")
    step_size = int(segment_length * (1.0 - overlap_rate))
    if step_size <= 0:
        raise ValueError(f"overlap_rate={overlap_rate} 导致 step_size <= 0。")

    if num_samples <= 0:
        return [(0, 0, segment_length)]

    total_segments = int(num_samples / step_size) + (
        1 if num_samples % step_size > overlap_rate * segment_length else 0
    )
    total_segments = max(1, int(total_segments))

    boundaries = []
    for chunk_id in range(total_segments):
        start = int(chunk_id * step_size)
        end = int(start + segment_length)
        boundaries.append((int(chunk_id), start, end))
    return boundaries

def load_audio_16k(wav_path: str) -> np.ndarray:
    """按 baseline 方式读取音频：librosa.load(sr=16000, mono=True, dtype=float32)。"""
    wav_path = canonical_wav_path(wav_path)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(f"[Stage2-MMSE] 找不到音频文件: {wav_path}")
    wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True, dtype=np.float32)
    return np.asarray(wav, dtype=np.float32).reshape(-1)

def load_chunk_with_padding_from_wav(y: np.ndarray, start: int, end: int, chunk_id: int) -> Tuple[np.ndarray, int]:
    """从已经重采样到 16k 的整条音频中取出一个 chunk，并按 audio_cut_xr.py 的尾段补齐逻辑处理。"""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    start = int(start)
    end = int(end)
    chunk_id = int(chunk_id)

    segment = y[start:end]

    if len(segment) < SEGMENT_LENGTH and chunk_id > 0:
        padding_needed = SEGMENT_LENGTH - len(segment)
        step_size = int(SEGMENT_LENGTH * (1.0 - float(OVERLAP_RATE)))
        prev_end = start + SEGMENT_LENGTH - step_size
        padding_start = max(0, prev_end - padding_needed)
        padding_values = y[padding_start:prev_end]
        segment = np.concatenate((segment, padding_values), axis=0)

    if len(segment) > SEGMENT_LENGTH:
        segment = segment[:SEGMENT_LENGTH]

    valid_len = int(min(len(segment), SEGMENT_LENGTH))
    if len(segment) < SEGMENT_LENGTH:
        pad = np.zeros((SEGMENT_LENGTH - len(segment),), dtype=np.float32)
        segment = np.concatenate([segment.astype(np.float32, copy=False), pad], axis=0)

    return segment.astype(np.float32, copy=False), valid_len

def load_chunk_with_padding(wav_path: str, start: int, end: int, chunk_id: int) -> Tuple[np.ndarray, int]:
    """读取 wav 并返回指定 chunk 的 30s waveform 与有效长度。"""
    y = load_audio_16k(wav_path)
    return load_chunk_with_padding_from_wav(y, start=start, end=end, chunk_id=chunk_id)

def build_chunk_entries_from_fold_entries(entries: List[Dict[str, Any]], split_name: str) -> List[Dict[str, Any]]:
    """将 foldY.json 中的 wav-level entries 展开为 chunk-level entries。"""
    chunk_entries: List[Dict[str, Any]] = []

    for wav_row in tqdm(entries, desc=f"构建 {split_name} chunks", dynamic_ncols=True):
        wav_row = copy.deepcopy(wav_row)
        wav_path = canonical_wav_path(wav_row["wav_path"])
        wav_name = str(wav_row.get("wav_name", Path(wav_path).name))

        y = load_audio_16k(wav_path)
        n_samples = int(len(y))
        boundaries = split_segment_boundaries(
            num_samples=n_samples,
            segment_length=SEGMENT_LENGTH,
            overlap_rate=OVERLAP_RATE,
        )

        for chunk_id, start, end in boundaries:
            _seg, valid_len = load_chunk_with_padding_from_wav(
                y, start=start, end=end, chunk_id=chunk_id
            )
            chunk_key = make_chunk_key(
                wav_path=wav_path,
                chunk_id=chunk_id,
                start=start,
                end=end,
            )

            row = copy.deepcopy(wav_row)
            row.update({
                "split": str(split_name).lower(),
                "wav_path": wav_path,
                "wav_name": wav_name,
                "chunk_key": chunk_key,
                "chunk_id": int(chunk_id),
                "chunk_start": int(start),
                "chunk_end": int(end),
                "chunk_valid_len": int(valid_len),
                "parent_num_samples_16k": int(n_samples),
                "segment_sec": float(SEGMENT_SEC),
                "overlap_rate": float(OVERLAP_RATE),
                "chunk_key_version": str(CHUNK_KEY_VERSION),
            })
            chunk_entries.append(row)

    if len(chunk_entries) == 0:
        raise ValueError(f"{split_name} chunk_entries 为空，请检查音频路径和 fold json。")

    return chunk_entries

def _strip_state_dict_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    """去掉 state_dict key 的公共前缀。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if k.startswith(prefix):
            out[k[len(prefix):]] = v
        else:
            out[k] = v
    return out

def load_stage1_ckpt_into_whisper_model(whisper_model: WhisperModel, ckpt_path: str) -> None:
    """把 Stage1 checkpoint 中的 Whisper 权重灌入当前 WhisperModel。"""
    if ckpt_path in [None, "None", ""]:
        return

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    candidates = [
        "whisper.model.",  # Stage1-MMSE 最常见：LightningModule.whisper.model.*
        "model.",          # 另一种常见命名
        "",                # 兜底：不去前缀
    ]

    loaded_any = False
    for pref in candidates:
        sd = _strip_state_dict_prefix(state, pref) if pref != "" else state
        missing, unexpected = whisper_model.load_state_dict(sd, strict=False)

        encoder_hit = any(k.startswith("encoder.") for k in sd.keys())
        if encoder_hit:
            print(f"[stage1_init] tried prefix='{pref}' -> missing={len(missing)} unexpected={len(unexpected)}")
            loaded_any = True
            break

    if not loaded_any:
        missing, unexpected = whisper_model.load_state_dict(state, strict=False)
        print(f"[stage1_init][warn] 前缀无法明确匹配，已 strict=False 加载，missing={len(missing)} unexpected={len(unexpected)}")

def find_stage1_checkpoint(stage1_save_dir: str, override_ckpt: Optional[str] = None) -> str:
    """no-val-last:"""
    if override_ckpt not in [None, "", "None"]:
        ckpt_path = str(override_ckpt)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"STAGE1_CKPT_OVERRIDE 不存在：{ckpt_path}")
        return ckpt_path

    stage1_dir = Path(stage1_save_dir)
    if not stage1_dir.exists():
        raise FileNotFoundError(f"Stage1 目录不存在：{stage1_save_dir}")

    last_ckpt = stage1_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt.resolve())

    all_ckpts = sorted(stage1_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(all_ckpts) > 0:
        return str(all_ckpts[0].resolve())

    raise FileNotFoundError(f"在 {stage1_save_dir} 下未找到任何 Stage1 checkpoint。")

def _extract_pitt_subject_visit(raw_name: str) -> Tuple[Optional[str], Optional[int]]:
    """与 Whisper_pitt_new(1).py 相同的 Pitt 文件名解析。"""
    if raw_name is None:
        return None, None
    stem = Path(str(raw_name)).stem
    m = re.match(r"^(\d+)-(\d+)(?:c)?(?:-|$)", stem, flags=re.IGNORECASE)
    if m is None:
        return None, None
    return str(int(m.group(1))), int(m.group(2))

def _normalize_pitt_subject_id(x) -> str:
    s = str(x).strip()
    try:
        return str(int(float(s)))
    except Exception:
        m = re.search(r"\d+", s)
        if m is None:
            raise ValueError(f"无法规范化 Pitt subject_id={x}")
        return str(int(m.group(0)))

def _load_pitt_canonical_split_audit() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """读取 Stage1 写出的 canonical Pitt split 审计文件。"""
    if not os.path.isfile(PITT_VISIT_MANIFEST_PATH) or not os.path.isfile(PITT_SUBJECT_ASSIGNMENT_PATH):
        raise FileNotFoundError(
            "[Pitt Stage2] 缺少 subject-disjoint 审计文件。请先运行 SCPA_stage1_pitt_new.py。\n"
            f"visit_manifest={PITT_VISIT_MANIFEST_PATH}\n"
            f"subject_assignment={PITT_SUBJECT_ASSIGNMENT_PATH}"
        )

    visit_df = pd.read_csv(PITT_VISIT_MANIFEST_PATH)
    subject_df = pd.read_csv(PITT_SUBJECT_ASSIGNMENT_PATH)

    required_visit = {"subject_id", "visit_index", "wav_path", "final_split"}
    required_subject = {"subject_id", "final_split"}
    if not required_visit.issubset(set(visit_df.columns)):
        raise ValueError(f"[Pitt Stage2] visit manifest 缺列：{sorted(required_visit - set(visit_df.columns))}")
    if not required_subject.issubset(set(subject_df.columns)):
        raise ValueError(f"[Pitt Stage2] subject assignment 缺列：{sorted(required_subject - set(subject_df.columns))}")

    visit_df = visit_df.copy()
    subject_df = subject_df.copy()
    visit_df["subject_id"] = visit_df["subject_id"].map(_normalize_pitt_subject_id)
    subject_df["subject_id"] = subject_df["subject_id"].map(_normalize_pitt_subject_id)
    visit_df["visit_index"] = pd.to_numeric(visit_df["visit_index"], errors="raise").astype(int)
    visit_df["final_split"] = visit_df["final_split"].astype(str).str.strip().str.lower()
    subject_df["final_split"] = subject_df["final_split"].astype(str).str.strip().str.lower()

    train_subjects = set(subject_df.loc[subject_df["final_split"] == "train", "subject_id"].tolist())
    test_subjects = set(subject_df.loc[subject_df["final_split"] == "test", "subject_id"].tolist())
    overlap = train_subjects & test_subjects
    if overlap:
        raise RuntimeError(f"[Pitt Stage2] canonical audit 本身存在 subject overlap: {sorted(overlap)}")

    train_visits = visit_df[visit_df["final_split"] == "train"]
    test_visits = visit_df[visit_df["final_split"] == "test"]
    checks = [
        ("train_subjects", len(train_subjects), PITT_EXPECTED_FINAL_TRAIN_SUBJECTS),
        ("test_subjects", len(test_subjects), PITT_EXPECTED_FINAL_TEST_SUBJECTS),
        ("train_visits", len(train_visits), PITT_EXPECTED_FINAL_TRAIN_VISITS),
        ("test_visits", len(test_visits), PITT_EXPECTED_FINAL_TEST_VISITS),
    ]
    for name, actual, expected in checks:
        if int(actual) != int(expected):
            raise RuntimeError(f"[Pitt Stage2] canonical audit {name}={actual}, expected={expected}")

    return visit_df, subject_df

def validate_pitt_stage1_train_manifest(entries: List[Dict[str, Any]]) -> None:
    """强制检查 Stage2 入口恰好是 canonical final train：184 subjects / 276 visits。"""
    if str(DATASET_OPTION) != "Pitt" or not bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        return

    visit_df, subject_df = _load_pitt_canonical_split_audit()
    canonical_train = visit_df[visit_df["final_split"] == "train"].copy()
    canonical_train_visits = {
        (_normalize_pitt_subject_id(r.subject_id), int(r.visit_index))
        for r in canonical_train.itertuples()
    }
    canonical_train_paths = {canonical_wav_path(p) for p in canonical_train["wav_path"].astype(str).tolist()}
    canonical_train_subjects = {
        _normalize_pitt_subject_id(x)
        for x in subject_df.loc[subject_df["final_split"] == "train", "subject_id"].tolist()
    }
    canonical_test_subjects = {
        _normalize_pitt_subject_id(x)
        for x in subject_df.loc[subject_df["final_split"] == "test", "subject_id"].tolist()
    }

    manifest_visits = set()
    manifest_paths = set()
    manifest_subjects = set()
    duplicate_visits = []
    for row in entries:
        fp = canonical_wav_path(row.get("wav_path", ""))
        sid = row.get("subject_id", None)
        vid = row.get("visit_index", None)
        if sid is None or vid is None:
            sid, vid = _extract_pitt_subject_visit(Path(fp).stem)
        if sid is None or vid is None:
            raise RuntimeError(f"[Pitt Stage2] Stage1 manifest 无法解析 subject/visit: {fp}")
        sid = _normalize_pitt_subject_id(sid)
        key = (sid, int(vid))
        if key in manifest_visits:
            duplicate_visits.append(key)
        manifest_visits.add(key)
        manifest_paths.add(fp)
        manifest_subjects.add(sid)

    if duplicate_visits:
        raise RuntimeError(f"[Pitt Stage2] Stage1 manifest 存在重复 subject-visit: {duplicate_visits[:10]}")
    if manifest_subjects & canonical_test_subjects:
        raise RuntimeError(
            f"[Pitt Stage2] Stage1 manifest 含 final-test subjects: {sorted(manifest_subjects & canonical_test_subjects, key=int)[:20]}"
        )
    if manifest_subjects != canonical_train_subjects:
        missing = sorted(canonical_train_subjects - manifest_subjects, key=int)
        extra = sorted(manifest_subjects - canonical_train_subjects, key=int)
        raise RuntimeError(f"[Pitt Stage2] Stage1 train subjects 不一致：missing={missing[:20]}, extra={extra[:20]}")
    if manifest_visits != canonical_train_visits:
        missing = sorted(canonical_train_visits - manifest_visits, key=lambda x: (int(x[0]), x[1]))
        extra = sorted(manifest_visits - canonical_train_visits, key=lambda x: (int(x[0]), x[1]))
        raise RuntimeError(f"[Pitt Stage2] Stage1 train visits 不一致：missing={missing[:20]}, extra={extra[:20]}")
    if manifest_paths != canonical_train_paths:
        missing = sorted(canonical_train_paths - manifest_paths)
        extra = sorted(manifest_paths - canonical_train_paths)
        raise RuntimeError(f"[Pitt Stage2] Stage1 train wav paths 不一致：missing={missing[:10]}, extra={extra[:10]}")

    print(
        "[Pitt Stage2] subject-disjoint Stage1 manifest check PASSED | "
        f"subjects={len(manifest_subjects)}, visits={len(manifest_visits)}, final-test-overlap=0"
    )

def _is_process2_ctd_manifest_entry(row: Dict[str, Any], dataset_key: str = None) -> bool:
    """PROCESS-2 专用 manifest safety filter。"""
    if dataset_key is None:
        dataset_key = DATASET_OPTION

    if str(dataset_key).strip().lower() != "process-2":
        return True

    wav_path = row.get("wav_path", row.get("wav_name", ""))
    stem = Path(str(wav_path)).stem
    return re.search(r"__CTD$", stem, flags=re.IGNORECASE) is not None

# Stage1 有效训练清单衔接
def load_stage1_valid_train_manifest(stage1_save_dir: str) -> List[Dict[str, Any]]:
    """读取 Stage1 已经过 MMSE 过滤后的 train_manifest.json。"""
    manifest_path = os.path.join(str(stage1_save_dir), "train_manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f"Stage1 train_manifest.json 不存在：{manifest_path}\n"
            "请先运行 whisper_ptapt_stage1.py。"
        )

    with open(manifest_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and isinstance(obj.get("entries", None), list):
        entries = obj["entries"]
    elif isinstance(obj, list):
        entries = obj
    else:
        raise ValueError(f"无法解析 Stage1 train_manifest.json：{manifest_path}")

    out: List[Dict[str, Any]] = []
    for row in entries:
        if not isinstance(row, dict):
            continue

        row = copy.deepcopy(row)
        if row.get("wav_path", None) in [None, ""]:
            continue

        row["wav_path"] = canonical_wav_path(row["wav_path"])
        row["wav_name"] = str(row.get("wav_name", Path(row["wav_path"]).name))
        row["split"] = "train"

        if not _is_process2_ctd_manifest_entry(row, dataset_key=DATASET_OPTION):
            continue

        out.append(row)

    if len(out) == 0:
        raise ValueError(f"Stage1 train_manifest 解析后为空：{manifest_path}")

    if str(DATASET_OPTION).strip().lower() == "process-2":
        print(
            f"[PROCESS-2][CTD] Stage2 loaded Stage1 train_manifest: "
            f"raw_entries={len(entries)}, kept_ctd_entries={len(out)}, "
            f"dropped_non_ctd={len(entries) - len(out)}"
        )

    out = sorted(out, key=lambda x: natural_key(x.get("wav_path", "")))
    if str(DATASET_OPTION) == "Pitt" and bool(PITT_ENABLE_SUBJECT_DISJOINT_SPLIT):
        validate_pitt_stage1_train_manifest(out)
    return out

def filter_entries_with_target_mmse_for_stage2(entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Stage2 保留 MMSE 过滤。"""
    valid: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []

    for row in entries:
        row = copy.deepcopy(row)
        val = row.get("target_mmse", None)
        try:
            if val in [None, ""]:
                raise ValueError("empty target_mmse")
            row["target_mmse"] = float(val)
            valid.append(row)
        except Exception:
            row["missing_reason"] = "missing_or_invalid_target_mmse_in_stage1_manifest"
            missing.append(row)

    return valid, missing

def check_all_clus_heads(all_clus_path: str, expected_ks: List[int]) -> None:
    """检查 all-clus.json 的 head 数是否和 KS 一致。"""
    if not os.path.isfile(all_clus_path):
        raise FileNotFoundError(f"[Stage2] all-clus.json not found: {all_clus_path}")

    with open(all_clus_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    keys = [k for k in raw.keys() if not str(k).startswith("__")]
    if len(keys) == 0:
        raise RuntimeError(f"[Stage2] empty all-clus.json: {all_clus_path}")

    expected_ks = [int(k) for k in expected_ks]
    n_heads = len(raw[keys[0]])

    if n_heads != len(expected_ks):
        raise RuntimeError(
            f"[Stage2] head mismatch: all-clus has {n_heads} heads, "
            f"but KS={expected_ks}. path={all_clus_path}"
        )

    for name in keys[:10]:
        if len(raw[name]) != len(expected_ks):
            raise RuntimeError(
                f"[Stage2] sample={name} has {len(raw[name])} heads, "
                f"but KS={expected_ks}. path={all_clus_path}"
            )

def write_stage2_run_meta(cfg: Dict[str, Any], stage1_ckpt_path: str) -> None:
    """保存 Stage2 运行配置。"""
    stage2_save_dir = cfg["STAGE2_SAVE_DIR"]

    train_all_clus_path = os.path.join(cfg["STAGE2_TRAIN_DIR"], "all-clus.json")

    meta = {
        "DATASET_OPTION": cfg.get("DATASET_OPTION", DATASET_OPTION),
        "MODEL_ROOT_TAG": MODEL_ROOT_TAG,
        "REP_LAYER": int(cfg.get("REP_LAYER", -1)),
        "FOLD_ID": int(cfg.get("FOLD_ID", -1)),
        "LAYER_NAME": cfg.get("LAYER_NAME", ""),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),

        "PATH_PARAM_TAGS": list(globals().get("PATH_PARAM_TAGS", [])),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(globals().get("PATH_PARAM_TAGS", [])),
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(globals().get("STAGE1_SOURCE_PATH_PARAM_TAGS", [])),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(
            globals().get("STAGE1_SOURCE_PATH_PARAM_TAGS", [])
        ),

        "KS": [int(k) for k in KS],
        "SAMPLE_MODE": str(SAMPLE_MODE),
        "SAMPLE_RATIO": float(SAMPLE_RATIO) if SAMPLE_RATIO is not None else None,
        "SAMPLE_PER_UTT": int(SAMPLE_PER_UTT),
        "CHUNK_LEVEL": bool(CHUNK_LEVEL),

        "SEGMENT_SEC": float(SEGMENT_SEC),
        "OVERLAP_RATE": float(OVERLAP_RATE),
        "CHUNK_KEY_VERSION": str(CHUNK_KEY_VERSION),

        "STAGE2_EXPORT_FOLD_VAL_LABELS": False,
        "upstream_uses_fold_val": False,
        "stage1_ckpt_path": str(stage1_ckpt_path),
        "train_all_clus_path": str(train_all_clus_path),
        "val_all_clus_path": None,
    }

    save_json(meta, os.path.join(stage2_save_dir, "stage2_run_meta.json"))

# Stage2 配置与表征抽取
def build_stage2_run_config(dataset_option: str, rep_layer: int, fold_id: int = PROTOCOL_FOLD_ID) -> Dict[str, Any]:
    """构造 Stage2 运行配置。"""
    rep_layer = int(rep_layer)
    fold_id = int(PROTOCOL_FOLD_ID)

    layer_name = build_layer_name(
        single_layer=rep_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=PATH_PARAM_TAGS,
    )

    stage1_layer_name = build_layer_name(
        single_layer=rep_layer,
        unfreeze_layer_count=UNFREEZE_ENCODER_LAYERS,
        path_param_tags=STAGE1_SOURCE_PATH_PARAM_TAGS,
    )

    run_root = os.path.join(
        MMSE_PTAPT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        layer_name,
    )

    stage1_run_root = os.path.join(
        MMSE_PTAPT_ROOT,
        dataset_option,
        MODEL_ROOT_TAG,
        stage1_layer_name,
    )

    stage1_dir = os.path.join(stage1_run_root, "stage1_mmse")

    stage2_save_dir = os.path.join(run_root, "stage2_clusters")
    stage2_train_dir = os.path.join(stage2_save_dir, "train")
    stage2_val_dir = os.path.join(stage2_save_dir, "val")
    meta_save_dir = os.path.join(run_root, "meta")

    return {
        "DATASET_OPTION": str(dataset_option),
        "REP_LAYER": int(rep_layer),
        "UNFREEZE_ENCODER_LAYERS": int(UNFREEZE_ENCODER_LAYERS),

        "FOLD_ID": int(fold_id),
        "FOLD_TAG": "not_used",
        "FOLD_JSON_PATH": "",
        "RUN_PROTOCOL_TAG": str(RUN_PROTOCOL_TAG),

        "LAYER_TAG": layer_name,
        "LAYER_NAME": layer_name,
        "PATH_PARAM_TAGS": list(PATH_PARAM_TAGS),
        "PATH_PARAM_SUFFIX": build_path_param_suffix(PATH_PARAM_TAGS),

        "STAGE1_LAYER_NAME": stage1_layer_name,
        "STAGE1_SOURCE_PATH_PARAM_TAGS": list(STAGE1_SOURCE_PATH_PARAM_TAGS),
        "STAGE1_SOURCE_PATH_PARAM_SUFFIX": build_path_param_suffix(STAGE1_SOURCE_PATH_PARAM_TAGS),

        "RUN_ROOT": run_root,
        "STAGE1_RUN_ROOT": stage1_run_root,
        "STAGE1_SAVE_DIR": stage1_dir,
        "STAGE1_DIR": stage1_dir,

        "STAGE2_SAVE_DIR": stage2_save_dir,
        "STAGE2_TRAIN_DIR": stage2_train_dir,
        "STAGE2_VAL_DIR": stage2_val_dir,
        "META_SAVE_DIR": meta_save_dir,
    }
class ChunkAudioDataset(data.Dataset):
    """Stage2 chunk-level 数据集。"""
    def __init__(self, entries: List[Dict[str, Any]]):
        self.entries = [copy.deepcopy(x) for x in entries]
        if len(self.entries) == 0:
            raise ValueError("ChunkAudioDataset 收到空 entries，无法构建数据集。")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int):
        row = self.entries[i]
        wav_path = canonical_wav_path(row["wav_path"])
        wav_name = str(row.get("wav_name", Path(wav_path).name))
        chunk_key = canonical_label_key(row["chunk_key"])
        chunk_id = int(row["chunk_id"])
        start = int(row["chunk_start"])
        end = int(row["chunk_end"])

        wav_30, valid_len = load_chunk_with_padding(
            wav_path=wav_path,
            start=start,
            end=end,
            chunk_id=chunk_id,
        )

        return wav_30, int(valid_len), chunk_key, wav_name

    def repeat30s_collate(self, batch):
        """兼容旧 DataLoader 调用名。"""
        return default_collate(batch)

class WhisperEncoderWrapper(nn.Module):
    """输出 Whisper encoder 的第 rep_layer 层 hidden states，形状 [B, T(=1500), C]。"""
    def __init__(self, model_name: str = "openai/whisper-small", rep_layer: int = 12):
        super().__init__()
        self.model = WhisperModel.from_pretrained(model_name)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
        self.hidden_size = self.model.config.d_model
        self.rep_layer = int(rep_layer)

        for p in self.model.parameters():
            p.requires_grad = False

    def _build_attn_mask(self, input_features: torch.Tensor, valid_wav_lens: torch.Tensor) -> Optional[torch.Tensor]:
        """构造 token 级 attention_mask：[B, tok_T]，True=可见，False=不可见。"""
        if valid_wav_lens is None:
            return None
        if not torch.is_tensor(valid_wav_lens):
            valid_wav_lens = torch.tensor(valid_wav_lens)

        vwl = valid_wav_lens.detach().cpu().long().view(-1)
        B = input_features.size(0)
        if vwl.numel() != B:
            raise RuntimeError(f"valid_wav_lens 数量与 batch 不一致：expect={B}, got={vwl.numel()}")

        mel_T = int(input_features.size(-1))
        tok_T = (mel_T + 1) // 2

        valid_Ts = []
        for i in range(B):
            vt = compute_valid_token_length(int(vwl[i].item()))
            valid_Ts.append(min(vt, tok_T))
        valid_Ts = torch.tensor(valid_Ts, device=input_features.device, dtype=torch.long)

        t = torch.arange(tok_T, device=input_features.device).view(1, tok_T).expand(B, tok_T)
        return (t < valid_Ts.view(B, 1)).bool()

    def forward_layerk(
        self,
        wav: torch.Tensor,
        sampling_rate: int = 16000,
        valid_wav_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """提取当前 rep_layer 的 Whisper hidden states。"""
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

        enc_out = self.model.encoder(
            input_features=input_features,
            attention_mask=attn_mask,
            return_dict=True,
            output_hidden_states=True,
        )
        hs = enc_out.hidden_states

        if self.rep_layer < 1 or self.rep_layer >= len(hs):
            raise ValueError(f"rep_layer={self.rep_layer} 非法；hidden_states len={len(hs)}")

        return hs[self.rep_layer]

# 训练 token 采样、FAISS 聚类与伪标签导出
@torch.no_grad()
def collect_train_samples(
    model: WhisperEncoderWrapper,
    loader,
    device: str,
    sample_ratio: Optional[float] = None,
    sample_per_utt: Optional[int] = None,
    sample_mode: str = "ratio",
) -> np.ndarray:
    """从 fold-train 抽取 token 表征，形成用于 KMeans 的训练样本矩阵 X_train: [N, C]。"""
    sample_mode = str(sample_mode).strip().lower()
    valid_modes = {"ratio", "per_chunk", "per_utt_cap", "all"}
    if sample_mode not in valid_modes:
        raise ValueError(f"SAMPLE_MODE 必须属于 {valid_modes}，当前={sample_mode}")

    if sample_ratio is not None:
        sample_ratio = float(sample_ratio)
        if not (0.0 < sample_ratio <= 1.0):
            raise ValueError(f"SAMPLE_RATIO 必须在 (0,1]，当前={sample_ratio}")

    if sample_per_utt is None:
        sample_per_utt = 0
    sample_per_utt = int(sample_per_utt)

    if sample_mode == "ratio" and sample_ratio is None:
        raise ValueError("SAMPLE_MODE='ratio' 时 SAMPLE_RATIO 不能为 None。")

    if sample_mode in {"per_chunk", "per_utt_cap"} and sample_per_utt <= 0:
        raise ValueError(
            f"SAMPLE_MODE='{sample_mode}' 时 SAMPLE_PER_UTT 必须 > 0，当前={sample_per_utt}"
        )

    def _parent_key_from_chunk_key(chunk_key: str, wav_name: str) -> str:
        """从 chunk_key 中取原始 wav key。"""
        ck = str(chunk_key)
        if "::chunk" in ck:
            return ck.split("::chunk", 1)[0]
        return str(wav_name)

    feats_all: List[np.ndarray] = []
    total_tokens = 0

    reservoir: Dict[str, List[np.ndarray]] = {}
    seen_count: Dict[str, int] = {}

    pbar = tqdm(loader, total=len(loader), desc="收集训练样本(fold-train)", dynamic_ncols=True)

    for wav_30, valid_len, chunk_keys, wav_names in pbar:
        wav_30 = wav_30.to(device)
        valid_len = valid_len.to(device)

        reps = model.forward_layerk(
            wav_30,
            sampling_rate=SAMPLE_RATE,
            valid_wav_lens=valid_len,
        )  # [B,T,C]

        B, T, C = reps.shape
        vwl_cpu = valid_len.detach().cpu().long().tolist()

        chunk_keys = list(chunk_keys)
        wav_names = list(wav_names)

        for i in range(B):
            vt = min(compute_valid_token_length(int(vwl_cpu[i])), T)
            if vt <= 0:
                continue

            x = reps[i, :vt, :].detach().cpu().numpy().astype(np.float32, copy=False)  # [vt,C]

            if sample_mode == "all":
                feats_all.append(x)
                total_tokens += int(vt)

            elif sample_mode == "ratio":
                n = int(math.ceil(vt * float(sample_ratio)))
                n = max(1, min(vt, n))

                if n >= vt:
                    feats_all.append(x)
                    total_tokens += int(vt)
                else:
                    idx = np.random.choice(vt, size=n, replace=False)
                    feats_all.append(x[idx])
                    total_tokens += int(n)

            elif sample_mode == "per_chunk":
                n = min(vt, int(sample_per_utt))

                if n >= vt:
                    feats_all.append(x)
                    total_tokens += int(vt)
                else:
                    idx = np.random.choice(vt, size=n, replace=False)
                    feats_all.append(x[idx])
                    total_tokens += int(n)

            elif sample_mode == "per_utt_cap":
                parent_key = _parent_key_from_chunk_key(
                    chunk_key=str(chunk_keys[i]),
                    wav_name=str(wav_names[i]),
                )

                if parent_key not in reservoir:
                    reservoir[parent_key] = []
                    seen_count[parent_key] = 0

                n_cand = min(vt, int(sample_per_utt))
                if n_cand < vt:
                    cand_idx = np.random.choice(vt, size=n_cand, replace=False)
                    cand = x[cand_idx]
                else:
                    cand = x

                for j in range(cand.shape[0]):
                    seen_count[parent_key] += 1
                    m = int(seen_count[parent_key])

                    if len(reservoir[parent_key]) < int(sample_per_utt):
                        reservoir[parent_key].append(cand[j].copy())
                    else:
                        r = np.random.randint(0, m)
                        if r < int(sample_per_utt):
                            reservoir[parent_key][r] = cand[j].copy()

        if sample_mode == "per_utt_cap":
            total_tokens = sum(len(v) for v in reservoir.values())
            pbar.set_postfix(wavs=len(reservoir), tokens=total_tokens)
        else:
            pbar.set_postfix(tokens=total_tokens)

    if sample_mode == "per_utt_cap":
        if len(reservoir) == 0:
            raise RuntimeError("per_utt_cap 没有收集到任何训练特征，无法进行聚类。")

        feats_per_wav = []
        for parent_key in sorted(reservoir.keys()):
            arr = np.stack(reservoir[parent_key], axis=0).astype(np.float32)
            feats_per_wav.append(arr)

        X = np.concatenate(feats_per_wav, axis=0).astype(np.float32)

        print(
            f"[Stage2-MMSE][KMeansSample] mode=per_utt_cap, "
            f"num_parent_wavs={len(reservoir)}, "
            f"cap_per_wav={sample_per_utt}, "
            f"total_tokens={X.shape[0]}"
        )

    else:
        if len(feats_all) == 0:
            raise RuntimeError("没有收集到任何训练特征，无法进行聚类。")

        X = np.concatenate(feats_all, axis=0).astype(np.float32)

        print(
            f"[Stage2-MMSE][KMeansSample] mode={sample_mode}, "
            f"total_tokens={X.shape[0]}"
        )

    return np.ascontiguousarray(X, dtype=np.float32)

def _build_faiss_index_from_centroids(centroids: np.ndarray, use_gpu: bool):
    """用聚类中心构建最近邻查询 index（IndexFlatL2）。"""
    centroids = np.ascontiguousarray(centroids.astype(np.float32))
    k, d = centroids.shape

    index = faiss.IndexFlatL2(d)

    if use_gpu:
        if not hasattr(faiss, "StandardGpuResources"):
            raise RuntimeError(
                "当前环境没有 faiss-gpu（缺少 StandardGpuResources），请安装 faiss-gpu 或把 FAISS_GPU=False。"
            )
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    index.add(centroids)
    return index

def fit_faiss_kmeans(
    X_train: np.ndarray,
    k: int,
    seed: int,
    niter: int,
    nredo: int,
    use_gpu: bool,
    max_points_per_centroid: int,
    verbose: bool,
):
    """训练 FAISS KMeans，并返回："""
    X_train = np.ascontiguousarray(X_train.astype(np.float32))
    d = int(X_train.shape[1])

    km = faiss.Kmeans(
        d=d,
        k=int(k),
        niter=int(niter),
        nredo=int(nredo),
        verbose=bool(verbose),
        gpu=bool(use_gpu),
    )

    try:
        km.seed = int(seed)
    except Exception:
        pass
    try:
        km.max_points_per_centroid = int(max_points_per_centroid)
    except Exception:
        pass

    km.train(X_train)
    centroids = km.centroids.reshape(int(k), d).astype(np.float32)
    centroids = np.ascontiguousarray(centroids, dtype=np.float32)

    index = _build_faiss_index_from_centroids(centroids, use_gpu=use_gpu)
    return centroids, index

@torch.no_grad()
def infer_and_export_split_json(
    model: WhisperEncoderWrapper,
    loader,
    index_list: List[Any],
    ks: List[int],
    device: str,
    outputdir: str,
    split_name: str,
    pad_to_full: bool,
    entries: List[Dict[str, Any]],
) -> None:
    """为单个 split（train 或 val）生成伪标签并导出 json。"""
    ensure_dir(outputdir)

    outputdata_list: List[Dict[str, List[int]]] = [dict() for _ in ks]
    combined_dict: Dict[str, List[List[int]]] = {}
    keys_meta: Dict[str, Dict[str, str]] = {}

    pbar = tqdm(loader, total=len(loader), desc=f"推断伪标签({split_name})", dynamic_ncols=True)
    for wav_30, valid_len, wav_keys, wav_names in pbar:
        wav_30 = wav_30.to(device)
        valid_len = valid_len.to(device)

        reps = model.forward_layerk(wav_30, sampling_rate=SAMPLE_RATE, valid_wav_lens=valid_len)  # [B,T,C]
        B, T, C = reps.shape
        vwl_cpu = valid_len.detach().cpu().long().tolist()

        for i in range(B):
            wav_key = str(wav_keys[i])
            wav_name = str(wav_names[i])
            vt = min(compute_valid_token_length(int(vwl_cpu[i])), T)
            vt = max(0, int(vt))

            if vt <= 0:
                if pad_to_full:
                    labels_full_list = [[IGNORE_INDEX] * T for _ in ks]
                else:
                    labels_full_list = [[] for _ in ks]
            else:
                x = reps[i, :vt, :].detach().cpu().numpy()
                x = np.ascontiguousarray(x.astype(np.float32))

                labels_full_list = []
                for index in index_list:
                    _, I = index.search(x, 1)
                    labels_valid = I[:, 0].astype(int).tolist()

                    if pad_to_full and vt < T:
                        labels_full = labels_valid + [IGNORE_INDEX] * (T - vt)
                    else:
                        labels_full = labels_valid

                    labels_full_list.append(labels_full)

            for j in range(len(ks)):
                outputdata_list[j][wav_key] = labels_full_list[j]
            combined_dict[wav_key] = labels_full_list
            keys_meta[wav_key] = {
                "wav_name": wav_name,
                "chunk_key": wav_key,
                "split": str(split_name).lower(),
            }

    for j, k in enumerate(ks):
        out_path = os.path.join(outputdir, f"{k}-clus.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(outputdata_list[j], f, ensure_ascii=False, indent=2)
        print(f"[Stage2-MMSE] Saved K={k} labels -> {out_path}")

    all_path = os.path.join(outputdir, "all-clus.json")
    with open(all_path, "w", encoding="utf-8") as f:
        json.dump(combined_dict, f, ensure_ascii=False, indent=2)
    print(f"[Stage2-MMSE] Saved ALL labels -> {all_path}")

    save_json(keys_meta, os.path.join(outputdir, "keys_meta.json"))
    save_json(
        {
            "split": str(split_name).lower(),
            "chunk_key_version": str(CHUNK_KEY_VERSION),
            "segment_sec": float(SEGMENT_SEC),
            "overlap_rate": float(OVERLAP_RATE),
            "num_chunks": int(len(entries)),
            "chunks": entries,
        },
        os.path.join(outputdir, "chunks_manifest.json"),
    )

# 单次实验
def run_one_experiment(run_config: Dict[str, Any]) -> None:
    """执行单个 (dataset, rep_layer, fold) 的 Stage2 实验。"""
    seed_everything_for_repro(SEED)

    ensure_dir(run_config["STAGE2_SAVE_DIR"])
    ensure_dir(run_config["STAGE2_TRAIN_DIR"])
    ensure_dir(run_config["STAGE2_VAL_DIR"])
    ensure_dir(run_config["META_SAVE_DIR"])

    print("\n" + "=" * 90)
    print(f"[Stage2-MMSE] DATASET = {run_config['DATASET_OPTION']}")
    print(f"[Stage2-MMSE] REP_LAYER = {run_config['REP_LAYER']}")
    print(f"[Stage2-MMSE] FOLD_ID = {run_config['FOLD_ID']}")
    print(f"[Stage2-MMSE] FOLD_JSON_PATH = {run_config['FOLD_JSON_PATH']}")
    print(f"[Stage2-MMSE] STAGE1_SAVE_DIR = {run_config['STAGE1_SAVE_DIR']}")
    print(f"[Stage2-MMSE] STAGE2_SAVE_DIR = {run_config['STAGE2_SAVE_DIR']}")
    print("=" * 90)

    train_wav_entries_raw = load_stage1_valid_train_manifest(run_config["STAGE1_SAVE_DIR"])

    train_wav_entries, missing_train_mmse = filter_entries_with_target_mmse_for_stage2(
        train_wav_entries_raw
    )

    val_wav_entries = []

    if len(train_wav_entries) == 0:
        raise ValueError(
            f"Stage2 official_train 在 target_mmse 过滤后为空："
            f"dataset={run_config['DATASET_OPTION']}"
        )

    train_entries = build_chunk_entries_from_fold_entries(train_wav_entries, split_name="train")
    val_entries = []

    save_json(
        {"entries": missing_train_mmse},
        os.path.join(run_config["META_SAVE_DIR"], "stage2_missing_train_mmse_entries.json"),
    )

    print(
        f"[Stage2-MMSE][OFFICIAL-TRAIN] "
        f"stage1_manifest_wavs={len(train_wav_entries_raw)}, "
        f"valid_mmse_wavs={len(train_wav_entries)}, "
        f"missing_mmse_wavs={len(missing_train_mmse)}, "
        f"train_chunks={len(train_entries)}"
    )

    print(
        f"[Stage2-MMSE][NO-VAL-LAST] chunk-level 展开完成："
        f"train_wavs={len(train_wav_entries)}, train_chunks={len(train_entries)}, "
    )

    print(
        f"[Stage2-MMSE] chunk-level 展开完成："
        f"train_wavs={len(train_wav_entries)}, train_chunks={len(train_entries)}, "
        f"val_wavs={len(val_wav_entries)}, val_chunks={len(val_entries)}"
    )

    stage1_ckpt_path = find_stage1_checkpoint(
        stage1_save_dir=run_config["STAGE1_SAVE_DIR"],
        override_ckpt=STAGE1_CKPT_OVERRIDE,
    )
    print(f"[Stage2-MMSE] 使用的 Stage1 checkpoint = {stage1_ckpt_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_faiss_gpu = bool(FAISS_GPU) and torch.cuda.is_available()
    if bool(FAISS_GPU) and not torch.cuda.is_available():
        print("[warn] 你设置了 FAISS_GPU=True，但当前没有 CUDA，将自动回退到 CPU FAISS。")

    model = WhisperEncoderWrapper(
        model_name=WHISPER_MODEL_NAME,
        rep_layer=int(run_config["REP_LAYER"]),
    ).to(device)
    model.eval()

    load_stage1_ckpt_into_whisper_model(model.model, stage1_ckpt_path)

    train_ds = ChunkAudioDataset(train_entries)

    train_loader = data.DataLoader(
        train_ds,
        batch_size=int(TRAIN_BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=train_ds.repeat30s_collate,
        pin_memory=True,
        persistent_workers=(int(NUM_WORKERS) > 0),
    )

    train_infer_loader = data.DataLoader(
        train_ds,
        batch_size=int(INFER_BATCH_SIZE),
        shuffle=False,
        num_workers=int(NUM_WORKERS),
        drop_last=False,
        collate_fn=train_ds.repeat30s_collate,
        pin_memory=True,
        persistent_workers=(int(NUM_WORKERS) > 0),
    )

    val_infer_loader = None

    print(f"[Stage2-MMSE][Layer{run_config['REP_LAYER']}] 开始收集 fold-train 特征用于聚类 ...")
    X_train = collect_train_samples(
        model=model,
        loader=train_loader,
        device=device,
        sample_ratio=SAMPLE_RATIO,
        sample_per_utt=SAMPLE_PER_UTT,
        sample_mode=SAMPLE_MODE,
    )
    print(f"[Stage2-MMSE][Layer{run_config['REP_LAYER']}] 收集完成：样本数={X_train.shape[0]}，维度={X_train.shape[1]}")

    ks = [int(x) for x in KS]
    if len(ks) == 0:
        raise ValueError("KS 不能为空。")

    index_list = []
    kmeans_stats = []
    for k in tqdm(ks, desc="训练 FAISS KMeans(各K)", dynamic_ncols=True):
        print(f"\n[Stage2-MMSE][Layer{run_config['REP_LAYER']}] 训练 KMeans: K={k} (gpu={use_faiss_gpu}) ...")
        centroids, index = fit_faiss_kmeans(
            X_train=X_train,
            k=int(k),
            seed=int(SEED),
            niter=int(KMEANS_MAX_ITER),
            nredo=int(FAISS_NREDO),
            use_gpu=use_faiss_gpu,
            max_points_per_centroid=int(FAISS_MAX_POINTS_PER_CENTROID),
            verbose=bool(FAISS_VERBOSE),
        )
        index_list.append(index)
        kmeans_stats.append(
            {
                "k": int(k),
                "centroids_shape": [int(x) for x in centroids.shape],
                "feature_dim": int(centroids.shape[1]),
            }
        )

    infer_and_export_split_json(
        model=model,
        loader=train_infer_loader,
        index_list=index_list,
        ks=ks,
        device=device,
        outputdir=run_config["STAGE2_TRAIN_DIR"],
        split_name="train",
        pad_to_full=bool(PAD_TO_FULL_LENGTH),
        entries=train_entries,
    )

    if bool(STAGE2_EXPORT_FOLD_VAL_LABELS):
        raise RuntimeError(
            "当前目标是 no-val-last：Stage2 不允许导出 fold_val pseudo labels。"
        )
    train_all_clus_path = os.path.join(run_config["STAGE2_TRAIN_DIR"], "all-clus.json")

    check_all_clus_heads(train_all_clus_path, KS)

    write_stage2_run_meta(
        cfg=run_config,
        stage1_ckpt_path=stage1_ckpt_path,
    )

    print(f"[Stage2][NO-VAL-LAST] checked train all-clus heads: KS={KS}")
    print(f"[Stage2][NO-VAL-LAST] wrote stage2_run_meta.json -> {os.path.join(run_config['STAGE2_SAVE_DIR'], 'stage2_run_meta.json')}")

    if bool(SAVE_EXPORT_METADATA):
        stage2_meta = {
            "dataset_option": str(run_config["DATASET_OPTION"]),
            "pitt_subject_disjoint": bool(str(run_config["DATASET_OPTION"]) == "Pitt" and PITT_ENABLE_SUBJECT_DISJOINT_SPLIT),
            "pitt_split_policy": (PITT_SPLIT_POLICY if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
            "pitt_expected_final_train_subjects": (PITT_EXPECTED_FINAL_TRAIN_SUBJECTS if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
            "pitt_expected_final_train_visits": (PITT_EXPECTED_FINAL_TRAIN_VISITS if str(run_config["DATASET_OPTION"]) == "Pitt" else None),
            "rep_layer": int(run_config["REP_LAYER"]),
            "fold_id": None,
            "fold_disabled": True,
            "run_protocol": str(run_config.get("RUN_PROTOCOL_TAG", "")),
            "fold_json_path": "",
            "run_root": str(run_config["RUN_ROOT"]),
            "stage1_save_dir": str(run_config["STAGE1_SAVE_DIR"]),
            "stage2_save_dir": str(run_config["STAGE2_SAVE_DIR"]),
            "stage1_ckpt_path": str(stage1_ckpt_path),
            "whisper_model_name": str(WHISPER_MODEL_NAME),

            "ks": [int(x) for x in KS],
            "sample_mode": str(SAMPLE_MODE),
            "sample_ratio": float(SAMPLE_RATIO) if SAMPLE_RATIO is not None else None,
            "sample_per_utt": int(SAMPLE_PER_UTT),
            "segment_sec": float(SEGMENT_SEC),

            "overlap_rate": float(OVERLAP_RATE),
            "chunk_key_version": str(CHUNK_KEY_VERSION),
            "unfreeze_encoder_layers": int(UNFREEZE_ENCODER_LAYERS),
            "layer_name": str(run_config["LAYER_NAME"]),
            "train_num_wavs": int(len(train_wav_entries)),

            "official_train_num_wavs_raw_from_stage1_manifest": int(len(train_wav_entries_raw)),
            "official_train_missing_mmse_wavs_stage2": int(len(missing_train_mmse)),
            "fold_val_num_wavs_held_out": 0,
            "train_num_chunks": int(len(train_entries)),
            "val_num_chunks": 0,
            "stage2_export_fold_val_labels": False,
            "upstream_uses_fold_val": False,

            "pad_to_full_length": bool(PAD_TO_FULL_LENGTH),
            "faiss_gpu": bool(use_faiss_gpu),
            "kmeans_stats": kmeans_stats,
        }
        save_json(stage2_meta, os.path.join(run_config["STAGE2_SAVE_DIR"], "clusters_meta.json"))
        save_json(stage2_meta, os.path.join(run_config["META_SAVE_DIR"], "stage2_meta.json"))
        save_text(stage1_ckpt_path, os.path.join(run_config["META_SAVE_DIR"], "stage1_ckpt_used.txt"))

    if bool(ENABLE_STAGE_QUALITY_EVAL):
        raise RuntimeError(
            "no-val-last 设置下应保持 ENABLE_STAGE_QUALITY_EVAL=False，"
            "避免 Stage2 使用 fold_val 做质量判断。"
        )

    print(f"\n[Stage2-MMSE] 完成：{run_config['STAGE2_SAVE_DIR']}")

# 批量运行入口
def main() -> None:
    """主入口："""
    global DATASET_OPTION, UNFREEZE_ENCODER_LAYERS

    if len(DATASET_OPTIONS) == 0:
        raise ValueError("DATASET_OPTIONS 为空，至少需要一个数据集。")
    if len(UNFREEZE_ENCODER_LAYERS_LIST) == 0:
        raise ValueError("UNFREEZE_ENCODER_LAYERS_LIST 为空，至少需要一个解冻层数。")

    print("[CONFIG]")
    print("  PROJECT_ROOT =", PROJECT_ROOT)
    print("  MMSE_PTAPT_ROOT =", MMSE_PTAPT_ROOT)
    print("  DATASET_OPTIONS =", DATASET_OPTIONS)
    print("  RUN_PROTOCOL_TAG =", RUN_PROTOCOL_TAG)
    print("  STAGE2_REP_LAYERS_TO_RUN =", STAGE2_REP_LAYERS_TO_RUN)
    print("  UNFREEZE_ENCODER_LAYERS_LIST =", UNFREEZE_ENCODER_LAYERS_LIST)
    print("  KS =", KS)
    print("  SAMPLE_MODE =", SAMPLE_MODE)
    print("  SAMPLE_PER_UTT =", SAMPLE_PER_UTT)
    print("  fold disabled = True")
    print("  protocol subdir disabled = True")

    for dataset_option in [str(x) for x in DATASET_OPTIONS]:
        for unfreeze_count in [int(x) for x in UNFREEZE_ENCODER_LAYERS_LIST]:
            DATASET_OPTION = dataset_option
            UNFREEZE_ENCODER_LAYERS = int(unfreeze_count)

            print("\n" + "=" * 100)
            print("[Stage2 BATCH ITEM]")
            print("  DATASET_OPTION =", DATASET_OPTION)
            print("  UNFREEZE_ENCODER_LAYERS =", UNFREEZE_ENCODER_LAYERS)
            print("=" * 100)

            for rep_layer in [int(x) for x in STAGE2_REP_LAYERS_TO_RUN]:
                run_config = build_stage2_run_config(
                    dataset_option=DATASET_OPTION,
                    rep_layer=int(rep_layer),
                    fold_id=PROTOCOL_FOLD_ID,
                )
                run_one_experiment(run_config)

if __name__ == "__main__":
    main()
