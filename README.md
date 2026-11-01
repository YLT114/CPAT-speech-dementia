# **Fine-Grained Acoustic-Only Speech-Based Cognitive Assessment in Dementia Through Cluster-Based Pseudo-Label Fine-Tuning of Whisper**

This repository contains the implementation of **Cluster Pseudo-Label Fine-Tuning (CPFT)**, a Whisper fine-tuning method for acoustic-only cognitive assessment. The method supports MMSE regression, binary dementia classification, and three-class HC/MCI/dementia classification.

## **Dataset**

Our experiments were conducted on four speech datasets: ADReSS, ADReSSo, DementiaBank (DB), and PROCESS-2. The dataset statistics are shown below.

| Dataset | Samples (HC/MCI/AD) | Segments (HC/MCI/AD) | MMSE |
|:---|---:|---:|---:|
| ADReSS | 78 / 0 / 78 | 257 / 0 / 325 | 23.37 +/- 6.90 |
| ADReSSo | 115 / 0 / 122 | 383 / 0 / 505 | 23.23 +/- 6.86 |
| DB | 222 / 0 / 255 | 691 / 0 / 960 | 22.92 +/- 6.55 |
| PROCESS-2 | 200 / 150 / 50 | 732 / 512 / 151 | 26.38 +/- 3.28 |

Sample and segment counts are reported in the order HC/MCI/AD. The Dementia category in PROCESS-2 is reported as AD in this table.

All audio recordings are loaded at 16 kHz and divided into 30-second segments with 30% overlap. In the code, `ADReSS-IS2020`, `ADReSSo21`, and `Pitt` correspond to the paper names ADReSS, ADReSSo, and DB, respectively.

## **Models**

The `data` folder contains frozen Whisper feature extraction, the Vanilla FT, self-supervised adaptation (SSA), and Task-Adaptive Pretraining (TAP) baselines, together with the three-stage CPFT pipelines for MMSE regression, binary classification, and three-class classification.

The `models` folder contains the downstream LRU model and the MLP, LSTM, Transformer, Mamba-3, and Slice comparison models. The main hyperparameters are documented in `hyperparameter.txt`. MMSE regression is evaluated using RMSE, MAE, R2, and Pearson correlation. Classification is evaluated using accuracy, precision, recall, and Macro-F1.

Segment-level features and predictions are mapped back to their original recordings through the saved `indices` arrays. The reported metrics are calculated at the recording level.

## Operations Guide

CPFT consists of four sequential upstream scripts followed by downstream model training. For MMSE regression, run the following scripts in order:

```text
CPFT_stage1.py -> CPFT_stage2.py -> CPFT_stage3.py -> CPFT_exteact.py -> lru.py
```

Stage 1 performs task-aware Whisper adaptation. Stage 2 applies K-means to the adapted representations and generates cluster pseudo-labels. Stage 3 performs masked cluster prediction together with the MMSE objective. The extraction script exports the adapted Whisper features, which are then used to train and evaluate the downstream LRU model.

For binary classification, use:

```text
CPFT_2class_stage1.py -> CPFT_2class_stage2.py -> CPFT_2class_stage3.py -> CPFT_2class_extract.py -> lru_2class.py
```

For PROCESS-2 three-class classification, use:

```text
CPFT_3class_stage1.py -> CPFT_3class_stage2.py -> CPFT_3class_stage3.py -> CPFT_3class_extract.py -> lru_3class.py
```

Before execution, configure the username, project root, dataset paths, local Whisper checkpoint, GPU ID, dataset option, Whisper layer, and number of unfrozen layers at the top of the corresponding scripts. The server code paths are:

```text
/home/<username>/alzheimer_environment/data
/home/<username>/alzheimer_environment/models
```

The CPFT scripts use the `whisper_ptapt_faiss` environment, while the LRU scripts use `audiojax_gpu`. All upstream stages use the official training split only; the official test split is used for final feature export and report the final results.

## Environment Specifications

### Software Environment

| Environment | Python | Main use |
|:---|:---:|:---|
| `audiojax_gpu` | 3.9.25 | LRU, LSTM, Transformer, and MLP |
| `audiojax_gpu_mamba3` | 3.10.20 | Mamba-3 |
| `slice_mmse` | 3.11.15 | Slice model |
| `whisper_ptapt_faiss` | 3.10.19 | Whisper adaptation, CPFT, and FAISS clustering |

The complete package versions for the four independent environments are recorded in `requirements.txt`.

## **Result**

The `results` folder contains the detailed multi-seed results for ADReSS, ADReSSo, DB, and PROCESS-2. Each workbook reports the mean and standard deviation of the evaluation metrics over five random seeds.
