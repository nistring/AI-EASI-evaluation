# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Research code for the manuscript *"Probabilistic Ordinal Segmentation for Assessing Eczema Area and Severity Index"*. Manuscript drafts and related working notes (including `menuscript/REVISION_PLAN.md`) live only locally in `menuscript/`. It is gitignored and must never be committed or pushed: the GitHub repo is public.

A Hierarchical Probabilistic U-Net (HPU-Net) predicts four pixel-wise severity maps (erythema, edema/papulation(induration), excoriation, lichenification; grades 0–3) through pixel-wise ordinal logistic regression. EASI is never a training target: it is derived at inference time from T = `test.mc_n` Monte-Carlo samples.

## Environment

- Conda env `atopy` (Python 3.12.0, torch 2.6.0+cu118, lightning 2.2.3, timm, pingouin): `conda activate atopy`.
- The host has **no internet access**. Only pretrained weights already in `~/.cache/torch/hub/checkpoints` (ResNet-18/34/50, EfficientNet-b0–b4 in `efficientnet_pytorch` format, MiT-b0–b2) can be used; anything else has to be copied in manually.
- There is no test suite, linter, or build step.
- Hardware: 4× RTX A6000 (48 GB), shared with other users — check `nvidia-smi` before launching; usually only 1–2 GPUs are free.

## Commands

Run everything from the repo root: dataset paths in `loader.py` are relative (e.g. `data/'19/...`, note the apostrophe in the directory names).

```bash
# Train (writes lightning_logs/version_N/; config.yaml is copied there after fit)
python main.py --phase train --devices 4 --cfg config.yaml

# Test a checkpoint on a named dataset: 19_int | 19_ext (ROI) | 20_int | 20_ext (whole-body)
python main.py --phase test --checkpoint lightning_logs/0.0/version_0/checkpoints/epoch=195-step=4704.ckpt \
    --devices 0 --test-dataset 19_int --cfg lightning_logs/0.0/version_0/config.yaml

# Predict on unlabeled images in data/roi_predict or data/wb_predict/{image,mask} -> results/images/
python main.py --phase predict --checkpoint <ckpt> --devices 0 --test-dataset wb_predict

# Statistics/figures from results.pkl (edit `root_dir` at the bottom first — CLI args are ignored)
python stat/analyze.py
```

Gotchas:
- `--devices` means a GPU **count** for `train` but a GPU **index** for `test`/`predict` (`pl.Trainer(devices=[args.devices])`).
- `main.py` hard-codes `os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"`, overriding any value set in the shell; GPU selection is only via `--devices`.
- Pass the run's own `config.yaml` with `--cfg` when testing: the model/ablation settings come from the config, not from the checkpoint.
- `predict` branches on `cfg.train.wholebody`, so the config must match the checkpoint type (ROI vs whole-body).
- `test.mc_n` must be divisible by 8 for whole-body inference (8 test-time orientations).
- Test outputs go to `<version_dir>/results/<test-dataset>/` (`results.pkl`, `easi.csv`, prediction-vs-GT overlay JPGs). The path is derived from the global `args.checkpoint` inside `LitModel`.

## Local-only directories (gitignored)

`lightning_logs/`, `results/`, `figure/`, `stat/`, `scripts/` exist only in the main checkout, not in fresh clones or worktrees. `data/` is huge and mostly untracked. Never delete them.

- `scripts/*.sh`: the loops actually used for the paper (multi-seed training, testing every checkpoint, predicting).
- `stat/analyze.py`: per-version regression plots + PCC/MAE/Spearman/GED → `<version>/figure/<dataset>/` and `<root_dir_name>_statistics.csv`. `stat/temp.py` aggregates mean/std over seeds; `stat/temp2.py` builds `results/Results_AI.xlsx` from `easi.csv`; `stat/temp3.py` aggregates whole-body photos into regions and computes ICCs against the three dermatologists in `results/Results_expert+AI.xlsx` (Korean sheet/column names).
- `lightning_logs/` folder name = experiment id: `0.0` ordinal HPU-Net (the main model) · `0.0_no_ordinal_reg` HPU-Net with softmax · `0.0_deterministic` ordinal U-Net · `0.0_both` plain U-Net · `0.2`–`1.0` class-imbalance-weight sweep (not in the paper) · `0.0_whole` whole-body fine-tune (single run). Each ROI experiment has 5 unseeded runs (`version_*`). Stat scripts reference `lightning_logs/0.0_wb_new` (5 whole-body runs), which is not in this checkout.

## Architecture

**Config (`config.yaml`)** has one active `train` block; the ROI and whole-body settings differ (the alternative is kept as a comment). `train.wholebody: True` switches the data module to a `CombinedLoader` of ROI images plus non-AD whole-body patches, and `batch_size` becomes a list `[roi, wb]`. Variants are selected with `model.ablation_study.deterministic_model` (prior uses latent means, no KL term → "U-Net") and `model.ablation_study.without_ordinal_regression` (4×4-channel softmax head instead of cumulative-logit ordinal output). `model.imbalance` mixes in the class weights (0.0 in the paper, so the weights are unused).

**Multi-rater labels (`loader.ROIDataset` + `LitModel.forward_step`).** Each ROI image has 2 binary lesion masks (labeller subfolders `동현`, `재성`) and 5 dermatologists' sign grades (`meta_result.pkl` `annotations[:, :4]` → shape (5, 4)). `forward_step` broadcasts mask × grade into 10 label maps and repeats the image 10×, so the effective batch is 10× the loader batch. Non-AD whole-body images (`WholeBodyTrainDataset`, `CustomCrop`) contribute 10 random crops each with all-zero labels.

**Model (`model/model.py`).** `HierarchicalProbUNet` = prior and posterior `_HierarchicalCore` (9-level residual U-Net encoder, base width 24, 4 latent scales with 1 channel each) + `_StitchingDecoder` (`f_comb`), with 4 output channels (one per sign) passed through `log_cumulative` (model_utils) using learnable `cutpoints`. The cutpoints are stored divided by `alpha` = 100 and multiplied back in use, which gives them an effective 100× learning rate. Loss = Dice over (sign × grade) + `beta`·KL. β is constant in the code (there is no per-epoch annealing). The model has **no BatchNorm or Dropout**, so `sync_batchnorm` is a no-op and gradient accumulation on fewer GPUs reproduces multi-GPU DDP exactly (same per-GPU micro-batch, `accumulate_grad_batches = 4 / n_gpus`). `sample()` returns per-pixel class probabilities of shape (B, mc_n, C, H, W, 4).

**EASI computation (`utils.cal_EASI`, `utils.area2score`).** A pixel is lesional when the sum of its predicted grades over the 4 signs is > 1. The area score is the continuous piecewise-linear `area2score` (not the clinical integer bins). Sign severity = mean grade ÷ sample-averaged lesion fraction. Per-sample EASI uses the area score of the **sample-mean** area. ROI ground-truth area also goes through `area2score` (applied to the mask fractions).

**Whole-body inference (`LitModel.inference_wholebody`, `loader.WholeBodyTestDataset`).** Mask the background → crop to the mask's bounding box → rescale by 256/384 → pad → 256×256 patches with **stride** `test.step` = 96 → 8 orientations (4 rotations × flip) × `mc_n/8` latent samples → weight the probabilities with a 2D cosine window and normalise by the summed window → unpad/mask → argmax. Samples are drawn independently per patch. Per-photo EASI (`easi.csv`) is aggregated into regions/total offline (`stat/temp3.py`: HN = (3·front + rear)/4, then region weights).

**`results.pkl` layout** (ROI): `gt_area` (B, 2), `gt_severity` (B, 5, 4), `gt_easi` (B, 10, 4), `area` (B, T), `severity`/`easi` (B, T, 4), `ged` (B,). Most re-analyses can be run from these files without a GPU.

**Datasets (`loader.get_dataset`)**: `roi_train` (SNUH `train.txt`, 80/20 train/tune split with `random_state=42`), `wb_train` (`'20 Int.` NL images), `19_int` (`test.txt`), `19_ext` (SNUBH), `20_int` / `20_ext` (whole-body AD; GT from `meta_result.pkl`, no pixel labels), `roi_predict`, `wb_predict`. Present on disk but not wired in: `'20 Ext. (SNUBH-adult)/whole` + `labels/{lesion_area,skin_area}` (353 photos with pixel lesion masks) and `'21 Ext. (SNUBH-child)` (944 pediatric images, single-rater grades plus lesion/skin masks).
