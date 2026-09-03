# Deepfake-Protection

Code, trained model weights, and frozen evaluation artifacts for **CAR — a multi-expert
deepfake detector with difficulty-aware routing and curriculum training**, from the paper
*"Graceful Degradation, Not Collapse: Compositional Artifact Routing for Robust Deepfake
Detection under Quality Degradation"* (under review).

CAR couples four complementary forensic experts — **temporal**, **motion**, **spectral**, and
**boundary** — with a learned, difficulty-conditioned router, and is trained in a staged
curriculum that ends with a noise-focused finetuning stage. All released evaluation results
follow a frozen protocol: decision thresholds are selected on the **validation** split only,
and the **raw prediction scores** for every model are shipped alongside the metrics, so every
reported number and statistical test can be reproduced exactly from this repository.

> **Naming note.** The paper's motion / spectral / boundary experts correspond to the
> flow / frequency / blending modules in this codebase (`FlowExpert`, `SpectralExpert`,
> `BlendingExpert`); the temporal expert keeps its name in both.

## Repository layout

| Path | Content |
|---|---|
| `src/` | Model code (experts, router/gating, fusion, CAR assembly), data loading, training loop, metrics |
| `configs/` | Training/eval configurations (`default.yaml` holds all paper-referenced hyperparameters) |
| `scripts/` | Training, evaluation, robustness, ablation, and statistical-analysis scripts |
| `weights/` | 8 released checkpoints (see manifest below) |
| `results/` | Frozen result JSONs and raw prediction scores (`.npz`), exactly as used for the paper |
| `requirements.txt` | Python dependencies |

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.11 + PyTorch 2.x (CUDA).

**Datasets are NOT included.** Experiments use Celeb-DF++ (Li et al., 2025,
arXiv:2507.18015, released by the original Celeb-DF authors; access via the
request form at github.com/OUC-VAS/Celeb-DF-PP) and FaceForensics++ (c23).
Download them from their official sources and point `data_root` / `ff_root` in
`configs/*.yaml` to your local copies. Consistent with the ethics statement, this
repository releases code, weights, and prediction scores only — no videos or frames.

## Reproduce CAR training (single seed, 4 stages)

The full pipeline per seed (`--seed 42|43|44`), as used for the published results
(see `configs/cloud.yaml` for the exact recipe):

```bash
# Stage 1: R3 pseudo-task pretraining (experts + router)
python scripts/run_pretrain.py --config configs/cloud.yaml --mode R3 --seed 42 --max_batches 30 --eval_val

# Stage 2: integrated training (28 epochs, specialization + consistency + routing curriculum)
python scripts/train_integrated.py --config configs/cloud.yaml \
    --pretrained results/pretrain/R3/pretrained.pt --output_dir results/final_car_s42 --seed 42

# Stage 3: quality-aware curriculum finetuning (15 epochs)
python scripts/train_phase4.py --config configs/cloud.yaml \
    --pretrained results/final_car_s42/checkpoints/best_model.pt \
    --output_dir results/final_car_v2_s42 --epochs 15 --seed 42

# Stage 4: noise-focused finetuning (8 epochs) -> final model
python scripts/train_phase4.py --config configs/cloud.yaml \
    --pretrained results/final_car_v2_s42/checkpoints/best_model.pt \
    --output_dir results/final_car_v3_s42 --epochs 8 --seed 42 --noise_focus 0.5
```

To skip Stage 1, use the released `weights/pretrained_R3.pt` as the `--pretrained` input of Stage 2.

## Train the baselines

```bash
python scripts/baseline_full.py --model mesonet|efficientnet_b0|efficientnet_b3|xception --seed 42
```

Supports `--resume <checkpoint>` for exact state recovery after interruption.
`scripts/run_baseline_queue.py` runs all baselines serially with automatic resume.

## Evaluation and analysis

Key entry points (all write frozen JSONs + raw scores under `results/`):

| Script | Produces |
|---|---|
| `scripts/eval_honest.py`, `scripts/run_honest_queue.py` | Per-model / per-seed test evaluation (`results/baseline_honest/`) |
| `scripts/robustness_honest.py` | Perturbation robustness: noise / blur / JPEG (`results/robustness_honest/`) |
| `scripts/robustness_transcode.py` | Transcoding robustness (`results/robustness_transcode/`) |
| `scripts/significance_test.py`, `scripts/significance_noise.py` | Paired statistical tests from raw scores (`results/significance/`) |
| `scripts/ablation_v3.py`, `scripts/ablation_robustness.py`, `scripts/expert_only_robustness.py` | Ablations (`results/ablation_v3/`, `results/ablation_robustness/`) |
| `scripts/collapse_analysis.py` | Routing-collapse analysis (`results/collapse_analysis/`) |
| `scripts/routing_compensation.py` | Routing compensation analysis (`results/routing_compensation/`) |
| `scripts/per_method_breakdown.py`, `scripts/baseline_per_method.py` | Per-manipulation-method breakdown (`results/per_method/`) |
| `scripts/efficiency_honest.py` | Efficiency measurements (`results/efficiency_honest/`) |
| `scripts/train_joint.py`, `scripts/eval_joint.py` | FF++ + Celeb-DF++ joint training / evaluation (`results/joint_*/`) |
| `scripts/make_paper_tables.py`, `scripts/aggregate_honest.py` | Aggregated tables (`results/paper_tables/`) |
| `scripts/audit_data_leak.py` | Split-integrity audit (`results/audit/`) |

## Model weights manifest

| File | Original location | Description |
|---|---|---|
| `weights/car_seed42_final.pt` | `results/final_car_v3/checkpoints/best_model.pt` | CAR seed 42, final model (after Stage 4) |
| `weights/car_seed42_step3.pt` | `results/final_car_v2/checkpoints/best_model.pt` | CAR seed 42, after Stage 3 (quality-aware curriculum) |
| `weights/car_seed43_final.pt` | `results/cloud_recovery/final_car_v3_s43_best.pt` | CAR seed 43, final model |
| `weights/car_seed44_final.pt` | `results/cloud_recovery/final_car_v3_s44_best.pt` | CAR seed 44, final model |
| `weights/joint_best_ff.pt` | `results/joint_ff_celebdf_e12/best_ff.pt` | FF++ + Celeb-DF++ joint training, best FF++ checkpoint |
| `weights/joint_best_joint.pt` | `results/joint_ff_celebdf_e12/best_joint.pt` | FF++ + Celeb-DF++ joint training, best joint-objective checkpoint |
| `weights/joint_v1_best_ff.pt` | `results/joint_ff_celebdf/best_ff.pt` | Joint training, first run, best FF++ checkpoint |
| `weights/pretrained_R3.pt` | `results/pretrain/R3/pretrained.pt` | R3 pseudo-task pretrained initialization |

## Raw prediction scores (`.npz`)

Every evaluation ships an `.npz` with three arrays, aligned by index (video-level scores on the
Celeb-DF++ test split, N = 5418):

- `preds` — float64, raw model scores
- `labels` — int64, 0 = real, 1 = fake
- `video_ids` — object array of video identifiers (load with `allow_pickle=True`)

```python
import numpy as np
d = np.load("results/robustness_honest/car_clean_preds.npz", allow_pickle=True)
```

These files are the exact inputs used for the paper's statistical tests; no aggregation or
thresholding is baked in.

## Ethics statement

Only model weights, code, and prediction scores are released. No dataset videos or frames
are included or redistributed. The datasets (Celeb-DF++, FaceForensics++) must be obtained
through their official access procedures.

## Citation

The accompanying paper is under review. A BibTeX entry will be added upon acceptance.
