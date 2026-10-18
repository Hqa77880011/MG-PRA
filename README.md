# MG-PRA

Implementation of **Mask-Guided Periodic Reference Aggregation for Geometry-Preserving SEM Defect Restoration in Lithography Inspection**.

## Installation

Python >=3.10.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5, NumPy 2.1.3, PyTorch 2.7.1+cu118, SciPy 1.15.3 and scikit-learn 1.6.1. Checks cover Python syntax, JSON/TOML parsing and CLI options against parser definitions; training and evaluation were not run.

## Data and preprocessing

Generate periodic line images and localized synthetic corruption with known masks.

```bash
periodic-reference demo --output data/generated.npz --count 96 --config config.json --seed 13
periodic-reference prepare --input data/generated.npz --output data/prepared
```

`x[N,1,H,W]` is defective SEM intensity, `mask[N,1,H,W]` is the supplied defect mask, and `y` is its paired clean image. Keep crops of the same clean SEM source in one group. The demo synthesizes periodic line images and localized corruptions.

Numeric NPZ input uses `group` to keep related samples together. `prepare` writes train/validation/calibration/test archives from ordered groups (60/15/10/15), or honors explicit `split` codes 0/1/2/3.

## Training and inference

```bash
periodic-reference train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16 --seed 13
```

Use `--device cuda` to select CUDA; CPU is the default. Keep the configuration, splits and seed fixed across comparisons.

## Evaluation

```bash
periodic-reference evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
```

## Ablations

Variants: `full`, `baseline`, `no_mask`, `no_period_loss`. Reuse the prepared data and training seed:

```bash
periodic-reference train --input data/prepared --output runs/baseline --config config.json --variant baseline --epochs 100 --batch-size 16 --seed 13
periodic-reference evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Outputs

- `data/prepared/*.npz`: grouped data splits.
- `runs/full/model.pt`: validation-selected parameters and model configuration.
- `runs/full/history.json` and `training.json`: objectives and run settings.
- `runs/full/evaluation/predictions.npz` and `metrics.json`: predictions/mechanism arrays and computed metrics.

```bash
periodic-reference plot --input runs --output runs/figures
```

`runs/figures/*.png` compares saved evaluation metrics. All data, checkpoint and output paths are CLI arguments.
