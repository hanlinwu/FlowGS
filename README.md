# FlowGS

Official implementation of **Flow-based Gaussian Splatting for
Continuous-Scale Remote Sensing Image Super-Resolution**.

[Paper](https://doi.org/10.1109/LGRS.2026.3729993) |
[arXiv](https://arxiv.org/abs/2605.22147) |
[Pretrained models](https://huggingface.co/hlwu/FlowGS)

FlowGS combines conditional flow matching with continuous-scale 2D Gaussian
splatting. It generates a detail latent in one flow step and renders the final
remote-sensing image at an arbitrary scale. This release contains the code,
configs, dataset split manifests, and pretrained weights needed for the main
experiments in the paper. Historical baselines, ablation experiments, local
logs, datasets, and generated results are intentionally excluded.

## Method at a glance

- One model supports arbitrary scale factors instead of one model per scale.
- A bank of 100 learnable Gaussian kernels and a 7 x 7 splatting region form
  the continuous feature field.
- Shortcut flow matching enables the reported single-step inference.
- Training uses online bicubic downsampling with scales sampled uniformly from
  `[1, 8]`.

## Installation

The recorded environment uses Python 3.10, PyTorch 1.13.1, torchvision 0.14.1,
and CUDA 11.7.

```bash
git clone https://github.com/hanlinwu/FlowGS.git
cd FlowGS
conda env create -f environment.yml
conda activate flowgs
```

`requirements.txt` is also provided for environments where PyTorch and CUDA
are installed separately.

## Pretrained models

Download the released checkpoints from Hugging Face:

```bash
hf download hlwu/FlowGS --include "*.ckpt" --local-dir checkpoints
```

Equivalent Python download:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="hlwu/FlowGS",
    allow_patterns=["*.ckpt"],
    local_dir="checkpoints",
)
```

The resulting layout must be:

```text
checkpoints/
├── first_stage_gs_res_v1.ckpt  # Gaussian renderer, required by the config
└── flowgs.ckpt                 # Final one-step FlowGS model
```

Both files are required for evaluation because the final config constructs the
Gaussian renderer before restoring the complete FlowGS checkpoint.

| File | Size | SHA-256 |
| --- | ---: | --- |
| `first_stage_gs_res_v1.ckpt` | 189 MiB | `68990a8884f2f27f6912664333698409b4047a1352e31317c49b5deaa1620f1b` |
| `flowgs.ckpt` | 1.3 GiB | `5874d635d1a86a6b33c44dc43c518008f094760b9e1067a54b9d610dff81f174` |

## Datasets

Download [AID](https://captain-whu.github.io/AID/),
[DOTA](https://captain-whu.github.io/DOTA/dataset.html), and
[DIOR](https://gcheng-nwpu.github.io/#Datasets) from their official sources.
Arrange the HR images as follows:

```text
load/
├── AID_split/
│   ├── train/HR/
│   ├── val/HR/
│   └── test/HR/
└── benchmark/
    ├── AID/HR/
    ├── DOTA_512/HR/
    └── DIOR/HR/
```

Use the filename manifests in `splits/` to populate each directory. The AID
split contains 8,000 training, 1,000 validation, and 1,000 test images. DOTA
images are 512 x 512 and DIOR images are 800 x 800 in the evaluation setup.

## Evaluation

The released model performs single-step inference. Reproduce the continuous-
scale AID and DOTA settings from Table I:

```bash
python test.py \
  --config configs/second_stage_fm_shortcut_1.yaml \
  --checkpoint checkpoints/flowgs.ckpt \
  --datasets AID,DOTA_512 \
  --scales 2.6,3.4,6.0,10.0 \
  --gpu 0
```

Reproduce the DIOR scales from Table II:

```bash
python test.py \
  --config configs/second_stage_fm_shortcut_1.yaml \
  --checkpoint checkpoints/flowgs.ckpt \
  --datasets DIOR \
  --scales 2.0,4.0,8.0 \
  --gpu 0
```

The evaluator reports PSNR, SSIM, FID, LPIPS, and runtime. Generated images
and metrics are written under `results/<dataset>/x<scale>/`.

## Training

Training follows three sequential stages. The paper uses online bicubic
downsampling of AID HR images and standard horizontal flip, vertical flip, and
90-degree rotation augmentation.

### 1. Initialize the autoencoder

```bash
python train.py --configs configs/first_stage_kl_v6.yaml --gpus 0
```

This stage runs for 1,000 epochs with batch size 4 and learning rate `4.5e-6`.
The loss weights are `1.0` for L1, `0.5` for the adversarial term, and `1e-6`
for KL regularization. Copy the selected checkpoint to:

```text
checkpoints/first_stage_kl_v6.ckpt
```

### 2. Train the Gaussian renderer

```bash
python train.py --configs configs/first_stage_gs_res_v1.yaml --gpus 0
```

Copy the selected checkpoint to:

```text
checkpoints/first_stage_gs_res_v1.ckpt
```

The released version of this file can be downloaded from Hugging Face when
only stage 3 training or evaluation is required.

### 3. Train shortcut flow matching

```bash
python train.py --configs configs/second_stage_fm_shortcut_1.yaml --gpus 0
```

This stage runs for 400 epochs with batch size 8, learning rate `1e-5`, 1,000
warmup steps, and EMA decay `0.999`. Each epoch contains 1,000 optimizer
iterations according to the released config.

Training logs and checkpoints are written under `logs/`. The configs are the
source of truth for model structure and all executable hyperparameters.

## Reported results

The paper reports FID / LPIPS for continuous-scale SR:

| Dataset | x2.6 | x3.4 | x6.0 | x10.0 |
| --- | --- | --- | --- | --- |
| AID | 6.59 / 0.106 | 13.56 / 0.159 | 31.94 / 0.256 | 61.57 / 0.452 |
| DOTA | 11.88 / 0.078 | 25.46 / 0.126 | 59.16 / 0.222 | 106.61 / 0.398 |

For DIOR, the reported LPIPS / FID values are `0.068 / 8.70` at x2,
`0.205 / 21.87` at x4, and `0.348 / 71.36` at x8.

## Repository scope

This repository intentionally excludes datasets, pretrained weights, generated
images, local caches, and all ablation-only code. Checkpoints are hosted at
[hlwu/FlowGS](https://huggingface.co/hlwu/FlowGS).

## Citation

If FlowGS is useful for your research, please cite:

```bibtex
@article{mo2026flow,
  title   = {Flow-Based Gaussian Splatting for Continuous-Scale Remote Sensing Image Superresolution},
  author  = {Mo, Jiangwei and Lu, Xi and Wu, Hanlin},
  journal = {IEEE Geoscience and Remote Sensing Letters},
  volume  = {23},
  pages   = {6019205--6019205},
  year    = {2026},
  doi     = {10.1109/LGRS.2026.3729993}
}
```

