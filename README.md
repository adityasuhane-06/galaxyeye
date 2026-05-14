# GalaxEye EO-SAR Binary Change Detection

## Project Title & Description

This repository contains an end-to-end solution for the GalaxEye AI Research Intern assignment on binary pixel-level change detection from paired Electro-Optical (EO) and Synthetic Aperture Radar (SAR) imagery.

The model receives a pre-event RGB EO image and a post-event single-channel SAR image, builds a 4-channel tensor, and predicts a binary change mask. The final approach uses a dual-encoder late-fusion U-Net with ImageNet-pretrained ResNet encoders, BCE-Dice loss, foreground-aware crop sampling, EO/SAR domain augmentation, and validation-threshold tuning.

The assignment-mandated label remapping is applied before all training and evaluation:

| Original value | Original class | Binary value | Binary class |
| ---: | --- | ---: | --- |
| 0 | Background | 0 | No-change |
| 1 | Intact | 0 | No-change |
| 2 | Damaged | 1 | Change |
| 3 | Destroyed | 1 | Change |

## Requirements

- Python 3.10 or newer
- CUDA-capable GPU recommended
- Pinned dependencies are listed in [requirements.txt](requirements.txt)

Main packages:

| Package | Version |
| --- | --- |
| torch | 2.5.1 |
| torchvision | 0.20.1 |
| torchaudio | 2.5.1 |
| numpy | 1.26.4 |
| tifffile | 2024.5.22 |
| PyYAML | 6.0.1 |
| tqdm | 4.66.4 |
| matplotlib | 3.9.0 |
| scikit-learn | 1.5.0 |

## Environment Setup

Using `venv` on Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Using `venv` on Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For CUDA 12.1 wheels, use:

```bash
pip install -r requirements-cu121.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## Dataset Structure

Place the provided dataset under `data/raw` with the fixed split layout below:

```text
data/raw/
  train/
    train/
      pre-event/
      post-event/
      target/
  val/
    val/
      pre-event/
      post-event/
      target/
  test/
    test/
      pre-event/
      post-event/
      target/
```

Each sample is a TIFF triplet:

```text
pre-event:  RGB EO image, 1024 x 1024 x 3
post-event: SAR image,    1024 x 1024 x 1
target:     mask,         1024 x 1024 x 1
```

Optional download helpers:

```bash
bash scripts/download_data.sh
```

```powershell
.\scripts\download_data.ps1
```

Validate extracted TIFF files:

```bash
python validate_data.py --data_path data/raw/train/train
python validate_data.py --data_path data/raw/val/val
python validate_data.py --data_path data/raw/test/test
```

Inspect split statistics:

```bash
python inspect_data.py --data_path data/raw/train/train --output outputs/metrics/train_data_report.json
python inspect_data.py --data_path data/raw/val/val --output outputs/metrics/val_data_report.json
python inspect_data.py --data_path data/raw/test/test --output outputs/metrics/test_data_report.json
```

Observed class imbalance after correct binary remapping:

| Split | Samples | Change pixel fraction |
| --- | ---: | ---: |
| Train | 2781 | 1.57% |
| Validation | 334 | 2.20% |
| Provided test | 77 | 0.75% |

## Training

Train from scratch with the final configuration:

```bash
python train.py --config config.yaml --device cuda
```

The final configuration logs the random seed, image size, augmentations, model, optimizer, learning rate, batch size, epoch count, scheduler, loss settings, and checkpoint directory.

Checkpoints are saved to:

```text
outputs/checkpoints_final_conservative/
  best.pth
  last.pth
  used_config.yaml
  history.json
  train_distribution.json
  val_distribution.json
```

## Evaluation

Run validation threshold sweep:

```bash
python eval.py \
  --config config.yaml \
  --data_path data/raw/val/val \
  --weights outputs/checkpoints_final_conservative/best.pth \
  --output outputs/metrics/val_sweep.json \
  --sweep_thresholds \
  --device cuda
```

Run test evaluation using the retained reporting threshold:

```bash
python eval.py \
  --config config.yaml \
  --data_path data/raw/test/test \
  --weights outputs/checkpoints_final_conservative/best.pth \
  --output outputs/metrics/test_metrics.json \
  --threshold 0.90 \
  --device cuda
```

Optional full-image tiled evaluation:

```bash
python eval.py \
  --config config.yaml \
  --data_path data/raw/test/test \
  --weights outputs/checkpoints_final_conservative/best.pth \
  --output outputs/metrics/test_full_metrics.json \
  --threshold 0.80 \
  --full_image \
  --tile_size 384 \
  --tile_stride 256 \
  --device cuda
```

The evaluator reports IoU, precision, recall, F1, accuracy, TP/FP/TN/FN, and confusion matrix for the change class.

## Model Weights

Final checkpoint download link:

[Download `best.pth` from Google Drive](https://drive.google.com/file/d/1A7vLlMNN-TF8t2u_WcTPXZldXE5HsKe8/view?usp=sharing)

## Results

Final retained metrics:

| Split | Threshold | IoU | Precision | Recall | F1 | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Validation | 0.80 | 0.4385 | 0.6210 | 0.5987 | 0.6097 | `[[47209631, 546090], [599707, 894876]]` |
| Provided test | 0.90 | 0.0573 | 0.2911 | 0.0666 | 0.1085 | `[[11252969, 14125], [81219, 5799]]` |

## Citation / References

- Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." MICCAI, 2015.
- He, K., Zhang, X., Ren, S., and Sun, J. "Deep Residual Learning for Image Recognition." CVPR, 2016.
- Daudt, R. C., Le Saux, B., and Boulch, A. "Fully Convolutional Siamese Networks for Change Detection." ICIP, 2018.
- Chen, H. and Shi, Z. "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection." Remote Sensing, 2020.
- Chen, H., Qi, Z., and Shi, Z. "Remote Sensing Image Change Detection with Transformers." IEEE TGRS, 2022.
- Bandara, W. G. C. and Patel, V. M. "A Transformer-Based Siamese Network for Change Detection." IGARSS, 2022.
- Fang, S. et al. "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images." IEEE GRSL, 2022.
- PyTorch and torchvision documentation/codebase for pretrained ResNet backbones.

## Submission Packaging

After training, exporting the report as `reports/technical_report.pdf`, and filling the README weight link:

```powershell
.\scripts\package_submission.ps1 -FirstName YourFirstName -LastName YourLastName
```

or:

```bash
bash scripts/package_submission.sh YourFirstName YourLastName
```
