# GalaxEye EO-SAR Binary Change Detection

This repository implements a reproducible baseline for binary pixel-level change detection on the GalaxEye EO-SAR assignment dataset. The model takes a co-registered pre-event EO image and post-event SAR image, concatenates them into a 4-channel tensor, and predicts a binary change mask.

The original four mask classes are remapped before training and evaluation:

| Original value | Meaning | Binary value |
| --- | --- | --- |
| 0 | Background | 0, no-change |
| 1 | Intact | 0, no-change |
| 2 | Damaged | 1, change |
| 3 | Destroyed | 1, change |

## Requirements

- Python 3.10 or newer
- CUDA-capable GPU recommended
- Dependencies are pinned in `requirements.txt`

## Environment Setup

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 tifffile==2024.5.22 PyYAML==6.0.1 tqdm==4.66.4 matplotlib==3.9.0 scikit-learn==1.5.0
```

Equivalent one-file CUDA install:

```bash
pip install -r requirements-cu121.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 tifffile==2024.5.22 PyYAML==6.0.1 tqdm==4.66.4 matplotlib==3.9.0 scikit-learn==1.5.0
```

Equivalent one-file CUDA install:

```powershell
pip install -r requirements-cu121.txt
```

Check GPU access:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## Dataset Structure

After downloading and extracting the dataset, this code expects:

```text
data/raw/train/train/
  pre-event/
  post-event/
  target/

data/raw/val/val/
  pre-event/
  post-event/
  target/

data/raw/test/test/
  pre-event/
  post-event/
  target/
```

The scripts also accept direct split roots such as `/path/to/train` if that folder already contains `pre-event`, `post-event`, and `target`.

To download from Windows PowerShell:

```powershell
.\scripts\download_data.ps1
```

To download from bash/WSL:

```bash
bash scripts/download_data.sh
```

## Data Inspection

```bash
python inspect_data.py --data_path data/raw/train/train --output outputs/metrics/train_data_report.json
python inspect_data.py --data_path data/raw/val/val --output outputs/metrics/val_data_report.json
```

Observed on the downloaded train/validation splits:

| Split | Samples | Change pixels after remap |
| --- | ---: | ---: |
| Train | 2781 | 1.57% |
| Validation | 334 | 2.20% |

This severe imbalance motivates the BCE + Dice loss and positive class weighting in `configs/baseline.yaml`.

## Training

```bash
python train.py --config configs/baseline.yaml
```

Fast smoke-training run for a quick checkpoint:

```bash
python train.py --config configs/fast.yaml
```

Higher-accuracy run for final submission:

```bash
python train.py --config configs/high_accuracy.yaml
```

This uses the full train split and foreground-aware crop sampling, which is important because change pixels are rare.

For an RTX 3050 4GB, start with:

```bash
python train.py --config configs/high_accuracy.yaml --epochs 20 --batch_size 2 --device cuda
```

If that is too slow, use the RTX 3050 balanced config:

```bash
python train.py --config configs/rtx3050_balanced.yaml --device cuda
```

If CUDA runs out of memory, lower the batch size:

```bash
python train.py --config configs/high_accuracy.yaml --epochs 20 --batch_size 1 --device cuda
```

Useful overrides:

```bash
python train.py --config configs/baseline.yaml --epochs 10 --batch_size 2
python train.py --config configs/baseline.yaml --train_dir /path/to/train --val_dir /path/to/val
```

Checkpoints are written to:

```text
outputs/checkpoints/best.pth
outputs/checkpoints/last.pth
```

## Evaluation

Validation:

```bash
python eval.py \
  --config configs/baseline.yaml \
  --data_path data/raw/val/val \
  --weights outputs/checkpoints/best.pth \
  --output outputs/metrics/val_metrics.json \
  --visualize
```

Provided test split:

```bash
python eval.py \
  --config configs/baseline.yaml \
  --data_path data/raw/test/test \
  --weights outputs/checkpoints/best.pth \
  --output outputs/metrics/test_metrics.json \
  --visualize \
  --vis_dir outputs/visualizations/test
```

The evaluator reports IoU, precision, recall, F1, accuracy, and the binary confusion matrix for the change class.

## Model Weights

Final checkpoint link: `TODO: add public Google Drive or Hugging Face link after training`

## Results

Fill this table after training and evaluation:

| Split | IoU | Precision | Recall | F1 | Confusion Matrix `[[TN, FP], [FN, TP]]` |
| --- | ---: | ---: | ---: | ---: | --- |
| Validation | TODO | TODO | TODO | TODO | TODO |
| Test | TODO | TODO | TODO | TODO | TODO |

## Method Summary

- Input: 4 channels, consisting of 3-channel pre-event EO and 1-channel post-event SAR.
- Model: compact U-Net baseline with skip connections.
- Loss: weighted BCE + Dice loss.
- Augmentation: random crop, flips, and 90-degree rotations.
- Metrics: change-class IoU, precision, recall, F1, and confusion matrix.

## References

- Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation", 2015.
- Daudt et al., "Fully Convolutional Siamese Networks for Change Detection", 2018.
- Chen and Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection", 2020.
- Chen et al., "Remote Sensing Image Change Detection with Transformers", 2021.
- Bandara and Patel, "A Transformer-Based Siamese Network for Change Detection", 2022.
- Fang et al., "SNUNet-CD: A Densely Connected Siamese Network for Change Detection of VHR Images", 2021.

## Submission Packaging

After training, exporting the report as `reports/technical_report.pdf`, and filling the README weight link:

```powershell
.\scripts\package_submission.ps1 -FirstName YourFirstName -LastName YourLastName
```

or:

```bash
bash scripts/package_submission.sh YourFirstName YourLastName
```
