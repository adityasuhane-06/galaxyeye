# Binary Change Detection on EO-SAR Image Pairs

## 1. Abstract

This work addresses binary pixel-level change detection on paired pre-event EO and post-event SAR imagery. The original four annotation classes are remapped into no-change and change classes before training and evaluation. The implemented baseline uses a 4-channel U-Net, combining the RGB EO image and the single-channel SAR image at the input level, trained with weighted BCE plus Dice loss to handle severe class imbalance. Final validation and test metrics should be filled after training the model.

## 2. Literature Survey

Classical remote-sensing change detection often relied on image differencing, ratioing, thresholding, and post-processing. These methods are simple and interpretable, but struggle with modality differences, illumination variation, speckle, registration noise, and complex disaster damage patterns.

Deep learning change detection reframed the task as dense prediction. U-Net-style encoder-decoder models are strong baselines because skip connections preserve spatial detail while deeper layers capture context. Daudt et al. introduced fully convolutional Siamese variants for change detection, including early fusion and Siamese feature-difference strategies. Later work such as STANet added spatial-temporal attention to better align and compare bitemporal features. Transformer-based methods such as BIT and ChangeFormer model longer-range context and can be stronger when sufficient data and compute are available. SNUNet-CD explores dense Siamese skip connections to improve multi-scale feature reuse.

EO-SAR fusion adds a cross-modal challenge: EO imagery captures spectral/visual texture, while SAR captures microwave backscatter and has different noise characteristics. For this assignment, a conservative early-fusion U-Net baseline is a practical first step because it is reproducible, fast enough to train, and directly learns from the provided data without using external remote-sensing datasets.

## 3. Methodology

The input tensor is formed by concatenating the pre-event RGB EO image and the post-event grayscale SAR image, yielding four channels. All images are scaled to `[0, 1]`. The target mask is remapped as `0,1 -> 0` and `2,3 -> 1`.

The model is a compact U-Net with convolutional encoder blocks, max-pooling downsampling, transposed-convolution upsampling, and skip connections. This architecture is chosen because binary change masks require both local boundary detail and larger contextual evidence.

The loss function is weighted BCE plus Dice loss. Weighted BCE improves sensitivity to the rare change class, while Dice directly optimizes overlap under imbalance. The downloaded training split contains approximately 1.57% change pixels after remapping, making imbalance handling necessary.

Training uses random 512 x 512 crops, horizontal/vertical flips, and 90-degree rotations. These augmentations preserve the semantic meaning of overhead imagery while increasing spatial variation.

## 4. Results

Fill after running:

```bash
python eval.py --config configs/baseline.yaml --data_path data/raw/val/val --weights outputs/checkpoints/best.pth --output outputs/metrics/val_metrics.json --visualize
python eval.py --config configs/baseline.yaml --data_path data/raw/test/test --weights outputs/checkpoints/best.pth --output outputs/metrics/test_metrics.json --visualize --vis_dir outputs/visualizations/test
```

| Split | IoU | Precision | Recall | F1 | Confusion Matrix |
| --- | ---: | ---: | ---: | ---: | --- |
| Validation | TODO | TODO | TODO | TODO | TODO |
| Test | TODO | TODO | TODO | TODO | TODO |

Qualitative examples should include at least five visualizations from `outputs/visualizations`, covering both success and failure cases.

Expected error modes to analyze:

- False positives around intact buildings with strong SAR backscatter.
- False negatives for small or thin damaged structures.
- Boundary errors caused by crop resolution and label uncertainty.
- Confusion in regions where EO/SAR appearance differs strongly for non-damage reasons.

## 5. Future Work

The next steps would be:

- Compare early fusion with Siamese dual-encoder fusion, allowing EO and SAR branches to learn modality-specific low-level features.
- Try attention-based change detection models such as STANet, BIT, or ChangeFormer if compute permits.
- Use hard-example mining or foreground-aware patch sampling to expose the model to more change pixels.
- Tune the prediction threshold using the validation split to balance recall and precision.
- Add test-time augmentation for more stable predictions.
- Perform event-wise error analysis to identify geographic or disaster-type generalization failures.

## 6. Conclusion

This baseline provides a reproducible end-to-end system for EO-SAR binary change detection, including mandatory label remapping, class-imbalance-aware training, and assignment-required metrics. Its main limitation is the simple early-fusion architecture, which may not fully exploit modality-specific EO and SAR cues. Nevertheless, it is a strong first-month deliverable because it is transparent, debuggable, and easy to extend.

## Time and Resource Log

- Data exploration: TODO hours
- Literature reading: TODO hours
- Implementation: TODO hours
- Training: TODO hours
- Evaluation: TODO hours
- Report writing: TODO hours
- Machine: TODO, local/cloud
- GPU: TODO model and VRAM
- Training time per epoch: TODO
- Total wall-clock training time: TODO
- Constraints: TODO
