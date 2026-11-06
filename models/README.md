# Model checkpoint

Place the trained weights here as **`best_model.pth`** (or point `MODEL_PATH` in `.env` elsewhere).
Weights are git-ignored. No checkpoint ships with this repository, and the backend never invents one.

## What the backend expects

* A PyTorch checkpoint for a `segmentation_models_pytorch` model: 3 input channels, **1 output class**
  (binary building mask, raw logits; the backend applies a sigmoid).
* Either a plain `state_dict`, or a dict containing one under `model_state_dict`, `state_dict`, `model`,
  `net` or `network`. A `module.` prefix (from `DataParallel`) is stripped automatically.
* Architecture/encoder in `.env` must match training: `MODEL_ARCHITECTURE=Unet`, `MODEL_ENCODER=resnet34`.
* Loaded with `torch.load(..., weights_only=True)`; save with `torch.save(model.state_dict(), "best_model.pth")`.

## Kaggle export snippet

```python
import segmentation_models_pytorch as smp, torch
model = smp.Unet("resnet34", encoder_weights=None, in_channels=3, classes=1)  # same as training
# ... train ...
torch.save(model.state_dict(), "best_model.pth")
```

## Preprocessing must match training

Default: RGB scaled to 0-1, then ImageNet mean/std (`NORMALIZE=imagenet`). If you trained with only `/255`,
set `NORMALIZE=none`. A mismatch will not crash, it will quietly give poor masks.
Also keep imagery resolution (ground sampling distance) similar to the training tiles.

## Optional: `training_report.json` (real results for the Performance page)

Export from your notebook; shown on the Performance page. Only put real numbers in it.

```json
{
  "summary":     {"IoU": 0.0, "Dice": 0.0, "Precision": 0.0, "Recall": 0.0, "Building-count error": 0},
  "comparison":  [{"metric": "IoU", "Edge": 0.0, "CNN": 0.0}],
  "training":    [{"epoch": 1, "loss": 0.0, "iou": 0.0}],
  "experiments": [{"name": "Tier 1", "iou": 0.0, "dice": 0.0, "precision": 0.0, "recall": 0.0}]
}
```
(The zeros above are placeholders for the format only.)
