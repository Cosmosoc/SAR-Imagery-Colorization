# QXSLAB_SAROPT Preprocessing for SAR2Optical Model

## Problem

The SAR2Optical pix2pix model produces poor results on QXSLAB_SAROPT dataset images because:

1. **Severe speckle noise**: QXSLAB_SAROPT images have much higher speckle noise than Sentinel-1 training data
2. **Different preprocessing**: Sentinel-1 data is typically speckle-filtered and radiometrically calibrated
3. **Domain shift**: Different intensity distributions and characteristics between datasets

## Solution

Preprocess QXSLAB_SAROPT images to match Sentinel-1 characteristics before running inference.

## Files Created

1. **[preprocess_qxslab.py](preprocess_qxslab.py)** - Standalone preprocessing script
2. **[batch_preprocess_and_infer.py](batch_preprocess_and_infer.py)** - Combined preprocessing + inference pipeline
3. **[PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md)** - Detailed documentation
4. **[example_usage.bat](example_usage.bat)** - Example commands

## Quick Start

### Option 1: Preprocess Only

```bash
# Single image
python preprocess_qxslab.py \
    --input /path/to/sar_image.png \
    --output /path/to/preprocessed_image.png

# Batch processing
python preprocess_qxslab.py \
    --input /path/to/sar_images/ \
    --output /path/to/preprocessed/
```

### Option 2: Preprocess + Inference (Recommended)

```bash
python batch_preprocess_and_infer.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/optical_output/ \
    --save-preprocessed
```

This will:
- Apply speckle filtering to reduce noise
- Normalize intensity distributions
- Run SAR2Optical inference
- Save both preprocessed SAR and output optical images

## Preprocessing Steps

The script automatically applies:

1. **Speckle Filtering** (Lee filter by default)
   - Reduces multiplicative noise in SAR images
   - Options: `lee`, `frost`, `median`, `gaussian`, `bilateral`

2. **Intensity Normalization**
   - Clips outliers using percentiles (2nd and 98th by default)
   - Stretches dynamic range to 0-255

3. **Gamma Correction** (optional)
   - Adjusts brightness if needed

## Common Usage Examples

### For Very Noisy Images

```bash
python preprocess_qxslab.py \
    --input /path/to/images/ \
    --output /path/to/output/ \
    --filter lee \
    --window-size 7  # Larger window = more smoothing
```

### If Output is Too Dark

```bash
python preprocess_qxslab.py \
    --input /path/to/images/ \
    --output /path/to/output/ \
    --gamma 0.8  # Values < 1.0 brighten the image
```

### For Grayscale SAR Images

```bash
python preprocess_qxslab.py \
    --input /path/to/images/ \
    --output /path/to/output/ \
    --grayscale
```

### Complete Pipeline with Custom Settings

```bash
python batch_preprocess_and_infer.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/results/ \
    --save-preprocessed \
    --filter lee \
    --window-size 5 \
    --gamma 0.9 \
    --percentile-low 2.0 \
    --percentile-high 98.0
```

## Parameters Reference

### Speckle Filter Options

- `lee` (recommended): Adaptive filter, good edge preservation
- `frost`: Exponential damping filter
- `median`: Simple median filter
- `gaussian`: Gaussian smoothing
- `bilateral`: Edge-preserving bilateral filter

### Window Size

- `3`: Light filtering, preserves detail
- `5`: Default, balanced filtering
- `7` or `9`: Aggressive filtering for very noisy images

### Gamma Correction

- `< 1.0`: Brightens image (try 0.7-0.9)
- `1.0`: No correction (default)
- `> 1.0`: Darkens image (try 1.1-1.3)

## Installation Requirements

Make sure you have these packages installed:

```bash
pip install numpy pillow opencv-python scipy torch torchvision tqdm
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Output too smooth | Use smaller `--window-size` (e.g., 3) |
| Output still noisy | Use larger `--window-size` (e.g., 7) or try `--filter frost` |
| Output too dark | Add `--gamma 0.8` (or lower) |
| Output too bright | Add `--gamma 1.2` (or higher) |
| Model still gives bad results | Try different filter types, adjust percentiles, or consider fine-tuning the model |

## Next Steps

1. Test preprocessing on a few sample images
2. Compare model output with/without preprocessing
3. Experiment with different filter parameters
4. If results improve but are still not satisfactory, consider fine-tuning the SAR2Optical model on preprocessed QXSLAB_SAROPT data

## Full Documentation

See [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) for complete documentation including:
- Detailed parameter explanations
- More examples
- Technical background
- Advanced troubleshooting

## Example Workflow

```bash
# 1. Test on single image first
python preprocess_qxslab.py \
    --input test_image.png \
    --output test_preprocessed.png

# 2. Check the result, adjust parameters if needed

# 3. Run full pipeline on all images
python batch_preprocess_and_infer.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/results/ \
    --save-preprocessed \
    --filter lee

# 4. Compare outputs with original inference results
```
