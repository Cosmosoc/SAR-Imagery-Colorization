# QXSLAB_SAROPT Preprocessing Guide

This guide explains how to preprocess QXSLAB_SAROPT SAR images to match Sentinel-1 characteristics for the SAR2Optical model.

## Quick Start

### Single Image Processing

```bash
python preprocess_qxslab.py --input /path/to/sar_image.png --output /path/to/output_image.png
```

### Batch Processing (Directory)

```bash
python preprocess_qxslab.py --input /path/to/sar_images/ --output /path/to/preprocessed/
```

## Command Line Arguments

### Required Arguments

- `--input`, `-i`: Input image file or directory containing SAR images
- `--output`, `-o`: Output file (for single image) or directory (for batch processing)

### Optional Arguments

#### Speckle Filtering

- `--filter`: Type of speckle filter to apply
  - Options: `lee` (default), `frost`, `median`, `gaussian`, `bilateral`
  - **Recommended**: `lee` or `frost` for SAR images

- `--window-size`: Size of the filter window (must be odd number)
  - Default: `5`
  - Larger values (7, 9) = more smoothing but less detail
  - Smaller values (3) = less smoothing but more detail preserved

#### Intensity Normalization

- `--percentile-low`: Lower percentile for intensity clipping
  - Default: `2.0`
  - Clips dark outliers

- `--percentile-high`: Upper percentile for intensity clipping
  - Default: `98.0`
  - Clips bright outliers

- `--no-normalize`: Disable intensity normalization (not recommended)

#### Brightness Adjustment

- `--gamma`: Gamma correction value
  - Default: `1.0` (no correction)
  - Values < 1.0: Brighten the image
  - Values > 1.0: Darken the image
  - Try values between 0.8-1.2 if results are too dark/bright

#### Image Format

- `--grayscale`: Treat input as grayscale and convert to RGB
  - Use this if your SAR images are single-channel

## Examples

### Example 1: Basic Preprocessing with Lee Filter

```bash
python preprocess_qxslab.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar_images/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/preprocessed/ \
    --filter lee \
    --window-size 5
```

### Example 2: Aggressive Speckle Filtering

For very noisy images, use a larger window:

```bash
python preprocess_qxslab.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar_images/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/preprocessed_aggressive/ \
    --filter lee \
    --window-size 7
```

### Example 3: Frost Filter with Custom Intensity Range

```bash
python preprocess_qxslab.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar_images/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/preprocessed_frost/ \
    --filter frost \
    --window-size 5 \
    --percentile-low 1.0 \
    --percentile-high 99.0
```

### Example 4: With Gamma Correction

If output images are too dark, try brightening:

```bash
python preprocess_qxslab.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar_images/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/preprocessed_bright/ \
    --filter lee \
    --gamma 0.8
```

### Example 5: Grayscale SAR Images

If your SAR images are single-channel grayscale:

```bash
python preprocess_qxslab.py \
    --input E:/SAR_Dataset/QXSLAB_SAROPT/sar_images/ \
    --output E:/SAR_Dataset/QXSLAB_SAROPT/preprocessed/ \
    --filter lee \
    --grayscale
```

## Recommended Workflow

1. **Start with default settings**:
   ```bash
   python preprocess_qxslab.py --input <input> --output <output>
   ```

2. **Test on a single image first** to find optimal parameters

3. **Compare different filters**:
   - Try `lee`, `frost`, and `bilateral` filters
   - Compare output quality

4. **Adjust window size** based on noise level:
   - Very noisy: window_size = 7 or 9
   - Moderately noisy: window_size = 5 (default)
   - Light noise: window_size = 3

5. **Fine-tune brightness** with gamma if needed

6. **Batch process** entire dataset once parameters are optimized

## Understanding the Preprocessing Steps

The script applies the following steps in order:

1. **Speckle Filtering**: Reduces multiplicative noise characteristic of SAR images
   - Lee filter: Adaptive filter based on local statistics
   - Frost filter: Exponential damping based on coefficient of variation
   - Median/Gaussian/Bilateral: Alternative smoothing methods

2. **Intensity Normalization**: Stretches the dynamic range to 0-255
   - Clips outliers based on percentiles
   - Ensures consistent brightness across images

3. **Gamma Correction**: Adjusts overall brightness/contrast
   - Compensates for differences in sensor characteristics

## Troubleshooting

### Problem: Output images are too smooth, losing detail

**Solution**: Reduce window size or try a different filter
```bash
--window-size 3
# or
--filter bilateral
```

### Problem: Output images are still very noisy

**Solution**: Increase window size or use stronger filtering
```bash
--window-size 7
# or try Frost filter
--filter frost
```

### Problem: Output images are too dark

**Solution**: Apply gamma correction
```bash
--gamma 0.8  # or try 0.7, 0.6
```

### Problem: Output images are too bright

**Solution**: Increase gamma value
```bash
--gamma 1.2  # or try 1.3, 1.4
```

### Problem: Output still has poor quality with model inference

**Solutions**:
1. Try different combinations of filters and parameters
2. Check if QXSLAB_SAROPT has different polarization than Sentinel-1
3. Consider fine-tuning the SAR2Optical model on preprocessed QXSLAB_SAROPT data
4. Verify that images are properly calibrated (radiometric correction)

## Next Steps After Preprocessing

1. Run inference on preprocessed images using the SAR2Optical model
2. Compare results with original (unpreprocessed) images
3. If results are still unsatisfactory:
   - Experiment with different preprocessing parameters
   - Consider collecting paired QXSLAB_SAROPT + optical data for fine-tuning
   - Check for fundamental differences in acquisition parameters (resolution, frequency band, etc.)

## Technical Details

### Speckle Filters

- **Lee Filter**: Optimal for preserving edges while reducing speckle
- **Frost Filter**: Good balance between smoothing and edge preservation
- **Median Filter**: Simple but effective for salt-and-pepper noise
- **Gaussian Filter**: Smooth but may blur edges
- **Bilateral Filter**: Edge-preserving smoothing, computationally intensive

### Why These Preprocessing Steps?

The SAR2Optical model was trained on Sentinel-1 data, which typically undergoes:
- Speckle filtering during preprocessing
- Radiometric calibration
- Terrain correction
- Specific intensity distribution characteristics

QXSLAB_SAROPT images may have:
- Different noise characteristics
- Different intensity distributions
- Different acquisition parameters

This preprocessing bridges the gap between the two datasets.
