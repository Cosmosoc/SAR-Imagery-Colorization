"""
Preprocessing script for QXSLAB_SAROPT SAR images to match Sentinel-1 characteristics
This script applies speckle filtering, normalization, and intensity adjustments
to align QXSLAB_SAROPT images with the training distribution of the SAR2Optical model.
"""

import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from typing import Tuple, Optional
import cv2
from scipy.ndimage import median_filter, uniform_filter


class SARPreprocessor:
    """Preprocesses SAR images to match Sentinel-1 characteristics"""

    def __init__(
        self,
        speckle_filter: str = "lee",
        window_size: int = 5,
        target_percentile_low: float = 2.0,
        target_percentile_high: float = 98.0,
        normalize: bool = True,
        gamma_correction: float = 1.0
    ):
        """
        Initialize the SAR preprocessor

        Args:
            speckle_filter: Type of speckle filter ('lee', 'frost', 'median', 'gaussian', 'bilateral')
            window_size: Window size for speckle filtering (odd number, e.g., 3, 5, 7)
            target_percentile_low: Lower percentile for intensity clipping (default: 2.0)
            target_percentile_high: Upper percentile for intensity clipping (default: 98.0)
            normalize: Whether to normalize to 0-255 range
            gamma_correction: Gamma correction value (1.0 = no correction)
        """
        self.speckle_filter = speckle_filter
        self.window_size = window_size
        self.target_percentile_low = target_percentile_low
        self.target_percentile_high = target_percentile_high
        self.normalize = normalize
        self.gamma_correction = gamma_correction

    def lee_filter(self, img: np.ndarray, window_size: int = 5) -> np.ndarray:
        """
        Apply Lee speckle filter to reduce multiplicative noise in SAR images

        Args:
            img: Input image as numpy array (0-255 or 0-1 range)
            window_size: Size of the filter window (must be odd)

        Returns:
            Filtered image as numpy array
        """
        img = img.astype(np.float64)

        # Calculate local mean and variance
        img_mean = uniform_filter(img, size=window_size)
        img_sqr_mean = uniform_filter(img**2, size=window_size)
        img_variance = img_sqr_mean - img_mean**2

        # Overall variance
        overall_variance = np.var(img)

        # Avoid division by zero
        img_variance = np.maximum(img_variance, 0)

        # Lee filter coefficient
        # Avoid division by zero in variance calculation
        weights = img_variance / (img_variance + overall_variance + 1e-10)

        # Apply filter
        img_filtered = img_mean + weights * (img - img_mean)

        return np.clip(img_filtered, 0, 255).astype(np.uint8)

    def frost_filter(self, img: np.ndarray, window_size: int = 5, damping: float = 1.0) -> np.ndarray:
        """
        Apply Frost speckle filter

        Args:
            img: Input image
            window_size: Size of the filter window
            damping: Damping factor for the filter

        Returns:
            Filtered image
        """
        img = img.astype(np.float64)

        # Calculate local statistics
        img_mean = uniform_filter(img, size=window_size)
        img_sqr_mean = uniform_filter(img**2, size=window_size)
        img_variance = np.maximum(img_sqr_mean - img_mean**2, 0)

        # Coefficient of variation
        coef_var = np.sqrt(img_variance) / (img_mean + 1e-10)

        # Frost filter weights
        weights = np.exp(-damping * coef_var)

        # Apply filter
        img_filtered = weights * img + (1 - weights) * img_mean

        return np.clip(img_filtered, 0, 255).astype(np.uint8)

    def apply_speckle_filter(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the selected speckle filter

        Args:
            img: Input image as numpy array

        Returns:
            Filtered image
        """
        if self.speckle_filter == "lee":
            return self.lee_filter(img, self.window_size)
        elif self.speckle_filter == "frost":
            return self.frost_filter(img, self.window_size)
        elif self.speckle_filter == "median":
            return median_filter(img, size=self.window_size).astype(np.uint8)
        elif self.speckle_filter == "gaussian":
            return cv2.GaussianBlur(img, (self.window_size, self.window_size), 0)
        elif self.speckle_filter == "bilateral":
            return cv2.bilateralFilter(img.astype(np.uint8), self.window_size, 75, 75)
        else:
            print(f"Warning: Unknown filter '{self.speckle_filter}', returning original image")
            return img

    def normalize_intensity(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize image intensity using percentile clipping

        Args:
            img: Input image

        Returns:
            Normalized image
        """
        # Calculate percentiles
        p_low = np.percentile(img, self.target_percentile_low)
        p_high = np.percentile(img, self.target_percentile_high)

        # Clip values
        img_clipped = np.clip(img, p_low, p_high)

        # Normalize to 0-255
        if p_high > p_low:
            img_normalized = ((img_clipped - p_low) / (p_high - p_low) * 255.0)
        else:
            img_normalized = img_clipped

        return img_normalized.astype(np.uint8)

    def apply_gamma_correction(self, img: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply gamma correction to adjust brightness

        Args:
            img: Input image (0-255)
            gamma: Gamma value (< 1 brightens, > 1 darkens)

        Returns:
            Gamma corrected image
        """
        if gamma == 1.0:
            return img

        # Normalize to 0-1
        img_normalized = img.astype(np.float64) / 255.0

        # Apply gamma correction
        img_gamma = np.power(img_normalized, gamma)

        # Scale back to 0-255
        return (img_gamma * 255.0).astype(np.uint8)

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """
        Apply full preprocessing pipeline

        Args:
            img: Input SAR image as numpy array

        Returns:
            Preprocessed image
        """
        # Ensure input is uint8
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)

        # Step 1: Apply speckle filter
        print(f"  Applying {self.speckle_filter} filter with window size {self.window_size}...")
        img_filtered = self.apply_speckle_filter(img)

        # Step 2: Normalize intensity
        if self.normalize:
            print(f"  Normalizing intensity (percentiles: {self.target_percentile_low}-{self.target_percentile_high})...")
            img_normalized = self.normalize_intensity(img_filtered)
        else:
            img_normalized = img_filtered

        # Step 3: Apply gamma correction
        if self.gamma_correction != 1.0:
            print(f"  Applying gamma correction (gamma={self.gamma_correction})...")
            img_corrected = self.apply_gamma_correction(img_normalized, self.gamma_correction)
        else:
            img_corrected = img_normalized

        return img_corrected


def preprocess_image(
    input_path: Path,
    output_path: Path,
    preprocessor: SARPreprocessor,
    is_grayscale: bool = False
) -> None:
    """
    Preprocess a single SAR image

    Args:
        input_path: Path to input image
        output_path: Path to save preprocessed image
        preprocessor: SARPreprocessor instance
        is_grayscale: Whether the input is grayscale
    """
    print(f"\nProcessing: {input_path.name}")

    # Load image
    img = Image.open(input_path)

    if is_grayscale or img.mode == 'L':
        # Grayscale image
        img_array = np.array(img)

        # Preprocess
        img_processed = preprocessor.preprocess(img_array)

        # Convert to RGB by replicating the channel
        img_processed_rgb = np.stack([img_processed] * 3, axis=-1)

    else:
        # RGB image - process each channel separately
        img_array = np.array(img)

        if img_array.ndim == 2:
            # If somehow it's 2D, treat as grayscale
            img_processed = preprocessor.preprocess(img_array)
            img_processed_rgb = np.stack([img_processed] * 3, axis=-1)
        else:
            # Process each channel
            channels_processed = []
            for i in range(min(3, img_array.shape[2])):
                channel = img_array[:, :, i]
                channel_processed = preprocessor.preprocess(channel)
                channels_processed.append(channel_processed)

            img_processed_rgb = np.stack(channels_processed, axis=-1)

    # Save
    output_img = Image.fromarray(img_processed_rgb, mode='RGB')
    output_img.save(output_path)
    print(f"  Saved to: {output_path}")


def preprocess_directory(
    input_dir: Path,
    output_dir: Path,
    preprocessor: SARPreprocessor,
    is_grayscale: bool = False,
    file_extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
) -> None:
    """
    Preprocess all images in a directory

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save preprocessed images
        preprocessor: SARPreprocessor instance
        is_grayscale: Whether inputs are grayscale
        file_extensions: Tuple of valid file extensions
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_files = []
    for ext in file_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"\nFound {len(image_files)} images to process")

    # Process each image
    for img_path in image_files:
        output_path = output_dir / img_path.name
        try:
            preprocess_image(img_path, output_path, preprocessor, is_grayscale)
        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess QXSLAB_SAROPT SAR images to match Sentinel-1 characteristics"
    )

    # Input/output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image file or directory containing SAR images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for preprocessed images"
    )

    # Preprocessing parameters
    parser.add_argument(
        "--filter",
        type=str,
        default="lee",
        choices=["lee", "frost", "median", "gaussian", "bilateral"],
        help="Speckle filter type (default: lee)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Filter window size, must be odd (default: 5)"
    )
    parser.add_argument(
        "--percentile-low",
        type=float,
        default=2.0,
        help="Lower percentile for intensity clipping (default: 2.0)"
    )
    parser.add_argument(
        "--percentile-high",
        type=float,
        default=98.0,
        help="Upper percentile for intensity clipping (default: 98.0)"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Gamma correction value (default: 1.0, no correction)"
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable intensity normalization"
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Treat input as grayscale and convert to RGB"
    )

    args = parser.parse_args()

    # Validate window size
    if args.window_size % 2 == 0:
        print("Warning: Window size should be odd. Adjusting to", args.window_size + 1)
        args.window_size += 1

    # Create preprocessor
    preprocessor = SARPreprocessor(
        speckle_filter=args.filter,
        window_size=args.window_size,
        target_percentile_low=args.percentile_low,
        target_percentile_high=args.percentile_high,
        normalize=not args.no_normalize,
        gamma_correction=args.gamma
    )

    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return

    print("=" * 60)
    print("SAR Image Preprocessing for SAR2Optical Model")
    print("=" * 60)
    print(f"Filter: {args.filter}")
    print(f"Window size: {args.window_size}")
    print(f"Intensity percentiles: {args.percentile_low}-{args.percentile_high}")
    print(f"Gamma correction: {args.gamma}")
    print(f"Normalization: {not args.no_normalize}")
    print("=" * 60)

    if input_path.is_file():
        # Single file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        preprocess_image(input_path, output_path, preprocessor, args.grayscale)
    elif input_path.is_dir():
        # Directory
        preprocess_directory(input_path, output_path, preprocessor, args.grayscale)
    else:
        print(f"Error: Invalid input path: {input_path}")
        return

    print("\n" + "=" * 60)
    print("Preprocessing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
