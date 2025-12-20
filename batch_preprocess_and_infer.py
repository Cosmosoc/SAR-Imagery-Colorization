"""
Batch preprocessing and inference script
Preprocesses QXSLAB_SAROPT images and runs SAR2Optical inference on them
"""

import argparse
from pathlib import Path
import torch
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from tqdm import tqdm

from preprocess_qxslab import SARPreprocessor
from utils.config import Config
from src.pix2pix import Pix2Pix


def run_inference_on_preprocessed(
    input_dir: Path,
    output_dir: Path,
    config_path: str = "config.yaml",
    save_preprocessed: bool = False,
    preprocessed_dir: Path = None,
    # Preprocessing parameters
    speckle_filter: str = "lee",
    window_size: int = 5,
    percentile_low: float = 2.0,
    percentile_high: float = 98.0,
    gamma: float = 1.0,
    is_grayscale: bool = False
):
    """
    Preprocess SAR images and run inference in one pipeline

    Args:
        input_dir: Directory containing input SAR images
        output_dir: Directory to save output optical images
        config_path: Path to config.yaml
        save_preprocessed: Whether to save preprocessed SAR images
        preprocessed_dir: Directory to save preprocessed images (if save_preprocessed=True)
        speckle_filter: Type of speckle filter
        window_size: Filter window size
        percentile_low: Lower percentile for normalization
        percentile_high: Upper percentile for normalization
        gamma: Gamma correction value
        is_grayscale: Whether input is grayscale
    """

    print("=" * 70)
    print("SAR2Optical Batch Preprocessing and Inference Pipeline")
    print("=" * 70)

    # Setup directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_preprocessed and preprocessed_dir:
        preprocessed_dir.mkdir(parents=True, exist_ok=True)

    # Load model configuration
    print("\n[1/4] Loading model configuration...")
    config = Config(config_path)
    device = torch.device(config["inference"]["device"])
    print(f"  Device: {device}")

    # Create and load model
    print("\n[2/4] Loading SAR2Optical model...")
    model = (
        Pix2Pix(
            c_in=config["model"]["c_in"],
            c_out=config["model"]["c_out"],
            is_train=False,
            use_upsampling=config["model"]["use_upsampling"],
            mode=config["model"]["mode"],
        )
        .to(device)
        .eval()
    )

    gen_checkpoint = Path(config["inference"]["gen_checkpoint"])
    if not gen_checkpoint.exists():
        raise FileNotFoundError(f"Generator checkpoint not found: {gen_checkpoint}")

    model.load_model(gen_path=gen_checkpoint)
    print(f"  Loaded checkpoint: {gen_checkpoint}")

    # Create preprocessor
    print("\n[3/4] Setting up SAR preprocessor...")
    preprocessor = SARPreprocessor(
        speckle_filter=speckle_filter,
        window_size=window_size,
        target_percentile_low=percentile_low,
        target_percentile_high=percentile_high,
        normalize=True,
        gamma_correction=gamma
    )
    print(f"  Filter: {speckle_filter}")
    print(f"  Window size: {window_size}")
    print(f"  Percentiles: {percentile_low}-{percentile_high}")
    print(f"  Gamma: {gamma}")

    # Get all image files
    print("\n[4/4] Processing images...")
    image_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_dir.glob(f"*{ext}"))
        image_files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {input_dir}")
        return

    print(f"  Found {len(image_files)} images")

    # Define inference transforms
    inference_transforms = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Process each image
    print("\n" + "=" * 70)
    print("Processing images...")
    print("=" * 70)

    successful = 0
    failed = 0

    for img_path in tqdm(image_files, desc="Progress"):
        try:
            # Load image
            img = Image.open(img_path)

            # Preprocess
            if is_grayscale or img.mode == 'L':
                img_array = np.array(img)
                img_processed = preprocessor.preprocess(img_array)
                img_processed_rgb = np.stack([img_processed] * 3, axis=-1)
            else:
                img_array = np.array(img)
                if img_array.ndim == 2:
                    img_processed = preprocessor.preprocess(img_array)
                    img_processed_rgb = np.stack([img_processed] * 3, axis=-1)
                else:
                    channels_processed = []
                    for i in range(min(3, img_array.shape[2])):
                        channel = img_array[:, :, i]
                        channel_processed = preprocessor.preprocess(channel)
                        channels_processed.append(channel_processed)
                    img_processed_rgb = np.stack(channels_processed, axis=-1)

            # Save preprocessed image if requested
            if save_preprocessed and preprocessed_dir:
                preprocessed_img = Image.fromarray(img_processed_rgb, mode='RGB')
                preprocessed_path = preprocessed_dir / img_path.name
                preprocessed_img.save(preprocessed_path)

            # Convert to PIL and prepare for inference
            img_pil = Image.fromarray(img_processed_rgb, mode='RGB')
            img_tensor = inference_transforms(img_pil).unsqueeze(0).to(device)

            # Run inference
            with torch.no_grad():
                pred = model.generate(img_tensor, is_scaled=True)
                pred = torch.clamp(pred, -1, 1)
                pred = (pred + 1) / 2.0
                pred = (pred * 255).to(torch.uint8)
                pred = pred.squeeze(0).cpu().numpy().transpose(1, 2, 0)

            # Save output
            output_img = Image.fromarray(pred)
            output_path = output_dir / f"{img_path.stem}_optical{img_path.suffix}"
            output_img.save(output_path)

            successful += 1

        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            failed += 1
            continue

    # Summary
    print("\n" + "=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_files)}")
    print(f"\nOutput directory: {output_dir}")
    if save_preprocessed and preprocessed_dir:
        print(f"Preprocessed images: {preprocessed_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Batch preprocess QXSLAB_SAROPT images and run SAR2Optical inference"
    )

    # Input/output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input directory containing SAR images"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output directory for generated optical images"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config.yaml (default: config.yaml)"
    )

    # Preprocessing save options
    parser.add_argument(
        "--save-preprocessed",
        action="store_true",
        help="Save preprocessed SAR images"
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default=None,
        help="Directory to save preprocessed images (default: <output>_preprocessed)"
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
        help="Filter window size (default: 5)"
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
        help="Gamma correction value (default: 1.0)"
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Treat input as grayscale"
    )

    args = parser.parse_args()

    # Setup paths
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Setup preprocessed directory
    if args.save_preprocessed:
        if args.preprocessed_dir:
            preprocessed_dir = Path(args.preprocessed_dir)
        else:
            preprocessed_dir = output_dir.parent / f"{output_dir.name}_preprocessed"
    else:
        preprocessed_dir = None

    # Validate window size
    window_size = args.window_size
    if window_size % 2 == 0:
        print(f"Warning: Window size should be odd. Adjusting to {window_size + 1}")
        window_size += 1

    # Run pipeline
    run_inference_on_preprocessed(
        input_dir=input_dir,
        output_dir=output_dir,
        config_path=args.config,
        save_preprocessed=args.save_preprocessed,
        preprocessed_dir=preprocessed_dir,
        speckle_filter=args.filter,
        window_size=window_size,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        gamma=args.gamma,
        is_grayscale=args.grayscale
    )


if __name__ == "__main__":
    main()
