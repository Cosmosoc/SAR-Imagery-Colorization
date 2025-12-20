# SAR2Optical: SAR to Optical Image Translation

A deep learning model that translates Synthetic Aperture Radar (SAR) images to optical RGB images using Pix2Pix (conditional GAN) with a U-Net generator.

<img width="883" height="545" alt="image" src="https://github.com/user-attachments/assets/6e6312ba-3c9c-4dd7-9dbf-40254460c9f3" />


## Overview

This project implements SAR-to-Optical image translation, enabling the conversion of radar imagery into natural-looking optical images. The model is based on the Pix2Pix architecture and has been fine-tuned on the QXSLAB_SAROPT dataset.

**Key Features:**
- Pix2Pix with U-Net Generator (54.5M parameters)
- PatchGAN Discriminator
- Pre-trained on Sentinel-1 SAR data
- Fine-tuned on QXSLAB_SAROPT (20,000 paired images)
- Supports LoRA fine-tuning for efficient adaptation
- Streamlit web app for interactive inference

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yuuIind/SAR2Optical.git
cd SAR2Optical
pip install -r requirements.txt
```

### 2. Download Checkpoints

Download pre-trained checkpoints from the [Results & Model Files](https://github.com/yuuIind/SAR2Optical?tab=readme-ov-file#results--model-files) section.
- Place checkpoint files in the `checkpoints/` directory

### 3. Run Inference

```bash
python inference.py
```

Configure paths in `config.yaml`:
```yaml
inference:
  image_path: "path/to/your/sar_image.png"
  output_path: "./output/result.jpg"
  gen_checkpoint: "checkpoints/pix2pix_gen_180.pth"
  device: "cuda"  # or "cpu"
```

### 4. Run Web App (Optional)

```bash
streamlit run app.py
```

## Project Structure

```
SAR2Optical/
├── src/                    # Model architecture
├── utils/                  # Utilities and config
├── checkpoints/            # Model weights
├── samples/                # Sample SAR images
├── output/                 # Inference outputs
├── inference.py            # Run inference
├── train.py                # Training script
├── app.py                  # Streamlit web app
├── preprocess.py           # SAR preprocessing
├── finetune.ipynb          # Fine-tuning notebook (Colab)
├── config.yaml             # Configuration
└── requirements.txt        # Dependencies
```

## Fine-Tuning on Custom Data

### Using Google Colab (Recommended)

Open `finetune.ipynb` in Google Colab for free GPU access:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Cosmosoc/SAR-Imagery-Colorization/blob/main/finetune.ipynb)

The notebook:
1. Downloads the dataset and pre-trained checkpoint
2. Configures training parameters
3. Fine-tunes with data augmentation
4. Saves checkpoints and visualizes results

### Datasets

**QXSLAB_SAROPT Dataset (Fine-tuning):**
- [Download from Google Drive](https://drive.google.com/file/d/1835G9HBouBqmk7tKNnIc5gkJ5B8-4v9I/view)
- [GitHub Repository](https://github.com/yaoxu008/QXS-SAROPT)
- [Paper (arXiv)](https://arxiv.org/pdf/2103.08259)

Dataset structure:
```
QXSLAB_SAROPT/
├── sar_256_oc_0.2/    # SAR images (20,000)
└── opt_256_oc_0.2/    # Optical images (20,000)
```

**Sentinel-1/2 Dataset (Pre-training):**
- [Kaggle: Sentinel-1/2 Image Pairs](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)

## Preprocessing SAR Images

If your SAR images are from a different source, preprocessing may improve results.

### Quick Usage

```bash
# Single image
python preprocess.py --input /path/to/sar.png --output /path/to/output.png

# Batch processing
python preprocess.py --input /path/to/sar_folder/ --output /path/to/output_folder/
```

### Preprocessing Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--filter` | Speckle filter type: `lee`, `frost`, `median`, `gaussian`, `bilateral` | `lee` |
| `--window-size` | Filter window size (odd number) | `5` |
| `--percentile-low` | Lower percentile for intensity clipping | `2.0` |
| `--percentile-high` | Upper percentile for intensity clipping | `98.0` |
| `--gamma` | Gamma correction (< 1 brightens) | `1.0` |
| `--grayscale` | Convert grayscale to RGB | `False` |

See [PREPROCESSING_GUIDE.md](PREPROCESSING_GUIDE.md) for detailed documentation.

## Model Architecture

**Generator (U-Net):**
- 8-layer encoder-decoder with skip connections
- Input: 256x256x3 SAR image
- Output: 256x256x3 optical image

**Discriminator (PatchGAN):**
- 70x70 receptive field
- Classifies image patches as real/fake

**Loss Function:**
- Adversarial loss (BCE) + L1 reconstruction loss (λ=100)

## Training Configuration

Key parameters in `config.yaml`:

```yaml
training:
  num_epochs: 200
  batch_size: 32
  lr: 0.0002
  lambda_L1: 100.0

model:
  c_in: 3
  c_out: 3
  netD: "patch"
  n_layers: 3
```

## Sample Results

| SAR Input | Generated Optical | Ground Truth |
|-----------|-------------------|--------------|
| ![SAR Input](https://github.com/user-attachments/assets/2018e388-ddd6-4fdf-b953-615f5ab18b7c) | ![Generated](https://github.com/user-attachments/assets/9da7cb9f-a9a4-4c5b-bab7-a37a8d44d689) | ![Ground Truth](https://github.com/user-attachments/assets/54f793f9-f716-4eae-a47f-b6bd14fed61a) |


## Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

Key dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
Pillow>=9.0.0
numpy>=1.21.0
scipy>=1.7.0
streamlit>=1.28.0
```

## Files Reference

| File | Description |
|------|-------------|
| `inference.py` | Single image inference |
| `train.py` | Full training from scratch |
| `finetune.ipynb` | Fine-tuning notebook for Colab |
| `app.py` | Streamlit web interface |
| `preprocess.py` | SAR image preprocessing |
| `PREPROCESSING_GUIDE.md` | Detailed preprocessing documentation |
| `config.yaml` | All configuration options |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [Pix2Pix Paper](https://arxiv.org/abs/1611.07004) - Isola et al.
- [QXS-SAROPT Dataset](https://github.com/yaoxu008/QXS-SAROPT) - Xu et al.
- [Sentinel-1/2 Image Pairs Dataset](https://www.kaggle.com/datasets/requiemonk/sentinel12-image-pairs-segregated-by-terrain)

## Citation

If you use this code, please cite:

```bibtex
@misc{sar2optical2024,
  author = {yuuIind},
  title = {SAR2Optical: SAR to Optical Image Translation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yuuIind/SAR2Optical}
}
```

If you use the QXS-SAROPT dataset, please also cite:

```bibtex
@article{xu2021qxs,
  title={QXS-SAROPT: A Benchmark Dataset for Multi-modal SAR-Optical Image Matching},
  author={Xu, Yao and others},
  journal={arXiv preprint arXiv:2103.08259},
  year={2021}
}
```
