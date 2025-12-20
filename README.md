# SAR2Optical: SAR to Optical Image Translation

A deep learning project for translating **Synthetic Aperture Radar (SAR)** images into **optical RGB images** using the **Pix2Pix GAN** architecture.

---

## Overview

This project implements and fine-tunes a **Pix2Pix** model for **SAR-to-Optical image translation**.
The base model was pre-trained on **Sentinel-1/2 satellite imagery** and adapted to the **QXSLAB-SAROPT** dataset using multiple fine-tuning strategies, including **full fine-tuning** and **LoRA-based adaptation**.

---

## Project Structure

```text
SAR2Optical/
├── src/
│   ├── layers.py              # Network building blocks
│   ├── networks.py            # Generator and Discriminator architectures
│   ├── pix2pix.py             # Pix2Pix GAN wrapper
│   ├── dataset.py             # Dataset loading utilities
│   ├── loss.py                # Loss functions
│   └── metric.py              # Evaluation metrics
├── utils/
│   ├── config.py              # Configuration loader
│   ├── data_downloader.py     # Dataset download utilities
│   └── utils.py               # General utilities
├── notebooks/
│   ├── sar2optical_train.ipynb # Training notebook
│   └── sar2optical_eval.ipynb  # Evaluation notebook
├── checkpoints/
│   └── pix2pix_gen_180.pth    # Pre-trained generator weights
├── samples/                   # Sample SAR images
├── inputs/                    # Input images for inference
├── output/                    # Generated optical images
├── data/
│   ├── plots/                 # Training loss curves
│   └── split.json             # Train/val/test split
├── train.py                   # Training script
├── inference.py               # Single-image inference
├── test.py                    # Model testing
├── torch2onnx.py              # ONNX export utility
├── onnx_inference.py          # ONNX runtime inference
├── preprocess_qxslab.py       # SAR preprocessing
├── app.py                     # Streamlit web application
├── config.yaml                # Configuration file
├── Dockerfile                 # Docker definition
├── docker-compose.yml         # Docker Compose config
├── finetune_qxslab_colab_v2.ipynb # Full fine-tuning (Colab)
└── finetune_qxslab_lora.ipynb     # LoRA fine-tuning (Colab)
```

---

## Model Architecture

### Generator (U-Net)

* **Encoder**: 8 downsampling blocks
  *(Conv2d → BatchNorm → LeakyReLU)*
* **Decoder**: 8 upsampling blocks
  *(ConvTranspose2d → BatchNorm → Dropout → ReLU)*
* **Skip Connections**: Encoder–decoder feature concatenation
* **Output**: 256×256 RGB image via **Tanh**
* **Parameters**: ~54.5M

### Discriminator (PatchGAN)

* 70×70 patch-based classifier
* 3-layer convolutional network
* **Parameters**: ~2.8M

---

## Loss Functions

* **Adversarial Loss**: Binary Cross-Entropy (BCE)
* **Reconstruction Loss**: L1 (Mean Absolute Error)
* **Total Loss**:

[
L_G = L_{adv} + \lambda \times L_{L1}, \quad \lambda = 100
]

---

## Installation

### Requirements

```bash
pip install torch torchvision
pip install pillow numpy scipy opencv-python
pip install streamlit
pip install tqdm matplotlib
```

### Clone and Setup

```bash
git clone https://github.com/yuuIind/SAR2Optical.git
cd SAR2Optical
pip install -r requirements.txt
```

---

## Usage

### 1. Inference (Single Image)

```bash
python inference.py
```

**Configure `config.yaml`:**

```yaml
inference:
  image_path: "path/to/sar_image.png"
  output_path: "./output/result.jpg"
  gen_checkpoint: "checkpoints/pix2pix_gen_180.pth"
  device: "cuda"  # or "cpu"
```

---

### 2. Training

```bash
python train.py
```

**Training configuration (`config.yaml`):**

```yaml
training:
  num_epochs: 200
  batch_size: 32
  lr: 0.0002
  device: "cuda"
```

---

### 3. Streamlit Web App

```bash
streamlit run app.py
```

Access at:
👉 [http://localhost:8501](http://localhost:8501)

---

### 4. Docker Deployment

```bash
docker-compose up --build
```

Access at:
👉 [http://localhost:8501](http://localhost:8501)

---

## Preprocessing for Domain Adaptation

For SAR datasets with different distributions (e.g., **QXSLAB-SAROPT**), preprocessing is recommended.

```bash
python preprocess_qxslab.py \
    --input path/to/sar_image.png \
    --output path/to/preprocessed.png \
    --filter gaussian \
    --window-size 7 \
    --gamma 0.1
```

### Available Filters

* `lee` – Lee speckle filter *(default)*
* `frost` – Frost speckle filter
* `median` – Median filter
* `gaussian` – Gaussian blur
* `bilateral` – Bilateral filter

---

## Fine-tuning

Two approaches are provided for domain adaptation:

### 1. Full Fine-tuning

Notebook: `finetune_qxslab_colab_v2.ipynb`

* Trains all ~54.5M parameters
* Requires careful learning rate tuning
* Risk of catastrophic forgetting

### 2. LoRA Fine-tuning (Recommended)

Notebook: `finetune_qxslab_lora.ipynb`

* Freezes pre-trained weights
* Trains ~1.5% parameters using LoRA adapters
* Faster training
* Reduced overfitting
* Preserves learned representations

### Fine-tuning Configuration

```python
CONFIG = {
    "num_epochs": 30,
    "batch_size": 32,
    "lr": 0.0002,        # LoRA
    # "lr": 0.00001,     # Full fine-tuning
    "lambda_L1": 100.0,
}

LORA_CONFIG = {
    "rank": 16,
    "alpha": 32,
    "dropout": 0.1,
}
```

---

## ONNX Export

Export the generator for deployment:

```bash
python torch2onnx.py
```

**Configuration (`config.yaml`):**

```yaml
export:
  gen_checkpoint: "pix2pix_gen_180.pth"
  export_path: "pix2pix_gen_sar2rgb.onnx"
  opset_version: 17
```

---

## Datasets

### Sentinel-1/2 (Pre-training)

* Co-registered SAR (Sentinel-1) and Optical (Sentinel-2) imagery
* Used for initial Pix2Pix training

### QXSLAB-SAROPT (Fine-tuning)

* 20,000 SAR–Optical image pairs
* Resolution: 256×256

```text
QXSLAB_SAROPT/
├── sar_256_oc_0.2/    # SAR images
└── opt_256_oc_0.2/    # Optical images
```

---

## Results

### Sentinel-1/2 Domain

* High-quality color reconstruction
* Sharp structural consistency

### QXSLAB-SAROPT (No Fine-tuning)

* Severe domain mismatch
* Artifacts:

  * Crystalline/blocky patterns
  * Excessive brightness
  * Structural distortion

### After Preprocessing

* Minor improvements
* Still insufficient for practical use

### After Fine-tuning

* Best performance achieved
* Significant artifact reduction
* Improved realism and structure

---

## Web Application Features

The **Streamlit app (`app.py`)** provides:

* Upload custom SAR images
* Sample image selection
* Real-time inference
* Side-by-side comparison
* Image download
* Optional LoRA weight loading
* GPU / CPU device information

---

## Citation

If you use this project, please cite:

```bibtex
@misc{sar2optical,
  author = {yuuIind},
  title = {SAR2Optical: SAR to Optical Image Translation using Pix2Pix},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yuuIind/SAR2Optical}
}
```

---

## References

* Pix2Pix — *Image-to-Image Translation with Conditional Adversarial Networks*
* LoRA — *Low-Rank Adaptation of Large Language Models*
* Sentinel-1 — ESA SAR Mission
* Sentinel-2 — ESA Optical Mission

---

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.


Just tell me 👍
