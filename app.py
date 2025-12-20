"""
SAR2Optical Streamlit Application
Converts SAR (Synthetic Aperture Radar) images to Optical images using a trained pix2pix model.
"""

import streamlit as st
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import v2
import io
import os
from pathlib import Path

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="SAR to Optical Image Converter",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL DEFINITION (Lightweight - only Generator needed for inference)
# ============================================================================

class DownsamplingBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2,
                 padding=1, negative_slope=0.2, use_norm=True):
        super().__init__()
        block = [torch.nn.Conv2d(c_in, c_out, kernel_size, stride, padding, bias=(not use_norm))]
        if use_norm:
            block += [torch.nn.BatchNorm2d(c_out)]
        block += [torch.nn.LeakyReLU(negative_slope)]
        self.conv_block = torch.nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)


class UpsamplingBlock(torch.nn.Module):
    def __init__(self, c_in, c_out, kernel_size=4, stride=2,
                 padding=1, use_dropout=False, use_upsampling=False, mode='nearest'):
        super().__init__()
        block = []
        if use_upsampling:
            block += [torch.nn.Sequential(
                torch.nn.Upsample(scale_factor=2, mode=mode),
                torch.nn.Conv2d(c_in, c_out, 3, 1, padding, bias=False))]
        else:
            block += [torch.nn.ConvTranspose2d(c_in, c_out, kernel_size, stride, padding, bias=False)]
        block += [torch.nn.BatchNorm2d(c_out)]
        if use_dropout:
            block += [torch.nn.Dropout(0.5)]
        block += [torch.nn.ReLU()]
        self.conv_block = torch.nn.Sequential(*block)

    def forward(self, x):
        return self.conv_block(x)


class UnetEncoder(torch.nn.Module):
    def __init__(self, c_in=3):
        super().__init__()
        self.enc1 = DownsamplingBlock(c_in, 64, use_norm=False)
        self.enc2 = DownsamplingBlock(64, 128)
        self.enc3 = DownsamplingBlock(128, 256)
        self.enc4 = DownsamplingBlock(256, 512)
        self.enc5 = DownsamplingBlock(512, 512)
        self.enc6 = DownsamplingBlock(512, 512)
        self.enc7 = DownsamplingBlock(512, 512)
        self.enc8 = DownsamplingBlock(512, 512)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
        x5 = self.enc5(x4)
        x6 = self.enc6(x5)
        x7 = self.enc7(x6)
        x8 = self.enc8(x7)
        return [x8, x7, x6, x5, x4, x3, x2, x1]


class UnetDecoder(torch.nn.Module):
    def __init__(self, use_upsampling=False, mode='nearest'):
        super().__init__()
        self.dec1 = UpsamplingBlock(512, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec2 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec3 = UpsamplingBlock(1024, 512, use_dropout=True, use_upsampling=use_upsampling, mode=mode)
        self.dec4 = UpsamplingBlock(1024, 512, use_upsampling=use_upsampling, mode=mode)
        self.dec5 = UpsamplingBlock(1024, 256, use_upsampling=use_upsampling, mode=mode)
        self.dec6 = UpsamplingBlock(512, 128, use_upsampling=use_upsampling, mode=mode)
        self.dec7 = UpsamplingBlock(256, 64, use_upsampling=use_upsampling, mode=mode)
        self.dec8 = UpsamplingBlock(128, 64, use_upsampling=use_upsampling, mode=mode)

    def forward(self, x):
        x9 = torch.cat([x[1], self.dec1(x[0])], 1)
        x10 = torch.cat([x[2], self.dec2(x9)], 1)
        x11 = torch.cat([x[3], self.dec3(x10)], 1)
        x12 = torch.cat([x[4], self.dec4(x11)], 1)
        x13 = torch.cat([x[5], self.dec5(x12)], 1)
        x14 = torch.cat([x[6], self.dec6(x13)], 1)
        x15 = torch.cat([x[7], self.dec7(x14)], 1)
        return self.dec8(x15)


class UnetGenerator(torch.nn.Module):
    def __init__(self, c_in=3, c_out=3, use_upsampling=False, mode='nearest'):
        super().__init__()
        self.encoder = UnetEncoder(c_in=c_in)
        self.decoder = UnetDecoder(use_upsampling=use_upsampling, mode=mode)
        self.head = torch.nn.Sequential(
            torch.nn.Conv2d(64, c_out, 3, 1, 1, bias=True),
            torch.nn.Tanh())

    def forward(self, x):
        return self.head(self.decoder(self.encoder(x)))


# ============================================================================
# LoRA LAYER FOR INFERENCE
# ============================================================================

class LoRAConv2d(torch.nn.Module):
    """LoRA adapter for Conv2d layers (inference only)."""
    def __init__(self, conv: torch.nn.Conv2d, rank: int = 16, alpha: float = 32):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.scaling = alpha / rank

        in_channels = conv.in_channels
        out_channels = conv.out_channels

        self.lora_A = torch.nn.Conv2d(in_channels, rank, kernel_size=1, bias=False)
        self.lora_B = torch.nn.Conv2d(rank, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        if lora_out.shape[2:] != out.shape[2:]:
            lora_out = torch.nn.functional.interpolate(lora_out, size=out.shape[2:], mode='bilinear', align_corners=False)
        return out + lora_out


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_model(model_path: str, lora_path: str = None):
    """Load the generator model with optional LoRA weights (cached for performance)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = UnetGenerator(c_in=3, c_out=3, use_upsampling=False)

    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()

        # Load LoRA weights if provided
        if lora_path and os.path.exists(lora_path):
            model = _apply_lora_weights(model, lora_path, device)

        return model, device, lora_path is not None and os.path.exists(lora_path)
    else:
        return None, device, False


def _apply_lora_weights(model, lora_path, device):
    """Apply LoRA weights to the model."""
    lora_state = torch.load(lora_path, map_location=device, weights_only=False)
    config = lora_state.get('config', {'rank': 16, 'alpha': 32})

    # Wrap encoder conv layers with LoRA
    for name in ['enc1', 'enc2', 'enc3', 'enc4', 'enc5', 'enc6', 'enc7', 'enc8']:
        lora_key = f"encoder_{name}"
        if f"{lora_key}.lora_A" in lora_state:
            block = getattr(model.encoder, name)
            if hasattr(block, 'conv_block'):
                # Get the conv layer from the sequential block
                conv = block.conv_block[0]
                lora_conv = LoRAConv2d(conv, rank=config['rank'], alpha=config['alpha'])
                lora_conv.lora_A.load_state_dict(lora_state[f"{lora_key}.lora_A"])
                lora_conv.lora_B.load_state_dict(lora_state[f"{lora_key}.lora_B"])
                block.conv_block[0] = lora_conv

    model.to(device)
    model.eval()
    return model


def predict(model, image: Image.Image, device) -> Image.Image:
    """Run inference on a single image"""

    # Define transforms
    transform = v2.Compose([
        v2.ToImage(),
        v2.Resize((256, 256)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5], std=[0.5]),
    ])

    # Preprocess
    img_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(img_tensor)

    # Post-process
    output = (output.squeeze(0) * 0.5 + 0.5).clamp(0, 1)
    output = (output.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

    return Image.fromarray(output)


# ============================================================================
# SAMPLE IMAGES
# ============================================================================

def get_sample_images():
    """Return paths to sample SAR images"""
    samples_dir = Path(__file__).parent / "samples"

    # Create samples directory if it doesn't exist
    samples_dir.mkdir(exist_ok=True)

    # Look for sample images
    sample_files = list(samples_dir.glob("*.png")) + list(samples_dir.glob("*.jpg"))

    return sample_files


def create_placeholder_samples():
    """Info about setting up sample images"""
    return """
    To add sample SAR images:
    1. Create a folder called `samples` in the app directory
    2. Add SAR images (PNG or JPG) to the folder
    3. Restart the app
    """


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Header
    st.title("🛰️ SAR to Optical Image Converter")
    st.markdown("""
    Convert **Synthetic Aperture Radar (SAR)** images to **Optical** images using deep learning.

    This app uses a Pix2Pix model fine-tuned on SAR-Optical image pairs.
    """)

    st.divider()

    # Sidebar - Model Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Model path input
        default_model_path = "checkpoints/pix2pix_gen_180.pth"
        model_path = st.text_input(
            "Base Model Checkpoint Path",
            value=default_model_path,
            help="Path to the pretrained generator checkpoint (.pth file)"
        )

        # LoRA weights path (optional)
        lora_path = st.text_input(
            "LoRA Weights Path (Optional)",
            value="",
            help="Path to LoRA adapter weights (.pth file). Leave empty for base model only."
        )

        # Load model
        lora_path_to_use = lora_path if lora_path.strip() else None
        model, device, lora_loaded = load_model(model_path, lora_path_to_use)

        if model is not None:
            if lora_loaded:
                st.success(f"✅ Model + LoRA loaded on {device}")
            else:
                st.success(f"✅ Model loaded on {device}")
                if lora_path.strip():
                    st.warning(f"⚠️ LoRA weights not found at: {lora_path}")
        else:
            st.error(f"❌ Model not found at: {model_path}")
            st.info("Please provide a valid model path")

        st.divider()

        # Device info
        st.subheader("📊 System Info")
        st.write(f"**Device:** {device}")
        st.write(f"**PyTorch:** {torch.__version__}")
        if torch.cuda.is_available():
            st.write(f"**GPU:** {torch.cuda.get_device_name(0)}")

    # Initialize session state for selected image
    if 'selected_image' not in st.session_state:
        st.session_state.selected_image = None
    if 'output_image' not in st.session_state:
        st.session_state.output_image = None

    # Main content area (single column for input section)
    col1, _ = st.columns(2)

    with col1:
        st.subheader("📤 Input SAR Image")

        # Upload your own image
        uploaded_file = st.file_uploader(
            "Upload a SAR image",
            type=["png", "jpg", "jpeg", "tif", "tiff"],
            help="Upload a SAR image to convert to optical"
        )

        if uploaded_file is not None:
            st.session_state.selected_image = Image.open(uploaded_file).convert('RGB')
            st.session_state.output_image = None  # Reset output when new image selected

        st.markdown("**Or select from sample images:**")

        # Get sample images
        sample_files = get_sample_images()

        if sample_files:
            # Create a grid of sample images (clickable)
            num_cols = min(len(sample_files), 5)  # Max 5 columns
            cols = st.columns(num_cols)

            for idx, sample_path in enumerate(sample_files):
                with cols[idx % num_cols]:
                    # Load and display thumbnail
                    sample_img = Image.open(sample_path).convert('RGB')

                    # Create a button with image
                    st.image(sample_img, use_container_width=True)
                    if st.button(f"Select", key=f"sample_{idx}", use_container_width=True):
                        st.session_state.selected_image = sample_img
                        st.session_state.output_image = None  # Reset output
                        st.rerun()
        else:
            st.warning("No sample images found!")
            st.markdown("""
            **To add sample images:**
            1. Create a `samples` folder in the app directory
            2. Add SAR images (PNG/JPG) to the folder
            3. Restart the app
            """)

    # Get selected image
    input_image = st.session_state.selected_image

    # Results section (full width, below the upload/sample area)
    st.divider()

    if input_image is not None and model is not None:
        # Generate button (centered)
        _, col_btn, _ = st.columns([1, 2, 1])
        with col_btn:
            if st.button("🔄 Generate Optical Image", type="primary", use_container_width=True):
                with st.spinner("Generating optical image..."):
                    st.session_state.output_image = predict(model, input_image, device)

        st.markdown("")  # Spacing

        # Side by side: Selected SAR | Predicted Optical
        img_col1, img_col2 = st.columns(2)

        with img_col1:
            st.markdown("**Selected SAR Image:**")
            st.image(input_image,width=500)
            st.caption(f"Size: {input_image.size[0]} x {input_image.size[1]}")

        with img_col2:
            st.markdown("**Predicted Optical Image:**")
            if st.session_state.output_image is not None:
                st.image(st.session_state.output_image, width=500)

                # Download button
                buf = io.BytesIO()
                st.session_state.output_image.save(buf, format="PNG")
                buf.seek(0)

                st.download_button(
                    label="📥 Download Result",
                    data=buf,
                    file_name="optical_output.png",
                    mime="image/png",
                    use_container_width=True
                )
            else:
                st.info("👆 Click 'Generate' to see result")

    elif model is None:
        st.warning("⚠️ Please load a valid model first (check sidebar)")
    else:
        st.info("⬆️ Upload or select a SAR image to get started")

    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>Built with Streamlit | Model: Pix2Pix (SAR2Optical)</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()