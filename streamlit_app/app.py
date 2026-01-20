"""
VAE/CVAE Interactive Playground
A beautiful Streamlit interface for exploring trained VAE and CVAE models on MNIST.

Features:
- Conditional digit generation with style control
- Style transfer: upload your own digit and transfer its style to any class
- Latent space exploration with interactive sliders
- Digit morphing/interpolation between two digits
- Random sampling with controllable diversity
- Reconstruction quality test
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io
import os

# Page configuration
st.set_page_config(
    page_title="VAE Playground",
    page_icon="V",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .digit-display {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== MODEL DEFINITIONS ====================

class ConvEncoder(nn.Module):
    """Convolutional encoder for VAE (matches notebook architecture)"""
    def __init__(self, latent_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var


class ConvDecoder(nn.Module):
    """Convolutional decoder for VAE (matches notebook architecture)"""
    def __init__(self, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(nn.Module):
    """Variational Autoencoder combining encoder and decoder"""
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def encode(self, x):
        return self.encoder(x)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var, z


class ConvEncoderCVAE(nn.Module):
    """Conditional encoder: maps (x, c) -> (mu, log_var)"""
    def __init__(self, latent_dim, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        # Match checkpoint: embed label to 784 and reshape to 28x28 (single channel)
        self.label_embedding = nn.Embedding(num_classes, 784)
        self.conv1 = nn.Conv2d(1 + 1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc_mu = nn.Linear(128 * 4 * 4, latent_dim)
        self.fc_log_var = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x, c):
        c_embed = self.label_embedding(c).view(-1, 1, 28, 28)
        c_map = c_embed
        x_cond = torch.cat([x, c_map], dim=1)
        x_cond = F.relu(self.bn1(self.conv1(x_cond)))
        x_cond = F.relu(self.bn2(self.conv2(x_cond)))
        x_cond = F.relu(self.bn3(self.conv3(x_cond)))
        x_cond = x_cond.view(x_cond.size(0), -1)
        mu = self.fc_mu(x_cond)
        log_var = self.fc_log_var(x_cond)
        return mu, log_var


class ConvDecoderCVAE(nn.Module):
    """Conditional decoder: maps (z, c) -> x_reconstructed"""
    def __init__(self, latent_dim, num_classes=10):
        super().__init__()
        self.num_classes = num_classes
        self.label_embedding = nn.Embedding(num_classes, num_classes)
        self.fc = nn.Linear(latent_dim + num_classes, 128 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(32)

    def forward(self, z, c):
        c_embed = self.label_embedding(c)
        z_cond = torch.cat([z, c_embed], dim=1)
        x = F.relu(self.fc(z_cond))
        x = x.view(-1, 128, 4, 4)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))
        return x


class CVAE(nn.Module):
    """Conditional Variational Autoencoder"""
    def __init__(self, latent_dim=2, num_classes=10):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.encoder = ConvEncoderCVAE(latent_dim, num_classes)
        self.decoder = ConvDecoderCVAE(latent_dim, num_classes)

    def encode(self, x, c):
        return self.encoder(x, c)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + std * epsilon

    def decode(self, z, c):
        return self.decoder(z, c)

    def forward(self, x, c):
        mu, log_var = self.encode(x, c)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z, c)
        return x_recon, mu, log_var, z


# ==================== UTILITY FUNCTIONS ====================

@st.cache_resource
def load_models():
    """Load pre-trained VAE and CVAE models"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    models = {}
    
    # Try to load VAE
    vae_path = 'models/vae_mnist.pth'
    if os.path.exists(vae_path):
        checkpoint = torch.load(vae_path, map_location=device)
        latent_dim = checkpoint.get('latent_dim', 2)
        vae = VAE(latent_dim).to(device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        vae.eval()
        models['vae'] = vae
        models['vae_latent_dim'] = latent_dim
    
    # Try to load CVAE
    cvae_path = 'models/cvae_mnist.pth'
    if os.path.exists(cvae_path):
        checkpoint = torch.load(cvae_path, map_location=device)
        latent_dim = checkpoint.get('latent_dim', 2)
        cvae = CVAE(latent_dim, num_classes=10).to(device)
        cvae.load_state_dict(checkpoint['model_state_dict'])
        cvae.eval()
        models['cvae'] = cvae
        models['cvae_latent_dim'] = latent_dim
    
    models['device'] = device
    return models


def tensor_to_image(tensor):
    """Convert a tensor to PIL Image"""
    img = tensor.squeeze().cpu().numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img, mode='L')


def preprocess_uploaded_image(uploaded_file):
    """Preprocess an uploaded image for the model"""
    img = Image.open(uploaded_file).convert('L')
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    # Invert if background is white (MNIST has black background)
    if img_array.mean() > 0.5:
        img_array = 1.0 - img_array
    
    tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
    return tensor, img


def create_digit_grid(images, cols=10, title=""):
    """Create a grid of digit images"""
    n = len(images)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(images[i], cmap='gray')
        ax.axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


# ==================== MAIN APP ====================

def main():
    # Header
    st.markdown('<h1 class="main-header">VAE Playground</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore Variational Autoencoders interactively</p>', unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    device = models['device']
    
    # Check if models are loaded
    if 'cvae' not in models and 'vae' not in models:
        st.error("No models found! Please place `vae_mnist.pth` and/or `cvae_mnist.pth` in the `models/` directory.")
        st.info("You can train and download models from the Jupyter notebook.")
        
        # File uploader as fallback
        st.subheader("Upload Models")
        col1, col2 = st.columns(2)
        
        with col1:
            vae_file = st.file_uploader("Upload VAE model (.pth)", type=['pth'], key='vae_upload')
            if vae_file:
                os.makedirs('models', exist_ok=True)
                with open('models/vae_mnist.pth', 'wb') as f:
                    f.write(vae_file.read())
                st.success("VAE model uploaded! Please refresh the page.")
        
        with col2:
            cvae_file = st.file_uploader("Upload CVAE model (.pth)", type=['pth'], key='cvae_upload')
            if cvae_file:
                os.makedirs('models', exist_ok=True)
                with open('models/cvae_mnist.pth', 'wb') as f:
                    f.write(cvae_file.read())
                st.success("CVAE model uploaded! Please refresh the page.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Features")
    
    feature = st.sidebar.radio(
        "Select a feature:",
        [
            "Conditional Generation",
            "Style Transfer",
            "Latent Space Explorer",
            "Digit Morphing",
            "Random Sampling",
            "Reconstruction Test"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"Running on: **{device}**")
    
    if 'cvae' in models:
        st.sidebar.success("CVAE loaded")
    else:
        st.sidebar.warning("CVAE not loaded")
    
    if 'vae' in models:
        st.sidebar.success("VAE loaded")
    else:
        st.sidebar.warning("VAE not loaded")
    
    # Main content based on selected feature
    if feature == "Conditional Generation":
        conditional_generation(models, device)
    elif feature == "Style Transfer":
        style_transfer(models, device)
    elif feature == "Latent Space Explorer":
        latent_explorer(models, device)
    elif feature == "Digit Morphing":
        digit_morphing(models, device)
    elif feature == "Random Sampling":
        random_sampling(models, device)
    elif feature == "Reconstruction Test":
        reconstruction_test(models, device)


def conditional_generation(models, device):
    """Generate specific digits with controllable style"""
    st.header("Conditional Digit Generation")
    st.markdown("Generate any digit (0-9) with full control over its style using the CVAE.")
    
    if 'cvae' not in models:
        st.error("CVAE model required for this feature. Please load the model first.")
        return
    
    cvae = models['cvae']
    latent_dim = models['cvae_latent_dim']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Controls")
        
        # Digit selection
        digit = st.selectbox("Select digit to generate:", list(range(10)), index=3)
        
        # Number of samples
        n_samples = st.slider("Number of samples:", 1, 20, 10)
        
        # Style control
        st.markdown("**Style Controls:**")
        
        if latent_dim == 2:
            z1 = st.slider("Z₁ (slant/rotation):", -3.0, 3.0, 0.0, 0.1)
            z2 = st.slider("Z₂ (thickness/scale):", -3.0, 3.0, 0.0, 0.1)
            use_fixed_z = st.checkbox("Use fixed style (otherwise random)", value=False)
        else:
            use_fixed_z = False
            st.info(f"Random sampling from {latent_dim}D latent space")
        
        # Diversity control
        diversity = st.slider("Style diversity:", 0.0, 2.0, 1.0, 0.1)
        
        generate_btn = st.button("Generate!", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Generated Digits")
        
        if generate_btn:
            with torch.no_grad():
                if use_fixed_z and latent_dim == 2:
                    z = torch.tensor([[z1, z2]], dtype=torch.float32).repeat(n_samples, 1).to(device)
                else:
                    z = torch.randn(n_samples, latent_dim).to(device) * diversity
                
                labels = torch.full((n_samples,), digit, dtype=torch.long).to(device)
                generated = cvae.decode(z, labels)
            
            images = [tensor_to_image(generated[i]) for i in range(n_samples)]
            
            # Display in grid
            cols = min(5, n_samples)
            rows = (n_samples + cols - 1) // cols
            
            for r in range(rows):
                row_cols = st.columns(cols)
                for c in range(cols):
                    idx = r * cols + c
                    if idx < n_samples:
                        with row_cols[c]:
                            st.image(images[idx], width=100, caption=f"#{idx+1}")


def style_transfer(models, device):
    """Transfer style from uploaded image to any digit class"""
    st.header("Style Transfer")
    st.markdown("""
    Upload your own handwritten digit and transfer its **style** to any other digit!
    
    This demonstrates that the CVAE has learned to separate *what digit* (class) from *how it looks* (style).
    """)
    
    if 'cvae' not in models:
        st.error("CVAE model required for this feature. Please load the model first.")
        return
    
    cvae = models['cvae']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Your Digit")
        
        uploaded_file = st.file_uploader(
            "Upload a digit image (any format)",
            type=['png', 'jpg', 'jpeg', 'bmp', 'gif'],
            help="Best results with clear handwritten digits on plain background"
        )
        
        if uploaded_file:
            tensor, preview = preprocess_uploaded_image(uploaded_file)
            
            st.image(preview, caption="Preprocessed (28×28)", width=150)
            
            # Source label (optional, for better encoding)
            source_label = st.number_input(
                "What digit is this? (helps encoding)",
                min_value=0, max_value=9, value=0
            )
            
            st.markdown("---")
            st.subheader("Transfer to:")
            
            # Target digits selection
            target_all = st.checkbox("All digits (0-9)", value=True)
            
            if not target_all:
                target_digits = st.multiselect(
                    "Select target digits:",
                    list(range(10)),
                    default=[0, 1, 2, 3]
                )
            else:
                target_digits = list(range(10))
            
            transfer_btn = st.button("Transfer Style!", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Style Transfer Results")
        
        if uploaded_file and transfer_btn:
            tensor = tensor.to(device)
            source = torch.tensor([source_label], dtype=torch.long).to(device)
            
            with torch.no_grad():
                # Encode the uploaded image
                mu, log_var = cvae.encode(tensor, source)
                z = mu  # Use mean for deterministic style
                
                # Generate with same z but different labels
                results = []
                for digit in target_digits:
                    label = torch.tensor([digit], dtype=torch.long).to(device)
                    generated = cvae.decode(z, label)
                    results.append(tensor_to_image(generated[0]))
            
            # Display results
            st.markdown("**Same style, different digits:**")
            
            cols = st.columns(min(5, len(results)))
            for i, (digit, img) in enumerate(zip(target_digits, results)):
                with cols[i % 5]:
                    st.image(img, caption=f"Digit {digit}", width=100)
                
                # New row if needed
                if (i + 1) % 5 == 0 and i + 1 < len(results):
                    cols = st.columns(min(5, len(results) - i - 1))
            
            st.success("Style successfully transferred across all target digits!")


def latent_explorer(models, device):
    """Explore the latent space interactively"""
    st.header("Latent Space Explorer")
    st.markdown("Move through the 2D latent space and see how generated digits change in real-time.")
    
    if 'cvae' not in models:
        st.error("CVAE model required for this feature.")
        return
    
    cvae = models['cvae']
    latent_dim = models['cvae_latent_dim']
    
    if latent_dim != 2:
        st.warning(f"This visualization works best with 2D latent space. Current: {latent_dim}D")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Latent Coordinates")
        
        z1 = st.slider("Z₁ (horizontal axis)", -4.0, 4.0, 0.0, 0.1, key="explorer_z1")
        z2 = st.slider("Z₂ (vertical axis)", -4.0, 4.0, 0.0, 0.1, key="explorer_z2")
        
        st.markdown("---")
        st.subheader("Display Options")
        
        show_all_digits = st.checkbox("Show all digits at this point", value=True)
        
        if not show_all_digits:
            selected_digit = st.selectbox("Select digit:", list(range(10)))
    
    with col2:
        st.subheader("Generated Output")
        
        z = torch.tensor([[z1, z2]], dtype=torch.float32).to(device)
        
        with torch.no_grad():
            if show_all_digits:
                images = []
                for digit in range(10):
                    label = torch.tensor([digit], dtype=torch.long).to(device)
                    generated = cvae.decode(z, label)
                    images.append(tensor_to_image(generated[0]))
                
                # Display in a row
                cols = st.columns(10)
                for i, (digit, img) in enumerate(zip(range(10), images)):
                    with cols[i]:
                        st.image(img, caption=str(digit), width=70)
            else:
                label = torch.tensor([selected_digit], dtype=torch.long).to(device)
                generated = cvae.decode(z, label)
                img = tensor_to_image(generated[0])
                st.image(img, width=200)
        
        # Latent space reference
        st.markdown("---")
        st.markdown("**Latent Space Map:**")
        st.markdown(f"Current position: **(Z₁={z1:.2f}, Z₂={z2:.2f})**")
        
        # Mini reference grid
        if st.checkbox("Show reference grid", value=False):
            with torch.no_grad():
                n = 7
                z_range = torch.linspace(-3, 3, n)
                grid_images = []
                
                for z2_val in reversed(z_range):
                    row = []
                    for z1_val in z_range:
                        z_grid = torch.tensor([[z1_val, z2_val]], dtype=torch.float32).to(device)
                        label = torch.tensor([3], dtype=torch.long).to(device)  # Use digit 3 as reference
                        generated = cvae.decode(z_grid, label)
                        row.append(tensor_to_image(generated[0]))
                    grid_images.append(row)
            
            fig = create_digit_grid([img for row in grid_images for img in row], cols=n, title="Latent Traversal (digit 3)")
            st.pyplot(fig)
            plt.close()


def digit_morphing(models, device):
    """Interpolate between two digits"""
    st.header("Digit Morphing")
    st.markdown("Watch one digit smoothly transform into another through latent space interpolation.")
    
    if 'cvae' not in models:
        st.error("CVAE model required for this feature.")
        return
    
    cvae = models['cvae']
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Morphing Settings")
        
        # Source and target
        source_digit = st.selectbox("Source digit:", list(range(10)), index=3, key="morph_source")
        target_digit = st.selectbox("Target digit:", list(range(10)), index=8, key="morph_target")
        
        # Number of interpolation steps
        n_steps = st.slider("Interpolation steps:", 5, 20, 10)
        
        # Interpolation type
        interp_type = st.radio(
            "Interpolation type:",
            ["Both class and style", "Style only (same digit)", "Class only (same style)"]
        )
        
        morph_btn = st.button("Morph!", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Morphing Sequence")
        
        if morph_btn:
            with torch.no_grad():
                # Get random styles for source and target
                z_source = torch.randn(1, models['cvae_latent_dim']).to(device)
                z_target = torch.randn(1, models['cvae_latent_dim']).to(device)
                
                images = []
                alphas = np.linspace(0, 1, n_steps)
                
                for alpha in alphas:
                    if interp_type == "Both class and style":
                        # Interpolate both z and one-hot class embedding
                        z_interp = (1 - alpha) * z_source + alpha * z_target
                        # For class, we need to do soft interpolation then generate
                        # Use source for first half, target for second half
                        current_digit = source_digit if alpha < 0.5 else target_digit
                    elif interp_type == "Style only (same digit)":
                        z_interp = (1 - alpha) * z_source + alpha * z_target
                        current_digit = source_digit
                    else:  # Class only
                        z_interp = z_source  # Fixed style
                        current_digit = source_digit if alpha < 0.5 else target_digit
                    
                    label = torch.tensor([current_digit], dtype=torch.long).to(device)
                    generated = cvae.decode(z_interp, label)
                    images.append(tensor_to_image(generated[0]))
            
            # Display sequence
            cols_per_row = min(n_steps, 10)
            for row_start in range(0, n_steps, cols_per_row):
                cols = st.columns(cols_per_row)
                for i, col in enumerate(cols):
                    idx = row_start + i
                    if idx < n_steps:
                        with col:
                            st.image(images[idx], width=70)
            
            st.success(f"Morphed from {source_digit} -> {target_digit} in {n_steps} steps!")


def random_sampling(models, device):
    """Generate random samples with controllable diversity"""
    st.header("Random Sampling")
    st.markdown("Generate diverse random digits by sampling from the latent space.")
    
    model_choice = st.radio("Select model:", ["CVAE (class-conditioned)", "VAE (unconditioned)"])
    
    if model_choice == "CVAE (class-conditioned)" and 'cvae' not in models:
        st.error("CVAE model not loaded.")
        return
    if model_choice == "VAE (unconditioned)" and 'vae' not in models:
        st.error("VAE model not loaded.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Sampling Parameters")
        
        n_samples = st.slider("Number of samples:", 10, 100, 30, 10)
        
        # Temperature / diversity control
        temperature = st.slider(
            "Temperature (diversity):",
            0.1, 2.0, 1.0, 0.1,
            help="Higher = more diverse but potentially noisier"
        )
        
        if model_choice == "CVAE (class-conditioned)":
            balanced = st.checkbox("Balanced classes (equal samples per digit)", value=True)
            if not balanced:
                specific_digit = st.selectbox("Generate only digit:", list(range(10)))
        
        sample_btn = st.button("Sample!", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Random Samples")
        
        if sample_btn:
            with torch.no_grad():
                if model_choice == "CVAE (class-conditioned)":
                    cvae = models['cvae']
                    latent_dim = models['cvae_latent_dim']
                    
                    z = torch.randn(n_samples, latent_dim).to(device) * temperature
                    
                    if balanced:
                        labels = torch.tensor([i % 10 for i in range(n_samples)], dtype=torch.long).to(device)
                    else:
                        labels = torch.full((n_samples,), specific_digit, dtype=torch.long).to(device)
                    
                    generated = cvae.decode(z, labels)
                else:
                    vae = models['vae']
                    latent_dim = models['vae_latent_dim']
                    
                    z = torch.randn(n_samples, latent_dim).to(device) * temperature
                    generated = vae.decode(z)
                    labels = None
                
                images = [tensor_to_image(generated[i]) for i in range(n_samples)]
            
            # Display in grid
            cols = 10
            for row_start in range(0, n_samples, cols):
                row_cols = st.columns(cols)
                for i, col in enumerate(row_cols):
                    idx = row_start + i
                    if idx < n_samples:
                        with col:
                            caption = str(labels[idx].item()) if labels is not None else ""
                            st.image(images[idx], width=65, caption=caption)


def reconstruction_test(models, device):
    """Test reconstruction quality with uploaded images"""
    st.header("Reconstruction Test")
    st.markdown("""
    Upload a digit image and see how well the VAE/CVAE can reconstruct it.
    This tests the model's ability to generalize to digits outside MNIST!
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Upload Test Image")
        
        uploaded_file = st.file_uploader(
            "Upload a digit image",
            type=['png', 'jpg', 'jpeg', 'bmp'],
            key="recon_upload"
        )
        
        if uploaded_file:
            tensor, preview = preprocess_uploaded_image(uploaded_file)
            st.image(preview, caption="Input (28×28)", width=150)
            
            model_choice = st.radio(
                "Model to use:",
                ["CVAE", "VAE"] if 'cvae' in models and 'vae' in models else
                ["CVAE"] if 'cvae' in models else ["VAE"]
            )
            
            if model_choice == "CVAE":
                digit_label = st.number_input("Digit label:", 0, 9, 0)
            
            recon_btn = st.button("Reconstruct!", type="primary", use_container_width=True)
    
    with col2:
        st.subheader("Results")
        
        if uploaded_file and recon_btn:
            tensor = tensor.to(device)
            
            with torch.no_grad():
                if model_choice == "CVAE":
                    cvae = models['cvae']
                    label = torch.tensor([digit_label], dtype=torch.long).to(device)
                    recon, mu, log_var, z = cvae(tensor, label)
                else:
                    vae = models['vae']
                    recon, mu, log_var, z = vae(tensor)
                
                recon_img = tensor_to_image(recon[0])
            
            # Display comparison
            comp_col1, comp_col2, comp_col3 = st.columns(3)
            
            with comp_col1:
                st.markdown("**Original**")
                st.image(preview, width=150)
            
            with comp_col2:
                st.markdown("**Reconstructed**")
                st.image(recon_img, width=150)
            
            with comp_col3:
                st.markdown("**Latent Code**")
                z_np = z.cpu().numpy()[0]
                if len(z_np) == 2:
                    st.write(f"Z₁ = {z_np[0]:.3f}")
                    st.write(f"Z₂ = {z_np[1]:.3f}")
                else:
                    st.write(f"Z = [{', '.join([f'{v:.2f}' for v in z_np[:5]])}...]")
            
            # Calculate reconstruction error
            recon_error = F.mse_loss(recon, tensor).item()
            st.metric("Reconstruction Error (MSE)", f"{recon_error:.4f}")
            
            if recon_error < 0.02:
                st.success("Excellent reconstruction! The model generalizes well to this input.")
            elif recon_error < 0.05:
                st.info("👍 Good reconstruction with minor differences.")
            else:
                st.warning("Higher error - the input might be quite different from MNIST style.")


if __name__ == "__main__":
    main()
