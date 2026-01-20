# 🎨 VAE/CVAE Interactive Playground

A beautiful Streamlit web application for exploring Variational Autoencoders trained on MNIST.

## Features

### 🔢 Conditional Generation
Generate any digit (0-9) with full control over its style using the CVAE. Adjust latent dimensions to control slant, thickness, and other visual attributes.

### 🎭 Style Transfer  
Upload your own handwritten digit and transfer its **style** to any other digit! This demonstrates that the CVAE has learned to separate *what digit* (class) from *how it looks* (style).

### 🎚️ Latent Space Explorer
Move through the 2D latent space interactively and see how generated digits change in real-time. Visualize the smooth manifold learned by the VAE.

### 🔀 Digit Morphing
Watch one digit smoothly transform into another through latent space interpolation. Choose to interpolate style, class, or both!

### 🎲 Random Sampling
Generate diverse random digits by sampling from the latent space with controllable temperature/diversity.

### 🔍 Reconstruction Test
Upload any digit image (even outside MNIST!) and see how well the model can reconstruct it. Tests generalization capability.

## Setup

### 1. Install dependencies

```bash
cd streamlit_app
pip install -r requirements.txt
```

### 2. Get the trained models

You need the trained model files from the Jupyter notebook:
- `vae_mnist.pth` - Standard VAE model
- `cvae_mnist.pth` - Conditional VAE model

Place them in a `models/` directory:

```
streamlit_app/
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── vae_mnist.pth
    └── cvae_mnist.pth
```

**Option A: Train in notebook and download**
1. Run the training cells in `vae_mnist_project.ipynb`
2. The model saving cells will auto-download the `.pth` files (on Colab)
3. Copy the files to `streamlit_app/models/`

**Option B: Upload directly in the app**
The app allows you to upload model files directly if they're not found.

### 3. Run the app

```bash
cd streamlit_app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Usage Tips

- **Style Transfer** works best with clear, high-contrast digit images
- **Latent Explorer** reveals what each latent dimension controls
- Try extreme values in **Random Sampling** with high temperature for creative outputs
- Use **Reconstruction Test** to check if the model generalizes to your handwriting

## Project Structure

```
gaidm/
├── vae_mnist_project.ipynb    # Main notebook with training code
├── models/                     # Saved model files
│   ├── vae_mnist.pth
│   └── cvae_mnist.pth
└── streamlit_app/
    ├── app.py                  # Streamlit application
    ├── requirements.txt        # Python dependencies
    └── README.md               # This file
```

## Screenshots

The app features a clean, modern interface with:
- Gradient-styled headers
- Interactive sliders for latent space control
- Side-by-side comparisons for style transfer
- Real-time digit generation

## Technical Details

- **Models**: Convolutional VAE/CVAE with 2D latent space
- **Training**: 30 epochs with KL annealing on MNIST
- **Backend**: PyTorch
- **Frontend**: Streamlit with custom CSS styling

---
*Part of the Generative AI and Diffusion Models course project*
