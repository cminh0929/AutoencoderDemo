# Autoencoder Experiments

Experiments with various types of Autoencoders on multiple datasets with CUDA support.

## Supported Datasets

1. **MNIST** - Handwritten digits (28×28, grayscale)
2. **Fashion-MNIST** - Clothing items (28×28, grayscale) 
3. **CIFAR-10** - Color images (32×32, RGB)

## Model Architectures

### 1. Linear Autoencoder

```
            ┌─────────────────────────────────────┐
            │        LINEAR AUTOENCODER           │
            └─────────────────────────────────────┘
                           │
                Input Image (28×28)
                           │
             Flatten -> vector 784d
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            ENCODER                │
            └───────────────────────────────────┘
                           │
                Linear: 784 → 128
                           │
                        ReLU
                           │
                Linear: 128 → 64
                           │
                        ReLU
                           ▼
                Latent Vector (64d)
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            DECODER                │
            └───────────────────────────────────┘
                           │
                Linear: 64 → 128
                           │
                        ReLU
                           │
                Linear: 128 → 784
                           │
                      Sigmoid
                           ▼
             Output (Reconstructed Image)
          Reshape → 28 × 28
```

### 2. CNN Autoencoder

```
            ┌─────────────────────────────────────┐
            │         CNN AUTOENCODER             │
            └─────────────────────────────────────┘
                           │
                Input Image (1, 28, 28)
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            ENCODER                │
            └───────────────────────────────────┘
                           │
         Conv2d: 1 → 16, kernel=3, stride=2
                    (16, 14, 14)
                           │
                        ReLU
                           │
         Conv2d: 16 → 32, kernel=3, stride=2
                    (32, 7, 7)
                           │
                        ReLU
                           │
         Conv2d: 32 → 64, kernel=7
                    (64, 1, 1)
                           │
                        ReLU
                           ▼
                Latent Vector (64, 1, 1)
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            DECODER                │
            └───────────────────────────────────┘
                           │
      ConvTranspose2d: 64 → 32, kernel=7
                    (32, 7, 7)
                           │
                        ReLU
                           │
      ConvTranspose2d: 32 → 16, kernel=3, stride=2
                    (16, 14, 14)
                           │
                        ReLU
                           │
      ConvTranspose2d: 16 → 1, kernel=3, stride=2
                    (1, 28, 28)
                           │
                      Sigmoid
                           ▼
             Output (Reconstructed Image)
```

### 3. Denoising Autoencoder

```
            ┌─────────────────────────────────────┐
            │      DENOISING AUTOENCODER          │
            └─────────────────────────────────────┘
                           │
                Input Image (1, 28, 28)
                           │
                Add Gaussian Noise
                  (noise_factor=0.32)
                           │
                Noisy Image (1, 28, 28)
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            ENCODER                │
            └───────────────────────────────────┘
                           │
         Conv2d: 1 → 16, kernel=3, stride=2
                    (16, 14, 14)
                           │
                        ReLU
                           │
         Conv2d: 16 → 32, kernel=3, stride=2
                    (32, 7, 7)
                           │
                        ReLU
                           │
         Conv2d: 32 → 64, kernel=7
                    (64, 1, 1)
                           │
                        ReLU
                           ▼
                Latent Vector (64, 1, 1)
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            DECODER                │
            └───────────────────────────────────┘
                           │
      ConvTranspose2d: 64 → 32, kernel=7
                    (32, 7, 7)
                           │
                        ReLU
                           │
      ConvTranspose2d: 32 → 16, kernel=3, stride=2
                    (16, 14, 14)
                           │
                        ReLU
                           │
      ConvTranspose2d: 16 → 1, kernel=3, stride=2
                    (1, 28, 28)
                           │
                      Sigmoid
                           ▼
          Denoised Image (Clean Output)
```

### 4. Variational Autoencoder (VAE)

```
            ┌─────────────────────────────────────┐
            │    VARIATIONAL AUTOENCODER (VAE)    │
            └─────────────────────────────────────┘
                           │
                Input Image (1, 28, 28)
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            ENCODER                │
            └───────────────────────────────────┘
                           │
         Conv2d: 1 → 16, kernel=3, stride=2
                    (16, 14, 14)
                           │
                        ReLU
                           │
         Conv2d: 16 → 32, kernel=3, stride=2
                    (32, 7, 7)
                           │
                        ReLU
                           │
         Conv2d: 32 → 64, kernel=7
                    (64, 1, 1)
                           │
                        ReLU
                           │
                    Flatten → 64d
                           │
                    ┌──────┴──────┐
                    │             │
              Linear: 64 → 50   Linear: 64 → 50
                    │             │
                   μ (mu)    log(σ²) (logvar)
                    │             │
                    └──────┬──────┘
                           │
                Reparameterization Trick
                  z = μ + σ * ε
                  (ε ~ N(0,1))
                           │
                           ▼
                Latent Vector z (50d)
                           │
                           ▼
            ┌───────────────────────────────────┐
            │            DECODER                │
            └───────────────────────────────────┘
                           │
                Linear: 50 → 64
                           │
                Reshape → (64, 1, 1)
                           │
      ConvTranspose2d: 64 → 32, kernel=7
                    (32, 7, 7)
                           │
                        ReLU
                           │
      ConvTranspose2d: 32 → 16, kernel=3, stride=2
                    (16, 14, 14)
                           │
                        ReLU
                           │
      ConvTranspose2d: 16 → 1, kernel=3, stride=2
                    (1, 28, 28)
                           │
                      Sigmoid
                           ▼
             Output (Reconstructed Image)
             
             Loss = BCE + KL Divergence
```

## Installation

```bash
pip install torch torchvision numpy matplotlib
```

## Usage

### Training with different datasets

**MNIST (default):**
```bash
python main.py
```

**Fashion-MNIST:**
```bash
python main.py --dataset fashion_mnist
```

**CIFAR-10:**
```bash
python main.py --dataset cifar10
```

### Run individual experiments

```bash
# Linear Autoencoder
python main.py --experiment linear --dataset mnist

# CNN Autoencoder
python main.py --experiment cnn --dataset fashion_mnist

# Denoising Autoencoder
python main.py --experiment denoising --dataset mnist

# VAE
python main.py --experiment vae --dataset fashion_mnist
```

### Demo with trained models

```bash
# Demo all models
python demo.py

# Demo specific model
python demo.py --model linear
python demo.py --model cnn
python demo.py --model denoising
python demo.py --model vae
```

## Directory Structure

```
Autoencoder/
├── data.py              # Data loading utilities
├── models.py            # Model architectures
├── training.py          # Training functions
├── visualization.py     # Visualization utilities
├── main.py              # Main training script
├── demo.py              # Demo script for saved models
├── results/             # Saved visualization results
│   ├── linear_ae_mnist/
│   ├── cnn_ae_fashion_mnist/
│   ├── denoising_ae_mnist/
│   └── vae_cifar10/
└── saved_models/        # Saved model weights
    ├── linear_autoencoder_mnist.pth
    ├── cnn_autoencoder_fashion_mnist.pth
    ├── denoising_autoencoder_mnist.pth
    └── vae_cifar10.pth
```

## Results

After training, results are automatically saved:

- **Images**: `results/<experiment>_<dataset>/`
- **Models**: `saved_models/<model>_<dataset>.pth`

Each experiment saves 3 checkpoints (beginning, middle, end) at high resolution (150 DPI) for reports.

## Technical Details

### Linear Autoencoder
- **Input**: 784 (28×28 flattened)
- **Encoder**: 784 → 128 → 64
- **Decoder**: 64 → 128 → 784
- **Epochs**: 10
- **Learning rate**: 0.0001
- **Loss**: MSE

### CNN Autoencoder
- **Input**: (1, 28, 28)
- **Encoder**: Conv layers with stride 2
- **Latent**: (64, 1, 1)
- **Decoder**: Transposed Conv layers
- **Epochs**: 5
- **Learning rate**: 0.0001
- **Loss**: MSE

### Denoising Autoencoder
- **Architecture**: Same as CNN
- **Noise factor**: 0.32 (Gaussian)
- **Epochs**: 3
- **Learning rate**: 0.001
- **Loss**: MSE (between clean and denoised)

### VAE
- **Architecture**: CNN-based
- **Latent dimension**: 50
- **Loss**: BCE + KL Divergence
- **Epochs**: 20
- **Learning rate**: 0.001

