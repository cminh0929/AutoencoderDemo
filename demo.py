"""Demo script to load saved models and show results."""

import torch
import matplotlib.pyplot as plt
from data import get_mnist_loaders
from models import LinearAutoencoder, CNNAutoencoder, DenoisingAutoencoder, VAE


def demo_linear_ae():
    """Demo linear autoencoder with saved model."""
    
    print("DEMO: Linear Autoencoder")
    
    # Load model
    model = LinearAutoencoder()
    model.load_state_dict(torch.load("saved_models/linear_autoencoder.pth", weights_only=True))
    model.eval()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get test data
    _, test_loader = get_mnist_loaders(batch_size=9)
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            images_flat = images.view(images.size(0), -1)
            recon_flat = model(images_flat)
            recon = recon_flat.view(-1, 1, 28, 28)
            break
    
    # Visualize
    images = images.cpu().numpy()
    recon = recon.cpu().numpy()
    
    plt.figure(figsize=(9, 2.5))
    plt.suptitle("Linear Autoencoder - Demo", fontsize=13)
    plt.gray()
    
    for i in range(9):
        plt.subplot(2, 9, i+1)
        plt.imshow(images[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original", fontsize=9)
    
    for i in range(9):
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(recon[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Reconstructed", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("results/demo_linear_ae.png", dpi=150, bbox_inches='tight')
    print("Saved: results/demo_linear_ae.png")
    plt.show()


def demo_cnn_ae():
    """Demo CNN autoencoder with saved model."""
    
    print("DEMO: CNN Autoencoder")
    
    # Load model
    model = CNNAutoencoder()
    model.load_state_dict(torch.load("saved_models/cnn_autoencoder.pth", weights_only=True))
    model.eval()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get test data
    _, test_loader = get_mnist_loaders(batch_size=9)
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            recon = model(images)
            break
    
    # Visualize
    images = images.cpu().numpy()
    recon = recon.cpu().numpy()
    
    plt.figure(figsize=(9, 2.5))
    plt.suptitle("CNN Autoencoder - Demo", fontsize=13)
    plt.gray()
    
    for i in range(9):
        plt.subplot(2, 9, i+1)
        plt.imshow(images[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original", fontsize=9)
    
    for i in range(9):
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(recon[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Reconstructed", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("results/demo_cnn_ae.png", dpi=150, bbox_inches='tight')
    print("Saved: results/demo_cnn_ae.png")
    plt.show()


def demo_denoising_ae():
    """Demo denoising autoencoder with saved model."""
    
    print("DEMO: Denoising Autoencoder")
    
    # Load model
    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load("saved_models/denoising_autoencoder.pth", weights_only=True))
    model.eval()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get test data
    _, test_loader = get_mnist_loaders(batch_size=9)
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            # Add noise
            noisy = images + 0.32 * torch.randn_like(images)
            noisy = torch.clamp(noisy, 0., 1.)
            recon = model(noisy)
            break
    
    # Visualize
    noisy = noisy.cpu().numpy()
    recon = recon.cpu().numpy()
    
    plt.figure(figsize=(9, 2.5))
    plt.suptitle("Denoising Autoencoder - Demo", fontsize=13)
    plt.gray()
    
    for i in range(9):
        plt.subplot(2, 9, i+1)
        plt.imshow(noisy[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Noisy", fontsize=9)
    
    for i in range(9):
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(recon[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Denoised", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("results/demo_denoising_ae.png", dpi=150, bbox_inches='tight')
    print("Saved: results/demo_denoising_ae.png")
    plt.show()


def demo_vae():
    """Demo VAE with saved model."""
    
    print("DEMO: Variational Autoencoder (VAE)")
    
    # Load model
    model = VAE(latent_dim=50)
    model.load_state_dict(torch.load("saved_models/vae.pth", weights_only=True))
    model.eval()
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get test data
    _, test_loader = get_mnist_loaders(batch_size=9)
    
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            recon, _, _ = model(images)
            break
        
        # Generate new images
        z = torch.randn(9, model.latent_dim).to(device)
        generated = model.decode(z)
    
    # Visualize reconstruction
    images = images.cpu().numpy()
    recon = recon.cpu().numpy()
    
    plt.figure(figsize=(9, 2.5))
    plt.suptitle("VAE Reconstruction - Demo", fontsize=13)
    plt.gray()
    
    for i in range(9):
        plt.subplot(2, 9, i+1)
        plt.imshow(images[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Original", fontsize=9)
    
    for i in range(9):
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(recon[i, 0])
        plt.axis("off")
        if i == 0:
            plt.ylabel("Reconstructed", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("results/demo_vae_reconstruction.png", dpi=150, bbox_inches='tight')
    print("Saved: results/demo_vae_reconstruction.png")
    plt.show()
    
    # Visualize generation
    generated = generated.cpu().numpy()
    
    plt.figure(figsize=(9, 1.5))
    plt.suptitle("VAE Generated Images - Demo", fontsize=13)
    plt.gray()
    
    for i in range(9):
        plt.subplot(1, 9, i+1)
        plt.imshow(generated[i, 0])
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("results/demo_vae_generation.png", dpi=150, bbox_inches='tight')
    print("Saved: results/demo_vae_generation.png")
    plt.show()


def main():
    """Run all demos."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo saved autoencoder models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['linear', 'cnn', 'denoising', 'vae', 'all'],
        default='all',
        help='Which model to demo (default: all)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.model == 'linear' or args.model == 'all':
            demo_linear_ae()
        
        if args.model == 'cnn' or args.model == 'all':
            demo_cnn_ae()
        
        if args.model == 'denoising' or args.model == 'all':
            demo_denoising_ae()
        
        if args.model == 'vae' or args.model == 'all':
            demo_vae()
        
        
        print("ALL DEMOS COMPLETED")
        
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please train the models first by running: python main.py")
        print("Or specify which experiment to run: python main.py --experiment <name>")


if __name__ == '__main__':
    main()
