"""Visualization utilities for autoencoder results."""

import torch
import matplotlib.pyplot as plt
from typing import List, Tuple


def plot_reconstructions(
    outputs: List[Tuple[int, torch.Tensor, torch.Tensor]],
    title_prefix: str = "Epoch",
    row1_label: str = "Original",
    row2_label: str = "Reconstructed",
    save_dir: str = None
):
    """
    Plot original and reconstructed images at checkpoints.
    
    Args:
        outputs: List of (epoch, original_images, reconstructed_images) tuples
        title_prefix: Prefix for plot title
        row1_label: Label for first row
        row2_label: Label for second row
        save_dir: Directory to save plots (if None, only show)
    """
    import os
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    checkpoints = [0, len(outputs)//2, len(outputs)-1]
    
    for idx in checkpoints:
        epoch, imgs, recon = outputs[idx]
        imgs = imgs.detach().cpu().numpy()
        recon = recon.detach().cpu().numpy()
        
        plt.figure(figsize=(9, 2.5))
        plt.suptitle(f"{title_prefix} {epoch+1}", fontsize=13)
        plt.gray()
        
        # First row
        for i in range(9):
            plt.subplot(2, 9, i+1)
            
            if imgs.ndim == 2:  # Flattened: (batch, 784)
                img = imgs[i].reshape(28, 28)
            elif imgs.ndim == 4:  # Image: (batch, 1, 28, 28)
                img = imgs[i, 0]
            else:
                img = imgs[i]
            
            plt.imshow(img)
            plt.axis("off")
            if i == 0:
                plt.ylabel(row1_label, fontsize=9)
        
        # Second row
        for i in range(9):
            plt.subplot(2, 9, 9+i+1)
            
            if recon.ndim == 2:  # Flattened: (batch, 784)
                rec = recon[i].reshape(28, 28)
            elif recon.ndim == 4:  # Image: (batch, 1, 28, 28)
                rec = recon[i, 0]
            else:
                rec = recon[i]
            
            plt.imshow(rec)
            plt.axis("off")
            if i == 0:
                plt.ylabel(row2_label, fontsize=9)
        
        plt.tight_layout()
        
        if save_dir:
            filename = f"{title_prefix.replace(' ', '_').lower()}_epoch_{epoch+1}.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"Saved: {filepath}")
        
        plt.show()
        plt.close()


def plot_vae_generation(model, test_loader, save_path: str = None):
    """
    Plot real vs generated images from VAE.
    
    Args:
        model: Trained VAE model
        test_loader: Test data loader
        save_path: Path to save the plot (if None, only show)
    """
    import os
    
    # Get device from model
    device = next(model.parameters()).device
    
    model.eval()
    with torch.no_grad():
        # Get real images
        for real_images, _ in test_loader:
            real_images = real_images[:9]
            break
        
        # Generate new images
        z = torch.randn(9, model.latent_dim).to(device)
        generated = model.decode(z).cpu()
    
    plt.figure(figsize=(9, 2.5))
    plt.suptitle("VAE: Real vs Generated Images", fontsize=13)
    plt.gray()
    
    # Real images
    for i in range(9):
        plt.subplot(2, 9, i+1)
        plt.imshow(real_images[i, 0], cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.ylabel("Real", fontsize=9)
    
    # Generated images
    for i in range(9):
        plt.subplot(2, 9, 9+i+1)
        plt.imshow(generated[i, 0], cmap='gray')
        plt.axis("off")
        if i == 0:
            plt.ylabel("Generated", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def show_noise_example(train_loader, noise_factor: float = 0.32, save_path: str = None):
    """
    Show example of original vs noisy image.
    
    Args:
        train_loader: Training data loader
        noise_factor: Standard deviation of Gaussian noise
        save_path: Path to save the plot (if None, only show)
    """
    import os
    
    for images, _ in train_loader:
        noisy_imgs = images + noise_factor * torch.randn_like(images)
        noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
        break
    
    plt.figure(figsize=(6, 2))
    plt.subplot(1, 2, 1)
    plt.imshow(images[0, 0], cmap='gray')
    plt.title("Original")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(noisy_imgs[0, 0], cmap='gray')
    plt.title("Noisy")
    plt.axis("off")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()
