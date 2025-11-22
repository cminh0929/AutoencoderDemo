"""Training utilities for autoencoder models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Compute VAE loss (reconstruction + KL divergence).
    
    Args:
        recon_x: Reconstructed images
        x: Original images
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        
    Returns:
        Total VAE loss
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD


def train_autoencoder(
    model: nn.Module,
    train_loader,
    test_loader,
    num_epochs: int,
    learning_rate: float = 0.0001,
    flatten_input: bool = False
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Train a standard autoencoder.
    
    Args:
        model: Autoencoder model
        train_loader: Training data loader
        test_loader: Test data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        flatten_input: Whether to flatten input images
        
    Returns:
        List of (epoch, original_images, reconstructed_images) tuples
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    outputs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            if flatten_input:
                images = images.view(images.size(0), -1)
            
            recon = model(images)
            loss = criterion(recon, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i == 0:
                outputs.append((epoch, images.detach().cpu(), recon.detach().cpu()))
        
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                if flatten_input:
                    images = images.view(images.size(0), -1)
                recon = model(images)
                loss = criterion(recon, images)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(test_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
    
    return outputs


def train_denoising_autoencoder(
    model: nn.Module,
    train_loader,
    num_epochs: int,
    noise_factor: float = 0.32,
    learning_rate: float = 0.001
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Train a denoising autoencoder.
    
    Args:
        model: Denoising autoencoder model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        noise_factor: Standard deviation of Gaussian noise
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of (epoch, noisy_images, reconstructed_images) tuples
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    outputs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            # Add noise
            noisy_imgs = images + noise_factor * torch.randn_like(images)
            noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
            
            # Forward pass
            recon = model(noisy_imgs)
            loss = criterion(recon, images)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx == 0:
                outputs.append((epoch, noisy_imgs[:9].cpu(), recon[:9].cpu()))
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")
    
    return outputs


def train_vae(
    model: nn.Module,
    train_loader,
    num_epochs: int,
    learning_rate: float = 0.001
) -> List[Tuple[int, torch.Tensor, torch.Tensor]]:
    """
    Train a Variational Autoencoder.
    
    Args:
        model: VAE model
        train_loader: Training data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of (epoch, original_images, reconstructed_images) tuples
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    outputs = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            # Forward pass
            recon, mu, logvar = model(images)
            loss = vae_loss(recon, images, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx == 0:
                outputs.append((epoch, images[:9].cpu(), recon[:9].cpu()))
        
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")
    
    return outputs
