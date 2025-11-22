"""Autoencoder model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAutoencoder(nn.Module):
    """Simple linear autoencoder with fully connected layers."""
    
    def __init__(self):
        """Initialize encoder and decoder networks."""
        super().__init__()
        # Encoder: 784 → 128 → 64
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        # Decoder: 64 → 128 → 784
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28*28),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 784)
            
        Returns:
            Reconstructed tensor of shape (batch, 784)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class CNNAutoencoder(nn.Module):
    """Convolutional autoencoder for image data."""
    
    def __init__(self):
        """Initialize encoder and decoder networks."""
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # (1,28,28) → (16,14,14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (16,14,14) → (32,7,7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),                        # (32,7,7) → (64,1,1)
            nn.ReLU(),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),              # (64,1,1) → (32,7,7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (32,7,7) → (16,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # (16,14,14) → (1,28,28)
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Reconstructed tensor of shape (batch, 1, 28, 28)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DenoisingAutoencoder(nn.Module):
    """CNN-based denoising autoencoder."""
    
    def __init__(self):
        """Initialize encoder and decoder networks."""
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # (1,28,28) → (16,14,14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (16,14,14) → (32,7,7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),                       # (32,7,7) → (64,1,1)
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),              # (64,1,1) → (32,7,7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (32,7,7) → (16,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # (16,14,14) → (1,28,28)
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Reconstructed tensor of shape (batch, 1, 28, 28)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class VAE(nn.Module):
    """Variational Autoencoder with reparameterization trick."""
    
    def __init__(self, latent_dim: int = 50):
        """
        Initialize VAE networks.
        
        Args:
            latent_dim: Dimension of latent space
        """
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),   # (1,28,28) → (16,14,14)
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # (16,14,14) → (32,7,7)
            nn.ReLU(),
            nn.Conv2d(32, 64, 7),                       # (32,7,7) → (64,1,1)
            nn.ReLU(),
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        
        # Decoder input
        self.fc_decode = nn.Linear(latent_dim, 64)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 7),              # (64,1,1) → (32,7,7)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),  # (32,7,7) → (16,14,14)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),   # (16,14,14) → (1,28,28)
            nn.Sigmoid()
        )
    
    def encode(self, x: torch.Tensor):
        """
        Encode input to latent parameters.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Tuple of (mu, logvar) tensors
        """
        h = self.encoder(x)
        h = h.view(h.size(0), -1)  # Flatten: (batch, 64, 1, 1) → (batch, 64)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + sigma * epsilon.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vector to image.
        
        Args:
            z: Latent vector of shape (batch, latent_dim)
            
        Returns:
            Reconstructed image of shape (batch, 1, 28, 28)
        """
        h = self.fc_decode(z)
        h = h.view(h.size(0), 64, 1, 1)  # Reshape: (batch, 64) → (batch, 64, 1, 1)
        recon = self.decoder(h)
        return recon
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through VAE.
        
        Args:
            x: Input tensor of shape (batch, 1, 28, 28)
            
        Returns:
            Tuple of (reconstruction, mu, logvar)
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
