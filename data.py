"""Data loading and preprocessing utilities for autoencoder experiments."""

import torch
from torchvision import datasets, transforms
from typing import Tuple


def get_data_loaders(
    dataset_name: str = 'mnist',
    batch_size: int = 64,
    data_dir: str = './data'
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and test data loaders for various datasets.
    
    Args:
        dataset_name: Name of dataset ('mnist', 'fashion_mnist', 'cifar10')
        batch_size: Number of samples per batch
        data_dir: Directory to store/load data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'mnist':
        return get_mnist_loaders(batch_size, data_dir)
    elif dataset_name == 'fashion_mnist':
        return get_fashion_mnist_loaders(batch_size, data_dir)
    elif dataset_name == 'cifar10':
        return get_cifar10_loaders(batch_size, data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from: mnist, fashion_mnist, cifar10")


def get_mnist_loaders(batch_size: int = 64, data_dir: str = './data'):
    """
    Create MNIST train and test data loaders.
    
    Args:
        batch_size: Number of samples per batch
        data_dir: Directory to store/load MNIST data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Train set
    train_dataset = datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Test set
    test_dataset = datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader


def get_fashion_mnist_loaders(batch_size: int = 64, data_dir: str = './data'):
    """
    Create Fashion-MNIST train and test data loaders.
    Fashion-MNIST: 28x28 grayscale images of clothing items.
    
    Args:
        batch_size: Number of samples per batch
        data_dir: Directory to store/load Fashion-MNIST data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Train set
    train_dataset = datasets.FashionMNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Test set
    test_dataset = datasets.FashionMNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader

'''
def get_cifar10_loaders(batch_size: int = 64, data_dir: str = './data'):
    """
    Create CIFAR-10 train and test data loaders.
    CIFAR-10: 32x32 RGB color images of 10 classes.
    
    Args:
        batch_size: Number of samples per batch
        data_dir: Directory to store/load CIFAR-10 data
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Train set
    train_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    # Test set
    test_dataset = datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader
'''

def add_noise(images: torch.Tensor, noise_factor: float = 0.32) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    Args:
        images: Input images tensor
        noise_factor: Standard deviation of Gaussian noise
        
    Returns:
        Noisy images clamped to [0, 1]
    """
    noisy_imgs = images + noise_factor * torch.randn_like(images)
    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
    return noisy_imgs
