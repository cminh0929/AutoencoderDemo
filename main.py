"""Main entry point for running autoencoder experiments."""

import argparse
import os
import torch
from data import get_data_loaders
from models import LinearAutoencoder, CNNAutoencoder, DenoisingAutoencoder, VAE
from training import train_autoencoder, train_denoising_autoencoder, train_vae
from visualization import plot_reconstructions, plot_vae_generation, show_noise_example


def run_linear_autoencoder(dataset: str = 'mnist'):
    """Run linear autoencoder experiment."""
    
    print(f"LINEAR AUTOENCODER EXPERIMENT - {dataset.upper()}")
    
    train_loader, test_loader = get_data_loaders(dataset_name=dataset, batch_size=64)
    model = LinearAutoencoder()
    
    outputs = train_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=10,
        learning_rate=0.0001,
        flatten_input=True
    )
    
    # Save results
    save_dir = f"results/linear_ae_{dataset}"
    plot_reconstructions(outputs, title_prefix=f"Linear AE ({dataset}) - Epoch", save_dir=save_dir)
    
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/linear_autoencoder_{dataset}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def run_cnn_autoencoder(dataset: str = 'mnist'):
    """Run CNN autoencoder experiment."""
    
    print(f"CNN AUTOENCODER EXPERIMENT - {dataset.upper()}")
    
    train_loader, test_loader = get_data_loaders(dataset_name=dataset, batch_size=64)
    model = CNNAutoencoder()
    
    outputs = train_autoencoder(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        num_epochs=5,
        learning_rate=0.0001,
        flatten_input=False
    )
    
    # Save results
    save_dir = f"results/cnn_ae_{dataset}"
    plot_reconstructions(outputs, title_prefix=f"CNN AE ({dataset}) - Epoch", save_dir=save_dir)
    
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/cnn_autoencoder_{dataset}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def run_denoising_autoencoder(dataset: str = 'mnist'):
    """Run denoising autoencoder experiment."""

    print(f"DENOISING AUTOENCODER EXPERIMENT - {dataset.upper()}")
    
    train_loader, _ = get_data_loaders(dataset_name=dataset, batch_size=64)
    
    # Show noise example
    save_dir = f"results/denoising_ae_{dataset}"
    os.makedirs(save_dir, exist_ok=True)
    show_noise_example(train_loader, noise_factor=0.32, save_path=f"{save_dir}/noise_example.png")
    
    model = DenoisingAutoencoder()
    
    outputs = train_denoising_autoencoder(
        model=model,
        train_loader=train_loader,
        num_epochs=3,
        noise_factor=0.32,
        learning_rate=0.001
    )
    
    # Save results
    plot_reconstructions(
        outputs, 
        title_prefix=f"Denoising AE ({dataset}) - Epoch",
        row1_label="Noisy",
        row2_label="Denoised",
        save_dir=save_dir
    )
    
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/denoising_autoencoder_{dataset}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def run_vae(dataset: str = 'mnist'):
    """Run VAE experiment."""
    
    print(f"VARIATIONAL AUTOENCODER (VAE) EXPERIMENT - {dataset.upper()}")
    
    train_loader, test_loader = get_data_loaders(dataset_name=dataset, batch_size=64)
    model = VAE(latent_dim=50)
    
    outputs = train_vae(
        model=model,
        train_loader=train_loader,
        num_epochs=20,
        learning_rate=0.001
    )
    
    # Save results
    save_dir = f"results/vae_{dataset}"
    plot_reconstructions(outputs, title_prefix=f"VAE ({dataset}) - Epoch", save_dir=save_dir)
    plot_vae_generation(model, test_loader, save_path=f"{save_dir}/vae_generation.png")
    
    # Save model
    os.makedirs("saved_models", exist_ok=True)
    model_path = f"saved_models/vae_{dataset}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved: {model_path}")


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Run autoencoder experiments')
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['linear', 'cnn', 'denoising', 'vae', 'all'],
        default='all',
        help='Which experiment to run (default: all)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['mnist', 'fashion_mnist'],
        #choices=['mnist', 'fashion_mnist', 'cifar10'],
        default='mnist',
        help='Which dataset to use (default: mnist)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"DATASET: {args.dataset.upper()}")
    print(f"{'='*60}\n")
    
    if args.experiment == 'linear' or args.experiment == 'all':
        run_linear_autoencoder(args.dataset)
    
    if args.experiment == 'cnn' or args.experiment == 'all':
        run_cnn_autoencoder(args.dataset)
    
    if args.experiment == 'denoising' or args.experiment == 'all':
        run_denoising_autoencoder(args.dataset)
    
    if args.experiment == 'vae' or args.experiment == 'all':
        run_vae(args.dataset)
    
if __name__ == '__main__':
    main()
