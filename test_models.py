import torch

from src.models import VAE, UNetAutoencoder


x = torch.rand(8, 3, 128, 128)

print("Probando VAE")
vae = VAE(input_channels=3, latent_dim=128)
reconstruction, mu, logvar, z = vae(x)

print("Input:", x.shape)
print("Reconstruction:", reconstruction.shape)
print("Mu:", mu.shape)
print("Logvar:", logvar.shape)
print("Z:", z.shape)

print("\nProbando U-Net Autoencoder")
unet = UNetAutoencoder(input_channels=3, latent_dim=128)
reconstruction, z = unet(x)

print("Input:", x.shape)
print("Reconstruction:", reconstruction.shape)
print("Z:", z.shape)