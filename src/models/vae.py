## https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=40xkM9yZlFYk
## https://www.codecademy.com/article/variational-autoencoder-tutorial-vaes-explained
import torch
import torch.nn as nn

# implementacion del VAE 

class VAEEncoder(nn.Module):
    # el encoder, comprime  imagen a un espacio latentez
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        # reduce progresivamente resolucion y aumenta la cantidad de canales 
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # tamaño del tensor aplanado después de las convoluciones
        self.flatten_dim = 256 * 8 * 8

        # general la media y logvar para el espacio latente
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

    def forward(self, x):
        # extrae características visulaes
        x = self.conv_layers(x)
        # convierte a un vector
        x = torch.flatten(x, start_dim=1)

        # media y logvar para el espacio latente
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


class VAEDecoder(nn.Module):
    # el decoder, reconstruye la imagen a partir del espacio latente
    def __init__(self, output_channels=3, latent_dim=128):
        super().__init__()

        # expande el vector latente a un tensor
        self.fc = nn.Linear(latent_dim, 256 * 8 * 8)

        # reduce progresivamente la cantidad de canales y aumenta la resolución
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            # limite la salida entre 0 y 1
            nn.Sigmoid(),
        )

    def forward(self, z):
        # proyecta z al tamaño del bottleneck
        x = self.fc(z)
        # reconstruye la imagen a partir del espacio latente
        x = x.view(-1, 256, 8, 8)
        # genera la imagen final
        x = self.deconv_layers(x)

        return x


class VAE(nn.Module):
    # modelo completo del VAE
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        # parte que comprime
        self.encoder = VAEEncoder(
            input_channels=input_channels,
            latent_dim=latent_dim,
        )

        # parte que reconstruye la imagen 
        self.decoder = VAEDecoder(
            output_channels=input_channels,
            latent_dim=latent_dim,
        )

    # formula = z = mu + epsilon * sigma, donde epsilon es ruido gaussiano
    def reparameterize(self, mu, logvar):
        # calcula la desviación estándar a partir del logvar
        std = torch.exp(0.5 * logvar)
        # genera ruido gaussiano
        epsilon = torch.randn_like(std)

        # aplica la reparametrización para obtener z
        z = mu + epsilon * std

        return z

    def encode(self, x):
        # comprime la imagen y obtiene la media y logvar del espacio latente
        return self.encoder(x)

    def decode(self, z):
        # reconstruye la imagen a partir del vector latente
        return self.decoder(z)

    def forward(self, x):
        # define el flujo de datos a través del modelo completo

        # obtiene mu y logvar del encoder
        mu, logvar = self.encode(x)
        # genera muestra latente y reconstruye
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar, z