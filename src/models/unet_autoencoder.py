## https://debuggercafe.com/unet-from-scratch-using-pytorch/

import torch
import torch.nn as nn

# ejecutado cuando en hydra se ejecuta python train.py model=unet 

# bloque basico de convolución doble, usado tanto en el encoder como en el decoder
# aplica dos convoluciones para extraer características
class DoubleConvolution(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # secuencia de dos convoluciones con activación ReLU
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # aplica las convoluciones al input
        return self.conv(x)


class UNetEncoder(nn.Module):
    # el encoder del UNet, que reduce la resolución 
    # guarda las salidas intermedias para las conexiones de salto
    def __init__(self, input_channels=3):
        super().__init__()

        # reduce la resolución a la mitad
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # bloques de convolución doble para cada nivel del encoder
        self.down1 = DoubleConvolution(input_channels, 64)
        self.down2 = DoubleConvolution(64, 128)
        self.down3 = DoubleConvolution(128, 256)
        self.down4 = DoubleConvolution(256, 512)
        # el cuello de botella, que es la representación más comprimida
        self.bottleneck = DoubleConvolution(512, 1024)

    def forward(self, x):
        # primer nivel del encoder reduce 128x128 a 64x64
        d1 = self.down1(x)
        p1 = self.pool(d1)

        # segundo nivel del encoder reduce 64x64 a 32x32
        d2 = self.down2(p1)
        p2 = self.pool(d2) 

        # tercer nivel del encoder reduce 64x64 a 32x32
        d3 = self.down3(p2)
        p3 = self.pool(d3)

        # cuarto nivel del encoder reduce 32x32 a 16x16
        d4 = self.down4(p3)
        p4 = self.pool(d4)

        # el cuello de botella reduce 16x16 a 8x8, con 1024 canales
        b = self.bottleneck(p4)

        # guarda las salidas
        skips = [d1, d2, d3, d4]

        return b, skips


class UNetDecoder(nn.Module):
    # el decoder del UNet, que reconstruye la imagen a partir de la representación comprimida
    def __init__(self, output_channels=3):
        super().__init__()

        # aumenta la resolución 8x8 a 16x16
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        # concatenacion entre del decoder y el encoder
        self.conv1 = DoubleConvolution(1024, 512)

        # aumenta la resolución 16x16 a 32x32
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConvolution(512, 256)

        # aumenta la resolución 32x32 a 64x64
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConvolution(256, 128)

        # aumenta la resolución 64x64 a 128x128
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConvolution(128, 64)

        # capa final para reconstruir la imagen con el número de canales de salida deseado
        self.out = nn.Sequential(
            nn.Conv2d(64, output_channels, kernel_size=1),
            # limite la salida entre 0 y 1
            nn.Sigmoid(),
        )

    def forward(self, bottleneck, skips):
        # recupera las salidas intermedias del encoder para las conexiones de salto
        d1, d2, d3, d4 = skips

        # primera etapa del decoder, concatena con salida equivalente del encoder y procesa con convoluciones dobles 
        x = self.up1(bottleneck)
        x = torch.cat([d4, x], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        x = torch.cat([d3, x], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        x = torch.cat([d2, x], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        x = torch.cat([d1, x], dim=1)
        x = self.conv4(x)

        # genera la imagen final
        reconstruction = self.out(x)

        return reconstruction


class UNetAutoencoder(nn.Module):
    # modelo completo del UNet que combina encoder, bottleneck y decoder
    def __init__(self, input_channels=3, latent_dim=128):
        super().__init__()

        # parte que comprime y reconstruye la imagen
        self.encoder = UNetEncoder(input_channels=input_channels)
        self.decoder = UNetDecoder(output_channels=input_channels)

        # convierte a espacio latente
        self.flatten = nn.Flatten()
        self.fc_latent = nn.Linear(1024 * 8 * 8, latent_dim)

    def encode(self, x):
        # obtiene el cuello de botella y las salidas intermedias del encoder
        bottleneck, skips = self.encoder(x)

        # convierte el cuello de botella a un vector latente y reduce su dimensionalidad
        z = self.flatten(bottleneck)
        z = self.fc_latent(z)

        return bottleneck, skips, z

    def decode(self, bottleneck, skips):
        # reconstruye imagen 
        return self.decoder(bottleneck, skips)

    def forward(self, x):
        # define el flujo de datos a través del modelo completo
        bottleneck, skips, z = self.encode(x)
        reconstruction = self.decode(bottleneck, skips)

        return reconstruction, z