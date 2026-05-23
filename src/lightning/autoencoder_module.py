import torch
import pytorch_lightning as pl

from src.models import VAE, UNetAutoencoder
from src.losses import get_loss

"""
contiene lógica de entrenamiento de los autoencoders

actúa como el puente entre:

dataset > modelo > función de pérdida > optimizador > métricas

se utiliza cuando se ejecuta train.py

define qué ocurre en cada paso del
entrenamiento
"""

class AutoencoderModule(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "vae",
        loss_name: str = "l1",
        input_channels: int = 3,
        latent_dim: int = 128,
        learning_rate: float = 1e-3,
        kl_weight: float = 1e-4,
    ):
        super().__init__()

        # guarda los hiperparámetros para que lightning y wandb puedan registrarlos
        self.save_hyperparameters()

        # guarda los hiperparámetros como atributos para usarlos en el resto del módulo
        self.model_name = model_name
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight

        if model_name == "vae":
            self.model = VAE(
                input_channels=input_channels,
                latent_dim=latent_dim,
            )

        elif model_name == "unet":
            self.model = UNetAutoencoder(
                input_channels=input_channels,
                latent_dim=latent_dim,
            )

        else:
            raise ValueError(f"Modelo no reconocido: {model_name}")

        # obtiene la función de pérdida configurada
        self.reconstruction_loss = get_loss(loss_name)

    # ejecuta el modelo en el paso de entrenamiento, validación o prueba
    def forward(self, x):
        return self.model(x)

    # logica comun para entrenamiento, validación y prueba
    def _shared_step(self, batch, stage: str):
        # extrae las imágenes del batch
        images = batch["image"]

        if self.model_name == "vae":
            # retorna reconstrucciones, media, varianza, espacio latente
            reconstructions, mu, logvar, z = self.model(images)

            # calcula la pérdida de reconstrucción
            reconstruction_loss = self.reconstruction_loss(
                reconstructions,
                images,
            )

            # calcula la pérdida KL para poder ordenar el espacio latente
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )

            loss = reconstruction_loss + self.kl_weight * kl_loss

            self.log(
                f"{stage}/reconstruction_loss",
                reconstruction_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

            self.log(
                f"{stage}/kl_loss",
                kl_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )

        # unet
        else:
            # retorna reconstrucciones y espacio latente
            reconstructions, z = self.model(images)

            # calcula error entre imagen original y reconstruida
            reconstruction_loss = self.reconstruction_loss(
                reconstructions,
                images,
            )

            loss = reconstruction_loss

            self.log(
                f"{stage}/reconstruction_loss",
                reconstruction_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        # retorna informacion para analisis posterior
        return {
            "loss": loss,
            "reconstructions": reconstructions,
            "images": images,
            "z": z,
        }

    # define qué ocurre en cada paso del entrenamiento
    # lightning se encarga de llamar a este método en cada iteración del entrenamiento
    def training_step(self, batch, batch_idx):
        output = self._shared_step(batch, stage="train")
        return output["loss"]

    # define qué ocurre en cada paso de la validación
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    # define qué ocurre en cada paso de la prueba
    # se usa al final del entrenamiento para evaluar el modelo con el conjunto de prueba
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    # define el optimizador que se usará para actualizar los pesos del modelo durante el entrenamiento
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        return optimizer