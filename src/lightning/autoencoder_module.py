import torch
import pytorch_lightning as pl
import wandb

from pytorch_lightning.loggers import WandbLogger

from src.models import VAE, UNetAutoencoder
from src.losses import get_loss
from src.utils import visualizations

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
        num_visualizations: int = 16,
        max_tsne_samples: int = 500,
    ):
        super().__init__()

        # guarda los hiperparámetros para que lightning y wandb puedan registrarlos
        self.save_hyperparameters()

        # guarda los hiperparámetros como atributos para usarlos en el resto del módulo
        self.model_name = model_name
        self.loss_name = loss_name
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.num_visualizations = num_visualizations
        self.max_tsne_samples = max_tsne_samples

        self._val_images = []
        self._val_reconstructions = []
        self._val_z = []
        self._val_z_labels = []

        self._test_images = []
        self._test_reconstructions = []
        self._test_labels_vis = []
        self._test_errors = []
        self._test_labels_all = []
        self._test_defect_types = []
        self._test_class_names = []

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
            "label": batch["label"],
            "class_name": batch["class_name"],
            "defect_type": batch["defect_type"],
        }

    # define qué ocurre en cada paso del entrenamiento
    # lightning se encarga de llamar a este método en cada iteración del entrenamiento
    def training_step(self, batch, batch_idx):
        output = self._shared_step(batch, stage="train")
        return output["loss"]

    # define qué ocurre en cada paso de la validación
    def validation_step(self, batch, batch_idx):
        output = self._shared_step(batch, stage="val")

        val_image_count = sum(t.size(0) for t in self._val_images)
        if val_image_count < self.num_visualizations:
            remaining = self.num_visualizations - val_image_count
            images = output["images"][:remaining].detach().cpu()
            reconstructions = output["reconstructions"][:remaining].detach().cpu()

            self._val_images.append(images)
            self._val_reconstructions.append(reconstructions)

        if sum(t.size(0) for t in self._val_z) < self.max_tsne_samples:
            remaining = self.max_tsne_samples - sum(t.size(0) for t in self._val_z)
            z = output["z"][:remaining].detach().cpu()
            class_names = output["class_name"][:remaining]

            self._val_z.append(z)
            self._val_z_labels.extend(list(class_names))

        return output

    # define qué ocurre en cada paso de la prueba
    # se usa al final del entrenamiento para evaluar el modelo con el conjunto de prueba
    def test_step(self, batch, batch_idx):
        output = self._shared_step(batch, stage="test")

        per_sample_error = torch.mean(
            torch.abs(output["reconstructions"] - output["images"]),
            dim=(1, 2, 3),
        )

        test_image_count = sum(t.size(0) for t in self._test_images)
        if test_image_count < self.num_visualizations:
            remaining = self.num_visualizations - test_image_count
            images = output["images"][:remaining].detach().cpu()
            reconstructions = output["reconstructions"][:remaining].detach().cpu()
            labels = output["label"][:remaining].detach().cpu()

            self._test_images.append(images)
            self._test_reconstructions.append(reconstructions)
            self._test_labels_vis.append(labels)

        self._test_errors.extend(per_sample_error.detach().cpu().tolist())
        self._test_labels_all.extend(output["label"].detach().cpu().tolist())
        self._test_defect_types.extend(list(output["defect_type"]))
        self._test_class_names.extend(list(output["class_name"]))

        return output

    # define el optimizador que se usará para actualizar los pesos del modelo durante el entrenamiento
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        return optimizer

    def on_validation_epoch_start(self):
        self._val_images = []
        self._val_reconstructions = []
        self._val_z = []
        self._val_z_labels = []

    def on_test_epoch_start(self):
        self._test_images = []
        self._test_reconstructions = []
        self._test_labels_vis = []
        self._test_errors = []
        self._test_labels_all = []
        self._test_defect_types = []
        self._test_class_names = []

    def on_validation_epoch_end(self):
        wandb_logger = self._get_wandb_logger()
        if wandb_logger is None:
            return

        if self._val_images:
            images = torch.cat(self._val_images, dim=0)
            reconstructions = torch.cat(self._val_reconstructions, dim=0)
            grid = visualizations.build_reconstruction_grid(
                images,
                reconstructions,
                max_images=self.num_visualizations,
            )

            if grid is not None:
                wandb_logger.experiment.log(
                    {"val/reconstructions": wandb.Image(grid)},
                    step=self.global_step,
                )

        if self._val_z:
            z = torch.cat(self._val_z, dim=0)
            labels = self._val_z_labels[: z.size(0)]
            fig = visualizations.plot_tsne(
                z,
                labels,
                max_points=self.max_tsne_samples,
            )

            if fig is not None:
                wandb_logger.experiment.log(
                    {"val/tsne": wandb.Image(fig)},
                    step=self.global_step,
                )
            
            table = visualizations.plot_tsne_wandb(
                z,
                labels,
                max_points=self.max_tsne_samples
                )
            if table is not None:
                wandb_logger.experiment.log(
                    {"val/tsne_table": table},
                    step=self.global_step,
                )

    def on_test_epoch_end(self):
        wandb_logger = self._get_wandb_logger()
        if wandb_logger is None:
            return

        if self._test_images:
            images = torch.cat(self._test_images, dim=0)
            reconstructions = torch.cat(self._test_reconstructions, dim=0)
            labels = torch.cat(self._test_labels_vis, dim=0)

            grid = visualizations.build_test_grid(
                images,
                reconstructions,
                labels,
                max_images=self.num_visualizations,
            )

            if grid is not None:
                wandb_logger.experiment.log(
                    {"test/reconstructions": wandb.Image(grid)},
                    step=self.global_step,
                )

        good_errors = []
        defect_errors_by_type = {}
        histogram_groups = {}

        for error, label, defect_type, class_name in zip(
            self._test_errors,
            self._test_labels_all,
            self._test_defect_types,
            self._test_class_names
        ):
            
            if label == 0:
                good_errors.append(error)
                histogram_groups.setdefault(class_name, {}).setdefault("good_errors", []).append(error)
            else:
                defect_errors_by_type.setdefault(defect_type, []).append(error)
                histogram_groups.setdefault(class_name, {}).setdefault("defect_errors", {}).setdefault(defect_type, []).append(error)

        hist_figs = visualizations.plot_error_histograms(
            good_errors,
            defect_errors_by_type,
        )

        hist_tables_wandb = visualizations.plot_error_histograms_wandb(histogram_groups)

        '''for class_name, defect_type, table in hist_tables_wandb:
            wandb_logger.experiment.log(
                {f"test/wberror_hist_{class_name}_{defect_type}": 
                 wandb.plot.histogram(table, "Reconstruction error",
                                      title=f"Reconstruction error: good vs {defect_type}")},
                 step=self.global_step,
            )
        '''
            
        for class_name, defect_type, fig in hist_tables_wandb:
            wandb_logger.experiment.log(
                {f"test/wberror_hist_{class_name}_{defect_type}": wandb.Image(fig)},
                 step=self.global_step,
            )

        for defect_type, fig in hist_figs:
            wandb_logger.experiment.log(
                {f"test/error_hist_{defect_type}": wandb.Image(fig)},
                step=self.global_step,
            )

        metrics = self.trainer.callback_metrics
        test_loss = metrics.get("test/loss")
        test_recon_loss = metrics.get("test/reconstruction_loss")

        if test_loss is not None or test_recon_loss is not None:
            table = wandb.Table(
                columns=["loss_name", "test_loss", "test_reconstruction_loss"],
            )
            table.add_data(
                self.loss_name,
                float(test_loss) if test_loss is not None else None,
                float(test_recon_loss) if test_recon_loss is not None else None,
            )
            wandb_logger.experiment.log(
                {"test/summary_table": table},
                step=self.global_step,
            )

    def _get_wandb_logger(self):
        if isinstance(self.logger, WandbLogger):
            return self.logger

        if self.trainer is not None:
            for logger in self.trainer.loggers:
                if isinstance(logger, WandbLogger):
                    return logger

        return None
    
    """
configurar el entrenamiento
=
crear modelo
crear loss
crear optimizador

y además

ejecutar el entrenamiento
ejecutar validación
ejecutar test
calcular pérdidas
registrar métricas
    """