import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from pytorch_lightning.loggers import WandbLogger

from src.data import MVTecDataModule
from src.lightning import AutoencoderModule


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="config",
)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed, workers=True)

    datamodule = MVTecDataModule(
        data_dir=cfg.data.data_dir,
        classes=list(cfg.data.classes),
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        val_split=cfg.data.val_split,
    )

    model = AutoencoderModule(
        model_name=cfg.model.name,
        loss_name=cfg.loss.name,
        input_channels=cfg.model.input_channels,
        latent_dim=cfg.model.latent_dim,
        learning_rate=cfg.trainer.learning_rate,
        kl_weight=cfg.model.kl_weight,
    )

    run_name = f"{cfg.model.name}_{cfg.loss.name}_z{cfg.model.latent_dim}"

    logger = WandbLogger(
        project=cfg.logger.project,
        entity=cfg.logger.entity,
        name=run_name,
        log_model=cfg.logger.log_model,
        group=cfg.logger.group,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=logger,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
    )

    trainer.fit(
        model=model,
        datamodule=datamodule,
    )

    trainer.test(
        model=model,
        datamodule=datamodule,
    )

    wandb.finish()


if __name__ == "__main__":
    main()


# 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"

"""
1. Hydra lee los YAML
   ↓
2. train.py recibe cfg
   ↓
3. train.py crea MVTecDataModule
   ↓
4. MVTecDataModule crea MVTecDataset
   ↓
5. train.py crea AutoencoderModule
   ↓
6. AutoencoderModule crea VAE y L1Loss
   ↓
7. train.py crea Trainer
   ↓
8. Trainer llama datamodule.train_dataloader()
   ↓
9. DataLoader usa MVTecDataset para cargar imágenes
   ↓
10. Trainer llama training_step()
   ↓
11. AutoencoderModule pasa imágenes al VAE
   ↓
12. VAE reconstruye imágenes
   ↓
13. AutoencoderModule calcula loss
   ↓
14. Lightning actualiza pesos
   ↓
15. WandB guarda métricas

"""