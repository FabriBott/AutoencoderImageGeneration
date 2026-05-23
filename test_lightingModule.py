from src.data import MVTecDataModule
from src.lightning import AutoencoderModule


dm = MVTecDataModule(
    data_dir="data/mvtec_ad",
    batch_size=4,
    num_workers=0,
)

dm.setup()

batch = next(iter(dm.train_dataloader()))

model = AutoencoderModule(
    model_name="vae",
    loss_name="l1",
    input_channels=3,
    latent_dim=128,
    learning_rate=1e-3,
)

output = model._shared_step(batch, stage="train")

print("Loss:", output["loss"].item())
print("Images:", output["images"].shape)
print("Reconstructions:", output["reconstructions"].shape)
print("Z:", output["z"].shape)