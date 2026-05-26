from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl

from src.data.mvtec_dataset import MVTecDataset

"""
organizar flujo de datos utilizado durante el entrenamiento,
validación y prueba de los modelos

encargado de alimentar imágenes al modelo
durante todo el proceso de entrenamiento
"""

class MVTecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/mvtec_ad",
        classes: list[str] | None = None,
        image_size: int = 128,
        batch_size: int = 32,
        num_workers: int = 4,
        val_split: float = 0.2,
    ):
        super().__init__()

        # ruta del dataset y otras configuraciones
        self.data_dir = data_dir
        self.classes = classes or ["cable", "capsule", "screw", "transistor"]
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        # transformaciones para preprocesar las imágenes
        self.transform = transforms.Compose(
            [
                # se redimensionan las imagenes a 128x128
                transforms.Resize((self.image_size, self.image_size)),
                # convierte a tensor y normaliza entre 0 y 1
                transforms.ToTensor(),
            ]
        )

    def setup(self, stage: str | None = None):
        
        # carga todas las imágenes de entrenamiento
        full_train = MVTecDataset(
            root_dir=self.data_dir,
            classes=self.classes,
            split="train",
            transform=self.transform,
        )

        # calcula cuántas imágenes se utilizarán para validación
        val_size = int(len(full_train) * self.val_split)
        # y el resto para entrenamiento
        train_size = len(full_train) - val_size

        # divide en entrenamiento y validación
        self.train_dataset, self.val_dataset = random_split(
            full_train,
            [train_size, val_size],
        )

        # carga el conjunto de prueba completo
        self.test_dataset = MVTecDataset(
            root_dir=self.data_dir,
            classes=self.classes,
            split="test",
            transform=self.transform,
        )

    def train_dataloader(self):
        # crea el DataLoader para el conjunto de entrenamiento
        # shuffle=True para mezclar los datos en cada época
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        # crea el DataLoader para el conjunto de validación
        # shuffle=False porque no es necesario mezclar los datos de validación
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        # crea el dataloader utilizado para pruebas finales
        # se utiliza una vez concluido el entrenamiento para medir el desempeño real del modelo
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )