from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

"""
convertir los archivos almacenados en información
que pytorch pueda utilizar

convierte data/mvtec_ad/cable/test/bent_wire/000.png en:
{
    "image": tensor(...),
    "label": 1,
    "class_name": "cable",
    "defect_type": "bent_wire" }
"""

class MVTecDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        classes: list[str],
        split: str,
        transform: Optional[Callable] = None,
    ):
        # carpeta raíz del dataset, clases a incluir, tipo de split (train/val/test) y transformaciones a aplicar
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.split = split
        self.transform = transform
        self.samples = []

        # construye la lista de muestras leyendo las imágenes del disco
        self._load_samples()

    # recorre las carpetas del dataset y carga la información de cada imagen en self.samples
    def _load_samples(self):
        # recorre cada clase y carga las imágenes según el tipo de split
        for class_name in self.classes:
            class_path = self.root_dir / class_name

            # durante el entrenamiento solo usa imagenes buenas
            if self.split == "train":
                image_dir = class_path / "train" / "good"

                for img_path in image_dir.glob("*.png"):
                    self.samples.append(
                        {
                            "path": img_path,
                            "class_name": class_name,
                            "defect_type": "good",
                            "label": 0,
                        }
                    )

            # para validación y prueba se cargan imágenes buenas y defectuosas
            elif self.split in ["val", "test"]:
                test_dir = class_path / "test"

                for defect_dir in test_dir.iterdir():
                    if not defect_dir.is_dir():
                        continue

                    defect_type = defect_dir.name
                    label = 0 if defect_type == "good" else 1

                    for img_path in defect_dir.glob("*.png"):
                        self.samples.append(
                            {
                                "path": img_path,
                                "class_name": class_name,
                                "defect_type": defect_type,
                                "label": label,
                            }
                        )

            else:
                raise ValueError(f"Split no válido: {self.split}")

    # devuelve numero de muestras cargadas
    def __len__(self):
        return len(self.samples)

    # devuelve la imagen procesada y su información asociada para un índice dado
    # este método es llamado automáticamente por pytorch cada vez que necesita una imagen para construir un batch
    def __getitem__(self, idx):
        # obtiene la muestra correspondiente al índice solicitado
        sample = self.samples[idx]

        # abre la imagen desde el disco
        image = Image.open(sample["path"]).convert("RGB")

        # aplica las transformaciones definidas
        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "class_name": sample["class_name"],
            "defect_type": sample["defect_type"],
            "path": str(sample["path"]),
        }