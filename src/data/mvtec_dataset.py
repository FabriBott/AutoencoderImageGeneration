from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset


class MVTecDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        classes: list[str],
        split: str,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.split = split
        self.transform = transform
        self.samples = []

        self._load_samples()

    def _load_samples(self):
        for class_name in self.classes:
            class_path = self.root_dir / class_name

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        image = Image.open(sample["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "class_name": sample["class_name"],
            "defect_type": sample["defect_type"],
            "path": str(sample["path"]),
        }