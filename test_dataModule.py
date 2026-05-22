from collections import Counter
from pathlib import Path

from PIL import Image

from src.data.mvtec_datamodule import MVTecDataModule


def main():
    print("=" * 60)
    print("VERIFICACIÓN DEL DATASET MVTec AD")
    print("=" * 60)

    dm = MVTecDataModule(
        data_dir="data/mvtec_ad",
        batch_size=8,
        num_workers=0,
    )

    dm.setup()

    # --------------------------------------------------
    # Tamaños de los conjuntos
    # --------------------------------------------------
    print("\n[1] Tamaños de los datasets")
    print(f"Train: {len(dm.train_dataset)}")
    print(f"Val:   {len(dm.val_dataset)}")
    print(f"Test:  {len(dm.test_dataset)}")

    # --------------------------------------------------
    # Clases presentes en train
    # --------------------------------------------------
    print("\n[2] Clases encontradas en TRAIN")

    train_classes = Counter()

    for sample in dm.train_dataset:
        train_classes[sample["class_name"]] += 1

    for cls, count in sorted(train_classes.items()):
        print(f"{cls:<12} -> {count}")

    # --------------------------------------------------
    # Clases y defectos presentes en test
    # --------------------------------------------------
    print("\n[3] Clases y defectos encontrados en TEST")

    test_classes = Counter()
    defect_types = Counter()

    for sample in dm.test_dataset:
        test_classes[sample["class_name"]] += 1
        defect_types[sample["defect_type"]] += 1

    print("\nClases:")

    for cls, count in sorted(test_classes.items()):
        print(f"{cls:<12} -> {count}")

    print("\nDefectos:")

    for defect, count in sorted(defect_types.items()):
        print(f"{defect:<25} -> {count}")

    # --------------------------------------------------
    # Verificación de imágenes
    # --------------------------------------------------
    print("\n[4] Verificación de imágenes")

    sample = dm.test_dataset[0]

    print("Clase:", sample["class_name"])
    print("Defecto:", sample["defect_type"])
    print("Label:", sample["label"])
    print("Tensor shape:", sample["image"].shape)

    # --------------------------------------------------
    # Verificación de DataLoader
    # --------------------------------------------------
    print("\n[5] Verificación de batch")

    batch = next(iter(dm.train_dataloader()))

    print("Batch keys:", batch.keys())
    print("Batch imágenes:", batch["image"].shape)
    print("Batch labels:", batch["label"].shape)

    # --------------------------------------------------
    # Verificar imagen original
    # --------------------------------------------------
    print("\n[6] Imagen original")

    img_path = Path(sample["path"])

    img = Image.open(img_path)

    print("Archivo:", img_path.name)
    print("Tamaño original:", img.size)
    print("Modo:", img.mode)

    print("\nDataset cargado correctamente.")
    print("=" * 60)


if __name__ == "__main__":
    main()