import os
import cv2
import numpy as np

from src.config import Config
from src.image_loader import ImageLoader
from src.preprocessing import Preprocessor
from src.detectors import UNetDetector


def main():
    loader = ImageLoader(Config.DEFAULT_MAX_DIM)
    preprocessor = Preprocessor()

    all_files = Config.get_image_files()
    train_names = [f for f in all_files if f.split("_")[0] in Config.TRAIN_PREFIXES]

    print(f"Obrazy treningowe: {train_names}")

    images = []
    masks = []

    for name in train_names:
        img = loader.load(os.path.join(Config.IMG_DIR, name))
        fov = loader.load(Config.get_fov_path(name), grayscale=True)
        gt = loader.load(Config.get_gt_path(name), grayscale=True)

        if img is None or gt is None:
            print(f"Pominięto {name}")
            continue

        g_clahe = preprocessor.extract_green_clahe(img)
        fov_bin = preprocessor.binarize_fov(fov)
        g_clahe = preprocessor.apply_fov(g_clahe, fov_bin)

        images.append(g_clahe)
        masks.append((gt > 127).astype(np.uint8))

    print(f"Załadowano {len(images)} obrazów.")

    detector = UNetDetector(preprocessor)
    detector.train_on_dataset(
        images, masks,
        epochs=Config.UNET_EPOCHS,
        crops_per_img=Config.UNET_CROPS_PER_IMG,
        batch_size=Config.UNET_BATCH_SIZE,
    )
    detector.save()
    print(f"Model zapisany do {Config.UNET_MODEL_PATH}")


if __name__ == "__main__":
    main()
