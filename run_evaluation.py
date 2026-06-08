import os
import cv2
import numpy as np

from src.config import Config
from src.image_loader import ImageLoader
from src.metrics import MetricsCalculator
from src.preprocessing import Preprocessor
from src.detectors import FrangiDetector, RFDetector, UNetDetector


class Evaluator:
    def __init__(self):
        self.loader = ImageLoader(Config.DEFAULT_MAX_DIM)
        self.preprocessor = Preprocessor()
        self.metrics = MetricsCalculator()
        self.results = []

    def run(self):
        print("=" * 80)
        print("EWALUACJA PORÓWNAWCZA METOD DETEKCJI NACZYŃ KRWIONOŚNYCH")
        print("=" * 80)

        frangi = FrangiDetector(self.preprocessor)

        rf = RFDetector(self.preprocessor)
        rf_loaded = rf.load()
        print(f"Random Forest: {'wczytano' if rf_loaded else 'brak modelu'}")

        unet = UNetDetector(self.preprocessor)
        unet_loaded = unet.load()
        print(f"U-Net: {'wczytano' if unet_loaded else 'brak modelu'}")
        print()

        for img_name in Config.TEST_IMAGES:
            print(f"--- Obraz: {img_name} ---")
            img, fov, gt = self._load_triplet(img_name)

            if img is None or gt is None:
                print("Pominięto (brak pliku).")
                continue

            row = {"image": img_name}

            mask, eroded_fov, _, _ = frangi.predict(img, fov)
            gt_resized = self._resize_gt(gt, mask)
            m = self.metrics.calculate(mask, gt_resized, eroded_fov)
            row["Frangi"] = m
            self._print_metrics("Frangi", m)

            if rf_loaded:
                mask, fov_bin, _, _ = rf.predict(img, fov)
                gt_resized = self._resize_gt(gt, mask)
                m = self.metrics.calculate(mask, gt_resized, fov_bin)
                row["RF"] = m
                self._print_metrics("RF", m)

            if unet_loaded:
                mask, fov_bin, _, _ = unet.predict(img, fov, stride=64)
                gt_resized = self._resize_gt(gt, mask)
                m = self.metrics.calculate(mask, gt_resized, fov_bin)
                row["UNet"] = m
                self._print_metrics("U-Net", m)

            self.results.append(row)
            print()

        self._print_summary()

    def _load_triplet(self, image_name):
        img = self.loader.load(os.path.join(Config.IMG_DIR, image_name))
        fov = self.loader.load(Config.get_fov_path(image_name), grayscale=True)
        gt = self.loader.load(Config.get_gt_path(image_name), grayscale=True)
        return img, fov, gt

    @staticmethod
    def _resize_gt(gt, mask):
        if gt.shape != mask.shape:
            return cv2.resize(gt, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        return gt

    @staticmethod
    def _print_metrics(name, m):
        print(f"  {name:<8} Acc={m['Accuracy']:.4f}, Sens={m['Sensitivity']:.4f}, "
              f"Spec={m['Specificity']:.4f}, G-Mean={m['Geometric Mean']:.4f}")

    def _print_summary(self):
        print("=" * 80)
        print("TABELA PORÓWNAWCZA")
        print("=" * 80)

        header = (f"{'Obraz':<15} | {'Metoda':<10} | {'Accuracy':>10} | {'Sensitivity':>12} | "
                  f"{'Specificity':>12} | {'G-Mean':>10} | {'Arith Mean':>12} | "
                  f"{'TP':>8} | {'TN':>8} | {'FP':>8} | {'FN':>8}")
        print(header)
        print("-" * len(header))

        methods = [("Frangi", "Frangi"), ("RF", "RF"), ("UNet", "U-Net")]

        for row in self.results:
            for key, name in methods:
                if key not in row:
                    continue
                m = row[key]
                print(f"{row['image']:<15} | {name:<10} | {m['Accuracy']:>10.4f} | "
                      f"{m['Sensitivity']:>12.4f} | {m['Specificity']:>12.4f} | "
                      f"{m['Geometric Mean']:>10.4f} | {m['Arithmetic Mean']:>12.4f} | "
                      f"{m['TP']:>8} | {m['TN']:>8} | {m['FP']:>8} | {m['FN']:>8}")

        print()
        print("Ewaluacja zakończona.")


if __name__ == "__main__":
    Evaluator().run()
