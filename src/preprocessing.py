import cv2
import numpy as np
from skimage.filters import frangi


class Preprocessor:
    def __init__(self, clip_limit=2.0, tile_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

    def extract_green_clahe(self, img_rgb):
        _, g, _ = cv2.split(img_rgb)
        return self.clahe.apply(g)

    def binarize_fov(self, fov_mask):
        if fov_mask is None:
            return None
        _, fov_bin = cv2.threshold(fov_mask, 127, 255, cv2.THRESH_BINARY)
        return fov_bin

    def apply_fov(self, image, fov_bin):
        if fov_bin is None:
            return image
        return cv2.bitwise_and(image, image, mask=fov_bin)

    def erode_fov(self, fov_bin, margin):
        if fov_bin is None or margin <= 0:
            return fov_bin
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        return cv2.erode(fov_bin, kernel, iterations=1)

    def compute_frangi(self, gray_image, sigmas):
        vessels = frangi(gray_image, sigmas=np.arange(*sigmas), black_ridges=True)
        return cv2.normalize(vessels, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def extract_features(self, img_rgb, fov_mask, frangi_sigmas):
        g_clahe = self.extract_green_clahe(img_rgb)
        fov_bin = self.binarize_fov(fov_mask)

        if fov_bin is None:
            fov_bin = np.ones_like(g_clahe) * 255

        g_masked = self.apply_fov(g_clahe, fov_bin)

        h, w = g_clahe.shape
        scale = max(h, w) / 800.0
        k_val = int(5 * scale)
        k_size = k_val if k_val % 2 == 1 else k_val + 1
        k_size = max(3, k_size)

        vessels = frangi(g_masked, sigmas=np.arange(*frangi_sigmas), black_ridges=True)
        f1 = cv2.normalize(vessels, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        f2 = cv2.morphologyEx(g_clahe, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)

        sx = cv2.Sobel(g_clahe, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(g_clahe, cv2.CV_32F, 0, 1, ksize=3)
        f3 = cv2.magnitude(sx, sy)
        f3 = cv2.normalize(f3, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

        gf = g_clahe.astype(np.float32)
        mean_val = cv2.blur(gf, (k_size, k_size))
        mean_sq = cv2.blur(gf ** 2, (k_size, k_size))
        f4 = np.sqrt(np.maximum(0, mean_sq - mean_val ** 2))

        features = np.dstack([f1, f2, f3, f4])
        return g_clahe, features, fov_bin, f1.astype(np.uint8)

    @staticmethod
    def prepare_ml_data(features, expert_mask, fov_mask, samples_per_class=5000):
        X_flat = features.reshape(-1, features.shape[-1])
        y_flat = (expert_mask.flatten() > 127).astype(np.uint8)
        fov_flat = fov_mask.flatten() > 127

        valid_indices = np.where(fov_flat)[0]
        X_valid = X_flat[valid_indices]
        y_valid = y_flat[valid_indices]

        pos_indices = np.where(y_valid == 1)[0]
        neg_indices = np.where(y_valid == 0)[0]

        n_pos = min(samples_per_class, len(pos_indices))
        n_neg = min(samples_per_class, len(neg_indices))

        if n_pos == 0 or n_neg == 0:
            return None, None

        np.random.seed(42)
        sampled_pos = np.random.choice(pos_indices, n_pos, replace=False)
        sampled_neg = np.random.choice(neg_indices, n_neg, replace=False)
        sampled = np.concatenate([sampled_pos, sampled_neg])

        return X_valid[sampled], y_valid[sampled]
