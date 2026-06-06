import os
import cv2
import numpy as np
import joblib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import RandomForestClassifier

from src.config import Config
from src.preprocessing import Preprocessor
from src.unet_model import MiniUNet, BCEDiceLoss


class FrangiDetector:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or Preprocessor()

    def predict(self, img_rgb, fov_mask, sigmas=None, threshold=None, fov_margin=None):
        sigmas = sigmas or Config.DEFAULT_FRANGI_SIGMAS
        threshold = threshold if threshold is not None else Config.DEFAULT_FRANGI_THRESHOLD
        fov_margin = fov_margin if fov_margin is not None else Config.DEFAULT_FOV_MARGIN

        g_clahe = self.preprocessor.extract_green_clahe(img_rgb)
        fov_bin = self.preprocessor.binarize_fov(fov_mask)

        vessels_norm = self.preprocessor.compute_frangi(g_clahe, sigmas)
        frangi_raw = vessels_norm.copy()

        eroded_fov = self.preprocessor.erode_fov(fov_bin, fov_margin)
        if eroded_fov is not None:
            vessels_norm = cv2.bitwise_and(vessels_norm, vessels_norm, mask=eroded_fov)

        _, mask = cv2.threshold(vessels_norm, threshold, 255, cv2.THRESH_BINARY)

        return mask, eroded_fov, g_clahe, frangi_raw


class RFDetector:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or Preprocessor()
        self.model = None

    def create_model(self, max_depth=None):
        max_depth = max_depth or Config.RF_MAX_DEPTH
        self.model = RandomForestClassifier(
            n_estimators=Config.RF_N_ESTIMATORS,
            max_depth=max_depth,
            random_state=Config.RF_RANDOM_STATE,
            n_jobs=-1,
        )
        return self.model

    def train(self, img_rgb, fov_mask, expert_mask, sigmas=None, samples_per_class=None):
        sigmas = sigmas or Config.DEFAULT_FRANGI_SIGMAS
        samples_per_class = samples_per_class or Config.RF_SAMPLES_PER_CLASS

        _, features, fov_bin, _ = self.preprocessor.extract_features(img_rgb, fov_mask, sigmas)
        X_train, y_train = Preprocessor.prepare_ml_data(features, expert_mask, fov_bin, samples_per_class)

        if X_train is None:
            return False

        if self.model is None:
            self.create_model()

        self.model.fit(X_train, y_train)
        return True

    def predict(self, img_rgb, fov_mask, sigmas=None):
        sigmas = sigmas or Config.DEFAULT_FRANGI_SIGMAS

        g_clahe, features, fov_bin, frangi_raw = self.preprocessor.extract_features(img_rgb, fov_mask, sigmas)

        X_flat = features.reshape(-1, features.shape[-1])
        valid_idx = np.where(fov_bin.flatten() > 127)[0]

        y_pred = self.model.predict(X_flat[valid_idx])

        mask_flat = np.zeros(X_flat.shape[0], dtype=np.uint8)
        mask_flat[valid_idx] = y_pred * 255
        mask = mask_flat.reshape(img_rgb.shape[:2])

        return mask, fov_bin, g_clahe, frangi_raw

    def save(self, path=None):
        path = path or Config.RF_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path=None):
        path = path or Config.RF_MODEL_PATH
        if not os.path.exists(path):
            return False
        self.model = joblib.load(path)
        return True


# ---------------------------------------------------------------------------
# Augmented patch datasets
# ---------------------------------------------------------------------------

def _augment(img: np.ndarray, mask: np.ndarray):
    """Random flips + 90° rotations applied identically to image and mask."""
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    mask = np.rot90(mask, k)
    if np.random.random() > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    if np.random.random() > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    return np.ascontiguousarray(img), np.ascontiguousarray(mask)


class PatchDataset(Dataset):
    def __init__(self, g_clahe, expert_mask, fov_bin, crop_size=256, num_crops=50,
                 augment=True):
        self.g_clahe = g_clahe
        self.expert_mask = (expert_mask > 127).astype(np.uint8)
        self.fov_bin = fov_bin
        self.crop_size = crop_size
        self.num_crops = num_crops
        self.augment = augment

    def __len__(self):
        return self.num_crops

    def __getitem__(self, idx):
        h, w = self.g_clahe.shape
        cs = self.crop_size

        for _ in range(100):
            y = np.random.randint(0, max(1, h - cs))
            x = np.random.randint(0, max(1, w - cs))
            crop_img = self.g_clahe[y: y + cs, x: x + cs]
            crop_mask = self.expert_mask[y: y + cs, x: x + cs]
            # Prefer patches that contain some FOV and some vessels
            fov_crop = self.fov_bin[y: y + cs, x: x + cs] if self.fov_bin is not None else None
            fov_ratio = np.mean(fov_crop > 127) if fov_crop is not None else 1.0
            vessel_ratio = np.mean(crop_mask)
            if fov_ratio > 0.3 and vessel_ratio > 0.01:
                break

        if self.augment:
            crop_img, crop_mask = _augment(crop_img, crop_mask)

        crop_img = crop_img.astype(np.float32) / 255.0
        crop_mask = crop_mask.astype(np.float32)

        return torch.tensor(crop_img).unsqueeze(0), torch.tensor(crop_mask).unsqueeze(0)


class MultiImageDataset(Dataset):
    def __init__(self, images, masks, crop_size=256, crops_per_img=30, augment=True):
        self.images = images
        self.masks = masks
        self.crop_size = crop_size
        self.crops_per_img = crops_per_img
        self.augment = augment

    def __len__(self):
        return len(self.images) * self.crops_per_img

    def __getitem__(self, idx):
        img_idx = idx // self.crops_per_img
        img = self.images[img_idx]
        mask = self.masks[img_idx]

        h, w = img.shape
        cs = self.crop_size

        for _ in range(100):
            y = np.random.randint(0, max(1, h - cs))
            x = np.random.randint(0, max(1, w - cs))
            crop_img = img[y: y + cs, x: x + cs]
            crop_mask = mask[y: y + cs, x: x + cs]
            if np.mean(crop_img > 0) > 0.3 and np.mean(crop_mask) > 0.01:
                break

        if self.augment:
            crop_img, crop_mask = _augment(crop_img, crop_mask)

        crop_img = crop_img.astype(np.float32) / 255.0
        crop_mask = crop_mask.astype(np.float32)

        return torch.tensor(crop_img).unsqueeze(0), torch.tensor(crop_mask).unsqueeze(0)


# ---------------------------------------------------------------------------
# U-Net detector
# ---------------------------------------------------------------------------

class UNetDetector:
    def __init__(self, preprocessor=None):
        self.preprocessor = preprocessor or Preprocessor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MiniUNet().to(self.device)

    # ------------------------------------------------------------------
    # Sliding-window inference
    # ------------------------------------------------------------------

    def _predict_sliding_window(self, img_norm: np.ndarray,
                                patch_size: int = 256,
                                stride: int = 128) -> np.ndarray:
        """Run inference on overlapping patches and average probabilities.

        This avoids border artefacts and gives substantially better results
        on large images where the model was trained on 256×256 crops.
        """
        h, w = img_norm.shape

        # Pad so the image is at least one patch large
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        if pad_h > 0 or pad_w > 0:
            img_norm = np.pad(img_norm, ((0, pad_h), (0, pad_w)), mode="reflect")

        H, W = img_norm.shape
        prob_map = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)

        # Build grid of top-left corners
        ys = list(range(0, H - patch_size + 1, stride))
        xs = list(range(0, W - patch_size + 1, stride))
        # Always include the last position so we cover the full image
        if not ys or ys[-1] + patch_size < H:
            ys.append(max(0, H - patch_size))
        if not xs or xs[-1] + patch_size < W:
            xs.append(max(0, W - patch_size))

        self.model.eval()
        with torch.no_grad():
            for y in ys:
                for x in xs:
                    patch = img_norm[y: y + patch_size, x: x + patch_size]
                    tensor = (
                        torch.tensor(patch, dtype=torch.float32)
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    pred = self.model(tensor).squeeze().cpu().numpy()
                    prob_map[y: y + patch_size, x: x + patch_size] += pred
                    count_map[y: y + patch_size, x: x + patch_size] += 1.0

        count_map = np.maximum(count_map, 1.0)
        prob_map /= count_map
        return prob_map[:h, :w]

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def train_on_image(self, g_clahe, expert_mask, fov_bin, epochs=20, num_crops=40,
                       batch_size=4, progress_callback=None):
        dataset = PatchDataset(g_clahe, expert_mask, fov_bin,
                               Config.UNET_CROP_SIZE, num_crops, augment=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.UNET_LR,
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = BCEDiceLoss()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for images, masks in dataloader:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)

            if progress_callback:
                progress_callback(epoch, epochs, avg_loss)

    def train_on_dataset(self, images, masks, epochs=20, crops_per_img=30, batch_size=8):
        dataset = MultiImageDataset(images, masks, Config.UNET_CROP_SIZE,
                                    crops_per_img, augment=True)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.UNET_LR,
                                     weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = BCEDiceLoss()

        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            n_batches = 0

            for images_batch, masks_batch in dataloader:
                images_batch = images_batch.to(self.device)
                masks_batch = masks_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images_batch)
                loss = criterion(outputs, masks_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"Epoch {epoch + 1:02d}/{epochs:02d} | Loss: {avg_loss:.4f} "
                  f"| LR: {scheduler.get_last_lr()[0]:.6f}")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, img_rgb, fov_mask, threshold=None,
                stride: int | None = None):
        """Segment vessels using sliding-window inference.

        Parameters
        ----------
        stride:
            Step between overlapping patches (default: Config.UNET_STRIDE).
            Smaller stride → more overlap → better quality but slower.
        """
        threshold = threshold if threshold is not None else Config.DEFAULT_UNET_THRESHOLD
        stride = stride if stride is not None else Config.UNET_STRIDE

        g_clahe = self.preprocessor.extract_green_clahe(img_rgb)
        fov_bin = self.preprocessor.binarize_fov(fov_mask)

        if fov_bin is not None:
            g_clahe_masked = self.preprocessor.apply_fov(g_clahe, fov_bin)
        else:
            g_clahe_masked = g_clahe
            fov_bin = np.ones_like(g_clahe) * 255

        img_norm = g_clahe_masked.astype(np.float32) / 255.0

        # Sliding-window inference (much better than single-pass on large images)
        prob_map = self._predict_sliding_window(img_norm,
                                                patch_size=Config.UNET_CROP_SIZE,
                                                stride=stride)

        # Apply FOV mask to probability map
        prob_map *= (fov_bin > 127).astype(np.float32)

        # Binary mask
        mask = (prob_map >= threshold).astype(np.uint8) * 255

        # Post-processing: morphological opening removes salt noise,
        # closing fills small gaps inside vessels
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        return mask, fov_bin, g_clahe, prob_map

    def save(self, path=None):
        path = path or Config.UNET_MODEL_PATH
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path=None):
        path = path or Config.UNET_MODEL_PATH
        if not os.path.exists(path):
            return False
        self.model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        return True
