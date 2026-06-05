import cv2
import os


class ImageLoader:
    def __init__(self, max_dim=800):
        self.max_dim = max_dim

    def load(self, path, grayscale=False):
        if not os.path.exists(path):
            return None

        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img is None:
            return None

        h, w = img.shape[:2]
        if max(h, w) > self.max_dim:
            scale = self.max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            interp = cv2.INTER_NEAREST if grayscale else cv2.INTER_AREA
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)

        return img
