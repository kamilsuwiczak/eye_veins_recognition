import os


class Config:
    IMG_DIR = "data/images/"
    FOV_DIR = "data/mask/"
    GT_DIR = "data/manual1/"
    RF_MODEL_PATH = "saved_models/rf_model.pkl"
    UNET_MODEL_PATH = "saved_models/unet_model.pth"

    DEFAULT_MAX_DIM = 800
    DEFAULT_FRANGI_SIGMAS = (1.5, 4.0, 0.5)
    DEFAULT_FRANGI_THRESHOLD = 15
    DEFAULT_FOV_MARGIN = 15
    DEFAULT_UNET_THRESHOLD = 0.5

    RF_N_ESTIMATORS = 50
    RF_MAX_DEPTH = 8
    RF_RANDOM_STATE = 42
    RF_SAMPLES_PER_CLASS = 5000

    UNET_CROP_SIZE = 256
    UNET_CROPS_PER_IMG = 30
    UNET_BATCH_SIZE = 8
    UNET_EPOCHS = 20
    UNET_LR = 0.001
    UNET_STRIDE = 128  # sliding-window step; smaller = better quality, slower

    TEST_IMAGES = [
        "11_dr.JPG",
        "11_g.jpg",
        "11_h.jpg",
        "12_dr.JPG",
        "12_g.jpg",
        "12_h.jpg",
        "13_dr.JPG",
        "13_g.jpg",
        "13_h.jpg",
        "14_dr.JPG",
        "14_g.jpg",
        "14_h.jpg",
        "15_dr.JPG",
        "15_g.jpg",
        "15_h.jpg", ]
    TRAIN_PREFIXES = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]

    @classmethod
    def get_image_files(cls):
        if not os.path.exists(cls.IMG_DIR):
            return []
        extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        arr = [f for f in os.listdir(cls.IMG_DIR) if f.lower().endswith(extensions)]
        return sorted([f for f in arr if f.split("_")[0] not in cls.TRAIN_PREFIXES])

    @classmethod
    def get_gt_path(cls, image_name):
        base = os.path.splitext(image_name)[0]
        path = os.path.join(cls.GT_DIR, image_name)
        if not os.path.exists(path):
            path = os.path.join(cls.GT_DIR, f"{base}.tif")
        return path

    @classmethod
    def get_fov_path(cls, image_name):
        base = os.path.splitext(image_name)[0]
        return os.path.join(cls.FOV_DIR, f"{base}_mask.tif")
