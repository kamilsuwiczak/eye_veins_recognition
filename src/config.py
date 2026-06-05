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

    TEST_IMAGES = ["04_g.jpg", "04_h.jpg", "04_dr.JPG", "05_g.jpg", "05_h.jpg"]
    TRAIN_PREFIXES = ["01", "02", "03"]

    @classmethod
    def get_image_files(cls):
        if not os.path.exists(cls.IMG_DIR):
            return []
        extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        return sorted([f for f in os.listdir(cls.IMG_DIR) if f.lower().endswith(extensions)])

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
