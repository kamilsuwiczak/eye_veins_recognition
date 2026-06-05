import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
from imblearn.metrics import specificity_score, geometric_mean_score


class MetricsCalculator:
    @staticmethod
    def calculate(pred_mask, true_mask, fov_mask=None):
        pred = (pred_mask > 127).astype(np.uint8)
        true = (true_mask > 127).astype(np.uint8)

        if fov_mask is not None:
            valid = fov_mask > 127
            y_pred = pred[valid]
            y_true = true[valid]
        else:
            y_pred = pred.flatten()
            y_true = true.flatten()

        if len(y_true) == 0:
            return None

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = accuracy_score(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred, zero_division=0)
        specificity = specificity_score(y_true, y_pred)
        g_mean = geometric_mean_score(y_true, y_pred)
        arithmetic_mean = (sensitivity + specificity) / 2.0

        return {
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "Accuracy": accuracy,
            "Sensitivity": sensitivity,
            "Specificity": specificity,
            "Arithmetic Mean": arithmetic_mean,
            "Geometric Mean": g_mean,
        }
