"""Same feature pipeline as the notebook — used by the Flask app."""
import cv2
import numpy as np


def preprocess(img_bgr, img_size):
    """BGR (cv2) -> grayscale -> resize -> normalize to [0,1]."""
    if img_bgr.ndim == 3:
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        img = img_bgr
    img = cv2.resize(img, (img_size, img_size))
    return img.astype(np.float32) / 255.0


def histo(img):
    img_u8 = (img * 255).astype(np.uint8)
    h, _   = np.histogram(img_u8.ravel(), bins=256, range=(0, 256))
    return h.astype(np.float64)


def extract_texture(img):
    variance = np.var(img)
    energy   = np.sum(img ** 2) / img.size
    h, _     = np.histogram((img * 255).astype(np.uint8).ravel(),
                            bins=256, range=(0, 256))
    p        = h / (h.sum() + 1e-10)
    entropy  = -np.sum(p * np.log2(p + 1e-10))
    contrast = float(np.mean(np.abs(img[:, 1:] - img[:, :-1])))
    homogen  = 1.0 / (1.0 + np.std(img))
    return np.array([variance, energy, entropy, contrast, homogen])


def extract_edges(img):
    img_u8 = (img * 255).astype(np.uint8)
    lap    = cv2.Laplacian(img_u8, cv2.CV_64F)
    lap_u8 = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    h, _   = np.histogram(lap_u8.ravel(), bins=256, range=(0, 256))
    return h.astype(np.float64)


def extract_features(img):
    return np.concatenate([histo(img), extract_texture(img), extract_edges(img)])
