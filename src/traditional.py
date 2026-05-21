import numpy as np
import cv2
from skimage.feature import hog, local_binary_pattern
from PIL import Image
from pathlib import Path


def extract_hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
    return hog(gray, orientations=orientations,
               pixels_per_cell=pixels_per_cell,
               cells_per_block=cells_per_block,
               feature_vector=True)


def extract_lbp(gray, P=24, R=3, n_bins=26):
    lbp = local_binary_pattern(gray, P=P, R=R, method="uniform")
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    return hist


def extract_color_histograms(img_rgb, bins_per_channel=16):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    feats = []
    for ch in range(3):
        hist = cv2.calcHist([hsv], [ch], None, [bins_per_channel], [0, 256])
        hist = hist.flatten() / (hist.sum() + 1e-8)
        feats.append(hist)
    return np.concatenate(feats)


def extract_intensity_stats(img_rgb):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    from scipy.stats import skew, kurtosis
    stats = [gray.mean(), gray.std(), skew(gray.flat), kurtosis(gray.flat)]
    for ch in range(3):
        ch_data = img_rgb[:, :, ch].astype(np.float32)
        stats.extend([ch_data.mean(), ch_data.std()])
    return np.array(stats)


def extract_traditional_features(img_path):
    img = Image.open(img_path).convert("RGB")
    img_rgb = np.array(img)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    hog_feat = extract_hog(gray)
    lbp_feat = extract_lbp(gray)
    color_feat = extract_color_histograms(img_rgb)
    stats_feat = extract_intensity_stats(img_rgb)

    return np.concatenate([hog_feat, lbp_feat, color_feat, stats_feat])


def extract_traditional_batch(image_paths):
    features = []
    for p in image_paths:
        feat = extract_traditional_features(p)
        features.append(feat)
    return np.array(features)
