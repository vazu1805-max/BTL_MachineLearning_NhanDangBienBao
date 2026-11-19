import numpy as np
from skimage.feature import hog

def pixel_features(X):
    return X.astype(np.float32) / 255.0

def hog_features(X, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)):
    rows = []
    for flat in X:
        image = flat.reshape(32, 32).astype(np.float32) / 255.0
        descriptor = hog(
            image, orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys"
        )
        rows.append(descriptor)
    return np.asarray(rows, dtype=np.float32)

def transform_features(X, method):
    if method == "pixel":
        return pixel_features(X)
    if method == "hog":
        return hog_features(X)
    raise ValueError(f"Phương pháp đặc trưng không hợp lệ: {method}")
