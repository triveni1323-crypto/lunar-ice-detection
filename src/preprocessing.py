import numpy as np
from skimage.transform import resize


def normalize(data):
    data = np.asarray(data, dtype=np.float32)
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    if max_val == min_val:
        return np.zeros_like(data)
    return (data - min_val) / (max_val - min_val)


def handle_missing(data):
    data = np.asarray(data, dtype=np.float32)
    if np.isnan(data).any():
        data = np.nan_to_num(data, nan=np.nanmean(data))
    return data


def resize_data(data, target_shape):
    return resize(data, target_shape, anti_aliasing=True)


def standardize(data):
    data = np.asarray(data, dtype=np.float32)
    mean = np.mean(data)
    std = np.std(data)
    if std == 0:
        return data - mean
    return (data - mean) / std
