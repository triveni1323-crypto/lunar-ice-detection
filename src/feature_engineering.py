import numpy as np
from skimage.feature import canny
from scipy import ndimage


def texture_features(image):
    laplacian = ndimage.laplace(image)
    edges = canny(image)
    variance = ndimage.generic_filter(image, np.var, size=3)
    return {
        "laplacian": laplacian,
        "edges": edges,
        "variance": variance,
    }


def radar_features(sar_image):
    magnitude = np.abs(sar_image)
    smooth = ndimage.gaussian_filter(magnitude, sigma=1)
    return {
        "magnitude": magnitude,
        "smooth": smooth,
    }


def create_feature_stack(*feature_maps):
    stack = []
    for feature in feature_maps:
        stack.append(np.asarray(feature, dtype=np.float32))
    return np.stack(stack, axis=-1)
