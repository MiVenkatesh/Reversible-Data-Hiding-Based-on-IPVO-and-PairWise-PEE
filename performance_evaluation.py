import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_eper(original_image, embedded_image):
    total_pixels = original_image.size
    false_predictions = np.count_nonzero(original_image - embedded_image)
    eper = false_predictions / total_pixels
    return eper

def payload_capacity(data):
    return len(data)

def embedding_efficiency(embedded_image, original_image):
    embedded_image = np.reshape(embedded_image, original_image.shape)  # Reshape the embedded_image
    diff = np.abs(embedded_image.astype(np.int64) - original_image.astype(np.int64))
    return np.sum(diff) / np.sum(np.abs(original_image))

def image_quality(original_image, restored_image):
    original_image = skimage.transform.resize(original_image, restored_image.shape)
    return psnr(original_image, restored_image)


def evaluate_performance(original_image, embedded_image, restored_image, data):
    capacity = payload_capacity(data)
    efficiency = embedding_efficiency(original_image, embedded_image)
    quality = image_quality(original_image, restored_image)
    eper = calculate_eper(original_image, embedded_image)

    return capacity, efficiency, quality
