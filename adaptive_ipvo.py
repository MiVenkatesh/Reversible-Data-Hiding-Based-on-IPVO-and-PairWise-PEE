import numpy as np

def calculate_histogram(image):
    height, width = image.shape
    histogram = np.zeros(256, dtype=np.int32)

    for y in range(height):
        for x in range(width):
            histogram[image[y, x]] += 1

    return histogram

def find_optimal_threshold(histogram):
    total_pixels = np.sum(histogram)
    background_sum = 0
    background_pixels = 0

    max_variance = 0
    optimal_threshold = 0

    for i in range(256):
        background_pixels += histogram[i]
        foreground_pixels = total_pixels - background_pixels

        if background_pixels == 0 or foreground_pixels == 0:
            continue

        background_sum += i * histogram[i]
        foreground_sum = total_pixels - background_sum

        background_mean = background_sum / background_pixels
        foreground_mean = foreground_sum / foreground_pixels

        between_class_variance = background_pixels * foreground_pixels * (background_mean - foreground_mean) ** 2

        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = i

    return optimal_threshold

def split_image(image, threshold):
    height, width = image.shape
    binary_image = np.zeros((height, width), dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if image[y, x] >= threshold:
                binary_image[y, x] = 1

    return binary_image

def ipvo(image):
    histogram = calculate_histogram(image)
    threshold = find_optimal_threshold(histogram)
    binary_image = split_image(image, threshold)

    reordered_image = np.zeros_like(image)
    height, width = image.shape

    low_pointer = 0
    high_pointer = height * width - 1

    for y in range(height):
        for x in range(width):
            if binary_image[y, x] == 0:
                reordered_image.ravel()[low_pointer] = image[y, x]
                low_pointer += 1
            else:
                reordered_image.ravel()[high_pointer] = image[y, x]
                high_pointer -= 1

    return reordered_image, binary_image

def generate_histogram(image):
    """Generate a histogram from a grayscale image."""
    histogram = np.zeros(256, dtype=int)
    for row in image:
        for value in row:
            histogram[value] += 1
    return histogram