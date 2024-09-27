import cv2
import numpy as np
from PIL import Image

def read_image(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def resize_image(image, width, height):
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

def normalize_image(image, min_value=0, max_value=255):
    image = np.clip(image, min_value, max_value)
    return image.astype(np.uint8)

def preprocess_image(file_path, width=None, height=None):
    image = read_image(file_path)

    if width and height:
        image = resize_image(image, width, height)

    image = normalize_image(image)
    return image

def load_image(image_path):
    """Load an image from a file."""
    return np.array(Image.open(image_path))

def save_image(image_path, image_data):
    """Save an image to a file."""
    img = Image.fromarray(image_data)
    img.save(image_path)

def convert_to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)