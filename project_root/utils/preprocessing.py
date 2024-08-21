import cv2
from torchvision.transforms import functional as F
import numpy as np

def load_image(image_path):
    """Loading an image from a file path and convert it to RGB."""

    file_bytes = np.asarray(bytearray(image_path.read()), dtype = np.uint8)
    image = cv2.imdecode(file_bytes, 1)  # Decoding image from bytes
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def preprocess_image(image_rgb):
    """Converting the RGB image to a tensor and normalize it."""
    
    image_tensor = F.to_tensor(image_rgb)
    return image_tensor
