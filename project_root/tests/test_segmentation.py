from torchvision.transforms import functional as F
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.segmentation_model import load_model, get_segmented_regions

def image_to_tensor(image_path):
    """Loading an image and convert it to a tensor."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)
    return image_tensor

def test_model():
    """Testing the Mask R-CNN model with a sample image."""
    model = load_model()
    image_path = 'data\\input_image\\lotus.jpg'
    image_tensor = image_to_tensor(image_path)
    
    masks, boxes, labels = get_segmented_regions(model, image_tensor)
    
    print("Masks shape:", masks.shape)
    print("Boxes shape:", boxes.shape)
    print("Labels shape:", labels.shape)
    
    # Printing the first mask, box, and label
    if masks.size > 0:
        print("First mask shape:", masks[0].shape)
    if boxes.size > 0:
        print("First box:", boxes[0])
    if labels.size > 0:
        print("First label:", labels[0])

if __name__ == "__main__":
    test_model()
