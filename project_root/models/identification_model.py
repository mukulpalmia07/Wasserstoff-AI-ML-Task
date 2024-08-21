import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from transformers import CLIPProcessor, CLIPModel
import csv

def load_detection_model():
    """Loading the pre-trained Faster R-CNN model."""
    model = fasterrcnn_resnet50_fpn(pretrained = True)
    model.eval()
    return model

def identify_objects(model, processor, object_image):
    """Identifying objects in the image using CLIP."""

    # Converting the image to the correct format
    object_image = object_image.convert("RGB")  # Ensure the image is in RGB format
    
    # Preparing inputs for the model
    inputs = processor(text = ["a photo of a {}.".format(label) for label in ['object']], 
                       images = object_image, 
                       return_tensors = "pt", 
                       padding = True)

    with torch.no_grad():
        outputs = model(**inputs)

    # Getting the predicted labels
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    predicted_label = logits_per_image.argmax(dim = 1).cpu().numpy()

    return predicted_label

def load_clip_model():
    """Loading the pre-trained CLIP model."""

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def generate_description(model, processor, object_image):
    """Generating a description for the identified object using CLIP."""
    
    # Ensuring the image is in RGB format
    object_image = object_image.convert("RGB")

    # Creating a list of descriptions or labels for input
    candidate_descriptions = ["a photo of an object"]

    # Preparing inputs for the model
    inputs = processor(text = candidate_descriptions, images = object_image, return_tensors = "pt", padding = True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted description
    logits_per_image = outputs.logits_per_image  # Image-text similarity score
    predicted_index = logits_per_image.argmax(dim = 1).item()  # Get index of highest score

    # Return the corresponding description
    return candidate_descriptions[predicted_index]

def save_identifications_to_csv(output_path, data):
    """Save identified objects and descriptions to a CSV file."""
    with open(output_path, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["Object ID", "Labels", "Description"])
        for entry in data:
            writer.writerow(entry)
