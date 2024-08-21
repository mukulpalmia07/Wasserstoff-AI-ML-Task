import json
import cv2
from PIL import Image

def create_mapping(objects_data):
    """Creating a mapping of object attributes."""
    mapping = {}

    for obj_id, attributes in objects_data.items():
        mapping[obj_id] = {
            "description": attributes.get("description", "N/A"),
            "extracted_text": attributes.get("extracted_text", "N/A"),
            "summary": attributes.get("summary", "No summary available"),
        }
    
    return mapping

objects_data = {"object_1": {"description": "A sample description", "text": "Extracted text", "summary": "Summary of the text"}}

mapping = create_mapping(objects_data)

# Saving to JSON file
with open("data_mapping.json", "w") as json_file:
    json.dump(mapping, json_file, indent = 4)

def create_summary_table(objects_data):
    """Creating a summary table from the extracted object data."""
    
    summary_table = []
    
    for obj_id, attributes in objects_data.items():
        summary_table.append({
            "Object ID": obj_id,
            "Description": attributes.get("description", "N/A"),  # Using get() to avoid KeyError
            "Extracted Text": attributes.get("extracted_text", "N/A"),
            "Summary": attributes.get("summary", "No summary available"), 
        })
    
    return summary_table

import numpy as np

def annotate_image(image, objects_data):
    """Annotating the image with detected objects."""

    # Converting PIL Image to OpenCV format if using a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR

    # Proceeding with annotation using OpenCV
    for obj_id, attributes in objects_data.items():
        # Example of annotating text on the image
        cv2.putText(image, f"{obj_id}: {attributes.get('description', 'N/A')}",
                    (10, 30 + 30 * list(objects_data.keys()).index(obj_id)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image