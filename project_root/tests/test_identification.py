from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.identification_model import (
    load_detection_model, 
    identify_objects, 
    load_clip_model, 
    generate_description, 
    save_identifications_to_csv
)

def test_object_identification_and_description():
    try:
        # Loading models
        detection_model = load_detection_model()
        clip_model, clip_processor = load_clip_model()
        
        # Loading a sample image
        image_path = 'data\\input_image\\lotus.jpg'
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"The image file at {image_path} was not found.")
        
        object_image = Image.open(image_path)
        
        # Identifying objects using Faster R-CNN
        object_labels = identify_objects(clip_model, clip_processor, object_image)
        
        # Generating a description using CLIP
        description = generate_description(clip_model, clip_processor, object_image)
        
        # Preparing descriptions data
        descriptions = [(i, description) for i in object_labels]
        
        # Saving results to CSV
        output_path = 'data\\output\\test_identification.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok = True)  # Ensure output directory exists
        save_identifications_to_csv(output_path, descriptions)

        print(f"Identifications and descriptions have been saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_object_identification_and_description()
