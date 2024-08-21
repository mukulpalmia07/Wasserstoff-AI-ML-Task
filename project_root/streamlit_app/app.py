import streamlit as st
import numpy as np
from PIL import Image
import sys
import os

# Add the parent directory to the path for importing modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.preprocessing import load_image, preprocess_image
from models.segmentation_model import load_model, get_segmented_regions
from models.identification_model import load_clip_model, identify_objects, generate_description
from models.text_extraction import extract_text_tesseract
from models.summarization import save_summaries_to_csv
from utils.data_mapping import create_mapping, create_summary_table, annotate_image

# Load models
segmentation_model = load_model()
clip_model, clip_processor = load_clip_model()

def main():
    st.title("Image Segmentation and Object Identification")

    # File Upload
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Load and preprocess image
        image = load_image(uploaded_file)  # Use the load_image function to load the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Segment the image
        image_tensor = preprocess_image(image)
        masks, boxes, labels = get_segmented_regions(segmentation_model, image_tensor)

        # Create a dictionary to hold object details
        objects_data = {}
        
        for i in range(masks.shape[0]):
            # Extract each object image using the mask
            mask = masks[i, 0] > 0.5
            object_id = f"object_{i+1}"

            # Crop the object from the original image
            coords = np.argwhere(mask)
            if coords.size == 0:  # Check if any pixels were found
                continue
            y0, x0 = coords.min(axis=0)
            y1, x1 = coords.max(axis=0)
            cropped_object = image[y0:y1+1, x0:x1+1]

            # Convert cropped object to an image for display
            object_image = Image.fromarray(cropped_object)
            st.image(object_image, caption=f'Segmented {object_id}', use_column_width=True)

            # Identify the object and extract information
            labels = identify_objects(clip_model, clip_processor, object_image)  # Pass clip_processor
            description = generate_description(clip_model, clip_processor, object_image)
            extracted_text = extract_text_tesseract(object_image)

            # Store object data in a structured format
            objects_data[object_id] = {
                "description": description,
                "extracted_text": extracted_text,
            }

        # Create summary table and mapping of object attributes
        summary_table = create_summary_table(objects_data)
        mapping = create_mapping(objects_data)

        # Annotate original image with object details
        annotated_image = annotate_image(image, objects_data)  # Pass the actual image instead of path
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)

        # Display summary table
        st.write("Summary Table:")
        st.dataframe(summary_table)

        # Option to save summaries to CSV
        if st.button("Save Summaries to CSV"):
            save_summaries_to_csv("object_summaries.csv", summary_table)
            st.success("Summaries saved to 'object_summaries.csv'.")

if __name__ == "__main__":
    main()
