# Wasserstoff-AI-ML-Task

## Image Processing and Object Recognition Project

### Overview

- This project involves multi-step image processing and object recognition, including image segmentation, object extraction, identification, and summarization. The final goal is to map and present the data using a Streamlit UI. The project leverages various machine learning models and tools to analyze and interpret images effectively.

#### Project Structure

- identification.py: Contains functions for object identification using Faster R-CNN and CLIP models. Includes functionalities for loading models, identifying objects, generating descriptions, and saving results to CSV.

- segmentation_model.py: Provides functions for image segmentation using the Mask R-CNN model. Includes model loading and extraction of segmented regions, masks, bounding boxes, and labels.

- identification_model.py: Provides function for identifying and extract objects using Faster R-CNN model and CLIP Model. Also generates description based on the object.
  
- summarization.py: Includes functions for summarizing text using the BART model. Provides functionalities for loading the model, generating summaries, and saving summaries to CSV.

- text_extraction.py: Contains functions for extracting text from images using Tesseract OCR. Includes methods for saving the extracted text to a CSV file.

- test_identification.py: Tests the object identification and description functionalities using sample images.

- test_segmentation.py: Tests the segmentation model with sample images to display masks, bounding boxes, and labels.

- test_summarization.py: Tests the summarization functionalities with sample data and saves summaries to a CSV file.

- test_text_extraction.py: Tests text extraction from images and saves the extracted data to a CSV file.

- data_mapping.py: Contains functions for creating a mapping of object attributes and annotating images with detected objects. Includes functionalities for saving mappings and annotations.

- postprocessing.py: Provides functions for extracting and saving segmented objects as separate images and initializing a database for metadata storage.

- preprocessing.py: Includes functions for loading and preprocessing images, converting them to tensors suitable for model input.

- visualization.py: Provides functions for visualizing segmented images, bounding boxes, and summary tables.

- streamlitapp.py: Implements a Streamlit application for interactive image segmentation, object identification, text extraction, and summarization. Provides a user interface for uploading images, displaying results, and saving summaries.

#### Results

- The pipeline effectively segments and identifies objects across diverse images. It efficiently processes and summarizes the extracted text and data, delivering a comprehensive and clear analysis of the input image.

#### Future Scope

- Enhance model accuracy and performance by refining algorithms and utilizing advanced techniques. Additionally, expand the pipeline to support a broader range of image formats and types, including raw and high-resolution images, to ensure greater versatility and robustness in image analysis.
