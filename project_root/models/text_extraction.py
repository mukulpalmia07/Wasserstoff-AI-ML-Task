import pytesseract
from PIL import Image
import csv
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_tesseract(object_image):
    """Extracting text from the given PIL Image using Tesseract OCR."""

    # If the input is a PIL Image, convert it to a format suitable for Tesseract
    if isinstance(object_image, Image.Image):
        # Converting the image to RGB if it isn't already
        object_image = object_image.convert("RGB")
    else:
        raise ValueError("Input must be a PIL Image.")

    # Use pytesseract to extract text
    extracted_text = pytesseract.image_to_string(object_image)

    return extracted_text



def save_extracted_data(output_path, data):
    """Saving extracted text/data from objects to a CSV file."""

    with open(output_path, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["Object ID", "Extracted Text/Data"])
        for entry in data:
            writer.writerow(entry)
