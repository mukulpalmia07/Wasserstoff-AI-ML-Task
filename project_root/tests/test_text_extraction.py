import pytesseract
from PIL import Image
import csv

# Set the Tesseract-OCR executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_image_and_save_data(image_path, output_csv_path):
    """Extracting text from the given image and save the data to a CSV file."""
    
    # Opening the image
    with Image.open(image_path) as img:
        # Converting image to RGB if it isn't already
        img = img.convert("RGB")

        # Extracting text using Tesseract OCR
        extracted_text = pytesseract.image_to_string(img)

    # Preparing data to be saved
    data = [("1", extracted_text)]

    # Saving extracted data to a CSV file
    with open(output_csv_path, mode = 'w', newline = '') as file:
        writer = csv.writer(file)
        writer.writerow(["Object ID", "Extracted Text/Data"])
        writer.writerows(data)

    print(f"Data has been saved to {output_csv_path}")

if __name__ == "__main__":

    image_path = 'data\\input_image\\lotus.jpg'
    output_csv_path = 'data\\output\\test_text_extraction.csv'

    process_image_and_save_data(image_path, output_csv_path)
