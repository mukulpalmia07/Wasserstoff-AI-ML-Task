import numpy as np
from PIL import Image
import sqlite3

def extract_and_save_objects(image_rgb, masks, output_dir, master_id):
    """Extracting each segmented object and save as a separate image."""

    object_ids = []
    for i in range(masks.shape[0]):
        mask = masks[i, 0] > 0.5
        # Extracting the object using the mask
        segmented_object = image_rgb * np.dstack([mask] * 3)

        # Cropping the object to its bounding box
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis = 0)[0], coords.min(axis = 0)[1]
        y1, x1 = coords.max(axis = 0)[0], coords.max(axis = 0)[1]
        
        cropped_object = segmented_object[y0 : y1 + 1, x0 : x1 + 1]

        # Saving the object as an image
        object_id = f"{master_id}_obj_{i + 1}"
        object_ids.append(object_id)
        object_image = Image.fromarray(cropped_object)
        object_image.save(f"{output_dir}/{object_id}.png")

    return object_ids


def initialize_database(db_path):
    """Initializing the database and create the necessary tables."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS images (
                      master_id TEXT PRIMARY KEY,
                      image_path TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS objects (
                      object_id TEXT PRIMARY KEY,
                      master_id TEXT,
                      object_path TEXT,
                      FOREIGN KEY(master_id) REFERENCES images(master_id))''')
    conn.commit()
    conn.close()

def save_metadata(master_id, image_path, object_ids, output_dir, db_path):
    """Saving metadata about the original image and its objects."""

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Saving master image information
    cursor.execute("INSERT INTO images (master_id, image_path) VALUES (?, ?)", (master_id, image_path))

    # Saveing objects' metadata
    for object_id in object_ids:
        object_path = f"{output_dir}/{object_id}.png"
        cursor.execute("INSERT INTO objects (object_id, master_id, object_path) VALUES (?, ?, ?)", (object_id, master_id, object_path))

    conn.commit()
    conn.close()