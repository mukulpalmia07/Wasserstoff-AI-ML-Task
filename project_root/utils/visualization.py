import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_mapping import objects_data, annotate_image, summary_table

def display_segmented_image(image_rgb, masks, boxes, labels):
    """Displaying the image with segmented regions and bounding boxes."""

    for i in range(masks.shape[0]):
        mask = masks[i, 0]
        mask = mask > 0.5  # Creating a binary mask
        image_rgb[mask] = np.array([0, 255, 0], dtype = np.uint8)

        # Drawing bounding boxes
        box = boxes[i].astype(int)
        cv2.rectangle(image_rgb, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)

    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()

def display_output(image, summary_table):
    """Displaying the annotated image alongside the summary table."""

    # Creating a figure with 2 subplots: one for the image, one for the table
    fig, ax = plt.subplots(1, 2, figsize = (15, 8))

    # Displaying the annotated image
    ax[0].imshow(image)
    ax[0].axis('off')  # Turn off axis
    ax[0].set_title('Annotated Image')

    # Displaying the summary table
    ax[1].axis('tight')
    ax[1].axis('off')
    ax[1].table(cellText = summary_table.values, colLabels = summary_table.columns, cellLoc = 'center', loc = 'center')

    plt.show()
