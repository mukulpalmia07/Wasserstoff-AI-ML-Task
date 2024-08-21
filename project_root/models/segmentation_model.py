import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

def load_model():
    """Loading the pre-trained Mask R-CNN model."""

    model = maskrcnn_resnet50_fpn(pretrained = True)
    model.eval()  # Setting the model to evaluation mode
    return model

def get_segmented_regions(model, image_tensor):
    """Apply the model to get segmentation masks, bounding boxes, and labels."""
    
    with torch.no_grad():
        predictions = model([image_tensor])
    
    masks = predictions[0]['masks'].cpu().numpy()
    boxes = predictions[0]['boxes'].cpu().numpy()
    labels = predictions[0]['labels'].cpu().numpy()
    return masks, boxes, labels
