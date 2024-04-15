

from torchvision import transforms as tr
from PIL import Image

def transform_tr(sample):
    """
    Apply training transformations to the input sample.

    Args:
        sample (dict): Input sample containing image, segmentation mask, and annotations.

    Returns:
        dict: Transformed sample.
    """
    print("Before transformations:")
    print("Image shape:", sample['image'].shape)
    print("Segmentation mask shape:", sample['segmentation_mask'].shape)

    # Define transformations for both image and segmentation mask
    image_transforms = tr.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomResizedCrop(size=(sample['image'].size[1], sample['image'].size[0]), scale=(0.5, 2.0)),
        tr.RandomApply([tr.GaussianBlur(kernel_size=7)], p=0.5),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()
    ])

    # Apply transformations to image
    transformed_image = image_transforms(sample['image'])

    # Apply same spatial transformations to segmentation mask
    mask_transforms = tr.Compose([
        tr.RandomHorizontalFlip(),
        tr.RandomResizedCrop(size=(sample['segmentation_mask'].size[1], sample['segmentation_mask'].size[0]), scale=(0.5, 2.0)),
        # Add more mask-specific transformations if needed
        tr.ToTensor()
    ])

    # Apply transformations to segmentation mask
    transformed_mask = mask_transforms(sample['segmentation_mask'])

    print("After transformations:")
    print("Transformed image shape:", transformed_image.shape)
    print("Transformed segmentation mask shape:", transformed_mask.shape)

    # Construct transformed sample
    transformed_sample = {
        'image': transformed_image,
        'segmentation_mask': transformed_mask,
        'annotations': sample['annotations']  # Annotations remain unchanged
    }

    return transformed_sample


def transform_val(sample):
    """
    Apply validation transformations to the input sample.

    Args:
        sample (dict): Input sample containing image, segmentation mask, and annotations.

    Returns:
        dict: Transformed sample.
    """
    composed_transforms = tr.Compose([
        tr.Resize((sample['image'].size[1], sample['image'].size[0])),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()
    ])

    transformed_sample = {
        'image': composed_transforms(sample['image']),
        'segmentation_mask': composed_transforms(sample['segmentation_mask']),
        #'annotations': transform_annotation(sample['annotations'], {'scale_factor': 1.0})  # No scaling for validation
    }

    return transformed_sample

def transform_annotation(annotation, transformation_params):
    """
    Transform the input annotation based on the provided parameters.
    
    Args:
        annotation (dict): Input annotation data.
        transformation_params (dict): Parameters for transformation.
            Example: {'scale_factor': 0.5, 'translation': (10, 10)}
    
    Returns:
        dict: Transformed annotation data.
    """
    scale_factor = transformation_params.get('scale_factor', 1.0)
    transformed_annotation = {
        'object_labels': annotation['object_labels'],
        'bounding_boxes': [[int(coord * scale_factor) for coord in bbox] for bbox in annotation['bounding_boxes']]
    }
    return transformed_annotation
