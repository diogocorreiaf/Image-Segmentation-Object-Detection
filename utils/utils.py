import random
import os
import numpy as np
root_dir = os.path.join(os.curdir,"my_logs")


dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

def dataset_randomizer(test_ratio = 0.65, val_ratio = 0.2):
    ''' Randomizes the Dataset, acesses the trainval.txt and splits it between the train, val and test.txt file
        
        Args:
        - dataset_path (str): The path to the dataset directory. 
        '''
    trainval_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","trainval.txt")
    train_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","train.txt")
    val_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","val.txt")
    test_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","test.txt")


    with open(trainval_path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    total_lines = len(lines)
    train_lines = int(total_lines * test_ratio)
    validation_lines = int(total_lines * val_ratio)

    with open(train_path, 'w') as train_file:
        train_file.truncate(0)
        train_file.writelines(lines[:train_lines])

    with open(val_path, 'w') as validation_file:
        validation_file.truncate(0)
        validation_file.writelines(lines[train_lines:train_lines + validation_lines])

    with open(test_path, 'w') as test_file:
        test_file.truncate(0)
        test_file.writelines(lines[train_lines + validation_lines:])


def get_path():
    import time
    id_ = time.strftime("run_%Y_%m_%D_%H_%M_%S")
    return os.path.join(root_dir,id_)


def calculate_iou(y_true, y_pred):
    """
    Calculate Intersection over Union (IoU) for bounding boxes.

    Args:
    y_true: Ground truth bounding boxes, shaped [N, 4], where N is the number of boxes and each box is represented as [x1, y1, x2, y2].
    y_pred: Predicted bounding boxes, shaped [N, 4], where N is the number of boxes and each box is represented as [x1, y1, x2, y2].

    Returns:
    IoU: Intersection over Union for each pair of boxes, shaped [N,].
    """
    # Calculate intersection coordinates
    x1 = np.maximum(y_true[:, 0], y_pred[:, 0])
    y1 = np.maximum(y_true[:, 1], y_pred[:, 1])
    x2 = np.minimum(y_true[:, 2], y_pred[:, 2])
    y2 = np.minimum(y_true[:, 3], y_pred[:, 3])

    # Calculate intersection area
    intersection_area = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    # Calculate bounding box areas
    true_area = (y_true[:, 2] - y_true[:, 0]) * (y_true[:, 3] - y_true[:, 1])
    pred_area = (y_pred[:, 2] - y_pred[:, 0]) * (y_pred[:, 3] - y_pred[:, 1])

    # Calculate Union area
    union_area = true_area + pred_area - intersection_area

    # Avoid division by zero
    epsilon = 1e-10

    # Calculate IoU
    iou = intersection_area / (union_area + epsilon)

    return iou