import random
import os
import numpy as np
root_dir = os.path.join(os.curdir,"my_logs")


dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

def dataset_randomizer(task, test_ratio = 0.65, val_ratio = 0.2):
    ''' Randomizes the Dataset, acesses the trainval.txt and splits it between the train, val and test.txt file
        
        Args:
        - dataset_path (str): The path to the dataset directory. 
        '''
    if task == "segmentation":
        trainval_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","trainsegmentation.txt")
        train_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","train.txt")
        val_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","val.txt")
        test_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","test.txt")

    elif task == "detection":
        trainval_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Main","trainval.txt")
        train_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Main","train.txt")
        val_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Main","val.txt")
        test_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Main","test.txt")


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


def compute_metrics(predictions, labels):
    # Extract confidence scores and bounding box coordinates from predictions and labels
    pred_confidence = predictions[..., 4]  # Confidence score
    true_confidence = labels[..., 4]  # Confidence score

    # Convert confidence scores to boolean (1 for object, 0 for no object)
    pred_conf_bool = tf.cast(pred_confidence > 0.5, dtype=tf.float32)
    true_conf_bool = tf.cast(true_confidence > 0.5, dtype=tf.float32)

    # Compute true positives, false positives, and false negatives
    true_positives = tf.reduce_sum(pred_conf_bool * true_conf_bool, axis=(1, 2))  # Sum over spatial dimensions (7, 7)
    false_positives = tf.reduce_sum(pred_conf_bool * (1 - true_conf_bool), axis=(1, 2))  # Sum over spatial dimensions (7, 7)
    false_negatives = tf.reduce_sum((1 - pred_conf_bool) * true_conf_bool, axis=(1, 2))  # Sum over spatial dimensions (7, 7)

    # Compute precision and recall
    precision = true_positives / tf.maximum(true_positives + false_positives, 1e-10)
    recall = true_positives / tf.maximum(true_positives + false_negatives, 1e-10)

    # Compute F1 score
    f1_score = 2 * (precision * recall) / tf.maximum(precision + recall, 1e-10)

    return precision, recall, f1_score

def compute_iou2(boxes1, boxes2):
    # Extract bounding box coordinates from predictions (boxes1) and labels (boxes2)
    boxes1_xy = boxes1[..., :2]
    boxes1_wh = boxes1[..., 2:4]
    boxes2_xy = boxes2[..., :2]
    boxes2_wh = boxes2[..., 2:4]

    # Calculate coordinates of intersection
    intersect_mins = tf.maximum(boxes1_xy - boxes1_wh / 2., boxes2_xy - boxes2_wh / 2.)
    intersect_maxes = tf.minimum(boxes1_xy + boxes1_wh / 2., boxes2_xy + boxes2_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    # Calculate area of boxes
    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
    boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

    # Calculate IoU
    union_area = boxes1_area + boxes2_area - intersect_area
    iou = intersect_area / tf.maximum(union_area, 1e-10)  # Avoid division by zero

    return iou