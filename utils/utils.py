import random
import os
import numpy as np
import tensorflow as tf
root_dir = os.path.join(os.curdir,"my_logs")


dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

def dataset_randomizer(task, test_ratio = 0.65, val_ratio = 0.2):
    ''' Randomizes the Dataset, acesses the trainval.txt and splits it between the train, val and test.txt file
        
        Args:
        - dataset_path (str): The path to the dataset directory. 
        '''
        
    trainval_path = ""
    train_path = ""
    val_path = ""
    test_path = ""   
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
    
    pred_confidence = predictions[..., 4]  
    true_confidence = labels[..., 4] 
    
    pred_conf_bool = tf.cast(pred_confidence > 0.5, dtype=tf.float32)
    true_conf_bool = tf.cast(true_confidence > 0.5, dtype=tf.float32)

    true_positives = tf.reduce_sum(pred_conf_bool * true_conf_bool, axis=(1, 2))  
    false_positives = tf.reduce_sum(pred_conf_bool * (1 - true_conf_bool), axis=(1, 2)) 
    false_negatives = tf.reduce_sum((1 - pred_conf_bool) * true_conf_bool, axis=(1, 2)) 

    precision = true_positives / tf.maximum(true_positives + false_positives, 1e-10)
    recall = true_positives / tf.maximum(true_positives + false_negatives, 1e-10)

    f1_score = 2 * (precision * recall) / tf.maximum(precision + recall, 1e-10)

    return precision, recall, f1_score

def compute_iou2(boxes1, boxes2):
    boxes1_xy = boxes1[..., :2]
    boxes1_wh = boxes1[..., 2:4]
    boxes2_xy = boxes2[..., :2]
    boxes2_wh = boxes2[..., 2:4]

    intersect_mins = tf.maximum(boxes1_xy - boxes1_wh / 2., boxes2_xy - boxes2_wh / 2.)
    intersect_maxes = tf.minimum(boxes1_xy + boxes1_wh / 2., boxes2_xy + boxes2_wh / 2.)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    boxes1_area = boxes1_wh[..., 0] * boxes1_wh[..., 1]
    boxes2_area = boxes2_wh[..., 0] * boxes2_wh[..., 1]

    union_area = boxes1_area + boxes2_area - intersect_area
    iou = intersect_area / tf.maximum(union_area, 1e-10) 

    return iou

