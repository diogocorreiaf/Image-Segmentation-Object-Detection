import os
import  xml.etree.ElementTree as ET
import tensorflow as tf
from dataset_preprocessing.data_preprocessing import create_data_loader

Pascal_VOC_classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']



def load_image_seg_class_paths(dataset_path, dataset_type):
    """
    Pairs Images and the corresponding segmentation class image from the dataset path

    Args:
    - dataset_path (str): The path to the dataset directory.
    - dataset_type (str): Type of dataset to load paths for: 'train', 'val', or 'test'. Default is 'train'.

    Returns:
    - List of lists: Each inner list contains two strings: the path to an image and the path to its corresponding segmentation class.
    """
    temp = []
    if dataset_type == 'train':
        txt_file = "train.txt"
    elif dataset_type == 'validation':
        txt_file = "val.txt"
    elif dataset_type == 'test':
        txt_file = "test.txt"
    else:
        raise ValueError("Invalid dataset_type. Use 'train', 'val', or 'test'.")

    new_path = os.path.join(dataset_path, "VOC2012_train_val", "VOC2012_train_val", "ImageSets", "Segmentation", txt_file)
    with open(new_path, "r") as file_:
        Instances = file_.read().split()
        for img in Instances:
            path_img = os.path.join(dataset_path, "VOC2012_train_val", "VOC2012_train_val", "JPEGImages", img + ".jpg")
            path_seg_class = os.path.join(dataset_path, "VOC2012_train_val", "VOC2012_train_val", "SegmentationClass", img + ".png")
            temp.append([path_img, path_seg_class])

    print(f"Number of images loaded for {dataset_type}: {len(temp)}")
    return temp



def load_image_annotations(dataset_path, dataset_type ):
    """
    Pairs Images and the corresponding annotations from the dataset path

    Args:
    - dataset_path (str): The path to the dataset directory.
    - dataset_type (str): Type of dataset to load paths for: 'train', 'val', or 'test'. Default is 'train'.

    Returns:
    - List of lists: Each inner list contains two strings: the path to an image and the path to its corresponding annotation
    """
    temp = []
    if dataset_type == 'train':
        txt_file = "train.txt"
    elif dataset_type == 'validation':
        txt_file = "val.txt"
    elif dataset_type == 'test':
        txt_file = "test.txt"
    else:
        raise ValueError("Invalid dataset_type. Use 'train', 'val', or 'test'.")
    new_path = os.path.join(dataset_path, "VOC2012_train_val", "VOC2012_train_val", "ImageSets", "Segmentation", txt_file)
    with open(new_path, "r") as file_:
        Instances = file_.read().split()
        for img in Instances:
            path_img = os.path.join(dataset_path, "VOC2012_train_val", "VOC2012_train_val", "JPEGImages", img + ".jpg")
            path_annotations = os.path.join(dataset_path, "VOC2012_train_val", "VOC2012_train_val", "Annotations", img + ".xml")
            temp.append([path_img, path_annotations])
    
    print(f"Number of images loaded for {dataset_type}: {len(temp)}")
    return temp



def load_and_shuffle_data(dataset_path, model_type):
    if model_type == 'segmentation':
        Train = load_image_seg_class_paths(dataset_path, dataset_type='train')
        Val = load_image_seg_class_paths(dataset_path, dataset_type='validation')
        Test = load_image_seg_class_paths(dataset_path, dataset_type='test')
    elif model_type == 'detection':
        Train = load_image_annotations(dataset_path, dataset_type='train')
        Val = load_image_annotations(dataset_path, dataset_type='validation')
        Test = load_image_annotations(dataset_path, dataset_type='test')
    else:
        raise ValueError("Invalid model_type. Use 'segmentation' or 'detection'.")
    Train = tf.random.shuffle(Train)
    Val = tf.random.shuffle(Val)
    Test = tf.random.shuffle(Test)

    return Train, Val, Test

def create_datasets(Train,Val,Test, model_type):
    Train = tf.data.Dataset.from_tensor_slices(Train)
    Val = tf.data.Dataset.from_tensor_slices(Val)
    Test = tf.data.Dataset.from_tensor_slices(Test)


    Train = create_data_loader(Train, train_type='train', data_type=model_type)
    Val = create_data_loader(Val, train_type='validation', data_type=model_type)
    Test = create_data_loader(Test, train_type='test', data_type=model_type)
   

    return Train, Val, Test