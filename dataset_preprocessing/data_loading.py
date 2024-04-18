import os
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import cv2 as cv 
import matplotlib.pyplot as plt
from PIL import Image



num_classes = 21 #Pascal VOC comes with 21 different classes


# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'



def load_image_seg_class_paths (dataset_path,is_train = True):
    """
    Pairs Images and the corresponding segmentation class image from the dataset path

    Args:
    - dataset_path (str): The path to the dataset directory.
    - is_train (bool): Indicates whether to load paths for training data (True) or validation data (False). Default is True.

    Returns:
    - List of lists: Each inner list contains two strings: the path to an image and the path to its corresponding segmentation class.
    """
    temp = []
    new_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","train.txt" if is_train else "val.txt")
    with open(new_path,"r") as file_:
        Instances = file_.read().split()
        for img in Instances:
            path_img = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","JPEGImages",img+".jpg")
            path_seg_class = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","SegmentationClass",img+".png")
            temp.append([path_img,path_seg_class])

    if is_train:
        print(f"Number of images loaded for training: {len(temp)}")
    else:
        print(f"Number of images loaded for validation: {len(temp)}")
    return temp


def main():
    # Storing the list of paired images and segmentation class image
    Train = load_image_seg_class_paths(dataset_path,is_train=True)
    Val = load_image_seg_class_paths(dataset_path,is_train=False)

    # shuffling the lists
    Train = tf.random.shuffle(Train)
    Val = tf.random.shuffle(Val)
    
    return 

if __name__ == "__main__":
    main()
