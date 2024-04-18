import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
from dataset_preprocessing.data_loading import load_image_seg_class_paths
from dataset_preprocessing.data_preprocessing import create_data_loader
from utils.utils import dataset_randomizer



# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

# Set Training Precision
mixed_precision.set_global_policy('mixed_float16')

def main():

    dataset_randomizer(dataset_path)


    # Storing the list of paired images and segmentation class image
    Train = load_image_seg_class_paths(dataset_path,dataset_type='train')
    Val = load_image_seg_class_paths(dataset_path,dataset_type='validation')
    Test = load_image_seg_class_paths(dataset_path,dataset_type='test')


    # shuffling the lists
    Train = tf.random.shuffle(Train)
    Val = tf.random.shuffle(Val)    
    Test = tf.random.shuffle(Test)


    #create Tensorflow Dataset
    Train = tf.data.Dataset.from_tensor_slices(Train)
    Val = tf.data.Dataset.from_tensor_slices(Val)
    Test = tf.data.Dataset.from_tensor_slices(Test)
    
    Train = create_data_loader(Train)
    Val = create_data_loader(Val)
    Test = create_data_loader(Test)


    return 

if __name__ == "__main__":
    main()