import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
from dataset_preprocessing.data_loading import load_image_seg_class_paths, load_image_annotations
from dataset_preprocessing.data_preprocessing import create_data_loader
from utils.utils import dataset_randomizer



# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

# Set Training Precision
mixed_precision.set_global_policy('mixed_float16')

def main():

    dataset_randomizer(dataset_path)


    # Storing the list of paired images and segmentation class image
    Train_Segmentation = load_image_seg_class_paths(dataset_path,dataset_type='train')
    Val_Segmentation = load_image_seg_class_paths(dataset_path,dataset_type='validation')
    Test_Segmentation = load_image_seg_class_paths(dataset_path,dataset_type='test')

    # Storing the list of paired images and annotations
    Train_Object_Detection = load_image_annotations(dataset_path,dataset_type='train')
    Val_Object_Detection = load_image_annotations(dataset_path,dataset_type='validation')
    Test_Object_Detection = load_image_annotations(dataset_path,dataset_type='test')

    # shuffling the lists
    Train_Segmentation = tf.random.shuffle(Train_Segmentation)
    Val_Segmentation = tf.random.shuffle(Val_Segmentation)    
    Test_Segmentation = tf.random.shuffle(Test_Segmentation)
    
    Train_Object_Detection = tf.random.shuffle(Train_Object_Detection)
    Val_Object_Detection = tf.random.shuffle(Val_Object_Detection)    
    Test_Object_Detection = tf.random.shuffle(Test_Object_Detection)

    # Create Tensorflow Dataset
    Train_Segmentation = tf.data.Dataset.from_tensor_slices(Train_Segmentation)
    Val_Segmentation = tf.data.Dataset.from_tensor_slices(Val_Segmentation)
    Test_Segmentation = tf.data.Dataset.from_tensor_slices(Test_Segmentation)

    Train_Object_Detection = tf.data.Dataset.from_tensor_slices(Train_Object_Detection)
    Val_Object_Detection = tf.data.Dataset.from_tensor_slices(Val_Object_Detection)
    Test_Object_Detection = tf.data.Dataset.from_tensor_slices(Test_Object_Detection)


        
    # Dataset Augmentation and Preprocessing 
    Train_Segmentation = create_data_loader(Train_Segmentation, 'train', 'segmentation')
    Val_Segmentation = create_data_loader(Val_Segmentation,'val', 'segmentation')
    Test_Segmentation = create_data_loader(Test_Segmentation,'test', 'segmentation')

    Train_Object_Detection = create_data_loader(Train_Segmentation, 'train', 'detection')
    Val_Object_Detection = create_data_loader(Train_Segmentation, 'val', 'detection')
    Test_Object_Detection = create_data_loader(Train_Segmentation, 'test', 'detection')



    return 

if __name__ == "__main__":
    main()