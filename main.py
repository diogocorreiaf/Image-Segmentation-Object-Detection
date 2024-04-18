import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras
from keras import mixed_precision
from dataset_preprocessing.data_loading import load_image_seg_class_paths
from dataset_preprocessing.data_preprocessing import create_data_loader




# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

# Set Training Precision
mixed_precision.set_global_policy('mixed_float16')

def main():
    # Storing the list of paired images and segmentation class image
    Train = load_image_seg_class_paths(dataset_path,is_train=True)
    Val = load_image_seg_class_paths(dataset_path,is_train=False)

    # shuffling the lists
    Train = tf.random.shuffle(Train)
    Val = tf.random.shuffle(Val)

    #create Tensorflow Dataset
    Train = tf.data.Dataset.from_tensor_slices(Train)
    Val = tf.data.Dataset.from_tensor_slices(Val)
    
    
    Train = create_data_loader(Train)
    Val = create_data_loader(Val)
    Q = []
    for X,Y in Val.take(1):
        print(X.shape)
        Q = Y
        print(Y.shape)
    
    print(Q.dtype)


    return 

if __name__ == "__main__":
    main()