import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from PIL import Image
from albumentations import RandomRotate90


# Pascal VOC constains 21 different classes 
num_classes = 21

# Size that Images will be resized to
Img_Width,Img_Height = 224,224 

import numpy as np
import tensorflow as tf

def Create_Mask(Img):
    '''
    Creates segmentation masks for each class in the input image.

    Args:
    - Img (numpy.ndarray): Input image 
                           

    Returns:
    - tf.Tensor: Segmentation masks for each class in the input image.
     
    '''
    Seg_Labels = np.zeros((Img.shape[0], Img.shape[1], num_classes), dtype=np.float16)
    for class_ in range(num_classes):
        Seg_Labels[:, :, class_] = (Img == class_)
    return tf.cast(Seg_Labels, dtype=tf.float16)



def create_preprocess_mask_img(Instance):
    '''
    Preprocesses an image and its corresponding mask for training.

    Args:
    - Instance (tuple): A tuple containing image and mask paths.

    Returns:
    - tuple: A tuple containing preprocessed image and mask.
    '''
    Img = Image.open(Instance[0].numpy())
    Img = Img.resize((Img_Width, Img_Height), resample=Image.BILINEAR)
    Img = np.asarray(Img)

    Mask = Image.open(Instance[1].numpy())
    Mask = Mask.resize((Img_Width, Img_Height), resample=Image.BILINEAR)
    Mask = np.asarray(Mask)


    Normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    if tf.random.uniform(()) > 0.5:
        aug = RandomRotate90(p=0.5) 
        Augmented = aug(image=Img, mask=Mask)

        Img = Augmented["image"]
        Mask = Augmented["mask"]

    return Normalization(Img), Create_Mask(Mask)

def seg_preprocess(Instance):
    '''
    Preprocesses an image and its corresponding mask for training.

    Args:
    - Instance (tuple): A tuple containing image and mask paths.


    Returns:
    - tuple: A tuple containing preprocessed image and mask.
             Preprocessed image: Scaled to specified dimensions and augmented if necessary.
             Preprocessed mask: Segmentation mask corresponding to the image.
    '''
    Img, Mask = tf.py_function(create_preprocess_mask_img, [Instance], [tf.float16, tf.float16])
    
    Img = tf.ensure_shape(Img, [None, None, 3])
    Mask = tf.ensure_shape(Mask, [None, None, num_classes]) 
    
    return Img, Mask



def create_data_loader(dataset, BATCH_SIZE=2, BUFFER_SIZE=2):
    '''
    Creates a TensorFlow data pipeline for training or validation data.

    Args:
    - dataset (tf.data.Dataset): Input dataset containing image and mask paths.
    - BATCH_SIZE (int): Batch size for training.
    - BUFFER_SIZE (int): Buffer size for shuffling the dataset.

    Returns:
    - tf.data.Dataset: A TensorFlow dataset pipeline for training or validation.
    '''
    data = dataset.map(seg_preprocess,num_parallel_calls = tf.data.AUTOTUNE)
    data = data.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(1)
    data = data.prefetch(buffer_size = tf.data.AUTOTUNE)
    
    return data


