import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import albumentations as A
import tensorflow as tf
import xml.etree.ElementTree as ET
import numpy as np
import cv2
from PIL import Image
from albumentations import RandomRotate90

# Pascal VOC constains 21 different classes 
num_classes = 21

# Size that Images will be resized to
Img_Width,Img_Height = 224,224 

# Split size
SPLIT_SIZE = 7

# Classes in the Pascal VOC dataset
classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']


transforms = A.Compose([
    A.Resize(Img_Height,Img_Width),
    A.RandomCrop(
         width=np.random.randint(int(0.9*Img_Width),Img_Width),
         height=np.random.randint(int(0.9*Img_Height),Img_Height), p=0.5),
    A.RandomScale(scale_limit=0.1, interpolation=cv2.INTER_LANCZOS4,p=0.5),
    A.HorizontalFlip(p=0.5,),
    A.Resize(Img_Height,Img_Width),
], bbox_params=A.BboxParams(format='yolo', ))

def aug_albument(image,bboxes):
  augmented=transforms(image=image,bboxes=bboxes)
  return [tf.convert_to_tensor(augmented["image"],dtype=tf.float32),
          tf.convert_to_tensor(augmented["bboxes"],dtype=tf.float32)]

def process_data(image,bboxes):
    aug= tf.numpy_function(func=aug_albument, inp=[image,bboxes], Tout=(tf.float32,tf.float32))
    return aug[0],aug[1]

def preprocess_augment(img, y):
    img = tf.image.random_brightness(img, max_delta=50.)
    img = tf.image.random_saturation(img, lower=0.5, upper=1.5)
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)
    img = tf.clip_by_value(img, 0, 255)
    labels=tf.numpy_function(func=generate_output, inp=[y], Tout=(tf.float32))
    return img, labels

def preprocess(img, y):
    img = tf.cast(tf.image.resize(img, size=[Image, Img_Width]), dtype=tf.float32)
    labels=tf.numpy_function(func=generate_output, inp=[y], Tout=(tf.float32))
    return img, labels

def preprocess_xml(filename):
    if isinstance(filename, np.ndarray):
        filename = filename.item()  # Convert numpy ndarray to native Python type
    tree = ET.parse(filename)
    root = tree.getroot()
    size_tree = root.find('size')
    height = float(size_tree.find('height').text)
    width = float(size_tree.find('width').text)
    bounding_boxes = []
    class_dict = {classes[i]: i for i in range(len(classes))}
    for object_tree in root.findall('object'):
        for bounding_box in object_tree.iter('bndbox'):
            xmin = float(bounding_box.find('xmin').text)
            xmax = float(bounding_box.find('xmax').text)
            ymin = float(bounding_box.find('ymin').text)
            ymax = float(bounding_box.find('ymax').text)
            break
        class_name = object_tree.find('name').text
        bounding_box = [
            (xmin + xmax) / (2 * width),
            (ymin + ymax) / (2 * height),
            (xmax - xmin) / width,
            (ymax - ymin) / height,
            class_dict[class_name]
        ]
        bounding_boxes.append(bounding_box)
    return tf.convert_to_tensor(bounding_boxes)



def get_imbboxes(im_path, xml_path):
    img = tf.io.decode_jpeg(tf.io.read_file(im_path))
    img = tf.cast(tf.image.resize(img, size=[Img_Height, Img_Width]), dtype=tf.float32)
    tf.py_function(lambda: print(f"Image shape: {img.shape}"), inp=[], Tout=[])
    bboxes = tf.numpy_function(func=preprocess_xml, inp=[xml_path], Tout=tf.float32)
    return img, bboxes



def generate_output(bounding_boxes):
  output_label=np.zeros((SPLIT_SIZE,SPLIT_SIZE,num_classes+5))
  for b in range(len(bounding_boxes)):
    grid_x=bounding_boxes[...,b,0]*SPLIT_SIZE
    grid_y=bounding_boxes[...,b,1]*SPLIT_SIZE
    i=int(grid_x)
    j=int(grid_y)

    output_label[i,j,0:5]=[1.,grid_x%1,grid_y%1,bounding_boxes[...,b,2],bounding_boxes[...,b,3]]
    output_label[i,j,5+int(bounding_boxes[...,b,4])]=1.

  return tf.convert_to_tensor(output_label,tf.float16)

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



def seg_preprocess_augment(Instance, is_training):
    '''
    Preprocesses an image and its corresponding mask for training.

    Args:
    - Instance (tuple): A tuple containing image and mask paths.
    - is_training: Checks the training type

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


    # Only apply augmentations if training dataset
    if is_training and tf.random.uniform(()) > 0.5:
        aug = RandomRotate90(p=0.5) 
        Augmented = aug(image=Img, mask=Mask)

        Img = Augmented["image"]
        Mask = Augmented["mask"]

    return Normalization(Img), Create_Mask(Mask)


def detect_preprocess_augment(Instance, is_training):
    '''
    Preprocesses an image and its corresponding mask for training.

    Args:
    - Instance (tuple): A tuple containing image and mask paths.
    - is_training: Checks the training type

    Returns:
    - tuple: A tuple containing preprocessed image and mask.
    '''
    Img = Image.open(Instance[0].numpy())
    Img = Img.resize((Img_Width, Img_Height), resample=Image.BILINEAR)
    Normalization = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

    bboxes = tf.numpy_function(func=preprocess_xml, inp=[Instance[1]], Tout=tf.float32)

    if is_training:
        Img = tf.image.random_brightness(Img, max_delta=50.)
        Img = tf.image.random_saturation(Img, lower=0.5, upper=1.5)
        Img = tf.image.random_contrast(Img, lower=0.5, upper=1.5)
        Img = tf.clip_by_value(Img, 0, 255)

    Img = tf.cast(Img, dtype=tf.float16)
    bboxes=tf.numpy_function(func=generate_output, inp=[bboxes], Tout=(tf.float32))

    return Normalization(Img), bboxes




def seg_preprocess(Instance, is_training = True):
    '''
    Preprocesses an image and its corresponding mask for training.

    Args:
    - Instance (tuple): A tuple containing image and mask paths.
    - is_training: checks the training type

    Returns:
    - tuple: A tuple containing preprocessed image and mask.
             Preprocessed image: Scaled to specified dimensions and augmented if necessary.
             Preprocessed mask: Segmentation mask corresponding to the image.
    ''' 
    Img, Mask = tf.py_function(seg_preprocess_augment, [Instance, is_training], [tf.float16, tf.float16])
    
    Img = tf.ensure_shape(Img, [None, None, 3])
    Mask = tf.ensure_shape(Mask, [None, None, num_classes]) 
    
    return Img, Mask

  

def detect_preprocess(Instance, is_training):
    '''
    Preprocess an Image and its corresponding annotation for training
    
    Args:
    - im_path (string): The path to the image file.
    - xml_path (string): The path to the XML file.
    - is_training (bool): checks the training type

    Returns:
    - tuple: A tuple containing preprocessed image and mask.
             Preprocessed image: Scaled to specified dimensions and augmented if necessary.
             Preprocessed mask: Segmentation mask corresponding to the image.'''

    Img, bboxes = tf.py_function(detect_preprocess_augment, [Instance, is_training], [tf.float16, tf.float16])
    Img = tf.ensure_shape(Img, [None, None, 3])

    return Img, bboxes



def create_data_loader(dataset, train_type, data_type, BATCH_SIZE=2, BUFFER_SIZE=2):
    '''
    Creates a TensorFlow data pipeline for training or validation data.

    Args:
    - dataset (tf.data.Dataset): Input dataset containing image and mask paths.
    - BATCH_SIZE (int): Batch size for training.
    - BUFFER_SIZE (int): Buffer size for shuffling the dataset.

    Returns:
    - tf.data.Dataset: A TensorFlow dataset pipeline for training or validation.
    '''

    if(data_type == 'segmentation'):
        if(train_type == 'train') :
            data = dataset.map(lambda x: seg_preprocess(x, is_training=True), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            data = dataset.map(lambda x: seg_preprocess(x, is_training=False), num_parallel_calls=tf.data.AUTOTUNE)

    elif(data_type == 'detection'):
        if(train_type == 'train') :
            data = dataset.map(lambda x: detect_preprocess(x, is_training=True), num_parallel_calls=tf.data.AUTOTUNE)
        else:
            data = dataset.map(lambda x: detect_preprocess(x, is_training=False), num_parallel_calls=tf.data.AUTOTUNE)


    data = data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat(1)
    data = data.prefetch(buffer_size = tf.data.AUTOTUNE)
    


    return data


