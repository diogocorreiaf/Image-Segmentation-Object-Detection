import cv2
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from models.detection_models import yolo_loss
import matplotlib.colors as mcolors
from PIL import Image


def segmentation_model_test(im_path, model_name, img_width=224, img_height=224):
    model = tf.keras.models.load_model(f'saved_models/{model_name}.keras')
    im_path = im_path + '.jpg'
    original_img = load_img(im_path)
    original_size = original_img.size 
    img = original_img.resize((img_width, img_height))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    pred = model.predict(img_array)
    pred = pred.squeeze() 
    pred = np.argmax(pred, axis=-1) 

    print("Predicted Mask Shape:", pred.shape)
    print("Unique Values in Predicted Mask:", np.unique(pred))

    pred_resized = zoom(pred, (original_size[1]/img_height, original_size[0]/img_width), order=1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_img)

    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmentation Mask")
    plt.imshow(pred_resized, cmap='jet', alpha=0.5)

    plt.show()

def gui_segmentation_model_test(model_name, original_img):
    model = tf.keras.models.load_model(f'saved_models/segmentation_models/{model_name}')
    original_size = original_img.shape[:2] 
    img = cv2.resize(original_img, (224, 224)) 
    img_array = np.expand_dims(img, axis=0) / 255.0 

    pred = model.predict(img_array)
    pred = pred.squeeze() 
    pred = np.argmax(pred, axis=-1) 
    pred_resized = zoom(pred, (original_size[0]/224, original_size[1]/224), order=1) 

    # Normalize to 0-1
    pred_resized = pred_resized / pred_resized.max()

    # Apply colormap
    cmap = plt.get_cmap('jet')
    pred_resized = cmap(pred_resized)

    # Convert to uint8 and save
    pred_resized = (pred_resized * 255).astype('uint8')
    Image.fromarray(pred_resized).save('output.png')

    return pred_resized

def process_output(pred):
    grid_size = pred.shape[1]
    num_boxes = 2 
    num_classes = 21  

    object_masks = pred[..., 0:2]
    boxes = pred[..., 2:10]  
    class_probs = pred[..., 10:]  

    scores = object_masks * class_probs

    classes = np.argmax(scores, axis=-1)
    scores = np.max(scores, axis=-1)

    nums = np.sum(scores > 0.5, axis=-1)

    return boxes, scores, classes, nums


def detection_model_test(im_path, model_name):
    model = tf.keras.models.load_model(f'saved_models/{model_name}.keras', custom_objects={'yolo_loss': yolo_loss})
    im_path = im_path + '.jpg'
    original_img = load_img(im_path)
    original_size = original_img.size
    img = original_img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    boxes, scores, classes, nums = process_output(pred)
    boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

    boxes = boxes * [original_size[0]/224, original_size[1]/224, original_size[0]/224, original_size[1]/224]

    img = cv2.imread(im_path)
    img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
    cv2.imshow('Output', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def testing_models(task, im_path, model_name):

    if task == 'segmentation':
        segmentation_model_test(im_path, model_name)
    elif task == 'detection':
        detection_model_test(im_path, model_name)
    return