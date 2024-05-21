import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model



def segmentation_model_test(im_path, model_name):
    model = load_model(f'saved_models/{model_name}')

    # Preprocess the image
    original_img = load_img(im_path)
    original_size = original_img.size
    img = original_img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred = np.argmax(pred, axis=-1) 

    # Post Processing
    pred_resized = zoom(pred[0], (original_size[0]/224, original_size[1]/224), order=1)

    plt.imshow(pred_resized, cmap='jet')
    plt.show()

def detection_model_test(im_path, model_name):
    model = load_model(f'saved_models/{model_name}')

    original_img = load_img(im_path)
    original_size = original_img.size
    img = original_img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)

    boxes, scores, classes, nums = pred
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