import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


#TODO preprocessing the image before testing against the model, and test to see if it actually works

def segmentation_model_test(im_path, model_name):
    model = load_model(model_name)

    img = load_img(im_path, target_size=(model.input_shape[1], model.input_shape[2]))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    pred = np.argmax(pred, axis=-1) 

    plt.imshow(pred[0], cmap='jet')
    plt.show()

def detection_model_test(im_path, model_name):
    model = load_model(model_name)

    img = load_img(im_path, target_size=(model.input_shape[1], model.input_shape[2]))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Predict the bounding boxes and class labels
    pred = model.predict(img_array)

    # Postprocess the predictions
    boxes, scores, classes, nums = pred
    boxes, scores, classes, nums = boxes[0], scores[0], classes[0], nums[0]

    # Display the image with bounding boxes
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