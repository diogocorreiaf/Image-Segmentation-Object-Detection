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
from keras.saving import register_keras_serializable

classes=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']

Width, Height = 224, 224



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
    num_classes = 20  

    object_masks = pred[..., 0:2]
    boxes = pred[..., 2:10]  
    class_probs = pred[..., 10:]  

    scores = object_masks * class_probs

    classes = np.argmax(scores, axis=-1)
    scores = np.max(scores, axis=-1)

    nums = np.sum(scores > 0.5, axis=-1)

    return boxes, scores, classes, nums


@register_keras_serializable('yolo_loss')
def yolo_loss(y_true, y_pred):
    return yolo_loss(y_true, y_pred)

def gui_detection_model_test(model_name, img):
    
    
    
    # Resize the image
    img = cv2.resize(img, (224, 224))

    # Save a copy of the original image for later
    original_img = img.copy()

    # Convert the image to a tensor and resize it
    image = tf.convert_to_tensor(img)
    image = tf.image.resize(image, [Width, Height])

    model = tf.keras.models.load_model(f'saved_models/detection_models/{model_name}')
    
    # Predict the output
    output = model.predict(np.expand_dims(image, axis=0))

    THRESH = .25

    object_positions = tf.concat(
        [tf.where(output[..., 0] >= THRESH), tf.where(output[..., 5] >= THRESH)], axis=0)
    selected_output = tf.gather_nd(output, object_positions)
    final_boxes = []
    final_scores = []

    for i, pos in enumerate(object_positions):
        for j in range(2):
            if selected_output[i][j*5] > THRESH:
                output_box = tf.cast(output[pos[0]][pos[1]][pos[2]][(j*5)+1:(j*5)+5], dtype=tf.float32)

                x_centre = (tf.cast(pos[1], dtype=tf.float32) + output_box[0]) * 32
                y_centre = (tf.cast(pos[2], dtype=tf.float32) + output_box[1]) * 32

                x_width, y_height = tf.math.abs(Height*output_box[2]), tf.math.abs(Width*output_box[3])

                x_min, y_min = int(x_centre - (x_width / 2)), int(y_centre - (y_height / 2))
                x_max, y_max = int(x_centre + (x_width / 2)), int(y_centre + (y_height / 2))

                if(x_min <= 0): x_min = 0
                if(y_min <= 0): y_min = 0
                if(x_max >= Width): x_max = Width
                if(y_max >= Height): y_max = Height
                final_boxes.append(
                    [x_min, y_min, x_max, y_max,
                    str(classes[tf.argmax(selected_output[..., 10:], axis=-1)[i]])])
                final_scores.append(selected_output[i][j*5])

    final_boxes = np.array(final_boxes)

    object_classes = final_boxes[..., 4]
    nms_boxes = final_boxes[..., 0:4]

    nms_output = tf.image.non_max_suppression(
        nms_boxes, final_scores, max_output_size=100, iou_threshold=0.2,
        score_threshold=float('-inf')
    )

    for i in nms_output:
        cv2.rectangle(
            img,
            (int(final_boxes[i][0]), int(final_boxes[i][1])),
            (int(final_boxes[i][2]), int(final_boxes[i][3])), (0, 0, 255), 1)
        cv2.putText(
            img,
            final_boxes[i][-1],
            (int(final_boxes[i][0]), int(final_boxes[i][1]) + 15),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (2, 225, 155), 1
        )

    # Convert BGR to RGB for matplotlib
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



    return img

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