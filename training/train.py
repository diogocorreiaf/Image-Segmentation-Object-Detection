import tensorflow as tf
import os
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import mixed_precision
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from utils.utils import get_path
from models.detection_models import yolo_loss,define_base_model, create_detection_model
from models.segmentation_models import create_segmentation_model
from collections import Counter
from utils.utils import calculate_iou





def log_det_model_performance(model, model_name, test, test_loss):
    os.makedirs('saved_models', exist_ok=True)
    
    # Configure logging to avoid duplicate log entries
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(filename=f'saved_models/{model_name}.log', level=logging.INFO)
    
    y_true_all = []
    y_pred_all = []

    for test_images, test_labels in test:
        y_pred = model.predict(test_images)
        y_true_all.extend(test_labels)
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
   # mAP = average_precision_score(y_true_all.ravel(), y_pred_all.ravel())
    
    #iou = calculate_iou(y_true_all, y_pred_all)
    #mean_iou = np.mean(iou)
    
    precision = precision_score(y_true_all, y_pred_all, average='macro')
    recall = recall_score(y_true_all, y_pred_all, average='macro')
    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    
    #logging.info(f'Mean Average Precision (mAP): {mAP:.4f}')
    #logging.info(f'Mean Intersection over Union (IoU): {mean_iou:.4f}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    
    true_class_counts = Counter(y_true_all)
    pred_class_counts = Counter(y_pred_all)
    logging.info(f'True class distribution: {true_class_counts}')
    logging.info(f'Predicted class distribution: {pred_class_counts}')
    
    logging.info(f'Hyperparameters: {model.optimizer.get_config()}')
    logging.info(f'Test loss: {test_loss:.4f}')
    
    model.summary(print_fn=logging.info)
    
    
    
def log_seg_model_performance(model, model_name, test, test_loss, test_acc):
    os.makedirs('saved_models', exist_ok=True)
    
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(filename=f'saved_models/{model_name}.log', level=logging.INFO)
    
    y_true_all = []
    y_pred_all = []

    for test_images, test_masks in test:
        y_pred = np.argmax(model.predict(test_images), axis=-1)
        y_true = np.argmax(test_masks, axis=-1)
        y_pred_all.extend(y_pred.flatten())
        y_true_all.extend(y_true.flatten())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    
    cm = confusion_matrix(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all, average='weighted')
    recall = recall_score(y_true_all, y_pred_all, average='weighted')
    f1 = f1_score(y_true_all, y_pred_all, average='weighted')
    
    logging.info(f'Confusion Matrix: \n{cm}')
    logging.info(f'Precision: {precision:.4f}')
    logging.info(f'Recall: {recall:.4f}')
    logging.info(f'F1 Score: {f1:.4f}')
    
    true_class_counts = Counter(y_true_all)
    pred_class_counts = Counter(y_pred_all)
    logging.info(f'True class distribution: {true_class_counts}')
    logging.info(f'Predicted class distribution: {pred_class_counts}')
    
    logging.info(f'Hyperparameters: {model.optimizer.get_config()}')
    logging.info(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}')
    
    model.summary(print_fn=logging.info)
    
    
    
def train_detection_model(model, Train, Val, Batchsize=2, Epochs=50):
    #Set callbacks
    mixed_precision.set_global_policy('mixed_float16')
    gc.collect()
    gc.enable()
    
    Early, Ceckpoint, Tensorboard, checkpoint_path = get_callbacks()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    #Train model
    history = model.fit(Train, validation_data=Val, batch_size=Batchsize, epochs=Epochs, callbacks=[Early, Ceckpoint, Tensorboard])
    best_checkpoint = Ceckpoint.filepath.format(epoch=Early.stopped_epoch, val_loss=min(history.history['val_loss']))
    model.load_weights(best_checkpoint)
    os.makedirs('saved_models/detection_models', exist_ok=True)
    
    #Logging the model
    test_loss = model.evaluate(Test)
    model.save(os.path.join('saved_models/detection_models', model_name + '.keras'))
    log_det_model_performance(model, model_name, Test, test_loss)
    pd.DataFrame(history.history).plot(figsize = (10,8))
    plt.grid('True')
    plt.savefig("Model_Learning_Curve.png")
    plt.show()




def train_segmentation_model(model,model_name, Train, Val, Test, Batchsize=2, Epochs=50):
    #Set callbacks
    mixed_precision.set_global_policy('mixed_float16')
    gc.collect()
    gc.enable()
    
    Early, Ceckpoint, Tensorboard, checkpoint_path = get_callbacks()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    #Train model
    history = model.fit(Train, validation_data=Val, batch_size=Batchsize, epochs=Epochs, callbacks=[Early, Ceckpoint, Tensorboard])
    best_checkpoint = Ceckpoint.filepath.format(epoch=Early.stopped_epoch, val_loss=min(history.history['val_loss']))
    model.load_weights(best_checkpoint)
    os.makedirs('saved_models/segmentation_models', exist_ok=True)
    
    #Logging the model
    test_loss, test_accuracy = model.evaluate(Test)
    model.save(os.path.join('saved_models', model_name + '.keras'))
    log_seg_model_performance(model, model_name, Test, test_loss, test_accuracy)
    pd.DataFrame(history.history).plot(figsize = (10,8))
    plt.grid('True')
    plt.savefig("Model_Learning_Curve.png")
    plt.show()

def train_models(task, model_name, train, val, test):
    mixed_precision.set_global_policy('mixed_float16')
    gc.collect()
    gc.enable()
    if task == 'segmentation':
        model = create_segmentation_model()
    elif task == 'detection':
        base_model = define_base_model()
        model = create_detection_model(base_model)

    EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    history = model.fit(train, validation_data=val, batch_size=2, epochs=2, callbacks=[EarlyStop, Checkpoint, Tensorboard])
    best_checkpoint = Checkpoint.filepath.format(epoch=EarlyStop.stopped_epoch, val_loss=min(history.history['val_loss']))
    model.load_weights(best_checkpoint)

    
    os.makedirs('saved_models', exist_ok=True)
    model.save(os.path.join('saved_models', model_name + '.keras'))

    if task == 'segmentation':
        test_loss, test_accuracy = model.evaluate(test)
        print(f"Test Loss: {test_loss}")
        print(f"Test Accuracy: {test_accuracy}")

        log_seg_model_performance(model, model_name, test, test_loss, test_accuracy)
    elif task == 'detection':
        test_loss = model.evaluate(test)
        print(f"Test Loss: {test_loss}")

        log_det_model_performance(model, model_name, test, test_loss)
        



    pd.DataFrame(history.history).plot(figsize = (10,8))
    plt.grid('True')
    plt.savefig("Model_Learning_Curve.png")
    plt.show()



def get_callbacks():
    EarlyStop = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights=True)
    checkpoint_path = os.path.join(os.curdir,"checkpoint")
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True)
    Tensorboard = tf.keras.callbacks.TensorBoard(get_path())
    return  EarlyStop, Checkpoint, Tensorboard, checkpoint_path