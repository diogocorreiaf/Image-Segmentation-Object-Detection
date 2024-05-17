import tensorflow as tf
import os
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras import mixed_precision
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils.utils import get_path
from models.detection_models import yolo_loss,define_base_model
from models.segmentation_models import create_segmentation_model

def get_callbacks():
    EarlyStop = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights=True)
    checkpoint_path = os.path.join(os.curdir,"checkpoint")
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True)
    root_dir = os.path.join(os.curdir,"my_logs")
    board_log_path = get_path()
    Tensorboard = tf.keras.callbacks.TensorBoard(board_log_path)
    return EarlyStop, Checkpoint, Tensorboard, checkpoint_path

def log_model_performance(model, test, test_loss, test_acc):
    y_pred = np.argmax(model.predict(test), axis=-1)
    y_true = np.argmax(test, axis=-1)
    cm = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    logging.info(f'Confusion Matrix: {cm}')
    logging.info(f'Precision: {precision}')
    logging.info(f'Recall: {recall}')
    logging.info(f'F1 Score: {f1}')
    logging.info(f'Hyperparameters: {model.optimizer.get_config()}')
    logging.info(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
    model.summary(print_fn=logging.info)


def train_segmentation_model(model, Train, Val, Batchsize=2, Epochs=100):
    EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
    history = model.fit(Train,validation_data=Val,batch_size=Batchsize,epochs=Epochs,callbacks=[EarlyStop,Checkpoint,Tensorboard])
    model.load_weights(checkpoint_path)
    return model, history


def train_detection_model(model, Train, Val, Batchsize=2, Epochs=100):
    def scheduler(epoch, lr):
        if epoch < 40:
            return 1e-3
        elif epoch >= 40 and epoch < 80:
            return 5e-4
        else:
            return 1e-4

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # Define the model checkpoint callback
    checkpoint_filepath = '/content/drive/MyDrive/Bang/yolo_efficientnet_b1_new.h5'
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    # Compile the model
    model.compile(
        loss=yolo_loss,
        optimizer=tf.keras.optimizers.Adam(1e-3),
    )

    # Train the model
    history = model.fit(
        Train,
        validation_data=Val,
        batch_size=Batchsize,
        verbose=1,
        epochs=Epochs,
        callbacks=[lr_callback, callback]
    )

    return model, history


def train_models(task, model_name, train, val, test):
    logging.basicConfig(filename=f'{model_name}.log', level=logging.INFO)
    if task == 'segmentation':
        # Set Training Precision
        mixed_precision.set_global_policy('mixed_float16')

        model = create_segmentation_model()
        EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
        gc.collect()
        gc.enable()
        model, history = train_segmentation_model(model, train, val)
        test_loss, test_acc = model.evaluate(test)
        model.save(f'{model_name}.h5')

        # Log the information
        log_model_performance(model, test, test_loss, test_acc)


    elif task == 'detection':  
        base_model = define_base_model()
        model = create_detection_model(base_model)
        EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
        gc.collect()    
        gc.enable()
        model, history = train_detection_model(model, train, val)
        test_loss, test_acc = model.evaluate(test)
        model.save(f'{model_name}.h5')

        # Log the information
        log_model_performance(model, test, test_loss, test_acc)