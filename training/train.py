import tensorflow as tf
import os
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import mixed_precision
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from utils.utils import get_path
from models.detection_models import yolo_loss,define_base_model, create_detection_model
from models.segmentation_models import create_segmentation_model




def log_model_performance(model, model_name, test, test_loss, test_acc):
    os.makedirs('saved_models', exist_ok=True)
    logging.basicConfig(filename=f'saved_models/{model_name}.log', level=logging.INFO)

    for test_images, test_masks in test: 
        y_pred = np.argmax(model.predict(test_images), axis=-1)
        y_true = np.argmax(test_masks, axis=-1) 
        cm = confusion_matrix(y_true.flatten(), y_pred.flatten()) 
        precision = precision_score(y_true.flatten(), y_pred.flatten(), average='weighted')
        recall = recall_score(y_true.flatten(), y_pred.flatten(), average='weighted')
        f1 = f1_score(y_true.flatten(), y_pred.flatten(), average='weighted') 
        logging.info(f'Confusion Matrix: {cm}')
        logging.info(f'Precision: {precision}')
        logging.info(f'Recall: {recall}')
        logging.info(f'F1 Score: {f1}')

    logging.info(f'Hyperparameters: {model.optimizer.get_config()}')
    logging.info(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
    model.summary(print_fn=logging.info)

def train_detection_model(model, Train, Val, Batchsize=2, Epochs=50):
    def scheduler(epoch, lr):
        if epoch < 40:
            return 1e-3
        elif epoch >= 40 and epoch < 80:
            return 5e-4
        else:
            return 1e-4

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    checkpoint_filepath = '/content/drive/MyDrive/Bang/yolo_efficientnet_b1_new.h5'
    callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True
    )

    model.compile(
        loss=yolo_loss,
        optimizer=tf.keras.optimizers.Adam(1e-3),
    )

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
    # Step 1: Set up the environment
    mixed_precision.set_global_policy('mixed_float16')
    gc.collect()
    gc.enable()

    # Step 2: Define the model
    if task == 'segmentation':
        model = create_segmentation_model()
    elif task == 'detection':
        base_model = define_base_model()
        model = create_detection_model(base_model)

    # Step 4: Set up callbacks
    EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()

    # Ensure checkpoint directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Step 5: Train the model
    history = model.fit(train, validation_data=val, batch_size=2, epochs=10, callbacks=[EarlyStop, Checkpoint, Tensorboard])
    best_checkpoint = Checkpoint.filepath.format(epoch=EarlyStop.stopped_epoch, val_loss=min(history.history['val_loss']))
    model.load_weights(best_checkpoint)

    test_loss, test_accuracy = model.evaluate(test)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")

    log_model_performance(model, model_name, test, test_loss, test_accuracy)

    os.makedirs('saved_models', exist_ok=True)
    model.save(os.path.join('saved_models', model_name + '.keras'))

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