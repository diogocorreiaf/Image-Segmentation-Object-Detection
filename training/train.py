import tensorflow as tf
import os
import gc
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import mixed_precision
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, average_precision_score
from utils.utils import get_path
from models.detection_models import create_detection_model, yolo_loss
from models.segmentation_models import create_segmentation_model
from collections import Counter
from utils.utils import compute_iou2, compute_metrics






def log_det_model_performance(model, model_name, test, test_loss):
    os.makedirs('saved_models/detection_models', exist_ok=True)
    log_file = f'saved_models/detection_models/{model_name}.log'
    
    with open(log_file, 'w') as f:
        mean_precision = tf.keras.metrics.Mean()
        mean_recall = tf.keras.metrics.Mean()
        mean_f1_score = tf.keras.metrics.Mean()
        mean_iou = tf.keras.metrics.Mean()
        
        for image, labels in test:
            predictions = model.predict(image)
            iou = compute_iou2(predictions, labels)
            precision, recall, f1_score = compute_metrics(predictions, labels)
            mean_precision.update_state(precision)
            mean_recall.update_state(recall)
            mean_f1_score.update_state(f1_score)
            mean_iou.update_state(iou)

        mean_precision_result = mean_precision.result().numpy().item()
        mean_recall_result = mean_recall.result().numpy().item()
        mean_f1_score_result = mean_f1_score.result().numpy().item()
        mean_iou_result = mean_iou.result().numpy().item()
        
        f.write(f'Test loss: {test_loss:.4f}\n')
        f.write(f'Precision: {mean_precision_result:.4f}\n')
        print("Precision: ", mean_precision_result)
        f.write(f'Recall: {mean_recall_result:.4f}\n')
        print("Recall: ", mean_recall_result) 
        f.write(f'F1 Score: {mean_f1_score_result:.4f}\n')
        print("F1: ", mean_f1_score_result) 
        f.write(f'Intersection over Union: {mean_iou_result:.4f}\n')
        print("IoU: ", mean_iou_result) 
        f.write(f'Hyperparameters: {model.optimizer.get_config()}\n')
        
        def model_summary_to_file(line):
            f.write(line + '\n')
        
        model.summary(print_fn=model_summary_to_file)
    
    
    
def log_seg_model_performance(model, model_name, test, test_loss, test_acc):
    os.makedirs('saved_models/segmentation_models', exist_ok=True)
    log_file = f'saved_models/segmentation_models/{model_name}.log'
    
    with open(log_file, 'w') as f:
        f.write(f'Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}\n')
        
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
        precision = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        recall = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        f1 = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
        
        f.write(f'Confusion Matrix: \n{cm}\n')
        print("Confusion Matrix: \n", cm)  # Debugging print
        f.write(f'Precision: {precision:.4f}\n')
        print("Precision: ", precision)  # Debugging print
        f.write(f'Recall: {recall:.4f}\n')
        print("Recall: ", recall)  # Debugging print
        f.write(f'F1 Score: {f1:.4f}\n')
        print("F1 Score: ", f1)  # Debugging print
        
        true_class_counts = Counter(y_true_all)
        pred_class_counts = Counter(y_pred_all)
        f.write(f'True class distribution: {true_class_counts}\n')
        f.write(f'Predicted class distribution: {pred_class_counts}\n')
        
        f.write(f'Hyperparameters: {model.optimizer.get_config()}\n')
        
        def model_summary_to_file(line):
            f.write(line + '\n')
        
        model.summary(print_fn=model_summary_to_file)
    
    
def train_detection_model(model, model_name, Train, Val, Test, Batchsize=2, Epochs=50):
    #Set callbacks
    mixed_precision.set_global_policy('mixed_float16')
    gc.collect()
    gc.enable()
    
    Early, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=1e-5)
    #Train model
    history = model.fit(Train, validation_data=Val, batch_size=Batchsize, epochs=Epochs, callbacks=[Early, Checkpoint, Tensorboard,reduce_lr ])
    best_checkpoint = Checkpoint.filepath.format(epoch=Early.stopped_epoch, val_loss=min(history.history['val_loss']))
    model.load_weights(best_checkpoint)
    
    #Save model
    os.makedirs('saved_models/detection_models', exist_ok=True)
    model_path = os.path.join('saved_models', 'detection_models', model_name + '.keras')
    model.save(model_path)
    
    #Logging the model
    pd.DataFrame(history.history).plot(figsize = (10,8))
    
    plt.grid('True')
    plt.savefig( model_name+".png")
    plt.show()
    test_loss = model.evaluate(Test)
    log_det_model_performance(model, model_name, Test, test_loss)





def train_segmentation_model(model, model_name, Train, Val, Test, Batchsize, Epochs):
    # Set mixed precision policy
    mixed_precision.set_global_policy('mixed_float16')
    
    # Garbage collection
    gc.collect()
    gc.enable()
    
    # Set callbacks
    Early, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-5)
    
    # Train model
    history = model.fit(Train, validation_data=Val, batch_size=Batchsize, epochs=Epochs, callbacks=[Early, Checkpoint, Tensorboard, reduce_lr])
    
    # Load best weights
    best_checkpoint = Checkpoint.filepath
    model.load_weights(best_checkpoint)
    
    # Save model
    os.makedirs('saved_models/segmentation_models', exist_ok=True)
    model.save(os.path.join('saved_models', 'segmentation_models', model_name + '.keras'))
    
    # Plot and save training history
    history_df = pd.DataFrame(history.history)
    history_df.plot(figsize=(10, 8))
    plt.grid(True)
    plt.savefig(model_name + ".png")
    plt.show()
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(Test)
    
    # Log model performance
    log_seg_model_performance(model, model_name, Test, test_loss, test_accuracy)


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
    EarlyStop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    checkpoint_path = os.path.join(os.curdir, "checkpoints", "best_model.h5")
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss', save_weights_only=True)
    Tensorboard = tf.keras.callbacks.TensorBoard(log_dir=get_path())
    return EarlyStop, Checkpoint, Tensorboard, checkpoint_path
