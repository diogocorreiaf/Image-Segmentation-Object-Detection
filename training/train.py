import tensorflow as tf
import os
from utils.utils import get_path
from models.detection_models import yolo_loss

def get_callbacks():
    EarlyStop = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights=True)
    checkpoint_path = os.path.join(os.curdir,"checkpoint")
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True)
    root_dir = os.path.join(os.curdir,"my_logs")
    board_log_path = get_path()
    Tensorboard = tf.keras.callbacks.TensorBoard(board_log_path)
    return EarlyStop, Checkpoint, Tensorboard, checkpoint_path

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