import tensorflow as tf
import os
from utils.utils import get_path


def get_callbacks():
    EarlyStop = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights=True)
    checkpoint_path = os.path.join(os.curdir,"checkpoint")
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True)
    root_dir = os.path.join(os.curdir,"my_logs")
    board_log_path = get_path()
    Tensorboard = tf.keras.callbacks.TensorBoard(board_log_path)
    return EarlyStop, Checkpoint, Tensorboard, checkpoint_path

