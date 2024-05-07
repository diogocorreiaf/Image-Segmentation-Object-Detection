import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import mixed_precision
from dataset_preprocessing.data_loading import load_and_shuffle_data, create_datasets
from dataset_preprocessing.data_preprocessing import create_data_loader
from utils.utils import dataset_randomizer, get_path
from models.segmentation_models import create_segmentation_model
from models.detection_models import create_detection_model, define_base_model
from training.train import get_callbacks, train_segmentation_model, train_detection_model



# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'


def main():
    task = input("Enter the task you want to run (either 'segmentation' or 'detection'): ")

    dataset_randomizer(dataset_path)
    Train, Val, Test = load_and_shuffle_data(dataset_path, task)
    Train, Val, Test = create_datasets(Train, Val, Test, task)

    
    if task == 'segmentation':
        # Set Training Precision
        mixed_precision.set_global_policy('mixed_float16')
        model = create_segmentation_model()
        EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
        gc.collect()
        gc.enable()
        model, history = train_segmentation_model(model, Train, Val)
    elif task == 'detection':
        base_model = define_base_model()
        model = create_detection_model(base_model)
        EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
        gc.collect()    
        gc.enable()
        model, history = train_detection_model(model, Train, Val)
    else:
        raise ValueError("Invalid task. Choose either 'segmentation' or 'detection'.")


    
    return
    

if __name__ == "__main__":
    main()