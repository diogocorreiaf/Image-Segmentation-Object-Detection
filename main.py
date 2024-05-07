import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import gc
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import mixed_precision
from tensorflow.keras.models import load_model
from test import model_test
from dataset_preprocessing.data_loading import load_and_shuffle_data, create_datasets
from dataset_preprocessing.data_preprocessing import create_data_loader
from utils.utils import dataset_randomizer, get_path
from models.segmentation_models import create_segmentation_model
from models.detection_models import create_detection_model, define_base_model
from training.train import get_callbacks, train_segmentation_model, train_detection_model
from test.testing import segmentation_model_test, detection_model_test 



# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'


def main():
    action = input("Do you want to 'train' a new model or 'load' an existing one? ")

    if action == 'train':
        model_name = input("Enter a name for the model: ")
        
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
            model.save(f'{model_name}.h5')
        elif task == 'detection':
            base_model = define_base_model()
            model = create_detection_model(base_model)
            EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()
            gc.collect()    
            gc.enable()
            model, history = train_detection_model(model, Train, Val)
            model.save(f'{model_name}.h5')
        else:
            raise ValueError("Invalid task. Choose either 'segmentation' or 'detection'.")
    elif action == 'load':
        model_name = input("Enter the name of the model to load: ")
        task = input("Enter the task you want to run (either 'segmentation' or 'detection'): ")
        if task == 'segmentation':
            model = load_model(f'{model_name}.h5')
            test_image_path = input("Enter the path of the image to test: ")
            segmentation_model_test(model, test_image_path)
        elif task == 'detection':
            model = load_model(f'{model_name}.h5')
            test_image_path = input("Enter the path of the image to test: ")
            detection_model_test(model, test_image_path)
    else:
        raise ValueError("Invalid action. Choose either 'train' or 'load'.")

    return
    

if __name__ == "__main__":
    main()