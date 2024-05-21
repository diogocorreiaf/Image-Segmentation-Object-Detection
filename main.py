import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataset_preprocessing.data_loading import load_and_shuffle_data, create_datasets
from dataset_preprocessing.data_preprocessing import create_data_loader
from utils.utils import dataset_randomizer, get_path
from training.train import train_models
from testing.test import testing_models

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
        train_models(task, model_name, Train, Val, Test)
        
    elif action == 'load':
        model_name = input("Enter the name of the model: ")
        task = input("Enter the task you want to run (either 'segmentation' or 'detection'):")
        img_name = input("Enter the path to the image you want to test: ")
        im_path = os.path.join("image_for_testing", img_name)
        testing_models(task, im_path, model_name)
        
    else:
        raise ValueError("Invalid action. Choose either 'train' or 'load'.")

    return
    

if __name__ == "__main__":
    main()