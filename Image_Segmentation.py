import gradio as gr
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from utils.utils import dataset_randomizer
from dataset_preprocessing.data_loading import load_and_shuffle_data, create_datasets
from models.segmentation_models import create_segmentation_model
from training.train import train_segmentation_model
from testing.test import gui_segmentation_model_test

model_files = [f for f in os.listdir('saved_models/segmentation_models') if f.endswith('.keras')]
task = "segmentation"




def datasetloader(train_split_value,val_split_value):
    if train_split_value + val_split_value >0.9:
        gr.Info("The sum of the train and validation split values must be less than 0.9 to allow space for the test split.")
        return
    dataset_randomizer(task,train_split_value, val_split_value)
    global Train, Val, Test
    Train, Val, Test = load_and_shuffle_data(task)
    Train, Val, Test = create_datasets(Train, Val, Test, task)
    return f"Loaded {len(Train)*2} training images, {len(Val)*2} validation images, and {len(Test)*2} test images."

def create_model(transfer_learning, learning_rate, momentum, optimizer, dropout_rate, activation, kernel_initializer):
    global model
    model = create_segmentation_model(transfer_learning, learning_rate, momentum, optimizer, dropout_rate, activation, kernel_initializer)
    if model is not None:
        return "Model created successfully."
    else:
        return "Error creating model."
    
    
def train_model(model_name, batch_size, epochs):
    train_segmentation_model(model,model_name, Train, Val, Test, batch_size, epochs)
    return "Model trained successfully."
    
def predict_model(loaded_model, testing_img):
    output = gui_segmentation_model_test(loaded_model, testing_img)
    return output    
    
def reset_values():
    train_split_value.value = 0.6
    val_split_value.value = 0.2
    learning_rate.value = 1e-4
    momentum.value = 0.9
    optimizer.value = "SGD"
    dropout_rate.value = 0.5
    activation.value = "relu"
    kernel_initializer.value = "zeros"
    transfer_learning.value = False
    model_name.value = ""
    epochs.value = 50
    batch_size.value = 2
    loaded_model.value = model_files[0] if model_files else None
    testing_img.value = None
    return "Values reset to default."



with gr.Blocks(title="Image Segmentation") as image_segmentation:
    #with gr.Column(min_width=400, scale=1):
        with gr.Row():
            with gr.Column(min_width=400, scale=1):
                gr.Label("Dataset Preprocessing",show_label=False)
                train_split_value = gr.Slider(minimum=0, maximum=0.9, label="Train Split Size", value=0.6)
                val_split_value = gr.Slider(minimum=0, maximum=0.9, label="Validation Split Size", value=0.2)
                randomize_btn = gr.Button("Loading and Preprocess Dataset")
                out_box = gr.Textbox(show_label=False,placeholder="Tab1")
                randomize_btn.click(fn=datasetloader, inputs=[train_split_value, val_split_value], outputs=out_box)
            
                gr.Label("Model Definition and Hyperparemeter Tuning", show_label=False)
                learning_rate = gr.Slider(minimum=1e-6, maximum=1e-2, label="Learning Rate", value=1e-4)
                momentum = gr.Slider(minimum=0, maximum=1, label="Momentum", value=0.9)
                optimizer = gr.Dropdown(label="Optimizer", choices=["SGD", "Adam", "RMSprop","Adagrad", "Nadam"], value="SGD")           
                dropout_rate = gr.Slider(minimum=0, maximum=1, label="Dropout Rate", value=0.5)
                activation = gr.Dropdown(label="Activation", choices=["relu", "tanh", "sigmoid"], value="relu")
                kernel_initializer = gr.Dropdown(label="Kernel Initializer", choices=["zeros", "ones", "random_normal", "random_uniform"], value="zeros") 
                transfer_learning = gr.Checkbox(label="Trainable", info='Utilize Transfer Learning')        
                create_model_btn = gr.Button("Create Model")
                create_model_btn.click(create_model, inputs=[transfer_learning, learning_rate, momentum, optimizer, dropout_rate, activation, kernel_initializer], outputs=gr.Textbox(show_label=False))    
                       
            with gr.Column():
                gr.Label("Model Training",show_label=False)
                model_name = gr.Textbox(label="Model Name", placeholder="Enter the model name")
                epochs = gr.Slider(minimum=1, maximum=200, label="Number of Epochs", value=50)
                batch_size = gr.Slider(minimum=1, maximum=32, label="Batch Size", value=2)
                train_btn = gr.Button("Train Model")
                train_btn.click(train_model, inputs=[model_name, epochs, batch_size], outputs=gr.Textbox(show_label=False))
                
                
            with gr.Column():
                gr.Label("Model Testing",show_label=False)
                loaded_model = gr.Dropdown(label="Select Model", choices=model_files)
                testing_img = gr.Image(label="Test Image") 
                predict_btn = gr.Button("Predict")  
                predict_btn.click(predict_model, inputs=[loaded_model, testing_img], outputs=gr.Image(label="Segmented Image", format="png"))