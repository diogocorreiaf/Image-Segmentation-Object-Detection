import gradio as gr
import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from utils.utils import dataset_randomizer
from dataset_preprocessing.data_loading import load_and_shuffle_data, create_datasets



model_files = os.listdir('saved_models')
task = "segmentation"



def datasetloader(train_split_value,val_split_value):
    if train_split_value + val_split_value >0.9:
        gr.Info("The sum of the train and validation split values must be less than 0.9 to allow space for the test split.")
        return
    dataset_randomizer(train_split_value, val_split_value)
    Train, Val, Test = load_and_shuffle_data(task)
    Train, Val, Test = create_datasets(Train, Val, Test, task)
    return f"Loaded {len(Train)*2} training images, {len(Val)*2} validation images, and {len(Test)*2} test images."

# Create the welcome interface block
with gr.Blocks(title="Image Segmentation") as image_segmentation:
    #with gr.Column(min_width=400, scale=1):
        with gr.Row():
            with gr.Column(min_width=400, scale=1):
                gr.Label("Dataset Preprocessing",show_label=False)
                train_split_value = gr.Slider(minimum=0, maximum=0.9, label="Train Split Size", value=0.6)
                val_split_value = gr.Slider(minimum=0, maximum=0.9, label="Validation Split Size", value=0.2)
                randomize_btn = gr.Button("Loading and Preprocess Dataset")
                randomize_btn.click(datasetloader, inputs=[train_split_value, val_split_value], outputs=gr.Textbox())
                
            
                gr.Label("Model Definition and Hyperparemeter Tuning")
                    
                    
                    
                       
            with gr.Column():
                gr.Label("Model Training",show_label=False)
                model_name = gr.Textbox(label="Model Name", placeholder="Enter the model name")
                epochs = gr.Slider(minimum=1, maximum=100, label="Number of Epochs", value=10)
                train_btn = gr.Button("Train Model")
                
            with gr.Column():
                gr.Label("Model Testing",show_label=False)
                loaded_model = gr.Dropdown(label="Select Model", choices=model_files)
                testing_img = gr.Image(label="Test Image") 
                predict_btn = gr.Button("Predict")  
                
                           
if __name__ == "__main__":
    image_segmentation.launch()