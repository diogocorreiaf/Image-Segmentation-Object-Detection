import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import gc
from tensorflow import keras
from keras import mixed_precision
from dataset_preprocessing.data_loading import load_and_shuffle_data, create_datasets
from dataset_preprocessing.data_preprocessing import create_data_loader
from utils.utils import dataset_randomizer, get_path
from models.segmentation_models import create_model
from training.train import get_callbacks



# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

# Set Training Precision
mixed_precision.set_global_policy('mixed_float16')

def main():
    dataset_randomizer(dataset_path)
    Train, Val, Test = load_and_shuffle_data(dataset_path, 'segmentation')
    Train, Val, Test = create_datasets(Train, Val, Test, 'segmentation')

    model = create_model()

    EarlyStop, Checkpoint, Tensorboard, checkpoint_path = get_callbacks()

    gc.collect()
    gc.enable()

    tf.keras.backend.clear_session()

    Epochs = 100
    Batchsize = 2

    history = model.fit(Train,validation_data=Val,batch_size=Batchsize,epochs=Epochs,callbacks=[EarlyStop,Checkpoint,Tensorboard])

    model.load_weights(checkpoint_path)
    pd.DataFrame(history.history).plot(figsize = (10,8))
    plt.grid('True')
    plt.savefig("Learning_Curve_Model1.png")
    plt.show()

    return
    

if __name__ == "__main__":
    main()