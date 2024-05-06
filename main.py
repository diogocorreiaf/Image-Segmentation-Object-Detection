import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import gc
from tensorflow import keras
from keras import mixed_precision
from dataset_preprocessing.data_loading import load_image_seg_class_paths, load_image_annotations
from dataset_preprocessing.data_preprocessing import create_data_loader
from utils.utils import dataset_randomizer, get_path
from models.segmentation_models import FCN_VGG8


# Define Path to Dataset
dataset_path = '/mnt/c/Users/diogo/Documents/UVT/THESIS/Dataset'

# Set Training Precision
mixed_precision.set_global_policy('mixed_float16')

def main():

    dataset_randomizer(dataset_path)


    # Storing the list of paired images and segmentation class image
    Train_Segmentation = load_image_seg_class_paths(dataset_path,dataset_type='train')
    Val_Segmentation = load_image_seg_class_paths(dataset_path,dataset_type='validation')
    Test_Segmentation = load_image_seg_class_paths(dataset_path,dataset_type='test')

    '''
    # Storing the list of paired images and annotations
    Train_Object_Detection = load_image_annotations(dataset_path,dataset_type='train')
    Val_Object_Detection = load_image_annotations(dataset_path,dataset_type='validation')
    Test_Object_Detection = load_image_annotations(dataset_path,dataset_type='test')
    '''
    # shuffling the lists
    Train_Segmentation = tf.random.shuffle(Train_Segmentation)
    Val_Segmentation = tf.random.shuffle(Val_Segmentation)    
    Test_Segmentation = tf.random.shuffle(Test_Segmentation)
    '''  
    Train_Object_Detection = tf.random.shuffle(Train_Object_Detection)
    Val_Object_Detection = tf.random.shuffle(Val_Object_Detection)    
    Test_Object_Detection = tf.random.shuffle(Test_Object_Detection)
    '''
    # Create Tensorflow Dataset
    Train_Segmentation = tf.data.Dataset.from_tensor_slices(Train_Segmentation)
    Val_Segmentation = tf.data.Dataset.from_tensor_slices(Val_Segmentation)
    Test_Segmentation = tf.data.Dataset.from_tensor_slices(Test_Segmentation)
    '''
    Train_Object_Detection = tf.data.Dataset.from_tensor_slices(Train_Object_Detection)
    Val_Object_Detection = tf.data.Dataset.from_tensor_slices(Val_Object_Detection)
    Test_Object_Detection = tf.data.Dataset.from_tensor_slices(Test_Object_Detection)
    '''

        
    # Dataset Augmentation and Preprocessing 
    Train_Segmentation = create_data_loader(Train_Segmentation, 'train', 'segmentation')
    Val_Segmentation = create_data_loader(Val_Segmentation,'val', 'segmentation')
    Test_Segmentation = create_data_loader(Test_Segmentation,'test', 'segmentation')

    '''
    Train_Object_Detection = create_data_loader(Train_Segmentation, 'train', 'detection')
    Val_Object_Detection = create_data_loader(Train_Segmentation, 'val', 'detection')
    Test_Object_Detection = create_data_loader(Train_Segmentation, 'test', 'detection')
    '''

    model = FCN_VGG8(224,224,21)
    VGG16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')

    for i in range(19):
        model.layers[i].set_weights(VGG16.layers[i].get_weights())

    root_dir = os.path.join(os.curdir,"my_logs")
    board_log_path = get_path()

    for layers in model.layers[:19]:
        layers.trainable = False


    EarlyStop = tf.keras.callbacks.EarlyStopping(patience = 10,restore_best_weights=True)
    checkpoint_path = os.path.join(os.curdir,"checkpoint")
    Checkpoint = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,save_best_only=True)
    Tensorboard = tf.keras.callbacks.TensorBoard(board_log_path)

    MeanIou = tf.keras.metrics.MeanIoU(num_classes=21)
    gc.collect()
    gc.enable()

    tf.keras.backend.clear_session()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-4,momentum=0.9),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[MeanIou])

    Epochs = 100
    Batchsize = 2

    history = model.fit(Train_Segmentation,validation_data=Val_Segmentation,batch_size=Batchsize,epochs=Epochs,callbacks=[EarlyStop,Checkpoint,Tensorboard])

    model.load_weights(checkpoint_path)
    pd.DataFrame(history.history).plot(figsize = (10,8))
    plt.grid('True')
    plt.savefig("Learning_Curve_Model1.png")
    plt.show()




    return 
    

if __name__ == "__main__":
    main()