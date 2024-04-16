import tensorflow as tf
from dataset_preprocessing.data_loading import read_data, data_loader
from dataset_preprocessing.data_preprocessing import preprocess
from models.models import create_model

def main():
    path = 'C:\\Users\\diogo\\Documents\\UVT\\THESIS\\Dataset'
    train_data = read_data(path=path, is_train=True)
    val_data = read_data(path=path, is_train=False)

    train_data = tf.random.shuffle(train_data)
    val_data = tf.random.shuffle(val_data)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
    val_dataset = tf.data.Dataset.from_tensor_slices(val_data)

    train_dataset = data_loader(train_dataset)
    val_dataset = data_loader(val_dataset)

    model = create_model()


if __name__ == "__main__":
    main()
