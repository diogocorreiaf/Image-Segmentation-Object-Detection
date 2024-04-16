import tensorflow as tf
from dataset_preprocessing.data_loading import read_data, create_dataset
from dataset_preprocessing.data_preprocessing import data_loader, preprocess

path = 'C:\\Users\\diogo\\Documents\\UVT\\THESIS\\Dataset'
Train = read_data(path=path, is_train=True)
Val = read_data(path=path, is_train=False)

Train = create_dataset(Train, preprocess)
Val = create_dataset(Val, preprocess)

Train = data_loader(Train)
Val = data_loader(Val)

Q = []
for X, Y in Val.take(1):
    print("Image shape:", X.shape)
    print("Mask shape:", Y.shape)
    Q = Y
    print("Mask dtype:", Q.dtype)
