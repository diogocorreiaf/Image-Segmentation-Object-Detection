import os
import tensorflow as tf


def read_data(path, is_train=True):
    temp = []
    updated_path = os.path.join(path, "VOC2012_train_val", "VOC2012_train_val", "ImageSets", "Segmentation", "train.txt" if is_train else "val.txt")
    with open(updated_path, "r") as file_:
        instances = file_.read().split()
        for img in instances:
            path_img = os.path.join(path, "VOC2012_train_val", "VOC2012_train_val", "JPEGImages", img + ".jpg")
            path_label = os.path.join(path, "VOC2012_train_val", "VOC2012_train_val", "SegmentationClass", img + ".png")
            temp.append([path_img, path_label])
    return temp

def data_loader(dataset, batch_size=2, buffer_size=2):
    data = dataset.cache().shuffle(buffer_size).batch(batch_size).repeat(1)
    data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
    return data

def create_dataset(data, preprocess_fn):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset

