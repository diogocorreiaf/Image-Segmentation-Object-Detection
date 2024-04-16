import numpy as np
import tensorflow as tf
from PIL import Image
from albumentations import RandomRotate90

def create_mask(img, num_classes=21):
    seg_labels = np.zeros((img.shape[0], img.shape[1], num_classes), dtype=np.float16)
    for class_ in range(num_classes):
        seg_labels[:, :, class_] = (img == class_)
    return tf.cast(seg_labels, dtype=tf.float16)

def create_preprocess_mask_img(instance, img_width=224, img_height=224):
    img = Image.open(instance[0].numpy())
    img = img.resize((img_width, img_height), resample=Image.BILINEAR)
    img = np.asarray(img)

    mask = Image.open(instance[1].numpy())
    mask = mask.resize((img_width, img_height), resample=Image.BILINEAR)
    mask = np.asarray(mask)

    normalization = tf.keras.layers.Rescaling(1./255)

    if tf.random.uniform(()) > 0.5: 
        aug = RandomRotate90(p=0.5)
        augmented = aug(image=img, mask=mask)
        img = augmented["image"]
        mask = augmented["mask"]

    return normalization(img), create_mask(mask)

def preprocess(instance):
    img, mask = tf.py_function(create_preprocess_mask_img, [instance], [tf.float16, tf.float16])
    return tf.ensure_shape(img, [None, None, 3]), tf.ensure_shape(mask, [None, None, 21])

def data_loader(dataset, batch_size=2, buffer_size=2):
    data = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    data = data.cache().shuffle(buffer_size).batch(batch_size).repeat(1)
    data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
    return data