import tensorflow as tf


# TODO Alter the model, make a tuner function to customize the hyperparemeters


def FCN_VGG8(Img_Width,Img_Height,num_classes, dropout_rate = 0.5, activation = "relu", kernel_initializer = "zeros"  ):
  Input = tf.keras.layers.Input(shape = [Img_Width,Img_Height,3])
  Conv1 = tf.keras.layers.Conv2D(64,kernel_size=3,strides = 1,padding="same",activation="relu")(Input)
  Conv2 = tf.keras.layers.Conv2D(64,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv1)
  Pool1 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(Conv2)

  Conv3 = tf.keras.layers.Conv2D(128,kernel_size=3,strides = 1,padding="same",activation="relu")(Pool1)
  Conv4 = tf.keras.layers.Conv2D(128,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv3)
  Pool2 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(Conv4)

  Conv5 = tf.keras.layers.Conv2D(256,kernel_size=3,strides = 1,padding="same",activation="relu")(Pool2)
  Conv6 = tf.keras.layers.Conv2D(256,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv5)
  Conv7 = tf.keras.layers.Conv2D(256,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv6)
  Pool3 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(Conv7)

  Conv8 = tf.keras.layers.Conv2D(512,kernel_size=3,strides = 1,padding="same",activation="relu")(Pool3)
  Conv9 = tf.keras.layers.Conv2D(512,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv8)
  Conv10 = tf.keras.layers.Conv2D(512,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv9)
  Pool4 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(Conv10)

  Conv11 = tf.keras.layers.Conv2D(512,kernel_size=3,strides = 1,padding="same",activation="relu")(Pool4)
  Conv12 = tf.keras.layers.Conv2D(512,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv11)
  Conv13 = tf.keras.layers.Conv2D(512,kernel_size=3,strides = 1,padding="same",activation="relu")(Conv12)
  Pool5 = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(Conv13)

  # Fully Convolutional Layer

  FC_Layer = tf.keras.layers.Conv2D(4096,kernel_size=7,activation="relu")(Pool5)
  FC_Drop = tf.keras.layers.Dropout(rate=dropout_rate)(FC_Layer)
  FC_Layer2 = tf.keras.layers.Conv2D(4096,kernel_size=1,activation="relu")(FC_Drop)
  FC_Drop2 = tf.keras.layers.Dropout(rate=dropout_rate)(FC_Layer2)

  # Classification Score Layer
  Score = tf.keras.layers.Conv2D(num_classes,kernel_size=1,activation="relu")(FC_Drop2)
 
  #Upsample Pool4

  Upscore = tf.keras.layers.Conv2DTranspose(num_classes,kernel_size=4,strides=2,kernel_initializer="zeros")(Score)
  
  Conv_Scale = tf.keras.layers.Conv2D(num_classes,kernel_size=1)(Pool4)
  Cropped = tf.keras.layers.Cropping2D(cropping=(5,5))(Conv_Scale)

  Fused = tf.keras.layers.add([Cropped,Upscore])
  
  Upsampled_Pool4 = tf.keras.layers.Conv2DTranspose(num_classes,kernel_size=4,strides=2,kernel_initializer="zeros")(Fused)

  # Upsample Pool3

  Conv_Scale2 = tf.keras.layers.Conv2D(num_classes,kernel_size=1)(Pool3)
  Cropped2 = tf.keras.layers.Cropping2D(cropping=(9,9))(Conv_Scale2)
  Fused2 = tf.keras.layers.add([Cropped2,Upsampled_Pool4])

  Upsampled_Pool3 = tf.keras.layers.Conv2DTranspose(num_classes,kernel_size=128,strides=16,kernel_initializer="zeros")(Fused2)

  # Score per Pixel

  Score = tf.keras.layers.Cropping2D(cropping=(24,24))(Upsampled_Pool3)
  Score = tf.keras.layers.Softmax(dtype = "float32")(Score)

  return tf.keras.Model(inputs = Input,outputs = Score)



def create_segmentation_model(transfer_learning=True, learning_rate=1e-4, momentum=0.9, optimizer_name='SGD', dropout_rate=0.5, activation='relu', kernel_initializer='zeros'):
    model = FCN_VGG8(224, 224, 21, dropout_rate, activation, kernel_initializer)
    
    VGG16 = tf.keras.applications.vgg16.VGG16(weights='imagenet')
    if transfer_learning:
        for i in range(19):
            model.layers[i].set_weights(VGG16.layers[i].get_weights())

        for layers in model.layers[:19]:
            layers.trainable = False

    tf.keras.backend.clear_session()
    MeanIou = tf.keras.metrics.MeanIoU(num_classes=21)

    if optimizer_name == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'RMSProp':
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == 'Nadam':
        optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=tf.keras.losses.categorical_crossentropy, metrics=[MeanIou])
    return model