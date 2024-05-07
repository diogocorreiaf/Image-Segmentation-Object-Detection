import tensorflow as tf

NUM_FILTERS=512
B=2
N_CLASSES = 21
OUTPUT_DIM=int(N_CLASSES+5*B)
SPLIT_SIZE = 7


def yolo_loss(y_true,y_pred):
    target=y_true[...,0]
    #--------------------------------------- for object
    y_pred_extract=tf.gather_nd(y_pred,tf.where(target[:]==1))
    y_target_extract=tf.gather_nd(y_true,tf.where(target[:]==1))
    
    rescaler=tf.where(target[:]==1)*32
    
    upscaler_1=tf.concat([rescaler[:,1:],tf.zeros([len(rescaler),2],dtype=tf.int64)],axis=-1)
    
    
    target_upscaler_2=tf.repeat([[32.,32.,224.,224.]],
                                repeats=[len(rescaler)],axis=0)*tf.cast(y_target_extract[...,1:5],dtype=tf.float32)
    pred_1_upscaler_2=tf.repeat([[32.,32.,224.,224.]],repeats=[len(rescaler)],axis=0)*tf.cast(y_pred_extract[...,1:5],dtype=tf.float32)
    pred_2_upscaler_2=tf.repeat([[32.,32.,224.,224.]],repeats=[len(rescaler)],axis=0)*tf.cast(y_pred_extract[...,6:10],dtype=tf.float32)
    
    
    target_original=tf.cast(upscaler_1,dtype=tf.float32)+target_upscaler_2
    pred_1_original=tf.cast(upscaler_1,dtype=tf.float32)+pred_1_upscaler_2
    pred_2_original=tf.cast(upscaler_1,dtype=tf.float32)+pred_2_upscaler_2
    
    mask=tf.cast(tf.math.greater(compute_iou(target_original,pred_2_original),compute_iou(target_original,pred_1_original)),dtype=tf.int32)
    
    y_pred_joined=tf.transpose(tf.concat([tf.expand_dims(y_pred_extract[...,0],axis=0),tf.expand_dims(y_pred_extract[...,5],axis=0)],axis=-1))

    obj_pred=tf.gather_nd(y_pred_joined,tf.stack([tf.range(len(rescaler)),mask],axis=-1))
    
    object_loss=difference(tf.cast(obj_pred,dtype=tf.float32),tf.cast(tf.ones([len(rescaler)]),dtype=tf.float32))
    
    #------------------------------------------------------ for no object
    
    y_pred_extract=tf.gather_nd(y_pred[...,0:B*5],tf.where(target[:]==0))
    
    y_target_extract=tf.zeros(len(y_pred_extract))

    no_obj_loss_1=difference(tf.cast(y_pred_extract[...,0],dtype=tf.float32),tf.cast(y_target_extract,dtype=tf.float32))
    
    no_obj_loss_2=difference(tf.cast(y_pred_extract[...,5],dtype=tf.float32),tf.cast(y_target_extract,dtype=tf.float32))
    
    no_obj_loss=no_obj_loss_1+no_obj_loss_2
    
    #-------------------------------------------------------- for object class loss
    
    y_pred_extract=tf.gather_nd(y_pred[...,10:],tf.where(target[:]==1))
    class_extract=tf.gather_nd(y_true[...,5:],tf.where(target[:]==1))
    
    class_loss=difference(tf.cast(y_pred_extract,dtype=tf.float32),tf.cast(class_extract,dtype=tf.float32))
    
    #--------------------------------------------------------- for object center loss
    
    y_pred_extract=tf.gather_nd(y_pred[...,0:B*5],tf.where(target[:]==1))
    center_joined=tf.stack([y_pred_extract[...,1:3],y_pred_extract[...,6:8]],axis=1)
    center_pred=tf.gather_nd(center_joined,tf.stack([tf.range(len(rescaler)),mask],axis=-1))
    center_target=tf.gather_nd(y_true[...,1:3],tf.where(target[:]==1))
    
    
    center_loss=difference(tf.cast(center_pred,dtype=tf.float32),tf.cast(center_target,dtype=tf.float32))
    
    
    
    #------------------------------------------------------- for width and height
    
    size_joined=tf.stack([y_pred_extract[...,3:5],y_pred_extract[...,8:10]],axis=-1)
    
    size_pred=tf.gather_nd(size_joined,tf.stack([tf.range(len(rescaler)),mask],axis=-1))
    size_target=tf.gather_nd(y_true[...,3:5],tf.where(target[:]==1))
    
    size_loss=difference(tf.math.sqrt(tf.math.abs(tf.cast(size_pred,dtype=tf.float32))),tf.math.sqrt(tf.math.abs(tf.cast(size_target,dtype=tf.float32))))
    
    
    box_loss=center_loss+size_loss
    
    lambda_coord=5
    lambda_no_obj=0.5
    
    loss=object_loss+(lambda_no_obj*no_obj_loss)+tf.cast(lambda_coord*box_loss,dtype=tf.float32)+tf.cast(class_loss,dtype=tf.float32)
    
    return loss

def difference(x,y):
    return tf.reduce_sum(tf.square(y-x))

def compute_iou(boxes1,boxes2):
    boxes1_t=tf.stack([boxes1[...,0]-boxes1[...,2]/2,
                       boxes1[...,1]-boxes1[...,3]/2,
                       boxes1[...,0]+boxes1[...,2]/2,
                       boxes1[...,1]+boxes1[...,3]/2],axis=-1)
    
    
    boxes2_t=tf.stack([boxes2[...,0]-boxes2[...,2]/2,
                       boxes2[...,1]-boxes2[...,3]/2,
                       boxes2[...,0]+boxes2[...,2]/2,
                       boxes2[...,1]+boxes2[...,3]/2],axis=-1)

    lu=tf.maximum(boxes1_t[...,:2],boxes2_t[...,2:])
    rd=tf.minimum(boxes1_t[...,2:],boxes2_t[...,2:])
    
    intersection=tf.maximum(0.0,rd-lu)
    inter_square=intersection[...,0]*intersection[...,1]
    
    square1=boxes1[...,2]*boxes1[...,3]
    square2=boxes2[...,2]*boxes2[...,3]
    
    union_square=tf.maximum(square1+square2-inter_square,1e-10)

    return tf.clip_by_value((inter_square/union_square),0.0,1.0)


def create_detection_model(base_model):
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Conv2D(NUM_FILTERS,(3,3),padding='same',kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(NUM_FILTERS,(3,3),padding='same',kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Conv2D(NUM_FILTERS,(3,3),padding='same',kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(NUM_FILTERS,kernel_initializer='he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(alpha=0.1),

        tf.keras.layers.Dense(int(SPLIT_SIZE*SPLIT_SIZE*OUTPUT_DIM),activation='sigmoid'),

        tf.keras.layers.Reshape((int(SPLIT_SIZE),int(SPLIT_SIZE),OUTPUT_DIM))
    ])
    return model

def define_base_model():

    base_model = tf.keras.applications.ResNet50(weights='imagenet', input_shape=(224,224,3), include_top=False)
    base_model.trainable = False
    return base_model