import random
import os

root_dir = os.path.join(os.curdir,"my_logs")


def dataset_randomizer(dataset_path):
    ''' Randomizes the Dataset, acesses the trainval.txt and splits it between the train, val and test.txt file
        
        Args:
        - dataset_path (str): The path to the dataset directory. 
        '''
    trainval_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","trainval.txt")
    train_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","train.txt")
    val_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","val.txt")
    test_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","test.txt")


    with open(trainval_path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    total_lines = len(lines)
    train_lines = int(total_lines * 0.7)
    validation_lines = int(total_lines * 0.15)

    with open(train_path, 'w') as train_file:
        train_file.truncate(0)
        train_file.writelines(lines[:train_lines])

    with open(val_path, 'w') as validation_file:
        validation_file.truncate(0)
        validation_file.writelines(lines[train_lines:train_lines + validation_lines])

    with open(test_path, 'w') as test_file:
        test_file.truncate(0)
        test_file.writelines(lines[train_lines + validation_lines:])


def get_path():
    import time
    id_ = time.strftime("run_%Y_%m_%D_%H_%M_%S")
    return os.path.join(root_dir,id_)