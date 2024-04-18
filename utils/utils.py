import random
import os

def dataset_randomizer(dataset_path):
    ''' Randomizes the Dataset, acesses the trainval.txt and splits it between the train, val and test.txt file
        
        Args:
        - dataset_path (str): The path to the dataset directory. 
        '''
    new_path = os.path.join(dataset_path,"VOC2012_train_val","VOC2012_train_val","ImageSets","Segmentation","trainval.txt")
    with open(new_path, 'r') as file:
        lines = file.readlines()

    random.shuffle(lines)

    total_lines = len(lines)
    train_lines = int(total_lines * 0.6)
    validation_lines = int(total_lines * 0.25)
    test_lines = total_lines - train_lines - validation_lines

    with open('train.txt', 'w') as train_file:
        train_file.writelines(lines[:train_lines])

    with open('validation.txt', 'w') as validation_file:
        validation_file.writelines(lines[train_lines:train_lines + validation_lines])

    with open('test.txt', 'w') as test_file:
        test_file.writelines(lines[train_lines + validation_lines:])

