import os
from PIL import Image
import numpy as np
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset

train_path = 'C:\\Users\\diogo\\Documents\\UVT\\THESIS\\Dataset\\VOC2012_train_val'
test_path = 'C:\\Users\\diogo\\Documents\\UVT\\THESIS\\Dataset\\VOC2012_test'

class PascalDataset(Dataset):

    def __init__(self,
                 args,
                 base_dir,
                 split='train',
                 test=False):

        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self._anno_dir = os.path.join(self._base_dir, 'Annotations')

        # Loading Test Dataset
        if test:
            self._base_dir = test_path
            self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
            self._anno_dir = os.path.join(self._base_dir, 'Annotations')

        # Ensure every split will always be a list, this makes coding easier
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        # Necessary Directories
        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []
        self.annotations = []  # Store annotations here

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                assert os.path.isfile(_image)
                self.im_ids.append(line)
                self.images.append(_image)

                # Load annotation if available
                _anno = os.path.join(self._anno_dir, line + ".xml")
                print(_anno)
                assert os.path.isfile(_anno)
                self.annotations.append(_anno)

        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __getitem__(self, index):
        image_path = self.images[index]
        annotation_path = self.annotations[index] if self.annotations else None
        # Load image
        image = Image.open(image_path)

        # Load annotation if available
        annotation = None
        if annotation_path:
            annotation = self._load_annotation(annotation_path)

        return image, annotation

def _load_annotation(self, annotation_path):
    # Parse XML annotation file
    tree = ET.parse(annotation_path)
    root = tree.getroot()

    # Initialize lists to store annotation data
    object_labels = []
    bounding_boxes = []

    # Extract object class labels and bounding box coordinates
    for obj in root.findall('object'):
        # Extract object class label
        object_label = obj.find('name').text
        object_labels.append(object_label)

        # Extract bounding box coordinates
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        bounding_boxes.append([xmin, ymin, xmax, ymax])

    # Combine object labels and bounding box coordinates into annotation data
    annotation_data = {
        'object_labels': object_labels,
        'bounding_boxes': bounding_boxes
    }
        # Print the number of annotations loaded
    num_annotations = len(object_labels)
    print(f"Number of annotations loaded from {annotation_path}: {num_annotations}")

    return annotation_data


    def __len__(self):
        return len(self.images)

if __name__ == '__main__':
    train_dataset = PascalDataset(args=None, base_dir=train_path, split='train')
    val_dataset = PascalDataset(args=None, base_dir=train_path, split='val')
    test_dataset = PascalDataset(args=None, base_dir=test_path, split='test', test=True)
