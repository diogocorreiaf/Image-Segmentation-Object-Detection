import os
from PIL import Image
import numpy as np
from xml.etree import ElementTree as ET
from torch.utils.data import Dataset

train_path = 'C:\\Users\\diogo\\Documents\\UVT\\THESIS\\Dataset\\VOC2012_train_val'
test_path = 'C:\\Users\\diogo\\Documents\\UVT\\THESIS\\Dataset\\VOC2012_test'

class PascalDataset(Dataset):
    def __init__(self, base_dir, split='train', is_test=False):
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')
        self._anno_dir = os.path.join(self._base_dir, 'Annotations')

        # Determine if it's a test dataset
        self.is_test = is_test

        # Ensure every split will always be a list, this makes coding easier
        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split

        # Necessary Directories
        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        # Storing
        self.im_ids = []
        self.images = []
        self.categories = []
        self.annotations = [] if not is_test else None  # No annotations for test dataset

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image), f"Image file not found: {_image}"
                assert os.path.isfile(_cat), f"Segmentation mask file not found: {_cat}"
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)
                _anno = os.path.join(self._anno_dir, line + ".xml")
                assert os.path.isfile(_anno), f"Annotation file not found: {_anno}"
                self.annotations.append(_anno)

            # Assertions for image and segmentation mask loading
            assert (len(self.images) == len(self.categories))

            if not is_test:
                assert len(self.images) == len(self.annotations), "Number of images does not match number of annotations"

                for img_file, anno_file in zip(self.images, self.annotations):
                    assert os.path.isfile(img_file), f"Image file not found: {img_file}"
                    assert os.path.isfile(anno_file), f"Annotation file not found: {anno_file}"

        print('Number of images in {}: {:d}'.format(split, len(self.images)))



    def _load_annotation(self, annotation_path):
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            object_labels = []
            bounding_boxes = []

            for obj in root.findall('object'):
                object_label = obj.find('name').text
                object_labels.append(object_label)

                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)
                bounding_boxes.append([xmin, ymin, xmax, ymax])

            annotation_data = {
                'object_labels': object_labels,
                'bounding_boxes': bounding_boxes
            }

            num_annotations = len(object_labels)
            print(f"Number of annotations loaded from {annotation_path}: {num_annotations}")

            return annotation_data

    def __len__(self):
            return len(self.images)



    def __getitem__(self, index):
        _img, _target, _anno = self._pair_img_mask_anno(index)
        sample = {'image': _img, 'segmentation_mask': _target, 'annotations': _anno}
        return sample


    def _pair_img_mask_anno(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _target = Image.open(self.categories[index])
        _anno = self._load_annotation(self.annotations[index])  

        return _img, _target, _anno


if __name__ == '__main__':
    train_dataset = PascalDataset(base_dir=train_path, split='train')
    val_dataset = PascalDataset(base_dir=train_path, split='val')
   # test_dataset = PascalDataset(base_dir=test_path, split='test', is_test=True)
    sample = train_dataset[0]
    print(sample)
