import matplotlib.pyplot as plt
from PIL import Image

def visualize_dataset(dataset, num_samples=5):
    plt.figure(figsize=(15, 15))
    idx = 0
    for instance in dataset.take(num_samples):
        plt.subplot(num_samples, 2, idx + 1)
        img = Image.open(instance[0])
        plt.imshow(img)
        plt.subplot(num_samples, 2, idx + 2)
        mask = Image.open(instance[1])
        plt.imshow(mask)
        idx += 2
    plt.show()
