import matplotlib.pyplot as plt
from dataset_preprocessing.pascal_dataset import train_loader, val_loader

def main():
    # Iterate over the train loader
    for images, segmentation_masks, annotations in train_loader:
        # Print the shape of the batch
        print("Images shape:", images.shape)
        print("Segmentation masks shape:", segmentation_masks.shape)
        print("Annotations:", annotations)

        # Plot a sample image and segmentation mask from the batch
        sample_image = images[0].permute(1, 2, 0)
        sample_segmentation_mask = segmentation_masks[0][0]
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(sample_image)
        plt.title("Sample Image")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.imshow(sample_segmentation_mask, cmap='gray')
        plt.title("Sample Segmentation Mask")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
