import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from U_net_optimized import OptimizedUNet

def load_test_image_from_dataset():
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("muchlisinadi/lung_dataset")
    test_dataset = dataset['train']  # Using train split as test

    # Get a random image
    idx = np.random.randint(len(test_dataset))
    image = np.array(test_dataset[idx]['image'])
    mask = np.array(test_dataset[idx]['label'])

    # Add channel dimension if needed
    if len(image.shape) == 2:
        image = image[np.newaxis, ...]
    if len(mask.shape) == 2:
        mask = mask[np.newaxis, ...]

    # Normalize image and mask
    image = (image - image.min()) / (image.max() - image.min()) if image.max() > image.min() else image
    mask = (mask - mask.min()) / (mask.max() - mask.min()) if mask.max() > mask.min() else mask

    # Convert to tensors
    image = torch.from_numpy(image).float().unsqueeze(0)  # Add batch dimension
    mask = torch.from_numpy(mask).float().unsqueeze(0)

    # Ensure mask has valid shape
    if mask.numel() == 1 or mask.dim() != 4:
        print("Warning: Mask has invalid shape. Replacing with zeros.")
        mask = torch.zeros_like(image)

    return image, mask

def predict(model, image):
    model.eval()
    with torch.no_grad():
        prediction = model(image)
        prediction = torch.sigmoid(prediction)
    return prediction

def visualize_results(original_image, true_mask, prediction):
    # Convert tensors to numpy arrays
    original_image = original_image.squeeze().numpy()
    true_mask = true_mask.squeeze().numpy()
    prediction = prediction.squeeze().numpy()

    # Handle invalid shapes
    if true_mask.ndim != 2:
        print("Warning: True mask has invalid shape. Skipping visualization.")
        true_mask = np.zeros_like(original_image)

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Plot original image
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot true mask
    ax2.imshow(true_mask, cmap='gray')
    ax2.set_title('True Mask')
    ax2.axis('off')

    # Plot prediction
    ax3.imshow(prediction, cmap='gray')
    ax3.set_title('Predicted Mask')
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

def dice_coefficient(pred, target):
    smooth = 1e-5
    pred = pred > 0.5  # Convert to binary
    target = target > 0.5
    intersection = (pred & target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def main():
    # Load model
    print("Loading model...")
    model = OptimizedUNet()
    if torch.cuda.is_available():
        model = model.cuda()
        checkpoint = torch.load('saved_unet_model.pth')
    else:
        checkpoint = torch.load('saved_unet_model.pth', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint)
    print("Model loaded successfully!")

    # Load test image and mask
    image, true_mask = load_test_image_from_dataset()
    print("Test image loaded successfully!")

    # Move to GPU if available
    if torch.cuda.is_available():
        image = image.cuda()
        true_mask = true_mask.cuda()

    # Get prediction
    print("Making prediction...")
    prediction = predict(model, image)

    # Move back to CPU for visualization
    if torch.cuda.is_available():
        image = image.cpu()
        true_mask = true_mask.cpu()
        prediction = prediction.cpu()

    # Visualize results
    print("Visualizing results...")
    visualize_results(image, true_mask, prediction)

    # Calculate and print Dice coefficient
    dice = dice_coefficient(prediction.numpy(), true_mask.numpy())
    print(f"Dice coefficient: {dice:.4f}")

if __name__ == "__main__":
    main()

