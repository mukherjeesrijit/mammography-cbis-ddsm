import cv2
import numpy as np

# Load the original image and the mask
image1 = cv2.imread('image1.jpg')
mask1 = cv2.imread('mask1.jpg', cv2.IMREAD_GRAYSCALE)  # Load the mask as a grayscale image

# Threshold the mask to create a binary image (if not already binary)
_, binary_mask = cv2.threshold(mask1, 1, 255, cv2.THRESH_BINARY)

# Find contours in the binary mask
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get the largest contour (if there are multiple)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Get the bounding box coordinates
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop the mask to the bounding box
    mask1 = mask1[y:y+h, x:x+w]

# Resize the mask to match the size of the original image
mask1_resized = cv2.resize(mask1, (image1.shape[1], image1.shape[0]))

# Create a color version of the mask (e.g., red color)
mask_colored = np.zeros_like(image1)  # Create an empty image with the same shape as the original
mask_colored[:, :, 2] = mask1_resized  # Assign the mask to the red channel (for a red overlay)

# Convert the grayscale mask to a 3-channel image for transparency
mask_colored = cv2.merge([np.zeros_like(mask1_resized), np.zeros_like(mask1_resized), mask1_resized])

# Define the transparency level (alpha). 0 = fully transparent, 1 = fully opaque
alpha = 0.2  # Adjust transparency level

# Overlay the mask onto the image using cv2.addWeighted
overlay_image = cv2.addWeighted(image1, 1, mask_colored, alpha, 0)

# Save the result
cv2.imwrite('overlay_image.jpg', overlay_image)

# Display the result
cv2.imshow('Overlay Image', overlay_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
