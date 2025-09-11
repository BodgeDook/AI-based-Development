# Importing the main libs for our task:

import cv2
import numpy as np

# Using the OpenCV methods to build the mask of the green apple:

# Step 1: Uploading the image (we will use synthetic data)
img = cv2.imread('green_apple.jpg')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Step 2: Making the ranges of green colour

# light green:
lower_green_light = np.array([30, 40, 60])
upper_green_light = np.array([85, 255, 255])

# dark green:
lower_green_dark = np.array([25, 30, 0])
upper_green_dark = np.array([95, 255, 100])

# Step 3: Masks for light and dark green
mask_light = cv2.inRange(hsv, lower_green_light, upper_green_light)
mask_dark = cv2.inRange(hsv, lower_green_dark, upper_green_dark)

# Step 4: Combining masks
mask = cv2.bitwise_or(mask_light, mask_dark)

# Step 5: Morphology
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
mask = cv2.dilate(mask, kernel, iterations = 2)

# Step 6: Filling contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
mask_filled = np.zeros_like(mask)
cv2.drawContours(mask_filled, contours, -1, 255, thickness = -1)
mask = mask_filled

# Step 7: Inverting and applying a mask
mask_inv = cv2.bitwise_not(mask)
result = cv2.bitwise_and(img, img, mask = mask_inv)

# Step 8: Saving and ending
cv2.imwrite('green_apple_mask.png', result)
print('done')
