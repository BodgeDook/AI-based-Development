# Importing the main libs for the task:
import cv2
import numpy as np
import os
import argparse


def load_and_convert(image_path: str):
    # Loads the image and converts it to HSV color space
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    except cv2.error as e:
        raise RuntimeError(f"Error converting image to HSV: {e}")

    return img, hsv


def create_green_masks(hsv):
    # Creates masks for light-green and dark-green shades
    try:
        lower_green_light = np.array([30, 40, 60])
        upper_green_light = np.array([85, 255, 255])

        lower_green_dark = np.array([25, 30, 0])
        upper_green_dark = np.array([95, 255, 100])

        mask_light = cv2.inRange(hsv, lower_green_light, upper_green_light)
        mask_dark = cv2.inRange(hsv, lower_green_dark, upper_green_dark)

        return cv2.bitwise_or(mask_light, mask_dark)
    except cv2.error as e:
        raise RuntimeError(f"Error creating green masks: {e}")


def apply_morphology(mask, kernel_size: int = 5, dilate_iter: int = 2):
    # Applies morphological operations (closing + dilation)
    if mask is None or mask.size == 0:
        raise ValueError("Invalid mask provided to morphology function")

    try:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
        return mask
    except cv2.error as e:
        raise RuntimeError(f"Error during morphological operations: {e}")


def fill_contours(mask):
    # Fills detected contours in the mask
    if mask is None or mask.size == 0:
        raise ValueError("Invalid mask provided to fill_contours")

    try:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        mask_filled = np.zeros_like(mask)
        cv2.drawContours(mask_filled, contours, -1, 255, thickness=-1)
        return mask_filled
    except cv2.error as e:
        raise RuntimeError(f"Error filling contours: {e}")


def apply_inverse_mask(img, mask):
    # Applies the inverted mask to the image
    if img is None or mask is None:
        raise ValueError("Invalid image or mask for applying inverse mask")

    try:
        mask_inv = cv2.bitwise_not(mask)
        return cv2.bitwise_and(img, img, mask=mask_inv)
    except cv2.error as e:
        raise RuntimeError(f"Error applying inverse mask: {e}")


def process_green_object(image_path: str, save_path: str):
    # Full pipeline for processing a green object
    try:
        img, hsv = load_and_convert(image_path)
        mask = create_green_masks(hsv)
        mask = apply_morphology(mask)
        mask = fill_contours(mask)
        result = apply_inverse_mask(img, mask)

        success = cv2.imwrite(save_path, result)
        if not success:
            raise IOError(f"Failed to save result to {save_path}")

        print(f"Done and saved in {save_path}")
    except Exception as e:
        print(f"Error: {e}")


def parse_args():
    # Parses arguments using argparse
    parser = argparse.ArgumentParser(
        description="Process an image to remove green objects and save the result."
    )
    parser.add_argument(
        "--input_image",
        type=str,
        required=True,
        help="Path to the input image (e.g., picture.jpg)"
    )
    parser.add_argument(
        "--output_image",
        type=str,
        default="output.png",
        help="Path to save the processed image (default: output.png)"
    )
    return parser.parse_args()


# Runs the program:
if __name__ == "__main__":
    args = parse_args()
    process_green_object(args.input_image, args.output_image)
