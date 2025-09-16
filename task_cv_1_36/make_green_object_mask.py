# Importing the main libs for our task:

import cv2
import numpy as np
import os
import argparse


def load_and_convert(image_path: str):
    """
    Loads an image and converts it to HSV color space.

    Args:
        image_path (str): Path to the input image.

    Returns:
        img (np.ndarray): Original BGR image.
        hsv (np.ndarray): Image converted to HSV color space.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return img, hsv


def create_color_mask(hsv, lower_hsv_list, upper_hsv_list):
    """
    Creates a combined mask from multiple HSV ranges.

    Args:
        hsv (np.ndarray): HSV image.
        lower_hsv_list (list of lists): List of lower HSV boundaries [[H,S,V], ...].
        upper_hsv_list (list of lists): List of upper HSV boundaries [[H,S,V], ...].

    Returns:
        mask (np.ndarray): Combined binary mask for all ranges.
    """
    if len(lower_hsv_list) != len(upper_hsv_list):
        raise ValueError("Lower and upper HSV lists must have the same length")

    combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in zip(lower_hsv_list, upper_hsv_list):
        mask = cv2.inRange(hsv, np.array(lower, dtype=np.uint8), np.array(upper, dtype=np.uint8))
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    return combined_mask


def apply_morphology(mask, kernel_size: int = 5, dilate_iter: int = 2):
    """Applies morphological closing and dilation to clean up the mask."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def fill_contours(mask):
    """Fills contours in the mask to make solid regions."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(mask)
    cv2.drawContours(mask_filled, contours, -1, 255, thickness=-1)
    return mask_filled


def apply_inverse_mask(img, mask):
    """Applies an inverted mask to the image."""
    mask_inv = cv2.bitwise_not(mask)
    return cv2.bitwise_and(img, img, mask=mask_inv)


def process_color_object(image_path, save_path, lower_hsv_list, upper_hsv_list):
    """Full pipeline: removes objects in specified HSV ranges from the image."""
    img, hsv = load_and_convert(image_path)
    mask = create_color_mask(hsv, lower_hsv_list, upper_hsv_list)
    mask = apply_morphology(mask)
    mask = fill_contours(mask)
    result = apply_inverse_mask(img, mask)

    success = cv2.imwrite(save_path, result)
    if not success:
        raise IOError(f"Failed to save result to {save_path}")

    print(f"Done and saved in {save_path}")


def parse_args():
    """Parses command-line arguments including multiple HSV ranges."""
    parser = argparse.ArgumentParser(
        description="Remove objects of specific color(s) from an image."
    )
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_image", type=str, default="output.png", help="Path to save result")
    parser.add_argument(
        "--lower_hsv", type=int, nargs="+", required=True,
        help="Lower HSV boundaries for one or more ranges: H S V [H S V ...]"
    )
    parser.add_argument(
        "--upper_hsv", type=int, nargs="+", required=True,
        help="Upper HSV boundaries for one or more ranges: H S V [H S V ...]"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Split the flat list of HSV values into list of 3-element lists
    if len(args.lower_hsv) % 3 != 0 or len(args.upper_hsv) % 3 != 0:
        raise ValueError("HSV arguments must be multiples of 3 (H S V)")

    lower_hsv_list = [args.lower_hsv[i:i+3] for i in range(0, len(args.lower_hsv), 3)]
    upper_hsv_list = [args.upper_hsv[i:i+3] for i in range(0, len(args.upper_hsv), 3)]

    if len(lower_hsv_list) != len(upper_hsv_list):
        raise ValueError("Number of lower and upper HSV ranges must match")

    process_color_object(args.input_image, args.output_image, lower_hsv_list, upper_hsv_list)
