import cv2
import sys
import argparse
import numpy as np
from skimage import data
import matplotlib.pyplot as plt

def load_test_image():
    """
    Loads a sample test image (astronaut) from skimage.
    
    Returns:
        np.ndarray: BGR image for OpenCV compatibility.
    """
    try:
        img = data.astronaut()
    except Exception as e:
        raise RuntimeError(f"Failed to load test image: {e}")

    if img is None or not isinstance(img, np.ndarray):
        raise RuntimeError("Loaded image is invalid or None.")

    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def create_exposures(img, alpha_dark=0.5, alpha_light=1.5):
    """
    Creates three exposure versions of the image: dark, normal, and light.
    """
    if img is None or not isinstance(img, np.ndarray):
        raise ValueError("Input image must be a valid numpy array.")

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Input image must be a color image with 3 channels (BGR).")

    dark = cv2.convertScaleAbs(img, alpha=alpha_dark, beta=0)
    normal = img.copy()
    light = cv2.convertScaleAbs(img, alpha=alpha_light, beta=0)

    return dark, normal, light


def combine_hdr(dark, normal, light, w_dark, w_normal, w_light):
    """
    Combines multiple exposure images into a single HDR-like effect.
    """
    for img in (dark, normal, light):
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError("All inputs must be valid numpy arrays.")
        if img.shape != dark.shape:
            raise ValueError("All input images must have the same shape.")

    hdr = cv2.addWeighted(dark, w_dark, normal, w_normal, 0)
    hdr = cv2.addWeighted(hdr, 1.0, light, w_light, 0)

    return hdr


def display_results(original, dark, normal, light, hdr):
    """
    Displays the original image, exposures, and HDR effect.
    """
    for name, img in zip(["original", "dark", "normal", "light", "hdr"], 
                         [original, dark, normal, light, hdr]):
        if img is None or not isinstance(img, np.ndarray):
            raise ValueError(f"Image '{name}' is invalid or None.")

    plt.figure(figsize=(15, 6))

    plt.subplot(1, 5, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 5, 2)
    plt.imshow(cv2.cvtColor(dark, cv2.COLOR_BGR2RGB))
    plt.title("Dark")
    plt.axis("off")

    plt.subplot(1, 5, 3)
    plt.imshow(cv2.cvtColor(normal, cv2.COLOR_BGR2RGB))
    plt.title("Normal")
    plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(cv2.cvtColor(light, cv2.COLOR_BGR2RGB))
    plt.title("Light")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB))
    plt.title("HDR effect")
    plt.axis("off")

    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Simulate HDR effect from multiple exposures.")
    parser.add_argument("--w_dark", type=float, default=0.3, help="Weight for dark exposure.")
    parser.add_argument("--w_normal", type=float, default=0.4, help="Weight for normal exposure.")
    parser.add_argument("--w_light", type=float, default=0.3, help="Weight for light exposure.")
    parser.add_argument("--alpha_dark", type=float, default=0.5, help="Brightness multiplier for dark exposure.")
    parser.add_argument("--alpha_light", type=float, default=1.5, help="Brightness multiplier for light exposure.")
    parser.add_argument("--output_image", type=str, default="output.png", help="File path to save HDR image.")
    parser.add_argument("--output_original", type=str, default="original.png", help="File path to save original image.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Check weight sum
    weight_sum = args.w_dark + args.w_normal + args.w_light
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        print(f"[ERROR] Weights must sum to 1. Current sum = {weight_sum}")
        sys.exit(1)

    try:
        img = load_test_image()
        dark, normal, light = create_exposures(img, args.alpha_dark, args.alpha_light)
        hdr = combine_hdr(dark, normal, light, args.w_dark, args.w_normal, args.w_light)
        display_results(img, dark, normal, light, hdr)

        # Save original
        success_original = cv2.imwrite(args.output_original, img)
        if success_original:
            print(f"Original image saved as '{args.output_original}'")
        else:
            print(f"[ERROR] Failed to save original image to '{args.output_original}'")

        # Save HDR result
        success_hdr = cv2.imwrite(args.output_image, hdr)
        if success_hdr:
            print(f"HDR image saved as '{args.output_image}'")
        else:
            print(f"[ERROR] Failed to save HDR image to '{args.output_image}'")

    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
