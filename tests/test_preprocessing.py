#!/usr/bin/env python3
"""
Test script for painting_preprocessing.py
This script demonstrates how to use the painting preprocessing functions.
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from art_recognition.preprocessing import preprocess_gallery_image, draw_preprocessing_result

def test_painting_preprocessing(tmp_path):
    """Test the painting preprocessing pipeline."""

    # Create a simple test image (you can replace this with loading a real image)
    # For demonstration, we'll create a synthetic gallery image
    test_image = np.zeros((600, 800, 3), dtype=np.uint8)

    # Create a wall background (light gray)
    test_image[:, :] = [200, 200, 200]

    # Add some paintings as rectangles
    # Painting 1
    cv2.rectangle(test_image, (50, 50), (250, 350), (100, 50, 25), -1)  # Brown frame
    cv2.rectangle(test_image, (60, 60), (240, 340), (255, 255, 255), -1)  # White canvas

    # Painting 2
    cv2.rectangle(test_image, (350, 100), (550, 400), (80, 40, 20), -1)  # Dark brown frame
    cv2.rectangle(test_image, (360, 110), (540, 390), (200, 150, 100), -1)  # Orange canvas

    # Painting 3
    cv2.rectangle(test_image, (100, 450), (300, 550), (60, 30, 15), -1)  # Very dark frame
    cv2.rectangle(test_image, (110, 460), (290, 540), (100, 200, 255), -1)  # Blue canvas

    print("Testing painting preprocessing...")

    try:
        # Run preprocessing
        result = preprocess_gallery_image(test_image)

        # Draw results
        output_image = draw_preprocessing_result(test_image, result["candidates"])

        # Display results
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(result["segmented"])
        plt.title("Mean Shift Segmentation")
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(result["wall_mask"], cmap='gray')
        plt.title("Wall Mask")
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(result["dilated_wall_mask"], cmap='gray')
        plt.title("Dilated Wall Mask")
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(result["inverted_wall_mask"], cmap='gray')
        plt.title("Inverted Wall Mask")
        plt.axis('off')

        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Paintings ({len(result['candidates'])} found)")
        plt.axis('off')

        plt.tight_layout()
        output_path = tmp_path / "synthetic_test_output.png"
        plt.savefig(output_path)
        plt.close()
        print(f"Saved synthetic test output to {output_path}")

        # Print details of each candidate
        for i, candidate in enumerate(result["candidates"]):
            x, y, w, h = candidate.bounding_rect
            print(f"Painting {i+1}: Position ({x}, {y}), Size {w}x{h}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

def run_with_real_image(image_path: str):
    """Test preprocessing with a real image file."""
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return

    print(f"Loading image: {image_path}")
    image = cv2.imread(image_path)

    if image is None:
        print("Failed to load image")
        return

    print(f"Image loaded: {image.shape}")

    try:
        result = preprocess_gallery_image(image)
        output_image = draw_preprocessing_result(image, result["candidates"])

        print(f"Found {len(result['candidates'])} painting candidates")
        for i, candidate in enumerate(result["candidates"]):
            x, y, w, h = candidate.bounding_rect
            print(f"Painting {i+1}: Position ({x}, {y}), Size {w}x{h}")

        plt.figure(figsize=(12, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detected Paintings ({len(result['candidates'])} found)")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig("real_image_test_output.png")
        plt.close()
        print("Saved real image test output to real_image_test_output.png")

    except Exception as e:
        print(f"Error processing image: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test painting preprocessing.")
    parser.add_argument("--image", help="Path to a real image file to test with")
    args = parser.parse_args()

    if args.image:
        run_with_real_image(args.image)
    else:
        # Test with synthetic image
        test_painting_preprocessing()
