#!/usr/bin/env python
import numpy as np
from pathlib import Path
from PIL import Image
import sys


def convert_labels_to_masks(labels_dir, masks_dir, suffix="regions.txt"):
    """
    Convert text label files to PNG masks.
    Handles negative values (unknown) by setting them to a specific class.
    """
    labels_path = Path(labels_dir)
    masks_path = Path(masks_dir)
    masks_path.mkdir(parents=True, exist_ok=True)

    label_files = sorted(labels_path.glob(f"*.{suffix}"))

    for label_file in label_files:
        # Read the text file as a matrix
        with open(label_file, 'r') as f:
            lines = f.read().strip().split('\n')

        # Parse the matrix (space-separated integers)
        matrix = []
        for line in lines:
            row = [int(x) for x in line.split()]
            matrix.append(row)

        # Convert to numpy array as int32 first
        mask = np.array(matrix, dtype=np.int32)

        # Handle negative values (unknown) by setting them to 0
        # Also clamp values to valid range [0, 7]
        mask = np.clip(mask, 0, 7)

        # Now convert to uint8
        mask = mask.astype(np.uint8)

        # Create image from mask
        image = Image.fromarray(mask, mode='L')

        # Save with same base name as original image
        base_name = label_file.stem.replace('.regions', '').replace('.layers', '').replace('.surfaces', '')
        output_path = masks_path / f"{base_name}.png"
        image.save(output_path)

    print(f"Converted {len(label_files)} labels to masks in {masks_path}")


if __name__ == "__main__":
    labels_dir = "/workspace/HW2/data/iccv09Data/labels"
    masks_dir = "/workspace/HW2/data/stanford_bg/masks"

    # Convert regions.txt to masks
    convert_labels_to_masks(labels_dir, masks_dir, "regions.txt")