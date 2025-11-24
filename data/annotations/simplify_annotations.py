#!/usr/bin/env python3
"""
Script to simplify annotation JSON files by extracting only dense_caption information.
Usage: python simplify_annotations.py [--d DIRECTORY]
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm


def simplify_annotations(annotation_dir=None, image_dir=None, keep_originals=False):
    """
    Simplify annotation JSON files by keeping only dense_caption information.

    Args:
        annotation_dir: Directory containing annotation files. If None, uses script directory.
        image_dir: Optional directory containing images. If provided, only JSON files
            whose stem matches a file in `image_dir` will be processed.
        keep_originals: If True, copy originals to original/ folder. If False, move them.
    """
    if annotation_dir is None:
        annotation_dir = Path(__file__).parent
    else:
        annotation_dir = Path(annotation_dir)

    # Prepare image stems filter if requested
    image_stems = None
    if image_dir:
        image_dir = Path(image_dir)
        if not image_dir.exists() or not image_dir.is_dir():
            print(f"Image directory does not exist or is not a directory: {image_dir}")
            return
        # Collect stems of files in image_dir (ignore subdirectories)
        image_stems = {p.stem for p in image_dir.iterdir() if p.is_file()}
    
    simple_dir = annotation_dir / "simple"
    simple_dir.mkdir(exist_ok=True)
    
    original_dir = annotation_dir / "original"
    original_dir.mkdir(exist_ok=True)
    
    # Find all JSON files in the main directory (not in subdirectories)
    json_files = [f for f in annotation_dir.glob("*.json")]

    # If image_dir provided, filter json_files to only those whose stem is in image_stems
    if image_stems is not None:
        original_count = len(json_files)
        json_files = [f for f in json_files if f.stem in image_stems]
        print(f"Filtered {original_count} -> {len(json_files)} files using images in: {image_dir}")
    
    if not json_files:
        print("No JSON files to process.")
        return
    
    print(f"Found {len(json_files)} files to process.")
    
    # Process files sequentially
    for json_file in tqdm(json_files, desc="Simplifying annotations"):
        filename = json_file.name
        
        try:
            # Read original JSON
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Extract image name (filename without extension + .jpg)
            image_name = json_file.stem + ".jpg"
            
            # Extract dense_caption - handle both keyed and direct structures
            dense_caption = None
            if isinstance(data, dict):
                if image_name in data:
                    # Structure: {"sa_123456.jpg": {"dense_caption": ...}}
                    dense_caption = data[image_name].get("dense_caption", {})
                elif "dense_caption" in data:
                    # Direct structure: {"dense_caption": ...}
                    dense_caption = data.get("dense_caption", {})
            
            # If dense_caption not found, use empty dict
            if dense_caption is None:
                dense_caption = {}
            
            # Create simplified structure
            simplified = {
                image_name: {
                    "dense_caption": dense_caption
                }
            }
            
            # Save simplified version with same filename
            simplified_path = simple_dir / filename
            with open(simplified_path, "w", encoding="utf-8") as f:
                json.dump(simplified, f, indent=2, ensure_ascii=False)
            
            # Handle original file
            if keep_originals:
                # Copy to original/ subdirectory
                original_path = original_dir / filename
                shutil.copy2(json_file, original_path)
            else:
                # Move to original/ subdirectory
                original_path = original_dir / filename
                json_file.rename(original_path)
            
        except Exception as e:
            tqdm.write(f"  ✗ ERROR processing {filename}: {e}")
    
    print(f"\n✓ Done! Simplified files saved to: {simple_dir}")
    if keep_originals:
        print(f"✓ Original files backed up to: {original_dir}")
    else:
        print(f"✓ Original files moved to: {original_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simplify annotation JSON files by extracting only dense_caption information."
    )
    parser.add_argument(
        "--d", 
        type=str, 
        default=None,
        help="Working directory containing annotation files (default: script directory)"
    )
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Keep original files in place and copy to original/ folder (default: move to original/)"
    )
    parser.add_argument(
        "-i", "--image-dir",
        dest="image_dir",
        type=str,
        default=None,
        help="Optional image directory: if provided, only process JSON files matching image basenames"
    )
    
    args = parser.parse_args()
    simplify_annotations(annotation_dir=args.d, image_dir=args.image_dir, keep_originals=args.keep_originals)