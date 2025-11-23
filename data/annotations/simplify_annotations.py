#!/usr/bin/env python3
"""
Script to simplify annotation JSON files by extracting only dense_caption information.
Usage: python simplify_annotations.py [--d DIRECTORY]
"""

import os
import json
import argparse
from pathlib import Path
from tqdm import tqdm


def simplify_annotations(annotation_dir=None):
    """
    Simplify annotation JSON files by keeping only dense_caption information.
    
    Args:
        annotation_dir: Directory containing annotation files. If None, uses script directory.
    """
    if annotation_dir is None:
        annotation_dir = Path(__file__).parent
    else:
        annotation_dir = Path(annotation_dir)
    
    original_dir = annotation_dir / "original"
    original_dir.mkdir(exist_ok=True)
    
    # Find all JSON files (excluding those starting with ssa_)
    json_files = [
        f for f in annotation_dir.glob("*.json") 
        if not f.name.startswith("ssa_")
    ]
    
    if not json_files:
        print("No JSON files to process.")
        return
    
    print(f"Found {len(json_files)} files to process.")
    
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
            
            # Move original file to original/ subdirectory
            original_path = original_dir / filename
            json_file.rename(original_path)
            
            # Save simplified version
            simplified_path = annotation_dir / f"ssa_{filename}"
            with open(simplified_path, "w", encoding="utf-8") as f:
                json.dump(simplified, f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            tqdm.write(f"  ✗ ERROR processing {filename}: {e}")
    
    print(f"\n✓ Done! Original files moved to: {original_dir}")
    print("✓ Simplified files created with ssa_ prefix")


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
    
    args = parser.parse_args()
    simplify_annotations(annotation_dir=args.d)