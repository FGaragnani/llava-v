#!/usr/bin/env python3
"""
Script to simplify annotation JSON files by extracting only dense_caption information.
Usage: python simplify_annotations.py [--d DIRECTORY] [--workers N]
"""

import os
import json
import argparse
import shutil
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def process_single_file(json_file, simple_dir, original_dir, keep_originals):
    """
    Process a single JSON file.
    
    Args:
        json_file: Path to the JSON file
        simple_dir: Directory to save simplified files
        original_dir: Directory for original files
        keep_originals: Whether to copy or move originals
        
    Returns:
        tuple: (success: bool, filename: str, error_msg: str or None)
    """
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
        
        return (True, filename, None)
        
    except Exception as e:
        return (False, filename, str(e))


def simplify_annotations(annotation_dir=None, keep_originals=False, workers=None):
    """
    Simplify annotation JSON files by keeping only dense_caption information.
    
    Args:
        annotation_dir: Directory containing annotation files. If None, uses script directory.
        keep_originals: If True, copy originals to original/ folder. If False, move them.
        workers: Number of parallel workers (default: CPU count)
    """
    if annotation_dir is None:
        annotation_dir = Path(__file__).parent
    else:
        annotation_dir = Path(annotation_dir)
    
    simple_dir = annotation_dir / "simple"
    simple_dir.mkdir(exist_ok=True)
    
    original_dir = annotation_dir / "original"
    original_dir.mkdir(exist_ok=True)
    
    # Find all JSON files in the main directory (not in subdirectories)
    json_files = [
        f for f in annotation_dir.glob("*.json")
    ]
    
    if not json_files:
        print("No JSON files to process.")
        return
    
    print(f"Found {len(json_files)} files to process.")
    
    # Process files in parallel
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_file, json_file, simple_dir, original_dir, keep_originals): json_file
            for json_file in json_files
        }
        
        # Process results with progress bar
        with tqdm(total=len(json_files), desc="Simplifying annotations") as pbar:
            for future in as_completed(futures):
                success, filename, error_msg = future.result()
                if not success:
                    errors.append((filename, error_msg))
                    tqdm.write(f"  ✗ ERROR processing {filename}: {error_msg}")
                pbar.update(1)
    
    print(f"\n✓ Done! Simplified files saved to: {simple_dir}")
    if keep_originals:
        print(f"✓ Original files backed up to: {original_dir}")
    else:
        print(f"✓ Original files moved to: {original_dir}")
    
    if errors:
        print(f"\n⚠ {len(errors)} file(s) had errors during processing.")


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
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    
    args = parser.parse_args()
    simplify_annotations(annotation_dir=args.d, keep_originals=args.keep_originals, workers=args.workers)