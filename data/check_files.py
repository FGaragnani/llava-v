import os
import tarfile
import json
import requests
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from collections import defaultdict

OUTPUT_DIR = Path("/leonardo_scratch/large/userexternal/fgaragna/dataset/GLAMM")
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations/simple"
IMAGES_DIR = OUTPUT_DIR / "images"

def check_files():

    for image_file in tqdm(IMAGES_DIR.iterdir(), total=500_000, desc="Checking images"):
        if image_file.is_file():
            image_stem = image_file.stem
            annotation_file = ANNOTATIONS_DIR / f"{image_stem}.json"
            if not annotation_file.exists():
                print(f"Missing annotation for image: {image_file.name}")
                continue
            
            # Check if image can be opened
            try:
                with Image.open(image_file) as img:
                    img.verify()  # Verify that it is, in fact an image
            except:
                os.remove(image_file)
                os.remove(annotation_file) if annotation_file.exists() else None
                print(f"Deleted corrupted image and its annotation: {image_file.name}, {annotation_file.name})")
    
    for annotation_file in tqdm(ANNOTATIONS_DIR.iterdir(), total=len(list(ANNOTATIONS_DIR.iterdir())), desc="Checking annotations"):
        if annotation_file.is_file():
            image_file = IMAGES_DIR / f"{annotation_file.stem}.jpg"
            try:
                with open(annotation_file, 'r', encoding='utf-8') as f:
                    json.load(f)
            except json.JSONDecodeError:
                os.remove(image_file) if image_file.exists() else None
                os.remove(annotation_file)
                print(f"Deleted corrupted annotation and its image: {annotation_file.name}, {image_file.name})")