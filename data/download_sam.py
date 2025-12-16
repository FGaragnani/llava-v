import os
import tarfile
import json
import requests
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# ---------------- CONFIG ----------------
OUTPUT_DIR = Path("/leonardo_scratch/large/userexternal/fgaragna/dataset/GLAMM")
ANNOTATIONS_DIR = OUTPUT_DIR / "annotations/simple"
IMAGES_DIR = OUTPUT_DIR / "images"
TARS_DIR = OUTPUT_DIR / "tars"
LINKS_FILE = OUTPUT_DIR / "links.txt"

IMAGES_PER_TAR = 11186
# ----------------------------------------

def load_links(file_path: Path) -> dict:
    """Load mapping from part filename -> CDN link."""
    if not file_path.exists():
        raise FileNotFoundError(f"Missing links file: {file_path}")
    links = {}
    for i in range(7):
        links[f"sa_{i:06d}.tar"] = f"https://huggingface.co/datasets/Aber-r/SA-1B_backup/resolve/main/sa_{i:06d}.tar?download=true"
    return links
    links = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            fname, url = parts[0], parts[1]
            links[fname] = url
    print(f"Loaded {len(links)} part links from {file_path}")
    return links


def download_file(url, dest):
    """Download a file with a progress bar (skip if already exists)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest


def get_part_index_from_name(name: str) -> int:
    """Infer which .tar contains this image based on its numeric ID."""
    try:
        num = int(name.split("_")[-1])
    except ValueError:
        raise ValueError(f"Unexpected name format: {name}")
    return num // IMAGES_PER_TAR


def extract_selected_images_from_tar(tar_path: Path, image_names: set, dest_dir: Path):
    """Extract only the images whose base names are in `image_names`."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_count = 0

    with tarfile.open(tar_path, "r:*") as tar:
        for member in tqdm(tar, desc=f"Extracting {tar_path.name}", unit="file", leave=False):
            filename = Path(member.name).stem
            if filename in image_names and member.name.lower().endswith(".jpg"):
                tar.extract(member, path=dest_dir)
                extracted_count += 1
                if extracted_count == len(image_names):
                    break

    return extracted_count


def main():
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TARS_DIR.mkdir(parents=True, exist_ok=True)
    link_map = load_links(LINKS_FILE)
    MAX_IMAGES = 500_000

    annotation_names = [p.stem for p in ANNOTATIONS_DIR.glob("*.json")]
    if not annotation_names:
        print("No annotations found.")
        return

    print(f"Found {len(annotation_names)} annotation files")

    from collections import defaultdict
    parts = defaultdict(set)
    for name in annotation_names:
        part_idx = get_part_index_from_name(name)
        parts[part_idx].add(name)
    print(f"Grouped into {len(parts)} tar parts.")

    before_start = sum(1 for _ in IMAGES_DIR.glob("*.jpg"))
    image_pbar = tqdm(total=MAX_IMAGES - before_start, desc="Extracting images", unit="img")
    for part_idx in tqdm(sorted(parts.keys()), desc="Processing TAR parts", unit="tar"):
        current_count = sum(1 for _ in IMAGES_DIR.glob("*.jpg")) - before_start
        new_images = current_count - image_pbar.n
        if new_images > 0:
            image_pbar.update(new_images)

        if current_count >= MAX_IMAGES:
            tqdm.write(f"Reached quota of {MAX_IMAGES} images already in {IMAGES_DIR}. Stopping.")
            break
        image_names = parts[part_idx]
        tar_name = f"sa_{part_idx:06d}.tar"
        tar_path = TARS_DIR / tar_name

        # Skip if all images already extracted
        remaining = [n for n in image_names if not (IMAGES_DIR / f"{n}.jpg").exists()]
        if not remaining or len(remaining) <= 5000:
            continue

        # Download if missing
        if not tar_path.exists():
            url = link_map.get(tar_name)
            if not url:
                print(f"⚠️ No link for {tar_name}")
                continue
            print(f"Fetching {tar_name} ...")
            download_file(url, tar_path)

        # Extract needed images
        print(f"📦 Extracting from {tar_name} ({len(remaining)} needed)...")
        image_names = parts[part_idx] | parts[part_idx + 1]
        remaining = [n for n in image_names if not (IMAGES_DIR / f"{n}.jpg").exists()]
        extracted_count = extract_selected_images_from_tar(tar_path, set(remaining), IMAGES_DIR)
        print(f"✅ Extracted {extracted_count}/{len(remaining)} images from {tar_name}")
        tqdm.write(f"✅ Extracted {extracted_count}/{len(remaining)} images from {tar_name} "
                   f"(total now: {current_count + extracted_count})")

        # Delete tar afterward
        try:
            tar_path.unlink()
            tqdm.write(f"🧹 Deleted {tar_name}")
        except Exception as e:
            tqdm.write(f"⚠️ Could not delete {tar_name}: {e}")

        # Stop if folder now exceeds quota
        if sum(1 for _ in IMAGES_DIR.glob("*.jpg")) >= MAX_IMAGES:
            tqdm.write(f"⏹️ Reached quota of {MAX_IMAGES} images. Stopping.")
            break
        # Optionally remove tar after extraction to save space
        # tar_path.unlink()

    print("🎉 All done.")


if __name__ == "__main__":
    main()
