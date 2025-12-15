
import os
import tarfile
import requests
from tqdm import tqdm
from pathlib import Path

EXTRACT_LIMIT = 2000000000
GRAND_URL = "https://huggingface.co/datasets/MBZUAI/GranD/resolve/main/part_{part}/part_{part}_{idx}.tar.gz?download=true"
OUTPUT_DIR = Path("/leonardo_scratch/large/userexternal/fgaragna/dataset/GLAMM/annotations/simple")
IMAGES_PER_TAR = 11186

def download_file(url, dest):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc="Downloading .tar.gz"
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))
    return dest

def extract_partial_tar(tar_path, dest_dir, limit):
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted_names = []
    with tarfile.open(tar_path, "r:gz") as tar:
        # json_members = [m for m in tar.getmembers() if m.name.endswith(".json")]
        # to_extract = [m for m in json_members if Path(m.name).stem not in existing][:needed]
        for member in tqdm(tar, desc="Extracting files"):
            # Skip directories
            if member.isdir():
                continue
            
            # Get just the filename (flatten directory structure)
            filename = Path(member.name).name
            file_stem = Path(filename).stem
            
            # Extract file directly to dest_dir
            member.name = filename
            tar.extract(member, path=dest_dir)
            
            extracted_names.append(file_stem)
    return extracted_names

def main():
    for part in range(3, 5):
        for idx in range(1, 5):
            tar_path = OUTPUT_DIR / f"../part_{part}_{idx}.tar.gz"
            if not tar_path.exists():
                download_file(GRAND_URL.format(part=part, idx=idx), tar_path)
            extract_partial_tar(tar_path, OUTPUT_DIR, EXTRACT_LIMIT)

if __name__ == "__main__":
    main()
