import argparse
import os
from typing import List
from PIL import Image
from tqdm import tqdm

def is_image_ok(path: str) -> bool:
    try:
        im = Image.open(path)
        return True
    except Exception:
        return False


def find_images(root: str, exts: List[str]) -> List[str]:
    out = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if any(fn.lower().endswith(ext) for ext in exts):
                out.append(os.path.join(dirpath, fn))
    return out


def main():
    parser = argparse.ArgumentParser(description="Prune malformed/unreadable images by attempting to open them.")
    parser.add_argument("image_dir", help="Root directory containing images")
    parser.add_argument("--exts", nargs="*", default=[".jpg", ".jpeg", ".png"], help="File extensions to check")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete, only report")
    parser.add_argument("--verbose", action="store_true", help="Print every file status")
    args = parser.parse_args()

    images = find_images(args.image_dir, args.exts)
    total = len(images)
    bad = []
    
    for i, path in tqdm(enumerate(images, 1), total=total):
        ok = is_image_ok(path)
        if args.verbose:
            print(f"[{i}/{total}] {'OK' if ok else 'BAD'} - {path}")
        if not ok:
            bad.append(path)
            print("Found bad image: ", path)

    if not bad:
        print("No bad images found.")
        return

    print(f"Found {len(bad)} bad images.")
    if args.dry_run:
        for p in bad:
            print(p)
        print("Dry run complete: no files were deleted.")
        return

    removed = 0
    for p in bad:
        try:
            os.remove(p)
            removed += 1
        except Exception as e:
            print(f"Failed to remove {p}: {e}")

    print(f"Removed {removed}/{len(bad)} bad images.")


if __name__ == "__main__":
    main()
