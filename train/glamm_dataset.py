import os
import json
from typing import List, Tuple, Optional, Dict

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image, ImageOps

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.train.train import preprocess_multimodal, preprocess


class GranDDataset(Dataset):
    """Per-image GranD dataset interface.

    Returns per item:
      input_ids, labels, image tensor, list of bboxes, dense_labels,
      dense_caption text, and is_grand flag.
    """

    def __init__(
        self,
        image_dir: str,
        annotation_dir: str,
        tokenizer=None,
        data_args=None,
        image_processor=None,
        patch_size: Tuple[int, int] = (224, 224),
        check_area: float = 0.05,
        prompt_template: Optional[str] = None,
        label_joiner: str = ", ",
    ):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.patch_size = patch_size
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_processor = image_processor if image_processor is not None else (
            getattr(data_args, "image_processor", None) if data_args else None
        )
        self.prompt_template = prompt_template or "Describe the scene."
        self.label_joiner = label_joiner
        self.check_area_fn = lambda img_size, patch_area: (patch_area / img_size) > check_area
        self.annotation_files = [
            os.path.join(annotation_dir, f)
            for f in os.listdir(annotation_dir)
            if f.lower().endswith(".json")
        ]
        self._image_index: List[Dict[str, str]] = []
        self._build_image_index()

    def _resize_and_pad(self, img: Image.Image) -> torch.Tensor:
        target_w, target_h = self.patch_size
        try:
            padded = ImageOps.pad(img, (target_w, target_h), color=0, centering=(0.5, 0.5))
        except TypeError:
            # Fallback manual padding
            w, h = img.size
            target_long = max(target_w, target_h)
            scale = target_long / float(max(w, h))
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            img = img.resize((new_w, new_h))
            pad_w = max(0, target_w - new_w)
            pad_h = max(0, target_h - new_h)
            left = pad_w // 2
            right = pad_w - left
            top = pad_h // 2
            bottom = pad_h - top
            padded = ImageOps.expand(img, border=(left, top, right, bottom), fill=0)
        return F.to_tensor(padded)

    def _build_image_index(self):
        for ann_path in self.annotation_files:
            self._image_index.append({"ann_path": ann_path, "image_name": ann_path.split(".")[0]})

    def __len__(self) -> int:
        return len(self._image_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | List | str]:
        if idx < 0 or idx >= len(self._image_index):
            raise IndexError(idx)
        entry = self._image_index[idx]
        ann_path = entry["ann_path"]
        image_name = entry["image_name"]
        with open(ann_path, "r", encoding="utf-8") as f:
            ann_file = json.load(f)
        ann = ann_file.get(image_name, {})

        dense = ann.get("dense_caption", {})
        if isinstance(dense, dict):
            dense_caption_text = dense.get("caption", "")
            details = dense.get("details", [])
        elif isinstance(dense, str):
            dense_caption_text = dense
            details = []
        else:
            dense_caption_text = ""
            details = []

        # Image
        image_path = os.path.join(self.image_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        img_area = w * h

        dense_labels: List[str] = []
        bboxes: List[Tuple[int, int, int, int]] = []
        
        if isinstance(details, list):
            for d in details:
                if not isinstance(d, dict):
                    continue

                # Extract phrase or fallback textual field
                text = None
                for k in ("phrase", "caption", "text", "detail", "value", "description"):
                    v = d.get(k)
                    if isinstance(v, str) and v.strip():
                        text = v.strip()
                        break

                # Extract bbox for this detail (annotation may store nested list of bboxes)
                bbox_values = d.get("bbox")
                if bbox_values and text:
                    # Some annotations store [[l,t,r,b]]
                    if isinstance(bbox_values, list) and len(bbox_values) == 1 and isinstance(bbox_values[0], list):
                        bbox_values = bbox_values[0]
                    try:
                        l, t, r, b = map(int, bbox_values)
                        area = max(0, r - l) * max(0, b - t)
                        if area > 0 and self.check_area_fn(img_area, area):
                            dense_labels.append(text)
                            bboxes.append((l, t, r, b))
                    except Exception:
                        pass  # Skip invalid bbox

        if self.image_processor is None:
            image_tensor = F.to_tensor(image)
        else:
            image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        # Conversation
        instruction = self.prompt_template
        conversation = [
            {"from": "human", "value": f"{DEFAULT_IMAGE_TOKEN}\n{instruction}"},
            {"from": "gpt", "value": dense_caption_text},
        ]
        sources = preprocess_multimodal([conversation], self.data_args)
        data_dict = preprocess(sources, self.tokenizer, has_image=True)

        return {
            "input_ids": data_dict["input_ids"][0],
            "labels": data_dict["labels"][0],
            "image": image_tensor,
            "bboxes": bboxes,
            "dense_labels": dense_labels,
            "dense_caption": dense_caption_text,
            "image_path": image_path,
            "is_grand": torch.tensor(True),
        }