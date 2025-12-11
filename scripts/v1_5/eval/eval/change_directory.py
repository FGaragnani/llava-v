from tqdm import tqdm

if __name__ == "__main__":
    import os

    for folder in tqdm(os.listdir(), desc="Processing folders"):
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.endswith("_eval.py"):
                continue
            with open(os.path.join(folder, file), "r+", encoding="utf-8") as f:
                content = f.read()
            content = content.replace(
                "from src.llava",
                "from llava"
            )
            with open(os.path.join(folder, file), "w+", encoding="utf-8") as f:
                f.write(content)