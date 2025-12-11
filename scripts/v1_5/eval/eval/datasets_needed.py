import tqdm as tqdm
import os

if __name__ == "__main__":

    print()
    for folder in tqdm.tqdm(os.listdir(), desc="Processing folders"):
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if not file.endswith("_eval.py"):
                continue
            with open(os.path.join(folder, file), "r+", encoding="utf-8") as f:
                lines = f.readlines()
            for line in lines:
                if "load_dataset(" in line:
                    print(line.strip(), end="\n")