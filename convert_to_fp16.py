import torch
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import os, sys, json

ROOT_PATH = "/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/stage_three"
sys.path.append("..")


def is_model_folder(path):
    # Basic check for HF model folder
    return (
        os.path.isdir(path) and
        (
            os.path.exists(os.path.join(path, "pytorch_model.bin")) or
            any(f.endswith(".safetensors") for f in os.listdir(path))
        )
    )

def is_already_fp16(model_path):
    # Check if model is already in fp16 format by reading config.json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return False
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        # Check if torch_dtype is set to float16
        torch_dtype = config.get("torch_dtype")
        return torch_dtype == "float16"
    except Exception as e:
        print(f"  Warning: Could not check config.json: {e}")
        return False

for folder_name in os.listdir(ROOT_PATH):
    model_path = os.path.join(ROOT_PATH, folder_name)

    if not is_model_folder(model_path):
        continue

    print(f"\nProcessing: {model_path}")
    
    # Check if already fp16
    if is_already_fp16(model_path):
        print("-> Already in FP16 format. Skipping.")
        continue

    try:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        model = model.half()
        model.save_pretrained(model_path)

        print("-> Converted to FP16 successfully.")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed: {e}")

print("\nAll done.")