import torch
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
import os

ROOT_PATH = "/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/stage_three"

def is_model_folder(path):
    # Basic check for HF model folder
    return (
        os.path.isdir(path) and
        (
            os.path.exists(os.path.join(path, "pytorch_model.bin")) or
            any(f.endswith(".safetensors") for f in os.listdir(path))
        )
    )

for folder_name in os.listdir(ROOT_PATH):
    model_path = os.path.join(ROOT_PATH, folder_name)

    if not is_model_folder(model_path):
        continue

    print(f"\nProcessing: {model_path}")

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