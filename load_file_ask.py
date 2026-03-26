# Utility file. Load a model with LLaVAModelForCausalLM and ask a question.

import sys
sys.path.insert(0, '/leonardo/home/userexternal/fgaragna')

import torch
from transformers import AutoTokenizer

from model.language_model.llava_llama import LlavaLlamaForCausalLM
from model.language_model.llava_qwen import LlavaQwenForCausalLM

def load_model(model_path):
    if "qwen" in model_path:
        model = LlavaQwenForCausalLM.from_pretrained(model_path)
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_path)
    return model.to("cuda:0") if torch.cuda.is_available() else model

def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path, use_fast=False)

def ask(model, tokenizer, question):
    device = next(model.parameters()).device
    model_inputs = tokenizer(question, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            inputs=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask", None),
            max_new_tokens=100,
        )
    new_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer


MODEL_PATH = "/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/llava-v_s2--ve-qwen2_5"
model = load_model(MODEL_PATH)
model.eval()
tokenizer = load_tokenizer(MODEL_PATH)

input_str = ""

while True:
    input_str = input("Enter your question (Ctrl-D to exit): ")
    if input_str == u'\u0004':
        break
    answer = ask(model, tokenizer, input_str)
    print("Answer:", answer)