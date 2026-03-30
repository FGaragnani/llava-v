# Utility file. Load a model with LLaVAModelForCausalLM and ask a question.

import sys
sys.path.insert(0, '/leonardo/home/userexternal/fgaragna')

import torch
from transformers import AutoTokenizer

from model.language_model.llava_llama import LlavaLlamaForCausalLM
from model.language_model.llava_qwen import LlavaQwenForCausalLM
from conversation import conv_vicuna_v1, conv_qwen2_5

import argparse

def load_model(model_path):
    if "qwen" in model_path:
        model = LlavaQwenForCausalLM.from_pretrained(model_path)
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_path)
    return model.to("cuda:0") if torch.cuda.is_available() else model

def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path, use_fast=False)

def ask(model, tokenizer, question, conversation):
    device = next(model.parameters()).device
    prompt = conversation.get_prompt(question)
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            inputs=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask", None),
            max_new_tokens=100,
        )
    new_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return answer

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-path", type=str)

args = argparser.parse_args()
MODEL_PATH = args.model_path

model = load_model(MODEL_PATH)
model.eval()
tokenizer = load_tokenizer(MODEL_PATH)
conv = conv_qwen2_5.copy() if "qwen" in MODEL_PATH else conv_vicuna_v1.copy()

questions = [
    "Hi! What is your name?",
    "What color is the sky?",
    "What color is the sky? Answer with a single word or phrase.",
    "Where is the sky located with respect to the ground?",
    "Where is the sky located with respect to the ground? Answer with a single word or phrase."
]

for question in questions:
    print("Question: ", question)
    answer = ask(model, tokenizer, question, conv)
    print("Answer: ", answer)