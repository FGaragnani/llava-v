# Utility file. Load a model with LLaVAModelForCausalLM and ask a question.

import sys
sys.path.insert(0, '/leonardo/home/userexternal/fgaragna')

import torch
from transformers import AutoTokenizer

from model.language_model.llava_llama import LlavaLlamaForCausalLM
from model.language_model.llava_qwen import LlavaQwenForCausalLM
from conversation import conv_templates, get_stopping_criteria

import argparse


def load_model(model_path):
    if "qwen" in model_path:
        model = LlavaQwenForCausalLM.from_pretrained(model_path)
    else:
        model = LlavaLlamaForCausalLM.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path, use_fast=False)


def build_prompt_from_template(template_conv, question):
    conv = template_conv.copy()
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def ask(model, tokenizer, question, template_conv):
    device = next(model.parameters()).device
    prompt = build_prompt_from_template(template_conv, question)
    model_inputs = tokenizer(prompt, return_tensors="pt").to(device)

    generate_kwargs = {
        "inputs": model_inputs["input_ids"],
        "attention_mask": model_inputs.get("attention_mask", None),
        "max_new_tokens": 100,
        "do_sample": False,
        "use_cache": True,
    }

    if tokenizer.eos_token_id is not None:
        generate_kwargs["eos_token_id"] = tokenizer.eos_token_id
    if tokenizer.pad_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.pad_token_id
    elif tokenizer.eos_token_id is not None:
        generate_kwargs["pad_token_id"] = tokenizer.eos_token_id

    generate_kwargs["stopping_criteria"] = get_stopping_criteria(tokenizer)

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)
    new_tokens = outputs[0][model_inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if not answer:
        answer = tokenizer.decode(new_tokens, skip_special_tokens=False).strip()
        if not answer:
            answer = "[EMPTY_GENERATION]"
    return answer


def infer_conv_mode(model_path, override=None):
    if override is not None:
        return override
    return "qwen2_5" if "qwen" in model_path.lower() else "vicuna_v1"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-path", type=str, required=True)
    argparser.add_argument("--conv-mode", type=str, default=None)
    args = argparser.parse_args()

    model_path = args.model_path
    conv_mode = infer_conv_mode(model_path, override=args.conv_mode)
    if conv_mode not in conv_templates:
        raise ValueError(f"Unknown conv mode: {conv_mode}. Available: {sorted(conv_templates.keys())}")

    model = load_model(model_path)
    model.eval()
    tokenizer = load_tokenizer(model_path)
    template_conv = conv_templates[conv_mode]

    questions = [
        "Hi! What is your name?",
        "What color is the sky?",
        "What color is the sky? Answer with a single word or phrase.",
        "Where is the sky located with respect to the ground?",
        "Where is the sky located with respect to the ground? Answer with a single word or phrase.",
    ]

    for question in questions:
        print("Question:", question)
        answer = ask(model, tokenizer, question, template_conv)
        print("Answer:", answer)