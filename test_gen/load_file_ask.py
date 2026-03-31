# Utility file. Load a model with LLaVAModelForCausalLM and ask a question with images.

import sys
sys.path.insert(0, '/leonardo/home/userexternal/fgaragna')

import torch
from PIL import Image
from conversation import conv_templates
from model.builder import load_pretrained_model
from mm_utils import get_model_name_from_path, process_images
from constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

import argparse


def load_image(image_path):
    """Load image from file path."""
    image = Image.open(image_path).convert('RGB')
    return image


def ask(model, tokenizer, image_processor, question, images, conv_mode="llava_v1"):
    """Ask a question given images using the model."""
    device = next(model.parameters()).device
    
    # Build prompt with image tokens
    conv = conv_templates[conv_mode].copy()
    image_str = DEFAULT_IMAGE_TOKEN + '\n'
    inp = image_str * len(images) + question
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    print("""Prompt:\n""" + prompt + "\n")

    # Tokenize prompt
    from mm_utils import tokenizer_image_token
    input_ids_result = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
    print(input_ids_result)
    if isinstance(input_ids_result, list):
        input_ids = torch.tensor(input_ids_result, dtype=torch.long)
    else:
        input_ids = input_ids_result
    
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(device)

    # Process images
    image_sizes = [img.size for img in images]
    images_tensor = process_images(images, image_processor, model.config)
    if isinstance(images_tensor, list):
        images_tensor = torch.stack([img.to(device, dtype=torch.float16) for img in images_tensor])
    else:
        images_tensor = images_tensor.to(device, dtype=torch.float16)

    # Generate
    generate_kwargs = {
        "inputs": input_ids,
        "images": images_tensor,
        "image_sizes": image_sizes,
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

    with torch.inference_mode():
        outputs = model.generate(**generate_kwargs)

    # Decode answer
    output_ids = outputs[0]
    answer = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    return answer


def infer_conv_mode(model_name, override=None):
    """Infer conversation mode from model name."""
    if override is not None:
        return override
    
    model_name_lower = model_name.lower()
    if "qwen" in model_name_lower:
        return "qwen2_5"
    else:
        return "llava_v1"


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-path", type=str, required=True)
    argparser.add_argument("--model-base", type=str, default=None)
    argparser.add_argument("--sky-image", type=str, default="test_gen/sky.png")
    argparser.add_argument("--ground-image", type=str, default="test_gen/ground.jpg")
    argparser.add_argument("--conv-mode", type=str, default=None)
    args = argparser.parse_args()

    # Load model and tokenizer using canonical loader
    model_name = get_model_name_from_path(args.model_path)
    print(f"Loading model: {model_name} from {args.model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=args.model_path,
        model_base=args.model_base,
        model_name=model_name,
        load_8bit=False,
        load_4bit=False,
    )

    # Infer or override conversation mode
    conv_mode = infer_conv_mode("qwen", override=args.conv_mode)
    print(f"Using conversation mode: {conv_mode}")
    
    print("Tokenizer: ", tokenizer)

    # model.eval()

    # Load images
    sky_image = load_image(args.sky_image)
    ground_image = load_image(args.ground_image)

    # Question-image pairs
    qa_pairs = [
        (sky_image, "Hi! What is your name?"),
        (sky_image, "What color is the sky?"),
        (sky_image, "What color is the sky? Answer with a single word or phrase."),
        (ground_image, "Where is the sky located with respect to the ground?"),
        (ground_image, "Where is the sky located with respect to the ground? Answer with a single word or phrase."),
    ]

    for image, question in qa_pairs:
        print("Question:", question)
        answer = ask(model, tokenizer, image_processor, question, [image], conv_mode=conv_mode)
        print("Answer:", answer)
        print()