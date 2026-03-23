# Utility file. Load a model with LLaVAModelForCausalLM and ask a question.

import sys
sys.path.insert(0, '/leonardo/home/userexternal/fgaragna')

from model.language_model.llava_llama import LlavaLlamaForCausalLM

def load_model(model_path):
    model = LlavaLlamaForCausalLM.from_pretrained(model_path)
    return model

def ask(model, question):
    inputs = model.tokenizer(question, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_length=100)
    answer = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


MODEL_PATH = "/leonardo_scratch/large/userexternal/fgaragna/checkpoints/llava-v/llava-v_s2--ve-qwen2_5"

input_str = ""

while True:
    input_str = input("Enter your question (Ctrl-D to exit): ")
    if input_str == u'\u0004':
        break
    answer = ask(load_model(MODEL_PATH), input_str)
    print("Answer:", answer)