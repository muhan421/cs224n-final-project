import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pickle
import os
from dotenv import load_dotenv

def get_model_and_tokenizer(model_name: str, hf_token: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quantization_config, token=hf_token
    )
    return model, tokenizer

def load_steering_vector(file_path):
    with open(file_path, 'rb') as file:
        steering_vector = pickle.load(file)
    return steering_vector

def generate_text(prompt: str, model, tokenizer) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def main():
    load_dotenv('keys.env')
    HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

    model_size = "7b"
    model_name = f"meta-llama/Llama-2-{model_size}-chat-hf"
    model, tokenizer = get_model_and_tokenizer(model_name, HUGGINGFACE_TOKEN)

    corr_path = 'steering_vectors/corrigibility_steering_vector_13.pkl'
    syco_path = 'steering_vectors/sycophancy_steering_vector_14.pkl'
    truth_path = 'steering_vectors/truthfulness_steering_vector_15.pkl'

    corrigibility_steering_vector = load_steering_vector(corr_path)
    sycophancy_steering_vector = load_steering_vector(syco_path)
    truthfulness_steering_vector = load_steering_vector(truth_path)

    prompt = "Say three words"

    with corrigibility_steering_vector.apply(model, multiplier=1.0, min_token_index=0):
        with sycophancy_steering_vector.apply(model, multiplier=1.0, min_token_index=0):
            with truthfulness_steering_vector.apply(model, multiplier=1.0, min_token_index=0):
                generated_text = generate_text(prompt, model, tokenizer)
                print(f"Steered model output for the prompt: {generated_text}")

if __name__ == "__main__":
    main()
