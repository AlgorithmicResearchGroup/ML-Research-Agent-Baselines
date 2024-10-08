import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from merging.LlamaAvg import average_models  # Assuming we're using LlamaAvg.py

# Step 1: Define the models to be merged
MODEL_NAMES = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1",
    "google/gemma-7b",
]
OUTPUT_DIR = "merged_model"

def load_models_and_tokenizers(model_names):
    models = []
    tokenizers = []
    for name in model_names:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(name)
        models.append(model)
        tokenizers.append(tokenizer)
    return models, tokenizers

# Step 2: Implement the merging process
def merge_models(models):
    merged_model = average_models(models)
    return merged_model

# Step 3: Evaluate the merged model on validation datasets
def evaluate_model(model, tokenizer, dataset_name):
    dataset = load_dataset('AlgorithmicResearchGroup/llm_merging', dataset_name)
    
    total_correct = 0
    total_samples = 0
    
    for sample in dataset['train']:
        if dataset_name == 'cosmosqa':
            prompt = f"{sample['input']}\nChoices:\n"
            for i, choice in enumerate(sample['answer_choices']):
                prompt += f"{i+1}. {choice}\n"
            prompt += "Answer: "
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=5)
            generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if str(sample['label'] + 1) in generated_answer:
                total_correct += 1
        
        elif dataset_name == 'xsum':
            inputs = tokenizer(sample['input'], return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=100)
            generated_summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # For simplicity, we'll just check if any words from the target summary are in the generated summary
            if any(word in generated_summary for word in sample['target'].split()):
                total_correct += 1
        
        total_samples += 1
    
    accuracy = total_correct / total_samples
    print(f"Accuracy on {dataset_name}: {accuracy:.4f}")

def main():
    # Load and merge models
    models, tokenizers = load_models_and_tokenizers(MODEL_NAMES)
    merged_model = merge_models(models)
    
    # Save the merged model
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizers[0].save_pretrained(OUTPUT_DIR)  # Save the tokenizer of the first model
    
    # Evaluate on validation datasets
    evaluate_model(merged_model, tokenizers[0], 'cosmosqa')
    evaluate_model(merged_model, tokenizers[0], 'xsum')