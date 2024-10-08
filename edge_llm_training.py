import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from tqdm import tqdm

# Define model architecture
def create_small_gpt2_model():
    config = GPT2Config(
        vocab_size=50257,  # GPT-2 tokenizer vocab size
        n_positions=512,
        n_embd=256,
        n_layer=6,
        n_head=8,
        activation_function="gelu_new",
        bos_token_id=50256,
        eos_token_id=50256,
    )
    model = GPT2LMHeadModel(config)
    return model

# Load and preprocess datasets
def load_datasets():
    c4_dataset = load_dataset('AlgorithmicResearchGroup/edge_llm_training', 'c4_combined_dataset')
    alpaca_dataset = load_dataset('AlgorithmicResearchGroup/edge_llm_training', 'alpaca_cleaned')
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)
    
    def tokenize_alpaca(examples):
        text = [f"Instruction: {instr}\nInput: {inp}\nOutput: {out}" 
                for instr, inp, out in zip(examples['instruction'], examples['input'], examples['output'])]
        return tokenizer(text, truncation=True, padding='max_length', max_length=512)
    
    c4_tokenized = c4_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    alpaca_tokenized = alpaca_dataset['train'].map(tokenize_alpaca, batched=True, remove_columns=['instruction', 'input', 'output'])
    
    combined_dataset = ConcatDataset([c4_tokenized['train'], alpaca_tokenized])
    return combined_dataset, tokenizer

# Training function
def train(model, dataset, tokenizer, num_epochs=3, batch_size=32, learning_rate=5e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()
            
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} average loss: {total_loss / len(loader)}")

def main():
    model = create_small_gpt2_model()
    dataset, tokenizer = load_datasets()
    
    print("Model size:", sum(p.numel() for p in model.parameters()) * 2 / 1024 / 1024, "MB")
    
    train(model, dataset, tokenizer)
    
    # Save the model
    output_dir = "edge_llm_training_model"
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()