import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments
import math

OUTPUT_DIR = "edge_llm_training_model"

# Load datasets
c4_dataset = load_dataset('AlgorithmicResearchGroup/edge_llm_training', 'c4_combined_dataset')
alpaca_dataset = load_dataset('AlgorithmicResearchGroup/edge_llm_training', 'alpaca_cleaned')

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

# Tokenize datasets
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

c4_tokenized = c4_dataset['train'].map(tokenize_function, batched=True, remove_columns=['text'])
alpaca_tokenized = alpaca_dataset['train'].map(lambda x: {'text': f"{x['instruction']} {x['input']} {x['output']}"}).map(tokenize_function, batched=True, remove_columns=['instruction', 'input', 'output'])

# Combine datasets
combined_dataset = ConcatDataset([c4_tokenized, alpaca_tokenized])

# Define model configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=512,
    n_embd=768,
    n_layer=6,
    n_head=12,
)

# Initialize model
model = GPT2LMHeadModel(config)

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    save_steps=1000,
    fp16=True,
    dataloader_num_workers=4,
)

# Define data collator
def data_collator(features):
    return {
        'input_ids': torch.stack([torch.tensor(f['input_ids']) for f in features]),
        'attention_mask': torch.stack([torch.tensor(f['attention_mask']) for f in features]),
        'labels': torch.stack([torch.tensor(f['input_ids']) for f in features]),
    }

def main():
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=combined_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training completed. Model saved.")

if __name__ == "__main__":
    main()