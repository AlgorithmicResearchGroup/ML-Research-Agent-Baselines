import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

# Step 1: Choose a base model
MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "fine_tuned_model"

# Step 2: Select an open-source dataset
DATASET_NAME = "databricks/databricks-dolly-15k"

# Step 3: Set up the training script
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)

# Add padding token to the tokenizer
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id

# Load the dataset
dataset = load_dataset(DATASET_NAME)

# Preprocess the dataset
def preprocess_function(examples):
    # Format the input as per the instruction-following format
    prompts = [
        f"Instruction: {instruction}\n\nContext: {context}\n\nResponse: {response}"
        for instruction, context, response in zip(examples['instruction'], examples['context'], examples['response'])
    ]
    
    # Tokenize the formatted prompts
    tokenized_inputs = tokenizer(prompts, truncation=True, padding="max_length", max_length=512)
    
    return tokenized_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=['instruction', 'context', 'response', 'category'])

# Set up LoRA for efficient fine-tuning
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)
model = get_peft_model(model, peft_config)

def main():
    # Step 4: Configure hyperparameters for efficient training
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        save_steps=1000,
        logging_steps=100,
        fp16=True,
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()