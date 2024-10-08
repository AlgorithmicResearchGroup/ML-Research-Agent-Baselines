import torch
from datasets import load_dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Step 1: Set up the model architecture
MODEL_NAME = "gpt2"  # We'll use GPT-2's architecture, but initialize from scratch
VOCAB_SIZE = 50257  # GPT-2's vocabulary size
OUTPUT_DIR = "minipile_gpt2_model"

def create_model_from_scratch():
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=6,
        n_head=12,
    )
    model = GPT2LMHeadModel(config)
    model.init_weights()
    return model

# Step 2: Load and preprocess the dataset
def load_and_preprocess_data():
    dataset = load_dataset("AlgorithmicResearchGroup/minipile")
    tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 doesn't have a pad token by default

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=1024, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets, tokenizer

# Step 3: Configure the training pipeline
def create_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Adjust based on performance and time constraints
        per_device_train_batch_size=8,  # Reduced batch size due to GPT-2's larger context
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        eval_steps=10000,
        save_steps=10000,
        warmup_steps=10000,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=500,
        fp16=True,  # Enable mixed precision training
        dataloader_num_workers=4,
        gradient_accumulation_steps=4,
    )

# Step 4: Train the model
def main():
    model = create_model_from_scratch()
    tokenized_datasets, tokenizer = load_and_preprocess_data()
    training_args = create_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    trainer.train()

    # Step 5: Save the model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()