import torch
from datasets import load_dataset
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Step 1: Set up the model architecture
MODEL_NAME = "roberta-base"  # We'll use RoBERTa's architecture, but initialize from scratch
VOCAB_SIZE = 50265  # RoBERTa's vocabulary size
OUTPUT_DIR = "minipile_model"

def create_model_from_scratch():
    config = RobertaConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    model = RobertaForMaskedLM(config)
    model.init_weights()
    return model

# Step 2: Load and preprocess the dataset
def load_and_preprocess_data():
    dataset = load_dataset("AlgorithmicResearchGroup/minipile")
    tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    return tokenized_datasets, tokenizer

# Step 3: Configure the training pipeline
def create_training_args():
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=1,  # Adjust based on performance and time constraints
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
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
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15),
    )

    trainer.train()

    # Step 5: Save the model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

if __name__ == "__main__":
    main()