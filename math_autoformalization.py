import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from sklearn.model_selection import train_test_split
import evaluate
import numpy as np

OUTPUT_DIR = "math_autoformalization_model"
# Load the dataset
dataset = load_dataset('AlgorithmicResearchGroup/math_reasoning_autoformalization_track', split='train')

# Initialize the tokenizer and model
model_name = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Preprocess function
def preprocess_function(examples):
    inputs = [f"Statement: {stmt}\nProof: {proof}" for stmt, proof in zip(examples['informal_statement'], examples['informal_proof'])]
    targets = examples['formal_proof']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Preprocess the dataset
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Split the dataset
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = tokenized_dataset['train']
val_dataset = tokenized_dataset['test']

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Metric
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    result = bleu.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["bleu"]}

def main():
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    # Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save the model
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Return the BLEU score
    print(f"Final BLEU score: {eval_results['eval_bleu']}")

if __name__ == "__main__":
    main()