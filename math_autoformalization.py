import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from datasets import load_dataset
from sklearn.model_selection import train_test_split

# Constants
MODEL_NAME = "google/flan-t5-base"  # We'll use FLAN-T5 as our base model
OUTPUT_DIR = "math_reasoning_autoformalization_model"
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 512

def load_and_preprocess_data():
    dataset = load_dataset('AlgorithmicResearchGroup/math_reasoning_autoformalization_track')
    
    # Split the dataset into train and validation sets
    train_val_data = dataset['train'].train_test_split(test_size=0.1)
    
    return train_val_data

def preprocess_function(examples, tokenizer):
    inputs = [f"Informal Statement: {stmt}\nInformal Proof: {proof}"
              for stmt, proof in zip(examples["informal_statement"], examples["informal_proof"])]
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    
    targets = [proof for proof in examples["formal_proof"]]
    with tokenizer.as_target_tokenizer():
        model_targets = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    
    model_inputs["labels"] = model_targets["input_ids"]
    return model_inputs

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # You might want to implement more sophisticated metrics here,
    # such as BLEU score or custom metrics for Lean 3 correctness
    return {"accuracy": sum(pred == label for pred, label in zip(decoded_preds, decoded_labels)) / len(decoded_preds)}

def main():
    # Load and preprocess data
    dataset = load_and_preprocess_data()
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    
    # Preprocess the datasets
    tokenized_datasets = dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    # Define training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=3,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )
    
    # Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation Results: {eval_results}")

if __name__ == "__main__":
    main()  