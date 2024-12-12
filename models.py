from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from typing import Dict, Any
import json

class ModelTrainer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def prepare_dataset(self, dataset_name: str, subset: str = None):
        """Load and prepare dataset for training"""
        dataset = load_dataset(dataset_name, subset) if subset else load_dataset(dataset_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        return tokenized_dataset
    
    def train(self, dataset, hyperparameters: Dict[str, Any]):
        """Train the model with given hyperparameters"""
        training_args = TrainingArguments(
            output_dir="./results",
            num_train_epochs=hyperparameters.get("num_epochs", 3),
            per_device_train_batch_size=hyperparameters.get("batch_size", 8),
            learning_rate=hyperparameters.get("learning_rate", 5e-5),
            weight_decay=hyperparameters.get("weight_decay", 0.01),
            logging_dir="./logs",
            logging_steps=100,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["validation"] if "validation" in dataset else None,
        )
        
        # Train the model
        train_result = trainer.train()
        
        # Get metrics
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics["train_runtime"],
            "samples_per_second": train_result.metrics["train_samples_per_second"]
        }
        
        if "validation" in dataset:
            eval_results = trainer.evaluate()
            metrics.update({
                "eval_loss": eval_results["eval_loss"],
                "eval_perplexity": torch.exp(torch.tensor(eval_results["eval_loss"])).item()
            })
            
        return metrics
