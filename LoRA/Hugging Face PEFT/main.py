import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import wandb
import numpy as np
import torch
import torch.nn as nn
import pickle
import xml.etree.ElementTree as ET
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    roc_auc_score, 
    matthews_corrcoef
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from accelerate import Accelerator
# Imports specific to the custom peft lora model
from peft import get_peft_config, PeftModel, PeftConfig, get_peft_model, LoraConfig, TaskType


# Helper Functions and Data Preparation
def truncate_labels(labels, max_length):
    """Truncate labels to the specified max_length."""
    return [label[:max_length] for label in labels]

def compute_metrics(p):
    """Compute metrics for evaluation."""
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    # Remove padding (-100 labels)
    predictions = predictions[labels != -100].flatten()
    labels = labels[labels != -100].flatten()
    
    # Compute accuracy
    accuracy = accuracy_score(labels, predictions)
    
    # Compute precision, recall, F1 score, and AUC
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    auc = roc_auc_score(labels, predictions)
    
    # Compute MCC
    mcc = matthews_corrcoef(labels, predictions) 
    
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc, 'mcc': mcc} 

def compute_loss(model, inputs):
    """Custom compute_loss function."""
    logits = model(**inputs).logits
    labels = inputs["labels"]
    loss_fct = nn.CrossEntropyLoss(weight=class_weights)
    active_loss = inputs["attention_mask"].view(-1) == 1
    active_logits = logits.view(-1, model.config.num_labels)
    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    loss = loss_fct(active_logits, active_labels)
    return loss

# Load the data from pickle files (replace with your local paths)
with open("train_sequences_chunked_by_family.pkl", "rb") as f:
    train_sequences = pickle.load(f)

with open("test_sequences_chunked_by_family.pkl", "rb") as f:
    test_sequences = pickle.load(f)

with open("train_labels_chunked_by_family.pkl", "rb") as f:
    train_labels = pickle.load(f)

with open("test_labels_chunked_by_family.pkl", "rb") as f:
    test_labels = pickle.load(f)

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D")
max_sequence_length = 1000

train_tokenized = tokenizer(train_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)
test_tokenized = tokenizer(test_sequences, padding=True, truncation=True, max_length=max_sequence_length, return_tensors="pt", is_split_into_words=False)

# Directly truncate the entire list of labels
train_labels = truncate_labels(train_labels, max_sequence_length)
test_labels = truncate_labels(test_labels, max_sequence_length)

train_dataset = Dataset.from_dict({k: v for k, v in train_tokenized.items()}).add_column("labels", train_labels)
test_dataset = Dataset.from_dict({k: v for k, v in test_tokenized.items()}).add_column("labels", test_labels)

# Compute Class Weights
classes = [0, 1]  
flat_train_labels = [label for sublist in train_labels for label in sublist]
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=flat_train_labels)
accelerator = Accelerator()
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(accelerator.device)

# Define Custom Trainer Class
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = compute_loss(model, inputs)
        return (loss, outputs) if return_outputs else loss
def train_function_no_sweeps(train_dataset, test_dataset):
    
    # Set the LoRA config
    config = {
        "lora_alpha": 1, #try 0.5, 1, 2, ..., 16
        "lora_dropout": 0.2,
        "lr": 5.701568055793089e-04,
        "lr_scheduler_type": "cosine",
        "max_grad_norm": 0.5,
        "num_train_epochs": 3,
        "per_device_train_batch_size": 12,
        "r": 2,
        "weight_decay": 0.2,
        # Add other hyperparameters as needed
    }
    # The base model you will train a LoRA on top of
    model_checkpoint = "facebook/esm2_t12_35M_UR50D"  
    
    # Define labels and model
    id2label = {0: "No binding site", 1: "Binding site"}
    label2id = {v: k for k, v in id2label.items()}
    model = AutoModelForTokenClassification.from_pretrained(model_checkpoint, num_labels=len(id2label), id2label=id2label, label2id=label2id)

    # Convert the model into a PeftModel
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS, 
        inference_mode=False, 
        r=config["r"], 
        lora_alpha=config["lora_alpha"], 
        target_modules=["query", "key", "value"], # also try "dense_h_to_4h" and "dense_4h_to_h"
        lora_dropout=config["lora_dropout"], 
        bias="none" # or "all" or "lora_only" 
    )
    model = get_peft_model(model, peft_config)

    # Use the accelerator
    model = accelerator.prepare(model)
    train_dataset = accelerator.prepare(train_dataset)
    test_dataset = accelerator.prepare(test_dataset)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Training setup
    training_args = TrainingArguments(
        output_dir=f"esm2_t12_35M-lora-binding-sites_{timestamp}",
        learning_rate=config["lr"],
        lr_scheduler_type=config["lr_scheduler_type"],
        gradient_accumulation_steps=1,
        max_grad_norm=config["max_grad_norm"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        num_train_epochs=config["num_train_epochs"],
        weight_decay=config["weight_decay"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        push_to_hub=False,
        logging_dir=None,
        logging_first_step=False,
        logging_steps=200,
        save_total_limit=7,
        no_cuda=False,
        seed=8893,
        fp16=True,
        report_to='wandb'
    )

    # Initialize Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Train and Save Model
    trainer.train()
    save_path = os.path.join("lora_binding_sites", f"best_model_esm2_t12_35M_lora_{timestamp}")
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)

train_function_no_sweeps(train_dataset, test_dataset) # Train!