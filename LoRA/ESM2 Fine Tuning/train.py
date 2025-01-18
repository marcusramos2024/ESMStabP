MODEL_ID = "facebook/esm2_t6_8M_UR50D"
OUTPUT_DIR = "OUTPUT_DIR_2"

import os
import glob
import pickle
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
import pandas as pd
import warnings
import torch
from sklearn.preprocessing import StandardScaler

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)

MAX_SEQUENCE_LENGTH = 512
LEARNING_RATE = 2e-5
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIMIT = 3
RUN_NAME = "Tm_Prediction_Run"


class Sequence:
    def __init__(self):
        self.sequence = None
        self.id = None
        self.ESM2_embedding = None
        self.labelTm = None

SEQUENCES_DIR = "Sequences"

def load_sequences(sequences_dir):
    """
    Loads all pickle files from the specified directory and extracts sequences and Tm labels.

    Args:
        sequences_dir (str): Path to the directory containing .pkl files.

    Returns:
        pd.DataFrame: DataFrame with 'sequence' and 'labelTm' columns.
    """
    sequences = []
    labels = []
    files = glob.glob(os.path.join(sequences_dir, "*.pkl"))
    for file in files:
        try:
            with open(file, "rb") as f:
                seq_obj = pickle.load(f)
                if seq_obj.sequence is not None and seq_obj.labelTm is not None:
                    sequences.append(seq_obj.sequence)
                    labels.append(seq_obj.labelTm)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    return pd.DataFrame({"sequence": sequences, "labelTm": labels})

df = load_sequences(SEQUENCES_DIR)


# Split the dataset into training and testing sets
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index).reset_index(drop=True)
train_df = train_df.reset_index(drop=True)

train_df.to_csv(os.path.join(OUTPUT_DIR, "train_data.csv"), index=False)
test_df.to_csv(os.path.join(OUTPUT_DIR, "test_data.csv"), index=False)


# Normalize the labels
scaler = StandardScaler()
train_labels = train_df["labelTm"].values.reshape(-1, 1)
test_labels = test_df["labelTm"].values.reshape(-1, 1)

# Fit scaler on training labels and transform labels
scaler.fit(train_labels)
train_df["labelTm"] = scaler.transform(train_labels)
test_df["labelTm"] = scaler.transform(test_labels)

# Save the scaler for inverse transformation later
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

# Create datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)
dataset = DatasetDict({
    'train': train_dataset,
    'test': test_dataset
})

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def preprocess_function(examples):
    """
    Tokenizes the input protein sequences with appropriate padding and truncation.

    Args:
        examples (dict): A batch of examples with 'sequence' and 'labelTm'.

    Returns:
        dict: Tokenized inputs and labels.
    """
    inputs = examples['sequence']
    
    # Tokenize inputs with padding and truncation
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        padding='max_length'  # For dynamic padding, use padding='longest' or remove
    )
    
    # Ensure labels are floats
    labels = examples['labelTm']
    model_inputs["labels"] = [float(label) for label in labels]
    
    return model_inputs

# Apply the preprocessing to the datasets
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Use DataCollatorWithPadding for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Initialize the model for regression
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=1,               # Single output for regression
    problem_type="regression"   # Specify regression problem
)

# Define the training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    save_total_limit=SAVE_TOTAL_LIMIT,
    run_name=RUN_NAME,                            # Specify run name to avoid W&B warnings
    logging_dir=os.path.join(OUTPUT_DIR, "logs"), # Directory for logs
    logging_steps=10,                             # Adjust as needed
    report_to="wandb",                            # Specify reporting to W&B
)

# Define a metric computation function
def compute_metrics_regression(p):
    preds = p.predictions.flatten()
    labels = p.label_ids
    # Inverse transform to get original scale
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
    labels = scaler.inverse_transform(labels.reshape(-1, 1)).flatten()
    mse = np.mean((preds - labels) ** 2)
    return {"mse": mse}

# Determine device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA backend for training.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend for training.")
else:
    device = torch.device("cpu")
    print("CUDA and MPS not available. Using CPU for training.")

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],      # Ensure correct split
    data_collator=data_collator,                  # Use dynamic data collator
    tokenizer=tokenizer,                          # Pass tokenizer to handle padding
    compute_metrics=compute_metrics_regression     # Compute MSE metric
)

model.to(device)

trainer.train()

trainer.evaluate()