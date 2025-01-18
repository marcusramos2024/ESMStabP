# test_model.py

import os
import pickle
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# Configuration
MODEL_ID = "facebook/esm2_t6_8M_UR50D"  # Base model used during training
OUTPUT_DIR = "OUTPUT_DIR_2"               # Directory where checkpoints are saved
CHECKPOINT = "checkpoint-20000"         # Specify the checkpoint to load
MAX_INPUT_LENGTH = 512                  # Maximum sequence length

# Device configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA backend for inference.")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS backend for inference.")
else:
    device = torch.device("cpu")
    print("CUDA and MPS not available. Using CPU for inference.")

# Path to the specific checkpoint
checkpoint_path = os.path.join(OUTPUT_DIR, CHECKPOINT)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Load the model configuration
config = AutoConfig.from_pretrained(checkpoint_path)

# Load the trained model from the checkpoint
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint_path,
    config=config,
)
model.to(device)
model.eval()

# Load the scaler for inverse transformation
scaler_path = os.path.join(OUTPUT_DIR, "scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

def preprocess_sequence(sequence):
    """
    Tokenizes the input protein sequence.

    Args:
        sequence (str): Protein sequence.

    Returns:
        dict: Tokenized inputs.
    """
    return tokenizer(
        sequence,
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors="pt"  # Return PyTorch tensors
    )

def predict_tm(sequence):
    """
    Predicts the Tm value for a given protein sequence.

    Args:
        sequence (str): Protein sequence.

    Returns:
        float: Predicted Tm value.
    """
    # Preprocess the sequence
    inputs = preprocess_sequence(sequence)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze()  # Shape: (1,)
        predicted_scaled = logits.item()   # Get scalar value

    # Inverse transform to get original scale
    predicted_original = scaler.inverse_transform(np.array([[predicted_scaled]]))[0][0]
    return predicted_original

def predict_tm(sequence):
    """
    Placeholder for the actual Tm prediction function.
    Replace this with your actual implementation.
    """
    # Example implementation (replace with your model's prediction)
    # For demonstration, let's assume Tm is the length of the sequence times a factor
    return len(sequence) * 0.5  # Dummy prediction

def main():
    csv_file = 'OUTPUT_DIR_2/train_data.csv'

    data = pd.read_csv(csv_file)

    sequences = data['sequence'].tolist()
    actual_tm = data['labelTm'].tolist()
    predicted_tm = []
    successful_indices = []

    print("Predicting Tm values for the provided sequences...\n")

    for idx, seq in enumerate(sequences, 1):
        try:
            pred = predict_tm(seq)
            predicted_tm.append(pred)
            successful_indices.append(idx)
            print(f"Sequence {idx}:")
            print(f"  Input Sequence: {seq}")
            print(f"  Actual Tm: {actual_tm[idx-1]}")
            print(f"  Predicted Tm: {pred:.2f}\n")
        except Exception as e:
            print(f"Error processing sequence {idx}: {e}\n")
            predicted_tm.append(np.nan)  # Placeholder for failed prediction

    # Filter out any failed predictions
    valid_predictions = ~np.isnan(predicted_tm)
    filtered_actual_tm = np.array(actual_tm)[valid_predictions]
    filtered_predicted_tm = np.array(predicted_tm)[valid_predictions]

    if len(filtered_predicted_tm) == 0:
        print("No successful predictions to evaluate.")
        return

    # Calculate evaluation metrics
    mae = mean_absolute_error(filtered_actual_tm, filtered_predicted_tm)
    mse = mean_squared_error(filtered_actual_tm, filtered_predicted_tm)
    rmse = np.sqrt(mse)
    r2 = r2_score(filtered_actual_tm, filtered_predicted_tm)
    try:
        pcc, _ = pearsonr(filtered_actual_tm, filtered_predicted_tm)
    except Exception as e:
        print(f"Error calculating Pearson Correlation Coefficient: {e}")
        pcc = np.nan

    # Display the results
    print("Evaluation Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  R² Score: {r2:.4f}")
    if not np.isnan(pcc):
        print(f"  Pearson Correlation Coefficient (PCC): {pcc:.4f}")
    else:
        print(f"  Pearson Correlation Coefficient (PCC): Calculation Failed")

if __name__ == "__main__":
    main()
