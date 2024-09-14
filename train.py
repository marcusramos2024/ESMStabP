import os
import pandas as pd
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold

class Protein:
    def __init__(self):
        self.labelTemp = None
        self.id = None
        self.sequence = None
        self.growthTemp = None
        self.lysate = None
        self.cell = None
        self.thermophilic = None
        self.nonThermophilic = None

def read_embedding(model, id):
    if model == "esm2_t33_650M_UR50D":
        embedding = torch.load(os.path.join(f"Embeddings/{model}", id + ".pt"))
        return embedding['mean_representations'][33]

    
def prepare_proteins(dataframe):
    proteins = []
    for index, row in dataframe.iterrows():
        protein = Protein()
        protein.id = row["Protein"]
        protein.sequence = row["sequence"]
        protein.growthTemp = row["growth_temp"]
        protein.lysate = row["lysate"]
        protein.cell = row["cell"]
        protein.labelTemp = row["label_tm"]
        protein.thermophilic = row["thermophilic"]
        protein.nonThermophilic = row["nonThermophilic"]
        proteins.append(protein)
    return proteins

def train_model():
    print("Loading dataset...")
    train_df = pd.read_csv("Datasets/Base.csv")
       
    train_proteins = prepare_proteins(train_df)
    
    xs_train = []
    ys_train = []

    print("Preparing features and labels...")
    for p in train_proteins:
        x = read_embedding("Embeddings", p.id)
        x = np.append(x, p.growthTemp) 
        x = np.append(x, p.lysate)
        x = np.append(x, p.cell)        
        x = np.append(x, p.thermophilic)
        x = np.append(x, p.nonThermophilic)

        y = p.labelTemp
        xs_train.append(x)
        ys_train.append(y)

    xs_train = np.array(xs_train)
    ys_train = np.array(ys_train)

    print("Initializing model...")
    
    regressionModel = RandomForestRegressor()


    print("Performing cross-validation...")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    r2_scores = []
    mse_scores = []
    mae_scores = []
    rmse_scores = []
    pcc_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(xs_train)):
        print(f"Training fold {fold + 1}...")
        
        x_train_fold, x_val_fold = xs_train[train_idx], xs_train[val_idx]
        y_train_fold, y_val_fold = ys_train[train_idx], ys_train[val_idx]
        
        regressionModel.fit(x_train_fold, y_train_fold)
        
        y_pred_fold = regressionModel.predict(x_val_fold)
        
        r2 = r2_score(y_val_fold, y_pred_fold)
        mse = mean_squared_error(y_val_fold, y_pred_fold)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_fold, y_pred_fold)
        pcc, _ = pearsonr(y_val_fold, y_pred_fold)
        
        r2_scores.append(r2)
        mse_scores.append(mse)
        mae_scores.append(mae)
        rmse_scores.append(rmse)
        pcc_scores.append(pcc)
        
        print(f"Fold {fold + 1} - R2: {r2:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, PCC: {pcc:.4f}")

    print("Cross-validation complete.")
    print(f"Average R2: {np.mean(r2_scores):.4f}")
    print(f"Average MSE: {np.mean(mse_scores):.4f}")
    print(f"Average RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Average MAE: {np.mean(mae_scores):.4f}")
    print(f"Average PCC: {np.mean(pcc_scores):.4f}")

    print("Retraining model on full dataset...")
    regressionModel.fit(xs_train, ys_train)

    model_path = os.path.join(f"Models/ESMStabP.joblib")

    joblib.dump(regressionModel, model_path)
    print(f"Model saved to {model_path}")


train_model()