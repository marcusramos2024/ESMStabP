import pandas as pd
import pathlib
import torch
from esm import FastaBatchedDataset, pretrained
import uuid
import os
import pandas as pd
import torch
import numpy as np
import joblib

class Protein:
    def __init__(self):
        pass

def evaluate_sequences(embeddingsDirectory, df, growthTemp=None, experimentalCondition=None):
    xs_test = []
    ys_test = []

    for index, row in df.iterrows():
        embeddingPath = embeddingsDirectory + "/" + str(row["ID"]) + ".pt"
        embedding = torch.load(embeddingPath)
        x = embedding['mean_representations'][33]

        if growthTemp:
            x = np.append(x, growthTemp) 
            if growthTemp > 60:
                x = np.append(x, 1) # thermophilic
            else:
                x = np.append(x, 0) # thermophilic

            if growthTemp < 30:
                x = np.append(x, 1) # nonThermophilic
            else:
                x = np.append(x, 0) # nonThermophilic
        
        if experimentalCondition:
            if experimentalCondition == "Cell":
                x = np.append(x, 1) # Cell
                x = np.append(x, 0) # Lysate
            
            else:
                x = np.append(x, 0) # Cell
                x = np.append(x, 1) # Lysate


        xs_test.append(x)

    xs_test = np.array(xs_test)

    model = "Models/"

    if growthTemp and experimentalCondition:
        model += "4.joblib"
    elif not growthTemp and experimentalCondition:
        model += "3.joblib"
    elif growthTemp and not experimentalCondition:
        model += "2.joblib"
    else:
        model += "1.joblib"

    regressionModel = joblib.load(model)

    y_pred = regressionModel.predict(xs_test)

    output_data = {
        "ID": [],
        "Tm": [],
        "Sequence": [],
    }

    for i in range(len(y_pred)):
        output_data["ID"].append(df.iloc[i]["ID"])
        output_data["Tm"].append(y_pred[i])
        output_data["Sequence"].append(df.iloc[i]["Sequence"])


    return pd.DataFrame(output_data)

def extract_embeddings(fasta_file, df, tokens_per_batch=4096, seq_length=1022):
    model, alphabet = pretrained.load_model_and_alphabet("esm2_t33_650M_UR50D")
    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()
    
    try:
        dataset = FastaBatchedDataset.from_file(fasta_file)
    except: Exception("Error: Invalid FASTA file")
    
    batches = dataset.get_batch_indices(tokens_per_batch, extra_toks_per_seq=1)

    data_loader = torch.utils.data.DataLoader(
        dataset, 
        collate_fn=alphabet.get_batch_converter(seq_length), 
        batch_sampler=batches
    )

    output_dir = pathlib.Path("Embeddings/" + fasta_file.replace("Uploads/", ""))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):

            print(f'Processing batch {batch_idx + 1} of {len(batches)}')

            if torch.cuda.is_available():
                toks = toks.to(device="cuda", non_blocking=True)

            out = model(toks, repr_layers=[33], return_contacts=False)

            logits = out["logits"].to(device="cpu")
            representations = {layer: t.to(device="cpu") for layer, t in out["representations"].items()}
            
            for i, label in enumerate(labels):
                entry_id = label.split()[0]
                
                filename = output_dir / f"{entry_id}.pt"
                truncate_len = min(seq_length, len(strs[i]))

                result = {"entry_id": entry_id}
                result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                df["ID"] = entry_id
                torch.save(result, filename)
    return df

def parse_fasta(filepath):
    sequences = []
    with open(filepath, 'r') as file:
        seq = ''
        for line in file:
            if line.startswith('>'):
                if seq:  
                    sequences.append(seq)
                    seq = ''
            else:
                seq += line.strip()  
        if seq:
            sequences.append(seq)
    if len(sequences) > 1000:
        raise Exception("Error: FASTA file exceeds 1000 sequences")

    df = pd.DataFrame({'Sequence': sequences, 'Tm': [None] * len(sequences)})
    return df