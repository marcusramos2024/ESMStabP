import pandas as pd
from Bio import SeqIO
import pickle
import os
from models import Sequence

'''
This file reads the sequences from the Base Datasets and converts them to pickles
'''

sequences = {}

deepstab = pd.read_csv("Base Datasets/DeepStabP.csv")
for _, row in deepstab.iterrows():
    sequence = Sequence()
    sequence.id = row["ID"]
    sequence.sequence = row["Sequence"]
    sequence.labelTm = int(row["label_tm"])
    
    sequences[sequence.sequence] = sequence

deeptm = pd.read_csv("Base Datasets/DeepTM.csv")
for _, row in deeptm.iterrows():
    sequence = Sequence()
    sequence.id = row["ID"]
    sequence.sequence = row["Sequence"]
    sequence.labelTm = int(row["tm"])

    sequences[sequence.sequence] = sequence    

temberture = pd.read_csv("Base Datasets/TemBERTure.csv")
for _, row in temberture.iterrows():
    sequence = Sequence()
    sequence.id = row["Protein_ID"]
    sequence.sequence = row["Sequence"]
    sequence.labelTm = int(row["Tm"])

    sequences[sequence.sequence] = sequence

for seq_key, sequence_obj in sequences.items():
    filename = f"{sequence_obj.id}.pkl"
    filepath = os.path.join("Sequences", filename)
    
    with open(filepath, "wb") as file:
        pickle.dump(sequence_obj, file)