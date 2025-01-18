import os
import pickle
from models import Sequence, ESM2

esm2 = ESM2()

for i, filename in enumerate(os.listdir("Sequences")):
    filepath = os.path.join("Sequences", filename)
    
    with open(filepath, "rb") as file:
        sequence_obj = pickle.load(file)
        try:
            embedding = esm2.extractEmbeddings(sequence_obj.sequence)
            sequence_obj.embedding = embedding
            
            with open(filepath, "wb") as save_file:
                pickle.dump(sequence_obj, save_file)
                
        except Exception as e:
            with open("Log.txt", "a") as log_file:
                log_file.write(f"Index: {i}, Error: {str(e)}\n")
        