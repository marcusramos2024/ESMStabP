from transformers import EsmTokenizer, EsmModel
import torch

class Sequence:
    def __init__(self):
        self.sequence = None
        self.id = None
        self.ESM2_embedding = None
        self.labelTm = None

class ESM2:
    def __init__(self):
        model = "facebook/esm2_t33_650M_UR50D"
        self.tokenizer = EsmTokenizer.from_pretrained(model)
        self.model = EsmModel.from_pretrained(model)

    def extractEmbeddings(self, sequence):    
        
        # Tokenize sequence
        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.squeeze(0)
        
        return embeddings