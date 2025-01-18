from flask import Flask, request, Response
from flask_wtf.csrf import CSRFProtect
from flask_cors import CORS  # Import CORS to handle cross-origin requests
import os
import uuid
from helper import extract_embeddings, evaluate_sequences, parse_fasta
import shutil
import pandas as pd
from io import StringIO

app = Flask(__name__)

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Enable CORS for all routes
CORS(app)

@csrf.exempt  # Exempt the route from CSRF protection
@app.route('/upload', methods=['POST'])
def upload_file():
        try:
            file = request.files['file']
            experimentalCondition = request.form.get('condition')
            growthTemp = request.form.get('growthTemp')
            if growthTemp:
                growthTemp = int(growthTemp)
            filename = f"{uuid.uuid4()}.txt"
            filepath = os.path.join("Uploads/", filename)
            embeddingsDirectory = f"Embeddings/{filename}"

            # Save FASTA
            file.save(filepath)
        except:
            print("Error: An Unexpected Error Occured")
            return "Error: An Unexpected Error Occured", 400


        # Create output df
        try:
            df = parse_fasta(filepath)
        except:
            os.remove(filepath)
            print("Error: FASTA file exceeds 1000 sequences")
            return "Error: FASTA file exceeds 1000 sequences", 400

        # Extract seqeunce embeddings from FASTA
        try:
            df = extract_embeddings(filepath, df)
        except:        
            # Delete embeddings
            for file in os.listdir(embeddingsDirectory):
                file_path = os.path.join(embeddingsDirectory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # # Delete FASTA after parsing
            os.remove(filepath)
            
            # Delete embedding directory
            shutil.rmtree(embeddingsDirectory)
            
            print("Error: Invalid FASTA file")
            return "Error: Invalid FASTA file", 400

        try:
            # Calculate Tm's from FASTA & Embeddings
            df = evaluate_sequences(embeddingsDirectory, df, growthTemp=growthTemp, experimentalCondition=experimentalCondition)
            
            # # Delete FASTA after parsing
            os.remove(filepath)

            # Delete embeddings
            for file in os.listdir(embeddingsDirectory):
                file_path = os.path.join(embeddingsDirectory, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            # Delete embedding directory
            shutil.rmtree(embeddingsDirectory)

            # Return the CSV as a response    
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            return Response(
                csv_buffer.getvalue(),
                mimetype='text/csv',
                headers={"Content-disposition": "attachment; filename=Results.csv"}
            )
        except:
            print("Error: An Unexpected Error Occured")
            return "Error: An Unexpected Error Occured", 400

if __name__ == '__main__':
    app.run(debug=True)