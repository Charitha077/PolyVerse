import json
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from rdkit import Chem
from rdkit.Chem import AllChem
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Load the trained model, scaler, and property names
model = load_model('model.h5')  # Load your pre-trained model
with open('scaler_properties.pkl', 'rb') as f:
    scaler_properties = pickle.load(f)
with open('property_names.json', 'r') as f:
    property_names = json.load(f)
print("Loaded model and data completely")

# Load polymer dataset
polymer_df = pd.read_excel('Polymer_final.xlsx')
polymer_mapping = polymer_df.set_index('Polymer_Name')['SMILES_Notation'].to_dict()
polymer_uses = polymer_df.set_index('Polymer_Name')['Uses'].to_dict()  # Column name 'Uses' as provided

# Function to normalize polymer names
def normalize_name(name):
    return ''.join(name.split()).lower()

# Function to convert polymer name to SMILES
def name_to_smiles(name):
    # Normalize the input name
    normalized_name = normalize_name(name)

    # Normalize the polymer names in the dataset
    normalized_mapping = {normalize_name(k): v for k, v in polymer_mapping.items()}

    # Look up the normalized polymer name in the normalized dataset dictionary
    smiles = normalized_mapping.get(normalized_name)
    return smiles

# Function to convert SMILES to vector using RDKit (Morgan Fingerprints)
def smiles_to_vector(smiles):
    if not smiles or not isinstance(smiles, str):
        raise ValueError("Invalid SMILES: SMILES string is either empty or not a string")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    # Get Morgan fingerprint and convert it to a bit vector
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp).reshape(1, -1)

# Function to predict properties based on input (name or SMILES)
def predict_properties(input_data):
    try:
        # Check if input is a polymer name
        if 'name' in input_data:
            polymer_name = input_data.get('name')
            input_smiles = name_to_smiles(polymer_name)
            if not input_smiles:
                return jsonify({"error": f"SMILES not found for polymer name: {polymer_name}"}), 400

        # If input is not a name, assume it's SMILES
        else:
            input_smiles = input_data.get('smiles')
            if not input_smiles:
                return jsonify({"error": "SMILES string is missing in request data"}), 400

        # Convert SMILES to vector
        input_vector = smiles_to_vector(input_smiles)

        # Predict properties using the model
        predicted_properties = model.predict(input_vector).flatten()

        # Unnormalize predictions
        unnormalized_properties = scaler_properties.inverse_transform([predicted_properties]).flatten()

        # Map predicted properties to corresponding names
        predictions = {property_names[i]: round(unnormalized_properties[i], 2) for i in range(len(property_names))}

        return predictions

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided in the request"}), 400

        # Predict properties based on input (name or SMILES)
        predictions = predict_properties(data)

        # Send predictions back in JSON format
        response = {
            "predicted_properties": predictions
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/analysis', methods=['POST'])
def analysis():
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({"error": "Polymer name is missing in request data"}), 400

        # Get polymer name from the request
        polymer_name = data.get('name')

        # Get the SMILES for the polymer name
        input_smiles = name_to_smiles(polymer_name)
        if not input_smiles:
            return jsonify({"error": f"SMILES not found for polymer name: {polymer_name}"}), 400

        # Get the uses information for the polymer
        polymer_uses_info = polymer_uses.get(polymer_name, "Uses information not available")

        # Return the SMILES and uses back in JSON format
        response = {
            "smiles": input_smiles,
            "uses": polymer_uses_info
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)