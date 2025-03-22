import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import json
from tensorflow.keras.models import load_model


# Step 1: Load Dataset
file_path = "Polymer_final.xlsx" 
data = pd.read_excel(file_path)

# Step 2: Preprocess Dataset
property_names = ['Tensile_Strength(Mpa)', 'Ionisation_Energy(eV)', 'Electron_Affinity(eV)', 'LogP', 'Refractive_Index', 'Molecular_Weight(g/mol)']
scaler_properties = MinMaxScaler()

# Normalize property values
data[property_names] = scaler_properties.fit_transform(data[property_names])

# Define the function to convert SMILES to vector
def smiles_to_vector(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    # Using GetMorganFingerprintAsBitVect to create the fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    return np.array(fp)

# Apply SMILES vectorization
data['SMILES_Vector'] = data['SMILES_Notation'].apply(smiles_to_vector)
X = np.array(data['SMILES_Vector'].tolist())
Y = data[property_names].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 3: Define Neural Network with Regularization (Dropout and Adam optimizer)
model = Sequential()

# Input layer
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))

# Hidden layers with dropout for regularization
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))  # Dropout layer to prevent overfitting

# Output layer (No activation for regression)
model.add(Dense(Y_train.shape[1]))

# Compile the model with Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Step 4: Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Step 5: Train Neural Network with EarlyStopping
history = model.fit(X_train, Y_train,
                    epochs=1000,
                    batch_size=32,
                    validation_data=(X_test, Y_test),
                    callbacks=[early_stopping])

# Step 6: Evaluate Model on Test Data
Y_train_pred = model.predict(X_train)
Y_test_pred = model.predict(X_test)

# Rescale predictions back to original scale
Y_train_pred = scaler_properties.inverse_transform(Y_train_pred)
Y_test_pred = scaler_properties.inverse_transform(Y_test_pred)


# Step 7: Predict Properties for Input SMILES
input_smiles = "CC"  # Example input
input_vector = smiles_to_vector(input_smiles).reshape(1, -1)
predicted_properties = model.predict(input_vector).flatten()

# Unnormalize predictions for readability
unnormalized_properties = scaler_properties.inverse_transform([predicted_properties]).flatten()

print(f"\nInput SMILES: {input_smiles}")
print("\nPredicted Properties:")
for name, value in zip(property_names, unnormalized_properties):
    print(f"  {name}: {value:.2f}")



# Step 1: Save the Trained Model (in H5 format)
model.save('model.h5')  # Save the model to a .h5 file

# Step 2: Save the Scaler for Property Values (using pickle)
with open('scaler_properties.pkl', 'wb') as f:
    pickle.dump(scaler_properties, f)

# Step 3: Save Property Names (using JSON)
with open('property_names.json', 'w') as f:
    json.dump(property_names, f)

print("Model, scaler, and property names saved locally!")

