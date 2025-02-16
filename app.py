import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
import pickle

# Load the saved model and components 
def load_model():
   with open('inha_qsar_model.pkl', 'rb') as f:
       return pickle.load(f)

# Calculate both descriptors and fingerprints
def calculate_features(smiles):
   mol = Chem.MolFromSmiles(smiles)
   if mol is not None:
       # Calculate descriptors
       descriptors = {
           'MW': Descriptors.ExactMolWt(mol),
           'LogP': Descriptors.MolLogP(mol),
           'TPSA': Descriptors.TPSA(mol),
           'HBA': Descriptors.NumHAcceptors(mol),
           'HBD': Descriptors.NumHDonors(mol),
           'RotBonds': Descriptors.NumRotatableBonds(mol)
       }
       
       # Generate Morgan fingerprints
       fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
       fingerprint_bits = list(fp.ToBitString())
       fingerprint_features = {f'bit_{i}': int(b) for i, b in enumerate(fingerprint_bits)}
       
       # Combine descriptors and fingerprints
       return {**descriptors, **fingerprint_features}
   return None

def main():
   st.title("InhA Inhibitor Activity Predictor")
   st.write("Predict pIC50 values for M. tuberculosis InhA inhibitors")

   # Sidebar
   st.sidebar.header("About")
   st.sidebar.write("""
   This app predicts the activity (pIC50) of potential InhA inhibitors.
   Enter a SMILES string to get predictions.
   """)

   # Main input area
   st.subheader("Input SMILES")
   smiles_input = st.text_area("Enter SMILES string:", height=100)
   
   if st.button("Predict"):
       if smiles_input:
           try:
               # Calculate all features
               features = calculate_features(smiles_input)
               
               if features:
                   # Convert to DataFrame with proper column order
                   model_components = load_model()
                   df = pd.DataFrame([features])
                   
                   # Ensure columns match training data
                   df = df[model_components['feature_names']]
                   
                   # Scale features
                   scaled_features = model_components['scaler'].transform(df)
                   
                   # Apply PCA
                   pca_features = model_components['pca'].transform(scaled_features)
                   
                   # Make prediction
                   prediction = model_components['model'].predict(pca_features)[0]
                   
                   # Display results
                   st.success(f"Predicted pIC50: {prediction:.2f}")
                   
                   # Display molecular descriptors (only the non-fingerprint features)
                   st.subheader("Molecular Descriptors")
                   descriptor_df = pd.DataFrame({k: [v] for k, v in features.items() 
                                              if not k.startswith('bit_')})
                   st.write(descriptor_df)
                   
               else:
                   st.error("Invalid SMILES string. Please check your input.")
           except Exception as e:
               st.error(f"An error occurred: {str(e)}")
       else:
           st.warning("Please enter a SMILES string.")

   # Example section
   with st.expander("See example SMILES"):
       st.code("O=C(Nc1ccccc1)C1CC(=O)N(C2CCCCC2)C1")
       if st.button("Use Example"):
           st.text_area("Enter SMILES string:", "O=C(Nc1ccccc1)C1CC(=O)N(C2CCCCC2)C1")

if __name__ == '__main__':
   main()