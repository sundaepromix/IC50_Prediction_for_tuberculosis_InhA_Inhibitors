# IC50 Prediction for Tuberculosis InhA Inhibitors

## Overview
A machine learning-based tool for predicting IC50 values of potential inhibitors targeting InhA (Enoyl-ACP Reductase) in Mycobacterium tuberculosis. This project combines cheminformatics, QSAR modeling, and web development to create a practical tool for TB drug discovery.

## Project Description
This project focuses on developing a predictive model for identifying potential drug candidates against tuberculosis by targeting InhA, a critical enzyme in M. tuberculosis's fatty acid synthesis pathway. The model predicts IC50 values (half maximal inhibitory concentration) of chemical compounds, helping researchers prioritize promising candidates for experimental testing.

## Key Features
- QSAR model for IC50 prediction
- Chemical structure processing and standardization
- Molecular descriptor generation
- Interactive web interface using Streamlit
- Real-time prediction capabilities

## Technical Details

### Data Source
- ChEMBL database (CHEMBL1849)
- 415 compounds with experimental IC50 values
- Validated InhA inhibitors

### Model Performance
- R² Score: 0.401
- RMSE: 0.948
- MAE: 0.598
- Y-scrambling validation confirms model robustness

### Features Used
- Molecular descriptors (MW, LogP, TPSA, HBA, HBD, RotBonds)
- Morgan fingerprints (1024 bits)
- PCA-reduced feature space (50 components)

## Installation and Setup

### Requirements
```txt
numpy==1.24.3
pandas==2.0.2
streamlit==1.24.0
rdkit==2023.3.2
scikit-learn==1.2.2
matplotlib==3.7.1
seaborn==0.12.2
pickle5==0.0.11

### Installation Steps

Clone the repository: git clone https://github.com/sundaepromix/IC50_Prediction_for_tuberculosis_InhA_Inhibitors.git

Install required packages: pip install -r requirements.txt

Run the Streamlit app: streamlit run app.py

## Usage

Input SMILES string of your compound
Get predicted IC50 value
View molecular descriptors
Assess prediction confidence

## Project Structure
├── README.md
├── requirements.txt
├── app.py                    # Streamlit application
├── model
│   └── inha_qsar_model.pkl  # Trained model
├── notebooks
│   ├── 1_data_preparation.ipynb
│   ├── 2_feature_engineering.ipynb
│   ├── 3_model_development.ipynb
│   └── 4_model_validation.ipynb
└── data
    └── processed_data.csv

## Model Development Process

Data cleaning and standardization
Feature engineering and selection
Model training and optimization
Validation and performance assessment
Web application development

## Limitations and Considerations

Model performs best for compounds structurally similar to training set
Predictions most reliable in medium to high activity ranges
Consider applicability domain when interpreting predictions

## Future Improvements

## Expand training dataset
Implement ensemble methods
Add uncertainty quantification
Include more molecular descriptors
Enhance user interface

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.
### Contact

Promise Sunday
Email: sundaepromix@gmail.com

## Acknowledgments

ChEMBL database for providing bioactivity data
RDKit community for cheminformatics tools
Streamlit team for the web framework


Made with ❤️ for advancing TB drug discovery
