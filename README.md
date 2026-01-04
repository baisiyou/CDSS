# Clinical Decision Support System (CDSS)

## System Overview

This system is a machine learning-based clinical decision support system designed to assist physicians in:

- Determining whether medication adjustments are necessary

- Predicting adverse drug reactions (liver and kidney dysfunction)

- Recommending treatment plans

- Providing early warnings for high-risk drug combinations

## Main Functions

### 1. Liver and Kidney Function Abnormality Prediction

- Based on the patient's medication history and laboratory indicators, predict the likelihood of:

- Kidney dysfunction (elevated creatinine, abnormal BUN)

- Liver dysfunction (abnormal transaminases, elevated INR)

### 2. Drug Combination Risk Warning

- Detecting high-risk drug combinations:

- Antibiotics + Nephrotoxic Drugs in combination

- Multiple nephrotoxic drugs in combination

- Hepatotoxic Drug Combinations

- Assessing the current risk status based on laboratory indicators

## System Architecture

``` Clinical Decision Support System (CDSS) /

├── data_preprocessing.py # Data preprocessing module

├── prediction_models.py # Predictive Model Module

├── drug_interaction_warning.py # Drug interaction warning module

├── drug_combination_analyzer.py # Drug combination analysis module

├── train_models.py # Model training script

├── cdss_api.py # Flask API service

├── drug_combination_analyzer.html # Drug combination analysis front-end interface

├── requirements.txt # Python dependencies

├── README.md # Documentation

├── One-click start and open.sh # One-click start script

├── eicu_mimic_lab_time.csv # Training data

└── models/ # Model save directory (generated after training)

├── organ_function_predictor.pkl

└── preprocessor.pkl

```

## Installation and Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt

```

### 2. Training the Model

First, you need to train the prediction model:

```bash
python train_models.py

```
This will:

- Load the `eicu_mimic_lab_time.csv` data

- Preprocess the data and create target labels

- Train the random forest model

- Save the model to the `models/` directory

**Note:** If the model file already exists, you can skip this step.

### 3. Starting the System

#### Method 1: One-Click Startup (Recommended)

```bash
chmod +x One-click start and open .sh

./ One-click start and open .sh

```

This script will:

- Automatically check and stop old processes

- Start the API service (port 5003)

- Wait for the service to be ready

- Automatically open the front-end interface

#### Method 2: Manual Startup

```bash

# Start the API service

python cdss_api.py

```

The API service will start at `http://localhost:5003`.

### 4. Using the Web Interface

1. Open the `drug_combination_analyzer.html` file (open in your browser)

2. Select or search for a drug (supports Chinese and English)

3. Click the "Start Analysis" button

4. View the analysis results:

- Overall risk assessment

- Multiple organ dysfunction prediction

- Recommended drugs

**Note:** The system supports multilingual switching (Chinese/English/Français), which can be done in the upper right corner of the page.

## API Interface

### Health Check

```
GET /health

```

### Predict Liver and Kidney Function Abnormalities

```
POST /predict
Content-Type: application/json

{
"bun": 1.2,

"inr": 0.8,

"vancomycin": 1,

"furosemide": 1,

...
}
```

### Drug Combination Risk Warning

```
POST /warn
Content-Type: application/json

{
"vancomycin": 1,

"furosemide": 1,

"bun": 2.0,

...
}
```

### Comprehensive Analysis

```
POST /analyze
Content-Type: application/json

{
"patientunitstayid": "12345",

"bun": 1.2,

"vancomycin": 1,
...

}
```

### Drug Combination Analysis
```
POST /drug_combinations
Content-Type: application/json

{
"aspirin": 1,

"prednisone": 1,

"piperacillin": 1

}
```

### Get Drug List
```
GET /drugs/list?limit=1000
```

### Get Recommended Drugs
```
POST /drugs/recommend
Content-Type: application/json

{
"drugs": ["aspirin", "prednisone"]

}
```

## Technical Implementation

### Predictive Model

- **Algorithm**: Random Forest

- **Features**: Drug usage, laboratory indicators, patient basic information

- **Target**: Binary classification (normal/abnormal)

### Risk Warning Rules

- **Nephrogenic Drugs**: Aminoglycosides, diuretics, NSAIDs, ACE inhibitors, contrast agents, etc.

- **Hepatotoxic Drugs**: Acetaminophen, amiodarone, statins, etc.

- **High-Risk Combinations**: Antibiotics + nephrotoxic drugs, multiple nephrotoxic drugs in combination

### Laboratory Indicator Assessment

- BUN (Blood Urea Nitrogen): Normal range (-2, 1.5)

- INR (International Normalized Ratio): Normal range (-2, 1.2)

- Albumin: Normal range (-1.0, 2)

## Usage Example

### Python Code Example

```python
from prediction_models import OrganFunctionPredictor
from drug_interaction_warning import DrugInteractionWarning
import pandas as pd

# Load the model
predictor = OrganFunctionPredictor()

predictor.load('models/organ_function_predictor.pkl')

# Prediction

patient_data = pd.DataFrame([{
'vancomycin': 1,

'furosemide': 1,

'bun': 2.0,

# ... Other features

}])

X = preprocessor.extract_features(patient_data)

X_scaled = preprocessor.scale_features(X)

predictions = predictor.predict_all(X_scaled)

# Warning

warning_system = DrugInteractionWarning()

warning = warning_system.generate_warning(
patient_data,

drug_columns,

lab_columns

)
```

## Notes

1. **Data Standardization**: Laboratory indicators need to be standardized before being input into the model.

2. **Model Update**: It is recommended to retrain the model regularly with new data.

3. **Clinical Validation**: System prediction results are for reference only and should be combined with clinical judgment.

4. **Data Privacy**: Ensure the security and privacy protection of patient data.

## Future Improvements

- [ ] Support more laboratory indicators

- [ ] Add time series prediction

- [ ] Integrate more drug interaction databases

- [ ] Optimize model performance

- [ ] Add model interpretability analysis

## License

This project is for learning and research purposes only.

## Contact Information

For questions or suggestions, please contact the development team.
