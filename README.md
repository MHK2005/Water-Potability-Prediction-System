# 💧 Water Potability Prediction System

A machine-learning web application that predicts water potability from nine measured water-quality parameters.

## Features

- Nine water-quality inputs
- Missing-value imputation learned from training data
- StandardScaler preprocessing
- Neural-network model
- Potable / Not Potable prediction

## Input Features

- `ph`
- `Hardness`
- `Solids`
- `Chloramines`
- `Sulfate`
- `Conductivity`
- `Organic_carbon`
- `Trihalomethanes`
- `Turbidity`

## Project Structure

```text
Water-Potability-Prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── water_potability.csv
├── models/
│   └── water_potability_pipeline.joblib
├── notebooks/
│   └── Water_Potability.ipynb
└── src/
    └── train_model.py
```

## Run Locally

```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
```

## Model

- `MLPRegressor`
- Hidden layers: 64 and 32 neurons
- ReLU activation
- Classification threshold: 0.5

## Disclaimer

This project is for educational and machine-learning demonstration purposes. Its predictions are not a substitute for certified laboratory water testing.
