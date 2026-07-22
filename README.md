# Water Potability Prediction System

A machine learning project that predicts whether a water sample is **potable (safe to drink)** or **not potable** based on its physicochemical properties.

## Live Demo

### [Link](https://water-potability-predict.streamlit.app/)

---

## Project Overview

Water quality depends on several physical and chemical characteristics. This project uses machine learning to analyze nine water-quality parameters and predict the potability of a given sample.

---

## Input Parameters

The model uses the following nine water-quality characteristics:

| Parameter | Description |
|---|---|
| `ph` | pH value of the water |
| `Hardness` | Water hardness |
| `Solids` | Total dissolved solids |
| `Chloramines` | Amount of chloramines |
| `Sulfate` | Sulfate concentration |
| `Conductivity` | Electrical conductivity |
| `Organic_carbon` | Organic carbon content |
| `Trihalomethanes` | Trihalomethane concentration |
| `Turbidity` | Turbidity level |

### Target Variable

- `0` — Not Potable
- `1` — Potable

---

## Machine Learning Pipeline

The application processes data through the following pipeline:

```text
Water Quality Parameters
          ↓
   Mean Imputation
          ↓
   StandardScaler
          ↓
     MLPRegressor
      64 → 32
          ↓
   Prediction Score
          ↓
    0.5 Threshold
          ↓
Potable / Not Potable
```

### Model Configuration

- **Algorithm:** Multi-Layer Perceptron (`MLPRegressor`)
- **Hidden Layers:** 64 and 32 neurons
- **Activation Function:** ReLU
- **Classification Threshold:** 0.5
- **Preprocessing:** Mean Imputation + Standard Scaling

---

## Project Structure

```text
Water-Potability-Prediction-System/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── water_potability.csv
│
├── models/
│   └── water_potability_pipeline.joblib
│
├── notebooks/
│   └── Water_Potability.ipynb
│
└── src/
    └── train_model.py
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

---

## Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/MHK2005/Water-Potability-Prediction-System.git
cd Water-Potability-Prediction-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

The trained model is already included in the repository. To retrain it:

```bash
python src/train_model.py
```

### 4. Start the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL that can be opened in a web browser.

---
