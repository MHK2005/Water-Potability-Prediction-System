from pathlib import Path
import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "water_potability.csv"
MODEL_PATH = ROOT / "models" / "water_potability_pipeline.joblib"

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
]
TARGET = "Potability"

def main():
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            random_state=42,
            max_iter=1000
        )),
    ])

    pipeline.fit(X_train, y_train)

    raw_scores = pipeline.predict(X_test)
    predictions = (raw_scores >= 0.5).astype(int)

    print(f"Test accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(classification_report(y_test, predictions, digits=4))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
