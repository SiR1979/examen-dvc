from pathlib import Path

import joblib
import pandas as pd
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

X_train = pd.read_csv(DATA / "X_train_scaled.csv")
y_train = pd.read_csv(DATA / "y_train.csv").squeeze("columns")
best_params = joblib.load(MODELS / "best_params.pkl")

model = SVR(**best_params)
model.fit(X_train, y_train)

joblib.dump(model, MODELS / "model.pkl")

print(f"Training rows: {len(X_train)}")
print("Model saved: models/model.pkl")
