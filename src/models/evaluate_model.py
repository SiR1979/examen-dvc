import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
METRICS = ROOT / "metrics"

X_test = pd.read_csv(DATA / "X_test_scaled.csv")
y_test = pd.read_csv(DATA / "y_test.csv").squeeze("columns")
model = joblib.load(ROOT / "models" / "model.pkl")

predictions = model.predict(X_test)

results = pd.DataFrame({
    "silica_concentrate": y_test,
    "prediction": predictions,
})
results.to_csv(ROOT / "data" / "predictions.csv", index=False)

scores = {
    "mse": float(mean_squared_error(y_test, predictions)),
    "r2": float(r2_score(y_test, predictions)),
}

METRICS.mkdir(parents=True, exist_ok=True)
with (METRICS / "scores.json").open("w", encoding="utf-8") as file:
    json.dump(scores, file, indent=2, allow_nan=False)
    file.write("\n")

print(f"Prediction rows: {len(results)}")
print(json.dumps(scores, indent=2))
