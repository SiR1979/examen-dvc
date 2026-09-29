from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

X_train = pd.read_csv(DATA / "X_train.csv")
y_train = pd.read_csv(DATA / "y_train.csv").squeeze("columns")

# Fit scaling separately within each cross-validation fold.
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(kernel="rbf")),
])

param_grid = {
    "svr__C": [1, 10, 100],
    "svr__epsilon": [0.01, 0.1],
    "svr__gamma": ["scale", 0.1],
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=1,
    error_score="raise",
)
search.fit(X_train, y_train)

best_params = {
    key.removeprefix("svr__"): value
    for key, value in search.best_params_.items()
}
best_params["kernel"] = "rbf"

MODELS.mkdir(parents=True, exist_ok=True)
joblib.dump(best_params, MODELS / "best_params.pkl")

print(f"Best parameters: {best_params}")
print(f"Best cross-validation MSE: {-search.best_score_:.6f}")
