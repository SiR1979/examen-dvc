from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data" / "processed"

data = pd.read_csv(ROOT / "data" / "raw" / "raw.csv")

features = [
    "ave_flot_air_flow",
    "ave_flot_level",
    "iron_feed",
    "starch_flow",
    "amina_flow",
    "ore_pulp_flow",
    "ore_pulp_pH",
    "ore_pulp_density",
]

X = data[features]
y = data[["silica_concentrate"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

OUTPUT.mkdir(parents=True, exist_ok=True)

datasets = {
    "X_train": X_train,
    "X_test": X_test,
    "y_train": y_train,
    "y_test": y_test,
}

for name, dataset in datasets.items():
    dataset.to_csv(OUTPUT / f"{name}.csv", index=False)
    print(f"{name}: {dataset.shape}")
