import os
os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"

from pathlib import Path
import tempfile
import zipfile
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from autogluon.tabular import TabularPredictor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


MODEL_PATH = "Trained regression model.zip"
INPUT_FILE = "Validation data-3.xlsx"
OUT_FILE = "regression_predictions_from_saved_model.xlsx"

COLUMN_MAPPING = {
    "Diffusion coefficient": "f0",
    "CN(FDCA–H)": "f1",
    "CN(FDCA–O)": "f2",
    "MEPS minimal": "f3",
    "MEPS maximal": "f4",
    "Polarity difference": "f5",
    "Solvation energy": "f6",
    "δH": "f7",
    "δT": "f8",
}


def find_predictor_dir(root: Path) -> Path:
    root = root.resolve()
    if (root / "predictor.pkl").exists():
        return root

    candidates = list(root.rglob("predictor.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"predictor.pkl was not found under {root}. "
            "Please provide a valid model folder or zip file."
        )

    dirs = [p.parent for p in candidates]
    dirs_sorted = sorted(dirs, key=lambda d: d.name)
    return dirs_sorted[-1]


def prepare_model_dir(model_path: str):
    p = Path(model_path).resolve()
    temp_dir = None

    if p.is_file() and p.suffix.lower() == ".zip":
        temp_dir = Path(tempfile.mkdtemp(prefix="tabpfnmodels_"))
        with zipfile.ZipFile(p, "r") as zf:
            zf.extractall(temp_dir)
        root = temp_dir
    elif p.is_dir():
        root = p
    else:
        raise FileNotFoundError(f"MODEL_PATH does not exist: {p}")

    predictor_dir = find_predictor_dir(root)
    return predictor_dir, temp_dir


predictor_dir, temp_dir = prepare_model_dir(MODEL_PATH)

print("====================================")
print("[Model loading] MODEL_PATH:", MODEL_PATH)
print("[Model loading] predictor_dir:", predictor_dir)
print("====================================\n")

predictor = TabularPredictor.load(predictor_dir, require_version_match=False)

feature_cols = predictor.features()
print("[Model information]")
print("Number of feature columns:", len(feature_cols))
print("Required feature columns:", feature_cols)
print("====================================\n")

df = pd.read_excel(INPUT_FILE, header=0)

print("[Input data]")
print("Original columns:", list(df.columns))
print("====================================\n")

missing_raw_cols = [raw_col for raw_col in COLUMN_MAPPING.keys() if raw_col not in df.columns]
if missing_raw_cols:
    raise ValueError(
        "The input file is missing the following raw feature columns:\n"
        + "\n".join(missing_raw_cols)
    )

mapped_feature_cols = list(COLUMN_MAPPING.values())
missing_model_cols = [c for c in feature_cols if c not in mapped_feature_cols]
if missing_model_cols:
    raise ValueError(
        "COLUMN_MAPPING does not cover the following model-required columns:\n"
        + "\n".join(missing_model_cols)
    )

if "Y" not in df.columns:
    raise ValueError("The input file must contain a column named 'Y' as the true target values.")

df_model = df.rename(columns=COLUMN_MAPPING).copy()
X_df = df_model[feature_cols].copy()

print("[Mapped model input columns]")
print(X_df.columns.tolist())
print("====================================\n")

y_pred = predictor.predict(X_df)
y_true = df["Y"].to_numpy()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

df_out = df.copy()
df_out["Y_pred"] = y_pred.values
df_out["Residual"] = df_out["Y"] - df_out["Y_pred"]

df_out.to_excel(OUT_FILE, index=False)

print("[Evaluation metrics]")
print(f"MSE: {mse:.6f}")
print(f"MAE: {mae:.6f}")
print(f"R2 : {r2:.6f}")
print("====================================\n")

print("[Prediction completed]")
print("Output file:", OUT_FILE)
print(df_out[["Y", "Y_pred", "Residual"]].head())

plt.figure(figsize=(6.6, 5.6))

plt.scatter(
    y_true, y_pred,
    s=70, alpha=0.85,
    edgecolors="black", linewidths=0.6
)

ymin = float(min(np.min(y_true), np.min(y_pred)))
ymax = float(max(np.max(y_true), np.max(y_pred)))
pad = 0.05 * (ymax - ymin + 1e-12)
lo, hi = ymin - pad, ymax + pad

plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4)
plt.xlim(lo, hi)
plt.ylim(lo, hi)
plt.gca().set_aspect("equal", adjustable="box")

plt.xlabel("True Y", fontsize=12)
plt.ylabel("Predicted Y", fontsize=12)
plt.title("Validation: True vs Predicted", fontsize=13)

metrics_text = (
    f"R2 = {r2:.4f}\n"
    f"MAE = {mae:.4f}\n"
    f"MSE = {mse:.4f}\n"
    f"n = {len(y_true)}"
)

plt.text(
    0.04, 0.96, metrics_text,
    transform=plt.gca().transAxes,
    va="top", ha="left",
    fontsize=11,
    bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="black", linewidth=0.8, alpha=0.9)
)

plt.grid(alpha=0.25, linestyle="--")
plt.tight_layout()
plt.show()

if temp_dir is not None:
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass