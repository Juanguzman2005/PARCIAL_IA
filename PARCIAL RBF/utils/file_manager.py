
import os, json, pickle, numpy as np, pandas as pd
from pathlib import Path
RESULTS_DIR = "resultados"
MODEL_DIR = "modelos"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def save_model_local(centers, weights, scaler, config):
    base = Path(MODEL_DIR) / f"rbf_model_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "centers.npy", centers)
    np.save(base / "weights.npy", weights)
    with open(base / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(base / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return str(base)

def load_model_local(folder_path):
    folder = Path(folder_path)
    centers = np.load(folder / "centers.npy")
    weights = np.load(folder / "weights.npy")
    scaler = None
    try:
        with open(folder / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
    except Exception:
        scaler = None
    return centers, weights, scaler

def save_simulation(base_name, Yr, Yd, metrics):
    base = Path(RESULTS_DIR) / f"{base_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    base.mkdir(parents=True, exist_ok=True)
    np.save(base / "Yr.npy", Yr)
    np.save(base / "Yd.npy", Yd)
    with open(base / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return str(base)

def save_csv(path, df):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
