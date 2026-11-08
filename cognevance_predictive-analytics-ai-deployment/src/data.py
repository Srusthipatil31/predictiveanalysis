"""Load the dataset (and create the CSV the first time).

Dataset: Breast Cancer Wisconsin (Diagnostic) - 569 patients, 30 numeric features computed from
digitised images of a fine-needle aspirate of a breast mass (UCI ML repository, bundled with scikit-learn).

To use your own data instead: replace data/breast_cancer_wdbc.csv with a CSV that has numeric feature
columns plus a `diagnosis` column containing M / B.
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src import config as cfg


def export_csv():
    raw = load_breast_cancer(as_frame=True)
    df = raw.data.copy()
    # sklearn encodes 0 = malignant, 1 = benign -> convert to readable M / B labels
    df[cfg.TARGET] = raw.target.map({0: "M", 1: "B"})
    cfg.DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(cfg.DATA_PATH, index=False)
    return df


def load_data():
    """Return (X, y) with y = 1 for malignant, 0 for benign."""
    df = pd.read_csv(cfg.DATA_PATH) if cfg.DATA_PATH.exists() else export_csv()
    y = (df[cfg.TARGET] == cfg.POSITIVE_LABEL).astype(int)
    X = df.drop(columns=[cfg.TARGET])
    return X, y


if __name__ == "__main__":
    X, y = load_data()
    print(f"{X.shape[0]} patients, {X.shape[1]} features, {y.mean():.1%} malignant -> {cfg.DATA_PATH}")
