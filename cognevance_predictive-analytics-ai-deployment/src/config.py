"""Paths and settings in one place."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "breast_cancer_wdbc.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
META_PATH = MODEL_DIR / "model_meta.json"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = OUTPUT_DIR / "figures"

TARGET = "diagnosis"          # "M" = malignant, "B" = benign in the CSV
POSITIVE_LABEL = "M"          # the class we want to catch (positive class = malignant)
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
