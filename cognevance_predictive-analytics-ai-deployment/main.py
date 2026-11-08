"""Train, tune, compare and evaluate every model, then save the best one.

    python main.py
"""
import os
os.environ.setdefault("PYTHONWARNINGS", "ignore::FutureWarning")   # hide scikit-learn deprecation chatter

from src import config as cfg
from src.train import run


def main():
    print("[1/2] Training and comparing models (tuning with 5-fold cross-validation) ...")
    comp, m = run()
    print("\n[2/2] Chosen model:", m["chosen_model"], "|", m["selected_by"])
    print("      best parameters :", m["best_params"])
    for k, v in m["test"].items():
        print(f"      test {k:10s} {v:.4f}")
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    print(f"      confusion matrix: {tp} malignant caught, {fn} missed, {fp} false alarms, {tn} benign correct")
    print(f"      features dropped as redundant: {len(m['dropped_correlated_features'])}")
    print(f"\nDone. See {cfg.OUTPUT_DIR}/ . Start the API + dashboard with:  python app.py")


if __name__ == "__main__":
    main()
