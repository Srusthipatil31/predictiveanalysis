"""Train, tune, compare, evaluate and save the models."""
import json

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split

from src import config as cfg
from src import plots
from src.data import load_data
from src.models import candidates


def run(verbose=True):
    cfg.MODEL_DIR.mkdir(exist_ok=True)
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, stratify=y, random_state=cfg.RANDOM_STATE)
    if verbose:
        print(f"      {len(X)} patients | {X.shape[1]} features | malignant {y.mean():.1%} | "
              f"train {len(X_train)} / test {len(X_test)}")

    # ---- exploratory charts ---------------------------------------------------
    plots.class_balance(y)
    plots.correlation_heatmap(X)
    plots.feature_distributions(X, y)

    # ---- tune + cross-validate every candidate on the TRAINING set only ----------
    cv = StratifiedKFold(n_splits=cfg.CV_FOLDS, shuffle=True, random_state=cfg.RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    tuned, rows = {}, []
    for name, (pipe, grid) in candidates().items():
        search = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
        best = search.best_estimator_
        cvres = cross_validate(best, X_train, y_train, cv=cv, scoring=scoring)
        pred = best.predict(X_test)
        proba = best.predict_proba(X_test)[:, 1]
        tuned[name] = best
        rows.append({
            "model": name,
            "best_params": json.dumps({k.replace("clf__", ""): v for k, v in search.best_params_.items()}),
            "cv_accuracy": cvres["test_accuracy"].mean(), "cv_recall": cvres["test_recall"].mean(),
            "cv_roc_auc": cvres["test_roc_auc"].mean(),
            "test_accuracy": accuracy_score(y_test, pred),
            "test_precision": precision_score(y_test, pred),
            "test_recall": recall_score(y_test, pred),
            "test_f1": f1_score(y_test, pred),
            "test_roc_auc": roc_auc_score(y_test, proba),
        })
        if verbose:
            r = rows[-1]
            print(f"      {name:22s} CV AUC {r['cv_roc_auc']:.3f} | test acc {r['test_accuracy']:.3f} "
                  f"recall {r['test_recall']:.3f} AUC {r['test_roc_auc']:.3f}")

    comp = pd.DataFrame(rows).round(4)
    # model selection uses cross-validation on the training set only (never the test set)
    comp = comp.sort_values(["cv_roc_auc", "cv_recall"], ascending=False).reset_index(drop=True)
    comp.to_csv(cfg.OUTPUT_DIR / "model_comparison.csv", index=False)
    best_name = comp.loc[0, "model"]
    best = tuned[best_name]

    # ---- charts + explanation for the chosen model -------------------------------
    plots.model_comparison(comp)
    plots.roc_curves(tuned, X_test, y_test)
    plots.confusion(best, X_test, y_test, best_name)
    imp = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=15,
                                 random_state=cfg.RANDOM_STATE)
    plots.importance(list(X.columns), imp.importances_mean, imp.importances_std, best_name)
    imp_df = pd.DataFrame({"feature": X.columns, "importance": imp.importances_mean}).sort_values(
        "importance", ascending=False)
    imp_df.round(5).to_csv(cfg.OUTPUT_DIR / "feature_importance.csv", index=False)

    # ---- save model + metadata for the API ---------------------------------------
    joblib.dump(best, cfg.MODEL_PATH)
    pred = best.predict(X_test)
    report = classification_report(y_test, pred, target_names=["benign", "malignant"], output_dict=True)
    row = comp.loc[0]
    metrics = {
        "chosen_model": best_name,
        "selected_by": "highest 5-fold cross-validated ROC-AUC on the training set",
        "best_params": json.loads(row["best_params"]),
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "test": {k: float(row[f"test_{k}"]) for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]},
        "confusion_matrix": pd.crosstab(y_test, pred).values.tolist(),
        "dropped_correlated_features": list(best.named_steps["decorrelate"].dropped_),
        "n_features_after_engineering": len(best.named_steps["decorrelate"].kept_),
        "classification_report": report,
    }
    (cfg.OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    full = pd.concat([X, y.rename("y")], axis=1)
    meta = {
        "feature_names": list(X.columns),
        "medians": X.median().round(5).to_dict(),
        "min": X.min().round(5).to_dict(), "max": X.max().round(5).to_dict(),
        "examples": {"malignant": full[full.y == 1].iloc[0].drop("y").round(5).to_dict(),
                     "benign": full[full.y == 0].iloc[0].drop("y").round(5).to_dict()},
        "model_name": best_name,
    }
    cfg.META_PATH.write_text(json.dumps(meta, indent=2))
    return comp, metrics
