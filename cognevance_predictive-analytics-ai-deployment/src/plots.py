"""All charts used in the report and dashboard."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

from src import config as cfg

BLUE, ORANGE, RED = "#2f6fb7", "#e07b39", "#c0392b"


def _save(fig, name):
    cfg.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(cfg.FIG_DIR / name, dpi=130)
    plt.close(fig)


def class_balance(y):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    bars = ax.bar(["Benign", "Malignant"], counts, color=[BLUE, RED])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 4, f"{c} ({c / len(y):.0%})", ha="center")
    ax.set_ylabel("Patients"); ax.set_title("Class balance"); ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "01_class_balance.png")


def correlation_heatmap(X):
    cols = [c for c in X.columns if c.startswith("mean ")]
    corr = X[cols].corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_yticks(range(len(cols)))
    short = [c.replace("mean ", "") for c in cols]
    ax.set_xticklabels(short, rotation=75, ha="right"); ax.set_yticklabels(short)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Correlation between the 10 'mean' features")
    _save(fig, "02_correlation_heatmap.png")


def feature_distributions(X, y):
    feats = ["worst concave points", "worst perimeter", "mean concave points", "worst radius",
             "mean texture", "mean smoothness"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, f in zip(axes.ravel(), feats):
        ax.hist(X.loc[y == 0, f], bins=25, alpha=.7, color=BLUE, label="Benign")
        ax.hist(X.loc[y == 1, f], bins=25, alpha=.7, color=RED, label="Malignant")
        ax.set_title(f, fontsize=10); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].legend()
    fig.suptitle("Feature distributions by diagnosis")
    _save(fig, "03_feature_distributions.png")


def model_comparison(df):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(df)); w = 0.27
    ax.bar(x - w, df["cv_roc_auc"], w, label="CV ROC-AUC (train)", color=BLUE)
    ax.bar(x, df["test_roc_auc"], w, label="Test ROC-AUC", color=ORANGE)
    ax.bar(x + w, df["test_recall"], w, label="Test recall (malignant)", color=RED)
    ax.set_xticks(x); ax.set_xticklabels(df["model"], rotation=18, ha="right")
    ax.set_ylim(0.85, 1.01); ax.set_ylabel("Score (axis starts at 0.85)")
    ax.set_title("Model comparison", pad=28)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "04_model_comparison.png")


def roc_curves(models, X_test, y_test):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for name, m in models.items():
        fpr, tpr, _ = roc_curve(y_test, m.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=name)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curves (test set)"); ax.legend(fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "05_roc_curves.png")


def confusion(model, X_test, y_test, name):
    fig, ax = plt.subplots(figsize=(4.8, 4.5))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=["Benign", "Malignant"],
                                          cmap="Blues", ax=ax, colorbar=False)
    ax.set_title(f"Confusion matrix - {name}")
    _save(fig, "06_confusion_matrix.png")


def importance(names, means, stds, model_name):
    order = np.argsort(means)[-12:]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.barh(np.array(names)[order], np.array(means)[order], xerr=np.array(stds)[order], color=BLUE)
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title(f"Top feature importances - {model_name}")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "07_feature_importance.png")
