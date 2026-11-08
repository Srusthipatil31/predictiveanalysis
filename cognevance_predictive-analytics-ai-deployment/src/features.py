"""Custom preprocessing / feature-engineering steps (they live inside the model pipeline, so the
API applies exactly the same steps to new patients as training did)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

EPS = 1e-9


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Winsorise: clip each feature to its [lower, upper] training quantiles."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.lo_ = X.quantile(self.lower)
        self.hi_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X.clip(self.lo_, self.hi_, axis=1)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds domain-inspired ratio / interaction features."""

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        X["area_perimeter_ratio"] = X["mean area"] / (X["mean perimeter"] + EPS)
        X["worst_mean_radius_ratio"] = X["worst radius"] / (X["mean radius"] + EPS)
        X["worst_mean_concave_ratio"] = X["worst concave points"] / (X["mean concave points"] + EPS)
        X["radius_variability"] = X["radius error"] / (X["mean radius"] + EPS)
        X["texture_x_smoothness"] = X["mean texture"] * X["mean smoothness"]
        X["shape_irregularity"] = X["mean concavity"] + X["mean compactness"]
        return X


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of every pair of features whose |correlation| exceeds `threshold` (learned on training data)."""

    def __init__(self, threshold=0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        self.dropped_ = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        self.kept_ = [c for c in X.columns if c not in self.dropped_]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.kept_]
