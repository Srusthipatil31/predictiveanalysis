"""Candidate models and their (small) hyper-parameter grids."""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src import config as cfg
from src.features import CorrelationFilter, FeatureEngineer, OutlierClipper


def make_pipeline(classifier) -> Pipeline:
    """impute -> clip outliers -> engineer features -> drop correlated -> scale -> classifier"""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median").set_output(transform="pandas")),
        ("clip", OutlierClipper()),
        ("engineer", FeatureEngineer()),
        ("decorrelate", CorrelationFilter(0.95)),
        ("scale", StandardScaler().set_output(transform="pandas")),
        ("clf", classifier),
    ])


def candidates():
    """name -> (pipeline, parameter grid for GridSearchCV)"""
    rs = cfg.RANDOM_STATE
    return {
        "Logistic Regression": (make_pipeline(LogisticRegression(max_iter=5000, random_state=rs)),
                                {"clf__C": [0.1, 1, 10]}),
        "SVM (RBF)": (make_pipeline(SVC(probability=True, random_state=rs)),
                      {"clf__C": [1, 10], "clf__gamma": ["scale", 0.01]}),
        "K-Nearest Neighbours": (make_pipeline(KNeighborsClassifier()),
                                 {"clf__n_neighbors": [5, 9, 15]}),
        "Random Forest": (make_pipeline(RandomForestClassifier(n_estimators=200, random_state=rs)),
                          {"clf__max_depth": [None, 6], "clf__min_samples_leaf": [1, 3]}),
        "Gradient Boosting": (make_pipeline(GradientBoostingClassifier(random_state=rs)),
                              {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1],
                               "clf__max_depth": [2, 3]}),
        "Neural Network (MLP)": (make_pipeline(MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=3000,
                                                             random_state=rs)),
                                 {"clf__alpha": [0.001, 0.01, 0.1]}),
    }
