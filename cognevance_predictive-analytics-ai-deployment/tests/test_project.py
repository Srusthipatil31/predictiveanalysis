"""Run:  python -m unittest discover -s tests -v   (train first with `python main.py`)"""
import unittest

import pandas as pd

from src.data import load_data
from src.features import CorrelationFilter, FeatureEngineer, OutlierClipper
from src.predict import Predictor


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = load_data()

    def test_dataset_shape(self):
        self.assertEqual(self.X.shape, (569, 30))
        self.assertTrue(set(self.y.unique()) <= {0, 1})

    def test_engineer_adds_features(self):
        out = FeatureEngineer().fit(self.X).transform(self.X)
        self.assertEqual(out.shape[1], 36)
        self.assertFalse(out.isna().any().any())

    def test_clipper_limits_range(self):
        c = OutlierClipper().fit(self.X)
        out = c.transform(self.X)
        self.assertTrue((out.max() <= c.hi_ + 1e-9).all())

    def test_correlation_filter_drops_columns(self):
        f = CorrelationFilter(0.95).fit(self.X)
        self.assertGreater(len(f.dropped_), 0)
        self.assertEqual(f.transform(self.X).shape[1], 30 - len(f.dropped_))


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = Predictor()

    def test_known_samples(self):
        ex = self.p.meta["examples"]
        self.assertEqual(self.p.predict(ex["malignant"])["prediction"], "Malignant")
        self.assertEqual(self.p.predict(ex["benign"])["prediction"], "Benign")

    def test_partial_input_uses_medians(self):
        r = self.p.predict({"mean radius": 12.0, "worst radius": 13.0})
        self.assertEqual(r["features_provided"], 2)
        self.assertEqual(r["features_filled_with_median"], 28)
        self.assertTrue(0 <= r["probability_malignant"] <= 1)

    def test_bad_input_rejected(self):
        with self.assertRaises(ValueError):
            self.p.predict({})
        with self.assertRaises(ValueError):
            self.p.predict({"mean radius": "abc"})
        with self.assertRaises(ValueError):
            self.p.predict({"mean radius": -5})


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app import app
        cls.c = app.test_client()

    def test_pages_load(self):
        for url in ["/", "/dashboard", "/health", "/features"]:
            self.assertEqual(self.c.get(url).status_code, 200, url)

    def test_predict_endpoint(self):
        ex = self.c.get("/features").get_json()["examples"]["malignant"]
        r = self.c.post("/predict", json=ex)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["prediction"], "Malignant")

    def test_predict_errors(self):
        self.assertEqual(self.c.post("/predict", json={}).status_code, 400)
        self.assertEqual(self.c.post("/predict", data="not json").status_code, 400)


if __name__ == "__main__":
    unittest.main()
