"""
Here we code what our model is. It may include all of feature engineering.
"""
import typing as t
from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler,KBinsDiscretizer
from sklearn.ensemble import BaggingRegressor

import data


EstimatorConfig = t.List[t.Dict[str, t.Any]]

def build_estimator(config: EstimatorConfig):
    estimator_mapping = get_estimator_mapping()
    steps = []
    for step in config:
        name = step["name"]
        params = step["params"]
        estimator = estimator_mapping[name](**params)
        steps.append((name, estimator))
    model = Pipeline(steps)
    return model


def get_estimator_mapping():
    return {
        "linear-regressor": LinearRegression,
        "logistic-regressor": LogisticRegression,
        "categorical-encoder": CategoricalEncoder,
        "standard-scaler": StandardScaler,
        "discretizer": Discretizer,
    }
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, *, one_hot: bool = False, force_dense_array: bool = False):
        self.one_hot = one_hot
        self.force_dense_array = force_dense_array
        self.categorical_column_names = (
            data.get_binary_column_names() + data.get_categorical_column_names()
        )
        mapping = data.get_categorical_variables_values_mapping()
        self.categories = [mapping[k] for k in self.categorical_column_names]

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        encoder_cls = (
            partial(OneHotEncoder, drop="first", sparse=not self.force_dense_array)
            if self.one_hot
            else OrdinalEncoder
        )
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "encoder",
                    encoder_cls(
                        categories=self.categories,
                    ),
                    self.categorical_column_names,
                ),
                ("pass-numeric", "passthrough", data.get_numeric_column_names()),
            ],
            remainder="drop",
        )
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        return self._column_transformer.transform(X)

class Discretizer(BaseEstimator, TransformerMixin):
    def __init__(self, *, bins_per_column: t.Mapping[str, int], strategy: str):
        self.bins_per_column = bins_per_column
        self.strategy = strategy

    def fit(self, X, y):
        X = X.copy()
        self.n_features_in_ = X.shape[1]
        self.original_column_order_ = X.columns.tolist()
        self.columns_, n_bins = zip(*self.bins_per_column.items())
        self.new_column_order_ = self.columns_ + tuple(
            name
            for name in self.original_column_order_
            if name not in self.bins_per_column
        )
        self._column_transformer = ColumnTransformer(
            transformers=[
                (
                    "encoder",
                    KBinsDiscretizer(
                        n_bins=n_bins, encode="ordinal", strategy=self.strategy
                    ),
                    self.columns_,
                ),
            ],
            remainder="passthrough",
        )
        self._column_transformer = self._column_transformer.fit(X, y=y)
        return self

    def transform(self, X):
        X = pd.DataFrame(
            self._column_transformer.transform(X), columns=self.new_column_order_
        )
        return X