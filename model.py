"""
Here we code what our model is. It may include all of feature engineering.
"""
import typing as t
from functools import partial

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import StandardScaler,KBinsDiscretizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

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
        "logistic-regressor": LogisticRegression,
        "categorical-encoder": CategoricalEncoder,
        "standard-scaler": StandardScaler,
        "discretizer": Discretizer,
        "average-thalachh": AverageThalachh,
        "bagging": BaggClassifier,
        "RandomForestModel": RandomForestModel,
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
            partial(
                OneHotEncoder,
                drop="if_binary",
                sparse=not self.force_dense_array,
            )
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
        X = self._column_transformer.transform(X)
        return X

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

class AverageThalachh(BaseEstimator, RegressorMixin):
    def fit(self, X, y):
        df = pd.DataFrame({"thalachh": X["thalachh"], "y": y})
        self.meanThalachh_ = df.groupby("y").mean().mean().thalachh
        return self

    def predict(self, X):
        """Predicts the mode computed in the fit method."""

        def validate_threshold(x):
            if x > self.meanThalachh_:
                return 1
            return 0

        y_pred = X["thalachh"].apply(validate_threshold)
        return y_pred

class BaggClassifier:

    def fit(self, X, y):
        #self._model = BaggingClassifier(base_estimator=LogisticRegression(),n_estimators=1000,max_features=0.8).fit(X,y)
        self._model = BaggingClassifier(base_estimator=GradientBoostingClassifier(subsample=0.8),n_estimators=300).fit(X,y)
        return self

    def predict(self,X):
        return self._model.predict(X)
    
class RandomForestModel:

    def fit(self, X, y):
        self._model = RandomForestClassifier(n_estimators=800,max_depth=8,max_features='log2').fit(X,y)
        return self

    def predict(self,X):
        return self._model.predict(X)