"""
In this module we store prepare the sataset for machine learning experiments.
"""

import typing as t
import typing_extensions as te

import pandas as pd
from sklearn.model_selection import train_test_split


class DatasetReader(te.Protocol):
    def __call__(self) -> pd.DataFrame:
        ...


SplitName = te.Literal["train", "test"]


def get_dataset(reader: DatasetReader, splits: t.Iterable[SplitName]):
    df = reader()
    y = df["y"]
    X = df.drop(columns=["y"])
    X = X.astype(
        {k: str for k in get_categorical_variables_values_mapping().keys()})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )
    split_mapping = {"train": (X_train, y_train), "test": (X_test, y_test)}
    return {k: split_mapping[k] for k in splits}


def get_categorical_column_names() -> t.List[str]:
    return 'cp,restecg,slp,caa,thall'.split(",")


def get_binary_column_names() -> t.List[str]:
    return "sex,exng,fbs".split(",")


def get_numeric_column_names() -> t.List[str]:
    return 'age,trtbps,chol,thalachh,oldpeak'.split(",")


def get_column_names() -> t.List[str]:
    return "cp,restecg,slp,caa,thall,sex,exng,fbs,age,trtbps,chol,thalachh,oldpeak".split(",")


def get_categorical_variables_values_mapping() -> t.Dict[str, t.Sequence[str]]:
    return {
        "sex":("1","0"),
        "exng":("1","0"),
        "fbs": ("1","0"),
        "cp": ("1","0","2","3"),
        "restecg":("0","1","2"),
        "slp":("0","1","2"),
        "caa":("0","1","2","3"),
        "thall":("0","1","2","3")
    }
