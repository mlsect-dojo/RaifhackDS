import pandas as pd


class BaseEvaluator:
    def __init__(self):
        raise NotImplemented

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        raise NotImplemented

    def predict(self, x: pd.DataFrame, **kwargs) -> pd.Series:
        raise NotImplemented
