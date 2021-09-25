import pandas as pd


class BaseEvaluator:
    def __init__(self):
        raise NotImplemented

    def fit(self, x: pd.DataFrame, y: pd.Series):
        raise NotImplemented

    def predict(self, x: pd.DataFrame) -> pd.Series:
        raise NotImplemented
