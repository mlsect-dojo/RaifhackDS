import pandas as pd
from sklearn.linear_model import LinearRegression
from .base_evaluator import BaseEvaluator


class LinregEvaluator(BaseEvaluator):
    def __init__(self, model_params: dict):
        self.model = LinearRegression(**model_params)

    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs):
        self.model.fit(x, y)

    def predict(self, x: pd.DataFrame, **kwargs) -> pd.Series:
        return self.model.predict(x)
