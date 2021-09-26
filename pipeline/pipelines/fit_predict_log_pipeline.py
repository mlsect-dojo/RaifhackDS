import pandas as pd
import numpy as np
from solutions.baseline.raifhack_ds.metrics import metrics_stat
from .fit_predict_pipeline import FitPreidctPipeline


class FitPreidctLogPipeline(FitPreidctPipeline):
    def fit_predict(self, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series) -> dict:
        """Fits model, measures metrics, returns scores"""
        self.evaluator.fit(train_x, train_y, test_x, test_y)
        predictions = self.evaluator.predict(test_x)
        print(predictions)
        metrics = metrics_stat(np.exp(predictions), np.exp(test_y))
        return metrics
