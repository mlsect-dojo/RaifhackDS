import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from solutions.baseline.raifhack_ds.metrics import deviation_metric
from .base_evaluator import BaseEvaluator


class CatboostEvaluator(BaseEvaluator):
    def __init__(self, model_params: dict):

        class RaifCatboostMetric(object):
            def get_final_error(self, error, weight):
                return error

            def is_max_optimal(self):
                # the larger metric value the better
                return True

            def evaluate(self, approxes, target, weight):
                assert len(approxes) == 1
                assert len(target) == len(approxes[0])
                preds = np.array(approxes[0])
                target = np.array(target)
                return np.corrcoef(target, preds)[0, 1], 0

        its = 1000
        self.model = CatBoostRegressor(iterations=its,
                                       verbose=int(its/30),
                                       depth=7,
                                       loss_function='MAPE',
                                       random_seed=42,
                                       use_best_model=True,
                                       # eval_metric=deviation_metric,
                                       **model_params)

    def fit(self, x: pd.DataFrame, y: pd.Series, x_test: pd.DataFrame, y_test: pd.Series, **kwargs):
        self.model.fit(x, y, eval_set=(x_test, y_test))

    def predict(self, x: pd.DataFrame, **kwargs) -> pd.Series:
        return self.model.predict(x)
