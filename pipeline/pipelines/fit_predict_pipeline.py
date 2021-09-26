from typing import List
import pandas as pd
from solutions.baseline.raifhack_ds.metrics import metrics_stat
from pipeline.transforms import BaseTransform
from pipeline.evaluators import BaseEvaluator
from .base_pipeline import BasePipeline


class FitPreidctPipeline(BasePipeline):
    def __init__(self, transforms: List[BaseTransform], evaluator: BaseEvaluator):
        self.transforms = transforms
        self.evaluator = evaluator

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies each transform, assigns result of it as new column(s)"""
        for transform in self.transforms:
            cols = transform(df)

            if isinstance(cols, pd.Series):
                df.loc[:, transform.name] = cols

            elif isinstance(cols, list):
                for i, col in enumerate(cols):

                    if not isinstance(col, pd.Series):
                        raise ValueError('Invalid Transform return type:', type(cols))

                    df.loc[:, f'{transform.name}_{i}'] = col

            else:
                raise ValueError('Invalid Transform return type:', type(cols))
        return df

    def postprocess(self,
                    df: pd.DataFrame,
                    drop_cols_names: List[str],
                    drop_na_cols: bool = False,
                    drop_na_rows: bool = False) -> pd.DataFrame:
        df = df.drop(drop_cols_names, axis=1)

        if drop_na_rows:
            df = df[df.isna().sum(axis=1) == 0]

        if drop_na_cols:
            not_na_cols = df.isna().sum(axis=0) == 0
            df = df[not_na_cols[not_na_cols].index.tolist()]

        return df

    def fit_predict(self, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series) -> dict:
        """Fits model, measures metrics, returns scores"""
        self.evaluator.fit(train_x, train_y, test_x, test_y)
        predictions = self.evaluator.predict(test_x)
        print(predictions)
        metrics = metrics_stat(predictions, test_y)
        return metrics
