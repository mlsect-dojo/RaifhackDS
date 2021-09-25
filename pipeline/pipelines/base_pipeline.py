from typing import List
import pandas as pd
from pipeline.transforms import BaseTransform
from pipeline.evaluators import BaseEvaluator


class BasePipeline:
    def __init__(self, transforms: List[BaseTransform], evaluator: BaseEvaluator):
        self.transforms = transforms
        self.evaluator = evaluator

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplemented

    def postprocess(self,
                    df: pd.DataFrame,
                    drop_cols_names: List[str],
                    drop_na_cols: bool = False,
                    drop_na_rows: bool = False) -> pd.DataFrame:
        raise NotImplemented

    def fit_predict(self, train_x: pd.DataFrame, train_y: pd.Series, test_x: pd.DataFrame, test_y: pd.Series) -> dict:
        raise NotImplemented
