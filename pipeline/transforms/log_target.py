import pandas as pd
import numpy as np
from pipeline.transforms import BaseTransform
from solutions.baseline.raifhack_ds.settings import TARGET


class LogTarget(BaseTransform):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        col = TARGET
        return df[col].apply(np.log)
