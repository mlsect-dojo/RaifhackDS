from typing import List
import pandas as pd
import numpy as np
from pipeline.transforms import BaseTransform
from solutions.baseline.raifhack_ds.settings import NUM_FEATURES


class LogNumericFeatures(BaseTransform):
    def __call__(self, df: pd.DataFrame) -> List[pd.Series]:
        features = NUM_FEATURES

        na_cols = df.isna().sum(axis=0) != 0
        na_cols = na_cols[na_cols].index.tolist()

        for val in ['lat', 'lng'] + na_cols:
            if val in features:
                features.remove(val)

        serieses = [df[col] for col in df[features].apply(np.log)]
        return serieses
