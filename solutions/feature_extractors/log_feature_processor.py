import pandas as pd
import numpy as np


def get_log_feature_processing(df: pd.DataFrame) -> pd.DataFrame:
    strange_features = ['lat', 'lng', 'price_type']
    log_df = df.select_dtypes(exclude=object).drop(strange_features, 1).apply(np.log1p)
    for feature in log_df:
        df[feature] = log_df[feature]
    return df