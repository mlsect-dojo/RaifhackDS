import pandas as pd
import numpy as np
from new_osm_features import FUNCTIONS_DICT
import random


LIMIT = 300


class FeatureGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.list_of_names = [name for name in df.select_dtypes(include=[int, float]).dropna(axis=1)]

    def __call__(self) -> pd.DataFrame:
        new_df = self.df.copy()
        cnt = 0
        for i in range(len(self.list_of_names)):
            for j in range(i, len(self.list_of_names)):
                cnt += 1
                if cnt > 300:
                    return new_df
                operation = random.choise(random)
                new_df[self.list_of_names[i] + f"_{operation}_" + self.list_of_names[j]] = \
                    FUNCTIONS_DICT[operation](new_df, self.list_of_names[i], self.list_of_names[j])
        return new_df


