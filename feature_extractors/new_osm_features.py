import pandas as pd
import numpy as np


def make_product(data: pd.DataFrame, feature1: str, feature2: str) -> pd.Series:
    """Вычисление произведения расстояния до метро и transport_stop(или других фичей, но это кажется круто)
        example: make_product(df_with_new_features, 'subway_dist', 'transport_stop_closest_dist')
    """
    return data[[feature1, feature2]].product(axis=1)


def make_sum(data: pd.DataFrame, feature1: str, feature2: str) -> pd.Series:
    return data[[feature1, feature2]].sum(axis=1)


def make_sub(data: pd.DataFrame, feature1: str, feature2: str) -> pd.Series:
    return data[[feature1, feature2]].sub(axis=1)


def make_div(data: pd.DataFrame, feature1: str, feature2: str) -> pd.Series:
    return data[[feature1, feature2]].div(axis=1)



