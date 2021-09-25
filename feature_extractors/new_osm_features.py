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


def min_distance_to_transport(data: pd.DataFrame, feature1: str, feature2: str) -> pd.Series:
    """
        находим минимальный путь до ближайшего транспорта('osm_subway_closest_dist' and 'osm_transport_stop_closest_dist')
    """
    return data[[feature1, feature2]].min(axis=1)


def min_distance_to_city(data: pd.DataFrame, feature1: str, feature2: str) -> pd.Series:
    """
    находим минимальный путь до ближайшего города('osm_city_closest_dist' + 'osm_transport_stop_closest_dist')
    """
    return data[[feature1, feature2]].sum(axis=1)


FUNCTIONS_DICT = {'*': make_product, '+': make_sum,
                  '-': make_sub, '/': make_div, 'min_dist': min_distance_to_transport,
                  'nearest_path_city': min_distance_to_city}
