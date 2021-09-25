from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from solutions.baseline.raifhack_ds.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET
from solutions.baseline.raifhack_ds.utils import PriceTypeEnum
from solutions.baseline.raifhack_ds.features import prepare_categorical
from solutions.baseline.raifhack_ds.metrics import metrics_stat


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = prepare_categorical(df)
    not_na_cols = df.isna().sum(axis=0) == 0
    df = df[not_na_cols[not_na_cols].index.tolist()]
    return df


def split_xy(df: pd.DataFrame, offer_type) -> Tuple[pd.DataFrame, pd.DataFrame]:
    features = [
        'osm_amenity_points_in_0.001',
        'osm_amenity_points_in_0.005',
        'osm_amenity_points_in_0.0075',
        'osm_amenity_points_in_0.01',

        'osm_building_points_in_0.001',
        'osm_building_points_in_0.005',
        'osm_building_points_in_0.0075',
        'osm_building_points_in_0.01',

        'osm_catering_points_in_0.001',
        'osm_catering_points_in_0.005',
        'osm_catering_points_in_0.0075',
        'osm_catering_points_in_0.01',

        'osm_city_closest_dist',
        'osm_city_nearest_population',

        'osm_crossing_closest_dist',
        'osm_crossing_points_in_0.001',
        'osm_crossing_points_in_0.005',
        'osm_crossing_points_in_0.0075',
        'osm_crossing_points_in_0.01',

        'osm_culture_points_in_0.001',
        'osm_culture_points_in_0.005',
        'osm_culture_points_in_0.0075',
        'osm_culture_points_in_0.01',

        'osm_finance_points_in_0.001',
        'osm_finance_points_in_0.005',
        'osm_finance_points_in_0.0075',
        'osm_finance_points_in_0.01',

        'osm_healthcare_points_in_0.005',
        'osm_healthcare_points_in_0.0075',
        'osm_healthcare_points_in_0.01',

        'osm_historic_points_in_0.005',
        'osm_historic_points_in_0.0075',
        'osm_historic_points_in_0.01',

        'osm_hotels_points_in_0.005',
        'osm_hotels_points_in_0.0075',
        'osm_hotels_points_in_0.01',

        'osm_leisure_points_in_0.005',
        'osm_leisure_points_in_0.0075',
        'osm_leisure_points_in_0.01',

        'osm_offices_points_in_0.001',
        'osm_offices_points_in_0.005',
        'osm_offices_points_in_0.0075',
        'osm_offices_points_in_0.01',

        'osm_shops_points_in_0.001',
        'osm_shops_points_in_0.005',
        'osm_shops_points_in_0.0075',
        'osm_shops_points_in_0.01',

        'osm_subway_closest_dist',

        'osm_train_stop_closest_dist',
        'osm_train_stop_points_in_0.005',
        'osm_train_stop_points_in_0.0075',
        'osm_train_stop_points_in_0.01',

        'osm_transport_stop_closest_dist',
        'osm_transport_stop_points_in_0.005',
        'osm_transport_stop_points_in_0.0075',
        'osm_transport_stop_points_in_0.01',

        'reform_count_of_houses_1000',
        'reform_count_of_houses_500',

        'reform_house_population_1000',
        'reform_house_population_500',
        'reform_mean_floor_count_1000',
        'reform_mean_floor_count_500',
        'reform_mean_year_building_1000',
        'reform_mean_year_building_500',

        'total_square',
        'price_type',

        'per_square_meter_price'
    ]
    features = list(set(features).intersection(set(df.columns)))
    drop_x_cols = ['price_type', 'per_square_meter_price']
    df = df[df.price_type == offer_type][features]
    y = df[TARGET]
    x = df.drop(drop_x_cols, axis=1)
    return x, y


def find_corr_coefficient(y_manual, predictions):
    """Вычисление корректирующего коэффициента
    :param X_manual: pd.DataFrame с ручными оценками
    :param y_manual: pd.Series - цены ручника
    """
    deviation = ((y_manual - predictions) / predictions).median()
    corr_coef = deviation
    return corr_coef


if __name__ == '__main__':
    train_df = pd.read_csv(Path('data/processed/train.csv'))
    test_df = pd.read_csv(Path('data/processed/test.csv'))

    train_x, train_y = split_xy(prepare_dataset(train_df), PriceTypeEnum.OFFER_PRICE)
    test_x, test_y = split_xy(prepare_dataset(test_df), PriceTypeEnum.OFFER_PRICE)

    train_log_y = np.log(train_y)

    model = LinearRegression()

    model.fit(train_x, train_log_y)

    test_log_preds = model.predict(test_x)
    test_preds = np.exp(test_log_preds)
    print(metrics_stat(test_y, test_preds))

    train_x_manual, train_y_manual = split_xy(prepare_dataset(train_df), PriceTypeEnum.MANUAL_PRICE)

    predictions_manual = model.predict(train_x_manual)
    predictions_offer = model.predict(train_x)
    corr_coef = find_corr_coefficient(train_y_manual, predictions_manual)

    metrics = metrics_stat(train_y.values, predictions_offer / (
                1 + corr_coef))  # для обучающей выборки с ценами из объявлений смотрим качество без коэффициента
    print(metrics)

    metrics = metrics_stat(train_y_manual.values, predictions_manual)
    print(metrics)
