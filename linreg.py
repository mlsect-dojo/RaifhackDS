from pathlib import Path
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from baseline.raifhack_ds.settings import MODEL_PARAMS, LOGGING_CONFIG, NUM_FEATURES, CATEGORICAL_OHE_FEATURES,CATEGORICAL_STE_FEATURES,TARGET
from baseline.raifhack_ds.utils import PriceTypeEnum
from baseline.raifhack_ds.features import prepare_categorical


if __name__ == '__main__':
    train_df = pd.read_csv(Path('data/processed/train.csv'))
    train_df = prepare_categorical(train_df)

    X_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][
        NUM_FEATURES + CATEGORICAL_OHE_FEATURES + CATEGORICAL_STE_FEATURES]
    y_offer = train_df[train_df.price_type == PriceTypeEnum.OFFER_PRICE][TARGET]

    X_offer = X_offer.drop(['region', 'city', 'realty_type'], axis=1)

    not_na_cols = X_offer.isna().sum(axis=0) == 0
    X_offer = X_offer[not_na_cols[not_na_cols].index.tolist()]

    model = LinearRegression()
    model.fit(X_offer, y_offer)