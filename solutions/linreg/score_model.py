import sys
sys.path.append('.')

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from solutions.baseline.raifhack_ds.metrics import metrics_stat
from solutions.linreg.linreg import split_xy, prepare_dataset


train_df = pd.read_csv(Path('data/processed/train.csv'))
test_df = pd.read_csv(Path('data/processed/test.csv'))

train_x, train_y = split_xy(prepare_dataset(train_df))
test_x, test_y = split_xy(prepare_dataset(test_df))

train_log_y = np.log(train_y)

model = LinearRegression()
# todo: add correction coef
model.fit(train_x, train_log_y)

test_log_preds = model.predict(test_x)
test_preds = np.exp(test_log_preds)
print(metrics_stat(test_y, test_preds))
