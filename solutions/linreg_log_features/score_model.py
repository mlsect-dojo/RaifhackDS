import sys
sys.path.append('.')

from pathlib import Path
import pandas as pd
from solutions.baseline.raifhack_ds.settings import TARGET, NOT_ENCODABLE_FEATURES, CATEGORICAL_STE_FEATURES
from pipeline.transforms import LogTarget, LogNumericFeatures
from pipeline.evaluators import LinregEvaluator
from pipeline.pipelines import FitPreidctLogPipeline


if __name__ == '__main__':
    train_df = pd.read_csv(Path('data/processed/train.csv'))
    test_df = pd.read_csv(Path('data/processed/test.csv'))

    pipeline = FitPreidctLogPipeline(transforms=[LogTarget(),
                                                 LogNumericFeatures()],
                                     evaluator=LinregEvaluator({}))
    train_df = pipeline.preprocess(train_df)
    test_df = pipeline.preprocess(test_df)

    # drop target variable from dfs
    extra_cols = ['floor', 'lat', 'lng', 'LogTarget', TARGET] + CATEGORICAL_STE_FEATURES + NOT_ENCODABLE_FEATURES
    train_x = pipeline.postprocess(train_df, extra_cols, drop_na_cols=True)
    test_x  = pipeline.postprocess(test_df, extra_cols, drop_na_cols=True)
    train_y = train_df['LogTarget']
    test_y  = test_df['LogTarget']

    metrics = pipeline.fit_predict(train_x, train_y, test_x, test_y)
    print(metrics)