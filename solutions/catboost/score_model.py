import sys
sys.path.append('.')

from pathlib import Path
import pandas as pd
from solutions.baseline.raifhack_ds.settings import TARGET, NOT_ENCODABLE_FEATURES, CATEGORICAL_STE_FEATURES
from pipeline.transforms import LogTarget, LogNumericFeatures
from pipeline.evaluators import CatboostEvaluator
from pipeline.pipelines import FitPreidctLogPipeline


if __name__ == '__main__':
    train_df = pd.read_csv(Path('data/processed/train.csv'))
    test_df = pd.read_csv(Path('data/processed/test.csv'))

    pipeline = FitPreidctLogPipeline(transforms=[LogTarget(),
                                                 LogNumericFeatures()],
                                     evaluator=CatboostEvaluator({}))
    train_df = pipeline.preprocess(train_df)
    test_df = pipeline.preprocess(test_df)

    train_df = train_df[train_df.price_type == 1]
    test_df = test_df[test_df.price_type == 1]

    # drop target variable from dfs
    extra_cols = ['floor', 'lat', 'lng', TARGET] + CATEGORICAL_STE_FEATURES + NOT_ENCODABLE_FEATURES
    train_df = pipeline.postprocess(train_df, extra_cols, drop_na_rows=True)
    test_df  = pipeline.postprocess(test_df, extra_cols, drop_na_rows=True)
    train_y = train_df['LogTarget']
    test_y = test_df['LogTarget']
    train_x = train_df.drop('LogTarget', axis=1)
    test_x  = test_df.drop('LogTarget', axis=1)


    metrics = pipeline.fit_predict(train_x, train_y, test_x, test_y)
    print(metrics)
