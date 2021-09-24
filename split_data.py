from pathlib import Path
import pandas as pd


if __name__ == '__main__':
    root_data = Path('./data/raw')
    root_processed = Path('./data/processed')
    if not root_processed.exists():
        root_processed.mkdir(parents=True)

    df = pd.read_csv(root_data / 'train.csv')

    dates = sorted(df.date.unique())
    split_date = dates[int(len(dates) * 0.8)]
    train = df[df.date < split_date]
    test  = df[df.date >= split_date]

    train.to_csv(root_processed / 'train.csv', index=False)
    test.to_csv(root_processed / 'test.csv', index=False)
