import pandas as pd


def mortage_rate(df: pd.DataFrame):
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    a1 = pd.to_datetime(pd.Series('2020-01-01'), format='%Y-%m-%d').values[0]
    a2 = pd.to_datetime(pd.Series('2020-02-09'), format='%Y-%m-%d').values[0]
    a3 = pd.to_datetime(pd.Series('2020-02-09'), format='%Y-%m-%d').values[0]
    a4 = pd.to_datetime(pd.Series('2020-04-26'), format='%Y-%m-%d').values[0]
    a5 = pd.to_datetime(pd.Series('2020-04-26'), format='%Y-%m-%d').values[0]
    a6 = pd.to_datetime(pd.Series('2020-06-21'), format='%Y-%m-%d').values[0]
    a7 = pd.to_datetime(pd.Series('2020-06-21'), format='%Y-%m-%d').values[0]
    a8 = pd.to_datetime(pd.Series('2020-07-26'), format='%Y-%m-%d').values[0]
    a9 = pd.to_datetime(pd.Series('2020-07-26'), format='%Y-%m-%d').values[0]
    a10 = pd.to_datetime(pd.Series('2020-12-31'), format='%Y-%m-%d').values[0]

    def add_stavka(row):
        if a1 < row < a2:
            row = 6.25
        elif a3 <= row < a4:
            row = 6.0
        elif a5 <= row < a6:
            row = 5.5
        elif a7 <= row < a8:
            row = 4.5
        elif a9 <= row < a10:
            row = 4.25
        return row


    stavka = df.date.apply(add_stavka)
    return stavka

