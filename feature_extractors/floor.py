from statistics import mean
from typing import Optional
import pandas as pd
import numpy as np


def floor_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    preprocess_floor = df['floor'].apply(parse_floor)
    floor = 1.93
    preprocess_floor = preprocess_floor.apply(lambda x: floor if pd.isna(x) else x)
    return preprocess_floor.apply(lambda x: np.log1p(x) if x > 0 else x)


def parse_floor(floor: Optional[str]) -> Optional[int]:
    if floor is None or pd.isna(floor):
        return
    if type(floor) == float or type(floor) == int:
        return int(floor)
    floor = floor.lower()
    # todo check -1 examples
    floor = floor.replace('подвал', ',0,')
    floor = floor.replace('цоколь', ',0,')
    floor = floor.replace('-', ',')

    # drop extra symbols
    allowed_chars = '1234567890,.'
    floors = list(filter(lambda c: c in allowed_chars, floor))
    floors = ''.join(floors).split(',')

    def safe_cast(number: str) -> Optional[int]:
        try:
            return int(float(number))
        except ValueError:
            pass

    # drop not numbers
    floors = list(map(safe_cast,  floors))
    floors = list(filter(lambda floor: floor is not None, floors))

    # calc mean floor
    if len(floors) == 0:
        return
    floor = int(mean(floors))
    return floor


if __name__ == '__main__':
    hard_cases = [
        '1',
        '18.0',
        'подвал, 1',
        '2',
        'подвал',
        'цоколь, 1',
        '1,2,антресоль',
        'цоколь',
        '4',
        '5',
        'тех.этаж (6)',
        '3',
        'Подвал',
        'Цоколь',
        '10',
        'фактически на уровне 1 этажа',
        '6',
        '1,2,3',
        '1, подвал',
        '1,2,3,4',
        '1,2',
        '1,2,3,4,5',
        '5, мансарда',
        '1-й, подвал',
        '12',
        '15',
        '13',
        '1, подвал, антресоль',
        'мезонин',
        'подвал, 1-3',
        '8',
        '7',
        '1 (Цокольный этаж)',
        '3, Мансарда (4 эт)',
        'подвал,1',
        '1, антресоль',
        '1-3',
        'мансарда (4эт)',
        '1, 2.',
        '9',
        'подвал , 1 ',
        '1, 2',
        'подвал, 1,2,3',
        '1 + подвал (без отделки)',
        'мансарда',
        '2,3',
        '4, 5',
        '1-й, 2-й',
        '18',
        '1 этаж, подвал',
        '1, цоколь',
        'подвал, 1-7, техэтаж',
        '3 (антресоль)',
        '1, 2, 3',
        'Цоколь, 1,2(мансарда)',
        'подвал, 3. 4 этаж',
        'подвал, 1-4 этаж',
        'подва, 1.2 этаж',
        '2, 3',
        '-1',
        '1.2',
        '11',
        '36',
        '7,8',
        '1 этаж',
        '1-й',
        '3 этаж',
        '4 этаж',
        '5 этаж',
        'подвал,1,2,3,4,5',
        '29',
        'подвал, цоколь, 1 этаж',
        '3, мансарда'
    ]

    for case in hard_cases:
        print(case, '->', parse_floor(case))
