from typing import List, Union
import pandas as pd


class BaseTransform:
    @property
    def name(self) -> str:
        return self.__class__.__name__

    def __call__(self, df: pd.DataFrame) -> Union[pd.Series, List[pd.Series]]:
        raise NotImplemented
