import pandas as pd
import numpy as np


def type_convert(x, type_=np.int16):
    """missing values are ' NA'."""
    try:
        return type_(x)
    except ValueError:
        return np.nan

# DtypeWarning: Columns (5,8,11,15) have mixed types
converters = {
    'age': type_convert,
    'antiguedad': type_convert,
    'fecha_dato': pd.to_datetime,
    'fecha_alta': pd.to_datetime
}
dtypes = {
    'indrel_1mes': pd.Categorical,
    'tiprel_1mes': pd.Categorical,
    'indrel': pd.Categorical,
    'indresi': pd.Categorical,
    'indext': pd.Categorical,
    'conyuemp': pd.Categorical,
    'conyuemp': pd.Categorical
}
df = pd.read_csv('../input/train_ver2.csv',
                 converters=converters, dtype=dtypes)

if __name__ == '__main__':
    print(df.dtypes)
