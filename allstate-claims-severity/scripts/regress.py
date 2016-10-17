from time import time
from json import dump
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.linear_model import (
#     SGDRegressor, RidgeCV, Ridge, ElasticNet, ElasticNetCV)
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error

np.random.seed(17)

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


def dummies(df):
    for col in df.columns:
        if col[0:3] == 'cat':
            df[col] = df[col].astype('category')
    df = pd.get_dummies(df, drop_first=True)
    return df

train = dummies(train)

y = train['loss']
train = train.drop(['id', 'loss'], axis=1)


x_train, x_test, y_train, y_test = train_test_split(
    train, y, test_size=0.2)

model = LinearSVR()
c_range = list()
params = {'C': range(0.001, 10), 'epsilon': (0.1, 0)}

clf = GridSearchCV(model, params, n_jobs=4, verbose=3)
clf.fit(x_train, y_train)

results = pd.DataFrame(clf.cv_results_)
results.to_json('model_results.json')

predictions = clf.predict(x_test)
mae = mean_absolute_error(y_true, y_pred)
print('mean_absolute_error: {}'.format(mae))
