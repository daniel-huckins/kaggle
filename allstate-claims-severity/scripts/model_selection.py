from sys import exit
from time import time
from json import dump
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error

np.random.seed(17)

scaler = StandardScaler()

train = pd.read_csv('../input/train.csv')
continuous_cols = [c for c in train.columns if c[0:4] == 'cont']
train[continuous_cols] = scaler.fit_transform(train[continuous_cols])

test = pd.read_csv('../input/test.csv')
test[continuous_cols] = scaler.transform(test[continuous_cols])


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
params = {'C': [10**-i for i in range(-2, 2)],
          'epsilon': (0.1, 0)}

clf = GridSearchCV(model, params, n_jobs=6, verbose=1)
print(x_train.columns)
exit()
clf.fit(x_train, y_train)

results = pd.DataFrame(clf.cv_results_)
results.to_json('./model_results.json')

predictions = clf.predict(x_test)
mae = mean_absolute_error(y_test, predictions)
print('mean_absolute_error: {}'.format(mae))
