import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVR

encoder = LabelEncoder()
scaler = StandardScaler()

train = pd.read_csv('../input/train.csv')
Y = train['loss']
train = train.drop(['id', 'loss'], axis=1)
continuous_cols = [c for c in train.columns if c[0:4] == 'cont']
train[continuous_cols] = scaler.fit_transform(train[continuous_cols])
cat_cols = set(c for c in train.columns if c[0:3] == 'cat')


test = pd.read_csv('../input/test.csv')
test_ids = test['id']
test = test.drop(['id'], axis=1)
test[continuous_cols] = scaler.transform(test[continuous_cols])

model = LinearSVR()
params = {'C': [10**-i for i in range(-2, 2)],
          'epsilon': (0.1, 0)}
clf = GridSearchCV(model, params, n_jobs=6, verbose=1)
clf.fit(train, Y)

predictions = clf.predict(test)
results = pd.DataFrame({'id': test_ids, 'loss': predictions})
results.to_csv('./results.csv')
