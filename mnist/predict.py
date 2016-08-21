"""create a model for mnist dataset."""

import numpy as np
import pandas as pd
from csv import DictWriter
from sklearn import svm

data = pd.read_csv('./data/train.csv', dtype=np.float, engine='c')
y = data['label'].astype(np.int)
X = data.drop('label', axis=1)
X = X.apply(lambda x: x * (1.0 / 255.0))

cls = svm.SVC()
cls.fit(X, y)

test = pd.read_csv('./data/test.csv', dtype=np.float, engine='c')
test = test.apply(lambda x: x * (1.0 / 255.0))
results = cls.predict(test)
with open('py_results.csv', 'w') as f:
    writer = DictWriter(f, ['ImageId', 'Label'])
    writer.writeheader()
    for i, v in enumerate(results):
        writer.writerow({
            'ImageId': i + 1,
            'Label': v
        })
