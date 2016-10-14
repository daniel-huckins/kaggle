from sys import exit
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

encoder = LabelEncoder()
encoder = encoder.fit(train['species'])
labels = encoder.transform(train['species'])
test_ids = test['id']


train = train.drop(['id', 'species'], axis=1)
test = test.drop(['id'], axis=1)

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=17)


for train_idx, test_idx in splitter.split(train, labels):
    X_train, X_test = train.values[train_idx], train.values[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]

# print(y_train)
# exit()

# LinearDiscriminantAnalysis has no tuning params?
# solver 'lsqr' (least squares) gives same accuracy but larger log loss
classifier = LinearDiscriminantAnalysis(solver='svd')
print('fitting')
classifier.fit(X_train, y_train)

# test params
# predictions = classifier.predict(X_test)
# acc = accuracy_score(y_test, predictions)
# print('Accuracy: {:.4%}'.format(acc))
#
# predictions = classifier.predict_proba(X_test)
# loss = log_loss(y_test, predictions)
# print('log loss: {}'.format(loss))

# submit
predictions = classifier.predict_proba(test)
submission = pd.DataFrame(predictions, columns=list(encoder.classes_))
submission.insert(0, 'id', test_ids)
submission.to_csv('submission.csv', index=False)
