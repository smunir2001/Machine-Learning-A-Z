# smunir2001@gmail.com | November 9th, 2022 | Magic_Example.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# dataset (magic04.data) from https://archive.ics.uci.edu/ml/datasets/MAGIC+Gamma+Telescope
cols = ['fLength', 'fWidth', 'fSize', 'fConc', 'fConc1', 'fAsym', 'fM3Long', 'fM3Trans', 'fAlpha', 'fDist', 'class']
df = pd.read_csv('magic04.data', names = cols)
# print(df.head)
# print(df['class'].unique())
df['class'] = (df['class'] == 'g').astype(int)
print('\ndf.head =\n', df.head, '\n')

# for label in cols[:-1]:
#     plt.hist(df[df['class'] == 1][label], color = 'blue', label = 'gamma', alpha = 0.7, density = True)
#     plt.hist(df[df['class'] == 0][label], color = 'red', label = 'gamma', alpha = 0.7, density = True)
#     plt.title(label)
#     plt.ylabel('Probability')
#     plt.xlabel(label)
#     plt.legend()
#     plt.show()

# train, validation, test datasets
train, valid, test = np.split(df.sample(frac = 1), [int(0.6 * len(df)), int(0.8 * len(df))])

def scale_dataset(dataframe, oversample = False):
    x = dataframe[dataframe.columns[:-1]].values
    y = dataframe[dataframe.columns[-1]].values
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    if oversample:
        ros = RandomOverSampler()
        x, y = ros.fit_resample(x, y)
    data = np.hstack((x, np.reshape(y, (-1, 1))))
    return data, x, y

# print("len(train[train['class'] == 1", len(train[train['class'] == 1])) # gamma
# print("\nlen(train[train['class'] == 0", len(train[train['class'] == 0])) # hydron

# train, x_train, y_train = scale_dataset(train, oversample = True)
# print('\nlen(y_train) = ', len(y_train))
# print('\nsum(y_train == 1) = ', sum(y_train == 1))
# print('\nsum(y_train == 0) = ', sum(y_train == 0))

train, x_train, y_train = scale_dataset(train, oversample = True)
valid, x_valid, y_valid = scale_dataset(valid, oversample = False)
test, x_test, y_test = scale_dataset(test, oversample = False)
# print('\nlen(y_train) = ', len(y_train))
# print('sum(y_train == 1) = ', sum(y_train == 1))
# print('sum(y_train == 0) = ', sum(y_train == 0), '\n')

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
nbrOfNeighbors = 3
kan_model = KNeighborsClassifier(n_neighbors = nbrOfNeighbors)
kan_model.fit(x_train, y_train)
y_pred  = kan_model.predict(x_test)
print('K-Nearest Neighbors\n-------------------')
print('ypred =\n', y_pred)
print('y_test =\n', y_test, '\n')
print(classification_report(y_test, y_pred))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model = nb_model.fit(x_train, y_train)
y_pred  = nb_model.predict(x_test)
print('Naive Bayes\n-----------')
print('y_pred =\n', y_pred)
print('y_test =\n', y_test, '\n')
print(classification_report(y_test, y_pred))

# Logistic Regression
from sklearn.linear_model import LogisticRegression
lg_model = LogisticRegression()
lg_model = lg_model.fit(x_train, y_train)
y_pred = lg_model.predict(x_test)
print('Logistic Regression\n-------------------')
print('y_pred =\n', y_pred)
print('y_test =\n', y_test, '\n')
print(classification_report(y_test, y_pred))

# Support Vector Machines (SVM)
from sklearn.svm import SVC
svm_model = SVC()
svm_model = svm_model.fit(x_train, y_train)
y_pred = svm_model.predict(x_test)
print('Support Vector Machines (SVM)\n-------------------')
print('y_pred =\n', y_pred)
print('y_test =\n', y_test, '\n')
print(classification_report(y_test, y_pred))