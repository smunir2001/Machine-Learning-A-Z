# smunir2001@gmail.com | November 3, 2022 | Data_Preprocessing_Tools.py
print('\nData_Preprocessing_Tools.py');
print('Sami Munir | smunir2001@gmail.com');
print('---------------------------------\n');

# Importing the libraries
print('importing the libraries...\n');
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
print('importing the dataset...')
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x, '\n')
print(y, '\n')

# Taking care of missing data
print('taking care of missing data...')
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])
print(x, '\n')

# Encoding categorical data
print('encoding categorical data...')
# Encoding the independent variable
print('\tencoding the independent variable...')
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder ='passthrough')
x = np.array(ct.fit_transform(x))
print(x, '\n')
# Encoding the dependent variable
print('\tencoding the dependent variable...')
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y, '\n')

# Splitting the dataset into training/test sets
print('splitting the dataset into training/test sets...')
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
print('x_train:\n', x_train, '\n')
print('x_test:\n', x_test, '\n')
print('y_train:\n', y_train, '\n')
print('y_test:\n', y_test, '\n')

# Feature scaling
print('feature scaling...')
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:])
x_test[:, 3:] = sc.transform(x_test[:, 3:])
print('\tx_train:\n', x_train)
print('\tx_test:\n', x_test)

# Conclusion