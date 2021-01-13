import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import compose, model_selection, preprocessing
from sklearn.compose import ColumnTransformer

glassTrain=pd.read_csv('glass_train.csv')
glassTest=pd.read_csv('glass_test.csv')
print(glassTrain.shape, glassTest.shape)
glassTrain.info()

plt.boxplot(glassTrain.values[:,1:-1])
plt.show()

metricCols=glassTrain.columns[1:-1]
print(metricCols)

filter=compose.ColumnTransformer(
    [
        ('minmax', preprocessing.MinMaxScaler(), metricCols),
        ('unveraendert', preprocessing.OneHotEncoder(categories='auto'), ['Type'])
    ]
)
filter.fit(glassTrain)

dataTrain=filter.transform(glassTrain)
dataTest=filter.transform(glassTest)
Xtrain=dataTrain[:,:-1]
ytrain=dataTrain[:,-1]
Xtest=dataTest[:,:-1]
ytest=dataTrain[:,-1]
print(Xtrain.shape, ytrain.shape)

plt.boxplot(Xtrain)
plt.show()