from sklearn import naive_bayes
from sklearn import linear_model
import pandas as pd
from sklearn import compose, model_selection, preprocessing, neighbors

avocado=pd.read_csv('avocado.csv')
avocadoTrain, avocadoTest = model_selection.train_test_split(avocado, test_size=0.3, random_state=23)
metricCols=['Total Volume', '4046', '4225', '4770', 'Total Bags',
       'Small Bags', 'Large Bags', 'XLarge Bags', 'year']
catCols=['Date', 'type', 'region']

filter=compose.ColumnTransformer(
    [
        ('num1', preprocessing.MinMaxScaler(), metricCols),
        ('cat', preprocessing.OneHotEncoder(categories='auto', sparse=False), catCols),
        ('objective', 'passthrough', ['AveragePrice'])
    ]
)

labelEncoder=preprocessing.LabelEncoder()
filter.fit(avocadoTrain)

dataTrain=filter.transform(avocadoTrain)
dataTest=filter.transform(avocadoTest)

Xtrain=dataTrain[:,:-1]
ytrain=dataTrain[:, -1]
ytrain=labelEncoder.fit_transform(ytrain)

Xtest=dataTest[:,:-1]
ytest=dataTest[:, -1]
ytest=labelEncoder.fit_transform(ytest)

classifier=neighbors.KNeighborsClassifier(n_neighbors=3)
classifier.fit(Xtrain, ytrain)
print('Training: ', classifier.score(Xtrain, ytrain))
print('Test: ', classifier.score(Xtest, ytest))

classifier=neighbors.KNeighborsClassifier(n_neighbors=8)
classifier.fit(Xtrain, ytrain)
print('Training: ', classifier.score(Xtrain, ytrain))
print('Test: ', classifier.score(Xtest, ytest))

classifier=neighbors.KNeighborsClassifier(n_neighbors=13)
classifier.fit(Xtrain, ytrain)
print('Training: ', classifier.score(Xtrain, ytrain))
print('Test: ', classifier.score(Xtest, ytest))

classifier=naive_bayes.GaussianNB()
classifier.fit(Xtrain, ytrain)
print('Training: ', classifier.score(Xtrain, ytrain))
print('Test: ', classifier.score(Xtest, ytest))

classifier=linear_model.LogisticRegression(
    solver='lbfgs',multi_class='ovr',max_iter=4000,C=0.01
)
classifier.fit(Xtrain, ytrain)
print('Training: ', classifier.score(Xtrain, ytrain))
print('Test: ', classifier.score(Xtest, ytest))

classifier=linear_model.LogisticRegression(
    solver='lbfgs',multi_class='ovr',max_iter=4000,C=1
)
classifier.fit(Xtrain, ytrain)
print('Training: ', classifier.score(Xtrain, ytrain))
print('Test: ', classifier.score(Xtest, ytest))

classifier=linear_model.LogisticRegression(
    solver='lbfgs',multi_class='ovr',max_iter=4000,C=1000
)
classifier.fit(Xtrain, ytrain)
print('Training: ', classifier.score(Xtrain, ytrain))
print('Test: ', classifier.score(Xtest, ytest))