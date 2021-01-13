from sklearn.neighbors import KNeighborsClassifier
from sklearn import naive_bayes
from sklearn import linear_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import compose, model_selection, preprocessing, neighbors
from sklearn.compose import ColumnTransformer

avocado=pd.read_csv('avocado.csv')
avocado=avocado.head(1000)
avocadoTrain, avocadoTest = model_selection.train_test_split(avocado, test_size=0.3, random_state=23)
metricCols=['Total Volume', '4046', '4225', '4770', 'Total Bags',
       'Small Bags', 'Large Bags', 'XLarge Bags', 'year']
catCols=['Date', 'type', 'region']

filter=compose.ColumnTransformer(
    [
        ('num', preprocessing.StandardScaler(), metricCols),
        ('cat', preprocessing.OrdinalEncoder(), catCols),
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

from sklearn import svm

paramGrid={
    'C': np.logspace(1, 5, 8),
    'gamma': np.logspace(-5, 1 ,10)
}

classifier=svm.SVR(gamma='auto', kernel='rbf')
classifier.fit(Xtrain, ytrain)

gridSearch=model_selection.GridSearchCV(estimator=classifier,
                                        param_grid=paramGrid,
                                        cv=3)

gridSearch.fit(Xtrain, ytrain)
gridSearch.best_params_

pC=gridSearch.cv_results_['param_C']
pGamma=gridSearch.cv_results_['param_gamma']
scores=gridSearch.cv_results_['mean_test_score']

lC=len(paramGrid['C'])
lg=len(paramGrid['gamma'])

plt.contourf(pC.reshape(lC, lg), pGamma.reshape(lC, lg), scores.reshape(lC, lg))
plt.xscale('log')
plt.yscale('log')
plt.xlabel("C")
plt.ylabel("gamma")
plt.title("Support Vector Machine")
plt.colorbar()
plt.show()

print("Der optimale Wert für C ist:", gridSearch.best_params_['C'])
print("Der optimale Wert für gamma ist:", gridSearch.best_params_['gamma'])

bestC=gridSearch.best_params_['C']
bestGamma=gridSearch.best_params_['gamma']

classifier=svm.SVR(C=bestC, gamma=bestGamma, kernel='rbf')
classifier.fit(Xtrain, ytrain)
print(" ")
print("Somit erhält man für:")
print("  - die Trainingsdaten einen Score von:", classifier.score(Xtrain, ytrain))
print("  - und für die Testdaten einen Score von:", classifier.score(Xtest, ytest))

from sklearn import tree

paramGrid={
    'max_depth' : np.arange(2,100,10),
    'min_samples_leaf': np.arange(10,50,10)
}

classifier = tree.DecisionTreeClassifier()
classifier.fit(Xtrain, ytrain)

gridSearch=model_selection.GridSearchCV(estimator=classifier,
                                        param_grid=paramGrid,
                                        cv=3)

gridSearch.fit(Xtrain, ytrain)
gridSearch.best_params_

pD=gridSearch.cv_results_['param_max_depth']
pS=gridSearch.cv_results_['param_min_samples_leaf']
scores=gridSearch.cv_results_['mean_test_score']

lD=len(paramGrid['max_depth'])
lL=len(paramGrid['min_samples_leaf'])

plt.contourf(pD.reshape(lD, lL), pS.reshape(lD, lL), scores.reshape(lD, lL))
plt.colorbar()
plt.xlabel("max_depth")
plt.ylabel("min_samples_leaf")
plt.title("Decision Tree")
plt.show()

print("Der optimale Wert für max_depth ist:", gridSearch.best_params_['max_depth'])
print("Der optimale Wert für min_samples_leaf ist:", gridSearch.best_params_['min_samples_leaf'])

bestDepth=gridSearch.best_params_['max_depth']
bestLeaf=gridSearch.best_params_['min_samples_leaf']

classifier=tree.DecisionTreeClassifier(max_depth=bestDepth, min_samples_leaf=bestLeaf)
classifier.fit(Xtrain, ytrain)
print(" ")
print("Somit erhält man für:")
print("  - die Trainingsdaten einen Score von:", classifier.score(Xtrain, ytrain))
print("  - und für die Testdaten einen Score von:", classifier.score(Xtest, ytest))