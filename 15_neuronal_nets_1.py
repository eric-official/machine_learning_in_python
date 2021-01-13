import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import compose, model_selection, preprocessing, neighbors
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import metrics

glassTrain = pd.read_csv('glass_train.csv')
glassTest = pd.read_csv('glass_test.csv')
metricCols = glassTrain.columns[1:-1]

filter = compose.ColumnTransformer(
    [
        ('minmax', preprocessing.MinMaxScaler(), metricCols),
        ('unveraendert', 'passthrough', ['Type'])
    ]
)
filter.fit(glassTrain)

dataTrain = filter.transform(glassTrain)
print(dataTrain.shape)
dataTest = filter.transform(glassTest)
Xtrain = dataTrain[:, :-1]
ytrain = dataTrain[:, -1]
Xtest = dataTest[:, :-1]
ytest = dataTest[:, -1]
print(Xtrain.shape, ytrain.shape)
print(Xtest.shape, ytest.shape)

input = layers.Input((9,))
h1 = layers.Dense(50, activation='relu')(input)
output = layers.Dense(8, activation='softmax')(h1)

model = keras.Model(input, output)
model.summary()

keras.utils.plot_model(model)

optimizer = keras.optimizers.Adam(0.001)
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy'])

history = model.fit(Xtrain, ytrain, epochs=5000, batch_size=30, verbose=2)

plt.subplot(1, 2, 1)
plt.plot(history.epoch, history.history['loss'])
plt.subplot(1, 2, 2)
plt.plot(history.epoch, history.history['accuracy'])

model.evaluate(Xtest, ytest, verbose=2)

dataPred = model.predict(Xtest)
print(dataPred)

yPred = np.argmax(dataPred, axis=1)
confusion = metrics.confusion_matrix(ytest, yPred)
print(confusion)
