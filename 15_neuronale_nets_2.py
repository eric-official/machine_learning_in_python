import pandas as pd
import matplotlib.pyplot as plt
from sklearn import compose, model_selection, preprocessing
from tensorflow import keras
from tensorflow.keras import layers

diamonds = pd.read_csv("diamonds.csv")
diamonds.info()
print(diamonds.head(), diamonds.shape)
diaTrain, diaTest = model_selection.train_test_split(diamonds, test_size=0.3, random_state=23)

tr = compose.ColumnTransformer([
    ('num', preprocessing.MinMaxScaler(), ['carat', 'depth', 'table', 'x', 'y', 'z']),
    ('cat', preprocessing.OneHotEncoder(categories='auto', sparse=False), ['cut', 'color', 'clarity']),
    ('objective', 'passthrough', ['price']),
])

tr.fit(diaTrain)
dataTrain = tr.transform(diaTrain)
dataTest = tr.transform(diaTest)
Xtrain = dataTrain[:, :-1]
Xtest  = dataTest[:, :-1]

ytrain = dataTrain[:, -1]
ytest  = dataTest[:, -1]

print(Xtest.shape)
print(ytest[:10])

trY = preprocessing.MinMaxScaler()
trY.fit(ytrain.reshape(-1,1))
yTtrain = trY.transform(ytrain.reshape(-1,1)).reshape(-1)
yTtest = trY.transform(ytest.reshape(-1,1)).reshape(-1)

input=layers.Input(26)
h1=layers.Dense(50, activation='relu')(input)
out=layers.Dense(1)(h1)
model=keras.Model(input, out)
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss='mean_squared_error'
)

history=model.fit(Xtrain,ytrain,epochs=100,validation_split=0.5,batch_size=32,verbose=2)

plt.plot(history.epoch, history.history['loss'], label='train')
plt.plot(history.epoch, history.history['val_loss'], label='valid')

yTpred=model.predict(Xtest)
ypred=trY.inverse_transform(yTpred.reshape(-1,1)).reshape(-1)
ypred[:10]

plt.scatter(ytest, ypred)