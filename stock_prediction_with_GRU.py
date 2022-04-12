import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#try with other stock

stock_title = 'SCB'
stock_link = 'https://query1.finance.yahoo.com/v7/finance/download/SCB.BK?period1=1491955200&period2=1649721600&interval=1d&events=history&includeAdjustedClose=true'
date_split_train_test = '2022-02'
step = 20 #to predict ahead (date_split_train_test shoud be valid with the step, ex; later than 2022-02 should be 20 or more days)
epochs = 50

df = pd.read_csv(stock_link, index_col='Date')
df.index = pd.to_datetime(df.index)
df = df.Close
data = df.values.reshape(-1,1)
sc = MinMaxScaler()
data_sc = sc.fit_transform(data)

def convertToMatrix(data, step=1):
    X, Y = [],[]
    for i in range(len(data)-step):
        d = i + step
        X.append(data[i:d,]) #Training data from i to i + d
        Y.append(data[d,]) #Target data d
    return np.array(X), np.array(Y)

n_train = df[:date_split_train_test].shape[0]
train, test = data_sc[0:n_train], data_sc[n_train:]
X_train, y_train = convertToMatrix(train, step)
X_test, y_test = convertToMatrix(test, step)

model = tf.keras.models.load_model('stock_prediction_model/')
pred = model.predict(X_test)
pred_inv = sc.inverse_transform(pred)
y_test_inv = sc.inverse_transform(y_test)

plt.figure(figsize=(12,3.1))
plt.plot(y_test_inv, lw=1.3, label='Actual data')
plt.plot(pred_inv, lw=2.4, label='Prediction')
plt.title(stock_title)
plt.legend()
plt.show()