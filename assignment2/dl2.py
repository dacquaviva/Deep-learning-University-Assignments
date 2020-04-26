from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from math import sqrt
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
# transform list into supervised learning format


def series_to_supervised(data, n_in=1, n_out=1):
    df = DataFrame(data)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    agg.dropna(inplace=True)
    return agg.values
# split a univariate dataset into train/test sets


def train_test_split(data, n_test):
    train = data[:-n_test]
    test = data[-n_test:]
    X_train, y_train = train[:, 0:-1], train[:, -1]
    X_test, y_test = test[:, 0:-1], test[:, -1]
    return X_train, X_test, y_train, y_test
# scale train and test data to [0, 1]


def scale(data):
    scaler = MinMaxScaler()
    scaler.fit(data)
    data_scale = scaler.transform(data)
    return scaler, data_scale


def invert_scale(scaler, data):
    data = data.reshape(data.shape[0], 1)
    return scaler.inverse_transform(data)
# fit an MLP network to training data


def fit_model(X_train, y_train, batch_size, nb_epoch, hidden_layers, neurons):
    model = Sequential()
    model.add(Dense(neurons, activation='relu', input_dim=X_train.shape[1]))
    for i in range(hidden_layers):
        # Add one hidden layer
        model.add(
            Dense(neurons - 20, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, y_train, epochs=nb_epoch,
              batch_size=batch_size, verbose=0, shuffle=False)
    return model
# run a repeated experiment


def experiment(config):

    repeats, series, epochs, batch_size, hidden_layers, neurons, lag = config

    # scale data
    scaler, series_scaled = scale(series)

    supervised_values = series_to_supervised(series_scaled, lag)

    # split data into train and test-sets
    X_train, X_test, y_train, y_test = train_test_split(supervised_values, 200)
    # run experiment
    error_scores = list()
    for r in range(repeats):
        # fit the model
        model = fit_model(X_train, y_train, batch_size,
                          epochs, hidden_layers, neurons)
        # walk_forward_validation
        y_pred = model.predict(X_test, batch_size=batch_size)
        y_pred_inverse = invert_scale(scaler, y_pred)
        y_test_inverse = invert_scale(scaler, y_test)
        rmse = sqrt(mean_squared_error(y_test_inverse, y_pred_inverse))
        #print('%d) Test RMSE: %.3f' % (r+1, rmse))
        error_scores.append(rmse)
    return error_scores


# load dataset
series = loadmat('./Xtrain.mat')
series = series['Xtrain']

# experiment
repeats = 2
results = DataFrame()
n_epochs = [1]
n_batch = [16]
# One hidden layer is already present by default, so it indicates the number of hidden layers after the first hidden layer
n_hidden_layers = [0, 1]
n_neurons = [32, 64, 128]
n_lags = [5, 12, 30, 50, 70, 100]
count = 0
dictionary = {}
for epochs in n_epochs:
    for batch in n_batch:
        for hidden_layers in n_hidden_layers:
            for neurons in n_neurons:
                for lag in n_lags:
                    count = count + 1
                    dictionary[str(count)] = "Model -> Number Epoch = " + str(epochs) + ", Batch size = " + str(batch) + \
                        ", Number Hidden layer = " + \
                        str(hidden_layers) + ", Number of neurons = " + \
                        str(neurons) + ", Number of lag = " + str(lag)
                    cfg = [repeats, series, epochs, batch,
                           hidden_layers, neurons, lag]
                    results[str(count)] = experiment(cfg)
# summarize results
print(results.describe())
# save boxplot
results.boxplot(figsize=(28, 12))
pyplot.savefig('boxplot_neurons.png')
