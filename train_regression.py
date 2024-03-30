import os
import csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns

from lstm_models import LSTM_Simple, LSTM_Deep

LOOKBACK = 60
EPOCHS = 500
BATCH_SIZE = 16
SET_TYPE = 3
MODEL = "simple"

TRAIN_SIZE = 0.8
PREDICT_NUM = 1
VALIDATION_SPLIT = 0.1
LOSS = "mse"
CLASSIFICATION = False

COLS = ["close", "open", "high", "low", "adjclose"] \
    + (["sentiment_nltk"] if SET_TYPE >= 2 else []) \
    + (["yield_rate", "vix_close", "cpi"])
OUTPUT = "close"

DEBUG = False
SAVE = False
PLOT = True

# TODO: 
# play with hyperparameters: LOOKBACK, EPOCHS, BATCHSIZE,
# build different lstm models: number of hidden nodes, layers...

def load_data(filename):
    df = pd.read_csv(filename)
    dates = pd.to_datetime(df['date'])
    data = df[COLS].astype(float)
    return data, dates, df

def split_x_y(data):
    X, y = [], [] # X: (1513, 14, 5), (1513, 1)
    for i in range(LOOKBACK, len(data) - PREDICT_NUM + 1):
        X.append(data[i - LOOKBACK:i, 0:data.shape[1]])
        y.append(data[i + PREDICT_NUM - 1:i + PREDICT_NUM, COLS.index(OUTPUT)]) # use close price as the output
    return X, y    

def split_train_test(X, y, train_size):
    # print(f"This is shape of the data: {data.shape}")
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size - LOOKBACK:], y[train_size - LOOKBACK:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


def train_lstm(X_train, y_train):
    num_features, lookback = X_train.shape[1], X_train.shape[2]

    # get the lstm model
    if MODEL == "simple":
        lstm = LSTM_Simple(num_features, lookback, 1, LOSS, CLASSIFICATION)
    elif MODEL == "deep":
        lstm = LSTM_Deep(num_features, lookback, 1, LOSS, CLASSIFICATION)

    model = lstm.get_model()
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=VALIDATION_SPLIT,
                        verbose=1)

    if PLOT:
        plt.plot(history.history['loss'], label='Training loss')
        plt.plot(history.history['val_loss'], label='Validation loss')
        plt.legend()
        
    if SAVE:
        version = 0
        val_filename = f"./output/val/val-{SET_TYPE}-{MODEL}-{EPOCHS}-{LOOKBACK}-v{version}.png"
        file_exists = os.path.isfile(val_filename)
        
        while file_exists:
            version += 1
            val_filename = val_filename[:-6] + f"v{version}" + ".png"
            file_exists = os.path.isfile(val_filename)
        plt.savefig(val_filename)

    plt.show()
    return model

def forecast(model, X_train, X_test, dates, scaler):
    train_size = len(X_train)
    test_dates = dates[train_size:]

    # make prediction
    prediction = model.predict(X_test) # X_train: shape = (num_rows, window_size, 1)
    prediction_copies = np.repeat(prediction, len(COLS), axis=-1)
    y_pred_output = scaler.inverse_transform(prediction_copies)[:,0]
    y_pred = pd.DataFrame({'date': np.array(test_dates), OUTPUT: y_pred_output})    
    y_pred['date'] = pd.to_datetime(y_pred['date'])

    # from pandas.tseries.holiday import USFederalHolidayCalendar
    # from pandas.tseries.offsets import CustomBusinessDay
    # n_days_for_prediction = len(X_test)
    # US_BD = CustomBusinessDay(calendar=USFederalHolidayCalendar())
    # train_dates = dates[:train_size]
    # forecast_dates = []
    # predict_period_dates = pd.date_range(list(train_dates)[-1], periods=n_days_for_prediction + 1, freq=US_BD).tolist()
    # for time_i in predict_period_dates[1:]:
    #     forecast_dates.append(time_i.date())
    # predict_data = pd.DataFrame({'date': np.array(forecast_dates), 'close': y_pred_future})

    if DEBUG:
        # print("Last train_date: ", list(train_dates)[-1])
        # print("This is where the train data ends: ", train_dates, "\n")
        # print("These are the dates we are predicting the price: ", predict_period_dates, "\n")
        print("X_train shape: ", X_train.shape)
        # print("predict shape: ", X_train[-n_days_for_prediction:].shape)
        print("This is the prediction before scaling: ", prediction, "\n")
        print("Actual price: ", )
        print("This is the predicted price in the furture: ", y_pred_output, "\n")
        print("test_dates length:", len(test_dates))
        print("pred_future length", len(y_pred_output))

    return y_pred

def plot_predict(original, predicted, save=True):
    original = original[['date', OUTPUT]]
    original['date'] = pd.to_datetime(original['date'])
    train_size = int(TRAIN_SIZE * len(original))
    train, test = original[:train_size], original[train_size:]

    if PLOT:
        plt.figure(figsize=(15, 9))
        sns.lineplot(data=train, x='date', y=OUTPUT, label='Train', color='blue')
        sns.lineplot(data=test, x='date', y=OUTPUT, label='Test', color='green')
        sns.lineplot(data=predicted, x='date', y=OUTPUT, label='Predicted', color='orange')
        plt.legend()

    if save:
        version = 0
        test_filename = f"./output/test/test-{SET_TYPE}-{MODEL}-{EPOCHS}-{LOOKBACK}-v{version}.png"
        file_exists = os.path.isfile(test_filename)
        
        while file_exists:
            version += 1
            test_filename = test_filename[:-6] + f"v{version}" + ".png"
            file_exists = os.path.isfile(test_filename)
        plt.savefig(test_filename)

    plt.show()

def evaluate_model(y_test, y_pred):
    y_pred = y_pred[[OUTPUT]].values
    mse = mean_squared_error(y_test, y_pred)
    return mse
    # error = np.sqrt(mse)
    # return error

def write_to_csv(error):
    filename = "./output/regression_data.csv"
    header = ["set_type","model", "epoch", "lookback", "error", "output"] # can add different variables
    row_formatted = f"{SET_TYPE},{MODEL},{EPOCHS},{LOOKBACK},{error:.4f},{OUTPUT}"
    row = row_formatted.split(",")
    file_exists = os.path.isfile(filename)

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)
    return

def main():

    # TODO: add arg parser for automating data collection using .sh
    # load the data
    filename = f"./data/set{SET_TYPE}.csv"
    data, dates, original = load_data(filename)

    # scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)

    # split the data
    X, y = split_x_y(data_scaled)
    train_size = int(len(original) * TRAIN_SIZE)
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_size)
    model = train_lstm(X_train, y_train)

    # predict and plot the data
    y_pred = forecast(model, X_train, X_test, dates, scaler)
    plot_predict(original, y_pred, SAVE)

    # evaluate the test and actual
    error = evaluate_model(y_test, y_pred)

    # write the result to a csv file for analysis
    write_to_csv(error)

if __name__ == "__main__":
    main()