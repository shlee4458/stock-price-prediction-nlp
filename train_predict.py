import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from lstm_models import LSTM_Simple

TRAIN_SIZE = 0.8
INITIAL_TRAIN_SIZE = 500
PREDICT_NUM = 1
LOOKBACK = 30
EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
COLS = ["close", "open", "high", "low", "sentiment_nltk"]

DEBUG = False

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
        y.append(data[i + PREDICT_NUM - 1:i + PREDICT_NUM, 0]) # use close price as the output
    return X, y    

def split_train_test(X, y, train_size):
    print(f"This is shape of the data: {data.shape}")
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size - LOOKBACK:], y[train_size - LOOKBACK:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def train_lstm(X_train, y_train):
    num_features, lookback = X_train.shape[1], X_train.shape[2]

    lstm = LSTM_Simple(num_features, lookback, 1)
    model = lstm.get_model()
    history = model.fit(X_train, y_train,
                        epochs=EPOCHS, 
                        batch_size=BATCH_SIZE, 
                        validation_split=VALIDATION_SPLIT,
                        verbose=1)

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

    return model

def forecast(model, X_train, X_test, y_test, dates, scaler):
    train_size = len(X_train)
    test_dates = dates[train_size:]

    # make prediction
    prediction = model.predict(X_test) # X_train: shape = (num_rows, window_size, 1)
    prediction_copies = np.repeat(prediction, len(COLS), axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]
    predict_data = pd.DataFrame({'date': np.array(test_dates), 'close': y_pred_future})    
    predict_data['date'] = pd.to_datetime(predict_data['date'])

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
        print("This is the predicted price in the furture: ", y_pred_future, "\n")
        print("test_dates length:", len(test_dates))
        print("pred_future length", len(y_pred_future))

    return predict_data

def plot_predict(original, predicted):
    original = original[['date', 'close']]
    original['date']=pd.to_datetime(original['date'])
    train_size = int(TRAIN_SIZE * len(original))
    train, test = original[:train_size], original[train_size:]
    
    test_filename = f"./output/test-ep-{EPOCHS}-lb-{LOOKBACK}.png"

    plt.figure(figsize=(15, 9))
    sns.lineplot(data=train, x='date', y='close', label='Train', color='blue')
    sns.lineplot(data=test, x='date', y='close', label='Test', color='green')
    sns.lineplot(data=predicted, x='date', y='close', label='Predicted', color='orange')
    plt.legend()
    plt.savefig(test_filename)

    plt.show()

if __name__ == "__main__":

    # load the data
    filename = "./data/yahoo_news_preprocessed.csv"
    data, dates, original = load_data(filename)
    # print(data)

    # scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)

    X, y = split_x_y(data_scaled)
    train_size = int(len(original) * TRAIN_SIZE)
    X_train, y_train, X_test, y_test = split_train_test(X, y, train_size)
    model = train_lstm(X_train, y_train)

    # predict and plot the data
    predicted = forecast(model, X_train, X_test, y_test, dates, scaler)
    plot_predict(original, predicted)


    # note--ignore for now please;
    # we can train using "train_size" and predict the price at the train_size + 1
    # we can have reasonable initial train_size, and iterate until the last day of which we have the data

    # split the data
    # X, y = split_x_y(data_scaled)
    # for train_size in range(INITIAL_TRAIN_SIZE, len(data_scaled)):
    #     X_train, y_train, _, _ = split_train_test(X, y, train_size)

    #     # train lstm
    #     # print(X_train)
    #     model = train_lstm(X_train, y_train)

    #     # predict and plot the data
    #     predicted = forecast(model, X_train, dates, scaler)
    #     plot_predict(original, predicted)