import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns

TRAIN_SIZE = 0.8
FUTURE_NUM = 1
PAST_NUM = 14
EPOCHS = 5
BATCH_SIZE = 16
VALIDATION_SPLIT = 0.1
COLS = ["close", "open", "high", "low", "sentiment_nltk"]

def load_data(filename):
    df = pd.read_csv(filename)
    dates = pd.to_datetime(df['date'])
    data = df[COLS].astype(float)
    return data, dates

def train_lstm(X_train, y_train):
    num_features, lookback = X_train.shape[1], X_train.shape[2]
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(num_features, lookback), return_sequences=True))
    model.add(LSTM(32, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_features))

    model.compile(optimizer='adam', loss='mse')
    model.summary()

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

def split_train_test(data):
    X, y = [], [] # X: (1513, 14, 5), (1513, 1)
    for i in range(PAST_NUM, len(data) - FUTURE_NUM + 1):
        X.append(data[i - PAST_NUM:i, 0:data.shape[1]])
        y.append(data[i + FUTURE_NUM - 1:i + FUTURE_NUM, 0]) # use close price as the output
    
    train_size = int(len(X) * TRAIN_SIZE)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

def forecast(model, X_train, dates):
    n_past = 16
    n_days_for_prediction = 15
    train_size = int(len(X_train) * TRAIN_SIZE)
    train_dates = dates[:train_size]
    predict_period_dates = pd.date_range(list(train_dates)[-n_past], periods=n_days_for_prediction).tolist()
    print(predict_period_dates)

    #Make prediction
    prediction = model.predict(X_train[-n_days_for_prediction:]) #shape = (n, 1) where n is the n_days_for_prediction

    #Perform inverse transformation to rescale back to original range
    #Since we used 5 variables for transform, the inverse expects same dimensions
    #Therefore, let us copy our values 5 times and discard them after inverse transform
    prediction_copies = np.repeat(prediction, len(COLS), axis=-1)
    y_pred_future = scaler.inverse_transform(prediction_copies)[:,0]

    forecast_dates = []
    for time_i in predict_period_dates:
        forecast_dates.append(time_i.date())
        
    df_forecast = pd.DataFrame({'Date':np.array(forecast_dates), 'Open':y_pred_future})
    df_forecast['Date']=pd.to_datetime(df_forecast['Date'])

    original = df[['Date', 'Open']]
    original['Date']=pd.to_datetime(original['Date'])
    original = original.loc[original['Date'] >= '2020-5-1']

    sns.lineplot(original['Date'], original['Open'])
    sns.lineplot(df_forecast['Date'], df_forecast['Open'])

if __name__ == "__main__":
    # load the data
    filename = "yahoo_news_preprocessed.csv"
    data, dates = load_data(filename)

    # scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(data)
    data_scaled = scaler.transform(data)

    # split the data
    X_train, y_train, X_test, y_test = split_train_test(data_scaled)
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    # train lstm
    model = train_lstm(X_train, y_train)
    forecast(model, X_train, dates)