from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras import backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

class LSTM_Simple:
    # (5, 14, 64) -> (64, 32) -> (32, 1)
    def __init__(self, num_features, lookback, output_size, loss, classification=False):
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(num_features, lookback), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(output_size))
        metrics=["accuracy"] if classification else []
        model.compile(optimizer='adam', loss=loss, metrics=metrics)

        self.model = model
    
    def get_model(self):
        return self.model