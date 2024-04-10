from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, RepeatVector, TimeDistributed
from keras import backend as K
from keras import optimizers

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def mase(y_true, y_pred):
    return 

class LSTMFactory:
    def __init__(self, num_features, lookback, output_size, loss, classification=False, model_type=1):
        model_map = {1: LSTM_1,
                     2: LSTM_2,
                    #  3: LSTM_3,
                     }

        self.model = model_map[model_type](num_features, lookback, output_size, loss, classification)

    def get_model(self):
        return self.model.get_model()

class LSTM_1():
    # (5, 14, 64) -> (64, 32) -> (32, 1)
    def __init__(self, num_features, lookback, output_size, loss, classification=False):
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=(num_features, lookback), return_sequences=True))
        model.add(LSTM(32, activation='relu', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(output_size))
        # optimizer = optimizers.Adam(learning_rate=0.001)
        metrics=["accuracy"] if classification else []
        # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        model.compile(loss=loss, metrics=metrics)
        self.model = model
    
    def get_model(self):
        return self.model
    
class LSTM_2():
    # Works better with set 3, high lookback
    def __init__(self, num_features, lookback, output_size, loss, classification=False):
        model = Sequential()
        model.add(LSTM(128, activation='relu', input_shape=(num_features, lookback), return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu', return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(128, activation='relu', return_sequences=False))
        model.add(Dropout(0.3))        
        model.add(Dense(output_size))
        model.compile(optimizer='adam', loss=loss)

        self.model = model
    
    def get_model(self):
        return self.model 

# class LSTM_3():
#     def __init__(self, num_features, lookback, output_size, loss, classification=False):
#         model = Sequential()
#         model.add(LSTM(128, activation='relu', input_shape=(num_features, lookback), return_sequences=True))
#         model.add(Dropout(0.4))
#         # model.add(RepeatVector(3))
#         model.add(TimeDistributed(Dense(output_size)))
#         # model.add(LSTM(128, activation='relu', return_sequences=True))
#         # model.add(Dropout(0.4))
#         # model.add(LSTM(128, activation='relu', return_sequences=False))
#         # model.add(Dropout(0.4))
#         model.add(Dense(output_size))
#         model.compile(optimizer='adam', loss=loss)

#         self.model = model
    
#     def get_model(self):
#         return self.model 
