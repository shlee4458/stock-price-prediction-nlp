import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import matplotlib.pyplot as plt

EPOCHS = 500
LOOK_BACK = 1
TRAIN_SIZE = 0.7
LSTM_PARAMS = {
    "input_size": LOOK_BACK,
    "hidden_size": 20,
    "num_layers": 1,
    "batch_first": True
}

class LSTMNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_size, input_size = LSTM_PARAMS["hidden_size"], LSTM_PARAMS["input_size"]
        self.lstm = nn.LSTM(**LSTM_PARAMS)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

def load_data(filename):
    cols = ['open', 'high', 'low', 'close', 'volume', 'sentiment_nltk', 'up']
    df = pd.read_csv(filename, usecols=cols)
    return df

def train_test_split(df):
    train_size = int(len(df) * TRAIN_SIZE)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    return train_data, test_data

def create_dataset(dataset, lookback):
    X, y = [], []
    for i in range(len(dataset) - lookback):
        feature = dataset["close"].iloc[i: i + lookback].values
        target = dataset["close"].iloc[i + 1: i + lookback + 1].values
        X.append(feature)
        y.append(target)

    return torch.Tensor(X), torch.Tensor(y)

def train_model(model, X_train, y_train, X_test, y_test):
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()
    loader = data.DataLoader(data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

    for epoch in range(EPOCHS):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch % 100 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train)
            train_rmse = np.sqrt(loss_fn(y_pred, y_train))
            y_pred = model(X_test)
            test_rmse = np.sqrt(loss_fn(y_pred, y_test))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

if __name__ == "__main__":
    filename = "yahoo_news_preprocessed.csv"
    df = load_data(filename)
    timeseries = df[["close"]].values.astype("float32")
    train_data, test_data = train_test_split(df)
    X_train, y_train = create_dataset(train_data, LOOK_BACK)
    X_test, y_test = create_dataset(test_data, LOOK_BACK)
    model = LSTMNetwork()
    train_model(model, X_train, y_train, X_test, y_test)

    # plot the graph
    with torch.no_grad():
    # shift train predictions for plotting
    
        train_size = int(len(df) * TRAIN_SIZE)
        train_plot = np.ones_like(timeseries) * np.nan
        train_plot[LOOK_BACK:train_size] = model(X_train)
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[train_size+LOOK_BACK:len(timeseries)] = model(X_test)
# plot
    plt.figure(figsize=(15, 9))
    plt.plot(timeseries, linewidth=0.7)
    plt.plot(train_plot, c='r', linewidth=0.7)
    plt.plot(test_plot, c='g', linewidth=0.7)
    plt.show()


    