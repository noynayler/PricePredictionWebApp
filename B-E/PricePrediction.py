import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load data from CSV
df = pd.read_csv('../../project1/.idea/data2.csv')
def pricePrediction(ItemCode):
    # Convert 'PriceUpdateDate' to datetime
    df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'])

    # Filter data for a specific item
    #item_code = 'P_7290000000121'  # Example item code
    item_code = ItemCode
    df_item = df[df['ItemCode'] == item_code].sort_values(by='PriceUpdateDate')

    # Prepare the data for LSTM
    prices = df_item[['PriceUpdateDate', 'ItemPrice']]
    prices.set_index('PriceUpdateDate', inplace=True)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))  # Adjust feature_range as necessary
    scaled_prices = scaler.fit_transform(prices[['ItemPrice']])
    prices['ScaledPrice'] = scaled_prices

    # Prepare sequences
    def create_sequences(data, seq_length):
        xs = []
        ys = []
        for i in range(len(data) - seq_length):
            x = data[i:i+seq_length]
            y = data[i+seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    seq_length = 5
    X, y = create_sequences(prices['ScaledPrice'].values, seq_length)

    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float().unsqueeze(-1)
    y = torch.from_numpy(y).float().unsqueeze(-1)

    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    test_size = len(X) - train_size

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    # Define the LSTM model
    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
            super(LSTM, self).__init__()
            self.hidden_layer_size = hidden_layer_size
            self.lstm = nn.LSTM(input_size, hidden_layer_size)
            self.linear = nn.Linear(hidden_layer_size, output_size)
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                                torch.zeros(1, 1, self.hidden_layer_size))

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 150

    for i in range(epochs):
        for seq, labels in zip(X_train, y_train):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 0:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

    # Make predictions
    model.eval()

    # Prepare test data for prediction
    test_inputs = X_test.tolist()
    predictions = []

    for i in range(len(X_test)):
        seq = torch.FloatTensor(test_inputs[i])
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            predictions.append(model(seq).item())

    # Inverse scale the predictions
    scaled_predictions = np.array(predictions).reshape(-1, 1)
    unscaled_predictions = scaler.inverse_transform(scaled_predictions).flatten()

    # Print the predictions and actual prices correctly aligned
    test_dates = prices.index[train_size + seq_length:].to_list()
    actual_prices = prices['ItemPrice'].values[train_size + seq_length:]

    print("Date\t\t\tPredicted Price\tActual Price")
    print("-----------------------------------------------")
    for date, pred_price, actual_price in zip(test_dates, unscaled_predictions, actual_prices):
        print(f"{date}\t{pred_price:.2f}\t\t{actual_price:.2f}")

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(prices.index, prices['ItemPrice'], label='Actual Prices')
    plt.plot(test_dates, unscaled_predictions, label='Predicted Prices', color='r')
    plt.xlabel('Date')
    plt.ylabel('Item Price')
    plt.title(f'Price Prediction for Item {item_code}')
    plt.legend()
    plt.show()
