import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Noa Added
import matplotlib
matplotlib.use('Agg')  # Use 'Agg' backend for environments without a display
import matplotlib.pyplot as plt
# Example of plotting
plt.figure(figsize=(10, 5))
# ... your plotting code ...
plt.savefig('path_to_save_graph.png')  # Save the figure to a file
plt.close()  # Close the figure


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['shufersal_data']
collection = db['products_labels']



def pricePrediction(df):
    item_code = df['ItemCode'].iloc[0]
    df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'])
    df_item = df.sort_values(by='PriceUpdateDate').copy()

    prices = df_item[['PriceUpdateDate', 'ItemPrice']].copy()
    prices.set_index('PriceUpdateDate', inplace=True)

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
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

    seq_length = 5  # Adjust based on your model input size
    X, y = create_sequences(prices['ScaledPrice'].values, seq_length)

    # Convert to PyTorch tensors
    X = torch.from_numpy(X).float().unsqueeze(-1)
    y = torch.from_numpy(y).float().unsqueeze(-1)

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

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
    epochs = 200
    for i in range(epochs):
        model.train()
        for seq, labels in zip(X_train, y_train):
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        # Validate the model
        model.eval()
        val_losses = []
        for seq, labels in zip(X_val, y_val):
            with torch.no_grad():
                model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                     torch.zeros(1, 1, model.hidden_layer_size))
                val_pred = model(seq)
                val_loss = loss_function(val_pred, labels)
                val_losses.append(val_loss.item())

        if i % 25 == 0:
            print(f'epoch: {i:3} train_loss: {single_loss.item():10.8f} val_loss: {np.mean(val_losses):10.8f}')

    # Make predictions for the next week (7 days)
    model.eval()

    # Prepare data for prediction
    last_seq = torch.FloatTensor(X[-1])
    predictions = []

    for _ in range(7):
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            pred = model(last_seq)
            predictions.append(pred.item())

            # Update the sequence to include the new prediction
            new_seq = torch.cat((last_seq[1:], pred.unsqueeze(0)))
            last_seq = new_seq

    # Inverse scale the predictions
    scaled_predictions = np.array(predictions).reshape(-1, 1)
    unscaled_predictions = scaler.inverse_transform(scaled_predictions).flatten()

    # Extend dates for prediction
    last_date = prices.index[-1]
    next_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=7, freq='D')

    # Create plot for predictions
    plt.figure(figsize=(10, 5))
    plt.plot(prices.index, prices['ItemPrice'], label='Actual Prices')
    plt.plot(next_dates, unscaled_predictions, label='Predicted Prices', color='r')
    plt.xlabel('Date')
    plt.ylabel('Item Price')
    plt.title(f'Price Prediction for Item {item_code}')
    plt.legend()
    plt.tight_layout()
    plt.savefig('../F-E/static/plots/predicted_prices_bar_chart.png')
    # Display the plot


    # Display the bar chart for predicted prices
    plt.figure(figsize=(8, 5))
    plt.bar(next_dates, unscaled_predictions, color='blue', alpha=0.7)
    plt.xlabel('Date')
    plt.ylabel('Predicted Price')
    plt.title('Predicted Prices for the next 7 days')
    plt.xticks(rotation=45, color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    plt.savefig('../F-E/static/plots/predicted_prices_bar_chart.png')
    # Display the bar chart



def prediction(itemcode):
    query = {
        'ItemCode': itemcode,
        'PriceUpdateDate': {"$gt": '2022-01-01'}
    }

    data = list(collection.find(query))
    df = pd.DataFrame(data)
    pricePrediction(df)
    return('/static/plots/predicted_prices_bar_chart.png')

