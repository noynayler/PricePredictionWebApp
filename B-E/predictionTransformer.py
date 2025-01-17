import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

def prediction(ItemCode):
    def get_season(date):
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
    db = client['shufersal_data']
    collection = db['combined_DB']

    # Fetch data from MongoDB
    query = {
        'ItemCode': ItemCode,
        'PriceUpdateDate': {"$gt": '2023-01-01'}
    }
    data = list(collection.find(query))
    df_db = pd.DataFrame(data)
    df_db = df_db.sort_values(by='PriceUpdateDate')

    ItemCode = df_db['ItemCode'][0]
    # Parse dates and convert prices
    df_db['PriceUpdateDate'] = pd.to_datetime(df_db['PriceUpdateDate'], errors='coerce')
    df_db['ItemPrice'] = pd.to_numeric(df_db['ItemPrice'], errors='coerce')

    # Drop rows with NaN values in 'PriceUpdateDate' or 'ItemPrice'
    df_db = df_db.dropna(subset=['PriceUpdateDate', 'ItemPrice'])
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df_db['ItemPrice'].quantile(0.25)
    Q3 = df_db['ItemPrice'].quantile(0.75)

    # Calculate IQR
    IQR = Q3 - Q1

    # Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the outliers
    df_db = df_db[(df_db['ItemPrice'] >= lower_bound) & (df_db['ItemPrice'] <= upper_bound)]
    #print(df_db['ItemPrice'])
    # Drop unnecessary columns
    df_db = df_db.drop(['_id', 'Vegetable'], axis=1)

    # Add season feature to the dataframe
    df_db['Season'] = df_db['PriceUpdateDate'].apply(get_season)

    # One-hot encode the season feature
    df_db = pd.get_dummies(df_db, columns=['Season'])

    # Prepare the data for Prophet
    df_prophet = df_db.rename(columns={'PriceUpdateDate': 'ds', 'ItemPrice': 'y'})

    # Ensure all season columns are present in the df_prophet dataframe
    for season in ['Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']:
        if season not in df_prophet.columns:
            df_prophet[season] = 0

    # Drop any remaining NaN values in the 'ds' column after processing
    df_prophet = df_prophet.dropna(subset=['ds'])
    df_prophet = df_prophet.sort_values(by='ds')

    # Define the Prophet model with possible tuning
    model = Prophet(seasonality_mode='multiplicative', changepoint_prior_scale=0.8)

    for season in ['Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']:
        model.add_regressor(season)

    # Fit the model to the data
    model.fit(df_prophet)

    # Create a dataframe for future dates
    future = model.make_future_dataframe(periods=7)  # Predict 7 days into the future

    # Add the season features to the future dataframe
    future['Season'] = future['ds'].apply(get_season)
    future = pd.get_dummies(future, columns=['Season'])

    # Ensure all season columns are present in the future dataframe
    for season in ['Season_Winter', 'Season_Spring', 'Season_Summer', 'Season_Autumn']:
        if season not in future.columns:
            future[season] = 0

    # Predict future prices
    forecast = model.predict(future)
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']])
    forecast['yhat'] = forecast['yhat'].clip(lower=0)
    forecast['yhat_smoothed'] = forecast['yhat'].rolling(window=3).mean()
    forecast['yhat_smoothed'] = forecast['yhat_smoothed'].fillna(forecast['yhat'])  # Handle the NaNs by filling them with original predictions

    # Filter the forecast to include only future dates
    forecast_future = forecast[forecast['ds'] > df_prophet['ds'].max()]

    # Performance Evaluation on Historical Data
    y_true = df_prophet['y'].values
    y_pred = forecast['yhat'][:len(y_true)].values

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"R-Squared (R2): {r2}")

    # Plot the original values and predictions
    plt.figure(figsize=(10, 5))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual Prices', color='blue')
    plt.plot(forecast_future['ds'], forecast_future['yhat'], label='Predicted Prices', color='red')
    plt.xlabel('Date')
    plt.ylabel('Item Price')
    plt.title(f'Price Prediction for Item {ItemCode}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'../F-E/static/plots/price_prediction.png')
    # plt.show()

    # Conclusion and advice
    average_predicted_price = forecast_future['yhat'].mean()
    last_known_price = df_prophet['y'].iloc[-1]

    if forecast_future['yhat'].iloc[-1] > last_known_price:
        price_trend = "up"
    elif forecast_future['yhat'].iloc[-1] < last_known_price:
        price_trend = "down"
    else:
        price_trend = "no_change"


    conclusion = (
        f"המחיר הממוצע החזוי לשבעת הימים הקרובים הוא {average_predicted_price:.2f}.\n"
        f"המחיר צפוי {'לעלות' if price_trend == 'up' else 'לרדת' if price_trend == 'down' else 'להיוותר ללא שינוי'} בהשוואה למחיר האחרון שהיה {last_known_price:.2f}.\n"
        f"{'כדאי לשקול לרכוש עכשיו לפני שהמחיר יעלה.' if price_trend == 'up' else 'כדאי לשקול להמתין מכיוון שהמחיר צפוי לרדת.' if price_trend == 'down' else ''}"    )

    #print(conclusion)

    return (None, '/static/plots/price_prediction.png', conclusion)

# Usage
# start_time = time.time()
#
# #pricePrediction("P_7290000965031")
# #pricePrediction("P_7290000000022")
# #pricePrediction("P_7296073440369")
# end_time = time.time()
# print(f"Total time: {end_time - start_time} seconds")

# Connect to MongoDB

# start_time = time.time()
# pricePrediction('P_7296073440314')
# end_time = time.time()
# print(f"Total time: {end_time - start_time} seconds")


# import os
# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from pymongo import MongoClient
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
#
# pd.set_option('display.max_columns', None)
# pd.set_option('display.width', None)
# pd.set_option('display.max_colwidth', None)
#
# def prediction(ItemCode ):
#     def get_season(date):
#         month = date.month
#         if month in [12, 1, 2]:
#             return 'Winter'
#         elif month in [3, 4, 5]:
#             return 'Spring'
#         elif month in [6, 7, 8]:
#             return 'Summer'
#         else:
#             return 'Autumn'
#
#     # Connect to MongoDB
#     client = MongoClient('mongodb+srv://admin:9491263@cluster0.nwcjgox.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
#     db = client['shufersal_data']
#     collection = db['combined_DB']
#
#     # Fetch data from MongoDB
#     query = {
#         'ItemCode': ItemCode,
#         'PriceUpdateDate': {"$gt": '2023-01-01'}
#     }
#     data = list(collection.find(query))
#     df_db = pd.DataFrame(data)
#
#     def parse_date(date_str):
#         try:
#             return pd.to_datetime(date_str, format='%Y-%m-%d %H:%M:%S', errors='raise')
#         except (ValueError, TypeError):
#             try:
#                 return pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
#             except (ValueError, TypeError):
#                 return pd.NaT
#
#     # Apply the custom function to convert dates
#     df_db['PriceUpdateDate'] = df_db['PriceUpdateDate'].apply(parse_date)
#
#     # Extract only the date part
#     df_db['PriceUpdateDate'] = df_db['PriceUpdateDate'].dt.date
#
#     df_db['ItemPrice'] = pd.to_numeric(df_db['ItemPrice'], errors='coerce')
#     df_db = df_db.drop('_id', axis=1)
#     df_db = df_db.drop('Vegetable', axis=1)
#
#     # Add season feature to the dataframe
#     df_db['Season'] = df_db['PriceUpdateDate'].apply(get_season)
#     # One-hot encode the season feature
#     one_hot_encoder = OneHotEncoder()
#     season_encoded = one_hot_encoder.fit_transform(df_db[['Season']]).toarray()
#     season_columns = one_hot_encoder.get_feature_names_out(['Season'])
#     df_season_encoded = pd.DataFrame(season_encoded, columns=season_columns, index=df_db.index)
#
#     df = pd.concat([df_db, df_season_encoded], axis=1)
#
#     print(df)
#
#     numOfEpoch = 100
#     prediction_days = 7
#     item_code = df['ItemCode'].iloc[0]
#     df['PriceUpdateDate'] = pd.to_datetime(df['PriceUpdateDate'])
#     df_item = df.sort_values(by='PriceUpdateDate').copy()
#
#     prices = df_item[['PriceUpdateDate', 'ItemPrice']].copy()
#     prices.set_index('PriceUpdateDate', inplace=True)
#
#     # Scale the data
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_prices = scaler.fit_transform(prices[['ItemPrice']])
#     prices['ScaledPrice'] = scaled_prices
#
#     # Prepare sequences
#     def create_sequences(data, seq_length, season_data):
#         xs = []
#         ys = []
#         for i in range(len(data) - seq_length):
#             x = np.hstack([data[i:i+seq_length].reshape(-1, 1), season_data[i:i+seq_length]])
#             y = data[i+seq_length]
#             xs.append(x)
#             ys.append(y)
#         return np.array(xs), np.array(ys)
#
#     seq_length = 5
#     season_features = df_item[season_columns].values
#     X, y = create_sequences(prices['ScaledPrice'].values, seq_length, season_features)
#
#     # Convert to PyTorch tensors
#     X = torch.from_numpy(X).float().view(-1, seq_length, X.shape[2])
#     y = torch.from_numpy(y).float().unsqueeze(-1)
#
#     # Split into training and validation sets
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
#
#     # Define the Transformer model with the best parameters
#     class TransformerModel(nn.Module):
#         def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
#             super(TransformerModel, self).__init__()
#             self.input_embedding = nn.Linear(input_dim, model_dim)
#             self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, model_dim))
#             self.encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout)
#             self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
#             self.fc_out = nn.Linear(model_dim, output_dim)
#
#         def forward(self, src):
#             src = self.input_embedding(src) + self.positional_encoding
#             src = src.permute(1, 0, 2)  # Transformer expects [seq_len, batch_size, model_dim]
#             output = self.transformer_encoder(src)
#             output = output.permute(1, 0, 2)  # Revert to [batch_size, seq_len, model_dim]
#             return self.fc_out(output[:, -1, :])
#
#     model = TransformerModel(input_dim=X.shape[2], model_dim=128, num_heads=4, num_layers=2, output_dim=1, dropout=0.1)
#     loss_function = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
#     # Train the model
#     epochs = numOfEpoch
#     for epoch in range(epochs):
#         model.train()
#         for seq, labels in zip(X_train, y_train):
#             optimizer.zero_grad()
#             y_pred = model(seq)
#             loss = loss_function(y_pred, labels)
#             loss.backward()
#             optimizer.step()
#
#         # Validate the model
#         model.eval()
#         val_losses = []
#         y_true = []
#         y_pred_list = []
#         with torch.no_grad():
#             for seq, labels in zip(X_val, y_val):
#                 val_pred = model(seq)
#                 val_loss = loss_function(val_pred, labels)
#                 val_losses.append(val_loss.item())
#
#                 y_true.append(labels.item())
#                 y_pred_list.append(val_pred.item())
#
#         if epoch % 25 == 0:
#             print(f'epoch: {epoch:3} train_loss: {loss.item():10.8f} val_loss: {np.mean(val_losses):10.8f}')
#
#     # Calculate error metrics
#     y_true = np.array(y_true).reshape(-1, 1)
#     y_pred_list = np.array(y_pred_list).reshape(-1, 1)
#
#     y_true_unscaled = scaler.inverse_transform(y_true)
#     y_pred_unscaled = scaler.inverse_transform(y_pred_list)
#
#     mae = mean_absolute_error(y_true_unscaled, y_pred_unscaled)
#     mse = mean_squared_error(y_true_unscaled, y_pred_unscaled)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_true_unscaled, y_pred_unscaled)
#
#     print(f'Mean Absolute Error (MAE): {mae}')
#     print(f'Mean Squared Error (MSE): {mse}')
#     print(f'Root Mean Squared Error (RMSE): {rmse}')
#     print(f'R-Squared (R2): {r2}')
#
#     # Prediction vs Actual Plot
#     plt.figure(figsize=(10, 5))
#     plt.scatter(y_true_unscaled, y_pred_unscaled, label='Predicted vs Actual')
#     plt.xlabel('Actual Prices')
#     plt.ylabel('Predicted Prices')
#     plt.title('Prediction vs Actual Prices')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('../F-E/static/plots/predicted_vs_actual.png')
#
#     # Residual Plot
#     residuals = y_true_unscaled - y_pred_unscaled
#     plt.figure(figsize=(10, 5))
#     plt.scatter(y_pred_unscaled, residuals, label='Residuals')
#     plt.axhline(0, color='red', linestyle='--')
#     plt.xlabel('Predicted Prices')
#     plt.ylabel('Residuals')
#     plt.title('Residuals Plot')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('../F-E/static/plots/residuals.png')
#
#     # Make predictions for the next specified number of days
#     model.eval()
#     last_seq = torch.FloatTensor(X[-1]).unsqueeze(0).view(1, seq_length, X.shape[2])
#     predictions = []
#
#     for _ in range(prediction_days):
#         with torch.no_grad():
#             pred = model(last_seq)
#             predictions.append(pred.item())
#             pred = pred.view(1, 1, -1)
#             new_seq = torch.cat((last_seq[:, 1:, :], torch.cat([pred] * (last_seq.shape[2] // pred.shape[2]), dim=2)), dim=1)
#             last_seq = new_seq
#
#     # Inverse scale the predictions
#     scaled_predictions = np.array(predictions).reshape(-1, 1)
#     unscaled_predictions = scaler.inverse_transform(scaled_predictions).flatten()
#
#     # Extend dates for prediction
#     last_date = prices.index[-1]
#     next_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=prediction_days, freq='D')
#
#
#
#     # Create plot for predictions
#     plt.figure(figsize=(10, 5))
#     plt.plot(prices.index, prices['ItemPrice'], label='Actual Prices')
#     plt.plot(next_dates, unscaled_predictions, label='Predicted Prices', color='r')
#     plt.xlabel('Date')
#     plt.ylabel('Item Price')
#     plt.title(f'Price Prediction for Item {item_code}')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('../F-E/static/plots/price_prediction.png')
#
#     # Display the bar chart for predicted prices
#     plt.figure(figsize=(10, 5))
#     plt.bar(next_dates, unscaled_predictions, color='blue', alpha=0.7)
#     plt.xlabel('Date')
#     plt.ylabel('Predicted Price')
#     plt.title(f'Predicted Prices for Next 7 Days')
#     plt.xticks(rotation=45)
#     plt.tight_layout()
#     plt.savefig('../F-E/static/plots/predicted_prices_bar_chart.png')
#
#     # Conclusion and advice
#     average_predicted_price = np.mean(unscaled_predictions)
#     last_known_price = df_db['ItemPrice'].iloc[-1]
#     price_trend = "up" if unscaled_predictions[-1] > last_known_price else "down"
#
#     conclusion = (
#         f"המחיר הממוצע החזוי לשבעת הימים הקרובים הוא {average_predicted_price:.2f}.\n"
#         f"המחיר צפוי {'לעלות' if price_trend == 'up' else 'לרדת'} בהשוואה למחיר האחרון שהיה {last_known_price:.2f}.\n"
#         f"{'כדאי לשקול לרכוש עכשיו לפני שהמחיר יעלה.' if price_trend == 'up' else 'כדאי לשקול להמתין מכיוון שהמחיר צפוי לרדת.'}"
#     )
#
#
#     print(conclusion)
#
#     return ('/static/plots/predicted_prices_bar_chart.png', '/static/plots/price_prediction.png',conclusion)
#
# #pricePrediction("P_7296073440314")