import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import unicodedata
import nltk
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt

nltk.download('vader_lexicon')

# Load the data
df_stocks = pd.read_pickle(r"C:\\Users\\lautr\\Downloads\\pickled_ten_year_filtered_lead_para (1).pkl")
df_stocks['prices'] = df_stocks['adj close'].apply(np.int64)

# Selecting the prices and articles
df_stocks = df_stocks[['prices', 'articles']]
df_stocks['articles'] = df_stocks['articles'].map(lambda x: x.lstrip('.-'))

# Sentiment analysis
sid = SentimentIntensityAnalyzer()
df_stocks['compound'] = 0.0
df_stocks['neg'] = 0.0
df_stocks['neu'] = 0.0
df_stocks['pos'] = 0.0

for date, row in df_stocks.iterrows():
    try:
        sentence = unicodedata.normalize('NFKD', row['articles']).encode('ascii', 'ignore').decode('utf-8')
        ss = sid.polarity_scores(sentence)
        df_stocks.at[date, 'compound'] = ss['compound']
        df_stocks.at[date, 'neg'] = ss['neg']
        df_stocks.at[date, 'neu'] = ss['neu']
        df_stocks.at[date, 'pos'] = ss['pos']
    except TypeError as e:
        print(row['articles'])
        print(date)
        print(e)

# Selecting prices and compound sentiment scores
features = df_stocks[['prices', 'compound']].values

# Scaling features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# Create sequences of data for the LSTM model
def create_sequences(data, time_step=100):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)

time_step = 100
X, y = create_sequences(scaled_features, time_step)

# Increase the model complexity
model = Sequential()
model.add(LSTM(units = 200, return_sequences = True, input_shape = (X.shape[1], X.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units = 200, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 200))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Use cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for train_index, test_index in tscv.split(X):
    X_train, X_val = X[train_index], X[test_index]
    Y_train, Y_val = y[train_index], y[test_index]
    history = model.fit(x = X_train, y = Y_train, epochs = 100, batch_size = 64, validation_data=(X_val, Y_val))

# Use the model to make predictions
predictions = model.predict(X)

# Scale the predictions back to their original range
predictions = scaler.inverse_transform(predictions)

# Create a DataFrame to compare the original data, fitted result and the error of that time point entry
comparison = pd.DataFrame({'Real DJIA': y.flatten(), 'Predicted DJIA': predictions.flatten()})
comparison['Error'] = comparison['Real DJIA'] - comparison['Predicted DJIA']

# Compute the standard deviation of the data
comparison['Standard Deviation'] = comparison.std(axis=1)

# Compute the Mean Absolute Error (MAE) and Mean Squared Error (MSE) of the data
mae = mean_absolute_error(comparison['Real DJIA'], comparison['Predicted DJIA'])
mse = mean_squared_error(comparison['Real DJIA'], comparison['Predicted DJIA'])

# Compute the Root Mean Squared Error (RMSE)
rmse = sqrt(mse)

# Compute the Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(comparison['Real DJIA'], comparison['Predicted DJIA'])

# Compute the R-squared and adjusted R-squared
r2 = r2_score(comparison['Real DJIA'], comparison['Predicted DJIA'])
adjusted_r2 = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)

# Compute the variance
variance = np.var(comparison['Error'])

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Percentage Error: {mape}")
print(f"R-squared: {r2}")
print(f"Adjusted R-squared: {adjusted_r2}")
print(f"Variance: {variance}")

# Plot the loss and validation loss for each process
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.show()

# Plot the predicted vs real data for the forecast data with time
plt.figure(figsize=(10, 6))
plt.plot(comparison['Real DJIA'], label='Real DJIA')
plt.plot(comparison['Predicted DJIA'], label='Predicted DJIA')
plt.title('Predicted vs Real DJIA')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend(loc='upper right')
plt.show()

# Print the comparison table
print(comparison)