import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# Load the data
data = pd.read_csv('C:\\Users\\lautr\\Downloads\\DJIA_data.csv')

# Convert the date to a suitable format
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')

# Normalize the data
scaler = MinMaxScaler()
data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']] = scaler.fit_transform(data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']])

# Split the data into training and testing sets
train_data = data[data['Date'] < '2016-01-01']
test_data = data[data['Date'] >= '2016-01-01']

# Define a function to create a sliding window of input data and output labels
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# Define the time step for the sliding window
time_steps = 10

# Reshape the training and testing data to include the time step dimension
x_train, y_train = create_dataset(train_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']], train_data['Close'], time_steps)
x_test, y_test = create_dataset(test_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']], test_data['Close'], time_steps)

# Define and train the LSTM model
model = Sequential()
model.add(LSTM(units = 200, return_sequences = True, input_shape = (x_train.shape[1], 6)))
model.add(Dropout(0.2))
model.add(LSTM(units = 200, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 200))
model.add(Dropout(0.2))
model.add(Dense(units = 1))

model.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Add validation data and monitor the validation loss
history = model.fit(x = x_train, y = y_train, epochs = 100, batch_size = 32, validation_data=(x_test, y_test))

# Use the model to make predictions and plot the results
predictions = model.predict(x_test)
plt.plot(range(len(y_test)), y_test, color = 'blue', label = 'Real DJIA')
plt.plot(range(len(predictions)), predictions, color = 'red', label = 'Predicted DJIA')
plt.title('DJIA Prediction')
plt.xlabel('Time')
plt.ylabel('DJIA')
plt.legend()
plt.show()
