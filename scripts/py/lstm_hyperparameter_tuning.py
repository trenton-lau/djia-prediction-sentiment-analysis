from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.optimizers import Adam, SGD, RMSprop

# Define a function to create the LSTM model
def create_model(optimizer='adam', dropout_rate=0.2, units=50, nb_layers=1):
    model = Sequential()
    for i in range(nb_layers):
        if i == 0:
            model.add(LSTM(units=units, return_sequences=True, input_shape=(x_train.shape[1], 6)))
        else:
            model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units=1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

# Create a KerasRegressor
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define the grid search parameters
batch_size = [10, 20, 40, 60, 80, 100]
epochs = [10, 50, 100]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
dropout_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
units = [10, 20, 30, 40, 50]
nb_layers = [1, 2, 3]
param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, dropout_rate=dropout_rate, units=units, nb_layers=nb_layers)

# Create Grid Search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(x_train, y_train)

# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stdds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
