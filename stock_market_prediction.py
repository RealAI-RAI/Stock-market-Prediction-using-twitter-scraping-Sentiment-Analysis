# -*- coding: utf-8 -*-

Original file is located at
    https://colab.research.google.com/drive/1UgILTWHFh2wjfwwTpmjaIiplBh5jE6ui

"""# LSTM Model

This code trains a **convolutional LSTM** neural network on historical stock prices, using data from a CSV file that contains combined data from Twitter sentiment analysis and Toronto Stock Exchange data. The data is preprocessed by splitting it into training and testing sets, scaling the data, and creating sequences of data to feed into the model. The model includes several layers of Conv1D and LSTM layers, and is trained using mean squared error as the loss function. The code then evaluates the model and calculates metrics such as mean squared error, mean absolute error, and R-squared, and plots the predictions and actual values.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
df=pd.read_csv("/content/combine_twitter_tsx .csv")
# Scrape London Stock Exchange data
df.dropna(inplace=True)

# Use only the closing prices
prices = df['Close'].values.reshape(-1, 1)

# Split data into training and testing sets
training_size = int(len(prices) * 0.8)
train_data = prices[:training_size]
test_data = prices[training_size:]

# Scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define function to create training and testing sequences
def create_sequences(data, time_steps):
    X = []
    y = []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Set time steps and number of features
time_steps = 30
num_features = 1

# Create the train and test sequences
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Reshape the data for the CNN
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))

# Define model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, activation='relu', input_shape=(time_steps, num_features)))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(LSTM(500, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(400, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(200, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(100, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(50, return_sequences=False))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
model.fit(X_train, y_train, epochs=500, batch_size=64, verbose=2)

# Evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Invert scaling of predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# Plot predictions and actual values
import matplotlib.pyplot as plt

plt.plot(predictions, label='Predictions')
plt.plot(actual_values, label='Actual Values')
plt.legend()
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Calculate evaluation metrics
mse = mean_squared_error(actual_values, predictions)
mae = mean_absolute_error(actual_values, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(actual_values, predictions)

# Print evaluation metrics
print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('Root Mean Squared Error (RMSE):', rmse)
print('R-squared (R2):', r2)

"""# Transfermor Model

This code is a TensorFlow implementation of a time series prediction model using a multi-head attention layer. The code loads data from a CSV file, splits it into training and testing sets, scales the data using a MinMaxScaler, and creates training and testing sequences using a function. The model architecture includes an input layer, a normalization layer, a multi-head attention layer, two dropout layers, a dense layer, and an output layer. The model is compiled with the mean squared error loss function and trained using early stopping. The code evaluates the model's performance using mean squared error, mean absolute error, root mean squared error, and r-squared metrics. Finally, the code plots the predicted values against the actual values.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv("/content/combine_twitter_tsx .csv")
df.dropna(inplace=True)
prices = df['Close'].values.reshape(-1, 1)

# Split data into training and testing sets
training_size = int(len(prices) * 0.8)
train_data = prices[:training_size]
test_data = prices[training_size:]

# Scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define function to create training and testing sequences
def create_sequences(data, time_steps):
    X = []
    y = []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Set time steps and number of features
time_steps = 30
num_features = 1

# Create the train and test sequences
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Define model
input_layer = Input(shape=(time_steps, num_features))
x = Normalization()(input_layer)
x = LayerNormalization()(x)
x = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=1)(x, x)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
output_layer = Dense(1)(x)
model = Model(inputs=[input_layer], outputs=[output_layer])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])

# Evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Invert scaling of predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# Plot predictions and actual values
import matplotlib.pyplot as plt

plt.plot(predictions, label='Predictions')
plt.plot(actual_values, label='Actual Values')
plt.legend()
plt.show()

"""#  Transformer Hugging face"""

!pip install --upgrade transformers

"""This code is an example of how to create a transformer-based neural network for time-series forecasting using the PyTorch library.

The code loads time-series data from a CSV file, splits it into training and testing sets, scales the data, and creates sequences of data with a specified number of time steps. Then, it defines a transformer-based neural network model with one transformer encoder layer, a dropout layer, and two dense layers. The model is compiled using the mean squared error as the loss function and the Adam optimizer.

The model is trained on the training data, with early stopping to prevent overfitting. The performance of the model is evaluated on the testing data using several metrics such as mean squared error, mean absolute error, root mean squared error, and R-squared. Finally, the predictions of the model and the actual values are plotted using matplotlib.
"""

from transformers import TransformerEncoder
import torch

# Load data
df = pd.read_csv("/content/combine_twitter_tsx .csv")
df.dropna(inplace=True)
prices = df['Close'].values.reshape(-1, 1)

# Split data into training and testing sets
training_size = int(len(prices) * 0.8)
train_data = prices[:training_size]
test_data = prices[training_size:]

# Scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define function to create training and testing sequences
def create_sequences(data, time_steps):
    X = []
    y = []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Set time steps and number of features
time_steps = 30
num_features = 1

# Create the train and test sequences
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Define model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
transformer_layer = TransformerEncoderLayer(d_model=num_features, nhead=8)
transformer_encoder = TransformerEncoder(transformer_layer, num_layers=1)
input_layer = layers.Input(shape=(time_steps, num_features))
x = Normalization()(input_layer)
x = LayerNormalization()(x)
x = layers.Lambda(lambda x: x.permute(0, 2, 1))(x)
x = layers.Reshape((-1, num_features))(x)
x = transformer_encoder(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dropout(0.2)(x)
output_layer = layers.Dense(1)(x)
model = Model(inputs=[input_layer], outputs=[output_layer])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train model
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2, validation_split=0.2, callbacks=[es])

# Evaluate model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("Mean Squared Error (MSE):", mse)
print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
print("R-squared (R2):", r2)

# Invert scaling of predictions and actual values
predictions = scaler.inverse_transform(predictions)
actual_values = scaler.inverse_transform(y_test)

# Plot predictions and actual values
import matplotlib.pyplot as plt

plt.plot(predictions, label='Predictions')
plt.plot(actual_values, label='Actual Values')
plt.legend()
plt.show()

"""# Transformer model using the Keras API

transformer model using the Keras API. This model consists of multiple transformer blocks, each with a specified number of attention heads and feedforward dimensions. The transformer blocks are then flattened and passed through a dense layer to output a single prediction. The model takes as input a sequence of integers with a specified input shape.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, MultiHeadAttention, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Read data
df = pd.read_csv("/content/combine_twitter_tsx .csv")

# Drop rows with missing values
df.dropna(inplace=True)

# Use only the closing prices
prices = df['Close'].values.reshape(-1, 1)

# Split data into training and testing sets
training_size = int(len(prices) * 0.8)
train_data = prices[:training_size]
test_data = prices[training_size:]

# Scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Define function to create training and testing sequences
def create_sequences(data, time_steps):
    X = []
    y = []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i])
    X = np.array(X)
    y = np.array(y)
    return X, y

# Set time steps
time_steps = 30

# Create the train and test sequences
X_train, y_train = create_sequences(train_data, time_steps)
X_test, y_test = create_sequences(test_data, time_steps)

# Reshape the data for the LSTM and Transformer models
X_train_lstm = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
X_train_trans = X_train.reshape((X_train.shape[0], X_train.shape[1]))
X_test_trans = X_test.reshape((X_test.shape[0], X_test.shape[1]))

# Define LSTM model
model_lstm = Sequential()
model_lstm.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(time_steps, 1)))
model_lstm.add(MaxPooling1D(pool_size=2))
model_lstm.add(Dropout(0.2))
model_lstm.add(Conv1D(filters=32, kernel_size=3, activation='sigmoid'))
model_lstm.add(MaxPooling1D(pool_size=2))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(100, return_sequences=True))
model_lstm.add(Dropout(0.2))
model_lstm.add(LSTM(50, return_sequences=False))
model_lstm.add(Dropout(0.3))
model_lstm.add(Dense(50, activation='relu'))
model_lstm.add(Dense(1))

# Compile LSTM model
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

# Define Transformer model
def transformer_model(input_shape):
    inputs = Input(shape=(input_shape,))
    embedding_layer = Embedding(input_dim=input_shape, output_dim=64)(inputs)
    transformer_block = Transformer(num_heads=8, ff_dim=64, dropout=0.2, name='transformer_block_1')(embedding_layer)
    transformer_block = Transformer(num_heads=8, ff_dim=64, dropout=0.2, name='transformer_block_2')(transformer_block)
    transformer_block = Transformer(num_heads=8, ff_dim=64, dropout=0.2, name='transformer_block_3')(transformer_block)
    transformer_block = Flatten()(transformer_block)
    outputs = Dense(1)(transformer_block)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='transformer')
    return model

history_lstm = model_lstm.fit(X_train_lstm, y_train, batch_size=32, epochs=50, validation_data=(X_test_lstm, y_test), callbacks=[EarlyStopping(patience=10)])

score_lstm = model_lstm.evaluate(X_test_lstm, y_test)
print('LSTM Test Loss:', score_lstm)

model_trans = transformer_model(X_train_trans.shape[1])
model_trans.compile(optimizer='adam', loss='mean_squared_error')
history_trans = model_trans.fit(X_train_trans, y_train, batch_size=32, epochs=50, validation_data=(X_test_trans, y_test), callbacks=[EarlyStopping(patience=10)])
score_trans = model_trans.evaluate(X_test_trans, y_test)
print('Transformer Test Loss:', score_trans)

"""# Generative Adversarial Network (GAN)

**Generative Adversarial Network (GAN)** to generate synthetic data that resembles real stock market closing prices. The code loads a CSV file containing historical stock market closing prices and preprocesses it by splitting it into training and testing sets and scaling it using MinMaxScaler. It then defines three models: a generator model, a discriminator model, and a GAN model that combines the generator and discriminator models. The generator model takes random noise as input and generates synthetic data that resembles the real stock market closing prices. The discriminator model takes real and synthetic data as input and predicts whether the data is real or fake. The GAN model trains the generator model to produce synthetic data that can fool the discriminator model. The code then trains the GAN model by alternating between training the discriminator model on real and fake data and training the generator model to produce synthetic data that can fool the discriminator model. Finally, it prints the mean squared error **(MSE)**, mean absolute error (**MAE**), root mean squared error **(RMSE)**, and **R2** score between the real and synthetic test data.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

# Load data
df = pd.read_csv("/content/combine_twitter_tsx .csv")
df.dropna(inplace=True)
prices = df['Close'].values.reshape(-1, 1)

# Split data into training and testing sets
training_size = int(len(prices) * 0.8)
train_data = prices[:training_size]
test_data = prices[training_size:]

# Scale data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)

# Set parameters
noise_dim = 100
batch_size = 32
epochs = 100
# Define generator model
generator = Sequential()
generator.add(Dense(128, activation='relu', input_dim=noise_dim))
generator.add(Dense(256, activation='relu'))
generator.add(Dense(512, activation='relu'))
generator.add(Dense(train_data.shape[1], activation='sigmoid'))
generator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Define discriminator model
discriminator = Sequential()
discriminator.add(Dense(512, activation='relu', input_dim=train_data.shape[1]))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(256, activation='relu'))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(128, activation='relu'))
discriminator.add(Dropout(0.3))
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

# Define GAN model
discriminator.trainable = False
gan_input = Input(shape=(noise_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)
gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

# Define function to create training batches
def get_batches(data, batch_size):
    batches = []
    num_batches = len(data) // batch_size
    for i in range(num_batches):
        batch = data[i*batch_size:(i+1)*batch_size]
        batches.append(batch)
    return np.array(batches)

# Define function to generate noise samples
def generate_noise_samples(num_samples, noise_dim):
    return np.random.normal(0, 1, size=[num_samples, noise_dim])

# Train GAN model
for epoch in range(epochs):
    # Train discriminator on real data
    real_batches = get_batches(train_data, batch_size)
    for batch in real_batches:
        noise = generate_noise_samples(batch_size, noise_dim)
        fake_data = generator.predict(noise)
        discriminator.train_on_batch(batch, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

    # Train generator by fooling discriminator
    noise = generate_noise_samples(batch_size, noise_dim)
    gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print loss
    if epoch % 100 == 0:
        noise = generate_noise_samples(len(test_data), noise_dim)
        generated_data = generator.predict(noise)
        generated_data = scaler.inverse_transform(generated_data)
        test_data = scaler.inverse_transform(test_data)
        mse = mean_squared_error(test_data, generated_data)
        mae = mean_absolute_error(test_data, generated_data)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_data, generated_data)
        print("Epoch:", epoch, "- MSE:", mse, "- MAE:", mae, "- RMSE:", rmse, "- R2:", r2)

# Train GAN model
for epoch in range(epochs):
    # Train discriminator on real data
    real_batches = get_batches(train_data, batch_size)
    for batch in real_batches:
        noise = generate_noise_samples(batch_size, noise_dim)
        fake_data = generator.predict(noise)
        discriminator.train_on_batch(batch, np.ones((batch_size, 1)))
        discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))

    # Train generator by fooling discriminator
    noise = generate_noise_samples(batch_size, noise_dim)
    gan.train_on_batch(noise, np.ones((batch_size, 1)))

    # Print loss
    if epoch % 100 == 0:
        noise = generate_noise_samples(len(test_data), noise_dim)
        generated_data = generator.predict(noise)
        generated_data = scaler.inverse_transform(generated_data)
        test_data_unscaled = scaler.inverse_transform(test_data)
        mse = mean_squared_error(test_data_unscaled, generated_data)
        mae = mean_absolute_error(test_data_unscaled, generated_data)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_data_unscaled, generated_data)
        print("Epoch:", epoch, "\tMSE:", mse, "\tMAE:", mae, "\tRMSE:", rmse, "\tR2 Score:", r2)

# Invert scaling of generated data
noise = generate_noise_samples(len(test_data), noise_dim)
generated_data = generator.predict(noise)
generated_data = scaler.inverse_transform(generated_data)

# Plot generated data
plt.plot(generated_data, label='Generated Data')
plt.plot(test_data_unscaled, label='Actual Data')
plt.legend()
plt.show()

discriminator.summary()
fig.

generator.summary()

with open('discriminator_summary.txt', 'w') as f:
    discriminator.summary(print_fn=lambda x: f.write(x + '\n'))

with open('generator_summary.txt', 'w') as f:
    generator.summary(print_fn=lambda x: f.write(x + '\n'))

gan.summary()



with open('gan_summary.txt', 'w') as f:
    gan.summary(print_fn=lambda x: f.write(x + '\n'))
