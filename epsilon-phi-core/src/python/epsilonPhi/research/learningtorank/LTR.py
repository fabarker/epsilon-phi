import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers

from epsilonPhi.core.dataModel.dataSources.GlobalDataSource import GlobalDataSource

# Generate synthetic data
np.random.seed(42)
T = 100  # Number of historical periods
N = 50  # Number of assets

# Generate synthetic past returns with a reversal effect
past_returns = np.random.randn(T, N)
# Introducing a reversal effect: if past return > 0.1, future return is negative
future_returns = np.where(past_returns[:-1, :] > 0.1, -np.abs(np.random.randn(T-1, N)), np.random.randn(T-1, N))

# Prepare listwise data
def create_listwise_data(past_returns, future_returns, prediction_horizon):
    X_list = []
    y_list = []
    for t in range(past_returns.shape[0] - prediction_horizon):
        X_t = past_returns[t].reshape(1, -1)  # Past returns at time t
        y_t = future_returns[t:t+prediction_horizon].mean(axis=0)  # Mean future return
        X_list.append(X_t)
        y_list.append(np.argsort(np.argsort(y_t)))  # Convert to ranks
    return np.vstack(X_list), np.vstack(y_list)

# Parameters
prediction_horizon = 1
X, y = create_listwise_data(past_returns, future_returns, prediction_horizon)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Listwise LTR model (ListNet example)
input_shape = (N,)
inputs = layers.Input(shape=input_shape)
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dense(32, activation='relu')(x)
x = layers.Dense(N, activation='softmax')(x)

model = tf.keras.Model(inputs, x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.2)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Predict ranks
predictions = model.predict(X_test)
print(predictions)