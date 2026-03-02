import h5py
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import json
from sklearn.model_selection import train_test_split

DATA_DIR = "datasets"

def get_data():
    with h5py.File(f'{DATA_DIR}/case_processed.h5', 'r') as f:
        x = f['x'][:]
        y = f['y'][:]
    return x, y

def get_model(time_steps, features, outputs):
    model = Sequential([
        LSTM(64, input_shape=(time_steps, features)),
        Dropout(0.4),
        Dense(outputs)
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


if __name__ == "__main__":
    X, Y = get_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)
    model = get_model(100, 3, 2)
    model.fit(X_train, Y_train, epochs=100, batch_size=100)
    loss, mae = model.evaluate(X_test, Y_test)
    y_pred = model.predict(X_test)
    print(y_pred.shape)
    print(Y_test[-1].shape)
    print(f"Loss: {loss}, MAE: {mae}")
    output = [{"y_test": Y_test[i].tolist(), "y_pred": y_pred[i].tolist()} for i in range(len(y_pred))]
    with open('output.json', 'w') as f:
        json.dump(output, f)