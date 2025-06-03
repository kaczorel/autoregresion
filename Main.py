import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.ar_model import AutoReg
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings


warnings.filterwarnings('ignore')

file_path = "alta_to_2025-05-11_akcje.xls"
try:
    df = pd.read_excel(file_path)
except FileNotFoundError:
    print(f"Błąd: Plik '{file_path}' nie został znaleziony.")
    exit()

df['Data'] = pd.to_datetime(df['Data'])
selected_quotation_column = 'Kurs zamknięcia'

if selected_quotation_column not in df.columns:
    print(f"Błąd: Kolumna '{selected_quotation_column}' nie została znaleziona.")
    exit()

data = df[['Data', selected_quotation_column]].copy()
data.set_index('Data', inplace=True)

if len(data) < 100:
    print("Błąd: Zbyt mało danych (mniej niż 100 notowań) do przeprowadzenia analizy.")
    exit()
elif len(data) > 1000:
    time_series_data = data[selected_quotation_column].tail(1000)
else:
    time_series_data = data[selected_quotation_column]

train_size = int(len(time_series_data) * 0.8)
train_data, test_data = time_series_data[0:train_size], time_series_data[train_size:len(time_series_data)]

# AR
ar_rmse_errors = {}
ar_mae_errors = {}
best_ar_rmse = float('inf')
best_ar_p = 0

for p in range(1, 51, 5):
    if len(train_data) <= p:
        continue
    try:
        predictions_ar = []
        history_ar = list(train_data)
        for t in range(len(test_data)):
            model_ar_step = AutoReg(history_ar, lags=p)
            model_ar_fit_step = model_ar_step.fit()
            yhat = model_ar_fit_step.predict(start=len(history_ar), end=len(history_ar))
            predictions_ar.append(yhat[0])
            history_ar.append(test_data.iloc[t])

        rmse_ar = np.sqrt(mean_squared_error(test_data, predictions_ar))
        mae_ar = mean_absolute_error(test_data, predictions_ar)

        ar_rmse_errors[p] = rmse_ar
        ar_mae_errors[p] = mae_ar

        if rmse_ar < best_ar_rmse:
            best_ar_rmse = rmse_ar
            best_ar_p = p
    except Exception:
        continue

plt.figure(figsize=(10, 6))
plt.plot(list(ar_rmse_errors.keys()), list(ar_rmse_errors.values()), marker='o', label='RMSE')
plt.plot(list(ar_mae_errors.keys()), list(ar_mae_errors.values()), marker='x', label='MAE')
plt.title('Błędy modelu AR w zależności od liczby opóźnień')
plt.xlabel('Liczba opóźnień (p)')
plt.ylabel('Wartość błędu')
plt.xticks(list(ar_rmse_errors.keys()))
plt.grid(True)
plt.legend()
plt.show()

# RNN (LSTM)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(time_series_data.values.reshape(-1, 1))

train_scaled = scaled_data[0:train_size, :]
test_scaled = scaled_data[train_size:len(time_series_data), :]

def create_sequences(data, n_steps):
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i:(i + n_steps), 0])
        y.append(data[i + n_steps, 0])
    return np.array(X), np.array(y)

rnn_rmse_errors = {}
rnn_mae_errors = {}
best_rnn_rmse = float('inf')
best_rnn_n_steps = 0

for n_steps in range(1, 51, 5):
    if len(train_scaled) <= n_steps:
        continue

    X_train, y_train = create_sequences(train_scaled, n_steps)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    model_rnn = Sequential()
    model_rnn.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model_rnn.add(Dense(1))
    model_rnn.compile(optimizer='adam', loss='mse')
    model_rnn.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0, shuffle=False)

    rnn_predictions_scaled = []
    input_sequence_rnn = train_scaled[-n_steps:].reshape(1, n_steps, 1)

    for i in range(len(test_data)):
        next_pred_scaled = model_rnn.predict(input_sequence_rnn, verbose=0)[0]
        rnn_predictions_scaled.append(next_pred_scaled)
        input_sequence_rnn = np.append(input_sequence_rnn[:, 1:, :], next_pred_scaled.reshape(1, 1, 1), axis=1)

    rnn_predictions = scaler.inverse_transform(np.array(rnn_predictions_scaled).reshape(-1, 1))
    y_test_rescaled = test_data.values.reshape(-1, 1)

    rmse_rnn = np.sqrt(mean_squared_error(y_test_rescaled, rnn_predictions))
    mae_rnn = mean_absolute_error(y_test_rescaled, rnn_predictions)

    rnn_rmse_errors[n_steps] = rmse_rnn
    rnn_mae_errors[n_steps] = mae_rnn

    if rmse_rnn < best_rnn_rmse:
        best_rnn_rmse = rmse_rnn
        best_rnn_n_steps = n_steps

plt.figure(figsize=(10, 6))
plt.plot(list(rnn_rmse_errors.keys()), list(rnn_rmse_errors.values()), marker='o', label='RMSE')
plt.plot(list(rnn_mae_errors.keys()), list(rnn_mae_errors.values()), marker='x', label='MAE')
plt.title('Błędy modelu RNN w zależności od liczby opóźnień')
plt.xlabel('Liczba opóźnień (n_steps)')
plt.ylabel('Wartość błędu')
plt.xticks(list(rnn_rmse_errors.keys()))
plt.grid(True)
plt.legend()
plt.show()