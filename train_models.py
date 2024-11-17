import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import save_model
import joblib
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_historical_data(start_date, end_date, num_floors, units_per_floor):
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    data = []
    
    for floor in range(1, num_floors + 1):
        for unit in range(1, units_per_floor + 1):
            room = f'A{floor:03d}'
            base_consumption = np.random.normal(150, 30)
            for date in date_range:
                consumption = base_consumption * (1 + 0.2 * np.sin(date.dayofyear * 2 * np.pi / 365))
                consumption *= np.random.uniform(0.8, 1.2)
                data.append({
                    'date': date,
                    'room': room,
                    'water_usage': round(consumption, 2)
                })
    
    return pd.DataFrame(data)

def train_ets_model(data):
    model = ExponentialSmoothing(data, seasonal_periods=7, trend='add', seasonal='add')
    return model.fit()

def train_arima_model(data):
    model = ARIMA(data, order=(1, 1, 1))
    return model.fit()

def train_lstm_model(data, look_back=7):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(scaled_data) - look_back):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back, 0])
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=100, batch_size=32, verbose=0)
    
    return model, scaler

# Generate 2 years of historical data
start_date = '2022-01-01'
end_date = '2023-12-31'
num_floors = 5
units_per_floor = 4

df = generate_historical_data(start_date, end_date, num_floors, units_per_floor)

# Create models directory if it doesn't exist
if not os.path.exists('models'):
    os.makedirs('models')

# Train and save models for each room
for room in df['room'].unique():
    room_data = df[df['room'] == room].set_index('date')['water_usage']
    
    # Train and save ETS model
    ets_model = train_ets_model(room_data)
    joblib.dump(ets_model, f'models/ets_model_{room}.joblib')
    
    # Train and save ARIMA model
    arima_model = train_arima_model(room_data)
    joblib.dump(arima_model, f'models/arima_model_{room}.joblib')
    
    # Train and save LSTM model
    lstm_model, lstm_scaler = train_lstm_model(room_data)
    save_model(lstm_model, f'models/lstm_model_{room}.h5')

    joblib.dump(lstm_scaler, f'models/lstm_scaler_{room}.joblib')

print("Models trained and saved successfully.")
