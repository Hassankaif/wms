import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def prepare_data(data, floor, unit):
    unit_data = data[(data['floor'] == floor) & (data['unit'] == unit)].set_index('date')
    unit_data = unit_data['water_usage'].resample('D').sum()  # Ensure daily frequency
    unit_data.index.freq = 'D'  # Explicitly set the frequency to daily
    return unit_data

def train_test_split(data, test_size=0.2):
    train_size = int(len(data) * (1 - test_size))
    train, test = data[:train_size], data[train_size:]
    return train, test

def ets_forecast(train, test):
    model = ExponentialSmoothing(train, seasonal_periods=7, trend='add', seasonal='add', freq='D')
    fit = model.fit()
    forecast = fit.forecast(len(test))
    rmse = sqrt(mean_squared_error(test, forecast))
    return forecast, rmse

def arima_forecast(train, test):
    model = ARIMA(train, order=(1, 1, 1), freq='D')
    fit = model.fit()
    forecast = fit.forecast(len(test))
    rmse = sqrt(mean_squared_error(test, forecast))
    return forecast, rmse

def hybrid_forecast(train, test):
    ets_pred, ets_rmse = ets_forecast(train, test)
    arima_pred, arima_rmse = arima_forecast(train, test)
    
    # Simple average of both models
    hybrid_pred = (ets_pred + arima_pred) / 2
    hybrid_rmse = sqrt(mean_squared_error(test, hybrid_pred))
    
    return hybrid_pred, hybrid_rmse

def forecast_all_units(data):
    results = {}
    floors = data['floor'].unique()
    units = data['unit'].unique()
    
    for floor in floors:
        for unit in units:
            unit_data = prepare_data(data, floor, unit)
            train, test = train_test_split(unit_data)
            
            forecast, rmse = hybrid_forecast(train, test)
            
            results[(floor, unit)] = {
                'forecast': forecast,
                'rmse': rmse
            }
    
    return results

# Load the data
df = pd.read_csv('water_consumption_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Run forecasts for all units
all_forecasts = forecast_all_units(df)

# Print results and save to CSV
results_df = pd.DataFrame(columns=['Floor', 'Unit', 'RMSE'])
for (floor, unit), result in all_forecasts.items():
    print(f"Floor {floor}, Unit {unit}:")
    print(f"  RMSE: {result['rmse']:.2f}")
    results_df = pd.concat([results_df, pd.DataFrame([{
        'Floor': floor,
        'Unit': unit,
        'RMSE': result['rmse']
    }])], ignore_index=True)

results_df.to_csv('forecasting_results.csv', index=False)
print("\nForecasting results saved to 'forecasting_results.csv'")

