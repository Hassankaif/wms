from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from forecasting2 import forecast_all_units

app = Flask(__name__)

# Load and preprocess data
df = pd.read_csv('water_consumption_data.csv')
df['date'] = pd.to_datetime(df['date'])

# Apply forecasting model
forecasting_results = forecast_all_units(df)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['GET'])
def get_forecast():
    floor = int(request.args.get('floor'))
    unit = int(request.args.get('unit'))
    
    if (floor, unit) in forecasting_results:
        result = forecasting_results[(floor, unit)]
        return jsonify({
            'forecast': result['forecast'].tolist(),
            'rmse': result['rmse'],
            'actual': result['actual'].tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in result['dates']]
        })
    else:
        return jsonify({'error': 'No forecast available for this unit'}), 404

@app.route('/visualizations')
def visualizations():
    images = create_visualizations(df, forecasting_results)
    return render_template('visualizations.html', images=images)

def create_visualizations(df, forecasting_results):
    images = []

    # 1. Total water consumption by floor
    plt.figure(figsize=(10, 6))
    df.groupby('floor')['water_usage'].sum().plot(kind='bar')
    plt.title('Total Water Consumption by Floor')
    plt.xlabel('Floor')
    plt.ylabel('Total Water Usage (Liters)')
    images.append(get_image_base64(plt))

    # 2. Average daily consumption per unit
    plt.figure(figsize=(10, 6))
    df.groupby(['floor', 'unit'])['water_usage'].mean().unstack().plot(kind='bar', stacked=True)
    plt.title('Average Daily Water Consumption per Unit')
    plt.xlabel('Floor')
    plt.ylabel('Average Water Usage (Liters)')
    plt.legend(title='Unit')
    images.append(get_image_base64(plt))

    # 3. Heatmap of RMSE values for each unit
    rmse_data = pd.DataFrame([(f, u, forecasting_results[(f, u)]['rmse']) for f, u in forecasting_results.keys()],
                             columns=['Floor', 'Unit', 'RMSE'])
    pivot_df = rmse_data.pivot(index='Floor', columns='Unit', values='RMSE')
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd')
    plt.title('RMSE Heatmap: Forecasting Performance by Floor and Unit')
    images.append(get_image_base64(plt))

    # 4. Line plot of actual vs forecasted values for a specific unit
    floor, unit = 1, 1  # You can change this to visualize different units
    result = forecasting_results[(floor, unit)]
    plt.figure(figsize=(12, 6))
    plt.plot(result['dates'], result['actual'], label='Actual', color='blue')
    plt.plot(result['dates'], result['forecast'], label='Forecast', color='red', linestyle='--')
    plt.title(f'Actual vs Forecasted Water Consumption (Floor {floor}, Unit {unit})')
    plt.xlabel('Date')
    plt.ylabel('Water Usage (Liters)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    images.append(get_image_base64(plt))

    return images

def get_image_base64(plt):
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import io
# import base64
# from forecasting2 import forecast_all_units

# app = Flask(__name__)

# # Load and preprocess data
# df = pd.read_csv('water_consumption_data.csv')
# df['date'] = pd.to_datetime(df['date'])

# # Apply forecasting model
# forecasting_results = forecast_all_units(df)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/forecast', methods=['GET'])
# def get_forecast():
#     floor = int(request.args.get('floor'))
#     unit = int(request.args.get('unit'))
    
#     if (floor, unit) in forecasting_results:
#         result = forecasting_results[(floor, unit)]
#         return jsonify({
#             'forecast': result['forecast'].tolist(),
#             'rmse': result['rmse']
#         })
#     else:
#         return jsonify({'error': 'No forecast available for this unit'}), 404

# @app.route('/visualizations')
# def visualizations():
#     images = create_visualizations(df, forecasting_results)
#     return render_template('visualizations.html', images=images)

# def create_visualizations(df, forecasting_results):
#     images = []

#     # 1. Total water consumption by floor
#     plt.figure(figsize=(10, 6))
#     df.groupby('floor')['water_usage'].sum().plot(kind='bar')
#     plt.title('Total Water Consumption by Floor')
#     plt.xlabel('Floor')
#     plt.ylabel('Total Water Usage (Liters)')
#     images.append(get_image_base64(plt))

#     # 2. Average daily consumption per unit
#     plt.figure(figsize=(10, 6))
#     df.groupby(['floor', 'unit'])['water_usage'].mean().unstack().plot(kind='bar', stacked=True)
#     plt.title('Average Daily Water Consumption per Unit')
#     plt.xlabel('Floor')
#     plt.ylabel('Average Water Usage (Liters)')
#     plt.legend(title='Unit')
#     images.append(get_image_base64(plt))

#     # 3. Heatmap of RMSE values for each unit
#     rmse_data = pd.DataFrame([(f, u, forecasting_results[(f, u)]['rmse']) for f, u in forecasting_results.keys()],
#                              columns=['Floor', 'Unit', 'RMSE'])
#     pivot_df = rmse_data.pivot(index='Floor', columns='Unit', values='RMSE')
#     plt.figure(figsize=(10, 8))
#     sns.heatmap(pivot_df, annot=True, fmt='.2f', cmap='YlOrRd')
#     plt.title('RMSE Heatmap: Forecasting Performance by Floor and Unit')
#     images.append(get_image_base64(plt))

#     return images

# def get_image_base64(plt):
#     img = io.BytesIO()
#     plt.savefig(img, format='png', bbox_inches='tight')
#     img.seek(0)
#     plt.close()
#     return base64.b64encode(img.getvalue()).decode()

# if __name__ == '__main__':
#     app.run(debug=True)
    
# # python -m venv venv
# # source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
# # pip install flask pandas matplotlib seaborn scikit-learn statsmodels