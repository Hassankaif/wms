from flask import Flask, request, jsonify, render_template
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from forecasting import forecast, check_usage_threshold
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)

# Load contact information (in a real-world scenario, this would be stored in a database)
contacts = pd.read_excel('contacts.xlsx')

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/forecast', methods=['POST'])
def get_forecast():
    data = request.json
    room = data['room']
    start_date = data['start_date']
    end_date = data['end_date']
    
    forecast_data = forecast(room, start_date, end_date)
    
    return jsonify(forecast_data.to_dict(orient='records'))

@app.route('/visualizations')
def visualizations():
    images = create_visualizations()
    return render_template('visualizations.html', images=images)

def create_visualizations():
    images = []
    
    # Example visualization (you can add more based on your needs)
    rooms = ['A101', 'A102', 'A103', 'A201', 'A202', 'A203']
    forecast_data = [forecast(room, '2023-05-01', '2023-05-07') for room in rooms]
    
    plt.figure(figsize=(12, 6))
    for i, room in enumerate(rooms):
        plt.plot(forecast_data[i]['date'], forecast_data[i]['ensemble_forecast'], label=room)
    plt.title('Water Consumption Forecast for Next Week')
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

def send_notification(room, actual_usage, expected_usage):
    contact = contacts[contacts['room'] == room].iloc[0]
    
    subject = f"High Water Usage Alert for Room {room}"
    body = f"Dear {contact['name']},\n\nYour water usage ({actual_usage:.2f} liters) has exceeded 90% of the expected usage ({expected_usage:.2f} liters) for today. Please check for any leaks or unnecessary water consumption.\n\nBest regards,\nWater Conservation System"
    
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = "your_email@example.com"
    msg['To'] = contact['email']
    
    # In a real-world scenario, you would use a proper email service or SMS gateway
    # For demonstration purposes, we'll just print the message
    print(f"Notification sent to {contact['email']}:")
    print(msg.as_string())

@app.route('/check_usage', methods=['POST'])
def check_usage():
    data = request.json
    room = data['room']
    date = data['date']
    
    exceeded, actual_usage, expected_usage = check_usage_threshold(room, date)
    
    if exceeded:
        send_notification(room, actual_usage, expected_usage)
    
    return jsonify({
        'exceeded': exceeded,
        'actual_usage': actual_usage,
        'expected_usage': expected_usage
    })

if __name__ == '__main__':
    app.run(debug=True)