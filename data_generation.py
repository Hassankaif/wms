import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(num_floors, units_per_floor, num_days):
    data = []
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=num_days)
    
    for floor in range(1, num_floors + 1):
        for unit in range(1, units_per_floor + 1):
            num_residents = np.random.randint(1, 5)
            unit_size = np.random.randint(50, 150)  # in square meters
            
            for day in range(num_days):
                current_date = start_date + timedelta(days=day)
                base_consumption = np.random.normal(150, 30)  # Base daily consumption in liters
                resident_factor = num_residents * np.random.uniform(0.8, 1.2)
                size_factor = (unit_size / 100) * np.random.uniform(0.9, 1.1)
                weekday_factor = 1 + 0.2 * np.sin(np.pi * current_date.weekday() / 3)  # Simulating weekly patterns
                
                water_usage = base_consumption * resident_factor * size_factor * weekday_factor
                
                data.append({
                    'date': current_date,
                    'floor': floor,
                    'unit': unit,
                    'water_usage': round(water_usage, 2),
                    'num_residents': num_residents,
                    'unit_size': unit_size
                })
    
    return pd.DataFrame(data)

# Generate data for a building with 5 floors, 4 units per floor, for the last 365 days
df = generate_data(num_floors=5, units_per_floor=4, num_days=365)

# Save the data to a CSV file
df.to_csv('water_consumption_data.csv', index=False)
print("Data generated and saved to 'water_consumption_data.csv'")