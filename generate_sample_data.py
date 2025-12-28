import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate sample real estate data for Baku, Azerbaijan
num_listings = 1500

# Districts in Baku
districts = ['Yasamal', 'Nasimi', 'Sabunchu', 'Nizami', 'Narimanov', 'Binagadi',
             'Surakhani', 'Sabail', 'Qaradagh', 'Khazar', 'Pirallahi']

# Metro stations
metro_stations = ['28 May', 'Sahil', 'Icheri Sheher', 'Koroglu', 'Qara Qarayev',
                  'Neftchilar', 'Khojasan', 'Elmlar Akademiyasi', 'Inshaatchilar',
                  'Avtovagzal', 'Memar Ajemi', 'Nizami', 'Ganjlik', 'Nariman Narimanov',
                  'Bakmil', 'Ulduz', 'Koroghlu', 'Darnagul', 'Ahmadli', 'Hazi Aslanov']

# Property types
property_types = ['Yeni tikili', 'Kohne tikili', 'Heyet evi']

# Number of rooms
room_options = ['1', '2', '3', '4', '5', '5+']

# Repair status
repair_status = ['Ela', 'Yaxshi', 'Orta', 'Temirsiz', 'Zemkli']

# Document types
document_types = ['Kupcha', 'Muqavile']

# Generate data
data = []

for i in range(num_listings):
    # Random dates over the last 6 months
    days_ago = random.randint(0, 180)
    posted_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')

    # Property characteristics
    district = random.choice(districts)
    property_type = random.choice(property_types)
    rooms = random.choice(room_options)

    # Area based on room count
    if rooms == '1':
        area = random.randint(35, 65)
    elif rooms == '2':
        area = random.randint(55, 90)
    elif rooms == '3':
        area = random.randint(80, 130)
    elif rooms == '4':
        area = random.randint(110, 170)
    else:
        area = random.randint(140, 250)

    # Floor (max 20 floors)
    total_floors = random.randint(5, 20)
    floor = random.randint(1, total_floors)
    floor_str = f"{floor}/{total_floors}"

    # Repair
    repair = random.choice(repair_status)

    # Document
    document = random.choice(document_types)

    # Price calculation with realistic variations
    # Base price per sqm varies by district
    district_multipliers = {
        'Yasamal': 1.15, 'Nasimi': 1.10, 'Sabail': 1.30, 'Narimanov': 1.05,
        'Nizami': 1.00, 'Binagadi': 0.85, 'Surakhani': 0.80, 'Sabunchu': 0.75,
        'Qaradagh': 0.90, 'Khazar': 0.95, 'Pirallahi': 0.70
    }

    # Property type multipliers
    type_multipliers = {'Yeni tikili': 1.0, 'Kohne tikili': 0.75, 'Heyet evi': 1.1}

    # Repair multipliers
    repair_multipliers = {'Ela': 1.15, 'Yaxshi': 1.05, 'Orta': 0.95, 'Temirsiz': 0.80, 'Zemkli': 1.10}

    base_price_per_sqm = 1200  # Average base price per sqm in AZN

    price_per_sqm = base_price_per_sqm * district_multipliers.get(district, 1.0) * \
                    type_multipliers[property_type] * repair_multipliers[repair]

    # Add some randomness
    price_per_sqm *= random.uniform(0.9, 1.1)

    total_price_azn = int(price_per_sqm * area)
    total_price_usd = int(total_price_azn / 1.7)  # Approximate exchange rate

    # Metro proximity (60% have metro nearby)
    metro = random.choice(metro_stations) if random.random() < 0.6 else None

    # Create record
    record = {
        'listing_id': f'TK{100000 + i}',
        'posted_date': posted_date,
        'property_type': property_type,
        'area': area,
        'rooms': rooms,
        'floor': floor_str,
        'repair': repair,
        'document': document,
        'price_azn': total_price_azn,
        'price_usd': total_price_usd,
        'district': district,
        'metro': metro,
        'price_per_sqm_azn': int(price_per_sqm),
        'price_per_sqm_usd': int(price_per_sqm / 1.7)
    }

    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('tikili_listings.csv', index=False, encoding='utf-8')
print(f"Generated {len(df)} listings")
print(f"Saved to tikili_listings.csv")

# Show sample
print("\nSample data:")
print(df.head(10))
print("\nData summary:")
print(df.describe())
