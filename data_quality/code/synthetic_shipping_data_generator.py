import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import numpy as np

# Set seed for reproducibility
random.seed(42)

# Generate synthetic data using Faker
fake = Faker()
num_customers = 100
num_products = 20
num_sales_agents = 10
num_shipments = 500
num_bookings = 700
transportation_modes = ['Ship', 'Air', 'Truck']
transportation_modes = np.array(transportation_modes)

# Generate customers
customers = [{'CustomerID': i, 'CustomerName': fake.company()} for i in range(1, num_customers + 1)]

# Generate products
products = [{'ProductID': i, 'ProductName': fake.word()} for i in range(1, num_products + 1)]

# Generate sales agents
sales_agents = [{'AgentID': i, 'AgentName': fake.name()} for i in range(1, num_sales_agents + 1)]

# Generate shipments
shipments = [{'ShipmentID': i,
              'CustomerID': random.randint(1, num_customers),
              'ProductID': random.randint(1, num_products),
              'TransportationMode': random.choice(transportation_modes),
              'ShipmentDate': fake.date_between(start_date='-30d', end_date='today')}
             for i in range(1, num_shipments + 1)]

# Generate bookings
bookings = [{'BookingID': i,
             'ShipmentID': random.randint(1, num_shipments),
             'AgentID': random.randint(1, num_sales_agents),
             'BookingDate': fake.date_between(start_date='-45d', end_date='-10d')}
            for i in range(1, num_bookings + 1)]

# Create DataFrames
df_customers = pd.DataFrame(customers)
df_products = pd.DataFrame(products)
df_sales_agents = pd.DataFrame(sales_agents)
df_shipments = pd.DataFrame(shipments)
df_bookings = pd.DataFrame(bookings)
df_transportation_mode = pd.DataFrame({"Transportation Mode": transportation_modes})

# Display sample data
print("Sample Customers Data:")
print(df_customers.head())

print("\nSample Products Data:")
print(df_products.head())

print("\nSample Sales Agents Data:")
print(df_sales_agents.head())

print("\nSample Shipments Data:")
print(df_shipments.head())

print("\nSample Bookings Data:")
print(df_bookings.head())

print("\nSample transportation Mode:")
print(df_transportation_mode.head())
