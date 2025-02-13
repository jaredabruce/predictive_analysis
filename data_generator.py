import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Ensure the data directory exists
os.makedirs('data', exist_ok=True)

### Synthetic Data for Marketing Campaign Performance Dashboard ###
def generate_campaign_data(num_rows=300):
    np.random.seed(42)  # For reproducibility

    # Generate random dates within 2023
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(num_rows)]
    dates = [d.strftime('%Y-%m-%d') for d in dates]

    # Random campaign names
    campaigns = np.random.choice(['Campaign A', 'Campaign B', 'Campaign C', 'Campaign D'], num_rows)

    # Generate random impressions between 1,000 and 50,000
    impressions = np.random.randint(1000, 50000, num_rows)

    # Generate clicks as a fraction of impressions (0.5% to 5% conversion)
    clicks = [np.random.randint(int(impr*0.005), int(impr*0.05)+1) for impr in impressions]

    # Generate conversions as a fraction of clicks (1% to 10%)
    conversions = [np.random.randint(max(1, int(c*0.01)), max(2, int(c*0.10)+1)) for c in clicks]

    # Generate spend values between $500 and $10,000
    spend = np.random.uniform(500, 10000, num_rows).round(2)

    # Generate CLV as a function of conversions (random factor between 50 and 500 per conversion)
    clv = [conv * np.random.randint(50, 501) for conv in conversions]

    # Create DataFrame
    campaign_data = pd.DataFrame({
        'Date': dates,
        'Campaign': campaigns,
        'Impressions': impressions,
        'Clicks': clicks,
        'Conversions': conversions,
        'Spend': spend,
        'CLV': clv
    })

    # Save to CSV
    campaign_data.to_csv('data/campaign_data.csv', index=False)
    print("Synthetic campaign_data.csv generated with {} rows.".format(num_rows))


### Synthetic Data for Customer Segmentation & Predictive Analytics ###
def generate_customer_data(num_customers=500):
    np.random.seed(24)  # For reproducibility

    # Create sequential customer IDs
    customer_ids = np.arange(1, num_customers + 1)

    # Generate random ages between 18 and 80
    ages = np.random.randint(18, 81, num_customers)

    # Generate random incomes between $20,000 and $150,000
    incomes = np.random.randint(20000, 150001, num_customers)

    # Generate random purchase frequency between 1 and 50 purchases per period
    purchase_frequency = np.random.randint(1, 51, num_customers)

    # Generate engagement status: probability of engagement increases with purchase frequency
    # Using a simple logistic function for probability
    prob_engaged = 1 / (1 + np.exp(-0.1 * (purchase_frequency - 25)))  # mid point at frequency 25
    engaged = np.random.binomial(1, prob_engaged)

    # Create DataFrame
    customer_data = pd.DataFrame({
        'CustomerID': customer_ids,
        'Age': ages,
        'Income': incomes,
        'Purchase_Frequency': purchase_frequency,
        'Engaged': engaged
    })

    # Save to CSV
    customer_data.to_csv('data/customer_data.csv', index=False)
    print("Synthetic customer_data.csv generated with {} rows.".format(num_customers))


if __name__ == '__main__':
    generate_campaign_data()
    generate_customer_data()
