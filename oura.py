import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Authentication and setup
OURA_ACCESS_TOKEN = "  "  # Replace with your token
headers = {"Authorization": f"Bearer {OURA_ACCESS_TOKEN}"}
base_url = "https://api.ouraring.com/v2/usercollection/daily_sleep"

# Date range parameters (last 30 days by default)
end_date = datetime.now().date()
start_date = end_date - timedelta(days=30)

# Fetch sleep data
params = {
    "start_date": start_date.isoformat(),
    "end_date": end_date.isoformat()
}

response = requests.get(base_url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()
    sleep_data = data.get("data", [])
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(sleep_data)
    
    # Display basic info
    print(f"Successfully retrieved {len(df)} days of sleep data")
    print("\nSample data:")
    print(df.head())
    
    # Basic visualization
    if not df.empty and 'total_sleep_duration' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.bar(df['day'], df['total_sleep_duration'] / 3600)  # Convert seconds to hours
        plt.title('Total Sleep Duration (Hours)')
        plt.xlabel('Date')
        plt.ylabel('Hours')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Calculate average sleep stats
        avg_sleep = df['total_sleep_duration'].mean() / 3600
        print(f"\nAverage sleep duration: {avg_sleep:.2f} hours")
else:
    print(f"Error: {response.status_code}")
    print(response.text)