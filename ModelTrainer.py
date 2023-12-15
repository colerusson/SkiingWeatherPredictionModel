from sklearn.linear_model import LinearRegression
import numpy as np

import DataProcessor

data_processor = DataProcessor.DataProcessor()

# Combining multiple data sources into feature set
years = list(DataProcessor.alta_monthly_snowfall.keys())
features = []

for year in years:
    alta_monthly = DataProcessor.alta_monthly_snowfall[year]
    eww_yearly = DataProcessor.eww_yearly_snowfall.get(year, 0)  # Using 0 as default if data is not available
    ots_average_snowfall = DataProcessor.ots_monthly_snowfall['Average Snowfall']
    sf_snow_amount_week = DataProcessor.sf_snow_amount_week
    uac_snowfall = DataProcessor.snowfall_data.get(year, [0] * 6)  # Using 0 as default if data is not available

    # Constructing a feature vector
    feature_vector = alta_monthly + [eww_yearly] + ots_average_snowfall + sf_snow_amount_week + uac_snowfall
    features.append(feature_vector)

# Convert lists to numpy arrays for model training
X = np.array(features)
y = np.array(list(DataProcessor.alta_snow_records.values()))  # Using AltaSnowRecords as the target variable

# Initialize the Linear Regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Predict for a new year (e.g., 2024)
# Consider the new_year_snowfall prediction using data from the respective sources
new_year_monthly_alta = DataProcessor.alta_monthly_snowfall.get('2023', [0.0] * 7)  # Using 0.0 as default if data is not available
new_year_eww_yearly = DataProcessor.eww_yearly_snowfall.get('2024', 0)  # Using 0 as default if data is not available
new_year_ots_average_snowfall = DataProcessor.ots_monthly_snowfall['Average Snowfall']
new_year_sf_snow_amount_week = DataProcessor.sf_snow_amount_week
new_year_uac_snowfall = DataProcessor.snowfall_data.get('2024', [0] * 6)  # Using 0 as default if data is not available

# Constructing a feature vector for the new year
new_year_snowfall = new_year_monthly_alta + [new_year_eww_yearly] + new_year_ots_average_snowfall + new_year_sf_snow_amount_week + new_year_uac_snowfall
new_year_snowfall = np.array(new_year_snowfall).reshape(1, -1)  # Reshape to match model input shape

# Predicted total snowfall for the upcoming year using the trained model
predicted_snowfall = model.predict(new_year_snowfall)
print("Predicted total snowfall for the upcoming year (2024):", predicted_snowfall[0])

print("Predicted total snowfall for the upcoming year:", predicted_snowfall[0])
