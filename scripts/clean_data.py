# Imports
import numpy as np
import pandas as pd

########################################################################################
# Load the data
base_df = pd.read_csv('./data/Base.csv')

target = 'fraud_bool'
X = base_df.drop(target, axis = 1)
y = base_df[target]

# First n months of the data
n = 1
X_train = X[X['month'] < n]
y_train = y[X['month'] < n]

########################################################################################
# Deal with missing data

# intended_balcon_amount is the only variable where missingness is represented by any negative value
# Replace all negative intended_balcon_amount values with -1
X_train.loc[X_train['intended_balcon_amount'] < 0, 'intended_balcon_amount'] = -1

# Add missingness indicator variables

# Remove variables with high percentage of missing values (bank_months_count, prev_address_months_count, intended_balcon_amount)

########################################################################################
# Other cleaning

# Drop the variable device_fraud_count (it's always 0)
X_train = X_train.drop('device_fraud_count', axis = 1)

########################################################################################
# Save the cleaned data