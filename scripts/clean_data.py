import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read in data
df = pd.read_csv('data/Base.csv')

# intended_balcon_amount is the only variable where missingness is represented by any negative value
# Replace all negative intended_balcon_amount values with -1
df.loc[df['intended_balcon_amount'] < 0, 'intended_balcon_amount'] = -1

# List of columns where missingness is indicated with -1
cols_missing_neg1 = ['prev_address_months_count',
                     'current_address_months_count',
                     'bank_months_count',
                     'session_length_in_minutes',
                     'device_distinct_emails_8w',
                     'intended_balcon_amount'] # already replaced all negative values with -1

# Add missingness indicator variables to cols_missing_neg1
for col in cols_missing_neg1:
    df[col + '_ismissing'] = (df[col] == -1).astype(int)

#remove columns with large amounts of missing data and columns that are always 0
df = df.drop(['bank_months_count',
              'prev_address_months_count',
              'intended_balcon_amount',
              'device_fraud_count'], # always 0
              axis = 1)
# df.shape

#write to new file
df.to_csv('data/cleaned_Base.csv')