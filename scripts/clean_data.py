import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#read in data
df = pd.read_csv('data/Base.csv')

#remove columns with large amounts of missing data and columns that are always 0
df = df.drop(['bank_months_count', 'prev_address_months_count', 'intended_balcon_amount', 'device_fraud_count'], axis = 1)

#add column to indicate missing values
missing = {'missing_data': df.isin([-1]).sum(axis=1)}

df = pd.concat([df,  pd.DataFrame(missing)], axis=1)

#write to new file
df.to_csv('data/cleaned_Base.csv')