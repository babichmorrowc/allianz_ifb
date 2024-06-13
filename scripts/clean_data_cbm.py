# Imports
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
# Kaggle classes
# Create ColumnDropper class
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.drop(self.columns_to_drop,axis=1)
        return X_transformed
    
# Create Missing values (-1 or negative) to nan transformer class
class MissingAsNan(BaseEstimator, TransformerMixin):
    def __init__(self, missing_neg1, missing_neg):
        self.missing_neg1 = missing_neg1
        self.missing_neg = missing_neg
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.missing_neg1] = X_transformed[self.missing_neg1]\
            .replace(-1, np.nan)
        X_transformed[self.missing_neg] = X_transformed[self.missing_neg]\
            .applymap(lambda x: np.nan if x < 0 else x)
        return X_transformed

# Create MissingFlagger class
class MissingFlagger(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_flag=None):
        self.columns_to_flag = columns_to_flag
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in self.columns_to_flag:
            X_transformed[f'MISSING_FLAG_{col}'] = X_transformed[col].isnull().astype(int)
        return X_transformed

# Create MissingValueFiller class
class MissingValueFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value=0):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.fillna(self.fill_value)
        return X_transformed
    
# Create IncomeRounder class
class IncomeRounder(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['income'] = X_transformed['income'].round(1)
        return X_transformed
    
# Create Merger class to merge some of the categories of categorical features
class Merger(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        # proposed_credit_limit
        X_transformed['proposed_credit_limit'] = X_transformed['proposed_credit_limit']\
        .apply(lambda x: min(4,max(0,1+x//500))).astype('int')
        # housing_status
        X_transformed['housing_status'] = X_transformed['housing_status']\
        .apply(lambda x: 'other' if x in {'BD','BF','BG'} else x)  
        # device_os
        X_transformed['device_os'] = X_transformed['device_os']\
        .apply(lambda x: 'other' if x == 'x11' else x)
        return X_transformed

# Create CategoricalConverter class, converting dtype of categorical features to 'category'
class CategoricalConverter(BaseEstimator, TransformerMixin):
    def __init__(self, cat_columns):
        self.cat_columns = cat_columns
        self.categories_ = {}

    def fit(self, X, y=None):
        for col in self.cat_columns:
            self.categories_[col] = X[col].astype('category').cat.categories
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for col in self.cat_columns:
            X_transformed[col] = pd.Categorical(X_transformed[col],
                                                categories=self.categories_[col], 
                                                ordered=False)
        return X_transformed    
    
# Create CustomOneHotEncoder class for one-hot-encoding, returning a dataframe
class CustomOneHotEncoder(TransformerMixin, BaseEstimator):
    def __init__(self, ohe_columns):
        self.ohe_columns = ohe_columns
        self.ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.feature_names = None

    def fit(self, X, y=None):
        self.ohe.fit(X[self.ohe_columns].astype('category'))
        self.feature_names = list(X.columns)
        return self

    def transform(self, X):
        # One-hot encode the specified columns
        X_ohe = X[self.ohe_columns].copy()
        X_ohe = self.ohe.transform(X_ohe)
        ohe_column_names = self.ohe.get_feature_names_out(self.ohe_columns)
        X_ohe = pd.DataFrame(X_ohe, columns=ohe_column_names, index=X.index)

        # Concatenate the one-hot-encoded columns with the remaining columns
        X_transformed = pd.concat([X.drop(self.ohe_columns,axis=1), X_ohe], axis=1).copy()

        return X_transformed

# Create CustomScalar class for standardization and column name adjustment
# If no columns given, scales all
class CustomScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_standardize=None):
        self.columns_to_standardize = columns_to_standardize

    def fit(self, X, y=None):
        if self.columns_to_standardize is None:
            self.columns_to_standardize = list(X.columns)
        if self.columns_to_standardize:
            self.scaler = StandardScaler()
            self.scaler.fit(X[self.columns_to_standardize])
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        if self.columns_to_standardize:
            X_transformed[self.columns_to_standardize] = self.scaler\
                .transform(X_transformed[self.columns_to_standardize])
        return X_transformed

########################################################################################
# Deal with missing data

# intended_balcon_amount is the only variable where missingness is represented by any negative value
# Replace all negative intended_balcon_amount values with -1
X_train.loc[X_train['intended_balcon_amount'] < 0, 'intended_balcon_amount'] = -1

cols_missing_neg1 = ['prev_address_months_count',
                     'current_address_months_count',
                     'bank_months_count',
                     'session_length_in_minutes',
                     'device_distinct_emails_8w',
                     'intended_balcon_amount'] # already replaced all negative values with -1

# Add missingness indicator variables
# Using same set of variables to flag as in the Kaggle notebook
cols_to_flag = ['prev_address_months_count', 
                'intended_balcon_amount', 
                'bank_months_count']

# Remove variables with high percentage of missing values (bank_months_count, prev_address_months_count, intended_balcon_amount)
# Drop the variable device_fraud_count (it's always 0)
# Drop the source variable
# Drop month (always 0 in this case)
cols_to_drop1 = ['device_fraud_count', 'source', 'month']
cols_to_drop2 = ['bank_months_count', 'prev_address_months_count', 'intended_balcon_amount']

########################################################################################
# One-hot encoding
ohe_cols = ['payment_type', 
            'employment_status', 
            'housing_status', 
            'device_os', 
            'device_distinct_emails_8w']

########################################################################################
# Create the pipeline
# Dropped columns as above, all missing values are set to -1, added missingness flags to 3 columns,
# one-hot encoded 5 columns, and scaled all columns
pipeline = make_pipeline(
    ColumnDropper(cols_to_drop1),
    MissingAsNan(cols_missing_neg1, []),
    MissingFlagger(cols_to_flag),
    ColumnDropper(cols_to_drop2),
    MissingValueFiller(fill_value=-1),
    CustomOneHotEncoder(ohe_cols),
    CustomScaler()
)

# Fit and transform
pipeline.fit(X_train)
X_train_processed = pipeline.transform(X_train)
X_train_processed.shape

########################################################################################
# Save the cleaned data
train_processed = pd.concat([y_train, X_train_processed], axis = 1)
train_processed.to_csv('./data/train_processed_cbm.csv', index = False)