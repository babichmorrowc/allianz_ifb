
#%% Imports

from prince import FAMD
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Load the data

data_path = "../Base.csv"
target = "fraud_bool"

X = pd.read_csv(data_path)
y = X[target]
X = X.drop(target, axis=1)

#%% Take the first n_datapoints datapoints from the first month and shuffle them

n_datapoints = 10000

shuffle_indices = np.random.permutation(n_datapoints)

X_months = X[X['month'] < 1][:n_datapoints].iloc[shuffle_indices]
y_months = y[X['month'] < 1][:n_datapoints].iloc[shuffle_indices]

#%% Remove the month column

X_months = X_months.drop('month', axis=1)

#%% Remove columns with too many missing values

remove_features = ["intended_balcon_amount","prev_address_months_count","bank_months_count"]
X_months = X_months.drop(remove_features, axis=1)

#%% Ignore other missing values for now

# Something about fixing missing values

#%% Fit the FAMD model. The prince library automatically standardizes the numerical data

famd = FAMD(
    n_components=X_months.shape[1],
    n_iter=100,
    copy=True,
    check_input=True,
    random_state=42,
    engine="sklearn",
    handle_unknown="error"  # same parameter as sklearn.preprocessing.OneHotEncoder
)

embedding_components = famd.fit_transform(X_months)

#%% Print statistics about the embedding
print(famd.eigenvalues_summary)

#%% Choose the two components
chosen_components = embedding_components.iloc[:, [0,1]]

#%% Plot the embedding
np_chosen_components = chosen_components.to_numpy()

plt.scatter(np_chosen_components[y_months==0, 0], np_chosen_components[y_months==0, 1], c='blue', label='No fraud', s=10)
plt.scatter(np_chosen_components[y_months==1, 0], np_chosen_components[y_months==1, 1], c='red', label='Fraud', s=10)
plt.xlim(-30, 30)  # Adjust the range as needed
plt.ylim(-30, 30)  # Adjust the range as needed
plt.legend()
plt.show()

#%% Do KDE on the embedding for fraud and no fraud

import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Plot KDE for the first dataset
sns.kdeplot(data=chosen_components[y_months == 0], x=0, y=1, fill=True, thresh=0,
           ax=axes[0])
axes[0].set_title('Density Plot for No Fraud')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')
axes[0].set_xlim(-30, 30)  # Adjust the range as needed
axes[0].set_ylim(-30, 30)  # Adjust the range as needed

# Plot KDE for the second dataset
sns.kdeplot(data=chosen_components[y_months == 1], x=0, y=1, fill=True, thresh=0,
           ax=axes[1], color='red')
axes[1].set_title('Density Plot for Fraud')
axes[1].set_xlabel('PC1')
axes[1].set_ylabel('PC2')
axes[1].set_xlim(-30, 30)  # Adjust the range as needed
axes[1].set_ylim(-30, 30)  # Adjust the range as needed

# Display the plot
plt.tight_layout()
# plt.savefig("kdeplot.png")
plt.show()

#%% Do a grid of plots for all pairs of the first n_comp components

n_comps = 8

fig, axes = plt.subplots(n_comps, n_comps, figsize=(3*n_comps, 3*n_comps))

np_embedding_components = embedding_components.to_numpy()

for i in range(n_comps):
    for j in range(n_comps):
        axes[i, j].scatter(np_embedding_components[y_months==0, i], np_embedding_components[y_months==0, j], c='blue', label='No fraud', s=10)
        axes[i, j].scatter(np_embedding_components[y_months==1, i], np_embedding_components[y_months==1, j], c='red', label='Fraud', s=10)
        axes[i, j].set_xlim(-30, 30)  # Adjust the range as needed
        axes[i, j].set_ylim(-30, 30)  # Adjust the range as needed

plt.tight_layout()
plt.show()

#%% Do a grid of plots for all pairs of the last n_comp components

n_comps = 8

fig, axes = plt.subplots(n_comps, n_comps, figsize=(3*n_comps, 3*n_comps))

np_embedding_components = embedding_components.to_numpy()

for i in range(n_comps):
    for j in range(n_comps):
        axes[i, j].scatter(np_embedding_components[y_months==0, -i], np_embedding_components[y_months==0, -j], c='blue', label='No fraud', s=10)
        axes[i, j].scatter(np_embedding_components[y_months==1, -i], np_embedding_components[y_months==1, -j], c='red', label='Fraud', s=10)
        axes[i, j].set_xlim(-30, 30)  # Adjust the range as needed
        axes[i, j].set_ylim(-30, 30)  # Adjust the range as needed

plt.tight_layout()
plt.show()
