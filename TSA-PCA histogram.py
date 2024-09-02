#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import plotly
import plotly.graph_objects as go
import warnings
import ast
warnings.filterwarnings("ignore")

col_names_hist = ["ResourceAttributes","ResourceSchemaUrl","ScopeName","ScopeVersion","ScopeAttributes","ScopeDroppedAttrCount","ScopeSchemaUrl","col1","MetricName","MetricDescription","MetricUnit","Attributes","StartTimeUnix","TimeUnix","Count","Sum","BucketCounts","ExplicitBounds","FilteredAttributes","TimeUnix1","Value","SpanId","TraceId","Flags","Min","Max"]

data_hist1 = pd.read_csv('output_hist_data.csv',encoding='utf-16',names = col_names_hist,header = None)

data_hist2 = pd.read_csv('output_histogram_2.csv',encoding='utf-16',names = col_names_hist,header = None) 

data_hist = data_hist1.merge(data_hist2,
                   on = col_names_hist, 
                   how = 'outer')

data_hist

data_hist.info()

data_hist.describe()

data_hist.nunique()

import pandas as pd

# Calculate the number of unique values in each column
unique_counts = data_hist.nunique()

# Filter columns with unique counts greater than 1
df_hist = data_hist.loc[:, unique_counts > 1]

print("DataFrame after dropping columns with <= 1 unique value:")
print(df_hist)


subset_cols = ['ResourceAttributes']

# Get unique combinations of values from the subset of columns
unique_combinations = df_hist.drop_duplicates(subset=subset_cols)

# Create a dictionary to store DataFrames
dfs = {}

# Iterate over unique combinations and filter DataFrame
for _, row in unique_combinations.iterrows():
    # Create a boolean mask for the combination
    mask = (df_hist[subset_cols] == row[subset_cols]).all(axis=1)
    # Use the combination values as the key for the dictionary
    key = tuple(row[subset_cols])
    # Filter the DataFrame and store it in the dictionary
    dfs[key] = df_hist[mask].copy()

# Access each DataFrame by its unique combination
for key, df_value in dfs.items():
    print(f"DataFrame for combination {key}:")
    print(df_value)
    print()


dfs.keys()

# Dictionary to store the results
unique_non_null_counts = {}

# Iterate over each DataFrame in the dictionary
for name, df in dfs.items():
    unique_non_null_counts[name] = df['ExplicitBounds'].dropna().nunique()

# Display the result
unique_non_null_counts

# Dictionary to store the results of unique non-null counts
unique_non_null_counts = {}

# Dictionary to store the split DataFrames
split_dfs = {}

# List to store keys of DataFrames to be removed
keys_to_remove = []

# Iterate over each DataFrame in the dictionary
for name, df in dfs.items():
    # Calculate the number of unique non-null values in 'ExplicitBounds' column
    unique_count = df['ExplicitBounds'].dropna().nunique()
    unique_non_null_counts[name] = unique_count
            
    # Check if unique_non_null_counts is greater than 1
    if unique_count > 1:
        # Split the DataFrame based on 'ExplicitBounds' column
        grouped = df.groupby('ExplicitBounds')
        split_occurred = False  # Track if any split occurred
        
        for key, group in grouped:
            split_dfs[f'{name}_{key}'] = group
            split_occurred = True
        
        # If a split occurred, add the key to keys_to_remove list
        if split_occurred:
            keys_to_remove.append(name)

# Remove the DataFrames from dfs based on keys_to_remove list
for key in keys_to_remove:
    del dfs[key]

# Display the keys of the split DataFrames to verify the splits
print(len(split_dfs))


dfs.update(split_dfs)

null_percentage_dict = {}

# Iterate through the dictionary of DataFrames
for key, df in dfs.items():
    # Calculate the percentage of null values in the 'value' column
    null_percentage = df['BucketCounts'].isnull().mean() * 100
    # Store the result in the results dictionary
    null_percentage_dict[key] = null_percentage

# Print the results
for key, percentage in null_percentage_dict.items():
    print(f"{key}: {percentage:.2f}% null values in 'value' column")

col_names_hist_uni = ["ResourceSchemaUrl","ScopeName","MetricName","MetricDescription","MetricUnit","Attributes","StartTimeUnix", "TimeUnix","BucketCounts","ExplicitBounds"]

col_names_uni_hist =   ["ResourceSchemaUrl","ScopeName","MetricName","MetricDescription","MetricUnit","Attributes","StartTimeUnix", "ExplicitBounds"]

def total_variance_first_n_components(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    total_variance = np.sum(pca.explained_variance_ratio_)
    return total_variance

def num_components_fn(X):
    pca = PCA()
    pca.fit(X)
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumsum_variance >=0.9) + 1
    return num_components

def total_variance_first_n_components(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    total_variance = np.sum(pca.explained_variance_ratio_)
    return total_variance

def num_components_fn(X):
    pca = PCA()
    pca.fit(X)
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumsum_variance >=0.9) + 1
    return num_components

def scale(arrays):
    scaler = StandardScaler()
    non_empty_arrays = [arr for arr in arrays if len(arr) > 0]
    if not non_empty_arrays:
        return arrays  # return original if all arrays are empty

    flattened_arrays = np.vstack(non_empty_arrays)
    original_shape = flattened_arrays.shape
    if original_shape[1] == 0:
        return arrays  # return original if there are no features

    flattened_arrays_scaled = scaler.fit_transform(flattened_arrays)
    arrays_scaled = flattened_arrays_scaled.reshape(original_shape[0], original_shape[1]).tolist()
    return arrays_scaled

def compute_reconstruction_loss_per_timestamp(original_df, reconstructed_df):
    losses = {}
    for timestamp in original_df.index:
        original_row = original_df.loc[timestamp].apply(np.array)
        reconstructed_row = reconstructed_df.loc[timestamp].apply(np.array)
        loss = mean_squared_error(
            np.concatenate(original_row.values), 
            np.concatenate(reconstructed_row.values)
        )
        losses[timestamp] = loss
    return losses



import warnings
warnings.filterwarnings("ignore")

for key in dfs.keys():
    print(f"Key: {key}")
    print("\n")

len(dfs)

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import numpy as np
import ast
import plotly.express as px
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def datetime_con(df1):
    subset = df1[col_names_uni_hist]
    df1['Servers'] = pd.factorize(subset.apply(tuple, axis=1))[0]

    df1 = df1.replace(r'\\N', np.nan, regex=True)
    
    time_series_df1 = df1.pivot_table(index='TimeUnix', columns='Servers', values='BucketCounts', aggfunc=lambda x: x)
    time_series_df1.index = pd.to_datetime(time_series_df1.index)
    return time_series_df1

def pad_list(lst, length):
    return lst + [0] * (length - len(lst))

def compute_reconstruction_loss_per_timestamp(original_df, reconstructed_df):
    losses = {}
    for index in original_df.index:
        original_values = np.concatenate([np.array(original_df.loc[index, col]).flatten() for col in original_df.columns])
        reconstructed_values = np.concatenate([np.array(reconstructed_df.loc[index, col]).flatten() for col in reconstructed_df.columns])
        loss = mean_squared_error(original_values, reconstructed_values)
        losses[index] = loss
    return losses

#split_timestamp = pd.Timestamp('2024-06-19 18:03:18.118482283')

for idx, i in enumerate(dfs):
    
    dfs_i = dfs[i]
    if dfs_i.empty:
        continue
    
    dfs_i = datetime_con(dfs_i)
    
    col_names = dfs_i.columns
    dfs_i = dfs_i.ffill().bfill()
    #print(dfs_i)
    for column_name in col_names:
        dfs_i[column_name] = dfs_i[column_name].apply(ast.literal_eval)
    
    # Split data into training and testing sets
    x_train = dfs_i['2024-06-25 10:08:22.882256861':'2024-06-25 11:08:22.882256861']
    x_test = dfs_i['2024-06-27 10:10:22.882256861':'2024-06-27 11:40:22.882256861']

    # Check if training and testing sets have samples
    if x_train.empty or x_test.empty:
        
        continue

    # Create empty DataFrames to hold the reconstructed data for training and testing sets
    reconstructed_train_df = pd.DataFrame(index=x_train.index, columns=x_train.columns)
    reconstructed_test_df = pd.DataFrame(index=x_test.index, columns=x_test.columns)
    
    # Iterate over each element position in the lists
    for jk in range(len(dfs_i.iloc[0, 0])):
        
        vectors_train = []
        vectors_test = []
        
        for col in dfs_i.columns:
            # Extract the i-th element from each list in the column
            max_len = max(len(item) for col in dfs_i.columns for item in dfs_i[col])
            dfs_i[col] = dfs_i[col].apply(lambda x: pad_list(x, max_len))
            
            elements_train = [row[jk] for row in x_train[col]]
            elements_test = [row[jk] for row in x_test[col]]
            
            vectors_train.append(elements_train)
            vectors_test.append(elements_test)
            
        # Convert vectors to a DataFrame for PCA
        vectors_train_df = pd.DataFrame(vectors_train).transpose()
        vectors_test_df = pd.DataFrame(vectors_test).transpose()

        # Ensure there are enough samples for PCA
        if vectors_train_df.empty or vectors_test_df.empty:
            print(f"Skipping element {jk} in index {idx} due to empty vectors.")
            continue

        # Determine the number of components for PCA
        num_components = min(len(vectors_train_df), len(vectors_train_df.columns))  # Example: Use minimum of rows and columns
        
        # Apply PCA
        pca = PCA(n_components=num_components)
        transformed_train = pca.fit_transform(vectors_train_df)

        # Inverse transform to reconstruct training data
        inverse_transformed_train = pca.inverse_transform(transformed_train)

        # Update the reconstructed DataFrame with the inverse transformed values for training data
        for j, col in enumerate(x_train.columns):
            for k, index in enumerate(x_train.index):
                if not isinstance(reconstructed_train_df.loc[index, col], list):
                    reconstructed_train_df.at[index, col] = [np.nan] * len(dfs_i.iloc[0, 0])
                reconstructed_train_df.at[index, col][jk] = inverse_transformed_train[k, j]

        # Transform and inverse transform testing data using the same PCA
        transformed_test = pca.transform(vectors_test_df)
        inverse_transformed_test = pca.inverse_transform(transformed_test)

        # Update the reconstructed DataFrame with the inverse transformed values for testing data
        for j, col in enumerate(x_test.columns):
            for k, index in enumerate(x_test.index):
                if not isinstance(reconstructed_test_df.loc[index, col], list):
                    reconstructed_test_df.at[index, col] = [np.nan] * len(dfs_i.iloc[0, 0])
                reconstructed_test_df.at[index, col][jk] = inverse_transformed_test[k, j]

    # Compute reconstruction errors
    original_train_flat = np.concatenate([np.array(item).flatten() for sublist in x_train.values for item in sublist])
    reconstructed_train_flat = np.concatenate([np.array(item).flatten() for sublist in reconstructed_train_df.values for item in sublist])
    train_error = mean_squared_error(original_train_flat, reconstructed_train_flat)
    
    original_test_flat = np.concatenate([np.array(item).flatten() for sublist in x_test.values for item in sublist])
    reconstructed_test_flat = np.concatenate([np.array(item).flatten() for sublist in reconstructed_test_df.values for item in sublist])
    test_error = mean_squared_error(original_test_flat, reconstructed_test_flat)
    
    # Compute the reconstruction loss for each timestamp for both training and testing sets
    train_reconstruction_losses = compute_reconstruction_loss_per_timestamp(x_train, reconstructed_train_df)
    test_reconstruction_losses = compute_reconstruction_loss_per_timestamp(x_test, reconstructed_test_df)

    # Plot the reconstruction loss using Plotly
    train_loss_df = pd.DataFrame({
        'Timestamp': list(train_reconstruction_losses.keys()),
        'Reconstruction Loss': list(train_reconstruction_losses.values())
    })

    test_loss_df = pd.DataFrame({
        'Timestamp': list(test_reconstruction_losses.keys()),
        'Reconstruction Loss': list(test_reconstruction_losses.values())
    })

    fig_train = px.line(train_loss_df, x='Timestamp', y='Reconstruction Loss', title='Training Reconstruction Loss per Timestamp')
    fig_train.show()

    fig_test = px.line(test_loss_df, x='Timestamp', y='Reconstruction Loss', title='Testing Reconstruction Loss per Timestamp')
    fig_test.show()

    # Print the overall reconstruction error
    print(i)
    print(f"Training Reconstruction Loss (Mean Squared Error): {train_error}")
    print(f"Testing Reconstruction Loss (Mean Squared Error): {test_error}")





