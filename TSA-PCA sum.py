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
warnings.filterwarnings("ignore")

def imported(path, headers, encode):
    data = pd.read_csv(path, encoding = encode, names = headers, header = None)
    
    return data

def filtering(data):
    unique_counts = data.nunique()
    df_gauge = data.loc[:, unique_counts > 1]
    
    return df_gauge

def slicing(data, cls):
    # Get unique combinations of values from the subset of columns
    unique_combinations = data.drop_duplicates(subset=cls)

    dfs = {}

    # Iterate over unique combinations and filter DataFrame
    for _, row in unique_combinations.iterrows():
        # Create a boolean mask for the combination
        mask = (data[cls] == row[cls]).all(axis=1)
        
        # Use the combination values as the key for the dictionary
        key = tuple(row[cls])
        # Filter the DataFrame and store it in the dictionary
        dfs[key] = data[mask].copy()
        
    return dfs
    

def pivot(dfs_i,col_names):
    
    
    subset = dfs_i[col_names]
    dfs_i['Servers'] = pd.factorize(subset.apply(tuple, axis=1))[0]

    dfs_i = dfs_i.replace(r'\\N', np.nan, regex=True)
    time_series_df = dfs_i.pivot_table(index='TimeUnix', columns='Servers', values='Value')

    # Extract datetime index before conversion
    datetime_index = time_series_df.index

    # Convert index to datetime
    time_series_df.index = pd.to_datetime(time_series_df.index)
    return time_series_df,datetime_index

def impute(time_series_df):
    window_size = 3
    for column in time_series_df.columns:
        if column != 'TimeUnix':  # Skip datetime column for imputation
            sma = time_series_df[column].rolling(window=window_size, min_periods=1).mean()
            time_series_df[column] = time_series_df[column].fillna(sma)
        time_series_df_filled = time_series_df.fillna(method='ffill')
        time_series_df_filled = time_series_df_filled.fillna(method='bfill')
    return time_series_df_filled

def process_dataframes(dfs_dict, col_names):
    processed_dict = {}
    datetime_indexes = {}

    for key, df in dfs_dict.items():
        try:
            pivoted_df, datetime_index = pivot(df, col_names)
            # Filter out DataFrames with less than 50 rows
            if len(pivoted_df) >= 50:
                processed_dict[key] = pivoted_df
                datetime_indexes[key] = datetime_index
        except KeyError as e:
            print(f"Skipping {key} due to missing columns: {e}")
    
    return processed_dict, datetime_indexes

def impute_dataframes(dfs_dict):
    imputed_dfs_dict = {}
    
    for key, df in dfs_dict.items():
        imputed_dfs_dict[key] = impute(df)
    
    return imputed_dfs_dict

def scale(time_series_df_filled, scaler):
    # Standardize numerical features
    numerical_features = time_series_df_filled.select_dtypes(include=[np.number])
    scaled_features = scaler.fit_transform(numerical_features)
    time_series_df_scaled = pd.DataFrame(scaled_features, columns=numerical_features.columns, index=time_series_df_filled.index)
    return time_series_df_scaled


def scale_dataframes(dfs_dict):
    scaled_dfs_dict = {}
    
    for key, df in dfs_dict.items():
        if not df.empty:  # Check if the DataFrame is not empty
            time_series_df_filled = df.copy()
            scaler = StandardScaler()
            scaled_df = scale(time_series_df_filled, scaler)
            scaled_dfs_dict[key] = scaled_df
    
    return scaled_dfs_dict


def total_variance_first_n_components(X, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(X)
    total_variance = np.sum(pca.explained_variance_ratio_)
    return total_variance

def num_components_fn(X):
    pca = PCA()
    pca.fit(X)
    cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumsum_variance >=0.9)
                               
    return num_components

def train(time_series_df,i):

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(time_series_df)
    X_std = pd.DataFrame(scaled_features, columns=time_series_df.columns, index=time_series_df.index)
    
    # Determine the number of PCA components
    num_components = num_components_fn(time_series_df)
    
    # Apply PCA
    pca = PCA(n_components=num_components, random_state=0)
    df_pca = pd.DataFrame(pca.fit_transform(time_series_df), index=time_series_df.index)
       
    # Inverse transform the PCA result to original feature space
    df_res = pd.DataFrame(pca.inverse_transform(df_pca), index=df_pca.index, columns=time_series_df.columns)
    
    # Compute the reconstruction loss
    loss = np.sum((np.array(time_series_df) - np.array(df_res)) ** 2, axis=1)
    loss = pd.Series(data=loss, index=time_series_df.index)
    
    # Store scaler and pca for future use
    trained_model = {
        'scaler': scaler,
        'pca': pca,
        'num_components': num_components,
        'columns': time_series_df.columns
    }
    
    #return loss
        
    trace = {
      'x': loss.index,
      'y': loss.values,
      'type': 'scatter',
      'mode': 'lines+markers',
      'name': f'Loss - {i}'  # Optional: Add filename or identifier to the plot
    }
    layout = {
      'title': f'Reconstruction Loss for {i}',
      'xaxis': {'title': 'Time'},
      'yaxis': {'title': 'Loss'}
    }
    fig = plotly.graph_objs.Figure(data=[trace], layout=layout)
    fig.show()
    
    return trained_model

def test(time_series_df, trained_model, i):
    # Standardize numerical features
    numerical_features = time_series_df.select_dtypes(include=[np.number])
    
    # Apply PCA transformation using the trained PCA model
    pca = trained_model['pca']
    X_pca = pd.DataFrame(pca.transform(numerical_features), index=numerical_features.index)
    
    # Inverse transform PCA to reconstruct the original space
    X_inv_pca = pd.DataFrame(pca.inverse_transform(X_pca), index=X_pca.index)
    
    # Compute reconstruction loss
    loss = np.sum((numerical_features - X_inv_pca) ** 2, axis=1)
    loss = pd.Series(data=loss, index=numerical_features.index)
    
    # Plot reconstruction loss (optional)
    trace = {
      'x': loss.index,
      'y': loss.values,
      'type': 'scatter',
      'mode': 'lines+markers',
      'name': f'Loss - {i}'  # Optional: Add filename or identifier to the plot
    }
    layout = {
      'title': f'Reconstruction Loss for {i}',
      'xaxis': {'title': 'Time'},
      'yaxis': {'title': 'Loss'}
    }
    fig = plotly.graph_objs.Figure(data=[trace], layout=layout)
    fig.show()
    
    return loss


def split_dataframe(df, start_timestamp_train, end_timestamp_train, start_timestamp_test, end_timestamp_test):
    train_df = df[(df.index > start_timestamp_train) & (df.index <= end_timestamp_train)]
    test_df = df[(df.index > start_timestamp_test) & (df.index <= end_timestamp_test)]
    return train_df, test_df

def train_dataframes(dfs_dict):
    trained_models = {}
    
    for key, df in dfs_dict.items():
        trained_model = train(df,key)
        trained_models[key] = trained_model
    
    return trained_models

def test_dataframes(test_dfs_dict, trained_models):
    test_results = {}
    
    for key, df in test_dfs_dict.items():
        if key in trained_models:
            loss = test(df, trained_models[key], key)
            test_results[key] = loss
        else:
            print(f"Skipping {key} as it does not have a corresponding trained model.")
    
    return test_results

col_names_sum = ["ResourceAttributes","ResourceSchemaUrl","ScopeName","ScopeVersion","ScopeAttributes","ScopeDroppedAttrCount","ScopeSchemaUrl","col1","MetricName","MetricDescription","MetricUnit","Attributes","StartTimeUnix","TimeUnix","Value","Flags","FilteredAttributes","TimeUnix1","Value1","SpanId","TraceId","AggTemp","IsMonotonic"]
path = 'output_sum_2.csv'
data = imported(path,col_names_sum,'utf-16')

data.head()

path_test = 'output_sum_data.csv'
data_test = imported(path_test,col_names_sum,'utf-16')

data = data.merge(data_test,
                   on = col_names_sum, 
                   how = 'outer')

data = filtering(data)
data = slicing(data, ['ResourceAttributes'])

data

dfs, datetime_indexes = process_dataframes(data, [ 'MetricName','MetricDescription','MetricUnit','Attributes','StartTimeUnix'])


dfs = impute_dataframes(dfs)


dfs = scale_dataframes(dfs)

train_dfs = {}
test_dfs = {}

for key, df in dfs.items():
    train_df, test_df = split_dataframe(df, '2024-06-25 10:08:22.882256861','2024-06-25 11:08:22.882256861','2024-06-27 10:10:22.882256861','2024-06-27 11:40:22.882256861')
    
    # Only add to dictionaries if both train and test are non-empty
    if not train_df.empty and not test_df.empty:
        train_dfs[key] = train_df
        test_dfs[key] = test_df

trained_models = train_dataframes(train_dfs)

test_results = test_dataframes(test_dfs, trained_models)




