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

# def imported(path, headers):
#     data = pd.read_csv(path, names = headers, header = None)
    
#     return data


def imported(path, headers):
    # Read the file and exclude the last line
    with open(path, 'r') as file:
        lines = file.readlines()

    lines = lines[:-1]

    # Use csv.reader to parse the remaining lines
    data = []
    for line in csv.reader(lines):
        data.append(line)

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=headers)
    
    return df

def filtering(data):
    unique_counts = data.nunique()
    df_gauge = data.loc[:, unique_counts > 1]
    
    return df_gauge

def slicing(data, cls):
    # Define the keys to keep
    keys_to_keep = {'host.name', 'os.type', 'process.command_line', 'process.runtime.description', 'process.runtime.name', 'service.instance.id', 'service.name'}
    
    # Define a function to clean keys and values and filter the dictionary
    def clean_and_filter_dict(d):
        cleaned_dict = {k.strip('\\'): v.strip('\\') for k, v in d.items()}
        return {k: v for k, v in cleaned_dict.items() if k in keys_to_keep}
    
    # Convert the string of dictionary to actual dictionary, clean, and filter it
    data[cls] = data[cls].apply(lambda x: clean_and_filter_dict(ast.literal_eval(x)))
    
    # Get unique combinations of values from the subset of columns
    unique_combinations = data.drop_duplicates(subset=[cls])

    dfs = {}

    # Iterate over unique combinations and filter DataFrame
    for _, row in unique_combinations.iterrows():
        # Create a boolean mask for the combination
        mask = data[cls].apply(lambda x: x == row[cls])
        
        # Use the combination values as the key for the dictionary
        key = tuple(row[cls].items())
        # Filter the DataFrame and store it in the dictionary
        dfs[key] = data[mask].copy()
        
    return dfs

def pivot(dfs_i, col_names):
    # Select subset of columns
    subset = dfs_i[col_names]
    
    # Factorize based on subset values
    unique_combinations, inverse_indices = np.unique(subset.apply(tuple, axis=1), return_inverse=True)
    
    # Assign unique combination labels
    dfs_i['Servers'] = inverse_indices
    
    # Create a DataFrame to store original values for each unique combination
    unique_combinations_df = pd.DataFrame(data=[list(comb) for comb in unique_combinations], columns=[f'{col}_value' for col in col_names])
    unique_combinations_df['Servers'] = range(len(unique_combinations))
    
    # Replace '\\N' with NaN
    dfs_i = dfs_i.replace(r'\\N', np.nan, regex=True)
    
    # Pivot table
    time_series_df = dfs_i.pivot_table(index='TimeUnix', columns='Servers', values='Value')
    
    # Extract datetime index before conversion
    datetime_index = time_series_df.index
    
    # Convert index to datetime
    time_series_df.index = pd.to_datetime(time_series_df.index)
    
    return time_series_df, datetime_index, unique_combinations_df

def pivot(dfs_i, col_names):
    # Replace '\\N' with NaN
    dfs_i = dfs_i.replace(r'\\N', np.nan, regex=True)
    
    # Select subset of columns
    subset = dfs_i[col_names]
    
    # Factorize based on subset values
    unique_combinations, inverse_indices = np.unique(subset.apply(tuple, axis=1), return_inverse=True)
    
    # Assign unique combination labels
    dfs_i['Servers'] = inverse_indices
    
    # Create a DataFrame to store original values for each unique combination
    unique_combinations_df = pd.DataFrame(data=[list(comb) for comb in unique_combinations], columns=[f'{col}_value' for col in col_names])
    unique_combinations_df['Servers'] = range(len(unique_combinations))
    
    # Convert 'Value' column to numeric, coerce errors to NaN
    dfs_i['Value'] = pd.to_numeric(dfs_i['Value'], errors='coerce')
    
    # Drop rows with NaN in 'Value' column
    dfs_i = dfs_i.dropna(subset=['Value'])
    
    # Pivot table
    time_series_df = dfs_i.pivot_table(index='TimeUnix', columns='Servers', values='Value')
    
    # Extract datetime index before conversion
    datetime_index = time_series_df.index
    
    # Convert index to datetime
    time_series_df.index = pd.to_datetime(time_series_df.index)
    
    return time_series_df, datetime_index, unique_combinations_df


def count_nulls_in_dataframes(dfs_dict):
    null_counts = {}

    for key, df in dfs_dict.items():
        null_counts[key] = df.isnull().sum()

    return null_counts

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
    unique_combinations_dict = {}

    for key, df in dfs_dict.items():
        try:
            pivoted_df, datetime_index, unique_combinations = pivot(df, col_names)
            unique_combinations_dict[key] = unique_combinations
            # Filter out DataFrames with less than 50 rows
            if len(pivoted_df) >= 50:
                processed_dict[key] = pivoted_df
                datetime_indexes[key] = datetime_index
        except KeyError as e:
            print(f"Skipping {key} due to missing columns: {e}")
    
    return processed_dict, datetime_indexes, unique_combinations_dict

def merge_train_test(dfs_dict, timestamp):

    timestamp = pd.to_datetime(timestamp)

    for key, df in dfs_dict.items():
        # Ensure the index ('TimeUnix') is in datetime format
        df.index = pd.to_datetime(df.index)

        # Iterate through columns
        for column in df.columns:
            if column == 'TimeUnix':  # Skip if the column is the index
                continue

            # Check if all values before the timestamp are NaN
            before_timestamp_all_nan = df[df.index < timestamp][column].isna().all()

            # Check if all values after the timestamp are NaN
            after_timestamp_all_nan = df[df.index >= timestamp][column].isna().all()

            # Remove the column if the condition is met
            if before_timestamp_all_nan or after_timestamp_all_nan:
                df.drop(columns=[column], inplace=True)

    return dfs_dict

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
    num_components = np.argmax(cumsum_variance >=0.8)+1
                               
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
      'name': f'Loss - {i}'  #
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

col_names_gauge= ['ResourceAttributes','ResourceSchemaUrl','ScopeName','ScopeVersion','ScopeAttributes','ScopeDroppedAttrCount','ScopeSchemaUrl','ServiceName','MetricName','MetricDescription','MetricUnit','Attributes','StartTimeUnix','TimeUnix','Value','Flags','FilteredAttributes','TimeUnix_','Value_','SpanId','TraceId']
path = 'metrics_gauge_23.csv'
data = imported(path,col_names_gauge)



path_test_1 = 'metrics_gauge_20.csv'
path_test_2 = 'metrics_gauge_21.csv'
data_test_1 = imported(path_test_1,col_names_gauge)
data_test_2 = imported(path_test_2,col_names_gauge)

data_test = data_test_1.merge(data_test_2,
                   on = col_names_gauge, 
                   how = 'outer')

data['MetricDescription'].unique()

def sort_dict_keys(d):
    try:
        # Parse string to dictionary
        d = ast.literal_eval(d)
        # Check if parsed object is a dictionary
        if isinstance(d, dict):
            # Sort dictionary keys and return as string
            return str({k: d[k] for k in sorted(d)})
    except (ValueError, SyntaxError):
        pass  # Handle cases where literal_eval fails or it's not a dictionary
    return d  # Return original string if parsing fails or it's not a dictionary


data['ResourceAttributes'] = data['ResourceAttributes'].apply(sort_dict_keys)
data_test['ResourceAttributes'] = data_test['ResourceAttributes'].apply(sort_dict_keys)

data = filtering(data)
data = slicing(data, 'ResourceAttributes')

data_test = filtering(data_test)
data_test = slicing(data_test, 'ResourceAttributes')

for key,value in data.items():
    print(key)
    print(value)

for key in data.keys():
    print(key)

for key in data_test.keys():
    print(key)

key_mapping = {
    (('service.name', 'recommendationservice'),): (('service.name', 'recommendationservice'),),
    (('service.name', 'loadgenerator'),): (('service.name', 'loadgenerator'),),
    (('host.name', '8a95a745cf54'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'ac2d5c0a-e8ae-4f85-8f13-d892e172ff1c'), ('service.name', 'frauddetectionservice')):(('host.name', '1f11ae35b957'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'd5f247e1-7968-4066-aba6-25e0af1127d2'), ('service.name', 'frauddetectionservice')),
    (('host.name', '765d26684dab'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '10a920c5-598f-4aec-ab92-9572db6fd57c'), ('service.name', 'frauddetectionservice')):(('host.name', '765d26684dab'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '10a920c5-598f-4aec-ab92-9572db6fd57c'), ('service.name', 'frauddetectionservice')),
    (('host.name', 'd1b82c21102a'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '5af87af8-d7be-4db0-81bc-f6ad331a89a3'), ('service.name', 'frauddetectionservice')):(('host.name', 'd1b82c21102a'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '5af87af8-d7be-4db0-81bc-f6ad331a89a3'), ('service.name', 'frauddetectionservice')),
    (('host.name', 'd1b82c21102a'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '39c3cd5d-7598-47ff-9e7d-7a2e0c63ee14'), ('service.name', 'frauddetectionservice')):(('host.name', 'd1b82c21102a'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '39c3cd5d-7598-47ff-9e7d-7a2e0c63ee14'), ('service.name', 'frauddetectionservice')),
    (('host.name', 'd1b82c21102a'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '39c3cd5d-7598-47ff-9e7d-7a2e0c63ee14'), ('service.name', 'frauddetectionservice')):(('host.name', 'd1b82c21102a'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '73305b45-c7be-4209-a8f7-af6e2710f80b'), ('service.name', 'frauddetectionservice')),
    (('host.name', '621839224abe'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'dd6d3363-bd67-4545-ac65-61b26e06b9e3'), ('service.name', 'adservice')):(('host.name', '0d3239c1afc4'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '6e987c5f-3638-4806-99fe-d89bab6fe764'), ('service.name', 'adservice')),
    (('host.name', 'a16a8d132263'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'ed934dc1-1b06-4906-8acc-e8089f1dc802'), ('service.name', 'adservice')):(('host.name', 'a16a8d132263'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'ed934dc1-1b06-4906-8acc-e8089f1dc802'), ('service.name', 'adservice')),
    (('host.name', '2362b3a6c565'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'd1b155aa-29f9-4876-aa60-1411a3349b3c'), ('service.name', 'adservice')):(('host.name', '2362b3a6c565'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'd1b155aa-29f9-4876-aa60-1411a3349b3c'), ('service.name', 'adservice')),
    (('host.name', '2362b3a6c565'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '4608a776-ce25-429d-a72f-5dfa0251b28e'), ('service.name', 'adservice')):(('host.name', '2362b3a6c565'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '4608a776-ce25-429d-a72f-5dfa0251b28e'), ('service.name', 'adservice')),
    (('host.name', '2362b3a6c565'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '173c5b15-c74a-4aac-86df-c1bd0defa732'), ('service.name', 'adservice')):(('host.name', '2362b3a6c565'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '173c5b15-c74a-4aac-86df-c1bd0defa732'), ('service.name', 'adservice')),
    (('host.name', '3843ea646f9f'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '4b876095-59ff-4dd7-be80-da6cef958562'), ('service.name', 'kafka')):(('host.name', '3843ea646f9f'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'bd09fa07-33ed-4c92-b8ae-3fec424a4d52'), ('service.name', 'kafka'))
}
#     (('host.name', 'a52fdb9cf875'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '6103ec55-cc79-462d-90c8-1f81593e9d82'), ('service.name', 'adservice')): (('host.name', '84518c826674'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -javaagent:/usr/src/app/opentelemetry-javaagent.jar oteldemo.AdService'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.3+9-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'a5c9f292-aefa-44b3-8b8a-deee173c37b8'), ('service.name', 'adservice')),:(('host.name', '1f11ae35b957'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'd5f247e1-7968-4066-aba6-25e0af1127d2'), ('service.name', 'frauddetectionservice')),
    
#     (('host.name', '5e8f08281084'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'c36d74eb-2a55-4d02-bb97-4cdb8c31eb78'), ('service.name', 'frauddetectionservice')): (('host.name', 'd5c7ea9dfff7'), ('os.type', 'linux'), ('process.runtime.description', 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'ca8e160c-2f6c-4799-b01d-1ada3d251a36'), ('service.name', 'frauddetectionservice')),
#     (('host.name', '2b789fb26ab5'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '10d2718a-92ce-42cd-9692-6ff671c4573c'), ('service.name', 'kafka')): (('host.name', 'b124b78b953e'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'b420875e-bb81-4c90-97c3-2b28cdc40660'), ('service.name', 'kafka')),
#     (('host.name', '2b789fb26ab5'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/storage.jsa -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/tools-log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.docker.KafkaDockerWrapper setup --default-configs-dir /etc/kafka/docker --mounted-configs-dir /mnt/shared/config --final-configs-dir /opt/kafka/config'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '11fadf36-aa29-464d-8df2-81a434afb76e'), ('service.name', 'kafka')): (('host.name', 'b124b78b953e'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/storage.jsa -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/tools-log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.docker.KafkaDockerWrapper setup --default-configs-dir /etc/kafka/docker --mounted-configs-dir /mnt/shared/config --final-configs-dir /opt/kafka/config'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'a54254b1-bb89-4223-b062-e7e197a07773'), ('service.name', 'kafka'))
# }

# Initialize a dictionary to store the merged DataFrames
merged_dfs = {}

# Iterate through the mapping and merge the DataFrames
for train_key, test_key in key_mapping.items():
    if train_key in data and test_key in data_test:
        # Merge the DataFrames by concatenating along the rows and reset the index
        merged_df = pd.concat([data[train_key], data_test[test_key]], axis=0).reset_index(drop=True)
        # Store the merged DataFrame in the new dictionary
        merged_dfs[str(train_key) + '_' + str(test_key)] = merged_df

# Display the merged DataFrames
for key, df in merged_dfs.items():
    print(f'Merged DataFrame for {key}:\n{df}\n')

len(merged_dfs)

# import ast

# data_string = "{'container.id': '5e8f082810844bbcbe6b731c6f775266a61f7ce11b95964c666d232cfe5a7a59', 'docker.cli.cobra.command_path': 'docker compose', 'host.arch': 'amd64', 'host.name': '5e8f08281084', 'os.description': 'Linux 5.15.153.1-microsoft-standard-WSL2', 'os.type': 'linux', 'process.command_args': '[\"/usr/lib/jvm/java-17-openjdk-amd64/bin/java\",\"-jar\",\"frauddetectionservice-1.0-all.jar\"]', 'process.executable.path': '/usr/lib/jvm/java-17-openjdk-amd64/bin/java', 'process.pid': '1', 'process.runtime.description': 'Debian OpenJDK 64-Bit Server VM 17.0.11+9-Debian-1deb11u1', 'process.runtime.name': 'OpenJDK Runtime Environment', 'process.runtime.version': '17.0.11+9-Debian-1deb11u1', 'service.instance.id': 'c36d74eb-2a55-4d02-bb97-4cdb8c31eb78', 'service.name': 'frauddetectionservice', 'telemetry.distro.name': 'opentelemetry-java-instrumentation', 'telemetry.distro.version': '2.4.0', 'telemetry.sdk.language': 'java', 'telemetry.sdk.name': 'opentelemetry', 'telemetry.sdk.version': '1.38.0'}"

# data_dict = ast.literal_eval(data_string)
# print(data_dict)

dfs, datetime_indexes,unique_combinations = process_dataframes(merged_dfs, [ 'ScopeName', 'ScopeVersion','MetricName','Attributes'])
#dfs, datetime_indexes,unique_combinations = process_dataframes(data, [ 'MetricDescription'])

null_counts = count_nulls_in_dataframes(dfs)

for df_name, null_count in null_counts.items():
    print(f"Null values in {df_name}:")
    print(null_count)

timestamp = '2024-07-21 03:10:22.882256861'
dfs = merge_train_test(dfs, timestamp)

for key, df in dfs.items():
    print(f"{key}:")
    print(df)

dfs = {key: df for key, df in dfs.items() if not df.empty}


unique_combinations

len(dfs)

dfs = impute_dataframes(dfs)


dfs = scale_dataframes(dfs)

dfs

train_dfs = {}
test_dfs = {}

for key, df in dfs.items():
    train_df, test_df = split_dataframe(df, '2024-07-22 16:47:22.882256861','2024-07-23 04:47:22.882256861','2024-07-19 14:40:22.882256861','2024-07-21 02:10:22.882256861')
    
    # Only add to dictionaries if both train and test are non-empty
    if not train_df.empty and not test_df.empty:
        train_dfs[key] = train_df
        test_dfs[key] = test_df



trained_models = train_dataframes(train_dfs)



import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_reconstruction_loss(numerical_features, pca):
    # Apply PCA transformation
    X_pca = pd.DataFrame(pca.transform(numerical_features), index=numerical_features.index)
    
    # Inverse transform PCA to reconstruct the original space
    X_inv_pca = pd.DataFrame(pca.inverse_transform(X_pca), index=X_pca.index)
    
    # Compute reconstruction loss
    reconstruction_error = np.sum((numerical_features.values - X_inv_pca.values) ** 2, axis=1)
    total_loss = pd.Series(data=reconstruction_error, index=numerical_features.index)
    
    return total_loss

def plot_feature_signals(timestamps, features_data, title):
    traces = []
    for feature, data in features_data.items():
        trace = go.Scatter(
            x=timestamps,
            y=data,
            mode='lines+markers',
            name=str(feature)  # Ensure feature name is a string
        )
        traces.append(trace)
    
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Time'},
        yaxis={'title': 'Value'},
        legend=dict(orientation='h', y=-0.3)
    )
    fig = go.Figure(data=traces, layout=layout)
    fig.show()

def analyze_anomalies_from_dict(train_dict, test_dict, trained_models, threshold, k):
    # Initialize a dictionary to store top k features for each dataframe
    top_k_features_dict = {}

    for key in test_dict.keys():
        if key not in train_dict or key not in trained_models:
            print(f"Warning: Missing training data or trained model for key: {key}")
            continue

        train_df = train_dict[key]
        test_df = test_dict[key]
        
        trained_model = trained_models[key]
        
        # Ensure both training and testing datasets have the same columns
        columns = trained_model['columns']
        train_df = train_df[columns]
        test_df = test_df[columns]
        
        # Standardize numerical features
        scaler = trained_model['scaler']
        pca = trained_model['pca']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        # Compute reconstruction losses
        test_loss = compute_reconstruction_loss(test_numerical, pca)
        
        # Identify timestamps with loss above the threshold in the test data
        anomalous_timestamps = test_loss[test_loss > threshold].index
        
        top_k_features_dict[key] = {}
        for timestamp in anomalous_timestamps:
            # Calculate reconstruction loss for all features at this timestamp
            X_pca = pca.transform(test_numerical.loc[[timestamp]])
            X_inv_pca = pca.inverse_transform(X_pca)
            reconstruction_error = np.sum((test_numerical.loc[[timestamp]].values - X_inv_pca) ** 2, axis=0)
            
            # Identify top k features with the highest loss for the current timestamp
            top_k_features = np.argsort(reconstruction_error)[-k:]
            top_k_features_names = [columns[i] for i in top_k_features]
            top_k_features_dict[key][timestamp] = top_k_features_names
    
    # Plot the original signals for the top k features in the training and test data
    for key, timestamp_features in top_k_features_dict.items():
        if key not in trained_models:
            continue
        
        train_df = train_dict[key]
        test_df = test_dict[key]
        trained_model = trained_models[key]
        scaler = trained_model['scaler']
        columns = trained_model['columns']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        for timestamp, features in timestamp_features.items():
            # Plot training data for these features
            train_features_data = {feature: train_numerical[feature] for feature in features}
            plot_feature_signals(train_numerical.index, train_features_data, f'Training Data - Top {k} Features Time-Series Signals at {str(timestamp)} ({key})')
            
            # Plot test data for these features
            test_features_data = {feature: test_numerical[feature] for feature in features}
            plot_feature_signals(test_numerical.index, test_features_data, f'Test Data - Top {k} Features Time-Series Signals at {str(timestamp)} ({key})')

# Example usage
# Assuming `train_dfs` and `test_dfs` are your dictionaries of training and testing dataframes respectively
# and `trained_models` is a dictionary of trained models for each key

# Set threshold and k
threshold = 50  # Example threshold
k = 5  # Number of top features to consider

# Train the models and store them in a dictionary
trained_models = {}
for key, df in train_dfs.items():
    trained_models[key] = train(df, key)

# Call the function
analyze_anomalies_from_dict(train_dfs, test_dfs, trained_models, threshold, k)


import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_reconstruction_loss(numerical_features, pca):
    # Apply PCA transformation
    X_pca = pd.DataFrame(pca.transform(numerical_features), index=numerical_features.index)
    
    # Inverse transform PCA to reconstruct the original space
    X_inv_pca = pd.DataFrame(pca.inverse_transform(X_pca), index=X_pca.index)
    
    # Compute reconstruction loss
    reconstruction_error = np.sum((numerical_features.values - X_inv_pca.values) ** 2, axis=1)
    total_loss = pd.Series(data=reconstruction_error, index=numerical_features.index)
    
    return total_loss

def plot_single_feature_signal(timestamps, feature_name, data, title):
    trace = go.Scatter(
        x=timestamps,
        y=data,
        mode='lines+markers',
        name=str(feature_name)  # Ensure feature name is a string
    )
    
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Time'},
        yaxis={'title': 'Value'},
        legend=dict(orientation='h', y=-0.3)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def analyze_anomalies_from_dict(train_dict, test_dict, trained_models, threshold, k, keys_to_include):
    # Initialize a dictionary to store top k features for each dataframe
    top_k_features_dict = {}

    for key in keys_to_include:
        if key not in train_dict or key not in test_dict or key not in trained_models:
            print(f"Warning: Missing data or trained model for key: {key}")
            continue

        train_df = train_dict[key]
        test_df = test_dict[key]
        
        trained_model = trained_models[key]
        
        # Ensure both training and testing datasets have the same columns
        columns = trained_model['columns']
        train_df = train_df[columns]
        test_df = test_df[columns]
        
        # Standardize numerical features
        scaler = trained_model['scaler']
        pca = trained_model['pca']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        # Compute reconstruction losses
        test_loss = compute_reconstruction_loss(test_numerical, pca)
        
        # Identify timestamps with loss above the threshold in the test data
        anomalous_timestamps = test_loss[test_loss > threshold].index
        
        top_k_features_dict[key] = {}
        for timestamp in anomalous_timestamps:
            # Calculate reconstruction loss for all features at this timestamp
            X_pca = pca.transform(test_numerical.loc[[timestamp]])
            X_inv_pca = pca.inverse_transform(X_pca)
            reconstruction_error = np.sum((test_numerical.loc[[timestamp]].values - X_inv_pca) ** 2, axis=0)
            
            # Identify top k features with the highest loss for the current timestamp
            top_k_features = np.argsort(reconstruction_error)[-k:]
            top_k_features_names = [columns[i] for i in top_k_features]
            top_k_features_dict[key][timestamp] = top_k_features_names
    
    # Plot the original signals for the top k features in the training and test data
    for key, timestamp_features in top_k_features_dict.items():
        if key not in trained_models:
            continue
        
        train_df = train_dict[key]
        test_df = test_dict[key]
        trained_model = trained_models[key]
        scaler = trained_model['scaler']
        columns = trained_model['columns']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        for timestamp, features in timestamp_features.items():
            for feature in features:
                # Plot training data for this feature
                train_feature_data = train_numerical[feature]
                plot_single_feature_signal(train_numerical.index, feature, train_feature_data, f'Training Data - {feature} Time-Series Signal at {str(timestamp)} ({key})')
                
                # Plot test data for this feature
                test_feature_data = test_numerical[feature]
                plot_single_feature_signal(test_numerical.index, feature, test_feature_data, f'Test Data - {feature} Time-Series Signal at {str(timestamp)} ({key})')

# Set threshold and k
threshold = 50  # Example threshold
k = 5  # Number of top features to consider

# Define the keys you want to include in the analysis
keys_to_include = ["(('host.name', '2b789fb26ab5'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '10d2718a-92ce-42cd-9692-6ff671c4573c'), ('service.name', 'kafka'))_(('host.name', 'b124b78b953e'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'b420875e-bb81-4c90-97c3-2b28cdc40660'), ('service.name', 'kafka'))"]  

# Train the models and store them in a dictionary
trained_models = {}
for key, df in train_dfs.items():
    trained_models[key] = train(df, key)

# Call the function
analyze_anomalies_from_dict(train_dfs, test_dfs, trained_models, threshold, k, keys_to_include)


import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_reconstruction_loss(numerical_features, pca):
    # Apply PCA transformation
    X_pca = pd.DataFrame(pca.transform(numerical_features), index=numerical_features.index)
    
    # Inverse transform PCA to reconstruct the original space
    X_inv_pca = pd.DataFrame(pca.inverse_transform(X_pca), index=X_pca.index)
    
    # Compute reconstruction loss
    reconstruction_error = np.sum((numerical_features.values - X_inv_pca.values) ** 2, axis=1)
    total_loss = pd.Series(data=reconstruction_error, index=numerical_features.index)
    
    return total_loss

def plot_single_feature_signal(timestamps, feature_name, data, title, unique_comb_value):
    trace = go.Scatter(
        x=timestamps,
        y=data,
        mode='lines+markers',
        name=str(feature_name)  # Ensure feature name is a string
    )
    
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Time'},
        yaxis={'title': 'Value'},
        legend=dict(orientation='h', y=-0.3),
        annotations=[
            dict(
                xref='paper',
                yref='paper',
                x=0,
                y=-0.2,
                showarrow=False,
                text=f"Unique Combination Value: {unique_comb_value}"
            )
        ]
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def analyze_anomalies_from_dict(train_dict, test_dict, trained_models, threshold, k, keys_to_include, unique_combinations):
    # Initialize a dictionary to store top k features for each dataframe
    top_k_features_dict = {}

    for key in keys_to_include:
        if key not in train_dict or key not in test_dict or key not in trained_models:
            print(f"Warning: Missing data or trained model for key: {key}")
            continue

        train_df = train_dict[key]
        test_df = test_dict[key]
        
        trained_model = trained_models[key]
        
        # Ensure both training and testing datasets have the same columns
        columns = trained_model['columns']
        train_df = train_df[columns]
        test_df = test_df[columns]
        
        # Standardize numerical features
        scaler = trained_model['scaler']
        pca = trained_model['pca']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        # Compute reconstruction losses
        test_loss = compute_reconstruction_loss(test_numerical, pca)
        
        # Identify timestamps with loss above the threshold in the test data
        anomalous_timestamps = test_loss[test_loss > threshold].index
        
        top_k_features_dict[key] = {}
        for timestamp in anomalous_timestamps:
            # Calculate reconstruction loss for all features at this timestamp
            X_pca = pca.transform(test_numerical.loc[[timestamp]])
            X_inv_pca = pca.inverse_transform(X_pca)
            reconstruction_error = np.sum((test_numerical.loc[[timestamp]].values - X_inv_pca) ** 2, axis=0)
            
            # Identify top k features with the highest loss for the current timestamp
            top_k_features = np.argsort(reconstruction_error)[-k:]
            top_k_features_names = [columns[i] for i in top_k_features]
            top_k_features_dict[key][timestamp] = top_k_features_names
    
    # Plot the original signals for the top k features in the training and test data
    for key, timestamp_features in top_k_features_dict.items():
        if key not in trained_models:
            continue
        
        train_df = train_dict[key]
        test_df = test_dict[key]
        trained_model = trained_models[key]
        scaler = trained_model['scaler']
        columns = trained_model['columns']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        for timestamp, features in timestamp_features.items():
            for feature in features:
                # Get unique combination value for the current key
                unique_comb_value = unique_combinations.get((key, feature), "N/A")
                
                # Plot training data for this feature
                train_feature_data = train_numerical[feature]
                plot_single_feature_signal(train_numerical.index, feature, train_feature_data, f'Training Data - {feature} Time-Series Signal at {str(timestamp)} ({key})', unique_comb_value)
                
                # Plot test data for this feature
                test_feature_data = test_numerical[feature]
                plot_single_feature_signal(test_numerical.index, feature, test_feature_data, f'Test Data - {feature} Time-Series Signal at {str(timestamp)} ({key})', unique_comb_value)

def train(df, key):
    # Mock training function
    numeric_df = df.select_dtypes(include=[np.number])  # Select only numeric columns
    scaler = StandardScaler().fit(numeric_df)
    pca = PCA(n_components=2).fit(scaler.transform(numeric_df))
    return {'scaler': scaler, 'pca': pca, 'columns': numeric_df.columns}

# Set threshold and k
threshold = 50  # Example threshold
k = 5  # Number of top features to consider

# Define the keys you want to include in the analysis
keys_to_include = ["(('host.name', '2b789fb26ab5'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dlog4j.configuration=file:/opt/kafka/config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '10d2718a-92ce-42cd-9692-6ff671c4573c'), ('service.name', 'kafka'))_(('host.name', 'b124b78b953e'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dlog4j.configuration=file:/opt/kafka/config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'b420875e-bb81-4c90-97c3-2b28cdc40660'), ('service.name', 'kafka'))"]



# Train the models and store them in a dictionary
trained_models = {}
for key, df in train_dfs.items():
    trained_models[key] = train(df, key)

# Call the function
analyze_anomalies_from_dict(train_dfs, test_dfs, trained_models, threshold, k, keys_to_include, unique_combinations)


import numpy as np
import pandas as pd
import plotly.graph_objs as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_reconstruction_loss(numerical_features, pca):
    # Apply PCA transformation
    X_pca = pd.DataFrame(pca.transform(numerical_features), index=numerical_features.index)
    
    # Inverse transform PCA to reconstruct the original space
    X_inv_pca = pd.DataFrame(pca.inverse_transform(X_pca), index=X_pca.index)
    
    # Compute reconstruction loss
    reconstruction_error = np.sum((numerical_features.values - X_inv_pca.values) ** 2, axis=1)
    total_loss = pd.Series(data=reconstruction_error, index=numerical_features.index)
    
    return total_loss

def plot_single_feature_signal(timestamps, feature_name, data, title):
    trace = go.Scatter(
        x=timestamps,
        y=data,
        mode='lines+markers',
        name=str(feature_name)  # Ensure feature name is a string
    )
    
    layout = go.Layout(
        title=title,
        xaxis={'title': 'Time'},
        yaxis={'title': 'Value'},
        legend=dict(orientation='h', y=-0.3)
    )
    fig = go.Figure(data=[trace], layout=layout)
    fig.show()

def analyze_anomalies_from_dict(train_dict, test_dict, trained_models, threshold, k, keys_to_include,unique_combinations):
    # Initialize a dictionary to store top k features for each dataframe
    top_k_features_dict = {}

    for key in keys_to_include:
        if key not in train_dict or key not in test_dict or key not in trained_models:
            print(f"Warning: Missing data or trained model for key: {key}")
            continue

        train_df = train_dict[key]
        test_df = test_dict[key]
        
        trained_model = trained_models[key]
        
        # Ensure both training and testing datasets have the same columns
        columns = trained_model['columns']
        train_df = train_df[columns]
        test_df = test_df[columns]
        
        # Standardize numerical features
        scaler = trained_model['scaler']
        pca = trained_model['pca']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        # Compute reconstruction losses
        test_loss = compute_reconstruction_loss(test_numerical, pca)
        
        # Identify timestamps with loss above the threshold in the test data
        anomalous_timestamps = test_loss[test_loss > threshold].index
        
        top_k_features_dict[key] = {}
        for timestamp in anomalous_timestamps:
            # Calculate reconstruction loss for all features at this timestamp
            X_pca = pca.transform(test_numerical.loc[[timestamp]])
            X_inv_pca = pca.inverse_transform(X_pca)
            reconstruction_error = np.sum((test_numerical.loc[[timestamp]].values - X_inv_pca) ** 2, axis=0)
            
            # Identify top k features with the highest loss for the current timestamp
            top_k_features = np.argsort(reconstruction_error)[-k:]
            top_k_features_names = [columns[i] for i in top_k_features]
            top_k_features_dict[key][timestamp] = top_k_features_names
    
    # Plot the original signals for the top k features in the training and test data
    for key, timestamp_features in top_k_features_dict.items():
        if key not in trained_models:
            continue
        
        train_df = train_dict[key]
        test_df = test_dict[key]
        trained_model = trained_models[key]
        scaler = trained_model['scaler']
        columns = trained_model['columns']
        
        train_numerical = pd.DataFrame(scaler.transform(train_df), columns=columns, index=train_df.index)
        test_numerical = pd.DataFrame(scaler.transform(test_df), columns=columns, index=test_df.index)
        
        for timestamp, features in timestamp_features.items():
            for feature in features:
                # Get unique combination value for the current key
                if key in unique_combinations:
                    unique_comb_df = unique_combinations[key]
                    if feature in unique_comb_df['Servers'].values:
                        row = unique_comb_df[unique_comb_df['Servers'] == feature].iloc[0]
                        unique_comb_value = row.to_dict()
                        unique_comb_value.pop('Servers', None)  # Remove the 'Servers' column
                    else:
                        unique_comb_value = "N/A"
                else:
                    unique_comb_value = "N/A"
                
                # Plot training data for this feature
                train_feature_data = train_numerical[feature]
                plot_single_feature_signal(train_numerical.index, feature, train_feature_data, f'Training Data - {feature} Time-Series Signal at {str(timestamp)} ({key})')
                print(unique_comb_value)
                
                # Plot test data for this feature
                test_feature_data = test_numerical[feature]
                plot_single_feature_signal(test_numerical.index, feature, test_feature_data, f'Test Data - {feature} Time-Series Signal at {str(timestamp)} ({key})')
                print(unique_comb_value)

# Set threshold and k
threshold = 50  # Example threshold
k = 5  # Number of top features to consider

# Define the keys you want to include in the analysis
keys_to_include = ["(('host.name', '2b789fb26ab5'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', '10d2718a-92ce-42cd-9692-6ff671c4573c'), ('service.name', 'kafka'))_(('host.name', 'b124b78b953e'), ('os.type', 'linux'), ('process.command_line', '/opt/java/openjdk/bin/java -Xmx400m -Xms400m -XX:SharedArchiveFile=/opt/kafka/kafka.jsa -Xlog:gc*:file=/opt/kafka/bin/../logs/kafkaServer-gc.log:time,tags:filecount=10,filesize=100M -Dcom.sun.management.jmxremote=true -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false -Dkafka.logs.dir=/opt/kafka/bin/../logs -Dlog4j.configuration=file:/opt/kafka/bin/../config/log4j.properties -javaagent:/tmp/opentelemetry-javaagent.jar -Dotel.jmx.target.system=kafka-broker kafka.Kafka /opt/kafka/config/server.properties'), ('process.runtime.description', 'Eclipse Adoptium OpenJDK 64-Bit Server VM 21.0.2+13-LTS'), ('process.runtime.name', 'OpenJDK Runtime Environment'), ('service.instance.id', 'b420875e-bb81-4c90-97c3-2b28cdc40660'), ('service.name', 'kafka'))"]  

# Train the models and store them in a dictionary
trained_models = {}
for key, df in train_dfs.items():
    trained_models[key] = train(df, key)

# Call the function
analyze_anomalies_from_dict(train_dfs, test_dfs, trained_models, threshold, k, keys_to_include,unique_combinations)


test_results = test_dataframes(test_dfs, trained_models)

for key in train_dfs.keys():
    print(key)

for key in test_dfs.keys():
    print(key)








