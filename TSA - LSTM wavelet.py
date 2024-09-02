#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import pywt
import pandas as pd
import numpy as np
import pywt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px


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

col_names_gauge= ['ResourceAttributes','ResourceSchemaUrl','ScopeName','ScopeVersion','ScopeAttributes','ScopeDroppedAttrCount','ScopeSchemaUrl','col1','MetricName','MetricDescription','MetricUnit','Attributes','StartTimeUnix','TimeUnix','Value','Flags','FilteredAttributes','TimeUnix_','Value_','SpanId','TraceId']
path = 'metrics_gauge_10.csv'
data = imported(path,col_names_gauge,'utf-16')


path_test = 'metrics_gauge_11.csv'
data_test = imported(path_test,col_names_gauge,'utf-16')

data = data.merge(data_test,
                   on = col_names_gauge, 
                   how = 'outer')

data.head()

data = filtering(data)
data = slicing(data, ['ResourceAttributes'])

dfs, datetime_indexes = process_dataframes(data, [ 'ScopeName', 'ScopeVersion','MetricName','MetricDescription','MetricUnit','Attributes','StartTimeUnix'])

dfs = impute_dataframes(dfs)


# Function to create sliding windows
def create_dataset(data, window_size):
    X, y, timestamps = [], [], []
    for i in range(len(data) - window_size):
        X.append(data.iloc[i:(i + window_size), 0].values)
        y.append(data.iloc[i + window_size, 0])
        timestamps.append(data.index[i + window_size])
    return np.array(X), np.array(y), timestamps

# Function to build and train LSTM model
def train_lstm(X, y, window_size):
    model = Sequential()
    model.add(LSTM(50, input_shape=(window_size, 1)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=32, verbose=2)
    return model

# Function to detect anomalies
def detect_anomalies(model, X, window_size):
    predictions = model.predict(X)
    mse = np.mean(np.power(X.reshape(X.shape[0], window_size) - predictions, 2), axis=1)
    threshold = np.mean(mse) + 3 * np.std(mse)
    anomalies = mse > threshold
    return anomalies, mse, threshold

# Function to apply wavelet transform and reconstruct data
def wavelet_transform_and_reconstruct(data, wavelet='db4'):
    coeffs = pywt.wavedec(data.iloc[:, 0], wavelet)
    coeffs[1:] = [np.zeros_like(coeff) for coeff in coeffs[1:]]
    reconstructed = pywt.waverec(coeffs, wavelet)
    # Ensure the lengths match
    reconstructed = reconstructed[:len(data)]
    reconstructed_df = pd.DataFrame(reconstructed, index=data.index[:len(reconstructed)], columns=['Reconstructed'])
    return reconstructed_df

# Function to preprocess the data and split into train and test sets
def preprocess_data(df, scaler, split_ratio=0.8):
    df_scaled = pd.DataFrame(scaler.fit_transform(df), index=df.index, columns=df.columns)
    split_index = int(len(df_scaled) * split_ratio)
    train_data = df_scaled['2024-07-01 09:30:22.882256861':'2024-07-01 12:30:22.882256861']
    test_data = df_scaled['2024-07-01 14:00:22.882256861':'2024-07-01 17:00:22.882256861']
    return train_data, test_data

# Function to plot reconstruction loss
def plot_reconstruction_loss(timestamps, mse, title):
    loss_df = pd.DataFrame({
        'Timestamp': timestamps,
        'Reconstruction Loss': mse
    })
    fig = px.line(loss_df, x='Timestamp', y='Reconstruction Loss', title=title)
    fig.show()

# Main function to process each dataframe and detect anomalies
def process_each_dataframe(key, df, window_size):
    scaler = MinMaxScaler()
    
    # Preprocess data
    train_data, test_data = preprocess_data(df, scaler)
    
    # Apply wavelet transform and reconstruct data
    train_reconstructed = wavelet_transform_and_reconstruct(train_data)
    test_reconstructed = wavelet_transform_and_reconstruct(test_data)
    
    # Create sliding windows for training and testing sets
    X_train, y_train, train_timestamps = create_dataset(train_reconstructed, window_size)
    X_test, y_test, test_timestamps = create_dataset(test_reconstructed, window_size)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Train LSTM model
    model = train_lstm(X_train, y_train, window_size)
    
    # Detect anomalies
    train_anomalies, train_mse, train_threshold = detect_anomalies(model, X_train, window_size)
    test_anomalies, test_mse, test_threshold = detect_anomalies(model, X_test, window_size)

    # Plot reconstruction loss
    plot_reconstruction_loss(train_timestamps, train_mse, f'Training Reconstruction Loss for {key}')
    plot_reconstruction_loss(test_timestamps, test_mse, f'Testing Reconstruction Loss for {key}')
    
    return {
        'model': model,
        'train_anomalies': train_anomalies,
        'train_mse': train_mse,
        'train_threshold': train_threshold,
        'test_anomalies': test_anomalies,
        'test_mse': test_mse,
        'test_threshold': test_threshold
    }

# Main function to process all dataframes
def process_dataframes(dfs_dict, window_size):
    results = {}
    for key, df in dfs_dict.items():
        results[key] = process_each_dataframe(key, df, window_size)
    return results

# Example usage
window_size = 50
results = process_dataframes(dfs, window_size)

for key, result in results.items():
    print(f"DataFrame: {key}")
    print(f"Training Anomalies detected at indices: {np.where(result['train_anomalies'])[0]}")
    print(f"Training MSE: {result['train_mse']}")
    print(f"Training Threshold: {result['train_threshold']}\n")
    print(f"Testing Anomalies detected at indices: {np.where(result['test_anomalies'])[0]}")
    print(f"Testing MSE: {result['test_mse']}")
    print(f"Testing Threshold: {result['test_threshold']}\n")



