#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Number of samples
n_samples = 50

# Generate dummy data
def generate_dummy_data(n_samples):
    np.random.seed(42)
    
    # Traces dataset
    traces_data = {
        'Timestamp': pd.date_range(start='1/1/2021', periods=n_samples, freq='T'),
        'TraceId': np.random.randint(1, 10, n_samples),
        'SpanId': np.random.randint(1, 100, n_samples),
        'ParentSpanId': np.random.randint(1, 100, n_samples),
        'TraceState': np.random.choice(['active', 'inactive'], n_samples),
        'SpanName': np.random.choice(['spanA', 'spanB', 'spanC'], n_samples),
        'SpanKind': np.random.choice(['client', 'server'], n_samples),
        'ServiceName': np.random.choice(['service1', 'service2'], n_samples),
        'ResourceAttributes': np.random.choice(['attr1', 'attr2'], n_samples),
        'ScopeName': np.random.choice(['scope1', 'scope2'], n_samples),
        'ScopeVersion': np.random.choice(['v1', 'v2'], n_samples),
        'SpanAttributes': np.random.choice(['attrA', 'attrB'], n_samples),
        'Duration': np.random.randint(1, 10, n_samples),
        'StatusCode': np.random.choice(['ok', 'error'], n_samples),
        'StatusMessage': np.random.choice(['msg1', 'msg2'], n_samples),
        'Events.Timestamp': pd.date_range(start='1/1/2021', periods=n_samples, freq='T'),
        'Events.Name': np.random.choice(['event1', 'event2'], n_samples),
        'Events.Attributes': np.random.choice(['attrE1', 'attrE2'], n_samples),
        'Links.TraceId': np.random.randint(1, 10, n_samples),
        'Links.SpanId': np.random.randint(1, 100, n_samples),
        'Links.TraceState': np.random.choice(['linked', 'unlinked'], n_samples),
        'Links.Attributes': np.random.choice(['attrL1', 'attrL2'], n_samples)
    }
    traces_df = pd.DataFrame(traces_data)

    # Metrics dataset
    metrics_data = {
        'Timestamp': pd.date_range(start='1/1/2021', periods=n_samples, freq='T'),
        'ResourceAttributes': np.random.choice(['attr1', 'attr2'], n_samples),
        'ResourceSchemaUrl': np.random.choice(['url1', 'url2'], n_samples),
        'ScopeName': np.random.choice(['scope1', 'scope2'], n_samples),
        'ScopeVersion': np.random.choice(['v1', 'v2'], n_samples),
        'ScopeAttributes': np.random.choice(['attrS1', 'attrS2'], n_samples),
        'ScopeDroppedAttrCount': np.random.randint(1, 5, n_samples),
        'ScopeSchemaUrl': np.random.choice(['url1', 'url2'], n_samples),
        'col1': np.random.choice(['col1_val1', 'col1_val2'], n_samples),
        'MetricName': np.random.choice(['metric1', 'metric2'], n_samples),
        'MetricDescription': np.random.choice(['desc1', 'desc2'], n_samples),
        'MetricUnit': np.random.choice(['unit1', 'unit2'], n_samples),
        'Attributes': np.random.choice(['attrM1', 'attrM2'], n_samples),
        'StartTimeUnix': np.random.randint(1609459200, 1609459200 + n_samples * 60, n_samples),
        'TimeUnix': np.random.randint(1609459200, 1609459200 + n_samples * 60, n_samples),
        'Value': np.random.rand(n_samples),
        'Flags': np.random.choice(['flag1', 'flag2'], n_samples),
        'FilteredAttributes': np.random.choice(['attrF1', 'attrF2'], n_samples),
        'TimeUnix_': np.random.randint(1609459200, 1609459200 + n_samples * 60, n_samples),
        'Value_': np.random.rand(n_samples),
        'SpanId': np.random.randint(1, 100, n_samples),
        'TraceId': np.random.randint(1, 10, n_samples)
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    return traces_df, metrics_df

# Fetch and preprocess data
def fetch_data(n_samples):
    traces_df, metrics_df = generate_dummy_data(n_samples)
    combined_df = pd.merge(traces_df, metrics_df, on='Timestamp', how='left')
    return combined_df

data = fetch_data(n_samples)

# Feature engineering
data['span_duration'] = data['Duration']
data['metric_value'] = data['Value']

# Create span chains
def build_span_chains(df, span_id):
    chain = []
    current_span = df[df['SpanId_x'] == span_id]
    while not current_span.empty:
        chain.append(current_span)
        parent_id = current_span.iloc[0]['ParentSpanId']
        current_span = df[df['SpanId_x'] == parent_id]
    return pd.concat(chain, ignore_index=True)

span_chains = pd.DataFrame()
unique_spans = data['SpanId_x'].unique()
for span_id in unique_spans:
    chain_df = build_span_chains(data, span_id)
    span_chains = pd.concat([span_chains, chain_df], ignore_index=True).drop_duplicates()

# Split data into training, validation, and testing
train_data, temp_data = train_test_split(span_chains, test_size=0.3, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Feature preparation
features_train = train_data[['span_duration', 'metric_value']]
features_val = val_data[['span_duration', 'metric_value']]
features_test = test_data[['span_duration', 'metric_value']]

# Standardize features
scaler = StandardScaler()
scaled_features_train = scaler.fit_transform(features_train)
scaled_features_val = scaler.transform(features_val)
scaled_features_test = scaler.transform(features_test)

# Train Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(scaled_features_train)

# Validate on validation set
val_anomalies = iso_forest.predict(scaled_features_val)
val_data['anomaly'] = val_anomalies
print("Validation Anomalies:\n", val_data[val_data['anomaly'] == -1])

# Test on test set
test_anomalies = iso_forest.predict(scaled_features_test)
test_data['anomaly'] = test_anomalies

# Output test results
print("Test Anomalies:\n", test_data[test_data['anomaly'] == -1])

# Plot the data
plt.scatter(data['span_duration'], data['metric_value'], c='blue', label='Normal')
plt.scatter(val_data[val_data['anomaly'] == -1]['span_duration'], val_data[val_data['anomaly'] == -1]['metric_value'], c='red', label='Validation Anomalies')
plt.scatter(test_data[test_data['anomaly'] == -1]['span_duration'], test_data[test_data['anomaly'] == -1]['metric_value'], c='green', label='Test Anomalies')
plt.xlabel('Span Duration')
plt.ylabel('Metric Value')
plt.legend()
plt.show()


# In[ ]:




