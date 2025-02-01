import matplotlib.pyplot as plt
import pandas as pd

# Load top talkers data
top_talkers = pd.read_csv('top_talkers.csv')

# Plot
plt.figure(figsize=(10, 6))
plt.barh(top_talkers['src_ip'], top_talkers['bytes'], color='skyblue')
plt.xlabel('Total Bytes Transferred')
plt.ylabel('Source IP Address')
plt.title('Top Talkers by Traffic Volume')
plt.savefig('top_talkers.png')

# Load protocol distribution data
protocol_dist = pd.read_csv('protocol_distribution.csv')

# Plot
plt.figure(figsize=(8, 8))
plt.pie(protocol_dist['count'], labels=protocol_dist['protocol'], autopct='%1.1f%%', startangle=140)
plt.title('Traffic Distribution by Protocol')
plt.savefig('protocol_distribution.png')


# Load anomalies data
anomalies = pd.read_csv('network_anomalies.csv')

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(anomalies['src_ip'], anomalies['z_score'], c=anomalies['z_score'], cmap='Reds', s=100)
plt.xlabel('Source IP Address')
plt.ylabel('Z-Score (Anomaly Score)')
plt.title('Network Anomalies Detected')
plt.colorbar(label='Z-Score')
plt.savefig('network_anomalies.png')


# Aggregate traffic by hour
df['hour'] = df['timestamp'].dt.hour
hourly_traffic = df.groupby('hour')['bytes'].sum().compute()

# Plot
plt.figure(figsize=(10, 6))
plt.plot(hourly_traffic.index, hourly_traffic.values, marker='o', linestyle='-')
plt.xlabel('Hour of Day')
plt.ylabel('Total Bytes Transferred')
plt.title('Network Traffic Over Time')
plt.grid(True)
plt.savefig('traffic_over_time.png')


# Plot histogram of z-scores
plt.figure(figsize=(10, 6))
plt.hist(anomalies['z_score'], bins=30, color='orange', edgecolor='black')
plt.xlabel('Z-Score (Anomaly Score)')
plt.ylabel('Frequency')
plt.title('Distribution of Anomaly Scores')
plt.savefig('anomaly_distribution.png')
