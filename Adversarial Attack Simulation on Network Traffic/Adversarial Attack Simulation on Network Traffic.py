import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scapy.all import *
from scapy.layers.inet import IP, TCP, UDP

# Configuration
DATA_PATH = "path/to/MachineLearningCSV/MachineLearningCVE.csv"
SELECTED_FEATURES = ['Source Port', 'Destination Port', 'Protocol', 'Packet Length', 'TCP Flags']
LABEL_COLUMN = 'Label'
EPSILON = 0.1  # Attack strength
PCAP_SIZE = 100  # Number of packets to generate

# Helper functions
def preprocess_data(data):
    """Preprocess dataset and return features/labels"""
    # Convert labels to binary (0: benign, 1: malicious)
    data[LABEL_COLUMN] = data[LABEL_COLUMN].apply(lambda x: 0 if x == 'BENIGN' else 1)
    
    # Handle infinite values
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Select and normalize features
    X = data[SELECTED_FEATURES]
    y = data[LABEL_COLUMN]
    
    # Convert flags to numeric (sum of TCP flags)
    if 'TCP Flags' in SELECTED_FEATURES:
        X['TCP Flags'] = X['TCP Flags'].fillna(0).astype(int)
    
    return X, y

def create_packet(features):
    """Create Scapy packet from feature vector"""
    sport = max(0, min(65535, int(features[0])))
    dport = max(0, min(65535, int(features[1])))
    protocol = int(features[2])
    length = max(60, min(65535, int(features[3])))  # Minimum packet size 60 bytes
    flags = max(0, min(255, int(features[4]))) if len(features) > 4 else 0

    ip = IP(src="192.168.1.100", dst="10.0.0.100", len=length, proto=protocol)
    
    if protocol == 6:
        return ip/TCP(sport=sport, dport=dport, flags=flags)
    elif protocol == 17:
        return ip/UDP(sport=sport, dport=dport)
    else:
        return ip/Raw(load="Perturbed Packet")

# Neural Network Model
class IDSModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Adversarial Attack Function
def fgsm_attack(model, X, y, epsilon):
    X_tensor = torch.tensor(X, dtype=torch.float32, requires_grad=True)
    y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)

    outputs = model(X_tensor)
    loss = nn.BCELoss()(outputs, y_tensor)
    loss.backward()

    perturbation = epsilon * X_tensor.grad.data.sign()
    X_adv = X_tensor + perturbation
    return X_adv.detach().numpy()

def main():
    # Load and preprocess data
    data = pd.read_csv(DATA_PATH)
    X, y = preprocess_data(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train surrogate model
    model = IDSModel(X_train_scaled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensors
    train_X = torch.tensor(X_train_scaled, dtype=torch.float32)
    train_y = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)

    # Training loop
    model.train()
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(train_X)
        loss = nn.BCELoss()(outputs, train_y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Generate adversarial examples
    malicious_idx = y_test[y_test == 1].index[:PCAP_SIZE]
    X_malicious = scaler.transform(X_test.loc[malicious_idx])
    X_adv = fgsm_attack(model, X_malicious, y_test.loc[malicious_idx], EPSILON)
    
    # Denormalize
    X_adv_denorm = scaler.inverse_transform(X_adv)
    X_orig_denorm = scaler.inverse_transform(X_malicious)

    # Create packets
    original_pkts = [create_packet(row) for row in X_orig_denorm]
    adversarial_pkts = [create_packet(row) for row in X_adv_denorm]

    # Save PCAPs
    wrpcap('original_traffic.pcap', original_pkts)
    wrpcap('adversarial_traffic.pcap', adversarial_pkts)
    print("PCAP files generated successfully!")

if __name__ == "__main__":
    main()
    
    
    
    
    # suricata -c /etc/suricata/suricata.yaml -r original_traffic.pcap
# suricata -c /etc/suricata/suricata.yaml -r adversarial_traffic.pcap



# The script assumes binary classification of network traffic

# Features are normalized before training

# FGSM attack perturbs features to evade detection

# Scapy creates realistic network packets for testing

# Results should show reduced alert counts for adversarial traffic