#!/usr/bin/env python3
# IoT IDS with Adversarial Training (PGD)
# Compatible with Raspberry Pi 4/5 (ARM64)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time

# Hardware Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64  # Reduced for RPi memory constraints
EPSILON = 0.1    # PGD attack strength
ALPHA = 0.01     # PGD step size
ITERATIONS = 7   # PGD attack iterations

# 1. Data Preparation (N-BaIoT Dataset)
def load_data():
    # Load preprocessed IoT dataset (example structure)
    data = pd.read_csv("n-baiot.csv")
    
    # Feature engineering for IoT devices
    features = data.drop(['label', 'device'], axis=1)
    labels = data['label'].values
    
    # Train-test split with temporal validation
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, shuffle=False
    )
    
    # Normalization critical for NN performance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, y_train), (X_test, y_test)

# 2. Lightweight Neural Network for RPi
class IoTIDSModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

# 3. Adversarial Training with PGD
class PGDAdversary:
    def __init__(self, model, epsilon=EPSILON, alpha=ALPHA, iterations=ITERATIONS):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
        
    def generate_attack(self, X, y):
        X_adv = X.clone().detach().requires_grad_(True)
        
        for _ in range(self.iterations):
            loss = nn.BCELoss()(self.model(X_adv).squeeze(), y.float())
            loss.backward()
            
            with torch.no_grad():
                perturbation = self.alpha * X_adv.grad.sign()
                X_adv += perturbation
                # Project back to epsilon ball
                eta = torch.clamp(X_adv - X, -self.epsilon, self.epsilon)
                X_adv = X + eta
            
            X_adv.grad.zero_()
        
        return X_adv.detach()

# 4. Training Loop with Adversarial Examples
def train(model, train_loader, test_loader, epochs=10):
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    adversary = PGDAdversary(model)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            
            # Generate adversarial examples
            X_adv = adversary.generate_attack(X_batch, y_batch)
            
            # Combined loss
            optimizer.zero_grad()
            loss_clean = nn.BCELoss()(model(X_batch).squeeze(), y_batch.float())
            loss_adv = nn.BCELoss()(model(X_adv).squeeze(), y_batch.float())
            loss = (loss_clean + loss_adv) / 2
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            acc = []
            for X_val, y_val in test_loader:
                X_val, y_val = X_val.to(DEVICE), y_val.to(DEVICE)
                outputs = model(X_val).squeeze()
                acc.append(((outputs > 0.5).float() == y_val).float().mean().item())
                
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {np.mean(acc):.4f}")

# 5. Raspberry Pi Deployment Setup
def deploy_to_pi(model):
    # Optimize for ARM processors
    model = model.to("cpu")
    example_input = torch.rand(1, input_dim)
    
    # Export for edge deployment
    torchscript_model = torch.jit.trace(model, example_input)
    torchscript_model.save("iot_ids_model.pt")
    print("Model exported for Raspberry Pi deployment")

# Main Execution
if __name__ == "__main__":
    # Load and preprocess data
    (X_train, y_train), (X_test, y_test) = load_data()
    input_dim = X_train.shape[1]
    
    # Convert to PyTorch tensors
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = IoTIDSModel(input_dim).to(DEVICE)
    
    # Train with adversarial examples
    train(model, train_loader, test_loader, epochs=15)
    
    # Prepare for deployment
    deploy_to_pi(model)
    
    # Raspberry Pi Inference Script (separate file)
    print("\nRPi Inference Script:")
    print("""
    import torch
    from torch import nn

    class IoTIDSModel(nn.Module):
        # Include model definition here
        
    def preprocess(input_data):
        # Add your preprocessing logic
        
    model = torch.jit.load('iot_ids_model.pt')
    with torch.no_grad():
        outputs = model(torch.FloatTensor(preprocessed_data))
        prediction = (outputs > 0.5).item()
    """)
    
    
    
    # Key Features:

# Hardware Optimization: Reduced batch size and model complexity for RPi compatibility

# Adversarial Robustness: Implements PGD attack during training

# Edge Deployment: Includes TorchScript export for ARM deployment

# IoT-Specific Features:

# Temporal validation split

# Lightweight network architecture

# Feature engineering guidance

# ARM-optimized model export

# To use this script:

# Prepare N-BaIoT dataset (preprocessed CSV)

# Install dependencies: pip install torch pandas scikit-learn

# Run on RPi: python3 iot_ids_train.py

# For production deployment:

# Use libtorch for C++ inference

# Implement MQTT integration for real-time monitoring

# Add hardware security measures (TPM integration)