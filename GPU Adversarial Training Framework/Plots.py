import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import resnet18

# Adversarial attack utility (PGD - Projected Gradient Descent)
def pgd_attack(model, inputs, labels, eps=0.3, alpha=0.01, iters=40):
    inputs = inputs.clone().detach().to(inputs.device)
    labels = labels.to(inputs.device)
    inputs.requires_grad = True
    for _ in range(iters):
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_data = inputs + alpha * inputs.grad.sign()
        eta = torch.clamp(adv_data - inputs.data, min=-eps, max=eps)
        inputs = torch.clamp(inputs.data + eta, 0, 1).detach_()
    return inputs

# Define the neural network (ResNet18)
def get_model():
    model = resnet18(pretrained=False, num_classes=10)
    return model

# Define the adversarial training function
def adversarial_training(model, train_loader, optimizer, device, eps=0.3, alpha=0.01, iters=40, epochs=5):
    model.train()
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Generate adversarial examples
            adv_inputs = pgd_attack(model, inputs, labels, eps, alpha, iters)

            # Forward pass
            outputs = model(adv_inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        loss_history.append(epoch_loss)
        accuracy_history.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

    return loss_history, accuracy_history

# Load CIFAR-10 dataset
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    return train_loader

# Plot training results
def plot_results(loss_history, accuracy_history):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss', color='blue', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.grid(True)
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracy_history, label='Training Accuracy', color='green', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_results.png')  # Save the plot as an image
    plt.show()

# Main function
def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    train_loader = load_data()

    # Initialize model, optimizer, and loss function
    model = get_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    loss_history, accuracy_history = adversarial_training(model, train_loader, optimizer, device, epochs=10)

    # Plot results
    plot_results(loss_history, accuracy_history)

if __name__ == "__main__":
    main()