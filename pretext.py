import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset_helper import JigsawPuzzleDataset, RotateImageDataset
from torchvision import transforms
from torchvision.transforms import ToTensor, Normalize, Compose

# Set the device to GPU if available, else CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model class
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 28 * 28, 128)  # Adjust accordingly if the image size changes
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class RotationCNN(nn.Module):
    def __init__(self, num_classes=200):  # Default to 200 for CUB
        super(RotationCNN, self).__init__()
        # Load a pre-trained resnet18 model
        self.resnet = models.resnet18(pretrained=True)
        
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features  # Get the number of features going into the last layer
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)  # Replace it for 200 classes

    def forward(self, x):
        # Use the modified resnet to perform the forward pass
        return self.resnet(x)




model = RotationCNN(num_classes=4).to(device)
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Freeze all layers in the model
for param in model.resnet.parameters():
    param.requires_grad = False

# Unfreeze the last fully connected layer to allow training
model.resnet.fc.requires_grad = True


# Load the dataset and create dataloaders
# dataset = JigsawPuzzleDataset(root_dir="CUB_200_2011/images", transform=transforms.ToTensor())


dataset = RotateImageDataset(root_dir='CUB_200_2011/images')

train_size = int(len(dataset) * 0.7)
train_dataset, valid_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_dataloader = DataLoader(train_dataset, batch_size=50, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=50, shuffle=False)
print("data loaded")

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10, model_path='model.pth', device='mps'):
    # Move the model to the specified device
    model.to(device)
    
    for epoch in range(num_epochs):
        # Initialize training variables
        total_train_loss = 0
        total_valid_loss = 0
        correct = 0
        total = 0
        
        # Training phase
        try:
            model.train()  # Set the model to training mode
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()  # Zero the gradients
                outputs = model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute the loss
                loss.backward()  # Backpropagation
                optimizer.step()  # Update the model parameters
                total_train_loss += loss.item()  # Accumulate the training loss
        except Exception as e:
            print(f'An error occurred during training: {e}')

        # Calculate average training loss
        avg_train_loss = total_train_loss / len(train_loader) if train_loader else 0
        
        # Validation phase
        try:
            model.eval()  # Set the model to evaluation mode
            with torch.no_grad():  # Turn off gradients for validation, saves memory and computations
                for inputs, labels in valid_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)  # Forward pass
                    loss = criterion(outputs, labels)  # Compute the loss
                    total_valid_loss += loss.item()  # Accumulate the validation loss
                    _, predicted = torch.max(outputs.data, 1)  # Get predictions from the maximum value
                    total += labels.size(0)  # Total number of labels
                    correct += (predicted == labels).sum().item()  # Total correct predictions
        except Exception as e:
            print(f'An error occurred during validation: {e}')

        # Calculate average validation loss and accuracy
        avg_valid_loss = total_valid_loss / len(valid_loader) if valid_loader else 0
        accuracy = (correct / total) * 100 if total > 0 else 0
        
        # Print training/validation statistics
        print(f'Epoch {epoch+1}/{num_epochs} - Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Save the trained model
    try:
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path} after training.')
    except Exception as e:
        print(f'Failed to save the model: {e}')



train_model(model, train_dataloader, valid_dataloader, criterion, optimizer, num_epochs=10, device=device)


