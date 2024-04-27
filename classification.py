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
from torchvision.datasets import ImageFolder


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

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



def load_dataset(root_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = ImageFolder(root_dir, transform=transform)
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_model(model, train_loader, num_epochs=10, model_path='clasmodel.pth', device=device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # Fine-tune hyperparameters as needed

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}')

    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')

# Load the model
model = RotationCNN(num_classes=200)  # Change this if your dataset has a different number of classes
# Load the state dict with a filter to exclude the final layer weights if they don't match
state_dict = torch.load('model.pth')
print(state_dict)
state_dict_filtered = {k: v for k, v in state_dict.items() if 'fc' not in k}
model.load_state_dict(state_dict_filtered, strict=False)


# Unfreeze the last layer
for param in model.resnet.fc.parameters():
    param.requires_grad = True

# Load the dataset
train_loader = load_dataset('CUB_200_2011/images')

# Train the model
train_model(model, train_loader, device=device) 