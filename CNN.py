# Import the main PyTorch library
import torch

# Import neural network module (contains layers like Conv2d, Linear, etc.)
import torch.nn as nn

# Import optimization algorithms (like Adam, SGD)
import torch.optim as optim

# Import datasets and image transformations
from torchvision import datasets, transforms

# DataLoader helps load data in batches
from torch.utils.data import DataLoader

# Detect device (we force CPU to keep it busy)
device = torch.device("cpu")

# -------------------------------
# IMAGE TRANSFORMATIONS
# -------------------------------

# Convert images to PyTorch tensors
# Normalize pixel values from [0,255] → [0,1]
transform = transforms.ToTensor()

# -------------------------------
# LOAD DATASET
# -------------------------------

# Download CIFAR-10 training dataset
train_dataset = datasets.CIFAR10(
    root="./data",          # folder where dataset is stored
    train=True,             # this is training data
    download=True,          # download if not already present
    transform=transform     # apply transformation
)

# Download test dataset
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,            # test data
    download=True,
    transform=transform
)

# Create training DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=128,         # number of images per batch
    shuffle=True,           # shuffle data every epoch
    num_workers=4           # use CPU cores to load data faster
)

# Create test DataLoader
test_loader = DataLoader(
    test_dataset,
    batch_size=1000
)

# -------------------------------
# BUILD CNN MODEL
# -------------------------------

# Define neural network class
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # First convolution layer:
        # 3 input channels (RGB)
        # 32 output feature maps
        # 3x3 filter size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)

        # Second convolution layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max pooling layer reduces image size by half
        self.pool = nn.MaxPool2d(2, 2)

        # Fully connected layer
        # After pooling twice: image becomes 8x8
        # 64 feature maps × 8 × 8
        self.fc1 = nn.Linear(64 * 8 * 8, 512)

        # Final output layer (10 classes)
        self.fc2 = nn.Linear(512, 10)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):

        # Pass input through first convolution
        x = self.relu(self.conv1(x))

        # Reduce spatial size
        x = self.pool(x)

        # Second convolution
        x = self.relu(self.conv2(x))

        # Reduce size again
        x = self.pool(x)

        # Flatten tensor into vector
        x = x.view(x.size(0), -1)

        # First fully connected layer
        x = self.relu(self.fc1(x))

        # Final classification layer
        x = self.fc2(x)

        return x


# Create model instance
model = CNN().to(device)

# -------------------------------
# LOSS + OPTIMIZER
# -------------------------------

# CrossEntropyLoss automatically applies Softmax
criterion = nn.CrossEntropyLoss()

# Adam optimizer updates weights using gradients
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# TRAINING LOOP
# -------------------------------

epochs = 10

for epoch in range(epochs):

    model.train()  # set model to training mode
    running_loss = 0

    for images, labels in train_loader:

        # Move data to CPU
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Reset gradients to zero
        optimizer.zero_grad()

        # Backpropagation (compute gradients)
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss:.4f}")

# -------------------------------
# TESTING
# -------------------------------

model.eval()  # evaluation mode

correct = 0
total = 0

with torch.no_grad():  # disable gradient calculation

    for images, labels in test_loader:

        outputs = model(images)

        # Get class with highest probability
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}%")
