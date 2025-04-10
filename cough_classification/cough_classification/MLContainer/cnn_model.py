import torch.nn as nn
import torch.optim as optim

# Define a simple CNN model for classifying 3 classes
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Convolutional layer: 3 input channels, 10 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        # Max pooling layer: 2x2 kernel, stride 2
        self.pool = nn.MaxPool2d(2, 2)
        # Calculate the number of features after pooling
        self.num_features = 10 * 150 * 300  # Adjusted based on output dimensions
        # Fully connected layer with the correct number of input features
        self.fc1 = nn.Linear(self.num_features, 3)  # Output: 3 classes

    def forward(self, x):
        # Apply convolution, activation, and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # print(f"Shape before flattening: {x.shape}")  # Debug: Check tensor shape
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, self.num_features)
        # Apply the fully connected layer
        x = self.fc1(x)
        return x