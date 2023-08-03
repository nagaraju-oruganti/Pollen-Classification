import torch
import torch.nn as nn

class Baseline3DConvNet(nn.Module):
    def __init__(self, config, num_classes):
        super(Baseline3DConvNet, self).__init__()
        
        in_channels = 20

        # Define the 3D convolution layers
        self.conv1 = nn.Conv3d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        # Define the pooling layer
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        #self.pool2 = nn.MaxPool3d(kernel_size=2, stride=1)

        self.dropout = nn.Dropout(0.2)
        
        # Define the fully connected layers
        self.fc1 = nn.Linear(64 * 1 * 112 * 112, 128)  # Adjust the input size based on your image dimensions
        self.fc2 = nn.Linear(128, num_classes)
        
    def loss_fn(self, logits, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        return loss
        
    def forward(self, x, y):
        
        # Apply 3D convolutions and pooling
        x = self.pool1(torch.relu(self.conv1(x)))
        x = torch.relu(self.conv2(x))

        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 64 * 1 * 112 * 112)

        # Apply fully connected layers
        x = self.dropout(x)                                 # added in aug_0.5_v2 version
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        loss = self.loss_fn(logits= logits, y = y)
        return logits, loss
