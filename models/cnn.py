import torch
from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # Convolutional block 1
            # 3x32x32 -> 32x32x32   
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # 32x32x32 -> 32x16x16
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Convolutional block 2
            # 32x16x16 -> 64x12x12 (No padding)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0),
            # 64x12x12 -> 64x6x6
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()         
        )
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64 * 6 * 6, 512),  
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


