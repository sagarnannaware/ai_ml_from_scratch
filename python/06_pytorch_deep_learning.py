"""
PyTorch — Deep Learning

Description:
PyTorch is a flexible and fast deep learning library favored in research and prototyping. It offers dynamic computation graphs and strong GPU acceleration.

Key Functionalities:
- Dynamic computation graphs (eager execution)
- Neural network modules (torch.nn)
- GPU acceleration
- Custom layers and loss functions
- Research and production workflows

Sample Code:
"""

import torch
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 3)
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
input = torch.randn(1, 4)
output = model(input)
print("PyTorch model output:", output)