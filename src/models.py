import torch.nn as nn
import torch

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        super(SimpleNeuralNetwork, self).__init__()
        # Input layer
        layers = [nn.Linear(input_size, hidden_size), nn.Sigmoid()]

        # Hidden layers
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.Sigmoid())

        # Output layer
        layers.append(nn.Linear(hidden_size, num_classes))

        # Create sequential model
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
