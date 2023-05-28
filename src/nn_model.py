"""Multilayer Perceptron for regression model."""
import torch.nn as nn


class MultipleRegression(nn.Module):
    """Multilayer Perceptron for regression for arbitrary number of hidden
    layers.
    """

    def __init__(self, input_size, layers_config, activation_function):
        super().__init__()
        # If layers_config has type of int then make it a tuple
        if type(layers_config) == int:
            layers_config = (layers_config,)

        layers = []
        layers.append(nn.Linear(input_size, layers_config[0]))
        layers.append(activation_function)

        for k in range(len(layers_config) - 1):
            layers.append(nn.Linear(layers_config[k], layers_config[k + 1]))
            layers.append(activation_function)

        layers.append(nn.Linear(layers_config[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.net(x)
