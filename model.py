from typing import Union

import torch
import torch.nn as nn
from torch.nn import functional
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size:int, action_size:int, seed:int, hidden_sizes:Union[list,tuple] = [64,32,16], dropout:float = 0.01):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # First layer
        self.hidden_layers = ModuleList([Linear(state_size,hidden_sizes[0])])
        
        # Adding layer hidden layers based on hidden_sizes
        layer_sizes = zip(hidden_sizes[:-1],hidden_sizes[1:])
        self.hidden_layers.extend([Linear(in_size,out_size) for in_size,out_size in layer_sizes])
        
        # Output layer
        self.output_layer = Linear(hidden_sizes[-1],action_size)
        
        # Dropout layer
        self.dropout_layer = Dropout(p = dropout)            
        
    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        for layer in self.hidden_layers:
            state = functional.relu(layer(state))
            state = self.dropout_layer(state)
            
        return self.output_layer(state)
