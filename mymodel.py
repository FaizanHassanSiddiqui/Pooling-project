
import torch
from torch import nn

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        
        super().__init__()
        
        
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        
        
    def forward(self, x):
    

        for linear in self.hidden_layers:
            x = torch.tanh(linear(x))
            
        x = self.output(x)

        x = torch.sigmoid(x)
    
         
        return x

