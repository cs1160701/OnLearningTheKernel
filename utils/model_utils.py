import math
import torch

import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """
    """
    def __init__(self, dim, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Parameters: 
        -----------
        x : tensor
            Input of shape (B, L, D) 
        """
        return x + self.pe[:x.size(1), :]

class ClassificationHead(nn.Module): 
    """
    """
    def __init__(self, in_features, out_features, n_classes, bias=True, 
                 weight_init=None, bias_init=None, activation='relu'): 
        super(ClassificationHead, self).__init__()

        self.layer1 = nn.Linear(in_features, out_features, bias=bias)
        self.layer2 = nn.Linear(out_features, n_classes, bias=bias)
        self.activation = F.relu if activation == "relu" else F.gelu

        # Custom initialization 
        if weight_init is not None or bias_init is not None: 
            self.reset_parameters(bias, weight_init, bias_init)

    def reset_parameters(self, bias, weight_initializer, bias_initializer): 

        # Weight matrices 
        if weight_initializer == 'xavier_uniform': 
            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)
        elif weight_initializer == 'xavier_normal':
            nn.init.xavier_normal_(self.layer1.weight)
            nn.init.xavier_normal_(self.layer2.weight) 

        # Bias vector 
        if bias and bias_initializer == 'zeros': 
            nn.init.zeros_(self.layer1.bias)
            nn.init.zeros_(self.layer2.bias)
        elif bias and bias_initializer == 'normal': 
            nn.init.normal_(self.layer1.bias, std=1e-3)
            nn.init.normal_(self.layer2.bias, std=1e-3)

    def forward(self, x): 
        # Run fully-connected network 
        x = self.activation(self.layer1(x))
        x = self.layer2(x)

        return x

class DualClassificationHead(nn.Module): 
    """
    """
    def __init__(self, in_features, out_features, n_classes, bias=True, 
                 weight_init=None, bias_init=None, activation='relu', 
                 interaction=None): 
        super(DualClassificationHead, self).__init__()

        self.interaction = interaction

        if self.interaction == 'NLI': 
            self.layer1 = nn.Linear(4*in_features, out_features, bias=bias)
        else:
            self.layer1 = nn.Linear(2*in_features, out_features, bias=bias)

        self.layer2 = nn.Linear(out_features, out_features//2, bias=bias)
        self.layer3 = nn.Linear(out_features//2, n_classes, bias=bias)
        
        self.activation = F.relu if activation == "relu" else F.gelu

        # Custom initialization 
        if weight_init is not None or bias_init is not None: 
            self.reset_parameters(bias, weight_init, bias_init)

    def reset_parameters(self, bias, weight_initializer, bias_initializer): 

        # Weight matrices 
        if weight_initializer == 'xavier_uniform': 
            nn.init.xavier_uniform_(self.layer1.weight)
            nn.init.xavier_uniform_(self.layer2.weight)
            nn.init.xavier_uniform_(self.layer3.weight)
        elif weight_initializer == 'xavier_normal':
            nn.init.xavier_normal_(self.layer1.weight)
            nn.init.xavier_normal_(self.layer2.weight) 
            nn.init.xavier_normal_(self.layer3.weight) 

        # Bias vector 
        if bias and bias_initializer == 'zeros': 
            nn.init.zeros_(self.layer1.bias)
            nn.init.zeros_(self.layer2.bias)
            nn.init.zeros_(self.layer3.bias)
        elif bias and bias_initializer == 'normal': 
            nn.init.normal_(self.layer1.bias, std=1e-3)
            nn.init.normal_(self.layer2.bias, std=1e-3)
            nn.init.normal_(self.layer3.bias, std=1e-3)

    def forward(self, x, y): 
        # Concatenate inputs 
        if self.interaction == 'NLI': 
            z = torch.cat([x, y, x * y, x - y], dim=1)
        else:
            z = torch.cat([x, y], dim=1)

        # Run fully-connected network 
        z = self.activation(self.layer1(z))
        z = self.activation(self.layer2(z))
        z = self.layer3(z)

        return z
