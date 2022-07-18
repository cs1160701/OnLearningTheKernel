#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Create the feature map interface and some commonly used feature maps.

All attention implementations that expect a feature map shall receive a factory
function that returns a feature map instance when called with the query
dimensions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

gelu = lambda x: F.gelu(x) # GELU 
pelu = lambda x: F.elu(x) + 1 # Positive ELU 

class FeatureMap(nn.Module):
    """Define the FeatureMap interface."""
    def __init__(self, query_dims):
        super().__init__()
        self.query_dims = query_dims

    def new_feature_map(self, device, dtype):
        """Create a new instance of this feature map. In particular, if it is a
        random feature map sample new parameters."""
        raise NotImplementedError()

    def forward_queries(self, x):
        """Encode the queries `x` using this feature map."""
        return self(x)

    def forward_keys(self, x):
        """Encode the keys `x` using this feature map."""
        return self(x)

    def forward(self, x):
        """Encode x using this feature map. For symmetric feature maps it
        suffices to define this function, but for asymmetric feature maps one
        needs to define the `forward_queries` and `forward_keys` functions."""
        raise NotImplementedError()

    @classmethod
    def factory(cls, *args, **kwargs):
        """Return a function that when called with the query dimensions returns
        an instance of this feature map.

        It is inherited by the subclasses so it is available in all feature
        maps.
        """
        def inner(query_dims):
            return cls(query_dims, *args, **kwargs)
        return inner


class ActivationFunctionFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, activation_function):
        super().__init__(query_dims)
        self.activation_function = activation_function

    def new_feature_map(self, device, dtype):
        return

    def forward(self, x):
        return self.activation_function(x)


elu_feature_map = ActivationFunctionFeatureMap.factory(
    lambda x: torch.nn.functional.elu(x) + 1
)

class GeneralizedFeatureMap(FeatureMap):
    """Define a feature map that is simply an element-wise activation
    function."""
    def __init__(self, query_dims, hidden_dims, dropout=0.1,
                 hidden_activation=gelu, output_activation=pelu):
        super(GeneralizedFeatureMap, self).__init__(query_dims)

        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(query_dims, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, query_dims)
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def new_feature_map(self, device, dtype):
        return

    def forward(self, x):
        # Feedforward network 
        y = self.dropout(self.hidden_activation(self.linear1(x)))
        y = self.dropout(self.linear2(y))
        y = self.output_activation(y)
        return y
        
        # # Residual connection 
        # z = x + y 

        # # Represent output in the log-space
        # u = torch.log(self.output_activation(z))

        # # Compute offset 
        # norm_z_squared = torch.einsum("...d,...d->...", z, z).unsqueeze(-1)
        # offset = 0.5 * norm_z_squared

        # phi = torch.exp(u - offset) 

        # return phi
