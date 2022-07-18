"""
Implement the Fastfood approximation to kernel expansions from the paper 
"Fastfood - Approximating Kernel Expansions in Loglinear Time".

Part of this code has been adapted from this source 
https://github.com/HazyResearch/structured-nets/blob/master/pytorch/structure/hadamard.py
"""
import torch
import scipy
import warnings
import numpy as np 

from torch.nn import init
from math import sqrt, log
from scipy.stats import chi 
from torch.nn.parameter import Parameter

from .base import FeatureMap

def hadamard_transform(u, normalize=False):
    """Multiply H_n @ u where H_n is the Hadamard matrix of dimension n x n.
    n must be a power of 2.
    Parameters:
        u: Tensor of shape (..., n)
        normalize: if True, divide the result by 2^{m/2} where m = log_2(n).
    Returns:
        product: Tensor of shape (..., n)
    """
    _, n = u.shape
    m = int(np.log2(n))
    assert n == 1 << m, 'n must be a power of 2'
    x = u[..., np.newaxis]
    for d in range(m)[::-1]:
        x = torch.cat((x[..., ::2, :] + x[..., 1::2, :], x[..., ::2, :] - x[..., 1::2, :]), dim=-1)
    return x.squeeze(-2) / 2**(m / 2) if normalize else x.squeeze(-2)


class FastFoodRandomFeatures(FeatureMap): 
    """
    Random Fastfood features for the RBF kernel according to [1].

    [1]: "Fastfood - Approximating Kernel Expansions in Loglinear Time" 
    by Quoc Le, Tamas Sarlos and Alexander Smola.

    Arguments
    ---------
        query_dimensions: int 
            The input query dimensions.
        softmax_temp: float 
            The temerature for the Gaussian kernel approximation 
            exp(-t * ||x-y||^2). (default: 1/sqrt(query_dimensions))
    """

    def __init__(self, query_dimensions, n_samples=None, softmax_temp=None, 
                 learn_S=False, learn_G_B=False): 
        super(FastFoodRandomFeatures, self).__init__(query_dimensions)

        # Check Fastfood condition 
        if n_samples is not None: 
            if n_samples < query_dimensions: 
                raise RuntimeError(('The dimension of the feature map must be '
                                    'greater or equal than the input dimension.'))

        self.learn_S = learn_S
        self.learn_G_B = learn_G_B
        self.n_samples = n_samples or query_dimensions # Currently ignored! 
        self.softmax_temp = softmax_temp or 1/sqrt(query_dimensions)

        # Declare structured matrices 
        self.P = None 

        if self.learn_G_B:
            self.B = Parameter(torch.Tensor(self.query_dims)) 
            self.G = Parameter(torch.Tensor(self.query_dims)) 
            init.normal_(self.B, std=sqrt(1./self.query_dims))
            init.normal_(self.G, std=sqrt(1./self.query_dims))
        else: 
            self.B = None 
            self.G = None 

        if self.learn_S: 
            self.S = Parameter(torch.Tensor(self.query_dims)) 
            init.normal_(self.S, std=sqrt(1./self.query_dims))
        else: 
            self.S = None 

    def new_feature_map(self, device, dtype): 
        # Permutation matrix P 
        self.P = torch.randperm(
            self.query_dims, 
            device=device 
        )

        if not self.learn_G_B:
            # Binary scaling matrix B 
            self.B = torch.tensor(
                np.random.choice([-1.0, 1.0], 
                    size=self.query_dims
                ),
                dtype=dtype, 
                device=device, 
                requires_grad=True
            )

            # Gaussian scaling matrix G 
            self.G = torch.zeros(
                self.query_dims, 
                dtype=dtype,
                device=device
            )
            self.G.normal_()

        if not self.learn_S: 
            # Scaling matrix S
            self.S = torch.tensor(
                chi.rvs( 
                    df=self.query_dims, 
                    size=self.query_dims
                ), 
                dtype=dtype,
                device=device 
            ) / torch.norm(self.G)

    def forward(self, x): 
        """
        Compute the FastFood feature map for the given input. 

        Arguments: 
        ----------
        x : (N, L, H, D)
            The input tensor.
        """
        # Original shape
        x_shape = x.shape
        x = x * sqrt(self.softmax_temp)

        # Reshape for Fastfood
        x = x.view(-1, self.query_dims)

        # Fastfood multiplication
        Bx = x * self.B
        HBx = hadamard_transform(Bx) 
        PHBx = HBx[:,self.P]
        GPHBx = PHBx * self.G
        HGPHBx = hadamard_transform(GPHBx)
        SHGPHBx = HGPHBx * self.S

        # Normalize and recover original shape
        Vx = (sqrt(1.0/self.query_dims) * SHGPHBx).view(x_shape)

        # Feature map
        phi = torch.cat([torch.cos(Vx), torch.sin(Vx)], dim=-1)
        phi = sqrt(1.0/self.query_dims) * phi
        return phi

class SmoothedFastFoodRandomFeatures(FastFoodRandomFeatures):
    """Simply add a constant value to the dot product in order to avoid
    possible numerical instabilities when the feature map is slightly
    negative.

    Implements K(x, y) = exp(-|x-y|^2) + s.

    Arguments
    ---------
        query_dimensions: int, 
            The input query dimensions.
        softmax_temp: float 
            The temerature for the Gaussian kernel approximation 
            exp(-t * ||x-y||^2). (default: 1/sqrt(query_dimensions))
        smoothing: float
            The smoothing parameter to add to the dot product.
    """
    def __init__(self, query_dimensions, n_samples=None, softmax_temp=None, 
                 learn_S=False, learn_G_B=False, smoothing=1.0):
        super(SmoothedFastFoodRandomFeatures, self).__init__(
            query_dimensions,
            n_samples=None, # Currently ignored! 
            softmax_temp=softmax_temp, 
            learn_S=learn_S, 
            learn_G_B=learn_G_B
        )
        self.smoothing = smoothing

    def forward(self, x):
        y = super().forward(x)
        smoothing = torch.full(
            y.shape[:-1] + (1,),
            self.smoothing,
            dtype=y.dtype,
            device=y.device
        )
        return torch.cat([y, smoothing], dim=-1)

class FastFoodPositiveFeatures(FastFoodRandomFeatures):
    def __init__(self, query_dimensions, n_samples=None, softmax_temp=None, 
                 learn_S=False, learn_G_B=False):
        super(SmoothedFastFoodRandomFeatures, self).__init__(
            query_dimensions,
            n_samples=None, # Currently ignored! 
            softmax_temp=softmax_temp, 
            learn_S=learn_S, 
            learn_G_B=learn_G_B
        )

    def forward(self, x): 
        """
        Compute the FastFood feature map for the given input. 

        Arguments: 
        ----------
        x : (N, L, H, D)
            The input tensor.
        """        
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)

        # Reshape for Fastfood
        x_shape = x.shape
        x = x.view(-1, self.query_dims)

        # Fastfood multiplication
        Bx = x * self.B
        HBx = hadamard_transform(Bx) 
        PHBx = HBx[:,self.P]
        GPHBx = PHBx * self.G
        HGPHBx = hadamard_transform(GPHBx)
        SHGPHBx = HGPHBx * self.S

        # Normalize and recover original shape
        Vx = (sqrt(1.0/self.query_dims) * SHGPHBx).view(x_shape)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared + 0.5 * log(self.query_dims)

        return torch.exp(Vx - offset)
