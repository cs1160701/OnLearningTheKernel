#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

import sys
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt, log
from torch.nn.parameter import Parameter

from .base import FeatureMap

def orthogonal_random_matrix_(w):
    """Initialize the matrix w in-place to compute orthogonal random features.

    The matrix is initialized such that its columns are orthogonal to each
    other (in groups of size `rows`) and their norms is drawn from the
    chi-square distribution with `rows` degrees of freedom (namely the norm of
    a `rows`-dimensional vector distributed as N(0, I)).

    Arguments
    ---------
        w: float tensor of size (rows, columns)
    """
    rows, columns = w.shape

    if rows == columns: 
        block = torch.randn(rows, rows, device=w.device)
        norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
        Q, _ = torch.qr(block)
        w = Q * norms[None]
    else: 
        start = 0
        while start < columns:
            end = min(start+rows, columns)
            block = torch.randn(rows, rows, device=w.device)
            norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
            Q, _ = torch.qr(block)
            w[:, start:end] = (
                Q[:, :end-start] * norms[None, :end-start]
            )
            start += rows

class GeneratorLayer(nn.Module): 
    def __init__(self, in_features, out_features, bias=True, 
                 batch_norm=True, activation='leaky'): 
        super(GeneratorLayer, self).__init__()

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.batch_norm = nn.BatchNorm1d(out_features) if batch_norm else nn.Identity()
        if activation == 'relu': 
            self.activation = F.relu
        elif activation == 'leaky': 
            self.activation = F.leaky_relu
        elif activation == 'gelu': 
            self.activation = F.gelu
        elif activation == 'tanh': 
            self.activation = F.tanh
        elif activation == 'elu': 
            self.activation = F.elu
        elif activation == 'linear':
            self.activation = nn.Identity()
        else: 
            raise ValueError("Unsupported activation: %s" % activation)

        self.reset_parameters()

    def reset_parameters(self): 
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x): 
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x

class GeneratorBlock(nn.Module): 
    def __init__(self, layer_dims, output_dim, hidden_act='leaky', 
                 output_act='tanh'):
        super(GeneratorBlock, self).__init__()

        generator_layers = []
        # Hidden layers
        for in_size, out_size in zip(layer_dims, layer_dims[1:]): 
            generator_layers.append(
                GeneratorLayer(
                    in_size, 
                    out_size, 
                    activation=hidden_act
                )
            )
        # Last layer 
        generator_layers.append(
            GeneratorLayer(
                layer_dims[-1],
                output_dim, 
                batch_norm=False, 
                activation=output_act
            )
        )
        self.generator_layers = nn.ModuleList(generator_layers) 

    def forward(self, x): 
        for layer in self.generator_layers: 
            x = layer(x)
        return x

class GenerativeRandomFourierFeatures(FeatureMap): 
    """
    """
    def __init__(self, query_dimensions, noise_dims, 
                 n_dims=None, softmax_temp=None, 
                 redraw=1, deterministic_eval=False): 
        super(GenerativeRandomFourierFeatures, self).__init__(
            query_dimensions
        )

        self.noise_dims = noise_dims
        self.n_dims = n_dims or query_dimensions
        self.softmax_temp = softmax_temp or 1/sqrt(query_dimensions)
        self.redraw = redraw
        self.deterministic_eval = deterministic_eval
        
        # Generator network 
        self.generator = GeneratorBlock(
            noise_dims, 
            query_dimensions, 
            hidden_act='leaky', 
            output_act='tanh'
        )    

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            'omega',
            torch.zeros(self.n_dims//2, self.noise_dims[0])
        )

        # Buffer for storing the counter 
        self.register_buffer(
            '_calls', 
            torch.tensor(-1, dtype=torch.int)
        )

    def new_feature_map(self, device, dtype):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1

        if (self._calls % self.redraw) != 0:
            return

        omega = torch.zeros(
            self.n_dims//2,
            self.noise_dims[0],
            dtype=dtype,
            device=device
        )
        omega.normal_()

        self.register_buffer('omega', omega)

    def forward(self, x): 
        # Run the generator 
        omega = self.generator(self.omega)

        # Scale input 
        x = x * sqrt(self.softmax_temp)

        # Compute feature map 
        u = torch.matmul(
            x.unsqueeze(-2), 
            omega.transpose(0,1)
        ).squeeze(-2)

        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2/self.n_dims)

class SmoothedGenerativeRandomFourierFeatures(GenerativeRandomFourierFeatures):
    """
    """
    def __init__(self, query_dimensions, noise_dims, n_dims=None, 
                 softmax_temp=None, smoothing=1.0, redraw=1, 
                 deterministic_eval=False):
        super(SmoothedGenerativeRandomFourierFeatures, self).__init__(
            query_dimensions, noise_dims=noise_dims,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp, redraw=redraw, 
            deterministic_eval=deterministic_eval
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

class GenerativePositiveRandomFeatures(GenerativeRandomFourierFeatures): 
    """
    """
    def __init__(self, query_dimensions, noise_dims, n_dims=None, 
                 softmax_temp=None, stabilize=False, redraw=1, 
                 deterministic_eval=False): 
        super(GenerativePositiveRandomFeatures, self).__init__(
            query_dimensions, noise_dims=noise_dims,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp, redraw=redraw,
            deterministic_eval=deterministic_eval
        )

        self.stabilize = stabilize
        
        # Generator network 
        self.generator = GeneratorBlock(
            noise_dims, 
            query_dimensions, 
            hidden_act='leaky', 
            output_act='tanh'
        )    

    def forward(self, x): 
        # Scale input 
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)

        # Run the generator 
        omega = self.generator(self.omega)

        # Compute feature map 
        u = torch.matmul(
            x.unsqueeze(-2), 
            omega.transpose(0,1)
        ).squeeze(-2)

        # Compute the exponential offset 
        offset = norm_x_squared + 0.5 * log(self.n_dims)

        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]

        return torch.exp(u - offset)

class RandomFourierFeatures(FeatureMap):
    """Random Fourier Features for the RBF kernel according to [1].

    [1]: "Weighted Sums of Random Kitchen Sinks: Replacing minimization with
         randomization in learning" by A. Rahimi and Benjamin Recht.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=False, redraw=1, deterministic_eval=False):
        super(RandomFourierFeatures, self).__init__(query_dimensions)

        self.n_dims = n_dims or query_dimensions
        self.orthogonal = orthogonal
        self.softmax_temp = (
            1/sqrt(query_dimensions) if softmax_temp is None
            else softmax_temp
        )

        self.redraw = redraw
        self.deterministic_eval = deterministic_eval

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            'omega',
            torch.zeros(self.query_dims, self.n_dims//2)
        )

        # Buffer for storing the counter 
        self.register_buffer(
            '_calls', 
            torch.tensor(-1, dtype=torch.int)
        )

    def new_feature_map(self, device, dtype):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1

        if (self._calls % self.redraw) != 0:
            return

        omega = torch.zeros(
            self.query_dims,
            self.n_dims//2,
            dtype=dtype,
            device=device
        )

        if self.orthogonal:
            orthogonal_random_matrix_(omega)
        else:
            omega.normal_()

        self.register_buffer('omega', omega)

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        phi = torch.cat([torch.cos(u), torch.sin(u)], dim=-1)
        return phi * sqrt(2/self.n_dims)


class SmoothedRandomFourierFeatures(RandomFourierFeatures):
    """Simply add a constant value to the dot product in order to avoid
    possible numerical instabilities when the feature map is slightly
    negative.

    Implements K(x, y) = exp(-|x-y|^2) + s.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the Gaussian kernel
                      approximation exp(-t * |x-y|^2)
                      (default: 1/sqrt(query_dimensions))
        orthogonal: bool, When True the random matrix is initialized for
                    orthogonal random features to reduce the approximation
                    variance (default: False)
        smoothing: float, The smoothing parameter to add to the dot product.
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=False, smoothing=1.0, redraw=1, 
                 deterministic_eval=False):
        super(SmoothedRandomFourierFeatures, self).__init__(
            query_dimensions,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal,
            redraw=redraw,
            deterministic_eval=deterministic_eval
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

class Favor(RandomFourierFeatures):
    """Positive orthogonal random features that approximate the softmax kernel.

    Basically implementation of Lemma 1 from "Rethinking Attention with
    Performers".

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=True, stabilize=False, redraw=1, 
                 deterministic_eval=False):
        super(Favor, self).__init__(query_dimensions, n_dims=n_dims,
                                    softmax_temp=softmax_temp,
                                    orthogonal=orthogonal, redraw=redraw, 
                                    deterministic_eval=deterministic_eval)
        self.stabilize = stabilize

    def _check_sequence_length(self, x):
        """Check that the 2nd dimension is larger than the 3rd as a heuristic
        that the sequence length will be larger than the number of heads. If
        not simply warn of a possible bug."""
        if len(x.shape) != 4:
            warnings.warn(("Favor.stabilize is set to True but the input "
                           "feature does not have the shape (N, L, H, D) "
                           "which may result in unexpected behaviour"))

        if x.shape[1] < x.shape[2]:
            warnings.warn(("Favor.stabilize is set to True but the 2nd "
                           "dimension of the input is smaller than the 3rd "
                           "which could indicate that the sequence length and "
                           "the heads are flipped. This may result in incorrect "
                           "behaviour. The shape of the input is "
                           "{!r}.").format(x.shape))

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared/2)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared * 0.5 + 0.5 * log(self.n_dims)

        # If stabilize is True then add the max norm per sequence in order to
        # ensure that exp_u1 and exp_u2 will be <1.
        #
        # NOTE: This is the only part of this feature map that assumes the
        #       2nd dimension is the sequence length. We call the
        #       _check_sequence_length dimension function to be able to catch
        #       some possible bugs ahead of time.
        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]

        exp_u1 = torch.exp(u - offset)
        exp_u2 = torch.exp(-u - offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)

        return phi

class GaussianPRF(Favor):
    """Positive orthogonal random features that approximate the RBF kernel.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=True, stabilize=False, redraw=1,
                 deterministic_eval=False):
        super(GaussianPRF, self).__init__(
            query_dimensions, n_dims=n_dims,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal, stabilize=stabilize, 
            redraw=redraw, deterministic_eval=deterministic_eval
        )

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared + 0.5 * log(self.n_dims)

        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]

        exp_u1 = torch.exp(u - offset)
        exp_u2 = torch.exp(-u - offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)

        return phi

class GaussianMixturePositive(FeatureMap):
    """Positive orthogonal random features that approximate the RBF kernel.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: False)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
    """
    def __init__(self, query_dimensions, n_heads, n_dims=None, 
                 softmax_temp=None, orthogonal=False, stabilize=False, 
                 redraw=1, deterministic_eval=False):
        super(GaussianMixturePositive, self).__init__(query_dimensions)

        self.n_dims = n_dims or query_dimensions
        self.orthogonal = orthogonal
        self.softmax_temp = (
            1/sqrt(query_dimensions) if softmax_temp is None
            else softmax_temp
        )
        self.stabilize = stabilize
        self.redraw = redraw
        self.deterministic_eval = deterministic_eval

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            'omega',
            torch.zeros(self.query_dims, self.n_dims)
        )

        # Buffer for storing the counter 
        self.register_buffer(
            '_calls', 
            torch.tensor(-1, dtype=torch.int)
        )

        # Parameters
        self.mean = Parameter(
            torch.Tensor(n_heads, query_dimensions)
        ) 
        self.sigma = Parameter(
            torch.Tensor(n_heads, query_dimensions)
        ) 
        self.reset_parameters()

    def reset_parameters(self):
        # Covariance matrix 
        nn.init.xavier_uniform_(self.sigma)
        
        # Mean vector 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.sigma)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.mean, -bound, bound)

    def new_feature_map(self, device, dtype):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1

        if (self._calls % self.redraw) != 0:
            return

        omega = torch.zeros(
            self.query_dims,
            self.n_dims,
            dtype=dtype,
            device=device
        )

        if self.orthogonal:
            orthogonal_random_matrix_(omega)
        else:
            omega.normal_()

        self.register_buffer('omega', omega)

    def _check_sequence_length(self, x):
        """Check that the 2nd dimension is larger than the 3rd as a heuristic
        that the sequence length will be larger than the number of heads. If
        not simply warn of a possible bug."""
        if len(x.shape) != 4:
            warnings.warn(("Favor.stabilize is set to True but the input "
                           "feature does not have the shape (N, L, H, D) "
                           "which may result in unexpected behaviour"))

        if x.shape[1] < x.shape[2]:
            warnings.warn(("Favor.stabilize is set to True but the 2nd "
                           "dimension of the input is smaller than the 3rd "
                           "which could indicate that the sequence length and "
                           "the heads are flipped. This may result in incorrect "
                           "behaviour. The shape of the input is "
                           "{!r}.").format(x.shape))

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)

        # Transform random samples 
        sigma = torch.exp(self.sigma/2)
        omega = self.omega * sigma.unsqueeze(-1)
        omega = omega + self.mean.unsqueeze(-1)

        # Project inputs 
        u = torch.einsum('nlhd,hdm->nlhm', x, omega)
        # u = x.unsqueeze(-2).matmul(omega).squeeze(-2)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared + 0.5 * log(self.n_dims)

        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]

        return torch.exp(u - offset)

class GaussianMixturePositiveHyperbolic(Favor):
    """Positive orthogonal random features that approximate the RBF kernel.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (should be divisible by 2)
                (default: query_dimensions)
        softmax_temp: float, The temerature for the softmax approximation
                     (default: 1/sqrt(query_dimensions))
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: False)
        stabilize: bool, If set to True subtract the max norm from the
                   exponentials to make sure that there are no infinities. It
                   is equivalent to a robust implementation of softmax where
                   the max is subtracted before the exponentiation.
                   (default: False)
    """
    def __init__(self, query_dimensions, n_heads, n_dims=None, 
                 softmax_temp=None, orthogonal=False, stabilize=False, 
                 redraw=1, deterministic_eval=False):
        super(GaussianMixturePositiveHyperbolic, self).__init__(
            query_dimensions, n_dims=n_dims,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal, stabilize=stabilize, 
            redraw=redraw, deterministic_eval=deterministic_eval
        )

        # Parameters
        self.mean = Parameter(torch.Tensor(n_heads, query_dimensions)) 
        self.sigma = Parameter(torch.Tensor(n_heads)) 
        self.reset_parameters()

    def reset_parameters(self):
        # Covariance matrix 
        nn.init.uniform_(self.sigma, -0.5, 0.5)
        
        # Mean vector 
        nn.init.uniform_(self.mean, -0.5, 0.5)

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)

        # Transform random samples 
        sigma = torch.exp(self.sigma/2)
        omega = self.omega * sigma[:, None, None]
        omega = omega + self.mean.unsqueeze(-1)

        # Project inputs 
        u = torch.einsum('nlhd,hdm->nlhm', x, omega)
        # u = x.unsqueeze(-2).matmul(omega).squeeze(-2)

        # Compute the offset for the exponential such that h(x) is multiplied
        # in logspace. In particular, we multiply with exp(-norm_x_squared)
        # and 1/sqrt(self.n_dims)
        offset = norm_x_squared + 0.5 * log(self.n_dims)

        if self.stabilize:
            self._check_sequence_length(norm_x_squared)
            offset = offset + norm_x_squared.max(1, keepdim=True)[0]

        exp_u1 = torch.exp(u - offset)
        exp_u2 = torch.exp(-u - offset)
        phi = torch.cat([exp_u1, exp_u2], dim=-1)

        return phi

# class GaussianMixturePRF(Favor):
#     """Positive orthogonal random features that approximate the RBF kernel.

#     Arguments
#     ---------
#         query_dimensions: int, The input query dimensions in order to sample
#                           the noise matrix
#         n_dims: int, The size of the feature map (should be divisible by 2)
#                 (default: query_dimensions)
#         softmax_temp: float, The temerature for the softmax approximation
#                      (default: 1/sqrt(query_dimensions))
#         orthogonal: bool, If set to true then the random matrix should be
#                     orthogonal which results in lower approximation variance
#                     (default: False)
#         stabilize: bool, If set to True subtract the max norm from the
#                    exponentials to make sure that there are no infinities. It
#                    is equivalent to a robust implementation of softmax where
#                    the max is subtracted before the exponentiation.
#                    (default: False)
#     """
#     def __init__(self, query_dimensions, n_heads, n_dims=None, 
#                  softmax_temp=None, orthogonal=False, stabilize=False, 
#                  redraw=1, deterministic_eval=False):
#         super(GaussianMixturePRF, self).__init__(
#             query_dimensions, n_dims=n_dims,
#             softmax_temp=softmax_temp,
#             orthogonal=orthogonal, stabilize=stabilize, 
#             redraw=redraw, deterministic_eval=deterministic_eval
#         )

#         # Parameters
#         self.mean = Parameter(
#             torch.Tensor(n_heads, query_dimensions)
#         ) 
#         self.sigma = Parameter(
#             torch.Tensor(n_heads, query_dimensions)
#         ) 
#         self.reset_parameters()

#     def reset_parameters(self):
#         # Covariance matrix 
#         nn.init.xavier_uniform_(self.sigma)
        
#         # Mean vector 
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.sigma)
#         bound = 1 / sqrt(fan_in)
#         nn.init.uniform_(self.mean, -bound, bound)

#     def forward(self, x):
#         x = x * sqrt(self.softmax_temp)
#         norm_x_squared = torch.einsum("...d,...d->...", x, x).unsqueeze(-1)

#         # Transform random samples 
#         sigma = torch.exp(self.sigma/2)
#         omega = self.omega * sigma.unsqueeze(-1)
#         omega = omega + self.mean.unsqueeze(-1)

#         # Normalize random samples so that they lie on a shphere 
#         norms = torch.norm(omega, dim=1)
#         omega = omega / norms.unsqueeze(-2)

#         # Project inputs 
#         u = torch.einsum('nlhd,hdm->nlhm', x, omega)
#         # u = x.unsqueeze(-2).matmul(omega).squeeze(-2)

#         # Compute the offset for the exponential such that h(x) is multiplied
#         # in logspace. In particular, we multiply with exp(-norm_x_squared)
#         # and 1/sqrt(self.n_dims)
#         offset = norm_x_squared + 0.5 * log(self.n_dims)

#         if self.stabilize:
#             self._check_sequence_length(norm_x_squared)
#             offset = offset + norm_x_squared.max(1, keepdim=True)[0]

#         exp_u1 = torch.exp(u - offset)
#         exp_u2 = torch.exp(-u - offset)
#         phi = torch.cat([exp_u1, exp_u2], dim=-1)

#         return phi

class GeneralizedRandomFeatures(RandomFourierFeatures):
    """Implements the generalized random Fourier features from Performers.

    It computes φ(χ) = [f(ω_1 χ), f(ω_2 χ), ..., f(ω_n χ)] where f(.) is the
    passed in `kernel_fn`.

    Arguments
    ---------
        query_dimensions: int, The input query dimensions in order to sample
                          the noise matrix
        n_dims: int, The size of the feature map (default: query_dimensions)
        softmax_temp: float, A normalizer for the dot products that is
                     multiplied to the input features before the feature map
                     application (default: 1.0)
        orthogonal: bool, If set to true then the random matrix should be
                    orthogonal which results in lower approximation variance
                    (default: True)
        kernel_fn: callable, defines the f used for the feature map.
                   (default: relu)
    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=True, kernel_fn=torch.relu, redraw=1,
                 deterministic_eval=False):
        super(GeneralizedRandomFeatures, self).__init__(
            query_dimensions,
            n_dims=2*query_dimensions if n_dims is None else 2*n_dims,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal, 
            redraw=redraw,
            deterministic_eval=deterministic_eval
        )
        self.kernel_fn = kernel_fn

    def forward(self, x):
        x = x * sqrt(self.softmax_temp)
        u = x.unsqueeze(-2).matmul(self.omega).squeeze(-2)
        return self.kernel_fn(u)

class GaussianFeatureMap(FeatureMap): 
    """
    TO DO: Add documentation 
    """
    def __init__(self, query_dimensions, n_dims, softmax_temp=None, 
        orthogonal=False, redraw=1, deterministic_eval=False): 
        super(GaussianFeatureMap, self).__init__(query_dimensions)

        self.orthogonal = orthogonal
        self.n_dims = n_dims or query_dimensions
        self.softmax_temp = softmax_temp or 1/sqrt(query_dimensions)
        self.redraw = redraw
        self.deterministic_eval = deterministic_eval

        # Make a buffer for storing the sampled omega
        self.register_buffer(
            'omega',
            torch.zeros(self.query_dims, self.n_dims//4)
        )

        # Buffer for storing the counter 
        self.register_buffer(
            '_calls', 
            torch.tensor(-1, dtype=torch.int)
        )

    def new_feature_map(self, device, dtype):
        # If we are not training skip the generation of a new feature map
        if self.deterministic_eval and not self.training:
            return

        # Only redraw the new feature map every self.redraw times
        self._calls += 1

        if (self._calls % self.redraw) != 0:
            return

        omega = torch.zeros(
            self.query_dims,
            self.n_dims//4,
            dtype=dtype,
            device=device
        )

        if self.orthogonal:
            orthogonal_random_matrix_(omega)
        else:
            omega.normal_()

        self.register_buffer('omega', omega)

class GaussianFourierFeatures(GaussianFeatureMap): 
    """

    TO DO: Add documentation

    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None, 
        orthogonal=False, redraw=1, deterministic_eval=False): 
        super(GaussianFourierFeatures, self).__init__(
            query_dimensions, n_dims=n_dims, 
            softmax_temp=softmax_temp,
            orthogonal=orthogonal, redraw=redraw, 
            deterministic_eval=deterministic_eval
        )

        # Parameters
        self.mean = Parameter(
            torch.Tensor(query_dimensions)
        ) 
        self.sigma = Parameter(
            torch.Tensor(query_dimensions, query_dimensions)
        ) 

        self.reset_parameters()

    def reset_parameters(self):
        # Covariance matrix 
        nn.init.xavier_uniform_(self.sigma)

        # Mean vector 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.sigma)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.mean, -bound, bound)

    def forward(self, x): 
        x = x * sqrt(self.softmax_temp)

        # Covariance transform 
        omega_gauss = torch.matmul(
            self.omega.transpose(0,1),
            self.sigma
        ).transpose(0,1)

        # Add mean vectors 
        omega_p = omega_gauss + self.mean.unsqueeze(-1)
        omega_m = omega_gauss - self.mean.unsqueeze(-1)

        # Project inputs 
        u_p = torch.matmul(x.unsqueeze(-2), omega_p).squeeze(-2)
        u_m = torch.matmul(x.unsqueeze(-2), omega_m).squeeze(-2)

        # Feature map 
        phi = torch.cat(
            [
                torch.cos(u_p), torch.sin(u_p), 
                torch.cos(u_m), torch.sin(u_m)
            ], 
            dim=-1
        )
        return phi * sqrt(4/self.n_dims)

class SmoothedGaussianFourierFeatures(GaussianFourierFeatures):
    """

    TO DO: Add documentation

    """
    def __init__(self, query_dimensions, n_dims=None, softmax_temp=None,
                 orthogonal=False, smoothing=1.0, redraw=1,
                 deterministic_eval=False):
        super(SmoothedGaussianFourierFeatures, self).__init__(
            query_dimensions,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp,
            orthogonal=orthogonal,
            redraw=redraw, 
            deterministic_eval=deterministic_eval
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

class GaussianMixtureFourierFeatures(GaussianFeatureMap): 
    """

    TO DO: Add documentation

    """
    def __init__(self, query_dimensions, n_heads, n_dims=None, 
        softmax_temp=None, orthogonal=False, redraw=1, 
        deterministic_eval=False): 
        super(GaussianMixtureFourierFeatures, self).__init__(
            query_dimensions, n_dims=n_dims, 
            softmax_temp=softmax_temp,
            orthogonal=orthogonal, redraw=redraw, 
            deterministic_eval=deterministic_eval
        )
        self.n_heads = n_heads

        # Parameters
        self.mean = Parameter(
            torch.Tensor(n_heads, query_dimensions)
        ) 
        self.sigma = Parameter(
            torch.Tensor(n_heads, query_dimensions, query_dimensions)
        ) 
        self.reset_parameters()

    def reset_parameters(self):
        # Covariance matrix 
        nn.init.xavier_uniform_(self.sigma)
        
        # Mean vector 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.sigma)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.mean, -bound, bound)

    def forward(self, x): 
        x = x * sqrt(self.softmax_temp)

        # Covariance transform 
        omega_gauss = torch.einsum('sm,hsd->hdm', self.omega, self.sigma) 
        # omega_gauss = torch.matmul(
        #     self.omega.transpose(0,1),
        #     self.sigma
        # ).transpose(1,2)

        # Add mean vectors 
        omega_p = omega_gauss + self.mean.unsqueeze(-1)
        omega_m = omega_gauss - self.mean.unsqueeze(-1)

        # Project inputs 
        u_p = torch.einsum('nlhd,hdm->nlhm', x, omega_p)
        u_m = torch.einsum('nlhd,hdm->nlhm', x, omega_m)

        # Feature map 
        phi = torch.cat(
            [
                torch.cos(u_p), torch.sin(u_p), 
                torch.cos(u_m), torch.sin(u_m)
            ], 
            dim=-1
        )
        return phi * sqrt(4/self.n_dims)

class SmoothedGaussianMixtureFourierFeatures(GaussianMixtureFourierFeatures):
    """

    TO DO: Add documentation

    """
    def __init__(self, query_dimensions, n_heads, n_dims=None, 
                softmax_temp=None, orthogonal=False, smoothing=1.0, 
                redraw=1, deterministic_eval=False):
        super(SmoothedGaussianMixtureFourierFeatures, self).__init__(
            query_dimensions, n_heads=n_heads,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp, orthogonal=orthogonal, 
            redraw=redraw, deterministic_eval=deterministic_eval
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

class GaussianMixtureAlaCarte(GaussianFeatureMap): 
    """

    TO DO: Add documentation

    """
    def __init__(self, query_dimensions, n_components, n_dims=None, 
                 softmax_temp=None, orthogonal=False, redraw=1, 
                 deterministic_eval=False): 
        super(GaussianMixtureAlaCarte, self).__init__(
            query_dimensions, n_dims=n_dims, 
            softmax_temp=softmax_temp, 
            orthogonal=orthogonal, redraw=redraw, 
            deterministic_eval=deterministic_eval
        )
        self.n_components = n_components

        # Parameters
        self.mean = Parameter(
            torch.Tensor(n_components, query_dimensions)
        ) 

        # Τo ensure a positve semi-definite covariance, 
        # we parametrize the covariance as log(σ^2) and
        # then take the exponential to retrieve σ^2
        self.sigma = Parameter(
            torch.Tensor(n_components, query_dimensions)
        ) 
        self.reset_parameters()

    def reset_parameters(self):
        # Covariance matrix 
        nn.init.xavier_uniform_(self.sigma)
        
        # Mean vector 
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.sigma)
        bound = 1 / sqrt(fan_in)
        nn.init.uniform_(self.mean, -bound, bound)

    def forward(self, x): 
        # Scale input 
        x = x * sqrt(self.softmax_temp)

        # Multiply by a diagnoal covariance matrix
        sigma = torch.exp(self.sigma/2)
        omega_s = self.omega * sigma.unsqueeze(-1)

        # Add/Subtract mean vectors 
        omega_p = omega_s + self.mean.unsqueeze(-1)
        omega_m = omega_s - self.mean.unsqueeze(-1)

        # Project inputs and compute feature map 
        u_p = torch.einsum('nlhd,cdm->nlchm', x, omega_p)
        u_m = torch.einsum('nlhd,cdm->nlchm', x, omega_m)
        # u_p = x.unsqueeze(-3).matmul(omega_p)
        # u_m = x.unsqueeze(-3).matmul(omega_m)

        phi = torch.cat(
            [
                torch.cos(u_p), torch.sin(u_p), 
                torch.cos(u_m), torch.sin(u_m)
            ], 
            dim=-1
        )
        return phi * sqrt(4/self.n_dims)

class SmoothedGaussianMixtureAlaCarte(GaussianMixtureAlaCarte):
    """

    TO DO: Add documentation

    """
    def __init__(self, query_dimensions, n_components, n_dims=None, 
                 softmax_temp=None, orthogonal=False, smoothing=1.0, 
                 redraw=1, deterministic_eval=False):
        super(SmoothedGaussianMixtureAlaCarte, self).__init__(
            query_dimensions, n_components=n_components,
            n_dims=query_dimensions-1 if n_dims is None else n_dims-1,
            softmax_temp=softmax_temp, orthogonal=orthogonal,
            redraw=redraw, deterministic_eval=deterministic_eval
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
