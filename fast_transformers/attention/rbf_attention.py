#
# Based on the code released by Idiap Research Institute
#

"""Implement unmasked linear attention."""
from math import sqrt

import torch
from torch.nn import Module

from ..attention_registry import AttentionRegistry, Optional, Float, \
    EventDispatcherInstance
from ..events import EventDispatcher

class RBFKernelAttention(Module):
    """
    Implement attention based on the RBF kernel between keys and queries. 

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we perform the following computation 

        V' = normalize(k(Q,K)).mm(V)

    where k(Q,K) is a (NxN) matrix whose [i,j] entry is given by 

        k(Q,K)[i,j] = k(Q[i],K[j]) = exp(- 0.5 * γ * ||Q[i] - K[j]||^2)

    Arguments
    ---------
        softmax_temp: The temerature γ for the RBF kernel
                      approximation exp(-0.5 * γ * ||x-y||^2)
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, softmax_temp=None, eps=1e-6,
                 event_dispatcher=""):
        super(RBFKernelAttention, self).__init__()
        self.eps = eps
        self.softmax_temp = softmax_temp
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):

        # Extract some shapes and compute the temperature
        N, L, H, E = queries.shape
        _, S, _, D = values.shape
        softmax_temp = self.softmax_temp or 1./sqrt(E)

        # Compute pairwise distances between keys and queries
        norm_Q_minus_K = torch.cdist(queries.view(N, H, L, E), 
                                     keys.view(N, H, S, E), p=2)

        # Compute the kernel matrix between queries and keys
        phi_Q_times_phi_K = torch.exp(-softmax_temp*(torch.square(norm_Q_minus_K)/2)) 

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("RBFKernelAttention does not support arbitrary "
                                "attention masks"))
        phi_Q_times_phi_K = phi_Q_times_phi_K * key_lengths.float_matrix[:, None, None]

        # Compute the normalizer
        Z = 1/(torch.sum(phi_Q_times_phi_K, dim=-1) + self.eps)

        # Compute and return the new values 
        V = torch.einsum("nhls,nshd,nhl->nlhd", phi_Q_times_phi_K, values, Z)

        return V.contiguous()

# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "rbf", RBFKernelAttention,
    [
        ("softmax_temp", Optional(Float)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
