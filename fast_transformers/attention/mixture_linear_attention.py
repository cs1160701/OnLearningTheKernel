#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement unmasked linear attention."""
import torch
from torch.nn import init, Module
from torch.nn.parameter import Parameter

from ..attention_registry import AttentionRegistry, Optional, Callable, Int, \
    EventDispatcherInstance
from ..events import EventDispatcher
from ..feature_maps import elu_feature_map


class MixtureLinearAttention(Module):
    """Implement unmasked attention using dot product of feature maps in
    O(N D^2) complexity.

    Given the queries, keys and values as Q, K, V instead of computing

        V' = softmax(Q.mm(K.t()), dim=-1).mm(V),

    we make use of a feature map function Φ(.) and perform the following
    computation

        V' = normalize(Φ(Q).mm(Φ(K).t())).mm(V).

    The above can be computed in O(N D^2) complexity where D is the
    dimensionality of Q, K and V and N is the sequence length. Depending on the
    feature map, however, the complexity of the attention might be limited.

    Arguments
    ---------
        n_components: int, the number of components in the mixture 
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, query_dimensions, n_components, feature_map,
                 eps=1e-6, event_dispatcher=""):
        super(MixtureLinearAttention, self).__init__()
        self.feature_map = feature_map(query_dimensions) 

        # Mixture weights 
        self.weights = Parameter(
            torch.Tensor(n_components)
        )
        init.uniform_(self.weights, -0.25, 0.25)

        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        # Apply the feature map to the queries and keys
        self.feature_map.new_feature_map(queries.device, queries.dtype)
        Q = self.feature_map.forward_queries(queries)
        K = self.feature_map.forward_keys(keys)

        # Apply the key padding mask and make sure that the attn_mask is
        # all_ones
        if not attn_mask.all_ones:
            raise RuntimeError(("MixtureLinearAttention does not support "
                                "arbitrary attention masks"))
        K = K * key_lengths.float_matrix[:, :, None, None, None]

        # Compute the KV matrix, namely the dot product of keys and values so
        # that we never explicitly compute the attention matrix and thus
        # decrease the complexity
        KV = torch.einsum("nsqhd,nshm->nqhmd", K, values)

        # Normalize mixture weights 
        W = torch.softmax(self.weights, dim=-1)

        # Compute the normalizer
        Z = 1/(torch.einsum("nlqhd,nqhd->nlqh", Q, K.sum(dim=1))+self.eps)

        # Compute and the new values
        V = torch.einsum("nlqhd,nqhmd,nlqh->nlqhm", Q, KV, Z)

        # Compute and return the weighted mixture
        M = torch.einsum("q,nlqhm->nlhm", W, V)

        # # Compute the normalizer
        # Z = 1/(torch.einsum("q,nlqhd,nqhd->nlh", W, Q, K.sum(dim=1))+self.eps)

        # # Finally compute and return the new values
        # V = torch.einsum("q,nlqhd,nqhmd,nlh->nlhm", W, Q, KV, Z)

        return M.contiguous()

# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "mixture_linear", MixtureLinearAttention,
    [
        ("query_dimensions", Int),
        ("n_components", Int),
        ("feature_map", Optional(Callable)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
