#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implement unmasked linear attention."""
import torch
from torch.nn import init, Linear, Module, ModuleList
from torch.nn.parameter import Parameter

from ..attention_registry import AttentionRegistry, Optional, Callable, Int, \
    List, EventDispatcherInstance
from ..events import EventDispatcher
from ..feature_maps import elu_feature_map


class GeneralizedLinearAttention(Module):
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
        feature_map: callable, a callable that applies the feature map to the
                     last dimension of a tensor (default: elu(x)+1)
        eps: float, a small number to ensure the numerical stability of the
             denominator (default: 1e-6)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, n_heads, query_dimensions, feature_maps, 
                 eps=1e-6, event_dispatcher=""):
        super(GeneralizedLinearAttention, self).__init__()
        # Feature maps 
        f_maps = []
        for feature_map in feature_maps: 
            f_maps.append(
                feature_map(query_dimensions)
            )

        self.n_kernels = len(f_maps)
        self.feature_maps = ModuleList(f_maps)

        # Mixture weight projection 
        self.pi_projection = Parameter(
            torch.Tensor(
                n_heads, 
                query_dimensions, 
                self.n_kernels
            )
        )
        init.xavier_uniform_(self.pi_projection)

        self.eps = eps
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        Vs = []
        for feature_map in self.feature_maps: 
            # Redraw random samples 
            feature_map.new_feature_map(queries.device, queries.dtype)

            # Compute feature representations
            K = feature_map.forward_keys(keys)
            Q = feature_map.forward_queries(queries)

            # Apply the key padding mask and make sure that the attn_mask is
            # all_ones
            if not attn_mask.all_ones:
                raise RuntimeError(("GeneralizedLinearAttention does not support "
                                    "arbitrary attention masks"))
            K = K * key_lengths.float_matrix[:, :, None, None]

            # Compute the KV matrix, namely the dot product of keys and values so
            # that we never explicitly compute the attention matrix and thus
            # decrease the complexity
            KV = torch.einsum("nshd,nshm->nhmd", K, values)

            # Compute the normalizer
            Z = 1/(torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1))+self.eps)

            # Finally compute the new values
            Vs.append(torch.einsum("nlhd,nhmd,nlh->nlhm", Q, KV, Z))

        # Stack all kernel outputs
        Vs = torch.stack(Vs, dim=-1)

        # Compute the mixture
        pi = torch.softmax(
            queries.unsqueeze(-2).matmul(self.pi_projection).squeeze(-2),
            dim=-1
        )
        
        V = torch.einsum('nlhmk,nlhk->nlhm', Vs, pi)
        
        return V.contiguous()

# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "generalized_linear", GeneralizedLinearAttention,
    [   
        ("n_heads", Int),
        ("query_dimensions", Int),
        ("feature_maps", List),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
