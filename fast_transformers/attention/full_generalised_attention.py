"""Implement the full attention with generalised softmax as defined in 
https://arxiv.org/pdf/1910.12554.pdf."""

from math import sqrt

import torch
from torch.nn import Dropout, Module

from ..attention_registry import AttentionRegistry, Optional, Float, \
    EventDispatcherInstance, Callable
from ..events import EventDispatcher, AttentionEvent

def NormSquared(V1,V2):
    V1squared=torch.einsum("nlhe,nlhe->nhl",V1,V1).unsqueeze(-1)
    V2squared=torch.einsum("nshe,nshe->nhs",V2,V2).unsqueeze(-2)
    V1TV2=torch.einsum("nlhe,nshe->nhls",V1,V2)
    return V1squared+V2squared-2*V1TV2
"""Implement various kernels. 
Arguments
---------
    V1,V2: vectors under consideration (N*L*H*E)
    args : kernel specific arguments

Returns  : Similarity between ever two positions (N*H*L*L) 
"""
def LinearKernel(V1, V2, args=""):
    return torch.einsum("nlhe,nshe->nhls",V1,V2)

def lOgarithmicKernel(V1, V2, args="2"): 
    p=float(args)
    return -torch.log(1+torch.pow(NormSquared(V1,V2),p/2))

def PowerKernel(V1, V2,args="2"):
    p=float(args)
    return -torch.pow(NormSquared(V1,V2),p/2)

def polYnomialKernel(V1, V2, args="2,1,1"):
    args=args.split(",")
    p=float(args[0])
    alpha=float(args[1])
    c=float(args[2])
    return torch.pow((alpha*torch.einsum("nlhe,nshe->nhls",V1,V2)+c),p)

def RbfKernel(V1, V2, args="0.5"):
    gamma=float(args)
    return torch.exp(-gamma*NormSquared(V1,V2))

def HyperbolicKernel(V1, V2, args=""):
    V1squared=torch.einsum("nlhe,nlhe->nhl",V1,V1).unsqueeze(-1)
    V2squared=torch.einsum("nshe,nshe->nhs",V2,V2).unsqueeze(-2)
    V1TV2=torch.einsum("nlhe,nshe->nhls",V1,V2)
    hyp=(V1squared+V2squared-2*V1TV2)/(1-V1squared-V2squared+V1squared*V2squared)
    return -torch.acos(1+hyp)

def WaveletKernel(V1,V2, args="1,1"):
    args=args.split(",")
    a=float(args[0])
    b=float(args[1])
    ns=NormSquared(V1,V2)
    return torch.cos(ns/a)*torch.exp(-ns/b)

map={
    'L':LinearKernel,
    'O':lOgarithmicKernel,
    'P':PowerKernel,
    'Y':polYnomialKernel,
    'R':RbfKernel,
    'H':HyperbolicKernel,
    'W':WaveletKernel
}
def Kernel(V1,V2,args="L"):
    return map[args[0]](V1,V2,args[1:])

class GeneralisedSoftmax(Module):
    """Implement a Generalised Softmax

    Arguments:
    ----------
    num_heads: Number of heads in the attention
    d_query  : Dimension of Query vector
    sm_temp  :temperature for each softmax
    type     : Sequence of dot product types to use for the generalised
               softmax. Individual types are separated by commas 

    """
    def __init__(self,num_heads,d_query,sm_temp=1,style="L;O2;P2;Y2,1,1;R0.5;H;W1,1"):
        super(GeneralisedSoftmax,self).__init__()
        self.kernel_args=style.split(";")
        self.kernel_count=len(self.kernel_args)
        self.M=torch.nn.Parameter(torch.rand(num_heads,d_query,self.kernel_count))
        self.C=torch.nn.Parameter(torch.rand(self.kernel_count,num_heads,d_query,d_query))
        self.softmax_temp=sm_temp
        self.pi=None
    def forward(self,query,key,attn_mask,key_lengths):
        kernel_outs=[]
        for c,k in enumerate(self.kernel_args):
            htilde=torch.tanh(torch.matmul(query.unsqueeze(-2),self.C[c]).squeeze(-2))
            QK=Kernel(htilde,key,k)
            if not attn_mask.all_ones:
                QK = QK + attn_mask.additive_matrix.unsqueeze(1)
            QK = QK + key_lengths.additive_matrix[:, None, None]
            kernel_outs.append(torch.softmax(self.softmax_temp * QK, dim=-1))
        pi=torch.softmax(torch.matmul(query.unsqueeze(-2),self.M).squeeze(-2),dim=-1)
        self.pi=torch.sum(
            pi * key_lengths.float_matrix[:, :, None, None],
            dim=(0,1,2)
        )
        kernel_out=torch.stack(kernel_outs,dim=-1)
        return torch.einsum("nhlsk,nlhk->nhls",kernel_out,pi)

        
class FullGeneralisedAttention(Module):
    """Implement the scaled dot product attention with softmax.

    Arguments
    ---------
        sm: Generalised Softmax Module
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.1)
        event_dispatcher: str or EventDispatcher instance to be used by this
                          module for dispatching events (default: the default
                          global dispatcher)
    """
    def __init__(self, sm ,attention_dropout=0.1,
                 event_dispatcher=""):
        super(FullGeneralisedAttention, self).__init__()
        self.softmaxModule=sm()
        self.dropout = Dropout(attention_dropout)
        self.event_dispatcher = EventDispatcher.get(event_dispatcher)

    def forward(self, queries, keys, values, attn_mask, query_lengths,
                key_lengths):
        """Implements the multihead softmax attention.

        Arguments
        ---------
            queries: (N, L, H, E) The tensor containing the queries
            keys: (N, S, H, E) The tensor containing the keys
            values: (N, S, H, D) The tensor containing the values
            attn_mask: An implementation of BaseMask that encodes where each
                       query can attend to
            query_lengths: An implementation of  BaseMask that encodes how
                           many queries each sequence in the batch consists of
            key_lengths: An implementation of BaseMask that encodes how
                         many queries each sequence in the batch consists of
        """
        # Compute the attention and the weighted average
        A = self.dropout(self.softmaxModule(queries,keys,attn_mask,key_lengths))
        V = torch.einsum("nhls,nshd->nlhd", A, values)

        # Let the world know of the attention matrix
        self.event_dispatcher.dispatch(AttentionEvent(self, A))

        # Make sure that what we return is contiguous
        return V.contiguous()


# Register the attention implementation so that it becomes available in our
# builders
AttentionRegistry.register(
    "FGA", FullGeneralisedAttention,
    [
        ("sm", Callable),
        ("attention_dropout", Optional(Float, 0.1)),
        ("event_dispatcher", Optional(EventDispatcherInstance, ""))
    ]
)
