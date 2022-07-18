#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>,
# Apoorv Vyas <avyas@idiap.ch>
#

"""Implementations of different types of attention mechanisms."""

from .rbf_attention import RBFKernelAttention
from .attention_layer import AttentionLayer
from .full_attention import FullAttention
from .linear_attention import LinearAttention
from .causal_linear_attention import CausalLinearAttention
from .clustered_attention import ClusteredAttention
from .improved_clustered_attention import ImprovedClusteredAttention
from .reformer_attention import ReformerAttention
from .conditional_full_attention import ConditionalFullAttention
from .exact_topk_attention import ExactTopKAttention
from .improved_clustered_causal_attention import ImprovedClusteredCausalAttention
from .local_attention import LocalAttention
from .mixture_linear_attention import MixtureLinearAttention
from .generalized_linear_attention import GeneralizedLinearAttention
from .full_generalised_attention import FullGeneralisedAttention, GeneralisedSoftmax


