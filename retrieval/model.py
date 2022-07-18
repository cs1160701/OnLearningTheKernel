import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from fast_transformers.masking import LengthMask
from fast_transformers.attention import AttentionLayer
from fast_transformers.builders import AttentionBuilder
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer

from fast_transformers.feature_maps import Favor, \
    ActivationFunctionFeatureMap, GaussianPRF, \
    GeneralizedRandomFeatures, GaussianMixturePositiveHyperbolic, \
    RandomFourierFeatures, SmoothedRandomFourierFeatures, \
    FastFoodRandomFeatures, SmoothedFastFoodRandomFeatures, \
    GaussianFourierFeatures, SmoothedGaussianFourierFeatures, \
    GaussianMixtureAlaCarte, SmoothedGaussianMixtureAlaCarte, \
    GaussianMixtureFourierFeatures, SmoothedGaussianMixtureFourierFeatures, \
    GenerativeRandomFourierFeatures, SmoothedGenerativeRandomFourierFeatures, \
    GenerativePositiveRandomFeatures, GeneralizedFeatureMap, \
    GaussianMixturePositive, FastFoodPositiveFeatures
    
from utils.model_utils import DualClassificationHead

def pow2cast(x):
    op = math.floor if bin(int(x))[3] != "1" else math.ceil
    return 2**(op(math.log(x,2)))

class DocumentMatchingClassifier(nn.Module): 
    """

    Parameters: 
    -----------
    """
    def __init__(self, n_classes, classifier_dim, vocab_size, max_len, 
                 d_model, attention_type, n_layers, n_heads, d_ff=None, 
                 d_query=None, d_values=None, n_mix_components=None, 
                 dropout=0.1, attention_dropout=0.1, activation='relu', 
                 classifier_activation='relu', output_norm=False, 
                 input_embeddings=None):
        super(DocumentMatchingClassifier, self).__init__()

        self.n_classes = n_classes
        self.classifier_dim = classifier_dim
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model 
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff or 4*d_model
        self.d_query = d_query or (d_model//n_heads)
        self.d_values = d_values or (d_model//n_heads)
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation = F.relu if activation == "relu" else F.gelu
        self.output_norm = nn.LayerNorm(d_model) if output_norm else None  

        # Input embeddings 
        if input_embeddings is not None: 
            input_embeddings = torch.from_numpy(input_embeddings)
            self.input_embeddings = nn.Embedding.from_pretrained(input_embeddings)
        else: 
            self.input_embeddings = nn.Embedding(
                vocab_size, 
                d_model, 
                padding_idx=0
            )
            nn.init.normal_(self.input_embeddings.weight)

        # CLS token 
        self.cls_embedding = nn.parameter.Parameter(torch.Tensor(1, 1, d_model))
        nn.init.zeros_(self.cls_embedding)
        self.max_len += 1 # Account for [CLS] token

        # Positional embeddings 
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_embedding', pe.unsqueeze(0))

        # Embedding dropout 
        self.embedding_dropout = nn.Dropout(dropout)

        # Attention module  
        if attention_type == 'softmax': 
            attn_type = 'full'
            attn_builder = AttentionBuilder.from_kwargs(
                attention_dropout=self.attention_dropout
            )

        elif attention_type == 'performer': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=Favor.factory(
                    n_dims=2*self.d_query, 
                    orthogonal=True,
                    stabilize=False, 
                    redraw=1000
                )
            )

        elif attention_type == 'rbf-positive': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=GaussianPRF.factory(
                    n_dims=2*self.d_query, 
                    orthogonal=True,
                    stabilize=False, 
                    redraw=1000
                )
            )

        elif attention_type == 'mix-gauss-positive': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=GaussianMixturePositive.factory(
                    n_heads=self.n_heads,                                                          
                    n_dims=min(
                        self.max_len,
                        pow2cast(np.log(self.d_query)*self.d_query)
                    ), 
                    orthogonal=False,
                    stabilize=False 
                )
            )

        elif attention_type == 'mix-gauss-positive-hyperbolic': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=GaussianMixturePositiveHyperbolic.factory(
                    n_heads=self.n_heads,                                                          
                    n_dims=min(
                        self.max_len,
                        2*pow2cast(np.log(self.d_query)*self.d_query)
                    ), 
                    orthogonal=False,
                    stabilize=False 
                )
            )

        elif attention_type == 'linear': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query
            )

        elif attention_type == 'generative-fourier': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query, 
                feature_map=SmoothedGenerativeRandomFourierFeatures.factory(
                    noise_dims=[
                        self.d_query,
                        4*self.d_query
                    ], 
                    n_dims=min(
                        self.max_len+1,
                        2*pow2cast(np.log(self.d_query)*self.d_query)+1
                    ) 
                )
            )

        elif attention_type == 'generative-positive': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query, 
                feature_map=GenerativePositiveRandomFeatures.factory(
                    noise_dims=[
                        self.d_query,
                        4*self.d_query
                    ], 
                    n_dims=min(
                        self.max_len,
                        pow2cast(np.log(self.d_query)*self.d_query)
                    ), 
                )
            )
            
        elif attention_type == 'rbf-fourier': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedRandomFourierFeatures.factory(
                    n_dims=min(
                        self.max_len+1,
                        2*pow2cast(np.log(self.d_query)*self.d_query)+1
                    ),
                    orthogonal=False 
                )
            )

        elif attention_type == 'rbf-fastfood': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedFastFoodRandomFeatures.factory()
            )

        elif attention_type == 'fs-fastfood': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedFastFoodRandomFeatures.factory(
                    learn_S=True
                )
            )

        elif attention_type == 'fsgb-fastfood': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedFastFoodRandomFeatures.factory(
                    learn_S=True, 
                    learn_G_B=True
                )
            )

        elif attention_type == 'fsgb-positive-fastfood': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=FastFoodPositiveFeatures.factory(
                    learn_S=True, 
                    learn_G_B=True
                )
            )

        elif attention_type == 'gauss-fourier': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query, 
                feature_map=SmoothedGaussianFourierFeatures.factory(
                    n_dims=2*self.d_query+1, 
                    orthogonal=False
                )
            )

        elif attention_type == 'mix-gauss-fourier': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedGaussianMixtureFourierFeatures.factory(
                    n_heads=self.n_heads,
                    n_dims=min(
                        self.max_len+1,
                        2*pow2cast(np.log(self.d_query)*self.d_query)+1
                    ),
                    smoothing=1.0, 
                    orthogonal=False
                )
            )

        elif attention_type == 'mix-gauss-alacarte': 
            attn_type = 'mixture_linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                n_components=n_mix_components or self.d_query//2,
                feature_map=SmoothedGaussianMixtureAlaCarte.factory(
                    n_components=n_mix_components or self.d_query//2,
                    n_dims=min(
                        self.max_len+1,
                        2*pow2cast(np.log(self.d_query)*self.d_query)+1
                    ), 
                    orthogonal=False
                )
            )

        else: 
            raise ValueError('%s is not a valid attention type.' %attention_type) 

        # Specify encoder model 
        self.transformer = TransformerEncoder(
            [
                TransformerEncoderLayer(
                    AttentionLayer(
                        attn_builder.get(attn_type), 
                        d_model=self.d_model, 
                        n_heads=self.n_heads, 
                        d_keys=self.d_query, 
                        d_values=self.d_values, 
                        bias=False, 
                        weight_init='xavier_uniform', 
                    ),
                    self.d_model, 
                    self.d_ff, 
                    self.dropout, 
                    self.activation, 
                    bias=True,
                    weight_init='xavier_uniform', 
                    bias_init='normal', 
                    axial_ordering=True
                )
                for _ in range(self.n_layers)
            ], 
            norm_layer=self.output_norm
        )

        # Clasification head 
        self.dual_classifier = DualClassificationHead(
            self.d_model, 
            self.classifier_dim,
            self.n_classes, 
            bias_init='zeros', 
            activation=classifier_activation
        )

    def forward(self, x1, x2, length_mask_x1, length_mask_x2): 
        """
        """
        ############################# First Document #############################
        x1 = self.input_embeddings(x1) 

        # Prepend [CLS] token 
        c = self.cls_embedding.repeat(
            [x1.shape[0], 1, 1]
        )
        x1 = torch.cat(
            (c, x1), 
            axis=1
        )

        # Account for [CLS] token
        length_mask_x1 += 1

        # Positional embeddings
        x1 = x1 + self.pos_embedding.narrow(1, 0, self.max_len) 

        # Embedding dropout 
        x1 = self.embedding_dropout(x1)

        # Check mask data type 
        if not isinstance(length_mask_x1, torch.Tensor) or length_mask_x1.dtype != torch.int64: 
            raise ValueError("ListOpsClassifier expects the \
                length_mask to be a a PyTorch long tensor.")

        # Transformer 
        z1 = self.transformer(
            x1, 
            length_mask=LengthMask(
                length_mask_x1, 
                max_len=self.max_len
            )
        )

        ############################# Second Document #############################
        x2 = self.input_embeddings(x2) 

        # Prepend [CLS] token 
        c = self.cls_embedding.repeat(
            [x2.shape[0], 1, 1]
        )
        x2 = torch.cat(
            (c, x2), 
            axis=1
        )

        # Account for [CLS] token
        length_mask_x2 += 1

        # Positional embeddings
        x2 = x2 + self.pos_embedding.narrow(1, 0, self.max_len)

        # Embedding dropout 
        x2 = self.embedding_dropout(x2)

        # Check mask data type 
        if not isinstance(length_mask_x2, torch.Tensor) or length_mask_x2.dtype != torch.int64: 
            raise ValueError("ListOpsClassifier expects the \
                length_mask to be a a PyTorch long tensor.")

        # Transformer 
        z2 = self.transformer(
            x2, 
            length_mask=LengthMask(
                length_mask_x2, 
                max_len=self.max_len
            )
        )

        # Use only [CLS] embedding for classification 
        y = self.dual_classifier(z1[:,0,:], z2[:,0,:])

        return y