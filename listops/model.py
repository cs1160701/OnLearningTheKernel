import math
import torch
import numpy as np 
import torch.nn as nn
import torch.nn.functional as F

from fast_transformers.masking import LengthMask
from fast_transformers.attention import AttentionLayer
from fast_transformers.builders import AttentionBuilder
from fast_transformers.transformers import TransformerEncoder, TransformerEncoderLayer
from fast_transformers.feature_maps import ActivationFunctionFeatureMap, \
    Favor, GaussianPRF, GaussianMixturePositiveHyperbolic, GeneralizedRandomFeatures, \
    SmoothedRandomFourierFeatures, SmoothedFastFoodRandomFeatures, \
    SmoothedGaussianFourierFeatures, SmoothedGenerativeRandomFourierFeatures, \
    SmoothedGaussianMixtureFourierFeatures, SmoothedGaussianMixtureAlaCarte, \
    GaussianMixturePositive, FastFoodPositiveFeatures, \
    GenerativePositiveRandomFeatures


from utils.model_utils import ClassificationHead

def pow2cast(x):
    op = math.floor if bin(int(x))[3] != "1" else math.ceil
    return 2**(op(math.log(x,2)))

class ListOpsClassifier(nn.Module): 
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
        super(ListOpsClassifier, self).__init__()

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

        # Token embeddings 
        if input_embeddings is not None: 
            input_embeddings = torch.from_numpy(input_embeddings)
            self.input_embeddings = nn.Embedding.from_pretrained(input_embeddings)
        else: 
            self.input_embeddings = nn.Embedding(
                self.vocab_size, 
                self.d_model, 
                padding_idx=0
            )
            nn.init.normal_(self.input_embeddings.weight)

        # CLS token 
        self.cls_embedding = nn.parameter.Parameter(torch.Tensor(1, 1, self.d_model))
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
        elif attention_type == 'local': 
            attn_type = 'window'
            attn_builder = AttentionBuilder.from_kwargs(
                attention_dropout=self.attention_dropout,
                window_size=29
            )
        elif attention_type == 'performer': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=Favor.factory(
                    n_dims=2*self.d_query, 
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
                    n_dims=2*self.d_query+1 
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
                        4*pow2cast(np.log(self.d_query)*self.d_query)+1
                    )
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
                        4*pow2cast(np.log(self.d_query)*self.d_query)+1
                    )
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
        self.classifier = ClassificationHead(
            self.d_model, 
            self.classifier_dim, 
            self.n_classes, 
            bias_init='zeros', 
            activation=classifier_activation
        )

    def forward(self, x, length_mask): 
        """
        """
        # Token embeddings 
        x = self.input_embeddings(x) 

        # Prepend [CLS] token 
        c = self.cls_embedding.repeat(
            [x.shape[0], 1, 1]
        )
        x = torch.cat(
            (c, x), 
            axis=1
        )

        # Account for [CLS] token
        length_mask += 1

        # Positional embeddings
        x = x + self.pos_embedding.narrow(1, 0, self.max_len)

        # Embedding dropout 
        x = self.embedding_dropout(x)

        # Check mask data type 
        if not isinstance(length_mask, torch.Tensor) or length_mask.dtype != torch.int64: 
            raise ValueError("ListOpsClassifier expects the \
                length_mask to be a a PyTorch long tensor.")

        # Transformer 
        z = self.transformer(
            x, 
            length_mask=LengthMask(
                length_mask, 
                max_len=self.max_len
            )
        )

        # Use only [CLS] embedding for classification 
        y = self.classifier(z[:,0,:])

        return y

class TiedListOpsClassifier(nn.Module): 
    """

    Parameters: 
    -----------
    """
    def __init__(self, n_classes, classifier_dim, vocab_size, max_len, d_model, 
                attention_type, n_layers, n_heads, d_ff=None, d_query=None, 
                d_values=None, dropout=0.1, attention_dropout=0.1, activation='relu', 
                classifier_activation='relu', output_norm=False, input_embeddings=None): 
        super(TiedListOpsClassifier, self).__init__()

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

        # Token embeddings 
        if input_embeddings is not None: 
            input_embeddings = torch.from_numpy(input_embeddings)
            self.input_embeddings = nn.Embedding.from_pretrained(input_embeddings)
        else: 
            self.input_embeddings = nn.Embedding(
                vocab_size, 
                d_model
            )
            nn.init.normal_(self.input_embeddings.weight)

        # Positional embeddings 
        self.pos_embedding = PositionalEncoding(
            self.d_model, 
            self.max_len
        )

        # Embedding dropout 
        self.embedding_dropout = nn.Dropout(dropout)

        # Attention module  
        if attention_type == 'softmax': 
            attn_type = 'full'
            attn_builder = AttentionBuilder.from_kwargs(
                attention_dropout=self.attention_dropout
            )

        elif attention_type == 'softmax-kernel': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=Favor.factory(
                    n_dims=2*self.d_query, 
                    stabilize=True
                )
            )

        elif attention_type == 'linear': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query
            )

        elif attention_type == 'rbf-fourier': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedRandomFourierFeatures.factory(
                    n_dims=2*self.d_query+1, 
                    orthogonal=False
                )
            )

        elif attention_type == 'rbf-fastfood': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedFastFoodRandomFeatures.factory()
            )

        elif attention_type == 'gauss-fourier': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query, 
                feature_map=SmoothedGaussFourierFeatures.factory(
                    n_dims=2*self.d_query+1, 
                    orthogonal=False
                )
            )

        elif attention_type == 'mix-gauss-fourier': 
            attn_type = 'linear'
            attn_builder = AttentionBuilder.from_kwargs(
                query_dimensions=self.d_query,
                feature_map=SmoothedMixGaussFourierFeatures.factory(
                    n_heads=self.n_heads,
                    n_dims=4*self.d_query+1, 
                    orthogonal=False
                )
            )

        else: 
            raise ValueError('%s is not a valid attention type.' %attention_type) 

        # Encoder layer 
        encoder_layer = TransformerEncoderLayer(
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
            bias_init='normal'
        ) 

        # Transformer (weight-tied)
        self.transformer = TransformerEncoder(
            [ 
                encoder_layer for _ in range(self.n_layers) 
            ], 
            norm_layer=self.output_norm
        )

        # Clasification head 
        self.classifier = ClassificationHead(
            self.d_model, 
            self.classifier_dim, 
            self.n_classes, 
            bias_init='zeros', 
            activation=classifier_activation
        )

    def forward(self, x_tok, length_mask): 
        """
        """
        # Token embeddings 
        x = self.input_embeddings(x_tok) 

        # Positional embedding s
        x = self.pos_embedding(x)

        # Embedding dropout 
        x = self.embedding_dropout(x)

        # Check mask data type 
        if not isinstance(length_mask, torch.Tensor) or length_mask.dtype != torch.int64: 
            raise ValueError("ListOpsClassifier expects the \
                length_mask to be a a PyTorch long tensor.")

        # Transformer 
        z = self.transformer(
            x, 
            length_mask=LengthMask(
                length_mask, 
                max_len=self.max_len
            )
        )

        # Use only [CLS] embedding for classification 
        y = self.classifier(z[:,0,:])

        return y