#
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Angelos Katharopoulos <angelos.katharopoulos@idiap.ch>
#

"""Implementations of feature maps to be used with linear attention and causal
linear attention."""

from .base import elu_feature_map, GeneralizedFeatureMap, \
    ActivationFunctionFeatureMap
from .fastfood import FastFoodRandomFeatures, \
    SmoothedFastFoodRandomFeatures, FastFoodPositiveFeatures
from .fourier_features import RandomFourierFeatures, Favor, \
    SmoothedRandomFourierFeatures, GeneralizedRandomFeatures, \
    GaussianFourierFeatures, SmoothedGaussianFourierFeatures, \
    GaussianMixtureFourierFeatures, SmoothedGaussianMixtureFourierFeatures, \
    GaussianPRF, GaussianMixturePositiveHyperbolic, GenerativePositiveRandomFeatures, \
    GenerativeRandomFourierFeatures, SmoothedGenerativeRandomFourierFeatures, \
    GaussianMixtureAlaCarte, SmoothedGaussianMixtureAlaCarte, \
    GaussianMixturePositive