# Code for Replicating Our the Primary Experiment of Our Paper "On  Learning the Transformer Kernel"

## Directories
* fast_transformers: Contains actual implementation of the transformer models. Most of the code in this folder is adapted from https://github.com/idiap/fast-transformers
* utils: Contains code shared between all experiments, not pertaining to the underlying transformer
* text: Contains code to replicate the Text Experiment
* retrieval: Contains code to replicate the Text Experiment
* listops: Contains code to replicate the Text Experiment

## Requirements
* Python(tesed with 3.6.12)
* PyTorch(tested with 1.5.0)
* Tensorflow(tested with 2.2.0)
* TensorBoard(tested with 2.4.0)
* TensorFlow Datasets(tested with 1.2.0)

## Instructions to Replicate

1. Install Requirements listed above
2. Install fast_transformers from the main directory, preferably in editable mode with `pip install -e .`
3. Download the LRA datasets for ListOps and Retrieval, and fix the "DATA_PATH" variables in retrieval/train.py and listops/train.py to pint to the folder containing the tsv files.
4. For Fastfood methods to work, one additionally needs to install [cuda kernels for fastfood](https://github.com/HazyResearch/structured-nets/tree/master/pytorch/structure/hadamard_cuda)
5. Experiments can now be run using `<experiment_name>/train.py <attention_type>` where "experiment name is one of "text", "retrieval" or "listops" and attention type is:
|Attention Name in Paper|attention_type to pass to code|
|-----------------------|------------------------------|
|Softmax Transformer|softmax|
|GMM-PRF|mix-gauss-positive|
|GMM-RKS|mix-gauss-fourier|
|FastFood-RKS|fsgb-fastfood|
|FastFood-PRF|fsgb-positive-fastfood|
|Generator-RKS|generative-fourier|
|Generator-PRF|generative-positive|

For text experiments, the maximum length can additionally be passed as the second parameter to replicate figure 4 from our table.