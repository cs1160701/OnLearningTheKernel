import os
import torch
import collections
import numpy as np 
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from torch.utils.data import Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class ListOpsDataset(Dataset): 
    def __init__(self, file_path, training_set=False, vocab=None, sequence_len=2000):

        self.n_classes = 10
        self.sequence_len = sequence_len
        self.training_set = training_set

        # Tokenizer 
        self.tokenizer = tfds.features.text.Tokenizer(
            reserved_tokens=['[MIN', '[MAX', '[MED', '[SM', ']']
        )

        # Load dataset 
        dataset = self.load_data(file_path)

        # Build vocabulary 
        if self.training_set: 
            self.vocab = self.build_vocabulary(dataset)
        elif vocab is not None: 
            self.vocab = vocab
        else: 
            ValueError('Either a vocabulary must be provided \
                or the training flag should be set to True.') 

        # Encoder 
        self.encoder = tfds.features.text.TokenTextEncoder(
            self.vocab, 
            tokenizer=self.tokenizer
        )

        # Vocab size 
        self.vocab_size = self.encoder.vocab_size

        # Encode data
        dataset = dataset.map(
            self.tokenize, 
            num_parallel_calls=AUTOTUNE
        )

        # Pad sequences
        max_shape = {
            'Source': [self.sequence_len], 
            'Target': []
        }

        dataset = dataset.padded_batch(
            batch_size=1, 
            padded_shapes=max_shape
        )

        # Convert tf dataset to arrays 
        source = []
        labels = []

        iterator = iter(dataset)

        for sample in iterator: 
            source.append(sample['Source'].numpy()[0])
            labels.append(sample['Target'].numpy()[0])

        self.source = np.array(source, dtype=int)
        self.labels = np.array(labels, dtype=int)

        self.length_mask = np.sum(
            self.source > 0, 
            axis=1
        )
        
    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        sequence = self.source[idx, :] 
        length_mask = self.length_mask[idx]

        sample = {
            'label': torch.tensor(label),
            'tokens': torch.tensor(sequence),  
            'length': torch.tensor(length_mask, dtype=torch.int64)
        }

        return sample

    def load_data(self, file_path, batch_size=1): 

        col_names = ['Source', 'Target']
        col_types = [tf.string, tf.int32]
        
        ds = tf.data.experimental.make_csv_dataset(
            [file_path],
            batch_size,
            column_defaults=col_types,
            select_columns=col_names,
            field_delim='\t',
            header=True,
            num_epochs=1, 
            shuffle=False
        )
        ds = ds.unbatch()

        return ds

    def encode(self, x): 
        result = tf.py_function(
            lambda s: tf.constant(self.encoder.encode(s.numpy())), 
            [x, ], 
            tf.int32
        )
        result.set_shape([None])
        return result

    def tokenize(self, dataset): 
        return {
            'Source': self.encode(dataset['Source'])[:self.sequence_len], 
            'Target': dataset['Target']
        }

    def build_vocabulary(self, dataset, cutoff=1000):
        vocab = collections.Counter()

        for i, sample in enumerate(dataset): 
            source = sample['Source']
            tokens = self.tokenizer.tokenize(source.numpy())
            tokens = np.reshape(tokens, (-1)).tolist()
            vocab.update(tokens)

            if i > cutoff: 
                break

        vocab = list(vocab.keys())

        return vocab

# class ListOpsDataset(Dataset): 
#     """
    
#     TO DO: Add documentation

#     """
#     def __init__(self, data_folder, data_file, training_set=False, 
#         vocab=None, sequence_len=2000, embedding_model=None): 

#         self.sequence_len = sequence_len
#         self.training_set = training_set

#         # Load data 
#         source, labels = self.read_data(os.path.join(data_folder, data_file))

#         # Specify vocabulary
#         if training_set: 
#             self.vocab, self.inverse_vocab = self.construct_vocab(source, model=embedding_model)
#         elif vocab is not None: 
#             self.vocab = vocab
#         else: 
#             raise ValueError('Either a vocabulary must be provided \
#                 or the training flag should be set to True.') 

#         # Vocabulary size
#         self.vocab_size = len(self.vocab)

#         # Encode corpus 
#         encoded_sequences, length_mask, long_sequence = self.encode_source(source)

#         # Remove long sequences
#         labels = labels[np.invert(long_sequence)]
#         length_mask = length_mask[np.invert(long_sequence)]
#         encoded_sequences = encoded_sequences[np.invert(long_sequence),:]

#         print('%i sequences were found having length longer than %i.' \
#             %(np.sum(long_sequence), self.sequence_len))

#         # Sanity check
#         if len(labels) != encoded_sequences.shape[0]:
#             raise ValueError('Data shapes are inconsistent.') 

#         self.labels = labels
#         self.length_mask = length_mask
#         self.encoded_sequences = encoded_sequences

#         # Number of classes 
#         self.n_classes = np.unique(self.labels)

#     def __len__(self): 
#         return len(self.labels)

#     def __getitem__(self, idx): 
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         length_mask = self.length_mask[idx]
#         sequence = self.encoded_sequences[idx, :] 

#         label = self.labels[idx]

#         sample = {
#             'label': torch.tensor(label),
#             'tokens': torch.tensor(sequence),  
#             'length': torch.tensor(length_mask, dtype=torch.int64)
#         }

#         return sample

#     def read_data(self, file_path): 
#         """
#         """
#         file = pd.read_csv(file_path, sep='\t')

#         labels = file['Target'].to_numpy()
#         source = file['Source'].to_numpy()

#         return source, labels

#     def encode_source(self, source): 
#         """
#         """
#         # Mask to indicate long sequences
#         long_sequence = np.zeros(source.shape[0], dtype=bool)

#         # Encoded sentences
#         encoded_sequences = np.full(shape=(source.shape[0], self.sequence_len), 
#                                     fill_value=self.vocab['<PAD>'], dtype=int)

#         for i, sample in enumerate(source): 
#             # Remove parentheses of type '(' and ')'
#             sample = sample.replace('(', '')
#             sample = sample.replace(')', '')

#             # White space tokenization
#             tokens = sample.strip().split()

#             if len(tokens) > self.sequence_len - 1: 
#                 long_sequence[i] = True 
#                 continue

#             # Reset pointer 
#             t = 0

#             # Begginning of sequence 
#             encoded_sequences[i,t] = self.vocab['<CLS>']
#             t += 1

#             for token in tokens: 
#                 if token in self.vocab.keys(): 
#                     encoded_sequences[i,t] = self.vocab[token]
#                 else: 
#                     encoded_sequences[i,t] = self.vocab['<UNK>']
#                 # Increment pointer
#                 t += 1

#         padding_mask = np.zeros(shape=encoded_sequences.shape)
#         padding_mask[encoded_sequences != self.vocab['<PAD>']] = 1

#         # Length of each sequence 
#         length_mask = np.sum(padding_mask, axis=1)

#         return encoded_sequences, length_mask, long_sequence

#     def construct_vocab(self, source, model=None, base_vocab={'<PAD>': 0, '<CLS>': 1, '<SEP>': 2, '<UNK>': 3}): 
#         ignored = 0
#         counter = collections.Counter()

#         for sample in source: 
#             # Remove parentheses of type '(' and ')'
#             sample = sample.replace('(', '')
#             sample = sample.replace(')', '')

#             # White space tokenization
#             tokens = sample.strip().split()

#             # Ignore long sequences
#             if len(tokens) > self.sequence_len - 1: 
#                 continue

#             counter.update(tokens)

#         # Initialie vocabulary 
#         vocab = dict(base_vocab)

#         # Associate IDs 
#         token_id = len(base_vocab)

#         for token, _ in counter.most_common(): 
#             if model is not None: 
#                 if token in model: 
#                     vocab[token] = token_id
#                     token_id += 1
#                 else: 
#                     ignored += 1
#             else: 
#                 vocab[token] = token_id 
#                 token_id += 1

#         # Inverse vocabulary
#         inverse_vocab = {v: k for k, v in vocab.items()}

#         print('Constructed vocabulary of size: ', len(vocab))

#         if model is not None: 
#             print('Number of tokens that were not found in the provided model: ', ignored)

#         return vocab, inverse_vocab

