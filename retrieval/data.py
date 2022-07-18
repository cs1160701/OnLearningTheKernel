import os
import torch
import collections
import numpy as np 
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from torch.utils.data import Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class AnthologyNetworkCorpus(Dataset): 
    def __init__(self, file_path, sequence_len=4000):

        self.n_classes = 2
        self.sequence_len = sequence_len

        # Load dataset 
        dataset = self.load_data(file_path)

        # Encoder 
        self.encoder = tfds.features.text.ByteTextEncoder()

        # Vocab size 
        self.vocab_size = self.encoder.vocab_size

        # Encode data
        dataset = dataset.map(
            self.tokenize_encode, 
            num_parallel_calls=AUTOTUNE
        )

        # Pad sequences
        max_shape = {
            'Source1': [self.sequence_len], 
            'Source2': [self.sequence_len], 
            'Targets': []
        }

        dataset = dataset.padded_batch(
            batch_size=1, 
            padded_shapes=max_shape
        )

        # Convert tf dataset to arrays 
        input1 = []
        input2 = []
        labels = []

        iterator = iter(dataset)

        for sample in iterator: 
            labels.append(sample['Targets'].numpy()[0])
            input1.append(sample['Source1'].numpy()[0])
            input2.append(sample['Source2'].numpy()[0])
            
        self.input1 = np.array(input1, dtype=int)
        self.input2 = np.array(input2, dtype=int)
        self.labels = np.array(labels, dtype=int)

        self.length_mask1 = np.sum(
            self.input1 > 0, 
            axis=1
        )

        self.length_mask2 = np.sum(
            self.input2 > 0, 
            axis=1
        )
        
    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, idx): 
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        input1 = self.input1[idx, :] 
        input2 = self.input2[idx, :] 
        length_mask1 = self.length_mask1[idx]
        length_mask2 = self.length_mask2[idx]

        sample = {
            'label': torch.tensor(label),
            'input1': torch.tensor(input1),  
            'input2': torch.tensor(input2),  
            'length1': torch.tensor(length_mask1, dtype=torch.int64),
            'length2': torch.tensor(length_mask2, dtype=torch.int64)
        }

        return sample

    def load_data(self, file_path, batch_size=1): 

        col_names = ['label', 'id1', 'id2', 'text1', 'text2']
        col_types = [tf.float32, tf.string, tf.string, tf.string, tf.string]
        
        ds = tf.data.experimental.make_csv_dataset(
            [file_path],
            batch_size,
            column_names=col_names,
            column_defaults=col_types,
            use_quote_delim=False,
            field_delim='\t',
            header=False,
            shuffle=False,
            num_epochs=1
        )

        ds = ds.unbatch()

        def adapt_sample(x): 
            return {
                'Source1': x['text1'], 
                'Source2': x['text2'],
                'Target':  x['label']
            }

        ds = ds.map(adapt_sample)

        return ds

    def encode(self, x): 
        result = tf.py_function(
            lambda s: tf.constant(self.encoder.encode(s.numpy()[:10000])), 
            [x, ], 
            tf.int32
        )
        result.set_shape([None])
        return result

    def tokenize_encode(self, dataset): 
        return {
            'Source1': self.encode(dataset['Source1'])[:self.sequence_len], 
            'Source2': self.encode(dataset['Source2'])[:self.sequence_len], 
            'Targets': tf.cast(dataset['Target'], tf.int32)
        }
