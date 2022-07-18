import torch
import numpy as np 
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from torch.utils.data import Dataset

AUTOTUNE = tf.data.experimental.AUTOTUNE

class IMBDbByteDataset(Dataset): 
    def __init__(self, file_name='imdb_reviews', training_set=False, sequence_len=1000): 

        self.sequence_len = sequence_len
        self.training_set = training_set

        # Load dataset 
        dataset = self.load_data(file_name)

        # Byte-Level encoder
        self.encoder = tfds.deprecated.text.ByteTextEncoder()

        # Vocabulary size
        self.vocab_size = self.encoder.vocab_size

        # Encode corpus 
        dataset = dataset.map(
            self.tokenize_encode, 
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
        source = np.empty(
            (len(dataset), self.sequence_len), 
            dtype=int 
        )

        labels = np.empty(
            len(dataset), 
            dtype=int
        )

        iterator = iter(dataset)

        for i, sample in enumerate(iterator): 
            labels[i] = sample['Target'].numpy()[0]
            source[i,:] = sample['Source'].numpy()[0]

        self.labels = labels 
        self.source = source

        self.length_mask = np.sum(
            self.source > 0, 
            axis=1
        )

        # Number of classes 
        self.n_classes = len(np.unique(self.labels))

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

    def load_data(self, file_name): 
        file = tfds.load(file_name)

        if self.training_set: 
            data = file['train']
        else: 
            data = file['test']

        def adapt_sample(x): 
            return {
                'Source': x['text'], 
                'Target': x['label']
            }

        data = data.map(adapt_sample)

        return data        

    def encode(self, x): 
        result = tf.py_function(
            lambda s: tf.constant(self.encoder.encode(s.numpy())), 
            [x, ], 
            tf.int32
        )
        result.set_shape([None])
        return result

    def tokenize_encode(self, dataset): 
        return {
            'Source': self.encode(dataset['Source'])[:self.sequence_len], 
            'Target': dataset['Target']
        }
