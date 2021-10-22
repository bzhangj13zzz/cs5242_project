from urllib.parse import unquote
from deep_translator.google_trans import GoogleTranslator
import pandas as pd
import os
import re
from pandas.core.frame import DataFrame
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from nltk.tokenize import word_tokenize

from vocab import Vocabulary

def translate_garbage_names(csv_path, translated_path='translated.csv', overwrite=False):
    if os.path.exists(translated_path) and not overwrite:
        print('Translated file already exists!')
    else:
        df = pd.read_csv(csv_path, names=["cn", "label"])
        chinese_names = df['cn'].tolist()
        # Call the Google translation API, this will take very long
        translator = GoogleTranslator(source='chinese', target='english')
        translated = translator.translate_batch(chinese_names)
        df['en'] = translated
        df.to_csv(translated_path, index=False)
    return translated_path

def preprocess(csv_path, preprocessed_path='preprocessed.csv', overwrite=False):
    if os.path.exists(preprocessed_path) and not overwrite:
        print('Preprocessed file already exists!')
    else:
        # Preprocessing includes:
        # Lowercasing
        # Remove all non-alphabet characters
        df = pd.read_csv(csv_path)
        original_words = df['en'].tolist()
        cleaned_words = [re.sub(r'[^A-Za-z ]+', '', w.lower()) for w in original_words]
        df['cleaned_en'] = cleaned_words
        df.to_csv(preprocessed_path, index=False)
    return preprocessed_path

def load_glove_to_dict(glove_path):
    embeddings_dict = {}
    with open(glove_path, 'r') as f:
        for line in tqdm(f):
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype=np.float)
            embeddings_dict[word] = vector
    return embeddings_dict

def embed_sequence(seq, max_len, embedding_dict: dict):
    # Embed a sequence using given embedding dict
    # The embedded sequence will have shape (max_len, emb_dim)
    emb_dim = len(list(embedding_dict.values())[0])
    seq_tensor = torch.zeros((max_len, emb_dim))
    for idx, word in enumerate(seq.split(' ')):
        if word in embedding_dict:
            seq_tensor[idx] = torch.tensor(embedding_dict[word])
    return seq_tensor

def build_wvecs(embedding_dict: dict, vocab: Vocabulary):
    # The word vectors, which will be used to initialise the embedding layer in our model
    wvecs = []

    for w in vocab.get_vocab_set():
        if w in embedding_dict:
            wvecs.append(embedding_dict[w])
        else:
            wvecs.append([0]*100)
    
    return wvecs

def build_vocab(df: DataFrame):
    tokenized_seqs = [word_tokenize(seq) for seq in df['cleaned_en']]
    vocab = Vocabulary()
    for s in tokenized_seqs:
        for w in s:
            vocab.add_word(w)
    return vocab

def collate_fn(batch):
    batch_labels = [l for _, l in batch]
    batch_features = [f for f, _ in batch]

    batch_features_len = [len(f) for f, _ in batch]

    seq_tensor = torch.zeros((len(batch), max(batch_features_len))).long()

    for idx, (seq, seqlen) in enumerate(zip(batch_features, batch_features_len)):
        seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

    batch_labels = torch.FloatTensor(batch_labels)

    return seq_tensor, batch_labels

def calc_acc(pred, label):
    classes = torch.argmax(pred, dim=1)
    return torch.mean((classes == label).float())

class GarbageDataset(Dataset):
    def __init__(self, csv_path, vocab: Vocabulary, max_length=10):
        self.label_mapping = {1: 0, 2: 1, 4: 2, 8: 3}
        df = pd.read_csv(csv_path)
        df = df[df['label'] != 16]

        seqs = df['cleaned_en'].tolist()
        tokenized_seqs = [word_tokenize(seq) for seq in seqs]
        vectorized_seqs = []
        for seq in tokenized_seqs:
            vectorized_seq = vocab.convert_words_to_idxs(seq)
            if len(vectorized_seq) > max_length:
                vectorized_seq = vectorized_seq[:max_length]
            else:
                while len(vectorized_seq) < max_length:
                    vectorized_seq.append(0)
            vectorized_seqs.append(vectorized_seq)
        
        labels = [self.label_mapping[l] for l in df['label'].tolist()]
        # Input vector will have shape (max_len * emb_dim)
        self.x = vectorized_seqs
        self.y = labels
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]