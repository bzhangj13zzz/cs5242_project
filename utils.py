from urllib.parse import unquote
from deep_translator.google_trans import GoogleTranslator
import pandas as pd
import os
import re
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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

def extract_

class GarbageDataset(Dataset):
    def __init__(self, csv_path, embedding_dict):
        self.label_mapping = {1: 'recyclable', 2: 'hazardous', 4: 'wet', 8: 'dry'}
        df = pd.read_csv(csv_path)
        seqs = df['cleaned_en'].tolist()
        max_len = max([len(seq.split(' ')) for seq in seqs])
        labels = df['label'].tolist()
        # Input vector will have shape (max_len * emb_dim)
        self.x = [embed_sequence(seq, max_len, embedding_dict) for seq in seqs]
        self.y = labels
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]