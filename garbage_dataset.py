class GarbageDataset(Dataset):
    def __init__(self, csv_path, embedding_dict):
        self.label_mapping = {1: 'recyclable', 2: 'hazardous', 4: 'wet', 8: 'dry'}
        df = pd.read_csv(csv_path)

        seqs = df['cleaned_en'].tolist()
        tokenized_seqs = [word_tokenize(seq) for seq in seqs]
        vocab = set([w for s in tokenized_seqs for w in s])

        labels = df['label'].tolist()
        # Input vector will have shape (max_len * emb_dim)
        self.y = labels
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]