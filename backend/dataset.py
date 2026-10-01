import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import ast
from collections import Counter
import os

class CornellMovieDataset(Dataset):
    def __init__(self, lines_path, conv_path, shakespeare_path, max_length=60, vocab_size=10000):
        self.max_length = max_length
        print("Loading datasets with pandas...")
        
        self.qa_pairs = []
        
        # Load Cornell Movie Dialogs
        if os.path.exists(lines_path) and os.path.exists(conv_path):
            self.lines_df = pd.read_csv(
                lines_path, 
                sep=r' \+\+\+\$\+\+\+ ', 
                engine='python', 
                names=['lineID', 'characterID', 'movieID', 'character', 'text'],
                on_bad_lines='skip',
                encoding='latin1'
            )
            
            self.conv_df = pd.read_csv(
                conv_path, 
                sep=r' \+\+\+\$\+\+\+ ', 
                engine='python', 
                names=['char1ID', 'char2ID', 'movieID', 'lineIDs'],
                on_bad_lines='skip',
                encoding='latin1'
            )
            self.qa_pairs.extend(self._extract_qa_pairs())
        
        # Load Shakespeare
        if os.path.exists(shakespeare_path):
            print("Loading Shakespeare...")
            with open(shakespeare_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # Simple heuristic: treat consecutive lines as Q & A if they have words
                clean_lines = [l.strip().lower() for l in lines if len(l.strip().split()) > 2]
                for i in range(len(clean_lines) - 1):
                    self.qa_pairs.append((clean_lines[i], clean_lines[i+1]))
        
        print("Building vocabulary...")
        self.vocab, self.word2idx, self.idx2word = self._build_vocab(vocab_size)
        
        # Pad and Unk tokens
        self.PAD_IDX = self.word2idx['<PAD>']
        self.UNK_IDX = self.word2idx['<UNK>']
        self.BOS_IDX = self.word2idx['<BOS>'] # Beginning of sentence
        self.EOS_IDX = self.word2idx['<EOS>'] # End of sentence
        
        print(f"Dataset ready! Total QA pairs: {len(self.qa_pairs)}")

    def _extract_qa_pairs(self):
        line_dict = pd.Series(self.lines_df.text.values, index=self.lines_df.lineID).to_dict()
        qa_pairs = []
        
        for idx, row in self.conv_df.iterrows():
            try:
                line_ids = ast.literal_eval(row['lineIDs'])
                for i in range(len(line_ids) - 1):
                    q = str(line_dict.get(line_ids[i], "")).strip().lower()
                    a = str(line_dict.get(line_ids[i+1], "")).strip().lower()
                    if q and a:
                        qa_pairs.append((q, a))
            except Exception:
                continue
                
        return qa_pairs

    def _build_vocab(self, max_size):
        counter = Counter()
        for q, a in self.qa_pairs:
            counter.update(q.split())
            counter.update(a.split())
            
        vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        vocab.extend([word for word, count in counter.most_common(max_size - 4)])
        
        word2idx = {word: idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        return vocab, word2idx, idx2word
        
    def _tokenize_and_pad(self, text):
        words = text.split()
        tokens = [self.word2idx.get(w, self.word2idx['<UNK>']) for w in words]
        return tokens

    def __len__(self):
        return len(self.qa_pairs)

    def __getitem__(self, idx):
        q, a = self.qa_pairs[idx]
        
        # Combine Q and A for causal modeling: <BOS> Q <EOS> A <EOS>
        q_tokens = self._tokenize_and_pad(q)
        a_tokens = self._tokenize_and_pad(a)
        
        # Sequence: BOS + Q + EOS + A + EOS
        full_seq = [self.BOS_IDX] + q_tokens + [self.EOS_IDX] + a_tokens + [self.EOS_IDX]
        
        # Truncate if too long
        if len(full_seq) > self.max_length + 1:
            full_seq = full_seq[:self.max_length + 1]
            
        # Pad sequence
        if len(full_seq) < self.max_length + 1:
            full_seq.extend([self.PAD_IDX] * ((self.max_length + 1) - len(full_seq)))
            
        # Input is everything except last token, Target is everything except first token
        inputs = torch.tensor(full_seq[:-1], dtype=torch.long)
        targets = torch.tensor(full_seq[1:], dtype=torch.long)
            
        return inputs, targets
