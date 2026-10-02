import ast
import os
import random
import re
from collections import Counter

import pandas as pd
import torch
from torch.utils.data import Dataset

# Words (keeping contractions like don't / i'm) OR single punctuation marks
TOKEN_RE = re.compile(r"\w+(?:'\w+)*|[^\w\s]")


def tokenize(text):
    text = re.sub(r"<[^>]+>", " ", str(text).lower())  # strip <u>, <i> tags etc.
    return TOKEN_RE.findall(text)


class SegawaDataset(Dataset):
    def __init__(self, data_dir, max_length=60, vocab_size=10000, answer_only_loss=True, load_all=True):
        self.max_length = max_length
        self.answer_only_loss = answer_only_loss

        raw_pairs = []  # (q_tokens, a_tokens, conversation_id)
        
        # Helper to load simple QA pairs
        def add_pairs(q_list, a_list, prefix):
            for i, (q_text, a_text) in enumerate(zip(q_list, a_list)):
                if not isinstance(q_text, str) or not isinstance(a_text, str): continue
                q = tokenize(q_text)
                a = tokenize(a_text)
                if q and a:
                    raw_pairs.append((q, a, f"{prefix}{i}"))

        # ---------- 1. Cornell Movie Dialogs ----------
        lines_path = os.path.join(data_dir, "movie_lines.txt")
        conv_path = os.path.join(data_dir, "movie_conversations.txt")
        if os.path.exists(lines_path) and os.path.exists(conv_path):
            print("Loading Cornell Movie Dialogs...")
            sep = r' \+\+\+\$\+\+\+ '
            lines_df = pd.read_csv(lines_path, sep=sep, engine='python', encoding='latin1', on_bad_lines='skip', names=['lineID', 'characterID', 'movieID', 'character', 'text'])
            conv_df = pd.read_csv(conv_path, sep=sep, engine='python', encoding='latin1', on_bad_lines='skip', names=['char1ID', 'char2ID', 'movieID', 'lineIDs'])
            line_dict = dict(zip(lines_df['lineID'], lines_df['text'].fillna('')))
            for conv_idx, line_ids_str in enumerate(conv_df['lineIDs']):
                try: line_ids = ast.literal_eval(line_ids_str)
                except: continue
                for i in range(len(line_ids) - 1):
                    q = tokenize(line_dict.get(line_ids[i], ""))
                    a = tokenize(line_dict.get(line_ids[i + 1], ""))
                    if q and a: raw_pairs.append((q, a, f"c{conv_idx}"))

        # ---------- 2. DailyDialogue ----------
        dd_dir = os.path.join(data_dir, "DailyDialogue")
        if load_all and os.path.exists(dd_dir):
            print("Loading DailyDialogue...")
            for split in ["train.csv", "validation.csv", "test.csv"]:
                path = os.path.join(dd_dir, split)
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    for idx, row in df.iterrows():
                        try:
                            dialog = ast.literal_eval(row['dialog'])
                            for i in range(len(dialog) - 1):
                                q, a = tokenize(dialog[i]), tokenize(dialog[i+1])
                                if q and a: raw_pairs.append((q, a, f"dd_{split}_{idx}"))
                        except: pass

        # ---------- 3. Dolly (databricks-dolly-15k) ----------
        dolly_path = os.path.join(data_dir, "Dolly", "train.csv")
        if load_all and os.path.exists(dolly_path):
            print("Loading Dolly...")
            df = pd.read_csv(dolly_path)
            # Combine instruction and context for the question
            qs = (df['instruction'].fillna('') + " " + df['context'].fillna('')).tolist()
            ans = df['response'].fillna('').tolist()
            add_pairs(qs, ans, "dolly")

        # ---------- 4. PersonaChat ----------
        persona_path = os.path.join(data_dir, "PersonaChat", "personachat_self_original.json")
        if load_all and os.path.exists(persona_path):
            print("Loading PersonaChat (this takes a moment)...")
            import json
            with open(persona_path, 'r', encoding='utf-8') as f:
                d = json.load(f)
                for split in ['train', 'valid']:
                    if split in d:
                        for idx, item in enumerate(d[split]):
                            history = item.get('history', [])
                            for i in range(len(history) - 1):
                                q, a = tokenize(history[i]), tokenize(history[i+1])
                                if q and a: raw_pairs.append((q, a, f"pc_{split}_{idx}"))

        # ---------- Vocabulary ----------
        print("Building vocabulary...")
        counter = Counter()
        for q, a, _ in raw_pairs:
            counter.update(q)
            counter.update(a)

        self.vocab = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.vocab += [w for w, _ in counter.most_common(vocab_size - 4)]
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.PAD_IDX = self.word2idx['<PAD>']
        self.UNK_IDX = self.word2idx['<UNK>']
        self.BOS_IDX = self.word2idx['<BOS>']
        self.EOS_IDX = self.word2idx['<EOS>']

        # ---------- Pre-encode, truncate safely ----------
        budget = max_length - 2
        max_q = budget // 2
        self.encoded, self.conv_ids = [], []
        dropped = 0
        
        # Add Pairs
        for q, a, cid in raw_pairs:
            q_ids = self.encode_tokens(q)[-max_q:]
            a_ids = self.encode_tokens(a)
            if len(a_ids) > budget - len(q_ids):
                dropped += 1
                continue
            self.encoded.append((q_ids, a_ids))
            self.conv_ids.append(cid)

        unk_rate = self._unk_rate()
        print(f"Dataset ready! Pairs kept: {len(self.encoded)} | dropped (answer too long): {dropped}")
        print(f"UNK rate: {unk_rate:.2f}% of tokens")

    # ---------- helpers (reuse these in your chat/inference script) ----------
    def encode_tokens(self, tokens):
        return [self.word2idx.get(t, self.UNK_IDX) for t in tokens]

    def encode_prompt(self, text):
        """Build the model prompt for generation: BOS + question + EOS."""
        budget = self.max_length - 2
        q = self.encode_tokens(tokenize(text))[-(budget // 2):]
        return [self.BOS_IDX] + q + [self.EOS_IDX]

    def decode(self, ids):
        out = []
        for i in ids:
            if i in (self.PAD_IDX, self.BOS_IDX):
                continue
            if i == self.EOS_IDX:
                break
            out.append(self.idx2word.get(int(i), '<UNK>'))
        text = " ".join(out)
        return re.sub(r"\s+([.,!?;:])", r"\1", text)

    def _unk_rate(self):
        total = unk = 0
        for q, a in self.encoded:
            for x in q + a:
                total += 1
                unk += (x == self.UNK_IDX)
        return 100.0 * unk / max(1, total)

    def split_by_conversation(self, val_frac=0.05, seed=42):
        """Split so no conversation appears in both train and val (Cornell lines
        are reused as question AND answer, so a random split leaks)."""
        ids = sorted(set(self.conv_ids))
        random.Random(seed).shuffle(ids)
        val_ids = set(ids[:max(1, int(len(ids) * val_frac))])
        train_idx = [i for i, c in enumerate(self.conv_ids) if c not in val_ids]
        val_idx = [i for i, c in enumerate(self.conv_ids) if c in val_ids]
        return train_idx, val_idx

    # ---------- Dataset API ----------
    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        q, a = self.encoded[idx]
        
        seq = [self.BOS_IDX] + q + [self.EOS_IDX] + a + [self.EOS_IDX]
        seq += [self.PAD_IDX] * (self.max_length + 1 - len(seq))
        inputs = torch.tensor(seq[:-1], dtype=torch.long)
        targets = torch.tensor(seq[1:], dtype=torch.long)

        if self.answer_only_loss:
            targets[:len(q) + 1] = self.PAD_IDX
                
        return inputs, targets