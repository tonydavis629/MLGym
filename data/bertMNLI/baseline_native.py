"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Constants
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
NUM_LABELS = 3
VOCAB_SIZE = 30522  # BERT vocabulary size
HIDDEN_SIZE = 768
NUM_ATTENTION_HEADS = 12
NUM_ENCODER_LAYERS = 12
DROPOUT = 0.1

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        seq_length = x.size(1)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, 3 * d_model)
        qkv = self.qkv(x)
        
        # Split into q, k, v and heads
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            # Reshape mask to match attention scores shape
            # mask: (batch_size, seq_length) -> (batch_size, 1, 1, seq_length)
            mask = mask.unsqueeze(1).unsqueeze(2)
            mask = mask.expand(-1, -1, seq_length, -1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        context = torch.matmul(attn, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.proj(context)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self attention block
        attn_output = self.self_attn(x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feedforward block
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class BERTClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, num_labels, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(MAX_LENGTH, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        # Create position ids once
        self.register_buffer(
            'position_ids',
            torch.arange(MAX_LENGTH).expand((1, -1))
        )
        
        # Transformer layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels)
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Get embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(self.position_ids[:, :input_ids.size(1)])
        
        # Combine embeddings
        x = token_embeds + position_embeds
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.encoder_layers:
            x = layer(x, attention_mask)
        
        # Pool the output (use [CLS] token representation)
        pooled_output = x[:, 0]
        
        # Classify
        return self.classifier(pooled_output)

class WordPieceTokenizer:
    def __init__(self):
        self.vocab = self._load_vocab()
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.pad_token = "[PAD]"
        self.special_tokens = [self.unk_token, self.cls_token, self.sep_token, self.pad_token]
        
    def _load_vocab(self):
        # Load the BERT vocabulary
        vocab = {}
        # Basic vocab with special tokens
        vocab["[PAD]"] = 0
        vocab["[UNK]"] = 1
        vocab["[CLS]"] = 2
        vocab["[SEP]"] = 3
        vocab["[MASK]"] = 4
        
        # Load the rest of the vocabulary
        # This is a simplified version of BERT's vocabulary
        with open("bert-base-uncased-vocab.txt", "r", encoding="utf-8") as f:
            for i, token in enumerate(f, start=len(vocab)):
                vocab[token.strip()] = i
                if i >= VOCAB_SIZE - 1:  # -1 to account for special tokens
                    break
        return vocab
    
    def tokenize(self, text):
        # Basic preprocessing
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        words = text.split()
        
        tokens = []
        for word in words:
            if word in self.vocab:
                tokens.append(word)
            else:
                tokens.append(self.unk_token)
        return tokens

class MNLIDataset(Dataset):
    def __init__(self, split="train"):
        self.dataset = load_dataset("multi_nli")[split]
        self.tokenizer = WordPieceTokenizer()
        
        # Cache the vocabulary mapping
        self.vocab = self.tokenizer.vocab
        self.pad_token_id = self.vocab[self.tokenizer.pad_token]
        self.cls_token_id = self.vocab[self.tokenizer.cls_token]
        self.sep_token_id = self.vocab[self.tokenizer.sep_token]
    
    def __len__(self):
        return len(self.dataset)
    
    def _tokenize_and_encode(self, premise, hypothesis):
        # Tokenize both sequences
        premise_tokens = self.tokenizer.tokenize(premise)
        hypothesis_tokens = self.tokenizer.tokenize(hypothesis)
        
        # Calculate maximum lengths for each sequence to fit within MAX_LENGTH
        # Account for [CLS] and two [SEP] tokens
        max_seq_length = MAX_LENGTH - 3
        max_premise_length = min(len(premise_tokens), max_seq_length // 2)
        max_hypothesis_length = min(len(hypothesis_tokens), 
                                  max_seq_length - max_premise_length)
        
        # Truncate if necessary
        premise_tokens = premise_tokens[:max_premise_length]
        hypothesis_tokens = hypothesis_tokens[:max_hypothesis_length]
        
        # Combine tokens with special tokens
        tokens = ["[CLS]"] + premise_tokens + ["[SEP]"] + hypothesis_tokens + ["[SEP]"]
        
        # Convert tokens to ids
        input_ids = [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]
        
        # Create attention mask
        attention_mask = [1] * len(input_ids)
        
        # Pad sequences
        padding_length = MAX_LENGTH - len(input_ids)
        input_ids += [self.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        return torch.tensor(input_ids), torch.tensor(attention_mask)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        premise = item['premise']
        hypothesis = item['hypothesis']
        label = item['label']
        
        input_ids, attention_mask = self._tokenize_and_encode(premise, hypothesis)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label)
        }

# Download BERT vocabulary if not exists
def download_bert_vocab():
    if not os.path.exists("bert-base-uncased-vocab.txt"):
        import requests
        url = "https://huggingface.co/bert-base-uncased/raw/main/vocab.txt"
        response = requests.get(url)
        with open("bert-base-uncased-vocab.txt", "w", encoding="utf-8") as f:
            f.write(response.text)

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = BERTClassifier(
        vocab_size=VOCAB_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_ENCODER_LAYERS,
        num_heads=NUM_ATTENTION_HEADS,
        num_labels=NUM_LABELS,
        dropout=DROPOUT
    ).to(device)
    
    # Setup data
    train_dataset = MNLIDataset("train")
    val_dataset = MNLIDataset("validation_matched")
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=4,
        pin_memory=True
    )
    
    # Setup optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        print(f'Epoch {epoch + 1}:')
        print(f'Average Training Loss: {train_loss / len(train_loader):.4f}')
        print(f'Average Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    download_bert_vocab()
    train_model()
