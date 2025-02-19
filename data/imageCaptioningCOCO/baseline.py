"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import json
import os
import random
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from nltk.tokenize import word_tokenize
from PIL import Image
from pycocotools.coco import COCO
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

nltk.download('punkt')
nltk.download('punkt_tab')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Vocabulary Class
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freqs = Counter()

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        for sentence in sentence_list:
            tokens = word_tokenize(sentence.lower())
            self.freqs.update(tokens)

        for word, freq in self.freqs.items():
            if freq >= self.freq_threshold:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx

    def numericalize(self, text):
        tokenized_text = word_tokenize(text.lower())
        return [
            self.stoi.get(token, self.stoi['<UNK>'])
            for token in tokenized_text
        ]

    def detokenize(self, tokens):
        text_tokens = []
        for token in tokens:
            if token.item() not in [0, 1, 2, 3]:
                text_tokens.append(self.itos[token.item()])
        return ' '.join(text_tokens)


# COCO Dataset Class
class COCODataset(Dataset):
    def __init__(self, root_dir, parquet_file, vocab, split="train", transform=None):
        self.root = root_dir
        # Read the parquet file using pandas
        self.df = pd.read_parquet(parquet_file)
        self.vocab = vocab
        self.split = split
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Get image path and captions from dataframe
        row = self.df.iloc[index]
        img_path = row['file_name']
        # Randomly select one of the 5 captions
        caption = random.choice(row['anns'])["caption"]
        
        # Load and process image
        image = Image.open(os.path.join(self.root, self.split, img_path)).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Process caption
        numericalized_caption = [self.vocab.stoi['<SOS>']]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<EOS>'])

        return image, torch.tensor(numericalized_caption)


# Collate Function
def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)

    images = torch.stack(images, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths


# Encoder CNN
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


# Decoder RNN
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, vocab, max_seq_length=30):
        "Generate captions for given image features using greedy search."
        sampled_ids = []
        inputs = features.unsqueeze(1)
        states = None

        for _ in range(max_seq_length):
            hiddens, states = self.lstm(inputs, states)  # hiddens: (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))    # outputs:  (batch_size, vocab_size)
            _, predicted = outputs.max(1)                # predicted: (batch_size)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)               # inputs: (batch_size, embed_size)
            inputs = inputs.unsqueeze(1)                 # inputs: (batch_size, 1, embed_size)

            if predicted.item() == vocab.stoi['<EOS>']:
                break

        return torch.tensor(sampled_ids)


# Training Function
def train():
    # Hyperparameters
    embed_size = 256
    hidden_size = 512
    num_layers = 1
    learning_rate = 0.001
    num_epochs = 2
    batch_size = 128
    freq_threshold = 5

    # Paths (adjust as necessary)
    root_dir = './data'
    train_parquet_file = './data/train_images.parquet'
    val_parquet_file = './data/val_images.parquet'

    # Load captions
    df = pd.read_parquet(train_parquet_file)
    captions_list = []
    for _, row in df.iterrows():
        captions_list.extend([sample["caption"] for sample in row["anns"]])

    # Build Vocabulary
    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(captions_list)

    # Data Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # Dataset and DataLoader
    dataset = COCODataset(root_dir=root_dir, parquet_file=train_parquet_file, vocab=vocab, split="train", transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    encoder = EncoderCNN(embed_size).to(device)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi['<PAD>'])
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        for (images, captions, lengths) in tqdm(dataloader, desc=f'Epoch [{epoch+1}/{num_epochs}]'):
            images = images.to(device)
            captions = captions.to(device)
            targets = pack_padded_sequence(captions, lengths, batch_first=True)[0]

            # Forward
            features = encoder(images)
            outputs = decoder(features, captions, lengths)

            # Loss
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if i % 100 == 0:
                # print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}')

    # Evaluation on Validation Set
    eval_dataset = COCODataset(root_dir=root_dir, parquet_file=val_parquet_file, vocab=vocab, split="val", transform=transform)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    encoder.eval()
    decoder.eval()
    bleu_scores = []
    with torch.no_grad():
        for (images, captions, lengths) in tqdm(eval_dataloader, desc='Running Validation'):
            images = images.to(device)
            captions = captions.to(device)

            # Process entire batch at once
            features = encoder(images)
            sampled_ids = []
            for feature in features:
                sampled_ids.append(decoder.sample(feature.unsqueeze(0), vocab))

            # Calculate BLEU scores for the batch
            for caption, pred_ids in zip(captions, sampled_ids):
                ground_truth = vocab.detokenize(caption)
                sampled_caption = vocab.detokenize(pred_ids)
                bleu_score = nltk.translate.bleu_score.sentence_bleu([ground_truth], sampled_caption)
                bleu_scores.append(bleu_score)

    with open("submission.csv", "w") as f:
        f.write("bleu_score\n")
        for score in bleu_scores:
            f.write(f"{score}\n")


if __name__ == '__main__':
    set_seed()
    train()
