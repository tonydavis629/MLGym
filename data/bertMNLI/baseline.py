"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import numpy as np
import torch
from datasets import load_dataset
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

# Constants
MAX_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 1e-7
NUM_LABELS = 3  # MNLI has 3 classes: entailment, contradiction, neutral


class MNLIDataset(Dataset):
    def __init__(self, split="train"):
        # Load MNLI dataset
        self.dataset = load_dataset("SetFit/mnli")[split]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Tokenize premise and hypothesis together
        encoding = self.tokenizer(
            item['text1'],
            item['text2'],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(item['label'])
        }

def train_model():
    # Setup device
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    # Load model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', 
        num_labels=NUM_LABELS
    ).to(device)

    # Setup data
    train_dataset = MNLIDataset("train")
    val_dataset = MNLIDataset("validation")
    
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
    
    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Setup gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{EPOCHS}'):
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass with mixed precision
            with autocast(device_type=device_str):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

            # Backward pass with gradient scaling
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

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        # Print metrics
        print(f'Epoch {epoch + 1}:')
        print(f'Average Training Loss: {train_loss / len(train_loader):.4f}')
        print(f'Average Validation Loss: {val_loss / len(val_loader):.4f}')
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

        # Save the model after training
        model_save_path = 'bert_mnli_model'
        model.save_pretrained(model_save_path)

if __name__ == "__main__":
    train_model()
