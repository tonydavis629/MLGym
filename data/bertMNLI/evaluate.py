"""
Copyright (c) Meta Platforms, Inc. and affiliates.
"""

import json

import torch
import transformers
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer

transformers.logging.set_verbosity_error()


# Constants
MAX_LENGTH = 128
BATCH_SIZE = 32

class MNLIDataset(Dataset):
    def __init__(self, split="train"):
        # Load MNLI dataset
        self.dataset = load_dataset("multi_nli")[split]
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        # Tokenize premise and hypothesis together
        encoding = self.tokenizer(
            item['premise'],
            item['hypothesis'],
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

def evaluate_model():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = BertForSequenceClassification.from_pretrained('bert_mnli_model')
    model.eval()
    model.to(device)

    # Setup data
    test_dataset = MNLIDataset("validation_matched")
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    result = {"validation_accuracy": 100 * correct / total}
    print(json.dumps(result))

if __name__ == "__main__":
    evaluate_model()
