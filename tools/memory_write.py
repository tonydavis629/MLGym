# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/home/agent/miniconda3/envs/mlgym_generic/bin/python

# @yaml
# signature: memory_write <content_str>
# docstring: Save important results, configs or findings to memory.
# arguments:
#   content_str:
#       type: string
#       description: content to be saved to persistent memory
#       required: true

from transformers import AutoTokenizer, AutoModel
import torch
import json
from pathlib import Path
import argparse


# initialize model and tokenizer
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

MEMORY_FILE_PATH = Path("/home/agent/memory.json")


def generate_ngrams(sequence, n):
    """
    Generates n-grams from a given sequence of tokens.
    """
    return [' '.join(sequence[i:i+n]) for i in range(len(sequence)-n+1)]

def extract_keywords(content_str: str, content_embedding: torch.Tensor, model, tokenizer, top_n: int = 1, ngram_size: int = 3) -> str:
    """
    Extracts top-n n-grams using contextual embeddings.
    """
    # split content into candidate words/phrases
    tokens = content_str.lower().split()
    candidate_phrases = generate_ngrams(tokens, ngram_size)
    candidate_embeddings = []

    # compute embeddings for each candidate phrase
    for phrase in candidate_phrases:
        inputs = tokenizer(phrase, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            phrase_embedding = model(**inputs).last_hidden_state.mean(dim=1).squeeze(0).numpy()
        candidate_embeddings.append(phrase_embedding)

    # compute similarity scores
    similarity_scores = [
        (phrase, (content_embedding @ phrase_embedding) /
         (torch.norm(torch.tensor(content_embedding)) * torch.norm(torch.tensor(phrase_embedding))))
        for phrase, phrase_embedding in zip(candidate_phrases, candidate_embeddings)
    ]
    # sort and return top-n n-grams
    sorted_keywords = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    keywords = [keyword[0] for keyword in sorted_keywords[:top_n]]
    return " ".join(keywords) if keywords else ""


def memory_write(content_str: str, memory_file_path: Path = MEMORY_FILE_PATH) -> None:
    """
    Converts text to embeddings, extracts tags, and saves them to a memory file in JSON format.
    """
    # tokenize and encode
    inputs = tokenizer(content_str, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    content_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0).numpy()  # Convert to NumPy array

    # extract keywords/tags
    tags = extract_keywords(content_str, content_embedding, model, tokenizer, top_n=1, ngram_size=3)

    # load existing memory or initialize if not present
    if memory_file_path.exists() and memory_file_path.stat().st_size > 0:
        with open(memory_file_path, "r") as file:
            memory_data = json.load(file)
    else:
        memory_data = {'texts': [], 'embeddings': [], 'tags': []}

    # append new content
    memory_data['texts'].append(content_str)
    memory_data['embeddings'].append(content_embedding.tolist())
    if tags:
        memory_data['tags'].append(tags)

    # save memory as JSON
    with open(memory_file_path, "w") as file:
        json.dump(memory_data, file, indent=4)

    print("Content and tags saved successfully to memory.")

parser = argparse.ArgumentParser(description="Save important results, configs or findings to memory.")
parser.add_argument(
    "content_str", type=str, help="Content to be saved to persistent memory"
)
args = parser.parse_args()

memory_write(content_str=str(args.content_str), memory_file_path=MEMORY_FILE_PATH)
