# Copyright (c) Meta Platforms, Inc. and affiliates.

#!/home/agent/miniconda3/envs/mlgym_generic/bin/python

# @yaml
# signature: memory_read
# docstring: Retrieve top-2 elements from memory most similar to a query.
# arguments:
#   query:
#       type: string
#       description: query to search for in memory
#       required: true

import argparse
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from pathlib import Path

# initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

MEMORY_FILE_PATH = Path("/home/agent/memory.json")

def memory_read(query_str: str, top_k: int = 2, memory_file_path: Path = MEMORY_FILE_PATH) -> None:
    """
    Retrieves top_k most similar texts to the query using cosine similarity.
    """
    if not memory_file_path.exists():
        print(f"Memory file not found: {memory_file_path}")
        return

    # load memory from JSON
    try:
        with open(memory_file_path, "r") as file:
            memory_data = json.load(file)
    except Exception as e:
        print(f"An error occurred while loading memory: {str(e)}")
        return

    if not memory_data.get('texts') or not memory_data.get('embeddings'):
        print("Memory is empty or improperly formatted.")
        return

    # tokenize and encode the query
    inputs = tokenizer(query_str, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

    # calculate cosine similarities
    memory_embeddings = np.array(memory_data['embeddings'])
    similarities = cosine_similarity([query_embedding], memory_embeddings).flatten()

    # get top_k results
    top_indices = similarities.argsort()[-top_k:][::-1]
    print(f"Query: {query_str}")
    print("Top Matches:")
    for i, idx in enumerate(top_indices):
        print(f"  Match {i+1}: {memory_data['texts'][idx]} | similarity score: {similarities[idx]:.2f}")


parser = argparse.ArgumentParser(description="Find top-2 most similar texts to the query")
parser.add_argument(
    "query_str", type=str, help="Query to search for in memory"
)
args = parser.parse_args()

memory_read(query_str=args.query_str, memory_file_path=MEMORY_FILE_PATH)
