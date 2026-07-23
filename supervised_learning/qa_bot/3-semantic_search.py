#!/usr/bin/env python3
"""
3-semantic_search.py
Perform semantic search on a corpus of documents.
"""

import os
from sentence_transformers import SentenceTransformer, util

def semantic_search(corpus_path, sentence):
    """
    Returns the most similar document from the corpus to the given sentence.

    Args:
        corpus_path (str): path to folder containing text documents
        sentence (str): query sentence for semantic search

    Returns:
        str: text of the most similar document
    """
    # Load corpus documents
    docs = []
    doc_names = []
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md") or filename.endswith(".txt"):
            path = os.path.join(corpus_path, filename)
            with open(path, "r", encoding="utf-8") as f:
                docs.append(f.read())
                doc_names.append(filename)

    if not docs:
        return None

    # Load pre-trained sentence transformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings
    corpus_embeddings = model.encode(docs, convert_to_tensor=True)
    query_embedding = model.encode(sentence, convert_to_tensor=True)

    # Compute cosine similarity
    similarities = util.cos_sim(query_embedding, corpus_embeddings)

    # Get index of most similar document
    best_idx = similarities.argmax().item()

    return docs[best_idx]