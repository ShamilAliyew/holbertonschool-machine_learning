#!/usr/bin/env python3
"""
4-qa.py
Interactive QA loop across multiple reference documents using semantic search.
"""

from 0-qa import question_answer as qa_single
from 3-semantic_search import semantic_search

def question_answer(corpus_path):
    """
    Interactive QA loop using multiple reference texts in a corpus.

    Args:
        corpus_path (str): path to folder containing reference documents
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ").strip()
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        # Find the most relevant document
        doc = semantic_search(corpus_path, question)
        if doc is None:
            print("A: Sorry, I do not understand your question.")
            continue

        # Get answer from BERT QA model
        answer = qa_single(question, doc)
        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")