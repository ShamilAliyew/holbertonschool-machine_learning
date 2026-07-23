#!/usr/bin/env python3
"""
0-qa.py
Simple question-answering using a BERT QA model.
"""

import tensorflow_hub as hub
import tensorflow_text
from transformers import BertTokenizer

def question_answer(question, reference):
    """
    Finds a snippet in `reference` that answers `question`.

    Args:
        question (str): The question to answer.
        reference (str): The document text.

    Returns:
        str or None: Answer snippet or None if not found.
    """
    try:
        # Load pre-trained tokenizer
        tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )

        # Tokenize question + context
        inputs = tokenizer.encode_plus(question, reference, return_tensors="tf")

        # Load BERT QA model from TF-Hub
        model = hub.load(
            "https://tfhub.dev/see--/bert-uncased-tf2-qa/1"
        )

        # Run model
        outputs = model([inputs["input_ids"], inputs["attention_mask"]])
        start_logits, end_logits = outputs["start_logits"], outputs["end_logits"]

        # Get start and end positions
        start_idx = tf.argmax(start_logits, axis=1).numpy()[0]
        end_idx = tf.argmax(end_logits, axis=1).numpy()[0] + 1

        # Decode answer
        answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

        # Return None if empty
        return answer if answer.strip() != "" else None

    except Exception:
        return None