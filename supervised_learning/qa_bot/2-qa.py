#!/usr/bin/env python3
"""
2-qa.py
Interactive QA loop using a reference document and BERT QA.
"""

from

0 - qa
import question_answer


def answer_loop(reference):
    """
    Continuously prompt user with Q: and answer using reference text.

    If the answer is not found, respond with fallback message.
    Exits on exit/quit/goodbye/bye (case-insensitive).
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ").strip()
        if question.lower() in exit_words:
            print("A: Goodbye")
            break

        answer = question_answer(question, reference)
        if answer is None or answer.strip() == "":
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")