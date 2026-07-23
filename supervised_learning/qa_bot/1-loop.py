#!/usr/bin/env python3
"""
1-loop.py
Simple interactive loop for Q&A.
"""

def main():
    """
    Continuously prompt user with Q: and respond with A:.
    Exits when the user types exit, quit, goodbye, or bye.
    """
    exit_words = {"exit", "quit", "goodbye", "bye"}

    while True:
        question = input("Q: ").strip()
        if question.lower() in exit_words:
            print("A: Goodbye")
            break
        print("A:")

if __name__ == "__main__":
    main()