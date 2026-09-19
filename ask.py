"""
ask.py
------
A tiny convenience script so you don't need curl to test the API.

Usage (with app.py already running in another terminal):
    python ask.py "How many days of annual leave do I get?"
"""

import sys
import requests

def main():
    if len(sys.argv) < 2:
        print('Usage: python ask.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    response = requests.post(
        "http://localhost:5000/ask",
        json={"question": question},
        timeout=30,
    )
    data = response.json()
    print()
    if "error" in data:
        print("Error:", data["error"])
    else:
        print("Q:", data["question"])
        print("A:", data["answer"])
        print("Sources:", ", ".join(data["sources"]))
    print()

if __name__ == "__main__":
    main()
