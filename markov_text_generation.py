"""
Markov Chain Text Generation - ProDigy Infotech ML Internship Task 03
Implements character-level and word-level Markov chain text generation
from scratch, plus an optional markovify-powered version.
"""

import random
import re
import json
from collections import defaultdict
from typing import Optional


# ──────────────────────────────────────────────
# 1.  WORD-LEVEL MARKOV CHAIN
# ──────────────────────────────────────────────

class WordMarkovChain:
    """
    Word-level Markov Chain text generator.

    Builds a statistical model where each state is a tuple of `order`
    consecutive words, and transitions are the words that follow.
    """

    def __init__(self, order: int = 2):
        """
        Parameters
        ----------
        order : int
            Number of preceding words used as context (n-gram order).
            order=1 → unigram context, order=2 → bigram context, etc.
        """
        self.order = order
        self.chain = defaultdict(list)
        self.start_states = []

    def train(self, text: str) -> None:
        """
        Build the Markov chain from a raw text string.

        Parameters
        ----------
        text : str
            Training corpus (any plain text).
        """
        # Tokenize into sentences then words
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())

        for sentence in sentences:
            words = sentence.split()
            if len(words) <= self.order:
                continue

            # Record valid start states (beginning of each sentence)
            start = tuple(words[:self.order])
            self.start_states.append(start)

            # Build transitions
            for i in range(len(words) - self.order):
                state = tuple(words[i:i + self.order])
                next_word = words[i + self.order]
                self.chain[state].append(next_word)

        print(f"✅ Word-level chain trained | order={self.order} | "
              f"{len(self.chain)} unique states | "
              f"{sum(len(v) for v in self.chain.values())} transitions")

    def generate(
        self,
        seed: Optional[str] = None,
        max_words: int = 100,
        num_sequences: int = 3,
    ) -> list[str]:
        """
        Generate text sequences using the trained chain.

        Parameters
        ----------
        seed       : optional starting phrase (must be >= order words)
        max_words  : maximum words per generated sequence
        num_sequences : how many sequences to produce
        """
        results = []

        for i in range(num_sequences):
            # Choose starting state
            if seed:
                seed_words = seed.strip().split()
                if len(seed_words) >= self.order:
                    current = tuple(seed_words[-self.order:])
                    words = list(seed_words)
                else:
                    print(f"⚠️  Seed too short for order={self.order}, using random start.")
                    current = random.choice(self.start_states)
                    words = list(current)
            else:
                current = random.choice(self.start_states)
                words = list(current)

            # Walk the chain
            for _ in range(max_words - self.order):
                next_words = self.chain.get(current)
                if not next_words:
                    break
                next_word = random.choice(next_words)
                words.append(next_word)
                current = tuple(words[-self.order:])

            text = " ".join(words)
            results.append(text)
            print(f"\n[Generated Text {i + 1}]\n{text}")

        return results

    def save(self, path: str) -> None:
        """Save the trained chain to a JSON file."""
        data = {
            "order": self.order,
            "chain": {str(k): v for k, v in self.chain.items()},
            "start_states": [list(s) for s in self.start_states],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        print(f"💾 Chain saved to '{path}'")

    def load(self, path: str) -> None:
        """Load a previously saved chain from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.order = data["order"]
        self.chain = defaultdict(list, {
            tuple(k.strip("()").replace("'", "").split(", ")): v
            for k, v in data["chain"].items()
        })
        self.start_states = [tuple(s) for s in data["start_states"]]
        print(f"📂 Chain loaded from '{path}'")


# ──────────────────────────────────────────────
# 2.  CHARACTER-LEVEL MARKOV CHAIN
# ──────────────────────────────────────────────

class CharMarkovChain:
    """
    Character-level Markov Chain text generator.

    Each state is a string of `order` consecutive characters;
    transitions are the characters that follow.
    """

    def __init__(self, order: int = 4):
        """
        Parameters
        ----------
        order : int
            Number of preceding characters used as context.
        """
        self.order = order
        self.chain = defaultdict(list)

    def train(self, text: str) -> None:
        """
        Build the character-level chain from raw text.

        Parameters
        ----------
        text : str
            Training corpus.
        """
        for i in range(len(text) - self.order):
            state = text[i:i + self.order]
            next_char = text[i + self.order]
            self.chain[state].append(next_char)

        print(f"✅ Char-level chain trained | order={self.order} | "
              f"{len(self.chain)} unique states")

    def generate(
        self,
        seed: Optional[str] = None,
        max_chars: int = 500,
        num_sequences: int = 2,
    ) -> list[str]:
        """
        Generate text sequences character by character.

        Parameters
        ----------
        seed         : optional starting string (>= order chars)
        max_chars    : maximum characters per sequence
        num_sequences: how many sequences to produce
        """
        results = []
        all_states = list(self.chain.keys())

        for i in range(num_sequences):
            # Choose starting state
            if seed and len(seed) >= self.order:
                current = seed[-self.order:]
                result = seed
            else:
                current = random.choice(all_states)
                result = current

            # Walk the chain
            for _ in range(max_chars - self.order):
                next_chars = self.chain.get(current)
                if not next_chars:
                    # Dead end — pick a new random state
                    current = random.choice(all_states)
                    continue
                next_char = random.choice(next_chars)
                result += next_char
                current = result[-self.order:]

            results.append(result)
            print(f"\n[Char-Level Generation {i + 1}]\n{result}")

        return results


# ──────────────────────────────────────────────
# 3.  MARKOVIFY-POWERED VERSION (optional)
# ──────────────────────────────────────────────

def markovify_generate(
    text: str,
    state_size: int = 2,
    num_sentences: int = 5,
    max_chars: int = 280,
) -> list[str]:
    """
    Generate text using the markovify library.

    Parameters
    ----------
    text         : training corpus string
    state_size   : n-gram order (markovify's state_size)
    num_sentences: how many sentences to generate
    max_chars    : max character length per sentence
    """
    try:
        import markovify
    except ImportError:
        print("⚠️  markovify not installed. Run: pip install markovify")
        return []

    model = markovify.Text(text, state_size=state_size)
    results = []

    print(f"\n{'─'*55}")
    print(f"  markovify | state_size={state_size}")
    print(f"{'─'*55}")

    for i in range(num_sentences):
        sentence = model.make_short_sentence(max_chars=max_chars, tries=100)
        if sentence:
            results.append(sentence)
            print(f"[{i + 1}] {sentence}")
        else:
            print(f"[{i + 1}] ⚠️  Could not generate sentence (corpus may be too small)")

    return results


# ──────────────────────────────────────────────
# 4.  PROBABILITY TABLE (for analysis)
# ──────────────────────────────────────────────

def show_transition_probabilities(chain: WordMarkovChain, top_n: int = 10) -> None:
    """
    Display the top N most common states and their transition probabilities.

    Parameters
    ----------
    chain : WordMarkovChain
        A trained word-level Markov chain.
    top_n : int
        How many states to display.
    """
    print(f"\n{'='*55}")
    print("  Top Transition Probabilities")
    print(f"{'='*55}")

    # Sort by number of transitions
    sorted_states = sorted(chain.chain.items(), key=lambda x: len(x[1]), reverse=True)

    for state, transitions in sorted_states[:top_n]:
        total = len(transitions)
        freq = defaultdict(int)
        for word in transitions:
            freq[word] += 1

        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:3]
        probs = ", ".join(f"'{w}' ({c/total:.0%})" for w, c in top)
        print(f"\n  State {state}")
        print(f"  → {probs}")


# ──────────────────────────────────────────────
# 5.  QUICK DEMO
# ──────────────────────────────────────────────

def quick_demo(train_file: str = "train_data.txt") -> None:
    """Run a full demonstration of all Markov chain variants."""

    print("\n" + "=" * 55)
    print("  Markov Chain Text Generation")
    print("=" * 55)

    # Load training data
    try:
        with open(train_file, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"\n📄 Loaded '{train_file}' ({len(text)} characters, {len(text.split())} words)")
    except FileNotFoundError:
        print(f"⚠️  '{train_file}' not found. Using built-in sample text.")
        text = (
            "Artificial intelligence is transforming the world. "
            "Machine learning enables computers to learn from data. "
            "Deep learning uses neural networks with many layers. "
            "Natural language processing allows machines to understand text. "
            "Generative models can create new content similar to training data. "
            "The future of AI holds tremendous promise for society. "
            "Transformers have revolutionized natural language understanding. "
            "Transfer learning reduces the need for large labeled datasets. "
        ) * 10

    print("\n" + "─" * 55)
    print("  1️⃣  Word-Level Markov Chain (order=1)")
    print("─" * 55)
    wmc1 = WordMarkovChain(order=1)
    wmc1.train(text)
    wmc1.generate(max_words=60, num_sequences=2)

    print("\n" + "─" * 55)
    print("  2️⃣  Word-Level Markov Chain (order=2)")
    print("─" * 55)
    wmc2 = WordMarkovChain(order=2)
    wmc2.train(text)
    wmc2.generate(max_words=80, num_sequences=2)

    print("\n" + "─" * 55)
    print("  3️⃣  Word-Level Markov Chain (order=3)")
    print("─" * 55)
    wmc3 = WordMarkovChain(order=3)
    wmc3.train(text)
    wmc3.generate(
        seed="Artificial intelligence is",
        max_words=80,
        num_sequences=2,
    )

    print("\n" + "─" * 55)
    print("  4️⃣  Transition Probability Analysis")
    print("─" * 55)
    show_transition_probabilities(wmc2, top_n=5)

    print("\n" + "─" * 55)
    print("  5️⃣  Character-Level Markov Chain (order=4)")
    print("─" * 55)
    cmc = CharMarkovChain(order=4)
    cmc.train(text)
    cmc.generate(max_chars=300, num_sequences=2)

    print("\n" + "─" * 55)
    print("  6️⃣  markovify Library")
    print("─" * 55)
    markovify_generate(text, state_size=2, num_sentences=4)

    print("\n✅ Demo complete!\n")


# ──────────────────────────────────────────────
# 6.  ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Markov Chain Text Generation — ProDigy Infotech Task 03"
    )
    subparsers = parser.add_subparsers(dest="command")

    # demo
    demo_p = subparsers.add_parser("demo", help="Run full demo")
    demo_p.add_argument("--train_file", default="train_data.txt")

    # generate (word-level)
    gen_p = subparsers.add_parser("generate", help="Generate text (word-level)")
    gen_p.add_argument("--train_file", default="train_data.txt")
    gen_p.add_argument("--order", type=int, default=2)
    gen_p.add_argument("--seed", type=str, default=None)
    gen_p.add_argument("--max_words", type=int, default=100)
    gen_p.add_argument("--num_sequences", type=int, default=3)

    # char-level generate
    char_p = subparsers.add_parser("char", help="Generate text (character-level)")
    char_p.add_argument("--train_file", default="train_data.txt")
    char_p.add_argument("--order", type=int, default=4)
    char_p.add_argument("--max_chars", type=int, default=500)
    char_p.add_argument("--num_sequences", type=int, default=2)

    # markovify
    mkv_p = subparsers.add_parser("markovify", help="Generate using markovify library")
    mkv_p.add_argument("--train_file", default="train_data.txt")
    mkv_p.add_argument("--state_size", type=int, default=2)
    mkv_p.add_argument("--num_sentences", type=int, default=5)

    args = parser.parse_args()

    if args.command == "demo" or args.command is None:
        train_file = getattr(args, "train_file", "train_data.txt")
        quick_demo(train_file)

    elif args.command == "generate":
        with open(args.train_file, "r", encoding="utf-8") as f:
            text = f.read()
        wmc = WordMarkovChain(order=args.order)
        wmc.train(text)
        wmc.generate(
            seed=args.seed,
            max_words=args.max_words,
            num_sequences=args.num_sequences,
        )

    elif args.command == "char":
        with open(args.train_file, "r", encoding="utf-8") as f:
            text = f.read()
        cmc = CharMarkovChain(order=args.order)
        cmc.train(text)
        cmc.generate(max_chars=args.max_chars, num_sequences=args.num_sequences)

    elif args.command == "markovify":
        with open(args.train_file, "r", encoding="utf-8") as f:
            text = f.read()
        markovify_generate(text, state_size=args.state_size, num_sentences=args.num_sentences)