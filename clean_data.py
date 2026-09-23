import argparse
import html
import random
import re
import unicodedata
from pathlib import Path


USER_PROMPTS = [
    "Tell me a joke.",
    "Tell me a short joke.",
    "Give me a funny joke.",
    "Make me laugh.",
    "Tell me one joke.",
    "Give me a clean joke.",
]


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = html.unescape(text)
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("—", "-").replace("–", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def remove_possible_id(text: str) -> str:
    # Handles lines like:
    # 123,Why did...
    # 123\tWhy did...
    # 123;Why did...
    text = re.sub(r"^\s*\d+\s*[,;\t|]\s*", "", text)
    return text.strip()


def is_valid_joke(text: str, min_chars: int, max_chars: int) -> bool:
    if len(text) < min_chars:
        return False
    if len(text) > max_chars:
        return False
    if text.count("http") > 0:
        return False
    if len(set(text)) < 8:
        return False
    if re.fullmatch(r"[\W\d_]+", text):
        return False
    return True


def to_training_sample(joke: str, prompt: str) -> str:
    return (
        "<|bos|>\n"
        f"<|user|>{prompt}\n"
        "<|assistant|>\n"
        f"{joke}\n"
        "<|eos|>\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data.txt")
    parser.add_argument("--output", default="data_clean.txt")
    parser.add_argument("--min_chars", type=int, default=20)
    parser.add_argument("--max_chars", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    seen = set()
    jokes = []

    with input_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            joke = normalize_text(line)
            joke = remove_possible_id(joke)
            joke = normalize_text(joke)

            if not is_valid_joke(joke, args.min_chars, args.max_chars):
                continue

            key = joke.lower()
            key = re.sub(r"\s+", " ", key)

            if key in seen:
                continue

            seen.add(key)
            jokes.append(joke)

    random.shuffle(jokes)

    with output_path.open("w", encoding="utf-8", newline="\n") as f:
        for i, joke in enumerate(jokes):
            prompt = USER_PROMPTS[i % len(USER_PROMPTS)]
            f.write(to_training_sample(joke, prompt))
            f.write("\n")

    print(f"Input lines cleaned: {len(jokes)}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()