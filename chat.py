import argparse
import json
import re
from pathlib import Path

import torch

from model import GPT, GPTConfig
from tokenizer import JokeTokenizer


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def detect_intent(user_text: str) -> str:
    text = user_text.lower()

    if any(
        word in text
        for word in [
            "program",
            "developer",
            "code",
            "python",
            "java",
            "javascript",
            "bug",
            "computer",
            "software",
        ]
    ):
        return "programming"

    if any(word in text for word in ["dad", "father"]):
        return "dad"

    if any(word in text for word in ["short", "quick", "small"]):
        return "short"

    if any(word in text for word in ["clean", "family", "kid", "child"]):
        return "clean"

    if any(word in text for word in ["animal", "cat", "dog", "bird"]):
        return "animal"

    return "general"


def build_prompt(user_text: str) -> str:
    intent = detect_intent(user_text)

    if intent == "programming":
        request = "Tell me a programming joke."
    elif intent == "dad":
        request = "Tell me a dad joke."
    elif intent == "short":
        request = "Tell me a short joke."
    elif intent == "clean":
        request = "Give me a clean joke."
    elif intent == "animal":
        request = "Tell me an animal joke."
    else:
        request = "Tell me a joke."

    return (
        "<|bos|>\n"
        f"<|user|>{request}\n"
        "<|assistant|>\n"
    )


def extract_answer(full_text: str) -> str:
    if "<|assistant|>" in full_text:
        text = full_text.split("<|assistant|>", 1)[1]
    else:
        text = full_text

    stop_markers = [
        "<|eos|>",
        "<|bos|>",
        "<|user|>",
        "<|assistant|>",
    ]

    for marker in stop_markers:
        if marker in text:
            text = text.split(marker, 1)[0]

    text = text.strip()
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text


def has_repetition(text: str) -> bool:
    words = text.lower().split()

    if len(words) < 10:
        return False

    for size in [2, 3, 4]:
        chunks = [
            tuple(words[i : i + size])
            for i in range(0, len(words) - size + 1)
        ]

        if len(chunks) != len(set(chunks)):
            return True

    return False


def is_good_joke(text: str) -> bool:
    if len(text) < 20:
        return False

    if len(text) > 450:
        return False

    if "<|" in text or "|>" in text:
        return False

    if has_repetition(text):
        return False

    if len(set(text)) < 8:
        return False

    return True


def score_joke(text: str) -> float:
    score = 0.0
    length = len(text)

    if 50 <= length <= 220:
        score += 3.0

    if "?" in text:
        score += 2.0

    if "." in text or "!" in text:
        score += 1.0

    if "\n" in text:
        score += 0.5

    if has_repetition(text):
        score -= 5.0

    if length > 300:
        score -= 2.0

    return score


@torch.no_grad()
def generate_one(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(prompt)

    x = torch.tensor([ids], dtype=torch.long, device=device)

    y = model.generate(
        x,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )

    full_text = tokenizer.decode(y[0].tolist(), skip_special_tokens=False)
    answer = extract_answer(full_text)

    return answer


def generate_best(
    model,
    tokenizer,
    prompt: str,
    device: str,
    attempts: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    candidates = []

    for _ in range(attempts):
        joke = generate_one(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        if is_good_joke(joke):
            candidates.append(joke)

    if not candidates:
        return "I could not generate a good joke. Try again."

    candidates.sort(key=score_joke, reverse=True)

    return candidates[0]


def load_model(model_dir: Path, device: str):
    config_path = model_dir / "config.json"
    model_path = model_dir / "model.pt"
    tokenizer_path = model_dir / "tokenizer.json"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Missing tokenizer: {tokenizer_path}")

    config_data = json.loads(config_path.read_text(encoding="utf-8"))

    model_config = GPTConfig(**config_data["model"])

    tokenizer = JokeTokenizer.load(tokenizer_path)

    model = GPT(model_config)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    model.to(device)
    model.eval()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", default="models/joke_bot_bpe")
    parser.add_argument("--device", default=get_device())

    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)

    args = parser.parse_args()

    model_dir = Path(args.model_dir)

    model, tokenizer = load_model(model_dir, args.device)

    print("Jokes bot ready. Type 'exit' to quit.")

    while True:
        user_text = input("\nYou: ").strip()

        if user_text.lower() in ["exit", "quit", "q"]:
            break

        prompt = build_prompt(user_text)

        answer = generate_best(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=args.device,
            attempts=args.attempts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

        print(f"\nBot: {answer}")


if __name__ == "__main__":
    main()