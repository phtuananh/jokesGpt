from pathlib import Path

from tokenizers import AddedToken, Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class JokeTokenizer:
    SPECIAL_TOKENS = [
        "<|unk|>",
        "<|pad|>",
        "<|bos|>",
        "<|user|>",
        "<|assistant|>",
        "<|eos|>",
    ]

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @classmethod
    def train_from_file(
        cls,
        file_path: str | Path,
        vocab_size: int = 2000,
        min_frequency: int = 2,
        show_progress: bool = True,
    ):
        file_path = Path(file_path)

        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        special_tokens = [
            AddedToken(tok, single_word=False, lstrip=False, rstrip=False)
            for tok in cls.SPECIAL_TOKENS
        ]

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            initial_alphabet=ByteLevel.alphabet(),
            show_progress=show_progress,
        )

        tokenizer.train([str(file_path)], trainer)

        return cls(tokenizer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: str | Path):
        path = Path(path)
        self.tokenizer.save(str(path))

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        tokenizer = Tokenizer.from_file(str(path))
        return cls(tokenizer)