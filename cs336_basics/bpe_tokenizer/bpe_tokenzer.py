from collections.abc import Iterable, Iterator

import regex as re

from cs336_basics.bpe_tokenizer.constants import GPT2_REGEX
from cs336_basics.bpe_tokenizer.serialization import load_merges, load_vocab
from cs336_basics.bpe_tokenizer.train_bpe import _apply_merge
from cs336_basics.bpe_tokenizer.types import FilePath, Merges, Vocab


class BPETokenizer:
    def __init__(
        self,
        vocab: Vocab,
        merges: Merges,
        special_tokens: list[str] | None = None,
        pre_tokenization_regex: str = GPT2_REGEX,
    ):
        if special_tokens is None:
            special_tokens = []

        for special_token in special_tokens:
            vocab[len(vocab)] = special_token.encode("utf-8")

        self._id_to_token = vocab
        self._token_to_id = {v: k for k, v in vocab.items()}
        self._merges = merges
        self._special_tokens = set(special_tokens)
        self._special_token_regex = re.compile("|".join(re.escape(special_token) for special_token in special_tokens))
        self._pre_tokenization_regex = re.compile(pre_tokenization_regex)

        full_regex = ""
        if special_tokens:
            full_regex = "|".join(re.escape(special_token) for special_token in special_tokens) + "|"
        full_regex += pre_tokenization_regex
        self._full_regex = re.compile(full_regex)

        self._max_special_token_length = (
            max(len(special_token) for special_token in special_tokens) if special_tokens else 0
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: FilePath,
        merges_filepath: FilePath,
        special_tokens: list[str] | None = None,
    ):
        vocab = load_vocab(vocab_filepath)
        merges = load_merges(merges_filepath)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        iterator = iter(iterable)
        buffer = next(iterator)

        while True:
            m = self._full_regex.search(buffer)
            if m is None:
                return
            elif m.end() >= len(buffer) - self._max_special_token_length - 1 and m.group() not in self._special_tokens:
                try:
                    buffer += next(iterator)
                except StopIteration:
                    yield from self._encode_pre_token(buffer)
                    return
            else:
                match_text = m.group()

                yield from self._encode_pre_token(match_text)

                buffer = buffer[m.end() :]

    def _encode_pre_token(self, pre_token: str) -> list[int]:
        if pre_token in self._special_tokens:
            return [self._token_to_id[pre_token.encode("utf-8")]]

        tokens = [bytes([b]) for b in pre_token.encode("utf-8")]
        for merge in self._merges:
            tokens = _apply_merge(tokens, merge)
        return [self._token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        token_bytes = [self._id_to_token[token_id] for token_id in ids]
        return b"".join(token_bytes).decode("utf-8", errors="replace")
