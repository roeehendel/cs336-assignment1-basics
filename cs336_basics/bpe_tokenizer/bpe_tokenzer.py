from collections.abc import Iterable, Iterator
from functools import lru_cache

import regex as re

from cs336_basics.bpe_tokenizer.bpe_tokenizer_training import _apply_merge
from cs336_basics.bpe_tokenizer.constants import GPT2_REGEX
from cs336_basics.bpe_tokenizer.serialization import load_merges, load_vocab
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
            if special_token.encode("utf-8") not in vocab.values():
                vocab[len(vocab)] = special_token.encode("utf-8")

        self._id_to_token = vocab
        self._token_to_id = {v: k for k, v in vocab.items()}
        self._merges = merges
        self._special_tokens = set(special_tokens)
        if special_tokens:
            escaped_tokens_pattern = "|".join(
                re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)
            )
            self._special_token_regex = re.compile(f"({escaped_tokens_pattern})")
        else:
            self._special_token_regex = re.compile(r"(?!)")
        self._pre_tokenization_regex = re.compile(pre_tokenization_regex)

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

        for _ in range(100):
            split_by_special_token = self._special_token_regex.split(buffer)

            for substr in split_by_special_token[:-1]:
                if substr in self._special_tokens:
                    yield self._token_to_id[substr.encode("utf-8")]
                else:
                    yield from self._encode_clean_chunk(substr)

            buffer = split_by_special_token[-1]
            chunk = next(iterator, None)
            if chunk is None:
                yield from self._encode_clean_chunk(buffer)
                return
            buffer += chunk

    def _encode_clean_chunk(self, chunk: str) -> Iterator[int]:
        for m in self._pre_tokenization_regex.finditer(chunk):
            yield from self._encode_pre_token(m.group())

    @lru_cache(maxsize=10_000)
    def _encode_pre_token(self, pre_token: str) -> list[int]:
        tokens = [bytes([b]) for b in pre_token.encode("utf-8")]
        token_pairs_set = {(t1, t2) for t1, t2 in zip(tokens, tokens[1:])}
        for merge in self._merges:
            if merge in token_pairs_set:
                tokens = _apply_merge(tokens, merge)
                token_pairs_set = {(t1, t2) for t1, t2 in zip(tokens, tokens[1:])}
        return [self._token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        token_bytes = [self._id_to_token[token_id] for token_id in ids]
        return b"".join(token_bytes).decode("utf-8", errors="replace")
