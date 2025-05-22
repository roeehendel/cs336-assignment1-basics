import os

from cs336_basics.bpe_tokenizer.bpe_tokenizer_training import train_bpe_fast
from cs336_basics.bpe_tokenizer.bpe_tokenzer import BPETokenizer
from cs336_basics.bpe_tokenizer.serialization import load_merges, load_vocab, save_merges, save_vocab

DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]


def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    output_dir: str,
    special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS,
):
    vocab, merges = train_bpe_fast(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
    )

    os.makedirs(output_dir, exist_ok=True)

    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.json")

    save_vocab(vocab, vocab_path)
    save_merges(merges, merges_path)

    loaded_vocab = load_vocab(vocab_path)
    loaded_merges = load_merges(merges_path)
    assert loaded_vocab == vocab
    assert loaded_merges == merges


def load_bpe_tokenizer(
    tokenizer_dir: str,
    special_tokens: list[str] = DEFAULT_SPECIAL_TOKENS,
):
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.json")
    return BPETokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )
