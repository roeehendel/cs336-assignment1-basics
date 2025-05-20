import os

from cs336_basics.bpe_tokenizer.serialization import load_merges, load_vocab, save_merges, save_vocab
from cs336_basics.bpe_tokenizer.train_bpe import fast_bpe

OUTPUTS_DIR = "outputs"
TS_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "tiny_stories")
TS_VOCAB_FILE = os.path.join(TS_OUTPUT_DIR, "vocab.json")
TS_MERGES_FILE = os.path.join(TS_OUTPUT_DIR, "merges.json")


def train_bpe_tinystories():
    vocab, merges = fast_bpe(
        # input_path="data/TinyStoriesV2-GPT4-sample.txt",
        # input_path="data/TinyStoriesV2-GPT4-valid.txt",
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    os.makedirs(TS_OUTPUT_DIR, exist_ok=True)

    save_vocab(vocab, TS_VOCAB_FILE)
    save_merges(merges, TS_MERGES_FILE)

    loaded_vocab = load_vocab(TS_VOCAB_FILE)
    loaded_merges = load_merges(TS_MERGES_FILE)
    assert loaded_vocab == vocab
    assert loaded_merges == merges


if __name__ == "__main__":
    train_bpe_tinystories()
