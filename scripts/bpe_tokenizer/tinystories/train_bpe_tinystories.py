from scripts.bpe_tokenizer.tokenizer_training_utils import load_bpe_tokenizer, train_bpe_tokenizer
from scripts.data.paths import TS_TOKENZIER_DIR, TS_TRAIN_FILE


def train_bpe_tinystories():
    train_bpe_tokenizer(
        input_path=TS_TRAIN_FILE,
        vocab_size=10_000,
        output_dir=TS_TOKENZIER_DIR,
    )


def load_tinystories_tokenizer():
    return load_bpe_tokenizer(TS_TOKENZIER_DIR)


if __name__ == "__main__":
    train_bpe_tinystories()
