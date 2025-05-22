from scripts.bpe_tokenizer.tokenizer_training_utils import load_bpe_tokenizer, train_bpe_tokenizer
from scripts.data.paths import OWT_TOKENIZER_DIR, OWT_TRAIN_FILE


def train_bpe_owt():
    train_bpe_tokenizer(
        input_path=OWT_TRAIN_FILE,
        vocab_size=32_000,
        output_dir=OWT_TOKENIZER_DIR,
    )


def load_openwebtext_tokenizer():
    return load_bpe_tokenizer(OWT_TOKENIZER_DIR)


if __name__ == "__main__":
    train_bpe_owt()
