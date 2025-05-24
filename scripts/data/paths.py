import os

DATA_DIR = "data"


TS_TRAIN_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-train.txt")
TS_VALID_FILE = os.path.join(DATA_DIR, "TinyStoriesV2-GPT4-valid.txt")


OWT_TRAIN_FILE = os.path.join(DATA_DIR, "owt_train.txt")
OWT_VALID_FILE = os.path.join(DATA_DIR, "owt_valid.txt")


OUTPUT_DIR = "output"

BPE_TOKENIZER_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "bpe_tokenizer")

TS_TOKENZIER_DIR = os.path.join(BPE_TOKENIZER_OUTPUT_DIR, "tiny_stories")
OWT_TOKENIZER_DIR = os.path.join(BPE_TOKENIZER_OUTPUT_DIR, "owt")
