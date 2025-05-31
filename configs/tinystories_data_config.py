from cs336_basics.training.train_config import DataConfig
from scripts.data.paths import TS_TOKENZIER_DIR

TINYSTORIES_DATA_CONFIG = DataConfig(
    train_path="output/token_ids/TinyStoriesV2-GPT4-train.txt.npy",
    valid_path="output/token_ids/TinyStoriesV2-GPT4-valid.txt.npy",
    tokenizer_path=TS_TOKENZIER_DIR,
    end_of_text_token="<|endoftext|>",
)
