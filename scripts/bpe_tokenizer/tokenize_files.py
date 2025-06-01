import os

import numpy as np

from cs336_basics.bpe_tokenizer.tokenize_file import tokenize_text_file_and_save
from scripts.data.paths import (
    OUTPUT_DIR,
    OWT_TOKENIZER_DIR,  # noqa: F401
    OWT_TRAIN_FILE,  # noqa: F401
    OWT_VALID_FILE,  # noqa: F401
    TS_TOKENZIER_DIR,  # noqa: F401
    TS_TRAIN_FILE,  # noqa: F401
    TS_VALID_FILE,  # noqa: F401
)

TOKENIZED_DIR = os.path.join(OUTPUT_DIR, "token_ids")


def main():
    configs = [
        # {"tokenizer_dir": TS_TOKENZIER_DIR, "input_files": [TS_VALID_FILE, TS_TRAIN_FILE]},
        {"tokenizer_dir": OWT_TOKENIZER_DIR, "input_files": [OWT_VALID_FILE, OWT_TRAIN_FILE]},
    ]

    os.makedirs(TOKENIZED_DIR, exist_ok=True)

    for config in configs:
        tokenizer_dir = config["tokenizer_dir"]
        input_files = config["input_files"]

        for input_file in input_files:
            output_path = os.path.join(TOKENIZED_DIR, f"{os.path.basename(input_file)}.npy")

            print(f"Tokenizing {input_file}")
            token_ids = tokenize_text_file_and_save(
                tokenizer_dir=tokenizer_dir,
                input_file=input_file,
                output_file=output_path,
                num_workers=4,
                show_progress=True,
            )
            loaded_token_ids = np.load(output_path)
            assert np.all(token_ids == loaded_token_ids)


if __name__ == "__main__":
    main()
