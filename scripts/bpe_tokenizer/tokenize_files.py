import os

import numpy as np

from cs336_basics.bpe_tokenizer.tokenize_file import tokenize_text_file_and_save
from scripts.bpe_tokenizer.owt.train_bpe_expts_owt import load_openwebtext_tokenizer
from scripts.bpe_tokenizer.tinystories.train_bpe_tinystories import load_tinystories_tokenizer
from scripts.data.paths import OUTPUT_DIR, OWT_TRAIN_FILE, OWT_VALID_FILE, TS_TRAIN_FILE, TS_VALID_FILE

TOKENIZED_DIR = os.path.join(OUTPUT_DIR, "token_ids")


def main():
    configs = [
        (load_tinystories_tokenizer(), [TS_VALID_FILE, TS_TRAIN_FILE]),
        (load_openwebtext_tokenizer(), [OWT_VALID_FILE, OWT_TRAIN_FILE]),
    ]

    os.makedirs(TOKENIZED_DIR, exist_ok=True)

    for tokenizer, input_files in configs:
        for input_file in input_files:
            output_path = os.path.join(TOKENIZED_DIR, f"{os.path.basename(input_file)}.npy")

            print(f"Tokenizing {input_file}")
            token_ids = tokenize_text_file_and_save(
                tokenizer=tokenizer,
                input_file=input_file,
                output_path=output_path,
                show_progress=True,
            )
            loaded_token_ids = np.load(output_path)
            assert np.all(token_ids == loaded_token_ids)


if __name__ == "__main__":
    main()
