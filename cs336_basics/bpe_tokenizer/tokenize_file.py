from collections.abc import Iterable

import numpy as np
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.bpe_tokenzer import BPETokenizer


def stream_token_ids_from_text_file(
    tokenizer: BPETokenizer, input_file: str, show_progress: bool = True
) -> Iterable[int]:
    with open(input_file) as f:
        line_count = sum(1 for _ in f)

    with open(input_file) as f:
        with tqdm(f, total=line_count, desc="Tokenizing", disable=not show_progress) as pbar_file:
            yield from tokenizer.encode_iterable(pbar_file)


def tokenize_text_file_and_save(
    tokenizer: BPETokenizer, input_file: str, output_path: str, show_progress: bool = True
) -> list[int]:
    token_ids = list(stream_token_ids_from_text_file(tokenizer, input_file, show_progress))
    # TODO: try to save in a streaming way
    np.save(output_path, token_ids)
    return token_ids
