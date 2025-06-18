import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.bpe_tokenzer import BPETokenizer
from cs336_basics.bpe_tokenizer.types import FilePath


def tokenize_text_file_and_save(
    input_file: str | Path,
    output_file: str | Path,
    tokenizer_dir: str | Path,
    special_tokens: list[str] = ["<|endoftext|>"],
    show_progress: bool = True,
) -> list[int]:
    """
    Tokenize a text file and save the results to a .npy file.

    Args:
        input_file: Path to the input text file
        output_file: Path to save the tokenized output (.npy file)
        tokenizer_dir: Path to tokenizer directory
        special_tokens: List of special tokens
        show_progress: Whether to show progress bars

    Returns:
        List of token IDs that were saved
    """
    token_ids = tokenize_file(
        input_file=input_file,
        tokenizer_dir=tokenizer_dir,
        special_tokens=special_tokens,
        show_progress=show_progress,
    )

    np.save(output_file, token_ids)
    return token_ids


def tokenize_file(
    input_file: FilePath,
    tokenizer_dir: FilePath,
    special_tokens: list[str] = ["<|endoftext|>"],
    show_progress: bool = True,
) -> list[int]:
    """
    Tokenize a text file.

    Args:
        input_file: Path to the input text file
        tokenizer_dir: Path to tokenizer directory
        special_tokens: List of special tokens
        show_progress: Whether to show progress bars

    Returns:
        List of token IDs
    """
    input_file = Path(input_file)
    tokenizer = _load_bpe_tokenizer(tokenizer_dir, special_tokens)

    with open(input_file, encoding="utf-8") as f:
        if show_progress:
            # Get total file size for better progress tracking
            f.seek(0, 2)  # Seek to end
            total_size = f.tell()
            f.seek(0)  # Seek back to beginning

            # Read in chunks and track progress by characters processed
            token_ids = []

            with tqdm(total=total_size, desc="Tokenizing", unit="chars", unit_scale=True) as pbar:
                for line in f:
                    line_tokens = list(tokenizer.encode_iterable([line]))
                    token_ids.extend(line_tokens)
                    pbar.update(len(line))

            return token_ids
        else:
            return list(tokenizer.encode_iterable(f))


def _load_bpe_tokenizer(tokenizer_dir: FilePath, special_tokens: list[str]) -> BPETokenizer:
    """Load BPE tokenizer from directory."""
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.json")
    return BPETokenizer.from_files(vocab_path, merges_path, special_tokens)
