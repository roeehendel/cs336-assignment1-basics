import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.bpe_tokenzer import BPETokenizer
from cs336_basics.bpe_tokenizer.types import FilePath
from cs336_basics.bpe_tokenizer.utils import find_chunk_boundaries


def tokenize_text_file_and_save(
    input_file: str | Path,
    output_file: str | Path,
    tokenizer_dir: str | Path,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_workers: int | None = None,
    split_token: str = "<|endoftext|>",
    show_progress: bool = True,
) -> list[int]:
    """
    Tokenize a text file and save the results to a .npy file.

    Args:
        input_file: Path to the input text file
        output_file: Path to save the tokenized output (.npy file)
        tokenizer_dir: Path to tokenizer directory
        special_tokens: List of special tokens
        num_workers: Number of worker processes. None (default) means single-threaded
        split_token: Token to split chunks on for multiprocessing
        show_progress: Whether to show progress bars

    Returns:
        List of token IDs that were saved
    """
    token_ids = tokenize_file(
        input_file=input_file,
        tokenizer_dir=tokenizer_dir,
        special_tokens=special_tokens,
        num_workers=num_workers,
        split_token=split_token,
        show_progress=show_progress,
    )

    np.save(output_file, token_ids)
    return token_ids


def tokenize_file(
    input_file: FilePath,
    tokenizer_dir: FilePath,
    special_tokens: list[str] = ["<|endoftext|>"],
    num_workers: int | None = None,
    split_token: str = "<|endoftext|>",
    show_progress: bool = True,
) -> list[int]:
    """
    Tokenize a text file, optionally using multiprocessing for speed.

    Args:
        input_file: Path to the input text file
        tokenizer_dir: Path to tokenizer directory
        special_tokens: List of special tokens
        num_workers: Number of worker processes. None (default) means single-threaded
        split_token: Token to split chunks on for multiprocessing
        show_progress: Whether to show progress bars

    Returns:
        List of token IDs
    """
    input_file = Path(input_file)

    if num_workers is None:
        return _tokenize_single_threaded(
            input_file=input_file,
            tokenizer_dir=tokenizer_dir,
            show_progress=show_progress,
            special_tokens=special_tokens,
        )
    else:
        return _tokenize_multi_threaded(
            input_file=input_file,
            tokenizer_dir=tokenizer_dir,
            special_tokens=special_tokens,
            num_workers=num_workers,
            split_token=split_token,
            show_progress=show_progress,
        )


def _tokenize_single_threaded(
    input_file: Path,
    tokenizer_dir: FilePath,
    show_progress: bool,
    special_tokens: list[str],
) -> list[int]:
    """Tokenize file using single thread."""
    tokenizer = _load_bpe_tokenizer(tokenizer_dir, special_tokens)

    with open(input_file, encoding="utf-8") as f:
        if show_progress:
            # Get total file size for better progress tracking
            f.seek(0, 2)  # Seek to end
            total_size = f.tell()
            f.seek(0)  # Seek back to beginning

            # Read in chunks and track progress by characters processed
            token_ids = []
            chars_processed = 0

            with tqdm(total=total_size, desc="Tokenizing", unit="chars", unit_scale=True) as pbar:
                for line in f:
                    line_tokens = list(tokenizer.encode_iterable([line]))
                    token_ids.extend(line_tokens)

                    chars_processed += len(line)
                    pbar.update(len(line))

            return token_ids
        else:
            return list(tokenizer.encode_iterable(f))


DEFAULT_MAX_CHUNK_SIZE = 1024 * 1024 * 10  # 10MB


def _tokenize_multi_threaded(
    input_file: Path,
    tokenizer_dir: str,
    special_tokens: list[str],
    num_workers: int,
    split_token: str,
    show_progress: bool,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> list[int]:
    """Tokenize file using multiple worker processes."""
    # Ensure split token is in special tokens
    if split_token not in special_tokens:
        special_tokens = [*special_tokens, split_token]

    split_token_bytes = split_token.encode("utf-8")

    file_size = os.path.getsize(input_file)
    desired_num_chunks = int(math.ceil(file_size / max_chunk_size))

    # Find chunk boundaries
    with open(input_file, "rb") as f:
        boundaries = find_chunk_boundaries(
            file=f,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_token_bytes,
        )

    # Read chunks and calculate sizes for better progress tracking
    chunks: list[str] = []
    chunk_sizes: list[int] = []
    total_size = 0

    with open(input_file, "rb") as f:
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk_bytes = f.read(end - start)
            chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
            chunks.append(chunk_text)

            chunk_size = len(chunk_text)
            chunk_sizes.append(chunk_size)
            total_size += chunk_size

    # Prepare worker arguments
    worker_args = [(chunk, tokenizer_dir, special_tokens) for chunk in chunks]

    # Process chunks in parallel with better progress tracking
    with mp.Pool(num_workers) as pool:
        if show_progress:
            # Create progress bar based on total characters to process
            with tqdm(total=total_size, desc="Tokenizing", unit="chars", unit_scale=True) as pbar:
                chunk_results = []
                for i, result in enumerate(pool.imap(_tokenize_chunk, worker_args)):
                    chunk_results.append(result)
                    # Update progress by the size of the chunk we just completed
                    pbar.update(chunk_sizes[i])
                    pbar.set_postfix({"chunks_done": f"{i + 1}/{len(chunks)}"})
        else:
            chunk_results = pool.map(_tokenize_chunk, worker_args)

    # Combine results
    token_ids: list[int] = []
    for chunk_tokens in chunk_results:
        token_ids.extend(chunk_tokens)

    return token_ids


def _load_bpe_tokenizer(tokenizer_dir: str, special_tokens: list[str]) -> BPETokenizer:
    """Load BPE tokenizer from directory."""
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.json")
    return BPETokenizer.from_files(vocab_path, merges_path, special_tokens)


def _tokenize_chunk(args: tuple[str, str, list[str]]) -> list[int]:
    """Worker function to tokenize a single chunk of text."""
    chunk_text, tokenizer_dir, special_tokens = args

    # Create tokenizer in worker process (avoids pickling compiled regex)
    tokenizer = _load_bpe_tokenizer(tokenizer_dir, special_tokens)
    return tokenizer.encode(chunk_text)
