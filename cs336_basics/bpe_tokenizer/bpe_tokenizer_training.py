import functools
import logging
import math
import multiprocessing
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import heapdict
import regex as re
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.constants import GPT2_REGEX
from cs336_basics.bpe_tokenizer.types import FilePath, Merges, Vocab
from cs336_basics.bpe_tokenizer.utils import find_chunk_boundaries
from cs336_basics.utils.logging_utils import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_CHUNK_SIZE = 2**23


def train_bpe_fast(
    input_path: FilePath,
    vocab_size: int,
    special_tokens: list[str],
    pre_tokenization_regex: str = GPT2_REGEX,
    split_special_token: bytes = b"<|endoftext|>",
    num_processes: int | None = None,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    verbose: bool = False,
    sample_file: float | None = None,
) -> tuple[Vocab, Merges]:
    old_logger_level = logger.getEffectiveLevel()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    logger.debug("Training BPE tokenizer")

    token_seq_to_count = _token_seq_to_count_from_file(
        input_path=input_path,
        special_tokens=special_tokens,
        pre_tokenization_regex=pre_tokenization_regex,
        split_special_token=split_special_token,
        num_processes=num_processes,
        max_chunk_size=max_chunk_size,
        verbose=verbose,
        sample_file=sample_file,
    )

    logger.debug(f"num pre-tokens: {len(token_seq_to_count)}")

    token_seqs = list(token_seq_to_count.keys())
    token_seq_counts = list(token_seq_to_count.values())

    logger.debug("Building token pair counts")
    tic = time.time()
    token_pair_to_token_seq_idx = defaultdict(set)

    token_pair_counts_raw = defaultdict(int)
    for i, (token_seq, count) in enumerate(zip(token_seqs, token_seq_counts)):
        for token_pair in zip(token_seq[:-1], token_seq[1:]):
            token_pair_counts_raw[token_pair] += count
            token_pair_to_token_seq_idx[token_pair].add(i)

    token_pair_counts = heapdict.heapdict()
    for token_pair, count in token_pair_counts_raw.items():
        token_pair_counts[token_pair] = _get_priority_for_heapdict(token_pair, count)

    logger.debug(f"Token pair counts built in {time.time() - tic:.2f} seconds. Unique pairs: {len(token_pair_counts)}")

    vocab = {i: bytes([i]) for i in range(256)}
    num_merges_to_perform = vocab_size - len(vocab) - len(special_tokens)
    merges = [None] * num_merges_to_perform

    for merge_idx in tqdm(range(num_merges_to_perform), desc="Merging", disable=not verbose):
        if not token_pair_counts:
            logger.warning("token_pair_counts (heapdict) is empty, stopping merges early.")
            break

        # get the next merge - the current most frequent token pair
        new_merge, _ = token_pair_counts.popitem()
        new_token = new_merge[0] + new_merge[1]
        merges[merge_idx] = new_merge
        vocab[len(vocab)] = new_token

        # update index
        idx_to_update = token_pair_to_token_seq_idx.pop(new_merge, set()).copy()
        for idx in idx_to_update:
            this_seq_occurrence_count = token_seq_counts[idx]

            # update index by removing token pairs before merge
            for old_pair in zip(token_seqs[idx][:-1], token_seqs[idx][1:]):
                if old_pair == new_merge:
                    continue

                if old_pair in token_pair_counts:
                    current_priority_value = token_pair_counts[old_pair]
                    current_count = -current_priority_value[0]
                    new_count = current_count - this_seq_occurrence_count

                    if new_count > 0:
                        token_pair_counts[old_pair] = _get_priority_for_heapdict(old_pair, new_count)
                    else:
                        del token_pair_counts[old_pair]

                if old_pair in token_pair_to_token_seq_idx:
                    token_pair_to_token_seq_idx[old_pair].discard(idx)
                    if not token_pair_to_token_seq_idx[old_pair]:
                        del token_pair_to_token_seq_idx[old_pair]

            # update token sequence - apply the merge
            token_seqs[idx] = _apply_merge(token_seqs[idx], new_merge)

            # update index by adding token pairs after merge
            for new_pair in zip(token_seqs[idx][:-1], token_seqs[idx][1:]):
                current_count = 0
                if new_pair in token_pair_counts:
                    current_priority_value = token_pair_counts[new_pair]
                    current_count = -current_priority_value[0]

                new_count = current_count + this_seq_occurrence_count
                token_pair_counts[new_pair] = _get_priority_for_heapdict(new_pair, new_count)
                token_pair_to_token_seq_idx[new_pair].add(idx)

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    logger.setLevel(old_logger_level)

    merges = [m for m in merges if m is not None]
    return vocab, merges


def _token_seq_to_count_from_file(
    input_path: FilePath,
    special_tokens: list[str],
    pre_tokenization_regex: str = GPT2_REGEX,
    split_special_token: bytes = b"<|endoftext|>",
    num_processes: int | None = None,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    verbose: bool = False,
    sample_file: float | None = None,
) -> Counter:
    file_size = os.path.getsize(input_path)
    if file_size == 0:
        return Counter()
    desired_num_chunks = int(math.ceil(file_size / max_chunk_size))

    with open(input_path, "rb") as f_for_boundaries:
        chunk_boundaries = find_chunk_boundaries(
            file=f_for_boundaries,
            desired_num_chunks=desired_num_chunks,
            split_special_token=split_special_token,
        )

    tasks = [
        Chunk(
            input_path=input_path,
            start_offset=start,
            end_offset=end,
            special_tokens=special_tokens,
            pre_tokenization_regex=pre_tokenization_regex,
        )
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]

    if sample_file:
        tasks = tasks[: int(len(tasks) * sample_file)]

    if not tasks:
        logger.debug("No processable chunks found after boundary calculations, returning empty counts.")
        return Counter()

    logger.debug(f"Created {len(tasks)} chunks for processing.")

    logger.debug("Pre-tokenizing")
    tic = time.time()
    token_seq_to_count = Counter()

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    if num_processes < 0:
        for task in tqdm(tasks, total=len(tasks), desc="Processing chunks sequentially", disable=not verbose):
            token_seq_to_count.update(_process_chunk(task))
    else:
        with multiprocessing.Pool(processes=num_processes) as pool:
            for chunk_result in tqdm(
                pool.imap_unordered(_process_chunk, tasks),
                total=len(tasks),
                desc="Processing chunks",
                disable=not verbose,
            ):
                token_seq_to_count.update(chunk_result)

    logger.debug(
        f"Pre-tokenization and aggregation completed in {time.time() - tic:.2f} seconds. Total unique tokens: {len(token_seq_to_count)}"
    )
    return token_seq_to_count


@dataclass
class Chunk:
    input_path: FilePath
    start_offset: int
    end_offset: int
    special_tokens: list[str]
    pre_tokenization_regex: str


def _process_chunk(chunk: Chunk) -> Counter:
    """
    Reads a specific chunk from the input file and processes it to get token sequence counts.
    """
    with open(chunk.input_path, encoding="utf-8") as f:
        f.seek(chunk.start_offset)
        chunk_text = f.read(chunk.end_offset - chunk.start_offset)
    if not chunk_text:
        return Counter()
    return _token_seq_to_count_from_text(
        text=chunk_text,
        special_tokens=chunk.special_tokens,
        pre_tokenization_regex=chunk.pre_tokenization_regex,
    )


def _token_seq_to_count_from_text(
    text: str,
    special_tokens: list[str],
    pre_tokenization_regex: str = GPT2_REGEX,
) -> Counter:
    special_tokens_pat = "|".join(re.escape(special_token) for special_token in special_tokens)
    text_split_by_special_tokens = re.split(special_tokens_pat, text)

    token_seq_to_count = Counter(
        match.group(0)
        for substr in text_split_by_special_tokens
        for match in re.finditer(pre_tokenization_regex, substr)
        if substr
    )

    token_seq_to_count = Counter(
        {tuple(bytes([b]) for b in k.encode("utf-8")): count for k, count in token_seq_to_count.items()}
    )

    return token_seq_to_count


@functools.total_ordering
class _InvertedBytes:
    """Helper class for heap tie-breaking for byte strings."""

    def __init__(self, b: bytes):
        self.b = b

    def __eq__(self, other):
        if not isinstance(other, _InvertedBytes):
            return NotImplemented
        return self.b == other.b

    def __lt__(self, other):
        if not isinstance(other, _InvertedBytes):
            return NotImplemented
        # Key: self is "less" if self.b is "greater", for min-heap to pick largest
        return self.b > other.b

    def __hash__(self):  # Important for objects used as dict keys or in sets if __eq__ is defined
        return hash(self.b)


def _get_priority_for_heapdict(pair: tuple[bytes, bytes], count: int) -> tuple:
    """Creates the priority tuple used as a value in heapdict."""
    # heapdict is a min-heap based on its values.
    # We want max count, then lexicographically largest pair.
    # _InvertedBytes handles the reverse lexicographical comparison correctly.
    return (-count, _InvertedBytes(pair[0]), _InvertedBytes(pair[1]))


def _apply_merge(token_seq: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    # TODO: can this be simplified and made faster?
    merge_token = merge[0] + merge[1]

    merged_token_seq = []
    i = 0
    while i < len(token_seq) - 1:
        if token_seq[i] == merge[0] and token_seq[i + 1] == merge[1]:
            merged_token_seq.append(merge_token)
            i += 2
        else:
            merged_token_seq.append(token_seq[i])
            i += 1
    if i == len(token_seq) - 1:
        merged_token_seq.append(token_seq[i])
    return tuple(merged_token_seq)


def train_bpe_simple(
    input_path: FilePath,
    vocab_size: int,
    special_tokens: list[str],
    pre_tokenization_regex: str = GPT2_REGEX,
) -> tuple[Vocab, Merges]:
    with open(input_path) as f:
        text = f.read()

    token_seq_to_count = _token_seq_to_count_from_text(
        text=text,
        special_tokens=special_tokens,
        pre_tokenization_regex=pre_tokenization_regex,
    )

    vocab = {i: bytes([i]) for i in range(256)}
    merges = []

    num_merges = vocab_size - len(vocab) - len(special_tokens)
    for _ in range(num_merges):
        token_pair_counts = defaultdict(int)
        for token_seq, count in token_seq_to_count.items():
            for token_pair in zip(token_seq[:-1], token_seq[1:]):
                token_pair_counts[token_pair] += count

        new_merge = max(token_pair_counts.keys(), key=lambda bp: (token_pair_counts[bp], bp))
        new_token = new_merge[0] + new_merge[1]

        merges.append(new_merge)
        vocab[len(vocab)] = new_token

        token_seq_to_count = {
            _apply_merge(token_seq, new_merge): counts for token_seq, counts in token_seq_to_count.items()
        }

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges
