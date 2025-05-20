import multiprocessing
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import regex as re
from tqdm import tqdm

from cs336_basics.bpe_tokenizer.constants import GPT2_REGEX
from cs336_basics.bpe_tokenizer.pretokenization_example import find_chunk_boundaries
from cs336_basics.bpe_tokenizer.types import FilePath, Merges, Vocab


def fast_bpe(
    input_path: FilePath,
    vocab_size: int,
    special_tokens: list[str],
    pre_tokenization_regex: str = GPT2_REGEX,
    split_special_token: bytes = b"<|endoftext|>",
    num_processes: int | None = None,
) -> tuple[Vocab, Merges]:
    num_processes = num_processes or multiprocessing.cpu_count()

    with open(input_path, "rb") as f_for_boundaries:
        chunk_boundaries = find_chunk_boundaries(
            file=f_for_boundaries,
            desired_num_chunks=num_processes * 10,
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

    print("Pre-tokenizing")
    tic = time.time()
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(_process_chunk, tasks)))
    print(f"Pre-tokenizated in {time.time() - tic:.2f} seconds")

    token_seq_to_count = sum(results, Counter())

    print(f"num pre-tokens: {len(token_seq_to_count)}")

    token_seqs = [token_seq for token_seq in token_seq_to_count.keys()]
    token_seq_counts = [count for count in token_seq_to_count.values()]

    print("Building token pair counts")
    tic = time.time()
    token_pair_to_token_seq_idx = defaultdict(set)
    token_pair_counts = defaultdict(int)
    for i, (token_seq, count) in enumerate(zip(token_seqs, token_seq_counts)):
        for b1, b2 in zip(token_seq[:-1], token_seq[1:]):
            token_pair_counts[(b1, b2)] += count
            token_pair_to_token_seq_idx[(b1, b2)].add(i)
    print(f"Token pair counts built in {time.time() - tic:.2f} seconds")

    vocab = {i: bytes([i]) for i in range(256)}
    num_merges = vocab_size - len(vocab) - len(special_tokens)
    merges = [None] * num_merges
    for merge_idx in tqdm(range(num_merges), desc="Merging"):
        new_merge = max(token_pair_counts.keys(), key=lambda bp: (token_pair_counts[bp], bp))
        new_token = new_merge[0] + new_merge[1]

        merges[merge_idx] = new_merge
        vocab[len(vocab)] = new_token

        idx_to_update = token_pair_to_token_seq_idx[new_merge].copy()
        for idx in idx_to_update:
            for b1, b2 in zip(token_seqs[idx][:-1], token_seqs[idx][1:]):
                if (b1, b2) in token_pair_counts:
                    token_pair_counts[(b1, b2)] -= token_seq_counts[idx]
                if token_pair_counts[(b1, b2)] == 0:
                    del token_pair_counts[(b1, b2)]
                if idx in token_pair_to_token_seq_idx[(b1, b2)]:
                    token_pair_to_token_seq_idx[(b1, b2)].remove(idx)

            token_seqs[idx] = _apply_merge(token_seqs[idx], new_merge)

            for b1, b2 in zip(token_seqs[idx][:-1], token_seqs[idx][1:]):
                token_pair_counts[(b1, b2)] += token_seq_counts[idx]
                token_pair_to_token_seq_idx[(b1, b2)].add(idx)

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode("utf-8")

    return vocab, merges


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
    with open(chunk.input_path, "rb") as f:
        f.seek(chunk.start_offset)
        chunk_bytes = f.read(chunk.end_offset - chunk.start_offset)
    chunk_text = chunk_bytes.decode("utf-8", errors="strict")
    if not chunk_text:
        return Counter()
    return _get_token_seq_to_count(
        text=chunk_text,
        special_tokens=chunk.special_tokens,
        pre_tokenization_regex=chunk.pre_tokenization_regex,
    )


def _get_token_seq_to_count(
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


def _apply_merge(token_seq: tuple[bytes, ...], merge: tuple[bytes, bytes]) -> tuple[bytes, ...]:
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


def slow_bpe(
    input_path: FilePath,
    vocab_size: int,
    special_tokens: list[str],
    pre_tokenization_regex: str = GPT2_REGEX,
) -> tuple[Vocab, Merges]:
    with open(input_path) as f:
        text = f.read()

    token_seq_to_count = _get_token_seq_to_count(
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
            for b1, b2 in zip(token_seq[:-1], token_seq[1:]):
                token_pair_counts[(b1, b2)] += count

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
