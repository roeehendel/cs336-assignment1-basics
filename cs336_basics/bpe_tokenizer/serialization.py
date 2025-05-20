import json

import regex as re


def save_vocab(vocab: dict[int, bytes], path: str) -> None:
    str_to_id = {bytes_to_str(v): k for k, v in vocab.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(str_to_id, f, ensure_ascii=True, indent=2)


def load_vocab(path: str) -> dict[int, bytes]:
    with open(path, encoding="utf-8") as f:
        str_to_id = json.load(f)
    return {v: str_to_bytes(k) for k, v in str_to_id.items()}


def save_merges(merges: list[tuple[bytes, bytes]], path: str) -> None:
    str_merges = [(bytes_to_str(a), bytes_to_str(b)) for a, b in merges]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(str_merges, f)


def load_merges(path: str) -> list[tuple[bytes, bytes]]:
    with open(path, encoding="utf-8") as f:
        str_merges = json.load(f)
    return [(str_to_bytes(a), str_to_bytes(b)) for a, b in str_merges]


def bytes_to_str(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"\\u{byte:04x}" for byte in b)


_ESCAPE_REGEX = re.compile(r"\\u00([0-9a-fA-F]{2})")


def str_to_bytes(s: str) -> bytes:
    if _ESCAPE_REGEX.search(s):
        return _ESCAPE_REGEX.sub(lambda m: bytes([int(m.group(1), 16)]).decode("latin1"), s).encode("latin1")
    return s.encode("utf-8")
