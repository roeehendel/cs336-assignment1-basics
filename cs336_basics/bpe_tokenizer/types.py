import os

Vocab = dict[int, bytes]
Merges = list[tuple[bytes, bytes]]

FilePath = str | os.PathLike
