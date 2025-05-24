# from tests.common import FIXTURES_PATH
# from tests.test_tokenizer import MERGES_PATH, VOCAB_PATH, _encode_iterable, get_tokenizer_from_vocab_merges_path


import time

from cs336_basics.bpe_tokenizer.bpe_tokenzer import BPETokenizer
from scripts.bpe_tokenizer.tinystories.train_bpe_tinystories import TS_MERGES_FILE, TS_VOCAB_FILE
from scripts.data.paths import TS_VALID_FILE


def main():
    # tokenizer = get_tokenizer_from_vocab_merges_path(
    #     vocab_path=VOCAB_PATH,
    #     merges_path=MERGES_PATH,
    #     special_tokens=["<|endoftext|>"],
    # )
    # with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
    #     ids = []
    #     for _id in _encode_iterable(tokenizer, f):
    #         ids.append(_id)

    # print(len(ids))

    tokenizer = BPETokenizer.from_files(
        vocab_filepath=TS_VOCAB_FILE,
        merges_filepath=TS_MERGES_FILE,
        special_tokens=["<|endoftext|>"],
    )

    with open(TS_VALID_FILE) as f:
        text = f.read()

    document_texts = text.split("<|endoftext|>")

    document_text = document_texts[0]

    # Measure encoding time
    start_encode = time.time()
    encoded = tokenizer.encode(document_text)
    encode_time = time.time() - start_encode
    print(f"Encoding time: {encode_time:.4f} seconds")

    # Measure decoding time
    start_decode = time.time()
    decoded = tokenizer.decode(encoded)
    decode_time = time.time() - start_decode
    print(f"Decoding time: {decode_time:.4f} seconds")

    # Print result
    print("\nDecoded text matches original:", decoded == text)


if __name__ == "__main__":
    main()
