from tests.common import FIXTURES_PATH
from tests.test_tokenizer import MERGES_PATH, VOCAB_PATH, _encode_iterable, get_tokenizer_from_vocab_merges_path


def main():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>"],
    )
    with open(FIXTURES_PATH / "tinystories_sample_5M.txt") as f:
        ids = []
        for _id in _encode_iterable(tokenizer, f):
            ids.append(_id)

    print(len(ids))


if __name__ == "__main__":
    main()
