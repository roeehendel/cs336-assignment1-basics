from cs336_basics.bpe_tokenizer.bpe_tokenizer_training import train_bpe_fast


def main():
    vocab, merges = train_bpe_fast(
        # input_path="data/TinyStoriesV2-GPT4-sample.txt",
        # input_path="data/TinyStoriesV2-GPT4-valid.txt",
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        verbose=True,
        # num_processes=-1,
    )


if __name__ == "__main__":
    main()
