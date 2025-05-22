# CS336 Assignment 1

## Problem (unicode1): Understanding Unicode
(a) In python, chr(0) returns the null character ('\x00')
(b) The printed representation shows noting while the __repr__ is '\x00'
(c) It doesn't appear visually when printed in text

## Problem (unicode2): Unicode Encodings
(a) In UTF-16 and UTF-32, many characters, especially common ones like english characters, include zero bytes in their representation. (TODO: explain why this is a problem - or why there are other problems)
(b) This function is incorrect because it assumes that each character is encoded into a single byte, which is not always correct. E.g. this function fails on the input "こ".
(c) b'\xff\xff'

## Problem (train_bpe_tinystories): BPE Training on TinyStories
(a) The run took 1 minute, ~1-2GB RAM. The longest token is `b' accomplishment'` with 15 bytes.
(b) The longest part is the pre-tokenization with ~30 seconds on 10 proccesses (iterating over the regex matches). However, the merges part is also ~30 seconds and non-parallizable.

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText
(a) The run took 1.5 hours, ~2-3GB RAM (estimated - need to verify). The longest token is `b'----------------------------------------------------------------'` with 64 bytes.
(b) The TinyStories tokenizer includes mostly common english words and names. The OpenWebText tokenizer is more diverse, and includes terms like "www", "Google", "YouTube", "gov", etc.

## Problem (tokenizer_experiments): Experiments with tokenizers
(a) The compression ratio of the TinyStories tokenizer is 4.1 bytes per token. The compression ratio of OpenWebText tokenizer is 4.4 bytes per token.
(b) The compression ration of OpenWebText using the TinyStories tokenizer is 3.3 bytes per token.
(c) The throughput of the TinyStories tokenizer is about 4M tokens per second. It would take about 50 hours to tokenize the Pile using this tokenizer.
(d) unit16 is an appropriate choice for storing token ids because they are integers between 0 and vocab_size - 1, and our largest vocab size is 32K which is less than the maximum value of a unit16, which is 65535.