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
