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

## Problem (transformer_accounting): Transformer LM resource accounting
(a) The GPT-2-XL model, with our architecture and the given configuration, has about 2B trainable parameters. Assuming each parameter is represented by a single-precision floating point (4 bytes) we need 8GB of memory to load the model.

(b) Following is a list of TFLOPs per operation (and percent of total):

**gpt-2-xl**
- total_flops=4.51 (100.00%)

- mha_flops=1.33 (29.44%)
- qkv_proj_flops=0.75 (16.73%)
- rope_flops=0.00 (0.00%)
- attn_flops=0.32 (7.14%)
- output_proj_flops=0.25 (5.58%)

- glu_flops=3.02 (66.91%)

- lm_head_flops=0.16 (3.65%)

(c) The linear layers in the MLP (GLU) require the most FLOPS - about 2/3, and the MHA comes second with about 1/3 - with the kqvo projections dominating within it, followed by the attention calculation.

(d) In the smaller models, the lm_head uses a much larger portion of the total flops (starting with ~22% in the smallest model and going down to ~6% in the largest). The main component that grows it's relative FLOP use is the MLP (GLU). 

**gpt-2-small**
- total_flops=0.35 (100.00%)

- mha_flops=0.10 (27.64%)
- qkv_proj_flops=0.04 (12.44%)
- rope_flops=0.00 (0.00%)
- attn_flops=0.04 (11.06%)
- output_proj_flops=0.01 (4.15%)

- glu_flops=0.17 (49.75%)

- lm_head_flops=0.08 (22.61%)

**gpt-2-medium**
- total_flops=1.03 (100.00%)

- mha_flops=0.31 (29.93%)
- qkv_proj_flops=0.15 (14.97%)
- rope_flops=0.00 (0.00%)
- attn_flops=0.10 (9.98%)
- output_proj_flops=0.05 (4.99%)

- glu_flops=0.62 (59.87%)

- lm_head_flops=0.11 (10.20%)

**gpt-2-large**
- total_flops=2.26 (100.00%)

- mha_flops=0.68 (29.96%)
- qkv_proj_flops=0.36 (16.05%)
- rope_flops=0.00 (0.00%)
- attn_flops=0.19 (8.56%)
- output_proj_flops=0.12 (5.35%)

- glu_flops=1.45 (64.20%)

- lm_head_flops=0.13 (5.84%)

(e) When increasing the context length to 16,384, the total FLOPs grow dramatically - about 33X up to ~150 TFLOPS! The multi-head attention now dominates, taking about 2/3 of the total FLOPs, compared to about 1/3 before - within it now the attention calculation dominates, taking ~55% the total FLOPs of the forward pass, compared to only about 7% before. This is due to the quadratic dependence of the attention calculation on the context length.

**gpt-2-xl with large context length**
- total_flops=149.52 (100.00%)

- mha_flops=98.57 (65.92%)
- qkv_proj_flops=12.08 (8.08%)
- rope_flops=0.00 (0.00%)
- attn_flops=82.46 (55.15%)
- output_proj_flops=4.03 (2.69%)

- glu_flops=48.32 (32.31%)

- lm_head_flops=2.63 (1.76%)

