# BPE Tokenizer

A tokenizer implementation based on the [BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding "The wikipedia page for BPE") (Byte Pair Encoding) algorithm.

> [!WARNING]
> This is a practice project intended to help understand how BPE algorithms and tokenizers work. It is not a production-ready implementation. Some performance optimizations and edge case handling may not be fully polished.

## Example

```bash
# Train model based on the configuration file
./BPE train examples/example.conf

# Encode text into tokens
./BPE encode --model models/example.bin --text "Hello, World!"

# Show the vocabulary in the tokenizer pmodel
./BPE show models/example.bin
```

## Build

This project depends on:
- anyhow: Error handling
- clap: CLI argument parser
- fancy-regex: Regex support

```bash
# Build
cargo build
```
