# BPE Tokenizer

A tokenizer implementation based on the [BPE](https://en.wikipedia.org/wiki/Byte-pair_encoding "The wikipedia page for BPE") (Byte Pair Encoding) algorithm.

> [!WARNING]
> This is a practice project intended to help understand how BPE algorithms and tokenizers work. It is not a production-ready implementation. Some performance optimizations and edge case handling may not be fully polished.

It supports following features:

- **BPE Training**: Train vocabulary and merge rules from text
- **Encode/Decode**: Convert between text and token sequences
- **Special Tokens**: Support custom special tokens
- **Pre-tokenization**: Use regex pattern pre-tokenization
- **Model Persistence**: Save and load tokenizer models (using rkyv serialization)

## Quick Example

```rust
use tokenizer::Tokenizer;

fn main() -> anyhow::Result<()> {
    // Create tokenizer with vocabulary size 100000, 2 special tokens
    let mut model = Tokenizer::new(100000, None, &["<|beginoftext|>", "<|endoftext|>"])?;

    // Train
    model.train("path/to/training_text.txt", false)?;

    // Encode
    let tokens = model.encode("Hello, World")?;
    println!("{:?}", tokens);

    // Decode
    let text = model.decode(&tokens)?;
    println!("{}", text);

    // Save model
    model.save("models/my_tokenizer.bin")?;

    // Load model
    let loaded = Tokenizer::load("models/my_tokenizer.bin")?;

    Ok(())
}
```

## Build & Run

The project depends on:
- fancy-regex: Regex support
- rkyv: Serialization/deserialization
- anyhow: Error handling

```bash
# Build
cargo build

# Run example
cargo run
```
