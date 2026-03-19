mod tests;
mod tokenizer;

fn main() {
    // Train the BPE tokenizer
    const MAX_VOCABULARY_SIZE: usize = 100000;
    const TRAIN_CROPS_PATH: &str = "src/tokenizer.rs";
    const SPECIAL_TOKENS: [&str; 3] = ["<|beginoftext|>", "<|middleoftext|>", "<|endoftext|>"];

    let mut model = tokenizer::Tokenizer::new(MAX_VOCABULARY_SIZE, None, &SPECIAL_TOKENS)
        .unwrap_or_else(|e| eprintln_error(e));
    model
        .train(TRAIN_CROPS_PATH, false)
        .unwrap_or_else(|e| eprintln_error(e));
    println!("{}", model.vocabulary_to_text());


    const TEXT: &str = "Hello, World";
    let tokens = model.encode(TEXT).unwrap_or_else(|e| eprintln_error(e));
    println!("Encode result: {:?}\n", tokens);
    let decoded_text = model.decode(&tokens).unwrap_or_else(|e| eprintln_error(e));
    println!("Decode result: {}\n", decoded_text);
}

fn eprintln_error(error: anyhow::Error) -> ! {
    eprintln!("{:?}", error);
    std::process::exit(69);
}
