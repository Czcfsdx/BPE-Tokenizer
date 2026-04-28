mod tests;
mod tokenizer;

fn main() {
    // Train the BPE tokenizer
    const CONFIG_PATH: &str = "examples/example.conf";
    let mut model = tokenizer::Tokenizer::new(CONFIG_PATH).unwrap_or_else(|e| eprintln_error(e));

    model
        .train(false)
        .unwrap_or_else(|e| eprintln_error(e));

    model.save("models/example.bin").unwrap_or_else(|e| eprintln_error(e));
    let loaded_model = tokenizer::Tokenizer::load("models/example.bin").unwrap_or_else(|e| eprintln_error(e));
    assert_eq!(model, loaded_model);

    const TEXT: &str = "<|beginoftext|>Hello,<|middleoftext|> World!<|endoftext|>";
    let tokens = model.encode(TEXT).unwrap_or_else(|e| eprintln_error(e));
    println!("Encode result: {:?}", tokens);
    let decoded_text = model.decode(&tokens).unwrap_or_else(|e| eprintln_error(e));
    println!("Decode result: {}", decoded_text);
}

fn eprintln_error(error: anyhow::Error) -> ! {
    eprintln!("{:?}", error);
    std::process::exit(69);
}
