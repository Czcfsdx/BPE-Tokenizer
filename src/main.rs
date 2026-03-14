mod tokenizer;

fn main() {
    // Train the BPE tokenizer
    const MAX_VOCABULARY_SIZE: usize = 100000;
    const VERBOSE: bool = false;
    const TRAIN_CROPS_PATH: &str = "datasets/bpe-wiki.txt";

    let mut model = tokenizer::Tokenizer::new();
    model
        .train(MAX_VOCABULARY_SIZE, TRAIN_CROPS_PATH, VERBOSE)
        .unwrap_or_else(|e| eprintln_error(e));
    println!("{}", model.vocabulary_to_text());

    // Test the save and load
    const MODEL_PATH: &str = "models/test.bin";
    model.save(MODEL_PATH).unwrap_or_else(|e| eprintln_error(e));
    let new_model = tokenizer::Tokenizer::load(MODEL_PATH).unwrap_or_else(|e| eprintln_error(e));
    assert_eq!(model, new_model);

    // Test the encode and the decode
    const TEXT: &str = "Byte-pair encoding (BPE) is a text compression algorithm from 1994 that iteratively replaces frequent byte pairs with placeholder symbols. Modern large language models use a modified version that converts text into \"tokens\" (natural numbers) by merging frequent character sequences, creating a fixed-size vocabulary. Unlike the original compression-focused approach, this tokenization method ensures any UTF-8 text can be encoded, handling unknown characters through byte-level processing or special tokens.";
    let tokens = model.encode(TEXT, VERBOSE);
    println!("Encode result: {:?}\n", tokens);
    let decoded_text = model.decode(&tokens).unwrap_or_else(|e| eprintln_error(e));
    println!("Decode result: {}\n", decoded_text);
    assert_eq!(TEXT, decoded_text);
}

fn eprintln_error(error: anyhow::Error) -> ! {
    eprintln!("{}", error);
    std::process::exit(69);
}
