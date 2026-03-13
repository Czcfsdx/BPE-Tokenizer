use std::process::exit;

mod tokenizer;

fn main() {
    const MAX_VOCABULARY_SIZE: usize = 100000;
    const VERBOSE: bool = true;
    const TRAIN_CROPS_PATH: &str = "datasets/bpe-wiki.txt";

    let mut model = tokenizer::Tokenizer::new();
    match model.train(MAX_VOCABULARY_SIZE, TRAIN_CROPS_PATH, VERBOSE) {
        Ok(()) => {}
        Err(error) => {
            println!("{}", error);
            exit(69);
        }
    }
}
