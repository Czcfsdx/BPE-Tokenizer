#[cfg(test)]
mod tests {
    use crate::tokenizer::Tokenizer;

    fn create_test_tokenizer() -> Tokenizer {
        const MAX_VOCABULARY_SIZE: usize = 100000;
        const TRAIN_CROPS_PATH: &str = "tests/bpe-wiki.txt";
        const SPECIAL_TOKENS: [&str; 3] = ["<|beginoftext|>", "<|middleoftext|>", "<|endoftext|>"];
        let mut model = Tokenizer::new(MAX_VOCABULARY_SIZE, None, &SPECIAL_TOKENS)
            .expect("Failed to create tokenizer");
        model
            .train(TRAIN_CROPS_PATH, false)
            .expect("Failed to train tokenizer");
        model
    }

    #[test]
    fn test_save_and_load() {
        let model = create_test_tokenizer();
        const MODEL_PATH: &str = "tests/test.bin";
        model.save(MODEL_PATH).expect("Failed to save model");
        let new_model = Tokenizer::load(MODEL_PATH).expect("Failed to load model");
        assert_eq!(model, new_model);
    }

    #[test]
    fn test_encode_and_decode() {
        let model = create_test_tokenizer();
        const TEXT: &str = "<|beginoftext|>+ Byte-pair encoding (BPE) is a text compression algorithm from 1994 that iteratively replaces frequent byte pairs with placeholder symbols. 这是一些混入的中文。 <|middleoftext|><|middleoftext|>Modern large language models use a modified version that converts text into \"tokens\" (natural numbers) by merging frequent character sequences, [😄😡😭 <>/?{}!@#$%^&*-=_+\\|;:`~] creating a fixed-size vocabulary. -<|endoftext|>";
        let tokens = model.encode(TEXT).expect("Failed to encode");
        let decoded_text = model.decode(&tokens).expect("Failed to decode");
        assert_eq!(TEXT, decoded_text);
    }
}
