use std::collections::HashMap;
use std::fmt;
use std::fs::read_to_string;
use std::io::Error as IoError;

#[derive(Debug)]
pub enum TokenizerError {
    Io(IoError),
    Param(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::Io(err) => write!(f, "IO error: {}", err),
            TokenizerError::Param(msg) => write!(f, "Parameter error: {}", msg),
        }
    }
}

impl From<IoError> for TokenizerError {
    fn from(err: IoError) -> TokenizerError {
        TokenizerError::Io(err)
    }
}

type TokenID = usize;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
struct Token(TokenID, TokenID);

pub struct Tokenizer {
    // regexp '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
    vocabulary: Vec<Token>,
}

impl Tokenizer {
    // Init the vocabulary to:
    // 0 => { left: 0, right: 0}
    // 1 => { left: 1, right: 0}
    // ...
    // 255 => { left: 255, right: 0}
    pub fn new() -> Self {
        let mut vocabulary = vec![];
        for i in u8::MIN..=u8::MAX {
            vocabulary.push(Token(i as usize, 0));
        }
        Self { vocabulary }
    }

    pub fn train(
        &mut self,
        max_vocabulary_size: usize,
        path: &str,
        verbose: bool,
    ) -> Result<(), TokenizerError> {
        if max_vocabulary_size < u8::MAX as usize {
            return Err(TokenizerError::Param(format!(
                "max_vocabulary_size must be greater than {}, now is {}",
                u8::MAX as usize,
                max_vocabulary_size
            )));
        }

        let mut token_id_seq: Vec<TokenID> = read_to_string(path)?
            .bytes()
            .map(|b| b as TokenID)
            .collect();

        while self.vocabulary.len() <= max_vocabulary_size {
            if let Some((token, times)) = Self::find_most_frequent_token_pair(&token_id_seq) {
                let new_token_id = self.vocabulary.len();
                token_id_seq = Self::replace_token_to_token_id(token_id_seq, token, new_token_id);
                self.vocabulary.push(token);
                if verbose {
                    let token_bytes =
                        Self::convert_token_id_to_bytes(&self.vocabulary, new_token_id);
                    match String::from_utf8(token_bytes) {
                        Ok(token_string) => println!(
                            "New token {:>3} ({:>2} times) => ({:>3}, {:>3}): {:<}",
                            new_token_id,
                            times,
                            token.0,
                            token.1,
                            format!("({:?})", token_string)
                        ),
                        Err(error) => println!(
                            "New token {:>3} ({:>2} times) => ({:>3}, {:>3}) But error occurs when converting it to String: {}",
                            new_token_id,
                            times,
                            token.0,
                            token.1,
                            error
                        ),
                    }
                }
            } else {
                if verbose {
                    println!("New token not found");
                }
                break;
            }
        }
        Ok(())
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tokenizer {
    // Find the token pair that appears most frequently in the given token_id_seq
    // WARN: Because HashMap does not maintain any order of the key-value pairs,
    // if there are multiple token_pairs that occur the most frequently,
    // we can't assure that the same token will be selected each time.
    fn find_most_frequent_token_pair(token_id_seq: &[TokenID]) -> Option<(Token, usize)> {
        let mut freq_table: HashMap<Token, usize> = HashMap::new();
        for w in token_id_seq.windows(2) {
            let token = Token(w[0], w[1]);
            freq_table.entry(token).and_modify(|v| *v += 1).or_insert(1);
        }

        // If no token appears more than once
        // stop merging token in new token.
        let (token, times) = freq_table
            .iter()
            .max_by_key(|kv| kv.1)
            .map(|(token, times)| (*token, *times))?;

        if times > 1 {
            Some((token, times))
        } else {
            None
        }
    }

    // Replace every form_token in token_id_seq to to_token_id
    fn replace_token_to_token_id(
        token_id_seq: Vec<TokenID>,
        from_token: Token,
        to_token_id: TokenID,
    ) -> Vec<TokenID> {
        let mut new_token_id_seq = Vec::with_capacity(token_id_seq.len());
        let mut i: usize = 0;
        while i < token_id_seq.len() {
            if i + 1 < token_id_seq.len()
                && Token(token_id_seq[i], token_id_seq[i + 1]) == from_token
            {
                new_token_id_seq.push(to_token_id);
                i += 2;
            } else {
                new_token_id_seq.push(token_id_seq[i]);
                i += 1;
            }
        }
        new_token_id_seq.shrink_to_fit();
        new_token_id_seq
    }

    // convert a token to a byte sequence
    fn convert_token_id_to_bytes(vocabulary: &[Token], token_id: usize) -> Vec<u8> {
        let mut bytes: Vec<u8> = vec![];
        let Token(left, right) = vocabulary[token_id];
        if token_id == left {
            debug_assert!(
                token_id >= u8::MIN as usize && token_id <= u8::MAX as usize,
                "token_id should in [0, 255], now is {}",
                token_id
            );
            bytes.push(token_id as u8);
        } else {
            let mut left_bytes = Self::convert_token_id_to_bytes(vocabulary, left);
            let mut right_bytes = Self::convert_token_id_to_bytes(vocabulary, right);
            bytes.append(&mut left_bytes);
            bytes.append(&mut right_bytes);
        }
        bytes
    }
}
