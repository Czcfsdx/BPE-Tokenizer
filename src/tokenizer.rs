use anyhow::{Context, Result};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::read_to_string;

type TokenID = usize;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, Archive, Deserialize, Serialize)]
pub struct Token(TokenID, TokenID);

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:>3}, {:>3})", self.0, self.1)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Tokenizer {
    // regexp '(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s
    vocabulary: Vec<Token>,              // for decode
    tokens_map: HashMap<Token, TokenID>, // for encode
}

// for `save` and `lode`
#[derive(Archive, Deserialize, Serialize)]
struct TokenizerData {
    pub vocabulary: Vec<Token>,
    // Store as a Vec for a more compact and faster loading experience.
    pub tokens_map_vec: Vec<(Token, TokenID)>,
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
        Self {
            vocabulary,
            tokens_map: HashMap::new(),
        }
    }

    pub fn train(&mut self, max_vocabulary_size: usize, path: &str, verbose: bool) -> Result<()> {
        if max_vocabulary_size < u8::MAX as usize {
            anyhow::bail!(
                "max_vocabulary_size must be greater than {}, now is {}",
                u8::MAX as usize,
                max_vocabulary_size
            );
        }

        let mut token_id_seq: Vec<TokenID> = read_to_string(path)
            .with_context(|| format!("Failed to read from the file: {}", path))?
            .bytes()
            .map(|b| b as TokenID)
            .collect();

        while self.vocabulary.len() <= max_vocabulary_size {
            let Some((token, times)) = Self::find_most_frequent_token(&token_id_seq) else {
                if verbose {
                    println!("New token not found");
                }
                break;
            };

            let new_token_id = self.vocabulary.len();
            token_id_seq = Self::replace_token_to_token_id(token_id_seq, token, new_token_id);
            self.vocabulary.push(token);
            self.tokens_map.insert(token, new_token_id);
            if verbose {
                let token_bytes = Self::decode_token_id_to_bytes(&self.vocabulary, new_token_id);
                match String::from_utf8(token_bytes) {
                        Ok(token_string) => println!(
                            "New token {:>3} ({:>2} times) => {}: ({:?})",
                            new_token_id,
                            times,
                            token,
                            token_string
                        ),
                    // Don't return this error, handle the error on-site.
                        Err(error) => println!(
                            "New token {:>3} ({:>2} times) => {} But error occurs when converting it to String: {}",
                            new_token_id,
                            times,
                            token,
                            error
                        ),
                    }
            }
        }
        Ok(())
    }

    // encode a given string text to a token id sequence
    pub fn encode(&self, text: &str, verbose: bool) -> Vec<TokenID> {
        let mut token_id_seq: Vec<TokenID> = text.bytes().map(|b| b as TokenID).collect();
        let mut tokens_not_in_vocabulary: HashSet<Token> = HashSet::new();
        while token_id_seq.len() >= 2 {
            let Some((token, _)) = Self::find_most_frequent_token_with_exclusion(
                &token_id_seq,
                &tokens_not_in_vocabulary,
            ) else {
                break;
            };

            let Some(&token_id) = self.tokens_map.get(&token) else {
                assert!(
                    !tokens_not_in_vocabulary.contains(&token),
                    "Token {} appear even be excluded",
                    token
                );
                // exclude the token which are not in the vocabulary
                tokens_not_in_vocabulary.insert(token);
                continue;
            };

            token_id_seq = Self::replace_token_to_token_id(token_id_seq, token, token_id);
            if verbose {
                println!("Replace: {} => {:>3}", token, token_id);
            }
        }
        token_id_seq
    }

    // decode a given token id sequence to a String
    pub fn decode(&self, token_id_seq: &[TokenID]) -> Result<String> {
        Ok(token_id_seq
            .iter()
            .map(|t| String::from_utf8(Self::decode_token_id_to_bytes(&self.vocabulary, *t)))
            .collect::<Result<Vec<String>, std::string::FromUtf8Error>>()
            .with_context(|| "Fail to decode the given token id sequence")?
            .concat())
    }

    // dump the tokenizer (`vocabulary` and `tokens_map`) into a file in the given path.
    pub fn save(&self, path: &str) -> Result<()> {
        let data = TokenizerData {
            vocabulary: self.vocabulary.clone(),
            tokens_map_vec: self.tokens_map.iter().map(|(k, v)| (*k, *v)).collect(),
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .with_context(|| "Fail to serialize the tokenizer model")?;
        std::fs::write(path, &bytes).with_context(|| {
            format!(
                "Fail to write the binary serialization of the tokenizer model to the file: {}",
                path,
            )
        })?;
        Ok(())
    }

    // load the tokenizer (`vocabulary` and `tokens_map`) form a file in the given path.
    pub fn load(path: &str) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("Fail to read from the file: {}", path))?;
        let data = rkyv::from_bytes::<TokenizerData, rkyv::rancor::Error>(&bytes)
            .with_context(|| format!("Fail to deserialize the data from the file: {}", path))?;
        Ok(Self {
            vocabulary: data.vocabulary,
            tokens_map: data.tokens_map_vec.into_iter().collect(),
        })
    }

    // render the vocabulary as a String.
    pub fn vocabulary_to_text(&self) -> String {
        let mut output = String::new();
        for (token_id, token) in self.vocabulary.iter().enumerate() {
            if token_id <= u8::MAX as TokenID {
                continue;
            }
            let temp =
                match String::from_utf8(Self::decode_token_id_to_bytes(&self.vocabulary, token_id))
                {
                    Ok(token_string) => {
                        format!("Token {:>3} => {}: ({:?})\n", token_id, token, token_string)
                    }
                    // Don't return this error, handle the error on-site.
                    Err(error) => format!(
                        "Token {:>3} => {} But error occurs when converting it to String: {}\n",
                        token_id, token, error
                    ),
                };
            output.push_str(&temp);
        }
        output
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
    fn find_most_frequent_token(token_id_seq: &[TokenID]) -> Option<(Token, usize)> {
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

    // Find the token pair that appears most frequently in the given token_id_seq
    // and also not in the exclude_set
    // WARN: the same as `find_most_frequent_token`
    fn find_most_frequent_token_with_exclusion(
        token_id_seq: &[TokenID],
        exclude_set: &HashSet<Token>,
    ) -> Option<(Token, usize)> {
        let mut freq_table: HashMap<Token, usize> = HashMap::new();
        for w in token_id_seq.windows(2) {
            let token = Token(w[0], w[1]);
            if !exclude_set.contains(&token) {
                freq_table.entry(token).and_modify(|v| *v += 1).or_insert(1);
            }
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

    // decode a token to a byte sequence
    fn decode_token_id_to_bytes(vocabulary: &[Token], token_id: TokenID) -> Vec<u8> {
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
            let mut left_bytes = Self::decode_token_id_to_bytes(vocabulary, left);
            let mut right_bytes = Self::decode_token_id_to_bytes(vocabulary, right);
            bytes.append(&mut left_bytes);
            bytes.append(&mut right_bytes);
        }
        bytes
    }
}
