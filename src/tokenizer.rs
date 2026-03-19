use anyhow::{Context, Result};
use rkyv::{Archive, Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::fs::read_to_string;

type Token = usize;

// Pattern from: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
const DEFAULT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, Archive, Deserialize, Serialize)]
pub struct Pair(Token, Token);

impl fmt::Display for Pair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({:>3}, {:>3})", self.0, self.1)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Tokenizer {
    max_vocabulary_size: usize,
    pre_tokenizer_pattern: String,
    vocabulary: Vec<Pair>,        // for decode
    merges: HashMap<Pair, Token>, // for encode
    special_tokens: Vec<String>,
    inverse_special_tokens: HashMap<String, Token>,
}

// for `save` and `lode`
#[derive(Archive, Deserialize, Serialize)]
struct TokenizerData {
    max_vocabulary_size: usize,
    pre_tokenizer_pattern: String,
    vocabulary: Vec<Pair>,
    merges_vec: Vec<(Pair, Token)>, // Store as a Vec for a more compact and faster loading experience.
    special_tokens: Vec<String>,
}

// Public Method
impl Tokenizer {
    pub fn train(&mut self, path: &str, verbose: bool) -> Result<()> {
        // Pre-tokenize
        let regex = fancy_regex::Regex::new(DEFAULT_PATTERN)?;
        let file_content = read_to_string(path)
            .with_context(|| format!("Failed to read from the file: {}", path))?;
        // PERF: Maybe we can use HashMap<&str, usize> to store the crops and save more space
        let crops: Vec<&str> = regex
            .find_iter(&file_content)
            .filter_map(|m| m.ok())
            .map(|m| m.as_str())
            .collect();

        let mut crops_tokens: Vec<Vec<Token>> = crops
            .iter()
            .map(|s| s.bytes().map(|b| b as Token).collect::<Vec<Token>>())
            .collect();

        while self.vocabulary.len() < self.max_vocabulary_size {
            let Some((pair, times)) = Self::find_most_frequent_pair(&crops_tokens, None) else {
                if verbose {
                    println!("New token not found");
                }
                break;
            };

            // If no token appears more than once, stop merging token in new token.
            if times <= 1 {
                if verbose {
                    println!("New token not found");
                }
                break;
            }

            let new_token = self.vocabulary.len();
            crops_tokens = crops_tokens
                .into_iter()
                .map(|v| Self::replace_pair_to_token(v, pair, new_token))
                .collect();
            self.vocabulary.push(pair);
            self.merges.insert(pair, new_token);
            if verbose {
                let token_bytes = self.decode_token_to_bytes(new_token);
                match String::from_utf8(token_bytes) {
                        Ok(token_str) => println!(
                            "New token {:>3} ({:>2} times) => {}: ({:?})",
                            new_token,
                            times,
                            pair,
                            token_str
                        ),
                    // Don't return this error, handle the error on-site.
                        Err(error) => println!(
                            "New token {:>3} ({:>2} times) => {} But error occurs when converting it to String: {}",
                            new_token,
                            times,
                            pair,
                            error
                        ),
                    }
            }
        }
        Ok(())
    }

    // encode a given string text to a token sequence
    pub fn encode(&self, text: &str) -> Result<Vec<Token>> {
        if !self.inverse_special_tokens.is_empty() {
            // special tokens match
            let special_pattern = self
                .inverse_special_tokens
                .keys()
                .map(|s| fancy_regex::escape(s).into_owned())
                .collect::<Vec<String>>()
                .join("|");
            let special_regex = fancy_regex::Regex::new(&special_pattern)
                .with_context(|| "Fail to turn special_tokens into regex expression")?;

            let mut result: Vec<Token> = Vec::new();
            let mut last_end = 0;

            for mat in special_regex.find_iter(text).flatten() {
                let start = mat.start();
                let end = mat.end();

                if last_end <= start {
                    result.append(&mut self.encode_ordinary(&text[last_end..start])?);
                    if let Some(&id) = self.inverse_special_tokens.get(mat.as_str()) {
                        result.push(id as Token);
                    }
                    last_end = end;
                }
            }
            if last_end < text.len() {
                result.append(&mut self.encode_ordinary(&text[last_end..])?);
            }

            Ok(result)
        } else {
            self.encode_ordinary(text)
        }
    }

    // decode a given token sequence to a String
    pub fn decode(&self, tokens: &[Token]) -> Result<String> {
        if !self.special_tokens.is_empty() {
            let mut result: String = String::new();
            let mut last_special = 0;

            for (i, &token) in tokens.iter().enumerate() {
                // whether the token is a special token
                if token >= self.max_vocabulary_size {
                    result.push_str(&self.decode_ordinary(&tokens[last_special..i])?);
                    if let Some(str) = self.special_tokens.get(token - self.max_vocabulary_size) {
                        result.push_str(str);
                    }
                    last_special = i + 1;
                }
            }
            if last_special < tokens.len() {
                result.push_str(&self.decode_ordinary(&tokens[last_special..])?);
            }

            Ok(result)
        } else {
            self.decode_ordinary(tokens)
        }
    }

    // dump the tokenizer into a file in the given path.
    pub fn save(&self, path: &str) -> Result<()> {
        let data = TokenizerData {
            max_vocabulary_size: self.max_vocabulary_size,
            pre_tokenizer_pattern: self.pre_tokenizer_pattern.clone(),
            vocabulary: self.vocabulary.clone(),
            merges_vec: self.merges.iter().map(|(k, v)| (*k, *v)).collect(),
            special_tokens: self.special_tokens.clone(),
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

    // load the tokenizer form a file in the given path.
    pub fn load(path: &str) -> Result<Self> {
        let bytes =
            std::fs::read(path).with_context(|| format!("Fail to read from the file: {}", path))?;
        let data = rkyv::from_bytes::<TokenizerData, rkyv::rancor::Error>(&bytes)
            .with_context(|| format!("Fail to deserialize the data from the file: {}", path))?;
        Ok(Self {
            max_vocabulary_size: data.max_vocabulary_size,
            pre_tokenizer_pattern: data.pre_tokenizer_pattern,
            vocabulary: data.vocabulary,
            merges: data.merges_vec.into_iter().collect(),
            inverse_special_tokens: data
                .special_tokens
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i + data.max_vocabulary_size))
                .collect(),
            special_tokens: data.special_tokens,
        })
    }

    // render the vocabulary as a String.
    pub fn vocabulary_to_text(&self) -> String {
        let mut result = String::new();
        for (token, pair) in self.vocabulary.iter().enumerate() {
            if token <= u8::MAX as Token {
                continue;
            }
            let temp = match String::from_utf8(self.decode_token_to_bytes(token)) {
                Ok(token_str) => {
                    format!("Token {:>3} => {}: ({:?})\n", token, pair, token_str)
                }
                // Don't return this error, handle the error on-site.
                Err(error) => format!(
                    "Token {:>3} => {} But error occurs when converting it to String: {}\n",
                    token, pair, error
                ),
            };
            result.push_str(&temp);
        }
        result
    }
}

// Private Method
impl Tokenizer {
    // encode a string that ignores any special tokens.
    fn encode_ordinary(&self, text: &str) -> Result<Vec<Token>> {
        // Pre-tokenize
        let regex = fancy_regex::Regex::new(DEFAULT_PATTERN)?;
        let chunks: Vec<&str> = regex
            .find_iter(text)
            .filter_map(|m| m.ok())
            .map(|m| m.as_str())
            .collect();

        Ok(chunks
            .iter()
            .flat_map(|&chunk| self.encode_chunk(chunk))
            .collect())
    }

    // encode a string chunk to a token sequence
    fn encode_chunk(&self, chunk: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = chunk.bytes().map(|c| c as Token).collect();

        if tokens.len() < 2 {
            return tokens;
        }

        let mut i: usize = 0;
        while i < tokens.len().saturating_sub(1) {
            let pair = Pair(tokens[i], tokens[i + 1]);
            if let Some(&new_token) = self.merges.get(&pair) {
                tokens[i] = new_token;
                tokens.remove(i + 1);
                if i > 0 {
                    i = i.saturating_sub(1);
                }
            } else {
                i += 1;
            }
        }

        tokens
    }

    // decode a tokens sequence that ignores any special tokens.
    fn decode_ordinary(&self, tokens: &[Token]) -> Result<String> {
        String::from_utf8(
            tokens
                .iter()
                .flat_map(|&t| self.decode_token_to_bytes(t))
                .collect(),
        )
        .with_context(|| "Fail to decode the given token sequence")
    }

    // decode a token to a byte sequence
    // PERF: Implement some kind of cache to not decode the same token over and over again.
    fn decode_token_to_bytes(&self, token: Token) -> Vec<u8> {
        let mut bytes: Vec<u8> = vec![];
        let Pair(left, right) = self.vocabulary[token];
        if token == left {
            debug_assert!(
                token >= u8::MIN as usize && token <= u8::MAX as usize,
                "token should in [0, 255], now is {}",
                token
            );
            bytes.push(token as u8);
        } else {
            let mut left_bytes = self.decode_token_to_bytes(left);
            let mut right_bytes = self.decode_token_to_bytes(right);
            bytes.append(&mut left_bytes);
            bytes.append(&mut right_bytes);
        }
        bytes
    }
}

// Associated Function
impl Tokenizer {
    // Init the vocabulary to:
    // 0 => { left: 0, right: 0}
    // 1 => { left: 1, right: 0}
    // ...
    // 255 => { left: 255, right: 0}
    pub fn new(
        max_vocabulary_size: usize,
        pre_tokenizer_pattern: Option<&str>,
        special_tokens: &[&str],
    ) -> Result<Self> {
        if max_vocabulary_size < u8::MAX as usize {
            anyhow::bail!(
                "max_vocabulary_size must be greater than {}, now is {}",
                u8::MAX as usize,
                max_vocabulary_size
            );
        }

        let special_tokens_vec: Vec<String> =
            special_tokens.iter().map(|&s| String::from(s)).collect();
        let mut vocabulary = Vec::with_capacity(u8::MAX as usize);
        for i in u8::MIN..=u8::MAX {
            vocabulary.push(Pair(i as usize, 0));
        }

        Ok(Self {
            max_vocabulary_size,
            vocabulary,
            pre_tokenizer_pattern: String::from(pre_tokenizer_pattern.unwrap_or(DEFAULT_PATTERN)),
            merges: HashMap::new(),
            inverse_special_tokens: special_tokens_vec
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), i + max_vocabulary_size))
                .collect(),
            special_tokens: special_tokens_vec,
        })
    }

    // Find the token pair that appears most frequently in the given tokens sequence
    // If the exclude_set is not None, it will also filter all tokens in the exclude_set
    // WARN: Because HashMap does not maintain any order of the key-value pairs,
    // if there are multiple token pairs that occur the most frequently,
    // we can't assure that the same token will be selected each time.
    fn find_most_frequent_pair(
        tokens: &[Vec<Token>],
        exclude_set: Option<&HashSet<Pair>>,
    ) -> Option<(Pair, usize)> {
        let mut freq_table: HashMap<Pair, usize> = HashMap::new();
        for text in tokens {
            for w in text.windows(2) {
                let pair = Pair(w[0], w[1]);
                if exclude_set.is_none_or(|set| !set.contains(&pair)) {
                    freq_table.entry(pair).and_modify(|v| *v += 1).or_insert(1);
                }
            }
        }

        freq_table
            .iter()
            .max_by_key(|kv| kv.1)
            .map(|(token, times)| (*token, *times))
    }

    // Replace every form_pair in tokens to to_token
    fn replace_pair_to_token(tokens: Vec<Token>, from_pair: Pair, to_token: Token) -> Vec<Token> {
        let mut new_tokens = Vec::with_capacity(tokens.len());
        let mut i: usize = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && Pair(tokens[i], tokens[i + 1]) == from_pair {
                new_tokens.push(to_token);
                i += 2;
            } else {
                new_tokens.push(tokens[i]);
                i += 1;
            }
        }
        new_tokens.shrink_to_fit();
        new_tokens
    }
}
