use anyhow::{Context, Result, anyhow, bail};
use fancy_regex::Regex;
use rkyv::{Archive, Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::io::{BufRead, BufReader};
use std::num::NonZero;
use std::thread;
use std::time::Instant;

// Pattern from: https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
const DEFAULT_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";

type Token = usize;

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash, Archive, Deserialize, Serialize)]
pub struct Pair(Token, Token);

impl fmt::Display for Pair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct Tokenizer {
    config: TokenizerConfig,
    vocabulary: Vec<Pair>,        // for decode
    merges: HashMap<Pair, Token>, // for encode
    inverse_special_tokens: HashMap<String, Token>,
    decode_cache: RefCell<HashMap<Token, Vec<u8>>>, // Not support for multiple threads
}

// for `save` and `load`
#[derive(Archive, Deserialize, Serialize)]
struct TokenizerData {
    max_vocabulary_size: usize,
    pre_tokenizer_pattern: String,
    special_tokens: Vec<String>,
    vocabulary: Vec<Pair>,
    merges_vec: Vec<(Pair, Token)>, // Store as a Vec for a more compact and faster loading experience.
}

// for configuration parse
#[derive(PartialEq, Eq, Debug)]
pub struct TokenizerConfig {
    max_vocabulary_size: usize,
    pre_tokenizer_pattern: String,
    special_tokens: Vec<String>,
    pub train_path: Option<String>,
    pub verbose: bool,
    pub save_path: Option<String>,
    pub num_threads: NonZero<usize>,
    pub interval: Option<NonZero<usize>>,
}

impl fmt::Display for TokenizerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Tokenizer Configuration:")?;
        writeln!(f, "    max_vocabulary_size: {}", self.max_vocabulary_size)?;
        writeln!(f, "    pre_tokenizer_pattern: {:#?}", self.pre_tokenizer_pattern)?;
        writeln!(f, "    special_tokens: [")?;
        for special_token in &self.special_tokens {
            writeln!(f, "        {:#?},", special_token)?;
        }
        writeln!(f, "    ]")?;
        match &self.train_path {
            Some(path) => writeln!(f, "    train_path: {:#?}", path)?,
            None => writeln!(f, "    train_path: None")?,
        }
        match &self.save_path {
            Some(path) => writeln!(f, "    save_path: {:#?}", path)?,
            None => writeln!(f, "    save_path: None")?,
        }
        match &self.interval {
            Some(i) => writeln!(f, "    report_interval: {}", i)?,
            None => writeln!(f, "    report_interval: None")?,
        }
        writeln!(f, "    verbose: {}", self.verbose)?;
        write!(f, "    jobs: {}", self.num_threads)?;
        Ok(())
    }
}

// Public Method
impl Tokenizer {
    pub fn train(&mut self) -> Result<()> {
        println!("{}\n", self.config);
        // Pre-tokenize
        let pattern = &self.config.pre_tokenizer_pattern;
        debug_assert!(!pattern.is_empty(), "pre-tokenize pattern is empty!");
        let regex = Regex::new(pattern)?;

        let Some(path) = self.config.train_path.as_deref() else {
            bail!(
                "Fail to found train_path in training! Maybe because you want to retrain a model which is loaded from a binary serialization."
            )
        };
        // TODO: Don't read all content in once.
        let file_content = fs::read_to_string(path)
            .with_context(|| format!("Fail to read from the file: {}", path))?;

        let corpus: Vec<&str> = regex
            .find_iter(&file_content)
            .filter_map(|m| m.ok())
            .map(|m| m.as_str())
            .collect();

        let mut corpus_tokens = HashMap::new();
        for str in corpus {
            let tokens = str.bytes().map(|b| b as Token).collect::<Vec<Token>>();
            corpus_tokens
                .entry(tokens)
                .and_modify(|v| *v += 1)
                .or_insert(1);
        }

        println!("Start training...");
        let verbose = self.config.verbose;
        let train_timer = Instant::now();
        let mut interval_timer = Instant::now();
        for index in 1..=self.config.max_vocabulary_size {
            let Some((pair, times)) =
                Self::find_most_frequent_pair(&corpus_tokens, self.config.num_threads)?
            else {
                if verbose {
                    println!("New token not found. Stop training.");
                }
                break;
            };

            // If no token appears more than once, stop merging token in new token.
            if times <= 1 {
                if verbose {
                    println!("New token not found. Stop training.");
                }
                break;
            }

            let new_token = index + u8::MAX as usize;
            corpus_tokens = Self::replace_pair_to_token(
                corpus_tokens,
                pair,
                new_token,
                self.config.num_threads,
            )?;
            self.vocabulary.push(pair);
            self.merges.insert(pair, new_token);
            if verbose {
                match self.decode(&[new_token]) {
                    Ok(token_str) => println!(
                        "New token {} ({} times) => {}: ({:?})",
                        new_token, times, pair, token_str
                    ),
                    // Don't return this error, handle the error on-site.
                    Err(error) => println!(
                        "New token {} ({} times) => {} But error occurs when converting it to String: {}",
                        new_token, times, pair, error
                    ),
                }
            }

            if self.config.interval.is_some_and(|x| index % x == 0) {
                println!("Episode {index}");
                println!("  time used: {}s", interval_timer.elapsed().as_secs_f64());
                println!("  vocabulary size: {}", u8::MAX as usize + index);
                interval_timer = Instant::now();
            }
        }
        println!("End training.");
        println!("  time used: {}s", train_timer.elapsed().as_secs_f64());
        println!("  vocabulary size: {}", self.vocabulary.len());
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
            let special_regex = Regex::new(&special_pattern)
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
        if !self.config.special_tokens.is_empty() {
            let mut result: String = String::new();
            let mut last_special = 0;
            let min_special_token = self.config.max_vocabulary_size + (u8::MAX as usize) + 1;

            for (i, &token) in tokens.iter().enumerate() {
                // whether the token is a special token
                if token >= min_special_token {
                    result.push_str(&self.decode_ordinary(&tokens[last_special..i])?);
                    if let Some(str) = self.config.special_tokens.get(token - min_special_token) {
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
    pub fn save(&self) -> Result<()> {
        let data = TokenizerData {
            max_vocabulary_size: self.config.max_vocabulary_size,
            pre_tokenizer_pattern: self.config.pre_tokenizer_pattern.clone(),
            special_tokens: self.config.special_tokens.clone(),
            vocabulary: self.vocabulary.clone(),
            merges_vec: self.merges.iter().map(|(k, v)| (*k, *v)).collect(),
        };
        let bytes = rkyv::to_bytes::<rkyv::rancor::Error>(&data)
            .with_context(|| "Fail to serialize the tokenizer model")?;

        let Some(path) = self.config.save_path.as_deref() else {
            bail!(
                "Fail to found save_path in saving! Maybe because you want to resave a model which is loaded from a binary serialization."
            )
        };
        if let Some(parent) = std::path::Path::new(path).parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(path, &bytes).with_context(|| {
            format!(
                "Fail to write the binary serialization of the tokenizer model to the file: {}",
                path,
            )
        })?;
        Ok(())
    }
}

impl fmt::Display for Tokenizer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (token, pair) in self.vocabulary.iter().enumerate() {
            if token <= u8::MAX as Token {
                if f.alternate() {
                    writeln!(f, "Token {} => {}: {:?}", token, pair, token as u8 as char)?
                }
                continue;
            }

            let temp = match self.decode(&[token]) {
                Ok(token_str) => format!("Token {} => {}: {:?}", token, pair, token_str),
                // Don't return this error, handle the error on-site.
                Err(error) => format!(
                    "Token {} => {} But error occurs when converting it to String: {}",
                    token, pair, error
                ),
            };
            writeln!(f, "{}", temp)?
        }

        if f.alternate() {
            for (token, str) in self.config.special_tokens.iter().enumerate() {
                let min_special_token = self.config.max_vocabulary_size + (u8::MAX as usize) + 1;
                writeln!(
                    f,
                    "Special Token {} => {:?}",
                    token + min_special_token,
                    str
                )?
            }
        }
        Ok(())
    }
}

// Private Method
impl Tokenizer {
    // encode a string that ignores any special tokens.
    fn encode_ordinary(&self, text: &str) -> Result<Vec<Token>> {
        // Pre-tokenize
        let pattern = &self.config.pre_tokenizer_pattern;
        debug_assert!(!pattern.is_empty(), "pre-tokenize pattern is empty!");
        let regex = Regex::new(pattern)?;
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
    fn decode_token_to_bytes(&self, token: Token) -> Vec<u8> {
        // Cache
        if let Some(cached_bytes) = self.decode_cache.borrow().get(&token) {
            return cached_bytes.clone();
        }

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
        self.decode_cache.borrow_mut().insert(token, bytes.clone());
        bytes
    }
}

// Associated Function
impl Tokenizer {
    pub fn new(config: TokenizerConfig) -> Result<Self> {
        // Initial the vocabulary to:
        // 0 => { left: 0, right: 0}
        // 1 => { left: 1, right: 0}
        // ...
        // 255 => { left: 255, right: 0}
        let mut vocabulary = Vec::with_capacity(u8::MAX as usize);
        for i in u8::MIN..=u8::MAX {
            vocabulary.push(Pair(i as usize, 0));
        }

        let min_special_token = config.max_vocabulary_size + (u8::MAX as usize) + 1;
        let inverse_special_tokens = config
            .special_tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i + min_special_token))
            .collect();

        Ok(Self {
            config,
            vocabulary,
            merges: HashMap::new(),
            inverse_special_tokens,
            decode_cache: RefCell::new(HashMap::new()),
        })
    }

    // load the tokenizer form a file in the given path.
    pub fn load(path: &str) -> Result<Self> {
        let bytes =
            fs::read(path).with_context(|| format!("Fail to read from the file: {}", path))?;
        let data = rkyv::from_bytes::<TokenizerData, rkyv::rancor::Error>(&bytes)
            .with_context(|| format!("Fail to deserialize the data from the file: {}", path))?;
        let config = TokenizerConfig {
            max_vocabulary_size: data.max_vocabulary_size,
            pre_tokenizer_pattern: data.pre_tokenizer_pattern,
            special_tokens: data.special_tokens,
            train_path: None,
            verbose: false,
            save_path: None,
            interval: None,
            num_threads: NonZero::<usize>::MIN,
        };
        let min_special_token = data.max_vocabulary_size + (u8::MAX as usize) + 1;
        let inverse_special_tokens = config
            .special_tokens
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i + min_special_token))
            .collect();
        Ok(Self {
            config,
            vocabulary: data.vocabulary,
            merges: data.merges_vec.into_iter().collect(),
            inverse_special_tokens,
            decode_cache: RefCell::new(HashMap::new()),
        })
    }

    // Find the token pair that appears most frequently in the given tokens sequence
    // WARN: Because HashMap does not maintain any order of the key-value pairs,
    // if there are multiple token pairs that occur the same times,
    // we can't assure that the same token will be selected each time.
    fn find_most_frequent_pair(
        corpus: &HashMap<Vec<Token>, usize>,
        num_threads: NonZero<usize>,
    ) -> Result<Option<(Pair, usize)>> {
        // To use chunk(), we need to transform HashMap to Vec
        let vec: Vec<(Vec<Token>, usize)> = corpus
            .iter()
            .map(|(tokens, value)| (tokens.clone(), *value))
            .collect();
        let chunk_size = corpus.len().div_ceil(num_threads.get());
        let chunks: Vec<_> = vec.chunks(chunk_size).collect();

        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let chunk = chunk.to_vec();
                thread::spawn(move || {
                    let mut local_freq_table = HashMap::new();
                    for (token, count) in chunk {
                        for w in token.windows(2) {
                            let pair = Pair(w[0], w[1]);
                            local_freq_table
                                .entry(pair)
                                .and_modify(|v| *v += count)
                                .or_insert(count);
                        }
                    }
                    local_freq_table
                })
            })
            .collect();

        let mut freq_table: HashMap<Pair, usize> = HashMap::new();
        for (i, h) in handles.into_iter().enumerate() {
            match h.join() {
                Ok(local_map) => {
                    for (pair, count) in local_map {
                        freq_table
                            .entry(pair)
                            .and_modify(|v| *v += count)
                            .or_insert(count);
                    }
                }
                Err(e) => bail!("Thread {i} panicked when counting pair frequency: {e:?}"),
            }
        }

        Ok(freq_table
            .iter()
            .max_by_key(|&(_, value)| value)
            .map(|(token, times)| (*token, *times)))
    }

    // Replace form_pair in every token in corpus to to_token
    fn replace_pair_to_token(
        corpus: HashMap<Vec<Token>, usize>,
        from_pair: Pair,
        to_token: Token,
        num_threads: NonZero<usize>,
    ) -> Result<HashMap<Vec<Token>, usize>> {
        let length = corpus.len();
        // To use chunk(), we need to transform HashMap to Vec
        let vec: Vec<(Vec<Token>, usize)> = corpus.into_iter().collect();
        let chunk_size = length.div_ceil(num_threads.get());
        let chunks: Vec<_> = vec.chunks(chunk_size).collect();

        let handles: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let chunk = chunk.to_vec();
                thread::spawn(move || {
                    let mut local_map = HashMap::with_capacity(chunk.len());
                    for (tokens, value) in chunk {
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
                        local_map.insert(new_tokens, value);
                    }
                    local_map.shrink_to_fit();
                    local_map
                })
            })
            .collect();

        let mut new_corpus = HashMap::with_capacity(length);
        for (i, h) in handles.into_iter().enumerate() {
            match h.join() {
                Ok(local_map) => {
                    for (tokens, value) in local_map {
                        new_corpus
                            .entry(tokens)
                            .and_modify(|v| *v += value)
                            .or_insert(value);
                    }
                }
                Err(e) => bail!("Thread {i} panicked when counting pair frequency: {e:?}"),
            }
        }
        Ok(new_corpus)
    }
}

impl TokenizerConfig {
    // Create TokenizerConfig by merging Config parse
    // from a configuration file with arguments passing.
    pub fn new(
        path: &str,
        train_path: Option<String>,
        verbose: bool,
        save_path: Option<String>,
        interval: Option<NonZero<usize>>,
        jobs: Option<NonZero<usize>>,
    ) -> Result<Self> {
        // merge argument and configuration
        let mut config = Self::parse_config_file(path)?;
        config.verbose = config.verbose || verbose;
        if train_path.is_some() {
            config.train_path = train_path;
        }
        if save_path.is_some() {
            config.save_path = save_path;
        }
        if interval.is_some() {
            config.interval = interval;
        }
        if let Some(num_threads) = jobs {
            config.num_threads = num_threads;
        }

        // check train_path, save_path
        if config.train_path.is_none() {
            bail!(
                "Fail to found train_path.\nPlease specify by --train-path or check your configuration file: {path}"
            )
        }
        if config.save_path.is_none() {
            bail!(
                "Fail to found save_path.\nPlease specify by --save-path or check your configuration file: {path}"
            )
        }

        Ok(config)
    }

    // Parse TokenizerConfig from a configuration file.
    fn parse_config_file(path: &str) -> Result<Self> {
        let mut max_vocabulary_size: Option<usize> = None;
        let mut pre_tokenizer_pattern: Option<String> = None;
        let mut special_tokens: Option<Vec<String>> = None;
        let mut train_path: Option<String> = None;
        let mut verbose: bool = false;
        let mut save_path: Option<String> = None;
        let mut interval: Option<NonZero<usize>> = None;
        let mut num_threads: NonZero<usize> = NonZero::<usize>::MIN;

        let file = fs::File::open(path)
            .with_context(|| format!("Fail to read from the configuration file: {}", path))?;
        let file = BufReader::new(file);
        for line in file.lines() {
            let line = line?;

            // Ignore comment
            let content = match line.split_once('#') {
                Some((before, _)) => before.trim(),
                None => line.trim(),
            };
            if content.is_empty() {
                continue;
            };

            // parse verbose
            if content == "verbose" {
                verbose = true;
                continue;
            }

            let Some((key, value)) = content.split_once('=') else {
                continue;
            };
            let key = key.trim();
            let value = value.trim();

            match key {
                "max_vocabulary_size" => max_vocabulary_size = Some(
                    value.parse().with_context(|| format!("Fail to parse max_vocabulary_size from \"{value}\". Please check your configuration file: {path}"))?
                ),
                "pre_tokenizer_pattern" => pre_tokenizer_pattern = Some(String::from(value)),
                "special_tokens" => special_tokens = Some(
                    value.split(',')
                        .filter_map(|s| {
                            let trimmed = s.trim();
                            if trimmed.is_empty() {
                                None
                            } else {
                                Some(trimmed.to_string())
                            }
                        })
                        .collect()
                ),
                "train_path" => train_path = Some(String::from(value)),
                "save_path" => save_path = Some(String::from(value)),
                "jobs" => num_threads = value.parse().with_context(|| format!("Fail to parse jobs from \"{value}\". Please check your configuration file: {path}"))?,
                "report_interval" => interval = Some(
                    value.parse().with_context(|| format!("Fail to parse report_interval from \"{value}\". Please check your configuration file: {path}"))?
                ),
                _ => continue,
            };
        }

        let max_vocabulary_size = max_vocabulary_size.ok_or(anyhow!(
            "Fail to found max_vocabulary_size.\nPlease check your configuration file: {path}"
        ))?;
        let special_tokens = special_tokens.unwrap_or_default();
        let pre_tokenizer_pattern = pre_tokenizer_pattern.unwrap_or(DEFAULT_PATTERN.to_string());
        if pre_tokenizer_pattern.is_empty() {
            bail!(
                "pre_tokenizer_pattern must not be empty.\nPlease check your configuration file: {path}"
            )
        }
        let maximum_num_threads = thread::available_parallelism().with_context(|| "Fail to get the maximum amount of parallelism available for this program using std::thread::available_parallelism.")?;
        if num_threads.get() != 1 && num_threads > maximum_num_threads {
            println!(
                "jobs is {num_threads}, which is greater than the maximum amount of parallelism available for this program. Now set jobs to {maximum_num_threads}"
            );
        }
        Ok(TokenizerConfig {
            max_vocabulary_size,
            pre_tokenizer_pattern,
            special_tokens,
            train_path,
            verbose,
            save_path,
            interval,
            num_threads,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_tokenizer() -> Tokenizer {
        const CONFIG_PATH: &str = "tests/test.conf";
        let config = TokenizerConfig::new(CONFIG_PATH, None, false, None, None, None)
            .expect("Fail to parse configuration file");
        let mut model = Tokenizer::new(config).expect("Fail to create tokenizer");
        model.train().expect("Fail to train tokenizer");
        model
    }

    #[test]
    fn test_create() {
        const MAX_VOCABULARY_SIZE: usize = 200;
        const SPECIAL_TOKENS: [&str; 3] = ["<|beginoftext|>", "<|middleoftext|>", "<|endoftext|>"];
        const PATTERN: &str =
            r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
        const TRAIN_CROPS_PATH: &str = "tests/bpe-wiki.txt";
        const VERBOSE: bool = true;
        const SAVE_PATH: &str = "tests/test.bin";
        const CONFIG_PATH: &str = "tests/test.conf";
        let config = TokenizerConfig::new(CONFIG_PATH, None, false, None, None, None)
            .expect("Fail to parse configuration file");
        let model = Tokenizer::new(config).expect("Fail to create tokenizer");
        assert_eq!(model.config.pre_tokenizer_pattern, PATTERN);
        assert_eq!(model.config.max_vocabulary_size, MAX_VOCABULARY_SIZE);
        assert_eq!(model.config.special_tokens, SPECIAL_TOKENS);
        assert_eq!(model.config.train_path.as_deref(), Some(TRAIN_CROPS_PATH));
        assert_eq!(model.config.verbose, VERBOSE);
        assert_eq!(model.config.save_path.as_deref(), Some(SAVE_PATH));
    }

    #[test]
    fn test_save_and_load() {
        let model = create_test_tokenizer();
        const MODEL_PATH: &str = "tests/test.bin";
        model.save().expect("Fail to save model");
        let new_model = Tokenizer::load(MODEL_PATH).expect("Fail to load model");
        assert_eq!(
            model.config.pre_tokenizer_pattern,
            new_model.config.pre_tokenizer_pattern
        );
        assert_eq!(
            model.config.max_vocabulary_size,
            new_model.config.max_vocabulary_size
        );
        assert_eq!(model.config.special_tokens, new_model.config.special_tokens);
        assert_eq!(model.vocabulary, new_model.vocabulary);
        assert_eq!(model.merges, new_model.merges);
        assert_eq!(
            model.inverse_special_tokens,
            new_model.inverse_special_tokens
        );
    }

    #[test]
    fn test_encode_and_decode() {
        let model = create_test_tokenizer();
        const TEXT: &str = "<|beginoftext|>+ Byte-pair encoding (BPE) is a text compression algorithm from 1994 that iteratively replaces frequent byte pairs with placeholder symbols. 这是一些混入的中文。 <|middleoftext|><|middleoftext|>Modern large language models use a modified version that converts text into \"tokens\" (natural numbers) by merging frequent character sequences, [😄😡😭 <>/?{}!@#$%^&*-=_+\\|;:`~] creating a fixed-size vocabulary. -<|endoftext|>";
        let tokens = model.encode(TEXT).expect("Fail to encode");
        let decoded_text = model.decode(&tokens).expect("Fail to decode");
        assert_eq!(TEXT, decoded_text);
    }
}
