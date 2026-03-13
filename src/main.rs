use {std::collections::HashMap, std::fs::read_to_string, std::process::exit};

type TokenID = usize;

fn main() {
    let mut vocabulary: Vec<(TokenID, TokenID)> = init_vocabulary();
    const MAX_VOCABULARY_LEN: usize = u8::MAX as usize + 1000;
    const VERBOSE: bool = true;
    const TRAIN_CROPS_PATH: &str = "datasets/bpe-wiki.txt";

    let mut token_seq: Vec<TokenID> = vec![];
    match read_to_string(TRAIN_CROPS_PATH) {
        Ok(crops) => {
            token_seq = crops.bytes().map(|b| b as TokenID).collect();
        }
        Err(error) => {
            println!("Error: {}", error);
            exit(69);
        }
    }

    while vocabulary.len() <= MAX_VOCABULARY_LEN {
        if let Some((token_pair, times)) = find_most_frequent_token_pair(&token_seq) {
            let new_token_id = vocabulary.len();
            token_seq = replace_token_pair_to_single_token(token_seq, token_pair, new_token_id);
            vocabulary.push(token_pair);
            if VERBOSE {
                let token_bytes = convert_token_to_bytes(&vocabulary, new_token_id);
                match String::from_utf8(token_bytes) {
                    Ok(token_string) => println!(
                        "New token {:>3} => ({:>3}, {:>3}): {:<25} had {:>2} occurrences",
                        new_token_id, token_pair.0, token_pair.1, format!("(\"{}\")", token_string), times
                    ),
                    Err(error) => println!(
                        "New token {:>3} => ({:>3}, {:>3}) had {:>2} occurrences. But error occurs when converting it to String: {}",
                        new_token_id, token_pair.0, token_pair.1, times, error
                    ),
                }
            }
        } else {
            if VERBOSE {
                println!("New token not found");
            }
            break;
        }
    }
}

// Init the vocabulary to:
// 0 => { left: 0, right: 0}
// 1 => { left: 1, right: 0}
// ...
// 255 => { left: 255, right: 0}
fn init_vocabulary() -> Vec<(TokenID, TokenID)> {
    let mut vocabulary = Vec::with_capacity(u8::MAX as usize + 1);
    for i in u8::MIN..=u8::MAX {
        vocabulary.push((i as usize, 0));
    }
    vocabulary
}

// Find the token pair that appears most frequently in the given token_seq
// WARN: Because HashMap does not maintain any order of the key-value pairs,
// if there are multiple token_pairs that occur the most frequently,
// we can't assure that the same token_pair will be selected each time.
fn find_most_frequent_token_pair(token_seq: &[TokenID]) -> Option<((TokenID, TokenID), usize)> {
    let mut freq_table: HashMap<(TokenID, TokenID), usize> = HashMap::new();
    for w in token_seq.windows(2) {
        let token_pair = (w[0], w[1]);
        freq_table
            .entry(token_pair)
            .and_modify(|v| *v += 1)
            .or_insert(1);
    }

    // If no token_pair appears more than once
    // stop merging token_pair in new token.
    let (token_pair, times) = freq_table
        .iter()
        .max_by_key(|kv| kv.1)
        .map(|(token_pair, times)| (*token_pair, *times))?;

    if times > 1 {
        Some((token_pair, times))
    } else {
        None
    }
}

// Replace every form_token_pair in token_seq to to_token
fn replace_token_pair_to_single_token(
    token_seq: Vec<TokenID>,
    from_token_pair: (TokenID, TokenID),
    to_token: TokenID,
) -> Vec<TokenID> {
    let mut new_token_seq = Vec::with_capacity(token_seq.len());
    let mut i: usize = 0;
    while i < token_seq.len() {
        if i + 1 < token_seq.len() && (token_seq[i], token_seq[i + 1]) == from_token_pair {
            new_token_seq.push(to_token);
            i += 2;
        } else {
            new_token_seq.push(token_seq[i]);
            i += 1;
        }
    }
    new_token_seq.shrink_to_fit();
    new_token_seq
}

// convert a token to a byte sequence
fn convert_token_to_bytes(vocabulary: &[(TokenID, TokenID)], token_id: usize) -> Vec<u8> {
    let mut bytes: Vec<u8> = vec![];
    let (left, right) = vocabulary[token_id];
    if token_id == left {
        debug_assert!(
            token_id >= u8::MIN as usize && token_id <= u8::MAX as usize,
            "token_id should in [0, 255], now is {}",
            token_id
        );
        bytes.push(token_id as u8);
    } else {
        let mut left_bytes = convert_token_to_bytes(vocabulary, left);
        let mut right_bytes = convert_token_to_bytes(vocabulary, right);
        bytes.append(&mut left_bytes);
        bytes.append(&mut right_bytes);
    }
    bytes
}
