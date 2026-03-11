use std::collections::HashMap;

type TokenID = usize;

fn main() {
    let mut vocabulary: Vec<(TokenID, TokenID)> = init_vocabulary();
    const MAX_VOCABULARY_LEN: usize = u8::MAX as usize + 10;

    let crops = [
        "Hello, World!",
        "The quick brown fox jumps over the lazy dog",
        // From https://en.wikipedia.org/wiki/Byte-pair_encoding
        "The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup freq_table of the replacements is required to rebuild the initial dataset. The modified version builds tokens (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words)."
    ];
    let mut token_seq: Vec<TokenID> = crops[1].bytes().map(|b| b as TokenID).collect();

    while vocabulary.len() <= MAX_VOCABULARY_LEN {
        println!("Old tokens sequence: {:?}", token_seq);

        if let Some((token_pair, times)) = find_most_frequent_token_pair(&token_seq) {
            let new_token_id = vocabulary.len();
            println!(
                "New Token {} => {:?}: {} times",
                new_token_id, token_pair, times
            );
            token_seq = replace_token_pair_to_single_token(token_seq, token_pair, new_token_id);
            vocabulary.push(token_pair);
        } else {
            println!("most frequent token_pair not found");
            break;
        }

        println!("New tokens sequence: {:?}\n", token_seq);
    }
}

// Init the vocabulary to:
// 0 => { left: 0, right: 0}
// 1 => { left: 1, right: 0}
// ...
// 255 => { left: 255, right: 0}
fn init_vocabulary() -> Vec<(TokenID, TokenID)> {
    let mut vocabulary = vec![];
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
    freq_table
        .iter()
        .max_by_key(|kv| kv.1)
        .map(|(token_pair, times)| (*token_pair, *times))
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
