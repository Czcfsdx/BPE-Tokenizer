use std::collections::HashMap;

fn main() {
    // From https://en.wikipedia.org/wiki/Byte-pair_encoding
    let text = "The original version of the algorithm focused on compression. It replaces the highest-frequency pair of bytes with a new byte that was not contained in the initial dataset. A lookup freq_table of the replacements is required to rebuild the initial dataset. The modified version builds tokens (units of recognition) that match varying amounts of source text, from single characters (including single digits or single punctuation marks) to whole words (even long compound words).".trim();
    // let text = "Hello, World!";
    let chars: Vec<char> = text.chars().collect();
    let mut freq_table: HashMap<[char; 2], u32> = HashMap::new();
    for pair in chars.windows(2) {
        let key = [pair[0], pair[1]];
        freq_table.entry(key).and_modify(|v| *v += 1).or_insert(1);
    }

    for (key, value) in &freq_table {
        println!("{:?}: {}", key, value);
    }
    if let Some(pair_with_max_count) = freq_table.iter().max_by_key(|kv| kv.1) {
        println!(
            "The pair which occurs most frequently => {:?}, which occurs {} times.",
            pair_with_max_count.0, pair_with_max_count.1
        );
    } else {
        println!("Pair no found!");
    }
}
