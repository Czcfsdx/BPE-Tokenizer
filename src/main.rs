mod tokenizer;

use std::num::NonZero;

use clap::{Args, Parser, Subcommand};
use tokenizer::{Tokenizer, TokenizerConfig};

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a tokenizer by a configuration file
    Train(TrainArg),
    /// Encode a text into tokens
    Encode(EncodeArg),
    /// Show the vocabulary in a tokenizer
    Show(ShowArg),
}

#[derive(Args)]
struct TrainArg {
    /// Path to Configuration file of the tokenizer model to train
    #[arg(value_name = "FILE")]
    config: String,
    /// The corpus path for the tokenizer model to train (Override configuration)
    #[arg(short, long)]
    train_path: Option<String>,
    /// Show information verbosely in training (Override configuration)
    #[arg(short, long)]
    verbose: bool,
    /// The path where the tokenizer model is saved after training (Override configuration)
    #[arg(short, long)]
    save_path: Option<String>,
    /// The episode interval for reporting in training (Override configuration)
    #[arg(short, long)]
    interval: Option<NonZero<usize>>,
    /// the number of paraller threads (Override configuration)
    #[arg(short, long)]
    jobs: Option<NonZero<usize>>,
}

#[derive(Args)]
struct EncodeArg {
    /// Path to the tokenizer to encode text
    #[arg(short, long, value_name = "FILE")]
    model: String,
    /// Text for the tokenizer to encode
    #[arg(short, long)]
    text: String,
    /// Enable to show the original tokens' IDs
    #[arg(long, group = "decode options")]
    no_decode: bool,
    /// Enable to show the tokens' IDs alongside the decoded text
    #[arg(long, group = "decode options")]
    show_id: bool,
}

#[derive(Args)]
struct ShowArg {
    /// Path to the tokenizer
    #[arg(value_name = "FILE")]
    model: String,
    /// Enable to show the special tokens and tokens in 0 - 255
    #[arg(short, long)]
    all_tokens: bool,
}

fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train(arg) => train(arg),
        Commands::Encode(arg) => encode(arg),
        Commands::Show(arg) => show(arg),
    }
}

fn train(arg: TrainArg) {
    let config = TokenizerConfig::new(
        &arg.config,
        arg.train_path,
        arg.verbose,
        arg.save_path,
        arg.interval,
        arg.jobs,
    )
    .unwrap_or_else(|e| eprintln_error(e));
    let mut model = Tokenizer::new(config).unwrap_or_else(|e| eprintln_error(e));
    model.train().unwrap_or_else(|e| eprintln_error(e));
    model.save().unwrap_or_else(|e| eprintln_error(e));
}

fn encode(arg: EncodeArg) {
    let model = Tokenizer::load(&arg.model).unwrap_or_else(|e| eprintln_error(e));
    let tokens = model
        .encode(&arg.text)
        .unwrap_or_else(|e| eprintln_error(e));
    if arg.no_decode {
        println!("{:?}", tokens);
    } else if arg.show_id {
        let result = tokens
            .iter()
            .map(|&t| match model.decode(&[t]) {
                Ok(str) => Ok((t, str)),
                Err(e) => Err(e),
            })
            .collect::<Result<Vec<(_, String)>, _>>()
            .unwrap_or_else(|e| eprintln_error(e));
        println!("{:?}", result);
    } else {
        let result = tokens
            .iter()
            .map(|&t| model.decode(&[t]))
            .collect::<Result<Vec<String>, _>>()
            .unwrap_or_else(|e| eprintln_error(e));
        println!("{:?}", result);
    }
}

fn show(arg: ShowArg) {
    let model = Tokenizer::load(&arg.model).unwrap_or_else(|e| eprintln_error(e));
    println!("The vocabulary in {}:", &arg.model);
    if arg.all_tokens {
        print!("{:#}", model);
    } else {
        print!("{}", model);
    }
}

fn eprintln_error(error: anyhow::Error) -> ! {
    eprintln!("{:?}", error);
    std::process::exit(69);
}
