use zevratafl_rust::dataset::play_random_games;
use zevratafl_rust::{ConsoleClient, UciRunState};
use zevratafl_rust::gen_train_data::gen_train_data;

fn generate_dataset(file_name: Option<String>) {
    match file_name {
        Some(f) => {
            play_random_games(1000000000, f);
        },
        None => {
            println!("Please provide a file name as the first argument.");
            return;
        }
    };
}

fn main() {
    let mut args = std::env::args();
    let _binary = args.next();

    match args.next() {
        Some(cmd) if cmd == "dataset" => {
            let target = args.next();
            // generate_dataset(target);
            println!("{:?}", target);
            if let Some(t) = target {
                gen_train_data(&t);
            } else {
                println!("Generating dataset and printing to console");
            }
        }
        Some(cmd) if cmd == "uci" => run_console_uci(),
        Some(file_name) => {
            gen_train_data(&file_name)
        },
        None => run_console_uci(),
    }
}

fn run_console_uci() {
    use std::io;

    let mut client = ConsoleClient::new(32);
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        line.clear();
        let bytes = stdin.read_line(&mut line).unwrap_or(0);
        if bytes == 0 {
            break;
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if matches!(client.run_line(trimmed), UciRunState::Quit) {
            break;
        }
    }
}
