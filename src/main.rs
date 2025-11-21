use zevratafl_rust::dataset::play_random_games;
use zevratafl_rust::{ConsoleClient, UciRunState};

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
            generate_dataset(target);
        }
        Some(cmd) if cmd == "uci" => run_console_uci(),
        Some(file_name) => generate_dataset(Some(file_name)),
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
