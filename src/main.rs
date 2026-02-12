use zevratafl_rust::{ConsoleClient, UciRunState};
use zevratafl_rust::gen_train_data::gen_train_data;
use zevratafl_rust::search::nn::NeuralNet;

struct CliArgs {
    net_path: String,
    datagen_path: Option<String>,
}

fn parse_args() -> CliArgs {
    let mut net_path = String::from("./gen1.onxx");
    let mut datagen_path: Option<String> = None;
    let mut args = std::env::args().skip(1);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--net" => {
                if let Some(path) = args.next() {
                    net_path = path;
                } else {
                    eprintln!("Missing value for --net");
                    std::process::exit(2);
                }
            }
            "--datagen" => {
                if let Some(path) = args.next() {
                    datagen_path = Some(path);
                } else {
                    eprintln!("Missing value for --datagen");
                    std::process::exit(2);
                }
            }
            _ => {
                eprintln!("Unknown arg: {arg}");
                eprintln!("Usage: zevratafl-rust [--net <model.onnx>] [--datagen <output.bin>]");
                std::process::exit(2);
            }
        }
    }

    CliArgs { net_path, datagen_path }
}

fn main() {
    let cli = parse_args();
    let mut nn = NeuralNet::new(&cli.net_path);

    if let Some(path) = cli.datagen_path {
        gen_train_data(&path, &mut nn);
        return;
    }

    run_console_uci(nn);
}

fn run_console_uci(nn: NeuralNet) {
    use std::io;

    let mut client = ConsoleClient::new(32, nn);
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
