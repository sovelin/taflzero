use taflzero::board::rules::{RulesEnum, get_rules_enum_from_str};
use taflzero::gen_train_data::{DatagenConfig, gen_train_data};
use taflzero::search::nn::NeuralNet;
use taflzero::{ConsoleClient, UciRunState};

struct CliArgs {
    net_path: String,
    datagen_path: Option<String>,
    datagen_count: Option<usize>,
    gamelog_path: Option<String>,
    dump_sample_path: Option<String>,
    variant: RulesEnum,
    curriculum_fraction: f64,
    curriculum_path: Option<String>,
    curriculum_max_size: usize,
}

fn parse_args() -> CliArgs {
    let mut net_path = String::from("./default_nn.onnx");
    let mut datagen_path: Option<String> = None;
    let mut datagen_count: Option<usize> = None;
    let mut gamelog_path: Option<String> = None;
    let mut dump_sample_path: Option<String> = None;
    let mut variant = RulesEnum::Copenhagen11x11;
    let mut curriculum_fraction = 0.0f64;
    let mut curriculum_path: Option<String> = None;
    let mut curriculum_max_size = 50_000usize;
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
            "--datagen-count" => {
                if let Some(raw) = args.next() {
                    match raw.parse::<usize>() {
                        Ok(0) => {
                            eprintln!("--datagen-count must be > 0");
                            std::process::exit(2);
                        }
                        Ok(v) => datagen_count = Some(v),
                        Err(_) => {
                            eprintln!("Invalid value for --datagen-count: {raw}");
                            std::process::exit(2);
                        }
                    }
                } else {
                    eprintln!("Missing value for --datagen-count");
                    std::process::exit(2);
                }
            }
            "--gamelog" => {
                if let Some(path) = args.next() {
                    gamelog_path = Some(path);
                } else {
                    eprintln!("Missing value for --gamelog");
                    std::process::exit(2);
                }
            }
            "--dump-sample" => {
                if let Some(path) = args.next() {
                    dump_sample_path = Some(path);
                } else {
                    eprintln!("Missing value for --dump-sample");
                    std::process::exit(2);
                }
            }
            "--variant" => {
                if let Some(raw) = args.next() {
                    match get_rules_enum_from_str(&raw) {
                        Some(v) => variant = v,
                        None => {
                            eprintln!("Unknown variant: {raw}");
                            std::process::exit(2);
                        }
                    }
                } else {
                    eprintln!("Missing value for --variant");
                    std::process::exit(2);
                }
            }
            "--curriculum-fraction" => {
                if let Some(raw) = args.next() {
                    match raw.parse::<f64>() {
                        Ok(v) if (0.0..=1.0).contains(&v) => curriculum_fraction = v,
                        _ => {
                            eprintln!("Invalid value for --curriculum-fraction: {raw}");
                            std::process::exit(2);
                        }
                    }
                } else {
                    eprintln!("Missing value for --curriculum-fraction");
                    std::process::exit(2);
                }
            }
            "--curriculum-path" => {
                if let Some(path) = args.next() {
                    curriculum_path = Some(path);
                } else {
                    eprintln!("Missing value for --curriculum-path");
                    std::process::exit(2);
                }
            }
            "--curriculum-max-size" => {
                if let Some(raw) = args.next() {
                    match raw.parse::<usize>() {
                        Ok(v) if v > 0 => curriculum_max_size = v,
                        _ => {
                            eprintln!("Invalid value for --curriculum-max-size: {raw}");
                            std::process::exit(2);
                        }
                    }
                } else {
                    eprintln!("Missing value for --curriculum-max-size");
                    std::process::exit(2);
                }
            }
            _ => {
                eprintln!("Unknown arg: {arg}");
                eprintln!(
                    "Usage: taflzero [--net <model.onnx>] [--datagen <output.bin>] [--datagen-count <games>] [--datagen-count <N>] [--variant <name>] [--dump-sample <output.bin>]"
                );
                std::process::exit(2);
            }
        }
    }

    CliArgs {
        net_path,
        datagen_path,
        datagen_count,
        gamelog_path,
        dump_sample_path,
        variant,
        curriculum_fraction,
        curriculum_path,
        curriculum_max_size,
    }
}

fn main() {
    let cli = parse_args();

    if cli.datagen_count.is_some() && cli.datagen_path.is_none() {
        eprintln!("--datagen-count can only be used together with --datagen");
        std::process::exit(2);
    }

    if let Some(path) = cli.dump_sample_path {
        taflzero::gen_train_data::dump_single_sample(&path);
        return;
    }

    if let Some(path) = cli.datagen_path {
        let mut nn = NeuralNet::new(&cli.net_path);

        let log_path = cli
            .gamelog_path
            .unwrap_or_else(|| format!("{}.gamelog", path));

        let curriculum_path = cli.curriculum_path.unwrap_or_else(|| {
            // Default: curriculum.bin next to selfplay.bin
            let p = std::path::Path::new(&path);
            p.with_file_name("curriculum.bin")
                .to_string_lossy()
                .into_owned()
        });

        let datagen_cfg = DatagenConfig {
            curriculum_fraction: cli.curriculum_fraction,
            curriculum_path: if cli.curriculum_fraction > 0.0 {
                Some(curriculum_path)
            } else {
                None
            },
            curriculum_max_size: cli.curriculum_max_size,
        };

        gen_train_data(&path, &log_path, &mut nn, cli.datagen_count, cli.variant, datagen_cfg);
        return;
    }

    run_console_uci(cli.net_path);
}

fn run_console_uci(net_path: String) {
    use std::io;

    let mut client = ConsoleClient::new(net_path);
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
