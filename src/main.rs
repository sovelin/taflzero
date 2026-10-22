use taflzero::board::Board;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
use taflzero::board::position_export::BitPosition;
use taflzero::board::rules::{RulesEnum, get_rules_enum_from_str};
use taflzero::gen_train_data::{
    DaemonConfig, DatagenConfig, DatagenSink, GameLimits, SearchConfig, WorkerOverrides,
    gen_train_data, gen_train_data_into,
};
use taflzero::match_play::{MatchConfig, play_match, replay_positions};
use taflzero::mcts::MCTSConfig;
use taflzero::mv::move_to_algebraic;
use taflzero::search::nn::{BatchServer, NeuralNet, fill_input};
use taflzero::{ConsoleClient, UciRunState, get_sample_size};

struct CliArgs {
    net_path: String,
    datagen_path: Option<String>,
    datagen_count: Option<usize>,
    gamelog_path: Option<String>,
    dump_sample_path: Option<String>,
    nn_eval_fen: Option<String>,
    variant: RulesEnum,
    curriculum_fraction: f64,
    curriculum_path: Option<String>,
    curriculum_max_size: usize,
    full_nodes: u64,
    cheap_nodes: u64,
    full_prob: f64,
    max_plies: usize,
    max_no_capture: usize,
    gumbel: Option<usize>,
    daemon: DaemonConfig,
    /// Game threads per datagen process, all served by one batching net.
    threads: usize,
    /// Leaves per net call inside one search (1 = no virtual loss).
    search_batch: usize,
    /// Game threads start spread over this many seconds, so a restarted worker does
    /// not finish its first games all at once (a burst of short games in the data).
    ramp_seconds: usize,
    /// `--match`: paired match `--a` vs `--b` instead of UCI (see match_play).
    match_mode: bool,
    match_a: String,
    match_b: String,
    pairs: usize,
    nodes: u64,
    opening_moves: usize,
    seed: Option<u64>,
    /// `--replay "<moves>"`: print the positions of a recorded game (see run_replay).
    replay: Option<String>,
}

fn parse_args() -> CliArgs {
    let mut net_path = String::from("./copenhagen11x11.onnx");
    let mut datagen_path: Option<String> = None;
    let mut datagen_count: Option<usize> = None;
    let mut gamelog_path: Option<String> = None;
    let mut dump_sample_path: Option<String> = None;
    let mut nn_eval_fen: Option<String> = None;
    let mut variant = RulesEnum::Copenhagen11x11;
    let mut curriculum_fraction = 0.0f64;
    let mut curriculum_path: Option<String> = None;
    let mut curriculum_max_size = 50_000usize;
    let mut full_nodes = 400u64;
    let mut cheap_nodes = 100u64;
    let mut full_prob = 1.0f64;
    let limits = GameLimits::default();
    let mut max_plies = limits.max_plies;
    let mut max_no_capture = limits.max_no_capture;
    let mut gumbel: Option<usize> = None;
    let mut daemon = DaemonConfig::default();
    let mut threads = 1usize;
    // Datagen and UCI play both default to 8; `--match` follows play.
    let mut search_batch = MCTSConfig::default_play().batch_size;
    let mut ramp_seconds = 0usize;
    let mut match_mode = false;
    let mut match_a = String::new();
    let mut match_b = String::new();
    let mut pairs = 50usize;
    let mut nodes = 400u64;
    let mut opening_moves = 16usize;
    let mut seed: Option<u64> = None;
    let mut replay: Option<String> = None;
    let mut args = std::env::args().skip(1);

    // `--flag <usize>` with a lower bound; exits with a usage error otherwise.
    fn usize_arg(args: &mut impl Iterator<Item = String>, flag: &str, min: usize) -> usize {
        let Some(raw) = args.next() else {
            eprintln!("Missing value for {flag}");
            std::process::exit(2);
        };
        match raw.parse::<usize>() {
            Ok(v) if v >= min => v,
            _ => {
                eprintln!("Invalid value for {flag}: {raw}");
                std::process::exit(2);
            }
        }
    }
    fn path_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> String {
        let Some(raw) = args.next() else {
            eprintln!("Missing value for {flag}");
            std::process::exit(2);
        };
        raw
    }

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
                        // 0 = no limit: play until the stop file appears (or forever).
                        Ok(0) => datagen_count = None,
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
            "--nn-eval" => {
                if let Some(fen) = args.next() {
                    nn_eval_fen = Some(fen);
                } else {
                    eprintln!("Missing value for --nn-eval");
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
            "--full-nodes" => {
                if let Some(raw) = args.next() {
                    match raw.parse::<u64>() {
                        Ok(v) if v > 0 => full_nodes = v,
                        _ => {
                            eprintln!("Invalid value for --full-nodes: {raw}");
                            std::process::exit(2);
                        }
                    }
                } else {
                    eprintln!("Missing value for --full-nodes");
                    std::process::exit(2);
                }
            }
            "--cheap-nodes" => {
                if let Some(raw) = args.next() {
                    match raw.parse::<u64>() {
                        Ok(v) if v > 0 => cheap_nodes = v,
                        _ => {
                            eprintln!("Invalid value for --cheap-nodes: {raw}");
                            std::process::exit(2);
                        }
                    }
                } else {
                    eprintln!("Missing value for --cheap-nodes");
                    std::process::exit(2);
                }
            }
            "--full-prob" => {
                if let Some(raw) = args.next() {
                    match raw.parse::<f64>() {
                        Ok(v) if (0.0..=1.0).contains(&v) && v > 0.0 => full_prob = v,
                        _ => {
                            eprintln!("Invalid value for --full-prob: {raw}");
                            std::process::exit(2);
                        }
                    }
                } else {
                    eprintln!("Missing value for --full-prob");
                    std::process::exit(2);
                }
            }
            "--max-plies" => max_plies = usize_arg(&mut args, "--max-plies", 1),
            "--max-no-capture" => max_no_capture = usize_arg(&mut args, "--max-no-capture", 1),
            "--gumbel" => gumbel = Some(usize_arg(&mut args, "--gumbel", 2)),
            "--net-watch" => daemon.net_watch = true,
            "--stop-file" => daemon.stop_file = Some(path_arg(&mut args, "--stop-file")),
            "--pause-file" => daemon.pause_file = Some(path_arg(&mut args, "--pause-file")),
            "--worker-id" => daemon.worker_id = Some(usize_arg(&mut args, "--worker-id", 0)),
            "--threads" => threads = usize_arg(&mut args, "--threads", 1),
            "--search-batch" => search_batch = usize_arg(&mut args, "--search-batch", 1),
            "--ramp-seconds" => ramp_seconds = usize_arg(&mut args, "--ramp-seconds", 0),
            "--match" => match_mode = true,
            "--a" => match_a = path_arg(&mut args, "--a"),
            "--b" => match_b = path_arg(&mut args, "--b"),
            "--pairs" => pairs = usize_arg(&mut args, "--pairs", 1),
            "--nodes" => nodes = usize_arg(&mut args, "--nodes", 1) as u64,
            "--opening-moves" => opening_moves = usize_arg(&mut args, "--opening-moves", 0),
            "--seed" => seed = Some(usize_arg(&mut args, "--seed", 0) as u64),
            "--replay" => replay = Some(path_arg(&mut args, "--replay")),
            _ => {
                eprintln!("Unknown arg: {arg}");
                eprintln!(
                    "Usage: taflzero [--net <model.onnx>] [--datagen <output.bin>] [--datagen-count <N|0=unlimited>] \
                     [--gamelog <path>] [--variant <name>] [--full-nodes N] [--cheap-nodes N] [--full-prob F] \
                     [--gumbel <considered>] [--max-plies N] [--max-no-capture N] \
                     [--net-watch] [--stop-file <path>] [--pause-file <path>] [--worker-id N] [--threads N] \
                     [--curriculum-fraction F] [--curriculum-path <path>] [--curriculum-max-size N] \
                     [--dump-sample <output.bin>] [--nn-eval <fen>]"
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
        nn_eval_fen,
        variant,
        curriculum_fraction,
        curriculum_path,
        curriculum_max_size,
        full_nodes,
        cheap_nodes,
        full_prob,
        max_plies,
        max_no_capture,
        gumbel,
        daemon,
        threads,
        search_batch,
        ramp_seconds,
        match_mode,
        match_a,
        match_b,
        pairs,
        nodes,
        opening_moves,
        seed,
        replay,
    }
}

/// `--match`: one line per finished game, `game <pair> <a_attacks 0|1> <winner> <reason>
/// <plies>`, followed by `moves <pair> <a_attacks> <m1> <m2> …` (algebraic, the opening
/// included) and `evals <pair> <a_attacks> <v1> …` (the mover's root value of each
/// searched move), then `done`. tools/match.py tallies them and stores the record.
fn run_match(cli: &CliArgs) {
    if cli.match_a.is_empty() || cli.match_b.is_empty() {
        eprintln!("--match needs --a and --b");
        std::process::exit(2);
    }
    let board_size = cli.variant.rules().board_size;
    let games = cli.pairs * 2;
    let threads = cli.threads.clamp(1, games);
    let search_batch = cli.search_batch.max(1);
    let server = |path: &str| {
        // The batch is the games on this net's move, `search_batch` leaves each.
        match BatchServer::start(
            path,
            board_size,
            threads * search_batch,
            std::time::Duration::from_millis(1),
            false,
        ) {
            Ok(s) if s.board_size() == board_size => s,
            Ok(s) => {
                eprintln!(
                    "neural net '{path}' is built for a {0}x{0} board but this run needs {board_size}x{board_size}",
                    s.board_size()
                );
                std::process::exit(2);
            }
            Err(err) => {
                eprintln!("{err}");
                std::process::exit(2);
            }
        }
    };
    let a = server(&cli.match_a);
    let b = server(&cli.match_b);
    let cfg = MatchConfig {
        pairs: cli.pairs,
        nodes: cli.nodes,
        opening_moves: cli.opening_moves,
        max_plies: cli.max_plies,
        threads,
        search_batch,
        variant: cli.variant,
        seed: cli.seed,
    };
    let out = std::sync::Mutex::new(std::io::stdout());
    play_match(&a, &b, &cfg, &|g| {
        use std::io::Write;
        let mut o = out.lock().unwrap();
        let _ = writeln!(
            o,
            "game {} {} {} {} {}",
            g.pair, g.a_attacks as u8, g.winner, g.reason, g.plies
        );
        // The record of the game for the dashboard's viewer, right after its `game`
        // line and under the same lock, so lines of concurrent games never interleave.
        let moves: Vec<String> = g
            .moves
            .iter()
            .map(|&mv| move_to_algebraic(mv, board_size))
            .collect();
        let evals: Vec<String> = g.evals.iter().map(|v| format!("{v:.4}")).collect();
        let _ = writeln!(
            o,
            "moves {} {} {}",
            g.pair,
            g.a_attacks as u8,
            moves.join(" ")
        );
        let _ = writeln!(
            o,
            "evals {} {} {}",
            g.pair,
            g.a_attacks as u8,
            evals.join(" ")
        );
        let _ = o.flush();
    });
    println!("done");
}

/// `--replay "<moves>"`: the positions of a recorded game for the dashboard's viewer,
/// one line `<ply> <hex>` per position (ply 0 is the start position, then one after
/// each move), the `BitPosition` as 100 hex digits (50 bytes), then `done`.
fn run_replay(cli: &CliArgs, moves: &str) {
    match replay_positions(cli.variant, moves) {
        Ok(positions) => {
            for (ply, p) in positions.iter().enumerate() {
                let hex: String = p.as_bytes().iter().map(|b| format!("{b:02x}")).collect();
                println!("{ply} {hex}");
            }
            println!("done");
        }
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(1);
        }
    }
}

/// `worker.overrides` next to the datagen file wins over the command line: it is how
/// a worker under a trainer that cannot be restarted gets new settings.
fn apply_worker_overrides(cli: &mut CliArgs) {
    let Some(path) = &cli.datagen_path else {
        return;
    };
    match WorkerOverrides::next_to(path) {
        None => {}
        Some((file, Err(err))) => {
            eprintln!("{file}: {err}");
            std::process::exit(2);
        }
        Some((file, Ok(o))) => {
            if let Some(t) = o.threads {
                println!("[datagen] {file}: threads {} -> {t}", cli.threads);
                cli.threads = t;
            }
            if let Some(b) = o.search_batch {
                println!("[datagen] {file}: search_batch {} -> {b}", cli.search_batch);
                cli.search_batch = b;
            }
            if let Some(r) = o.ramp_seconds {
                println!("[datagen] {file}: ramp_seconds {} -> {r}", cli.ramp_seconds);
                cli.ramp_seconds = r;
            }
        }
    }
}

/// Loads a net and refuses to continue unless it fits the board this run will play on.
/// `batch_size` is the shape the session is warmed up on — see `NeuralNet::batch_size`.
fn load_net_for(path: &str, board_size: usize, batch_size: usize) -> NeuralNet {
    let nn = match NeuralNet::new(path, board_size, batch_size) {
        Ok(nn) => nn,
        Err(err) => {
            eprintln!("{err}");
            std::process::exit(2);
        }
    };

    if nn.board_size() != board_size {
        eprintln!(
            "neural net '{path}' is built for a {0}x{0} board but this run needs {board_size}x{board_size}",
            nn.board_size()
        );
        std::process::exit(2);
    }

    nn
}

fn main() {
    let mut cli = parse_args();
    apply_worker_overrides(&mut cli);
    if let Some(moves) = cli.replay.clone() {
        run_replay(&cli, &moves);
        return;
    }
    if cli.match_mode {
        run_match(&cli);
        return;
    }

    if cli.datagen_path.is_none()
        && (cli.datagen_count.is_some()
            || cli.daemon.net_watch
            || cli.daemon.stop_file.is_some()
            || cli.daemon.pause_file.is_some())
    {
        eprintln!("--datagen-count / --net-watch / --stop-file / --pause-file need --datagen");
        std::process::exit(2);
    }

    if let Some(path) = cli.dump_sample_path {
        taflzero::gen_train_data::dump_single_sample(&path);
        return;
    }

    // Debug: dump NN input tensor + raw policy/value for one FEN, as JSON to stdout.
    // Used to cross-check the Rust inference pipeline against the Python model.
    if let Some(fen) = cli.nn_eval_fen {
        let mut board = Board::new();
        board.set_rules(cli.variant);
        board.set_fen(&fen).expect("Invalid FEN");

        let mut nn = load_net_for(
            &cli.net_path,
            board.board_size(),
            MCTSConfig::default_play().batch_size,
        );

        let bit_pos = BitPosition::from_board(&board, 1);
        let mut input = vec![0f32; get_sample_size(board.board_size())];
        fill_input(&mut input, &bit_pos, board.precomputed());

        let out = nn.evaluate_position(&bit_pos, board.precomputed());

        let f2s = |xs: &[f32]| {
            xs.iter()
                .map(|v| format!("{v}"))
                .collect::<Vec<_>>()
                .join(",")
        };
        let bytes = bit_pos
            .as_bytes()
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(",");
        println!("{{");
        println!("  \"stm\": {}, \"rep\": {},", bit_pos.stm, bit_pos.rep);
        println!("  \"bitpos_bytes\": [{bytes}],");
        println!("  \"input\": [{}],", f2s(&input));
        println!("  \"value\": {},", out.value);
        println!("  \"policy\": [{}]", f2s(&out.policy));
        println!("}}");
        return;
    }

    if let Some(path) = cli.datagen_path {
        let board_size = cli.variant.rules().board_size;
        let search_batch = cli.search_batch;

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
            search: SearchConfig {
                full_nodes: cli.full_nodes,
                cheap_nodes: cli.cheap_nodes,
                full_prob: cli.full_prob,
                gumbel: cli.gumbel,
                search_batch,
            },
            limits: GameLimits {
                max_plies: cli.max_plies,
                max_no_capture: cli.max_no_capture,
            },
            daemon: cli.daemon,
        };

        if cli.threads > 1 {
            if datagen_cfg.curriculum_fraction > 0.0 {
                eprintln!("--curriculum-fraction is not supported with --threads > 1");
                std::process::exit(2);
            }
            // One session for all game threads: a batch closes when every thread has
            // a request in or after this long, whichever is first.
            let server = match BatchServer::start(
                &cli.net_path,
                board_size,
                cli.threads * search_batch,
                std::time::Duration::from_millis(2),
                datagen_cfg.daemon.net_watch,
            ) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("{err}");
                    std::process::exit(2);
                }
            };
            if server.board_size() != board_size {
                eprintln!(
                    "neural net '{}' is built for a {1}x{1} board but this run needs {board_size}x{board_size}",
                    cli.net_path,
                    server.board_size()
                );
                std::process::exit(2);
            }
            let sink = DatagenSink::open(&path, &log_path);
            let per_thread = cli.datagen_count.map(|n| n.div_ceil(cli.threads));
            let variant = cli.variant;
            std::thread::scope(|scope| {
                for t in 0..cli.threads {
                    let server = &server;
                    let sink = &sink;
                    let mut cfg = datagen_cfg.clone();
                    cfg.daemon.thread = Some(t);
                    // The server reloads the net; threads must not try on their handle.
                    cfg.daemon.net_watch = false;
                    let delay = std::time::Duration::from_secs_f64(
                        cli.ramp_seconds as f64 * t as f64 / cli.threads as f64,
                    );
                    scope.spawn(move || {
                        std::thread::sleep(delay);
                        let mut nn = NeuralNet::shared(std::sync::Arc::clone(server));
                        server.client_started();
                        gen_train_data_into(sink, &mut nn, per_thread, variant, cfg);
                        server.client_stopped();
                    });
                }
            });
            let (batches, served) = server.stats();
            println!(
                "[NN] batch server: {served} samples in {batches} batches ({:.1} per batch)",
                if batches > 0 {
                    served as f64 / batches as f64
                } else {
                    0.0
                }
            );
            return;
        }

        let mut nn = load_net_for(&cli.net_path, board_size, search_batch);
        gen_train_data(
            &path,
            &log_path,
            &mut nn,
            cli.datagen_count,
            cli.variant,
            datagen_cfg,
        );
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
