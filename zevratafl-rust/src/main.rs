use zevratafl_rust::board::board::Board;
use zevratafl_rust::board::constants::INITIAL_FEN;
use zevratafl_rust::engine::Engine;
use zevratafl_rust::moves::movegen::MoveGen;
use zevratafl_rust::moves::undo::UndoMove;
use zevratafl_rust::search::search_root::SearchIterationResponse;
use zevratafl_rust::dataset::play_random_games;
use zevratafl_rust::nnue::{load_fc1_from_raw, load_fc1_single_line, load_fc2_from_raw, load_fc2_single_line};
use zevratafl_rust::WasmClient;

struct SearchData {
    move_gens: Vec<MoveGen>,
    undos: Vec<UndoMove>,
}

impl SearchData {
    pub fn new(max_depth: usize) -> Self {
        let mut move_gens = Vec::with_capacity(max_depth);
        let mut undos = Vec::with_capacity(max_depth);
        for _ in 0..max_depth {
            move_gens.push(MoveGen::new());
            undos.push(UndoMove::new());
        }
        Self { move_gens, undos }
    }
}

fn perft_test(board: &mut Board, depth: usize, search_data: &mut SearchData) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut nodes = 0;
    search_data.move_gens[depth].generate_moves(board);

    for i in 0..search_data.move_gens[depth].count() {
        let mv = search_data.move_gens[depth].moves()[i];
        board.make_move(mv, &mut search_data.undos[depth]).unwrap();
        nodes += perft_test(board, depth - 1, search_data);
        board.unmake_move(&mut search_data.undos[depth]).unwrap();
    }

    nodes
}


fn generate_dataset(file_name: Option<String>) {
    match file_name {
        Some(f) => {
            play_random_games(10000000, f);
        },
        None => {
            println!("Please provide a file name as the first argument.");
            return;
        }
    };
}

fn main() {
    let file_name = std::env::args().nth(1);
    generate_dataset(file_name);
    return;
    // let mut ss = WasmClient::new("ss".to_string(), 32);
    //
    // ss.run("uci");
    // ss.run("position fen 3aaaaa3/5a5/11/a4d4a/a3ddd3a/aa1ddkdd1aa/a3ddd3a/a4d4a/11/5a5/3aaaaa3 a moves g1g2");
    //     ss.print_board();
    // ss.run("go movetime 1000");
    // return;;

    // let file_name = std::env::args().nth(1);
    //
    // if let Some(f) = file_name {
    //     play_random_games(10000000, f);
    // }
    //
    // return;

    // let w1 = load_fc1_single_line("nnue-gen2/fc1.62.weights.csv");
    // let w2 = load_fc2_single_line("nnue-gen2/fc2.62.weights.csv");

    let w1 = load_fc1_from_raw();
    let w2 = load_fc2_from_raw();
    //
    let mut engine = Engine::new(32, &w1, &w2);
    engine.set_position_and_moves(INITIAL_FEN, vec![]);
    //
    // engine.set_position_and_moves("akaaaaaaaaa/k9d/dddddddd/11/11/11/11/11/11/11/aaaaaaaaaaa a", vec![]);
    // engine.board().nnue.debug_float_full();
    // println!("Eval at start: {}", engine.board().get_sided_eval());
    // return;

    engine.make_search(
        10000000,
        Some(&|iteration: SearchIterationResponse| {
            println!(
                "Depth: {}, Best Move: {:?}, Score: {}, Nodes: {}, Time: {}ms, Speed: {} knps",
                iteration.depth,
                iteration.mv,
                iteration.score,
                iteration.nodes,
                iteration.time,
                iteration.speed
            );
        })
    );


    // let mut board = Board::new();
    // board.set_fen(INITIAL_FEN).unwrap();
    //
    //
    // let max_depth = 6;
    // let mut search_data = SearchData::new(1000);
    // for depth in 1..=max_depth {
    //     let time_start = std::time::Instant::now();
    //     let nodes = perft_test(&mut board, depth, &mut search_data);
    //     let time_elapsed = time_start.elapsed();
    //     let speed_knps = (nodes as f64) / (time_elapsed.as_secs_f64() * 1000.0);
    //     println!("Perft to depth {}: {} nodes in {}s, speed: {}", depth, nodes, time_elapsed.as_secs_f64(), speed_knps);
    // }
}
