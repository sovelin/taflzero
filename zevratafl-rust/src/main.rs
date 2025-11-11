use std::sync::LazyLock;
use zevratafl_rust::board::board::Board;
use zevratafl_rust::board::constants::INITIAL_FEN;
use zevratafl_rust::engine::Engine;
use zevratafl_rust::moves::movegen::MoveGen;
use zevratafl_rust::moves::undo::UndoMove;
use zevratafl_rust::search::search_root::SearchIterationResponse;

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


fn main() {
    let mut engine = Engine::new(1024);

    engine.set_position_and_moves(INITIAL_FEN, vec![]).unwrap();
    //engine.set_position_and_moves("4aaaa3/4da5/4k6/3a1d4a/1aa2dd3a/a1ad2dd1aa/a2dddd3a/a4d4a/11/5a5/3aaaaa3 d", vec![]).unwrap();
    engine.print_board();

    engine.make_search(
        100000,
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
