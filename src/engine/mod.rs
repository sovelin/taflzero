use wasm_bindgen::prelude::*;

use crate::board::Board;
use crate::mcts::mcts::MCTSTree;
use crate::mv::Move;
use crate::nnue::{Weights1, Weights2};
use crate::search::nn::NeuralNet;
use crate::search::search_data::SearchData;
use crate::search::search_root::{search_root, search_root_nodes, SearchIterationResponse, SearchResponse};
use crate::search::transposition::TranspositionTable;
use crate::terminal::check_terminal;
use crate::types::Side;

#[wasm_bindgen]
pub struct Engine {
    tt: TranspositionTable,
    search_data: SearchData,
    board: Board,
    nn: NeuralNet,
    best_move: Option<Move>,
    tree: MCTSTree,
    config: EngineConfig
}

pub struct EngineConfig {
    pub net_path: String,
}

impl Engine {

    pub fn new(tt_size_mb: usize, net_path: String) -> Self {
        let net_path = String::from(net_path);

        let config = EngineConfig {
            net_path: net_path.clone(),
        };
        let nn = NeuralNet::new(config.net_path.as_str());

        let mut board = Board::new();
        board.setup_initial_position().expect("Setup initial position failed");

        Self {
            tt: TranspositionTable::new(tt_size_mb),
            search_data: SearchData::new(),
            nn,
            best_move: None,
            tree: MCTSTree::new(),
            config,
            board
        }
    }

    pub fn set_nn(&mut self, path: String) {
        self.config.net_path = String::from(path);

        // try catching error here and returning it instead of panicking

        self.nn = NeuralNet::new(self.config.net_path.as_str());
    }

    pub fn best_move(&self) -> Option<Move> {
        self.best_move
    }

    pub fn set_position_and_moves(&mut self, fen: &str, moves: Vec<Move>) {
        self.board.clear();
        self.board.set_fen(fen).unwrap();

        for mv in moves {
            self.board.make_move_simple(mv).unwrap();
        }
    }

    pub fn make_search(&mut self, time: u64, depth: u32, on_iteration: Option<&dyn Fn(SearchIterationResponse)>) -> SearchResponse {
        self.search_data.start_timer(time, depth);
        let res = search_root(&mut self.board, &mut self.search_data, &mut self.tt, &mut self.nn, on_iteration, &mut self.tree);
        self.best_move = Some(res.best_move);
        res
    }

    pub fn make_search_nodes(&mut self, nodes: u64, on_iteration: Option<&dyn Fn(SearchIterationResponse)>) -> SearchResponse {
        let res = search_root_nodes(&mut self.board, &mut self.search_data, &mut self.tt, &mut self.nn, on_iteration, &mut self.tree, nodes);
        self.best_move = Some(res.best_move);
        res
    }

    pub fn make_move(&mut self, mv: Move) -> Result<(), &'static str> {
        self.board.make_move_simple(mv)
    }

    pub fn print_board(&self) {
        println!("{:?}", self.board);
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn check_terminal(&mut self) -> Option<Side> {
        check_terminal(&mut self.board)
    }

    pub fn get_board_mutable(&mut self) -> &mut Board {
        &mut self.board
    }
}
