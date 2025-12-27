use wasm_bindgen::prelude::*;

use crate::board::Board;
use crate::mv::Move;
use crate::nnue::{Weights1, Weights2};
use crate::search::search_data::SearchData;
use crate::search::search_root::{search_root, SearchIterationResponse, SearchResponse};
use crate::search::transposition::TranspositionTable;
use crate::terminal::check_terminal;
use crate::types::Side;

#[wasm_bindgen]
pub struct Engine {
    tt: TranspositionTable,
    search_data: SearchData,
    board: Board,
    best_move: Option<Move>,
}

impl Engine {
    pub fn new(tt_size_mb: usize, w1: &Weights1, w2: &Weights2) -> Self {
        Self {
            tt: TranspositionTable::new(tt_size_mb),
            search_data: SearchData::new(),
            board: Board::new_with_nnue(w1.clone(), w2.clone()),
            best_move: None,
        }
    }
    pub fn new_no_nnue(tt_size_mb: usize) -> Self {
        Self {
            tt: TranspositionTable::new(tt_size_mb),
            search_data: SearchData::new(),
            board: Board::new(),
            best_move: None,
        }
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
        let res = search_root(&mut self.board, &mut self.search_data, &mut self.tt, on_iteration);
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