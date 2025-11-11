use wasm_bindgen::prelude::*;

use crate::board::Board;
use crate::mv::Move;
use crate::search::search_data::SearchData;
use crate::search::search_root::{search_root, SearchIterationResponse, SearchResponse};
use crate::search::transposition::TranspositionTable;

#[wasm_bindgen]
pub struct Engine {
    tt: TranspositionTable,
    search_data: SearchData,
    board: Board,
    best_move: Option<Move>,
}

impl Engine {
    pub fn new(tt_size_mb: usize) -> Self {
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

    pub fn set_position_and_moves(&mut self, fen: &str, moves: Vec<Move>) -> Result<(), JsValue> {
        self.board.clear();

        if let Err(_) =self.board.set_fen(fen) {
            return Err(JsValue::from_str("Invalid FEN string"));
        }

        for mv in moves {
            self.board.make_move(mv, &mut self.search_data.undos[0])?;
        }

        Ok(())
    }

    pub fn make_search(&mut self, time: u64, on_iteration: Option<&dyn Fn(SearchIterationResponse)>) -> SearchResponse {
        self.search_data.start_timer(time);
        let res = search_root(&mut self.board, &mut self.search_data, &mut self.tt, on_iteration);
        self.best_move = Some(res.best_move);
        res
    }

    pub fn print_board(&self) {
        println!("{:?}", self.board);
    }

    pub fn board(&self) -> &Board {
        &self.board
    }
}