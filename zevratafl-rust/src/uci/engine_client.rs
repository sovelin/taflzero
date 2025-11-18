use wasm_bindgen::prelude::wasm_bindgen;
use crate::Engine;
use crate::mv::Move;
use crate::nnue::{load_fc1_from_raw, load_fc2_from_raw, Weights2};
use crate::search_root::SearchResponse;
use crate::terminal::check_terminal;
use crate::types::Side;

#[wasm_bindgen]
struct EngineClient {
    engine: Engine
}

#[wasm_bindgen]
impl EngineClient {
    #[wasm_bindgen(constructor)]
    pub fn new(tt_size_mb: usize) -> Self {
        let w1 = load_fc1_from_raw();
        let w2 = load_fc2_from_raw();

        let engine = Engine::new(tt_size_mb, &w1, &w2);

        Self { engine }
    }

    #[wasm_bindgen]
    pub fn set_position_and_moves(&mut self, fen: &str, moves: Vec<u32>) {
        let moves = moves.into_iter().map(|mv_u32| Move::from_u32(mv_u32)).collect();
        self.engine.set_position_and_moves(fen, moves);
    }

    #[wasm_bindgen]
    pub fn check_terminal_state(&mut self) -> Option<Side> {
        check_terminal(self.engine.get_board_mutable())
    }

    #[wasm_bindgen]
    pub fn make_search(&mut self, time: u32) -> u32 {
        let res = self.engine.make_search(time as u64, None);
        res.best_move.raw()
    }

    #[wasm_bindgen]
    pub fn side_to_move(&self) -> Side {
        self.engine.board().side_to_move
    }

    #[wasm_bindgen]
    pub fn get_board_str(&self) -> String {
        format!("{}", self.engine.board())
    }

    #[wasm_bindgen]
    pub fn get_w2_first(&self) -> f32 {
        self.engine.board().nnue.w2[0]
    }
}