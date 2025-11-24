use wasm_bindgen::prelude::wasm_bindgen;
use crate::{Board, Engine};
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::nnue::{load_fc1_from_raw, load_fc2_from_raw};
use crate::terminal::check_terminal;
use crate::types::{Piece, Side, Square};

#[wasm_bindgen]
struct EngineClient {
    engine: Engine,
    move_gen: MoveGen
}

#[wasm_bindgen]
impl EngineClient {
    #[wasm_bindgen(constructor)]
    pub fn new(tt_size_mb: usize) -> Self {
        let w1 = load_fc1_from_raw();
        let w2 = load_fc2_from_raw();

        let engine = Engine::new(tt_size_mb, &w1, &w2);

        Self { engine, move_gen: MoveGen::new() }
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
    pub fn check_terminal_state_for_fen(&mut self, fen: &str) -> Option<Side> {
        let board = self.engine.get_board_mutable();
        board.set_fen(fen).unwrap();
        check_terminal(board)
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

    #[wasm_bindgen]
    pub fn get_board_state(&self) -> Vec<Piece> {
        self.engine.board().board.iter().cloned().collect()
    }

    #[wasm_bindgen]
    pub fn get_fen(&self) -> String {
        self.engine.board().get_fen()
    }

    #[wasm_bindgen]
    pub fn set_fen(&mut self, fen: &str) {
        self.engine.get_board_mutable().set_fen(fen).unwrap();
    }

    #[wasm_bindgen]
    pub fn move_str_to_num(&self, mv_str: &str) -> Result<u32, String> {
        let mv = crate::mv::create_move_from_algebraic(mv_str)?;
        Ok(mv.raw())
    }

    #[wasm_bindgen]
    pub fn move_num_to_str(&self, mv_num: u32) -> String {
        let mv = Move::from_u32(mv_num);
        format!("{}", mv)
    }

    #[wasm_bindgen]
    pub fn is_move_available(&mut self, from: Square, to: Square) -> bool {
        let mv = Move::new(from, to);
        self.move_gen.generate_moves(self.engine.get_board_mutable());

        for index in 0..self.move_gen.count {
            if self.move_gen.moves[index] == mv {
                return true;
            }
        }

        false
    }
}