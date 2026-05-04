use crate::board::constants::{BOARD_SIZE, INITIAL_FEN, SQS};
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub fn get_initial_board_fen() -> String {
    INITIAL_FEN.to_string()
}

#[wasm_bindgen]
pub fn get_board_size() -> usize {
    BOARD_SIZE
}

#[wasm_bindgen]
pub fn get_total_squares() -> usize {
    SQS
}
