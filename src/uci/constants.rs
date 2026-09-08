use crate::board::constants::INITIAL_FEN;
use wasm_bindgen::prelude::wasm_bindgen;

#[wasm_bindgen]
pub fn get_initial_board_fen() -> String {
    INITIAL_FEN.to_string()
}
