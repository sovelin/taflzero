use crate::board::constants::BOARD_SIZE;
use crate::board::types::{Col, Row, Square};
use crate::board::utils::get_sq_algebraic;
use std::fmt::{Debug, Display};
use wasm_bindgen::prelude::wasm_bindgen;

#[repr(transparent)]
#[derive(Copy, Clone)]
#[wasm_bindgen]
pub struct Move(u32);

impl Default for Move {
    #[inline]
    fn default() -> Self {
        Move(0)
    }
}

impl Debug for Move {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from_sq = self.from();
        let to_sq = self.to();
        let alg_from = get_sq_algebraic(from_sq);
        let alg_to = get_sq_algebraic(to_sq);
        write!(f, "{}{}", alg_from, alg_to)
    }
}

impl Display for Move {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let from_sq = self.from();
        let to_sq = self.to();
        let alg_from = get_sq_algebraic(from_sq);
        let alg_to = get_sq_algebraic(to_sq);
        write!(f, "{}{}", alg_from, alg_to)
    }
}

#[wasm_bindgen]
impl Move {
    #[inline]
    #[wasm_bindgen(constructor)]
    pub fn new(from: Square, to: Square) -> Self {
        Move(((from as u32) << 16) | (to as u32))
    }

    #[inline]
    #[wasm_bindgen]
    pub fn from(&self) -> Square {
        ((self.0 >> 16) & 0xFFFF) as Square
    }

    #[inline]
    #[wasm_bindgen]
    pub fn to(&self) -> Square {
        (self.0 & 0xFFFF) as Square
    }

    #[inline]
    #[wasm_bindgen]
    pub fn raw(&self) -> u32 {
        self.0
    }

    #[inline]
    #[wasm_bindgen]
    pub fn is_null(&self) -> bool {
        self.0 == 0
    }

    #[inline]
    #[wasm_bindgen]
    pub fn create_null() -> Self {
        Move(0)
    }

    #[inline]
    #[wasm_bindgen]
    pub fn from_u32(mv_u32: u32) -> Self {
        Move(mv_u32)
    }
}

pub fn create_move_from_algebraic(s: &str) -> Result<Move, String> {
    if s.len() < 4 || s.len() > 6 {
        return Err(format!("Invalid move format length: {s}"));
    }

    let bytes = s.as_bytes();

    let from_file = bytes[0];
    if !(b'a'..=b'k').contains(&from_file) {
        return Err(format!("Invalid from-file: {s}"));
    }
    let from_col: Col = (from_file - b'a') as Col;

    let mut i = 1;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i >= bytes.len() {
        return Err(format!("Invalid move format (missing to-file): {s}"));
    }

    let to_file = bytes[i];
    if !(b'a'..=b'k').contains(&to_file) {
        return Err(format!("Invalid to-file: {s}"));
    }
    let to_col: Col = (to_file - b'a') as Col;

    let from_rank_str = &s[1..i];
    let to_rank_str = s
        .get(i + 1..)
        .ok_or_else(|| format!("Invalid move format (missing to-rank): {s}"))?;

    let from_rank_num: usize = from_rank_str
        .parse()
        .map_err(|_| format!("Invalid from-rank: {from_rank_str}"))?;
    let to_rank_num: usize = to_rank_str
        .parse()
        .map_err(|_| format!("Invalid to-rank: {to_rank_str}"))?;

    if !(1..=BOARD_SIZE).contains(&from_rank_num) || !(1..=BOARD_SIZE).contains(&to_rank_num) {
        return Err(format!("Rank out of range (1..={BOARD_SIZE}): {s}"));
    }

    let from_row: Row = (from_rank_num - 1) as Row;
    let to_row: Row = (to_rank_num - 1) as Row;

    let from_sq: Square = from_row * BOARD_SIZE + from_col;
    let to_sq: Square = to_row * BOARD_SIZE + to_col;

    Ok(Move::new(from_sq, to_sq))
}

impl PartialEq for Move {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::utils::{get_col, get_row};

    #[test]
    fn algebraic_move_parsing() {
        let mv = create_move_from_algebraic("a10a9").expect("parse");
        let from_sq = mv.from();
        let to_sq = mv.to();

        let from_row = get_row(from_sq);
        let to_row = get_row(to_sq);

        let from_col = get_col(from_sq);
        let to_col = get_col(to_sq);

        assert_eq!(from_col, 0);
        assert_eq!(from_row, 9);
        assert_eq!(to_col, 0);
        assert_eq!(to_row, 8);
    }

    #[test]
    fn pack_unpack_roundtrip() {
        let from_sq: Square = 37;
        let to_sq: Square = 99;
        let mv = Move::new(from_sq, to_sq);
        assert_eq!(mv.from(), from_sq);
        assert_eq!(mv.to(), to_sq);
    }

    #[test]
    fn bad_format_rejected() {
        assert!(create_move_from_algebraic("z10a9").is_err());
        assert!(create_move_from_algebraic("a0a1").is_err());
        assert!(create_move_from_algebraic("a12a1").is_err());
        assert!(create_move_from_algebraic("a1a").is_err());
        assert!(create_move_from_algebraic("a1  a2").is_err());
    }
}
