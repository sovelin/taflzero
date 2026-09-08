#![cfg(test)]

use std::error::Error;
use std::io::Cursor;

use crate::Engine;
use crate::board::Board;
use crate::board::types::{Piece, Side, Square};
use crate::board::utils::{get_col, get_row, get_square_from_algebraic};
use crate::mv::{Move, create_move_from_algebraic};

#[derive(Debug, serde::Deserialize)]
struct Record {
    moves: String,
    _attacker_captures: u64,
    _defender_captures: u64,
    _status: String,
}

pub fn aagenielsen_dk_game_records() -> Vec<Vec<Move>> {
    let copenhagen_csv = include_str!("copenhagen.csv");
    let cursor = Cursor::new(copenhagen_csv);

    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(false)
        .from_reader(cursor);

    let mut game_records = Vec::with_capacity(1_800);

    for result in rdr.deserialize() {
        let record: Record = result.unwrap();
        let mut record_taflzero = Vec::new();

        for play in record.moves.split_ascii_whitespace() {
            if play.contains('-') {
                let vertexes: Vec<_> = play.split('-').collect();
                let from = vertexes[0];
                let to_capture: Vec<_> = vertexes[1].split('x').collect();
                let to = to_capture[0];

                let play = create_move_from_algebraic(&format!("{from}{to}")).unwrap();
                record_taflzero.push(play);
            }
        }

        game_records.push(record_taflzero);
    }

    game_records
}

#[test]
fn play_outs() -> Result<(), Box<dyn Error>> {
    let mut client = Engine::new("default_nn.onnx".to_string());

    for mvs in aagenielsen_dk_game_records() {
        client.set_start_position();
        for mv in mvs {
            client.make_move(mv)?;
        }
    }

    Ok(())
}

fn expect_occupied(board: &Board, sq: Square) {
    let row = get_row(sq);
    let col = get_col(sq);

    assert_ne!(board.row_occ[row] & (1 << col), 0);
    assert_ne!(board.col_occ[col] & (1 << row), 0);
}

fn expect_not_occupied(board: &Board, sq: Square) {
    let row = get_row(sq);
    let col = get_col(sq);

    assert_eq!(board.row_occ[row] & (1 << col), 0);
    assert_eq!(board.col_occ[col] & (1 << row), 0);
}

pub fn expect_defender_in_defenders_array(board: &Board, sq: Square) {
    assert!(board.defenders_count > 0);
    assert!(board.defenders[..board.defenders_count as usize].contains(&sq));
}

pub fn expect_attacker_in_attackers_array(board: &Board, sq: Square) {
    assert!(board.attackers_count > 0);
    assert!(board.attackers[..board.attackers_count as usize].contains(&sq));
}

fn expect_defender_not_in_defenders_array(board: &Board, sq: Square) {
    if board.defenders_count == 0 {
        return;
    }

    assert!(!board.defenders[..board.defenders_count as usize].contains(&sq));
}

fn expect_attacker_not_in_attackers_array(board: &Board, sq: Square) {
    if board.attackers_count == 0 {
        return;
    }

    assert!(!board.attackers[..board.attackers_count as usize].contains(&sq));
}

pub fn expect_defender_on(board: &Board, sq_str: &str) {
    let sq = get_square_from_algebraic(sq_str);
    assert_eq!(board.board[sq], Piece::DEFENDER);
    expect_defender_in_defenders_array(board, sq);
    expect_occupied(board, sq);
}

pub fn expect_attacker_on(board: &Board, sq_str: &str) {
    let sq = get_square_from_algebraic(sq_str);
    assert_eq!(board.board[sq], Piece::ATTACKER);
    expect_attacker_in_attackers_array(board, sq);
    expect_occupied(board, sq);
}

pub fn expect_king_on(board: &Board, sq_str: &str) {
    let sq = get_square_from_algebraic(sq_str);
    assert_eq!(board.board[sq], Piece::KING);
    assert_eq!(board.king_sq as Square, sq);
    expect_occupied(board, sq);
}

pub fn expect_attackers_count(board: &Board, count: u8) {
    assert_eq!(board.attackers_count, count)
}

pub fn expect_defenders_count(board: &Board, count: u8) {
    assert_eq!(board.defenders_count, count)
}

pub fn expect_no_pice_on(board: &Board, sq_str: &str) {
    let sq = get_square_from_algebraic(sq_str);
    assert_eq!(board.board[sq], Piece::EMPTY);

    expect_not_occupied(board, sq);
    expect_attacker_not_in_attackers_array(board, sq);
    expect_defender_not_in_defenders_array(board, sq);
    assert_ne!(board.king_sq as Square, sq);
}

pub fn expect_king_not_exists(board: &Board) {
    assert_eq!(board.king_sq, crate::board::constants::HOLE);
}

pub fn expect_side_to_be(board: &Board, side: Side) {
    assert_eq!(board.side_to_move, side);
}
