#![cfg(test)]

use std::error::Error;
use std::io::Cursor;

use crate::Engine;
use crate::board::Board;
use crate::board::rules::RulesEnum;
use crate::board::types::{Piece, Side, Square};
use crate::board::utils::{get_col, get_row, get_square_from_algebraic};
use crate::mv::Move;

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

    // The records are Copenhagen 11x11; a board of that variant reads their coordinates.
    let board = Board::from_rules(RulesEnum::Copenhagen11x11);
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

                let play = board
                    .create_move_from_algebraic(&format!("{from}{to}"))
                    .unwrap();
                record_taflzero.push(play);
            }
        }

        game_records.push(record_taflzero);
    }

    game_records
}

#[test]
fn play_outs() -> Result<(), Box<dyn Error>> {
    let mut client = Engine::new("copenhagen11x11.onnx".to_string());

    for mvs in aagenielsen_dk_game_records() {
        client.set_start_position();
        for mv in mvs {
            client.make_move(mv)?;
        }
    }

    Ok(())
}

fn expect_occupied(board: &Board, sq: Square) {
    let row = get_row(sq, 11);
    let col = get_col(sq, 11);

    assert_ne!(board.row_occ[row] & (1 << col), 0);
    assert_ne!(board.col_occ[col] & (1 << row), 0);
}

fn expect_not_occupied(board: &Board, sq: Square) {
    let row = get_row(sq, 11);
    let col = get_col(sq, 11);

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
    let sq = get_square_from_algebraic(sq_str, 11);
    assert_eq!(board.board[sq], Piece::DEFENDER);
    expect_defender_in_defenders_array(board, sq);
    expect_occupied(board, sq);
}

pub fn expect_attacker_on(board: &Board, sq_str: &str) {
    let sq = get_square_from_algebraic(sq_str, 11);
    assert_eq!(board.board[sq], Piece::ATTACKER);
    expect_attacker_in_attackers_array(board, sq);
    expect_occupied(board, sq);
}

pub fn expect_king_on(board: &Board, sq_str: &str) {
    let sq = get_square_from_algebraic(sq_str, 11);
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
    let sq = get_square_from_algebraic(sq_str, 11);
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

// Exact tactics at expansion: with the king one step from a corner, the escape is
// found when the root is expanded, before a single child has been visited.
#[test]
fn corner_escape_in_one_is_proven_at_expansion() {
    let mut client = Engine::new("copenhagen11x11.onnx".to_string());
    client.set_position_and_moves("11/11/5a5/11/11/11/11/11/11/11/1k9 - d 0 0", vec![]);

    let res = client.make_search_nodes(1, None);

    assert_eq!(client.board().move_to_algebraic(res.best_move), "b1a1");
}

// Game-length shaping: a win 100 plies away loses alpha/2 of its value, one 400 plies
// away loses the full alpha, and a net without the head (or alpha 0) changes nothing.
#[test]
fn moves_left_shaping_scales_with_predicted_length() {
    use crate::mcts::shaped_value;
    assert_eq!(shaped_value(0.8, Some(1.0), 0.15), 0.8 * (1.0 - 0.075));
    assert_eq!(shaped_value(-0.8, Some(4.0), 0.15), -0.8 * 0.85);
    assert_eq!(shaped_value(0.8, None, 0.15), 0.8);
    assert_eq!(shaped_value(0.8, Some(1.0), 0.0), 0.8);
}

// `result` reports the winner of the current UCI position without a search, so a
// match driver can stop a game on the engine's own rules.
#[test]
fn uci_result_reports_terminal_winner() {
    use crate::uci::{UciController, UciOutput};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Sink(Arc<Mutex<Vec<String>>>);
    impl UciOutput for Sink {
        fn send(&self, message: &str) {
            self.0.lock().unwrap().push(message.to_string());
        }
    }

    let out = Sink(Arc::new(Mutex::new(vec![])));
    let mut uci = UciController::new(out.clone(), "copenhagen11x11.onnx".to_string());
    let last = |out: &Sink| out.0.lock().unwrap().last().cloned().unwrap();

    uci.run("position startpos");
    uci.run("result");
    assert_eq!(last(&out), "result none");

    // King already on a corner: defenders have won.
    uci.run("position fen 11/11/5a5/11/11/11/11/11/11/11/k10 - a 0 0");
    uci.run("result");
    assert_eq!(last(&out), "result defenders");

    // No king on the board: attackers have won.
    uci.run("position fen 11/11/5a5/11/11/5d5/11/11/11/11/11 - d 0 0");
    uci.run("result");
    assert_eq!(last(&out), "result attackers");

    // Defenders to move with only a boxed-in king (no legal move): attackers have
    // won, and `go` must not answer with a null move (it used to say `bestmove a1a1`).
    uci.run("position fen 2a2aka3/1a2aaa1a2/1aa3a4/a1a8/a8a1/a10/a10/a10/a10/1a8a/4aaa4 - d 0 0");
    uci.run("result");
    assert_eq!(last(&out), "result attackers");
    uci.run("go nodes 10");
    assert_eq!(last(&out), "bestmove (none)");
}

// `randommove` returns a legal move that the engine then accepts.
#[test]
fn uci_randommove_is_legal() {
    use crate::uci::{UciController, UciOutput};
    use std::sync::{Arc, Mutex};

    #[derive(Clone)]
    struct Sink(Arc<Mutex<Vec<String>>>);
    impl UciOutput for Sink {
        fn send(&self, message: &str) {
            self.0.lock().unwrap().push(message.to_string());
        }
    }

    let out = Sink(Arc::new(Mutex::new(vec![])));
    let mut uci = UciController::new(out.clone(), "copenhagen11x11.onnx".to_string());
    uci.run("position startpos");
    let mut seen = std::collections::HashSet::new();
    for _ in 0..20 {
        uci.run("randommove");
        let line = out.0.lock().unwrap().last().cloned().unwrap();
        let mv = line.split_whitespace().nth(1).unwrap().to_string();
        assert_ne!(mv, "(none)");
        let legal: Vec<String> = uci
            .engine()
            .legal_moves()
            .iter()
            .map(|m| uci.engine().board().move_to_algebraic(*m))
            .collect();
        assert!(legal.contains(&mv), "{mv} not legal");
        seen.insert(mv);
    }
    assert!(seen.len() > 3, "random moves never vary");

    uci.run("position fen 11/11/5a5/11/11/11/11/11/11/11/k10 - a 0 0");
    uci.run("randommove");
    // Terminal position: the king is already on a corner, but attackers still have
    // moves; whatever is returned must parse.
    let line = out.0.lock().unwrap().last().cloned().unwrap();
    assert!(line.starts_with("randommove "));
}
