use crate::board::{Board, PRECOMPUTED};
use crate::board::constants::HOLE;
use crate::board::types::{Piece, Square};
use crate::board::utils::is_edge_square;
use crate::utils::bfs::bfs;

fn is_calculate_needed(board: &Board) -> bool {
    if board.last_move_to == HOLE {
        return true;
    }

    if board.board[board.last_move_to as usize] != Piece::ATTACKER {
        return false;
    }

    if board.was_capture {
        return true;
    }

    let all_neighbors = &PRECOMPUTED.all_neighbors[board.last_move_to as usize];

    let mut attackers_nearby_count = 0;

    for &neighbor in all_neighbors.iter() {
        if board.board[neighbor] == Piece::ATTACKER {
            attackers_nearby_count += 1;
        }

        if attackers_nearby_count >= 2 {
            return true;
        }
    }

    false
}

pub fn defenders_is_surrounded(board: &Board) -> bool {
    if !is_calculate_needed(board) {
        return false;
    }

    let mut start_sqs = vec![];

    if board.king_sq != HOLE {
        start_sqs.push(board.king_sq as Square);
    }

    for i in 0..board.defenders_count {
        let sq = board.defenders[i as usize];
        start_sqs.push(sq);
    }

    let inner_squares = bfs(
        |sq| {
            board.board[sq] != Piece::ATTACKER
        },
        &PRECOMPUTED.vertical_horizontal_neighbors,
        &start_sqs,
    );

    for i in 0..inner_squares.len() {
        if inner_squares[i] && is_edge_square(i as Square) {
            return false
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;
    use crate::mv::create_move_from_algebraic;
    use super::defenders_is_surrounded;

    #[test]
    fn by_default_not_surrounded() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("d4"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_corners_but_with_open_edges_not_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("j1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("k2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("j11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("k10"), Piece::ATTACKER).unwrap();

        board.set_piece(get_square_from_algebraic("d4"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_corners_and_edges_no_escape_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c9"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b9"), Piece::ATTACKER).unwrap();

        board.set_piece(get_square_from_algebraic("b10"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(is_surrounded);
    }

    #[test]
    fn surrounded_corners_and_edges_one_defender_outside_not_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c9"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b9"), Piece::ATTACKER).unwrap();

        board.set_piece(get_square_from_algebraic("b10"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("j1"), Piece::DEFENDER).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_corners_and_edges_but_defender_in_chain_not_surrounded() {
        let mut board = Board::new();

        board.set_piece(get_square_from_algebraic("a10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d10"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c9"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b9"), Piece::DEFENDER).unwrap();

        board.set_piece(get_square_from_algebraic("b10"), Piece::KING).unwrap();

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn not_surrounded_if_at_leadt_one_piece_could_go_to_edge() {
        let mut board = Board::new();
    board.set_fen("2a8/11/aa3aa2a1/1da2a5/ad1aaa5/1aa1k1a2aa/3a1aa3a/1a2a6/11/11/11 d");

        let is_surrounded = defenders_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn edge_case_when_capture_happen_with_surrounding() {
        let mut board = Board::new();
        board.set_fen("11/11/3aa5a/2akdaaa1a1/3a1d1da1d/4a2d1aa/4ad1daa1/2aaadda3/5aa4/11/6a4 a");
        // make move
        board.make_move_simple(create_move_from_algebraic("k9k8").unwrap()).unwrap();
        let is_surrounded = defenders_is_surrounded(&board);
        assert!(is_surrounded);
    }
}
