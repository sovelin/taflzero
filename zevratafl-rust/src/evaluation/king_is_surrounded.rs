use crate::board::{Board, PRECOMPUTED};
use crate::board::types::Piece;
//use web_sys::console;

pub fn king_is_surrounded(board: &Board) -> bool {
    let king_sq = board.king_sq;

    // console::log_1(&format!("{}", board.get_fen()).into());
    // console::log_1(&format!("king_sq = {}", board.king_sq).into());


    if PRECOMPUTED.vertical_horizontal_neighbors.len() < 4 {
        return false;
    }

    let mut surround_count = 0;
    for sq in PRECOMPUTED.vertical_horizontal_neighbors[king_sq as usize].iter() {
        if *sq == PRECOMPUTED.throne_sq || board.board[*sq] == Piece::ATTACKER {
            surround_count += 1;
        }
    }


    surround_count >= 4
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;
    use super::*;

    #[test]
    fn test_king_is_surrounded() {
        let mut board = Board::new();
        // Set up a scenario where the king is surrounded
        board.set_piece(get_square_from_algebraic("b2"), Piece::KING).unwrap();
        assert_eq!(king_is_surrounded(&board), false);
    }

    #[test]
    fn surrounded_by_attackers_not_from_all_sides() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("b2"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER).unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_by_attackers_from_all_sides() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("b2"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("b3"), Piece::ATTACKER).unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(is_surrounded);
    }

    #[test]
    fn surrounded_by_defenders_from_all_sides() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("b2"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("a2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("b1"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("b3"), Piece::DEFENDER).unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_by_three_attackers_on_board_side() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("e1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("d1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("f1"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("e2"), Piece::ATTACKER).unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_by_three_attackers_and_throne() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("e6"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("d6"), Piece::ATTACKER).unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(is_surrounded);
    }

    #[test]
    fn surrounded_by_corner_and_by_corner() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("b1"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("b2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER).unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn on_throne_and_surrounded_by_four_attackers() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("f7"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("e6"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("g6"), Piece::ATTACKER).unwrap();
        println!("Board State:\n{:?}", board);

        let is_surrounded = king_is_surrounded(&board);
        assert!(is_surrounded);
    }
}