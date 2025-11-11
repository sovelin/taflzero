use std::sync::LazyLock;
use crate::board::{get_side_by_piece, Board, PRECOMPUTED};
use crate::board::types::{Piece, Square};

pub static ENEMY_SQUARES: LazyLock<[usize; 5]> = LazyLock::new(|| {
    [
        PRECOMPUTED.corners_sq[0],
        PRECOMPUTED.corners_sq[1],
        PRECOMPUTED.corners_sq[2],
        PRECOMPUTED.corners_sq[3],
        PRECOMPUTED.throne_sq,
    ]
});

pub fn is_potential_thread(board: &Board, target_sq: Square, enemy_sq: Square) -> bool {
    if board.board[enemy_sq] == Piece::EMPTY && ENEMY_SQUARES.contains(&enemy_sq) {
        return true;
    }

    if board.board[enemy_sq] == Piece::EMPTY {
        return false;
    }


    let target_side = get_side_by_piece(board.board[target_sq]);
    let enemy_side = get_side_by_piece(board.board[enemy_sq]);

    target_side != enemy_side
}

pub fn is_capture_possible(board: &Board, target_sq: Square, enemy_sq_1: Square, enemy_sq_2: Square) -> bool {
    if board.board[target_sq] == Piece::KING || board.board[target_sq] == Piece::EMPTY {
        return false;
    }


    let is_threat_1 = is_potential_thread(board, target_sq, enemy_sq_1);
    let is_threat_2 = is_potential_thread(board, target_sq, enemy_sq_2);

    is_threat_1 && is_threat_2
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::board::Board;
    use crate::board::types::{Piece, Side, Square};
    use crate::board::utils::get_square_from_algebraic;

    #[test]
    fn attacker_capture_defender_simple() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c3"), Piece::DEFENDER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));
    }

    #[test]
    fn defender_capture_attacker_simple() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("c4"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("c3"), Piece::ATTACKER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));
    }

    #[test]
    fn king_can_capture_attacker_simple() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("c4"), Piece::KING);
        board.set_piece(get_square_from_algebraic("c3"), Piece::ATTACKER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));
    }

    #[test]
    fn attacker_cannot_capture_king_simple() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c3"), Piece::KING);

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));
    }

    // --- Corners ---

    #[test]
    fn king_can_capture_on_corners() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("a3"), Piece::KING);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));
    }

    #[test]
    fn king_can_capture_on_corners_swapped_params() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("a3"), Piece::KING);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a3"),
            get_square_from_algebraic("a1"),
        ));
    }

    #[test]
    fn defender_can_capture_on_corners() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("a3"), Piece::DEFENDER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));
    }

    #[test]
    fn attacker_can_capture_defender_on_corners() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("a3"), Piece::ATTACKER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));
    }

    #[test]
    fn attacker_can_capture_defender_on_other_corner() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("k10"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("k9"), Piece::ATTACKER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("k10"),
            get_square_from_algebraic("k11"),
            get_square_from_algebraic("k9"),
        ));
    }

    // --- Near throne ---

    #[test]
    fn king_can_capture_near_throne() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("f4"), Piece::KING);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));
    }

    #[test]
    fn attacker_can_capture_defender_near_throne() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f5"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("f4"), Piece::ATTACKER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));
    }

    #[test]
    fn defender_can_capture_attacker_near_throne() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER);

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));
    }

    // --- Negative cases / special rules ---

    #[test]
    fn king_cannot_capture_defender_on_corner_because_same_side_rule() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("a3"), Piece::KING);

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));
    }

    #[test]
    fn no_capture_when_both_sides_not_same() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c4"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("c3"), Piece::DEFENDER);

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));
    }

    #[test]
    fn no_capture_if_all_pieces_same() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c3"), Piece::ATTACKER);

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));
    }

    #[test]
    fn no_capture_when_between_is_empty() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER);
        // c3 — пусто

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));
    }

    #[test]
    fn no_capture_when_king_on_throne_and_trying_to_capture_defender() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING);     // трон
        board.set_piece(get_square_from_algebraic("f5"), Piece::DEFENDER); // цель
        board.set_piece(get_square_from_algebraic("f4"), Piece::ATTACKER); // вторая сторона

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));
    }

    #[test]
    fn capture_when_king_on_throne_captures_attacker_with_help_of_defender() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING);     // трон
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER); // цель
        board.set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER); // вторая сторона

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));
    }

    #[test]
    fn no_capture_when_king_on_throne_but_no_second_defender() {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING);     // трон
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER); // цель
        // нет второй фигуры, которая замкнет захват

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"), // пусто — не считается угрозой, т.к. не трон/угол
        ));
    }
}
