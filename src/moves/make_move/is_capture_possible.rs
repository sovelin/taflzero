use crate::board::types::{Piece, Square};
use crate::board::{Board, PRECOMPUTED, get_side_by_piece};
use std::sync::LazyLock;

pub static HOSTILE_CORNERS_SQS: LazyLock<[usize; 5]> = LazyLock::new(|| {
    [
        PRECOMPUTED.corners_sq[0],
        PRECOMPUTED.corners_sq[1],
        PRECOMPUTED.corners_sq[2],
        PRECOMPUTED.corners_sq[3],
        PRECOMPUTED.throne_sq,
    ]
});

pub static NOT_HOSTILE_SQS: LazyLock<[usize; 1]> = LazyLock::new(|| [PRECOMPUTED.throne_sq]);

pub fn is_potential_thread(board: &Board, target_sq: Square, enemy_sq: Square) -> bool {
    let is_hostile_corners = board.get_rules().is_corners_hostile;

    if board.board[enemy_sq] == Piece::EMPTY
        && ((is_hostile_corners && HOSTILE_CORNERS_SQS.contains(&enemy_sq))
            || (!is_hostile_corners && NOT_HOSTILE_SQS.contains(&enemy_sq)))
    {
        return true;
    }

    if board.board[enemy_sq] == Piece::EMPTY {
        return false;
    }

    let target_side = get_side_by_piece(board.board[target_sq]);
    let enemy_side = get_side_by_piece(board.board[enemy_sq]);

    target_side != enemy_side
}

pub fn is_capture_possible(
    board: &Board,
    target_sq: Square,
    enemy_sq_1: Square,
    enemy_sq_2: Square,
) -> bool {
    println!("Checking capture: target_sq={}, enemy_sq_1={}, enemy_sq_2={}", target_sq, enemy_sq_1, enemy_sq_2);
    if board.rules.rules().is_king_strong && board.board[target_sq] == Piece::KING || board.board[target_sq] == Piece::EMPTY {
        return false;
    }

    let is_threat_1 = is_potential_thread(board, target_sq, enemy_sq_1);
    let is_threat_2 = is_potential_thread(board, target_sq, enemy_sq_2);

    is_threat_1 && is_threat_2
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::board::board::Board;
    use crate::board::rules::RulesEnum;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;

    #[test]
    fn attacker_capture_defender_simple() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c3"), Piece::DEFENDER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));

        Ok(())
    }

    #[test]
    fn defender_capture_attacker_simple() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("c4"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("c3"), Piece::ATTACKER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));

        Ok(())
    }

    #[test]
    fn king_can_capture_attacker_simple() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("c4"), Piece::KING)?;
        board.set_piece(get_square_from_algebraic("c3"), Piece::ATTACKER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));

        Ok(())
    }

    #[test]
    fn attacker_cannot_capture_king_simple() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c3"), Piece::KING)?;

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));

        Ok(())
    }

    // --- Corners ---

    #[test]
    fn king_can_capture_on_corners() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("a3"), Piece::KING)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));

        Ok(())
    }

    #[test]
    fn king_can_capture_on_corners_swapped_params() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("a3"), Piece::KING)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a3"),
            get_square_from_algebraic("a1"),
        ));

        Ok(())
    }

    #[test]
    fn defender_can_capture_on_corners() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("a3"), Piece::DEFENDER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));

        Ok(())
    }

    #[test]
    fn attacker_can_capture_defender_on_corners() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("a3"), Piece::ATTACKER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));

        Ok(())
    }

    #[test]
    fn attacker_can_capture_defender_on_other_corner() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("k10"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("k9"), Piece::ATTACKER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("k10"),
            get_square_from_algebraic("k11"),
            get_square_from_algebraic("k9"),
        ));

        Ok(())
    }

    // --- Near throne ---

    #[test]
    fn king_can_capture_near_throne() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("f4"), Piece::KING)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));

        Ok(())
    }

    #[test]
    fn attacker_can_capture_defender_near_throne() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f5"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("f4"), Piece::ATTACKER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));

        Ok(())
    }

    #[test]
    fn defender_can_capture_attacker_near_throne() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER)?;

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));

        Ok(())
    }

    // --- Negative cases / special rules ---

    #[test]
    fn king_cannot_capture_defender_on_corner_because_same_side_rule() -> Result<(), Box<dyn Error>>
    {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("a2"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("a3"), Piece::KING)?;

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("a2"),
            get_square_from_algebraic("a1"),
            get_square_from_algebraic("a3"),
        ));

        Ok(())
    }

    #[test]
    fn no_capture_when_both_sides_not_same() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c4"), Piece::DEFENDER)?;
        board.set_piece(get_square_from_algebraic("c3"), Piece::DEFENDER)?;

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));

        Ok(())
    }

    #[test]
    fn no_capture_if_all_pieces_same() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c3"), Piece::ATTACKER)?;

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));

        Ok(())
    }

    #[test]
    fn no_capture_when_between_is_empty() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER)?;
        board.set_piece(get_square_from_algebraic("c4"), Piece::ATTACKER)?;

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("c3"),
            get_square_from_algebraic("c2"),
            get_square_from_algebraic("c4"),
        ));

        Ok(())
    }

    #[test]
    fn no_capture_when_king_on_throne_and_trying_to_capture_defender() -> Result<(), Box<dyn Error>>
    {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING)?; // throne
        board.set_piece(get_square_from_algebraic("f5"), Piece::DEFENDER)?; // target
        board.set_piece(get_square_from_algebraic("f4"), Piece::ATTACKER)?; // second side

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));

        Ok(())
    }

    #[test]
    fn capture_when_king_on_throne_captures_attacker_with_help_of_defender()
    -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING)?; // throne
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER)?; // target
        board.set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER)?; // second side

        assert!(is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"),
        ));

        Ok(())
    }

    #[test]
    fn no_capture_when_king_on_throne_but_no_second_defender() -> Result<(), Box<dyn Error>> {
        let mut board = Board::new();
        board.set_piece(get_square_from_algebraic("f6"), Piece::KING)?; // throne
        board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER)?; // target

        assert!(!is_capture_possible(
            &board,
            get_square_from_algebraic("f5"),
            get_square_from_algebraic("f6"),
            get_square_from_algebraic("f4"), // empty square, no defender
        ));

        Ok(())
    }

    mod historical_variant {
        use super::*;
        use crate::board::rules::RulesEnum;

        #[test]
        fn no_capture_on_first_corner() -> Result<(), Box<dyn Error>> {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);
            board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER)?;
            board.set_piece(get_square_from_algebraic("a3"), Piece::DEFENDER)?;

            assert!(!is_capture_possible(
                &board,
                get_square_from_algebraic("a2"),
                get_square_from_algebraic("a1"),
                get_square_from_algebraic("a3"),
            ));

            Ok(())
        }

        #[test]
        fn no_capture_on_second_corner() -> Result<(), Box<dyn Error>> {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);
            board.set_piece(get_square_from_algebraic("k10"), Piece::ATTACKER)?;
            board.set_piece(get_square_from_algebraic("k9"), Piece::DEFENDER)?;

            assert!(!is_capture_possible(
                &board,
                get_square_from_algebraic("k10"),
                get_square_from_algebraic("k11"),
                get_square_from_algebraic("k9"),
            ));

            Ok(())
        }

        #[test]
        fn no_capture_on_third_corner() -> Result<(), Box<dyn Error>> {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);
            board.set_piece(get_square_from_algebraic("a10"), Piece::DEFENDER)?;
            board.set_piece(get_square_from_algebraic("a9"), Piece::ATTACKER)?;

            assert!(!is_capture_possible(
                &board,
                get_square_from_algebraic("a10"),
                get_square_from_algebraic("a11"),
                get_square_from_algebraic("a9"),
            ));

            Ok(())
        }

        #[test]
        fn no_capture_on_fourth_corner() -> Result<(), Box<dyn Error>> {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);
            board.set_piece(get_square_from_algebraic("k2"), Piece::DEFENDER)?;
            board.set_piece(get_square_from_algebraic("k3"), Piece::ATTACKER)?;

            assert!(!is_capture_possible(
                &board,
                get_square_from_algebraic("k2"),
                get_square_from_algebraic("k1"),
                get_square_from_algebraic("k3"),
            ));

            Ok(())
        }

        #[test]
        fn no_capture_king_on_corner() -> Result<(), Box<dyn Error>> {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);
            board.set_piece(get_square_from_algebraic("k2"), Piece::KING)?;
            board.set_piece(get_square_from_algebraic("k3"), Piece::ATTACKER)?;

            assert!(!is_capture_possible(
                &board,
                get_square_from_algebraic("k2"),
                get_square_from_algebraic("k1"),
                get_square_from_algebraic("k3"),
            ));

            Ok(())
        }

        #[test]
        fn capture_where_throne() -> Result<(), Box<dyn Error>> {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);
            board.set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER)?;
            board.set_piece(get_square_from_algebraic("f4"), Piece::DEFENDER)?;

            assert!(is_capture_possible(
                &board,
                get_square_from_algebraic("f5"),
                get_square_from_algebraic("f6"), // throne, but can be used for capture in this variant
                get_square_from_algebraic("f4"),
            ));

            Ok(())
        }
    }
}
