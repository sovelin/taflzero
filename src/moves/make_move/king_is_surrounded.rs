use crate::board::types::Piece;
use crate::board::{Board, PRECOMPUTED};

pub fn king_is_surrounded(board: &Board) -> bool {
    let king_sq = board.king_sq as usize;
    let rules = board.get_rules();

    if rules.is_king_strong {
        // Copenhagen: king captured when all 4 neighbors are hostile (attacker or throne)
        let surround_count = PRECOMPUTED.vertical_horizontal_neighbors[king_sq]
            .iter()
            .filter(|&&sq| sq == PRECOMPUTED.throne_sq || board.board[sq] == Piece::ATTACKER)
            .count();
        surround_count >= 4
    } else {
        // Historical: weak king
        // On throne: all 4 neighbors must be attackers
        // Off throne: 2 attackers on the same axis (left+right or top+bottom)
        let is_attacker =
            |sq_opt: Option<usize>| sq_opt.is_some_and(|sq| board.board[sq] == Piece::ATTACKER);

        if king_sq == PRECOMPUTED.throne_sq {
            PRECOMPUTED.vertical_horizontal_neighbors[king_sq]
                .iter()
                .all(|&sq| board.board[sq] == Piece::ATTACKER)
        } else {
            let left_right = is_attacker(PRECOMPUTED.left_neighbor[king_sq])
                && is_attacker(PRECOMPUTED.right_neighbor[king_sq]);
            let top_bottom = is_attacker(PRECOMPUTED.top_neighbor[king_sq])
                && is_attacker(PRECOMPUTED.bottom_neighbor[king_sq]);
            left_right || top_bottom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::Board;
    use crate::board::types::Piece;
    use crate::board::utils::get_square_from_algebraic;

    #[test]
    fn test_king_is_surrounded() {
        let mut board = Board::new();
        // Set up a scenario where the king is surrounded
        board
            .set_piece(get_square_from_algebraic("b2"), Piece::KING)
            .unwrap();
        assert_eq!(king_is_surrounded(&board), false);
    }

    #[test]
    fn surrounded_by_attackers_not_from_all_sides() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("b2"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER)
            .unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_by_attackers_from_all_sides() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("b2"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b3"), Piece::ATTACKER)
            .unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(is_surrounded);
    }

    #[test]
    fn surrounded_by_defenders_from_all_sides() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("b2"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("a2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b1"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b3"), Piece::DEFENDER)
            .unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_by_three_attackers_on_board_side() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("e1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d1"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f1"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e2"), Piece::ATTACKER)
            .unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn surrounded_by_three_attackers_and_throne() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("e6"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("d6"), Piece::ATTACKER)
            .unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(is_surrounded);
    }

    #[test]
    fn surrounded_by_corner_and_by_corner() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("b1"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b2"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER)
            .unwrap();

        let is_surrounded = king_is_surrounded(&board);
        assert!(!is_surrounded);
    }

    #[test]
    fn on_throne_and_surrounded_by_four_attackers() {
        let mut board = Board::new();
        board
            .set_piece(get_square_from_algebraic("f6"), Piece::KING)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f5"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("f7"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("e6"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("g6"), Piece::ATTACKER)
            .unwrap();
        println!("Board State:\n{:?}", board);

        let is_surrounded = king_is_surrounded(&board);
        assert!(is_surrounded);
    }

    mod historical_variant {
        use super::*;
        use crate::board::Board;
        use crate::board::rules::RulesEnum;
        use crate::board::set_board_from_str;
        use crate::board::types::Piece;
        use crate::board::utils::get_square_from_algebraic;

        #[test]
        fn surrounded_by_attackers_from_all_sides_on_throne() {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);

            set_board_from_str(
                &mut board,
                "...........
                     ...........
                     ...........
                     ...........
                     .....A.....
                     ....AKA....
                     .....A.....
                     ...........
                     ...........
                     ...........
                     ...........",
            );

            let is_surrounded = king_is_surrounded(&board);
            assert!(is_surrounded);
        }

        #[test]
        fn surrounded_by_attackers_from_3_sides_on_throne_is_not_win() {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);

            set_board_from_str(
                &mut board,
                "...........
                     ...........
                     ...........
                     ...........
                     .....D.....
                     ....AKA....
                     .....A.....
                     ...........
                     ...........
                     ...........
                     ...........",
            );

            let is_surrounded = king_is_surrounded(&board);
            assert!(!is_surrounded);
        }

        #[test]
        fn surrounded_by_attackers_from_3_sides_and_throne_is_win() {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);

            set_board_from_str(
                &mut board,
                "...........
                     ...........
                     ...........
                     .....A.....
                     ....AKA....
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........",
            );

            let is_surrounded = king_is_surrounded(&board);
            assert!(is_surrounded);
        }

        #[test]
        fn surrounded_by_attackers_from_2_horizontal_sides_on_other_sq_is_win() {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);

            set_board_from_str(
                &mut board,
                "...........
                     ..D........
                     .AKA.......
                     ..D........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........",
            );

            let is_surrounded = king_is_surrounded(&board);
            assert!(is_surrounded);
        }

        #[test]
        fn surrounded_by_attackers_from_2_vertical_sides_on_other_sq_is_win() {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);

            set_board_from_str(
                &mut board,
                "...........
                     ..A........
                     .DKD.......
                     ..A........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........",
            );

            let is_surrounded = king_is_surrounded(&board);
            assert!(is_surrounded);
        }

        #[test]
        fn surrounded_by_attack0ers_from_2_different_sides_is_not_win() {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);

            set_board_from_str(
                &mut board,
                "...........
                     ..A........
                     .AKD.......
                     ..D........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........
                     ...........",
            );

            let is_surrounded = king_is_surrounded(&board);
            assert!(!is_surrounded);
        }
    }
}
