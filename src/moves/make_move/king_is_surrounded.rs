use crate::board::{Board, PRECOMPUTED};
use crate::board::types::Piece;

pub fn king_is_surrounded(board: &Board) -> bool {
    let king_sq = board.king_sq;

    if board.get_rules().is_king_strong && PRECOMPUTED.vertical_horizontal_neighbors.len() < 4 {
        return false;
    }

    let mut is_near_throne = king_sq as usize == PRECOMPUTED.throne_sq;

    let mut surround_count = 0;
    for sq in PRECOMPUTED.vertical_horizontal_neighbors[king_sq as usize].iter() {
        if *sq == PRECOMPUTED.throne_sq || board.board[*sq] == Piece::ATTACKER {
            surround_count += 1;
        }
    }

    if board.get_rules().is_king_strong {
        return surround_count >= 4
    }

    for sq in PRECOMPUTED.vertical_horizontal_neighbors[PRECOMPUTED.throne_sq].iter() {
        if *sq == king_sq as usize {
            is_near_throne = true;
        }
    }


    if is_near_throne {
        return surround_count >= 4
    }

    println!("here");

    let top_neighbor = PRECOMPUTED.top_neighbor[king_sq as usize];
    let bottom_neighbor = PRECOMPUTED.bottom_neighbor[king_sq as usize];
    let left_neighbor = PRECOMPUTED.left_neighbor[king_sq as usize];
    let right_neighbor = PRECOMPUTED.right_neighbor[king_sq as usize];

    if let Some(top) = top_neighbor {
        if let Some(bottom) = bottom_neighbor {
            if board.board[top] == Piece::ATTACKER && board.board[bottom] == Piece::ATTACKER {
                return true;
            }
        }
    }

    if let Some(left) = left_neighbor {
        if let Some(right) = right_neighbor {
            if board.board[left] == Piece::ATTACKER && board.board[right] == Piece::ATTACKER {
                return true;
            }
        }
    }

    false
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

    mod historical_variant {
        use crate::board::Board;
        use crate::board::types::Piece;
        use crate::board::utils::get_square_from_algebraic;
        use crate::rules::RulesEnum;
        use crate::set_board_from_str;
        use super::*;

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
                     ..........."
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
                     ..........."
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
                     ..........."
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
                     ..........."
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
                     ..........."
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
                     ..........."
            );

            let is_surrounded = king_is_surrounded(&board);
            assert!(!is_surrounded);
        }
    }
}