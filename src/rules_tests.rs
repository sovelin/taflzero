#![cfg(test)]

//! Rule tests grouped by variant.
//!
//! These sit apart from the per-module unit tests because a "rule" is rarely owned by
//! one module: the throne, for instance, is a movegen restriction, a capture helper and
//! a terminal condition at once. Keeping a variant's rules in one place also makes the
//! file readable as a specification of that variant.

use crate::board::Board;
use crate::board::rules::RulesEnum;
use crate::board::types::{Piece, Side};
use crate::mv::Move;

fn board_of(variant: RulesEnum) -> Board {
    Board::from_rules(variant)
}

/// Destination squares the side to move can reach, in algebraic form.
fn destinations_from(board: &Board, from: &str) -> Vec<String> {
    let mut move_gen = crate::movegen::MoveGen::new();
    move_gen.generate_moves(board);

    let from_sq = board.get_square_from_algebraic(from);
    move_gen
        .moves()
        .iter()
        .filter(|mv| mv.from() == from_sq)
        .map(|mv| board.get_sq_algebraic(mv.to()))
        .collect()
}

fn place(board: &mut Board, squares: &[(&str, Piece)]) {
    for (sq, piece) in squares {
        let sq = board.get_square_from_algebraic(sq);
        board.set_piece(sq, *piece).expect("set_piece");
    }
}

fn play(board: &mut Board, mv: &str) {
    let mv: Move = board.create_move_from_algebraic(mv).expect("parse move");
    board.make_move_simple(mv).expect("make move");
}

/// Sets up a position, plays one attacker move, and reports whether the king fell.
/// The move is always the piece that *completes* the surround — capture rules care
/// which move closed it, so pre-placing the last attacker would test something else.
fn king_survives_after(variant: RulesEnum, pieces: &[(&str, Piece)], mv: &str) -> bool {
    let mut board = board_of(variant);
    place(&mut board, pieces);
    assert_ne!(board.king_sq, -1, "test setup must place a king");
    play(&mut board, mv);
    board.king_sq != -1
}

mod tablut {
    use super::*;

    fn tablut() -> Board {
        board_of(RulesEnum::Tablut9x9)
    }

    #[test]
    fn initial_position_is_canonical() {
        let mut board = tablut();
        board.setup_initial_position().expect("initial position");

        assert_eq!(board.board_size(), 9);
        assert_eq!(board.attackers_count, 16, "attackers");
        assert_eq!(board.defenders_count, 8, "defenders");
        assert_eq!(
            board.king_sq as usize,
            board.precomputed().throne_sq,
            "king starts on the throne"
        );
        assert_eq!(board.side_to_move, Side::ATTACKERS);
    }

    mod throne {
        use super::*;

        #[test]
        fn regular_piece_cannot_land_on_it() {
            let mut board = tablut();
            place(&mut board, &[("e1", Piece::ATTACKER), ("a9", Piece::KING)]);

            let dests = destinations_from(&board, "e1");
            assert!(
                !dests.contains(&"e5".to_string()),
                "an attacker must not occupy the throne, got {dests:?}"
            );
        }

        #[test]
        fn regular_piece_may_slide_over_it() {
            let mut board = tablut();
            place(&mut board, &[("e1", Piece::ATTACKER), ("a9", Piece::KING)]);

            let dests = destinations_from(&board, "e1");
            assert!(
                dests.contains(&"e9".to_string()),
                "sliding across the empty throne is allowed, got {dests:?}"
            );
        }

        // Regression: the throne used to be blocked by a hardcoded 11x11 bit index, so on
        // a 9x9 board f6 was forbidden and the real throne at e5 was free.
        #[test]
        fn blocks_the_throne_and_nothing_else() {
            let mut board = tablut();
            place(&mut board, &[("f1", Piece::ATTACKER), ("a9", Piece::KING)]);

            let dests = destinations_from(&board, "f1");
            assert!(
                dests.contains(&"f6".to_string()),
                "f6 is an ordinary square, got {dests:?}"
            );
        }

        #[test]
        fn king_may_return_to_it() {
            let mut board = tablut();
            place(&mut board, &[("e1", Piece::KING)]);
            board.set_side(Side::DEFENDERS);

            let dests = destinations_from(&board, "e1");
            assert!(
                dests.contains(&"e5".to_string()),
                "the king may stand on the throne, got {dests:?}"
            );
        }
    }

    mod king_capture {
        use super::*;

        #[test]
        fn two_attackers_suffice_away_from_the_throne() {
            let mut board = tablut();
            place(
                &mut board,
                &[
                    ("b2", Piece::KING),
                    ("a2", Piece::ATTACKER),
                    ("c1", Piece::ATTACKER),
                ],
            );

            play(&mut board, "c1c2");
            assert_eq!(board.king_sq, -1, "weak king falls to a plain sandwich");
        }

        #[test]
        fn two_attackers_are_not_enough_beside_the_empty_throne() {
            let mut board = tablut();
            place(
                &mut board,
                &[
                    ("e4", Piece::KING),
                    ("d4", Piece::ATTACKER),
                    ("f1", Piece::ATTACKER),
                ],
            );

            play(&mut board, "f1f4");
            assert_ne!(
                board.king_sq, -1,
                "the empty throne stands in for the fourth side, so three are needed"
            );
        }

        #[test]
        fn three_attackers_take_the_king_beside_the_empty_throne() {
            let mut board = tablut();
            place(
                &mut board,
                &[
                    ("e4", Piece::KING),
                    ("d4", Piece::ATTACKER),
                    ("f4", Piece::ATTACKER),
                    ("e1", Piece::ATTACKER),
                ],
            );

            play(&mut board, "e1e3");
            assert_eq!(board.king_sq, -1, "three sides plus the throne");
        }

        #[test]
        fn king_on_the_throne_needs_all_four() {
            let mut board = tablut();
            place(
                &mut board,
                &[
                    ("e5", Piece::KING),
                    ("d5", Piece::ATTACKER),
                    ("f5", Piece::ATTACKER),
                    ("e6", Piece::ATTACKER),
                    ("e1", Piece::ATTACKER),
                ],
            );

            play(&mut board, "e1e3");
            assert_ne!(
                board.king_sq, -1,
                "three sides are not enough on the throne"
            );

            board.set_side(Side::ATTACKERS);
            place(&mut board, &[("a4", Piece::ATTACKER)]);
            play(&mut board, "a4e4");
            assert_eq!(board.king_sq, -1, "the fourth side completes the capture");
        }
    }

    mod terminal {
        use super::*;
        use crate::terminal::check_terminal;

        #[test]
        fn king_wins_in_a_corner() {
            let mut board = tablut();
            place(&mut board, &[("a1", Piece::KING)]);

            assert_eq!(check_terminal(&mut board), Some(Side::DEFENDERS));
        }

        #[test]
        fn a_plain_edge_square_is_not_a_win() {
            let mut board = tablut();
            place(&mut board, &[("e1", Piece::KING)]);

            assert_eq!(
                check_terminal(&mut board),
                None,
                "only the four corners are escape squares — reaching the edge is not enough"
            );
        }

        /// Corners are hostile, so they close a sandwich the way an attacker would.
        /// This is the other half of `has_corners_win`: an empty corner is not neutral
        /// ground for a defender standing next to it.
        #[test]
        fn a_corner_closes_a_sandwich() {
            let mut board = tablut();
            place(
                &mut board,
                &[("b1", Piece::DEFENDER), ("d1", Piece::ATTACKER)],
            );

            play(&mut board, "d1c1");

            let b1 = board.get_square_from_algebraic("b1");
            assert_eq!(
                board.board[b1],
                Piece::EMPTY,
                "b1 is sandwiched between the attacker on c1 and the hostile corner a1"
            );
        }

        #[test]
        fn king_in_the_middle_is_not_terminal() {
            let mut board = tablut();
            place(&mut board, &[("e4", Piece::KING)]);

            assert_eq!(check_terminal(&mut board), None);
        }
    }
}

/// Copenhagen has a strong king: four hostile sides are needed. These tests exist to
/// pin that down while the weak-king path next door is being changed.
mod copenhagen {
    use super::*;

    const V: RulesEnum = RulesEnum::Copenhagen11x11;

    #[test]
    fn two_attackers_leave_the_king_alone() {
        assert!(
            king_survives_after(
                V,
                &[
                    ("f4", Piece::KING),
                    ("e4", Piece::ATTACKER),
                    ("g1", Piece::ATTACKER)
                ],
                "g1g4",
            ),
            "a plain sandwich must not touch a strong king"
        );
    }

    #[test]
    fn three_attackers_are_not_enough() {
        assert!(
            king_survives_after(
                V,
                &[
                    ("f4", Piece::KING),
                    ("e4", Piece::ATTACKER),
                    ("g4", Piece::ATTACKER),
                    ("f1", Piece::ATTACKER),
                ],
                "f1f3",
            ),
            "three sides still leave one open"
        );
    }

    #[test]
    fn four_attackers_capture() {
        assert!(
            !king_survives_after(
                V,
                &[
                    ("f4", Piece::KING),
                    ("e4", Piece::ATTACKER),
                    ("g4", Piece::ATTACKER),
                    ("f3", Piece::ATTACKER),
                    ("f8", Piece::ATTACKER),
                ],
                "f8f5",
            ),
            "all four sides closed"
        );
    }
}

/// Historical shares the weak-king rules with Tablut, only on an 11x11 board, so the
/// same three cases must hold there too.
mod historical {
    use super::*;

    const V: RulesEnum = RulesEnum::Historical11x11;

    #[test]
    fn two_attackers_capture_away_from_the_throne() {
        assert!(
            !king_survives_after(
                V,
                &[
                    ("c3", Piece::KING),
                    ("b3", Piece::ATTACKER),
                    ("d1", Piece::ATTACKER)
                ],
                "d1d3",
            ),
            "weak king falls to a plain sandwich"
        );
    }

    #[test]
    fn two_attackers_are_not_enough_beside_the_empty_throne() {
        assert!(
            king_survives_after(
                V,
                &[
                    ("f5", Piece::KING),
                    ("e5", Piece::ATTACKER),
                    ("g1", Piece::ATTACKER)
                ],
                "g1g5",
            ),
            "the empty throne stands in for the fourth side"
        );
    }

    #[test]
    fn three_attackers_capture_beside_the_empty_throne() {
        assert!(
            !king_survives_after(
                V,
                &[
                    ("f5", Piece::KING),
                    ("e5", Piece::ATTACKER),
                    ("g5", Piece::ATTACKER),
                    ("f1", Piece::ATTACKER),
                ],
                "f1f4",
            ),
            "three sides plus the throne"
        );
    }

    #[test]
    fn king_on_the_throne_needs_all_four() {
        assert!(
            king_survives_after(
                V,
                &[
                    ("f6", Piece::KING),
                    ("e6", Piece::ATTACKER),
                    ("g6", Piece::ATTACKER),
                    ("f1", Piece::ATTACKER),
                ],
                "f1f5",
            ),
            "three sides are not enough on the throne"
        );
    }
}

/// A position taken from a real game, kept because it looks like a kill and is not one.
///
/// ```text
///  9 . . . . . . . . .
///  8 . . . A . . . . .
///  7 . . . D A A . . .
///  6 A . D K A . . . .
///  5 D . A A . A . . .
///  4 . . A . . . . . .
///  3 . A . . . . . . .
///  2 . . . . . . . . .
///  1 . . . . . . . . .
/// ```
mod real_game_position {
    use super::*;
    use crate::board::set_board_from_str;
    use crate::terminal::check_terminal;

    const POSITION: &str = ".........
                            ...A.....
                            ...DAA...
                            A.DKA....
                            D.AA.A...
                            ..A......
                            .A.......
                            .........
                            .........";

    fn position() -> Board {
        let mut board = board_of(RulesEnum::Tablut9x9);
        set_board_from_str(&mut board, POSITION);
        board
    }

    #[test]
    fn is_not_terminal_for_either_side() {
        for side in [Side::ATTACKERS, Side::DEFENDERS] {
            let mut board = position();
            board.set_side(side);
            assert_eq!(check_terminal(&mut board), None, "with {side:?} to move");
        }
    }

    // The king has an attacker on two of its four sides, but on different axes — e6 to
    // the right and d5 below. Its own defenders hold c6 and d7, so neither axis can be
    // closed while they stand.
    #[test]
    fn two_attackers_on_different_axes_leave_the_king_alone() {
        let board = position();
        let king = board.king_sq as usize;
        assert_eq!(board.get_sq_algebraic(king), "d6");

        let attacker_neighbours: Vec<String> = board.precomputed().vertical_horizontal_neighbors
            [king]
            .iter()
            .filter(|&&sq| board.board[sq] == Piece::ATTACKER)
            .map(|&sq| board.get_sq_algebraic(sq))
            .collect();
        assert_eq!(attacker_neighbours.len(), 2, "two sides are attacked");

        let mut board = position();
        assert_eq!(check_terminal(&mut board), None, "yet the king stands");
    }

    // The one terminal condition that depends on history rather than on the pieces: the
    // same position seen a third time ends the game, and in tafl it ends it for the
    // attackers. This is how a position like the one above shows up as terminal in a
    // self-play log while looking perfectly alive on the board — and why the game is
    // then thrown away instead of stored, since repetitions carry no outcome worth
    // learning from.
    #[test]
    fn a_third_repetition_ends_it_for_the_attackers() {
        let mut board = position();
        board.set_side(Side::DEFENDERS);
        assert_eq!(check_terminal(&mut board), None, "first visit");

        board.rep_table.insert(board.zobrist, 3);
        assert_eq!(check_terminal(&mut board), Some(Side::ATTACKERS));
    }

    // `defenders_is_surrounded` asks whether the group can still reach the border. Here
    // a5 is a defender sitting on the border itself, so the answer is trivially yes.
    #[test]
    fn defenders_are_not_surrounded_while_one_stands_on_the_border() {
        let mut board = position();
        board.set_side(Side::ATTACKERS);

        assert_eq!(
            check_terminal(&mut board),
            None,
            "the group touches the edge at a5"
        );
    }
}
