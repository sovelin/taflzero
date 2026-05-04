pub mod is_capture_possible;
mod mask_shield_captures;
pub mod king_is_surrounded;

use crate::board::{Board, PRECOMPUTED};
use crate::board::types::OptionalSquare;
use crate::moves::make_move::is_capture_possible::is_capture_possible;
use crate::moves::make_move::king_is_surrounded::king_is_surrounded;
use crate::moves::make_move::mask_shield_captures::make_shield_wall_captures;
use crate::moves::mv::Move;
use crate::moves::undo::{CapturedPiece, UndoMove};
use crate::types::{Piece, Side, Square};

impl Board {
    fn add_position_to_rep_table(&mut self) {
        if self.rep_table.contains_key(&self.zobrist) {
            self.rep_table.insert(self.zobrist, self.rep_table.get(&self.zobrist).unwrap() + 1);
        } else {
            self.rep_table.insert(self.zobrist, 1);
        }
    }

    pub fn make_move(&mut self, mv: Move, undo: &mut UndoMove) -> Result<(), &'static str> {
        undo.clear_captured_pieces();
        undo.last_move_to = self.last_move_to;
        undo.was_capture = false;
        undo.mv = mv;

        let from = mv.from();
        let to = mv.to();
        let piece = self.board[from as usize];

        self.clear_piece(from);
        self.set_piece(to, piece)?;


        for sandwich in PRECOMPUTED.sandwich_captures[to as usize].iter() {
            if is_capture_possible(&self, sandwich.between_sq, to, sandwich.captor_sq) {
                undo.add_captured_piece(CapturedPiece {
                    square: sandwich.between_sq,
                    piece: self.board[sandwich.between_sq as usize],
                });
                self.clear_piece(sandwich.between_sq);
            }
        }

        self.last_move_to = to as OptionalSquare;

        if self.get_rules().has_shield_walls {
            make_shield_wall_captures(self, to, undo);
        }


        // Capture king special handling
        if self.king_sq != -1 && self.side_to_move == Side::ATTACKERS && PRECOMPUTED.vertical_horizontal_neighbors[self.king_sq as usize].contains(
            &to
        ) && king_is_surrounded(self) {
            // Capture the king
            undo.add_captured_piece(CapturedPiece {
                square: self.king_sq as Square,
                piece: Piece::KING,
            });

            self.clear_piece(self.king_sq as Square);
        }

        self.flip_side();


        self.add_position_to_rep_table();
        self.was_capture = undo.captured_pieces_count > 0;
        Ok(())
    }

    pub fn make_move_simple(&mut self, mv: Move) -> Result<(), &'static str> {
        self.make_move(mv, &mut UndoMove::new())
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::constants::HOLE;
    use crate::board::types::{Piece, Side};
    use crate::board::utils::get_square_from_algebraic;
    use crate::moves::mv::create_move_from_algebraic;
    use crate::moves::undo::{CapturedPiece, UndoMove};
    use crate::tests::{expect_attacker_on, expect_attackers_count, expect_defenders_count, expect_king_on, expect_no_pice_on, expect_side_to_be};

    fn expect_undo_has_captured_piece(undo: &UndoMove, square: &str, piece: Piece) {
        let sq = get_square_from_algebraic(square);
        let captured_piece = CapturedPiece {
            square: sq,
            piece,
        };
        assert!(
            undo.captured_pieces().contains(&captured_piece),
            "Expected undo to have captured piece {:?} on square {}",
            piece,
            square
        );
    }

    #[test]
    fn simple_move() {
        let mut board = Board::new();
        board.side_to_move = Side::ATTACKERS;
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER);
        let mv = create_move_from_algebraic("a2a4").unwrap();

        board.make_move_simple(mv);

        expect_attacker_on(&board, "a4");
        expect_no_pice_on(&board, "a2");
        expect_attackers_count(&board, 1);
        expect_defenders_count(&board,0);
        expect_side_to_be(&board, Side::DEFENDERS)
    }


    #[test]
    fn capture_defender_in_sandwich_prepare() {
        let mut board = Board::new();
        board.side_to_move = Side::ATTACKERS;
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("a4"), Piece::DEFENDER);
        board.set_piece(get_square_from_algebraic("a5"), Piece::ATTACKER);

        let mv = create_move_from_algebraic("a2a3").unwrap();
        board.make_move_simple(mv);

        expect_attacker_on(&board, "a3");
        expect_attacker_on(&board, "a5");
        expect_no_pice_on(&board, "a4");
        expect_attackers_count(&board, 2);
        expect_defenders_count(&board, 0);
    }

    #[test]
    fn king_not_captured_in_sandwich() {
        let mut board = Board::new();
        board.side_to_move = Side::ATTACKERS;
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER);
        board.set_piece(get_square_from_algebraic("a4"), Piece::KING);
        board.set_piece(get_square_from_algebraic("a5"), Piece::ATTACKER);

        let mv = create_move_from_algebraic("a2a3").unwrap();
        board.make_move_simple(mv);

        expect_attacker_on(&board, "a3");
        expect_attacker_on(&board, "a5");
        expect_king_on(&board, "a4");
        expect_attackers_count(&board, 2);
        expect_defenders_count(&board, 0);
    }

    #[test]
    fn not_overflow_pieces() {
        let mut board = Board::new();
        board.setup_initial_position().expect("setup failed");
        board.side_to_move = Side::DEFENDERS;
        let mv = create_move_from_algebraic("d1d1").unwrap();
        let res = board.make_move_simple(mv);
        assert!(res.is_ok());
    }

    #[test]
    fn undo_should_fill_correctly() {
        let mut board = Board::new();
        board.set_side(Side::DEFENDERS);
        board.set_piece(get_square_from_algebraic("a2"), Piece::ATTACKER).unwrap();
        board.set_piece(get_square_from_algebraic("a5"), Piece::DEFENDER).unwrap();
        board.set_piece(get_square_from_algebraic("a6"), Piece::ATTACKER).unwrap();
        let mv = create_move_from_algebraic("a2a4").unwrap();

        let mut undo = UndoMove::new();
        board.make_move(mv, &mut undo).expect("make move failed");

        assert_eq!(undo.mv, mv);
        assert_eq!(undo.captured_pieces_count, 1);

        assert_eq!(undo.captured_pieces()[0], CapturedPiece {
            square: get_square_from_algebraic("a5"),
            piece: Piece::DEFENDER
        });

        assert_eq!(undo.last_move_to, HOLE);
    }

    mod shieldwall_rule {
        use crate::rules::RulesEnum;
        use super::*;

        #[test]
        fn capture_2_surrounded_pieces() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("f2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("g1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("f1"), Piece::DEFENDER).unwrap();
            println!("{:?}", board);

            let mv = create_move_from_algebraic("c1d1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");
            println!("{:?}", board);

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "e1", Piece::DEFENDER);
            expect_undo_has_captured_piece(&undo, "f1", Piece::DEFENDER);
        }

        #[test]
        fn two_not_fully_rounded_on_edge_no_capture() {
            // JS: "if there are 2 not fully rounded pieces on edge (without e2) -> not captured"
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("f2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("g1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("f1"), Piece::DEFENDER).unwrap();

            let mv = create_move_from_algebraic("c1d1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 0);
            assert_eq!(board.board[get_square_from_algebraic("e1") as usize], Piece::DEFENDER);
            assert_eq!(board.board[get_square_from_algebraic("f1") as usize], Piece::DEFENDER);
        }

        #[test]
        fn one_piece_surrounded_no_capture() {
            // JS: "one piece surrounded -> no capture"
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("d3"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("d1"), Piece::DEFENDER).unwrap();

            let mv = create_move_from_algebraic("d3d2").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 0);
            assert_eq!(board.board[get_square_from_algebraic("d1") as usize], Piece::DEFENDER);
        }

        #[test]
        fn two_rounded_on_edge_captured_but_king_not_captured() {
            // JS: "if there are 2 rounded pieces on edge, it could be captured (but king not captured)"
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("f2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("g1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("f1"), Piece::KING).unwrap();

            let mv = create_move_from_algebraic("c1d1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 1);
            expect_undo_has_captured_piece(&undo, "e1", Piece::DEFENDER);
            assert_eq!(board.board[get_square_from_algebraic("e1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("f1") as usize], Piece::KING);
        }

        #[test]
        fn defenders_can_capture_too() {
            // JS: "defenders also could capture"
            let mut board = Board::new();
            board.set_side(Side::DEFENDERS);
            board.set_piece(get_square_from_algebraic("c1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("e2"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("f2"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("f1"), Piece::ATTACKER).unwrap();

            let mv = create_move_from_algebraic("c1d1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "e1", Piece::ATTACKER);
            expect_undo_has_captured_piece(&undo, "f1", Piece::ATTACKER);
            assert_eq!(board.board[get_square_from_algebraic("e1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("f1") as usize], Piece::EMPTY);
        }

        #[test]
        fn capture_on_corner_with_shieldwall() {
            // JS: "capture on corner with shieldwall"
            let mut board = Board::new();
            board.set_side(Side::DEFENDERS);
            board.set_piece(get_square_from_algebraic("b2"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("c2"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c1"), Piece::ATTACKER).unwrap();

            let mv = create_move_from_algebraic("e1d1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "b1", Piece::ATTACKER);
            expect_undo_has_captured_piece(&undo, "c1", Piece::ATTACKER);
            assert_eq!(board.board[get_square_from_algebraic("b1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("c1") as usize], Piece::EMPTY);
        }

        #[test]
        fn capture_on_corner_with_shieldwall_bottom_left_attackers() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("b2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("b1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("c1"), Piece::DEFENDER).unwrap();

            let mv = create_move_from_algebraic("e1d1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "b1", Piece::DEFENDER);
            expect_undo_has_captured_piece(&undo, "c1", Piece::DEFENDER);
            assert_eq!(board.board[get_square_from_algebraic("b1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("c1") as usize], Piece::EMPTY);
        }

        #[test]
        fn capture_on_corner_with_shieldwall_bottom_right() {
            let mut board = Board::new();
            board.set_side(Side::DEFENDERS);
            board.set_piece(get_square_from_algebraic("i2"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("j2"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("g1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("i1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("j1"), Piece::ATTACKER).unwrap();

            let mv = create_move_from_algebraic("g1h1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "i1", Piece::ATTACKER);
            expect_undo_has_captured_piece(&undo, "j1", Piece::ATTACKER);
            assert_eq!(board.board[get_square_from_algebraic("i1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("j1") as usize], Piece::EMPTY);
        }

        #[test]
        fn capture_on_corner_with_shieldwall_top_left() {
            let mut board = Board::new();
            board.set_side(Side::DEFENDERS);
            board.set_piece(get_square_from_algebraic("b10"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("c10"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("e11"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("b11"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c11"), Piece::ATTACKER).unwrap();

            let mv = create_move_from_algebraic("e11d11").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "b11", Piece::ATTACKER);
            expect_undo_has_captured_piece(&undo, "c11", Piece::ATTACKER);
            assert_eq!(board.board[get_square_from_algebraic("b11") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("c11") as usize], Piece::EMPTY);
        }

        #[test]
        fn capture_on_corner_with_shieldwall_top_right() {
            let mut board = Board::new();
            board.set_side(Side::DEFENDERS);
            board.set_piece(get_square_from_algebraic("i10"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("j10"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("g11"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("i11"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("j11"), Piece::ATTACKER).unwrap();

            let mv = create_move_from_algebraic("g11h11").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "i11", Piece::ATTACKER);
            expect_undo_has_captured_piece(&undo, "j11", Piece::ATTACKER);
            assert_eq!(board.board[get_square_from_algebraic("i11") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("j11") as usize], Piece::EMPTY);
        }

        #[test]
        fn capture_on_corner_with_shieldwall_top_right_attackers() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("i10"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("j10"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("g11"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("i11"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("j11"), Piece::DEFENDER).unwrap();

            let mv = create_move_from_algebraic("g11h11").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 2);
            expect_undo_has_captured_piece(&undo, "i11", Piece::DEFENDER);
            expect_undo_has_captured_piece(&undo, "j11", Piece::DEFENDER);
            assert_eq!(board.board[get_square_from_algebraic("i11") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("j11") as usize], Piece::EMPTY);
        }

        #[test]
        fn shieldwall_capture_disabled_on_historical_variant() {
            let mut board = Board::new();
            board.set_rules(RulesEnum::Historical11x11);
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("i10"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("j10"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("g11"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("i11"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("j11"), Piece::DEFENDER).unwrap();

            let mv = create_move_from_algebraic("g11h11").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 0);
        }

        #[test]
        fn capture_full_right_edge_with_shieldwall_attackers() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);

            board.set_piece(get_square_from_algebraic("k1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("k11"), Piece::ATTACKER).unwrap();

            for rank in 2..=10 {
                let coord = format!("k{}", rank);
                board.set_piece(get_square_from_algebraic(&coord), Piece::DEFENDER).unwrap();
            }

            for rank in 2..=10 {
                if rank == 6 { continue; }
                let coord = format!("j{}", rank);
                board.set_piece(get_square_from_algebraic(&coord), Piece::ATTACKER).unwrap();
            }

            board.set_piece(get_square_from_algebraic("h6"), Piece::ATTACKER).unwrap();
            println!("{:?}", board);

            let mv = create_move_from_algebraic("h6j6").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            // we cant eat all because capture should be flanked
            assert_eq!(undo.captured_pieces_count, 0);
        }

        #[test]
        fn standard_captures_complex() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("b2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("g1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e3"), Piece::ATTACKER).unwrap();

            board.set_piece(get_square_from_algebraic("d1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("b1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("e2"), Piece::DEFENDER).unwrap();

            board.set_piece(get_square_from_algebraic("c1"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("d2"), Piece::ATTACKER).unwrap();

            let mv = create_move_from_algebraic("g1e1").unwrap();
            let mut undo = UndoMove::new();
            println!("{:?}", board);
            board.make_move(mv, &mut undo).expect("make move failed");

            println!("{:?}", board);

            assert_eq!(undo.captured_pieces_count, 3);
            expect_undo_has_captured_piece(&undo, "b1", Piece::DEFENDER);
            expect_undo_has_captured_piece(&undo, "d1", Piece::DEFENDER);
            expect_undo_has_captured_piece(&undo, "e2", Piece::DEFENDER);

            assert_eq!(board.board[get_square_from_algebraic("b1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("d1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("e2") as usize], Piece::EMPTY);
        }

        #[test]
        fn capture_non_flanking_is_not_capture() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("b1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("d3"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("d1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("f6"), Piece::KING).unwrap();

            let mv = create_move_from_algebraic("d3d2").unwrap();

            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).expect("make move failed");

            assert_eq!(undo.captured_pieces_count, 0);
        }
    }

    mod king_capture_rule {
        use crate::Board;
        use crate::board::utils::get_square_from_algebraic;
        use crate::mv::create_move_from_algebraic;
        use crate::types::{OptionalSquare, Piece, Side, Square};
        use crate::undo::{CapturedPiece, UndoMove};

        #[test]
        fn king_capture_possible_not_near_throne()  {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("d5"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("d4"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c5"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("d7"), Piece::ATTACKER).unwrap();

            // make move d7d6
            let mv = create_move_from_algebraic("d7d6").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo);
            assert_eq!(undo.captured_pieces_count, 1);
            assert_eq!(undo.captured_pieces()[0], CapturedPiece {
                square: get_square_from_algebraic("d5"),
                piece: Piece::KING
            });
        }

        #[test]
        fn king_capture_not_possible_near_throne() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("e6"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("a6"), Piece::ATTACKER).unwrap();
            println!("{:?}", board);
            // make move e7e6
            let mv = create_move_from_algebraic("a6d6").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo);
            assert_eq!(undo.captured_pieces_count, 1);
            assert_eq!(undo.captured_pieces()[0], CapturedPiece {
                square: get_square_from_algebraic("e6"),
                piece: Piece::KING
            });
        }

        #[test]
        fn king_capture_impossible_if_not_surrounded() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("e6"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("a6"), Piece::ATTACKER).unwrap();
            println!("{:?}", board);
            // make move a6d6
            let mv = create_move_from_algebraic("a6d6").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo);
            assert_eq!(undo.captured_pieces_count, 0);
        }

        #[test]
        fn king_capture_impossible_if_not_surrounded_fully_by_attackers() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("e6"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("a6"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e7"), Piece::DEFENDER).unwrap();
            println!("{:?}", board);
            // make move a6d6
            let mv = create_move_from_algebraic("a6d6").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo);
            assert_eq!(undo.captured_pieces_count, 0);
        }

        #[test]
        fn king_already_surrounded_but_not_capture_move() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("e6"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("d6"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("f6"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("d1"), Piece::ATTACKER).unwrap();
            println!("{:?}", board);
            // make move e5e4
            let mv = create_move_from_algebraic("d1e1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo);
            assert_eq!(undo.captured_pieces_count, 0);
        }

        #[test]
        fn king_is_removing_and_go_back_working_correctly() {
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("e6"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("e5"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e7"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("a6"), Piece::ATTACKER).unwrap();
            println!("{:?}", board);
            // make move e7e6
            let mv = create_move_from_algebraic("a6d6").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo);
            assert_eq!(undo.captured_pieces_count, 1);
            assert_eq!(undo.captured_pieces()[0], CapturedPiece {
                square: get_square_from_algebraic("e6"),
                piece: Piece::KING
            });
            assert_eq!(board.king_sq, -1);

            // Now undo the move and check if king is restored
            board.unmake_move(&mut undo).expect("undo move failed");
            assert_eq!(board.king_sq, get_square_from_algebraic("e6") as OptionalSquare);
        }

        #[test]
        fn another_move_should_not_trigger_shield_wall() {
            let mut board = Board::new();
            // set fen
            board.set_fen("3aaaa4/11/11/6a4/3aaaa3a/a2ad5a/3adka4/3adda4/4a1a4/5d1aa2/3a2adda1 a");
            let mv = create_move_from_algebraic("d1e1").unwrap();
            let mut undo = UndoMove::new();
            board.make_move(mv, &mut undo).unwrap();

            assert_eq!(undo.captured_pieces_count, 0);
        }
    }
}