mod is_capture_possible;
mod mask_shield_captures;

use crate::board::{Board, PRECOMPUTED};
use crate::board::types::{OptionalSquare};
use crate::moves::make_move::is_capture_possible::is_capture_possible;
use crate::moves::make_move::mask_shield_captures::make_shield_wall_captures;
use crate::moves::mv::Move;
use crate::moves::undo::{CapturedPiece, UndoMove};

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

        make_shield_wall_captures(self, to, undo);

        self.last_move_to = to as OptionalSquare;
        self.flip_side();
        self.add_position_to_rep_table();
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
        fn standard_captures_complex() {
            // JS: "standard captures should works too. more complex test"
            let mut board = Board::new();
            board.set_side(Side::ATTACKERS);
            board.set_piece(get_square_from_algebraic("b2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("c2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("f2"), Piece::ATTACKER).unwrap();
            board.set_piece(get_square_from_algebraic("e1"), Piece::ATTACKER).unwrap();

            board.set_piece(get_square_from_algebraic("d1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("b1"), Piece::DEFENDER).unwrap();
            board.set_piece(get_square_from_algebraic("d3"), Piece::DEFENDER).unwrap();

            board.set_piece(get_square_from_algebraic("c1"), Piece::KING).unwrap();
            board.set_piece(get_square_from_algebraic("d4"), Piece::ATTACKER).unwrap();

            let mv = create_move_from_algebraic("f2d2").unwrap();
            let mut undo = UndoMove::new();
            println!("{:?}", board);
            board.make_move(mv, &mut undo).expect("make move failed");

            println!("{:?}", board);

            assert_eq!(undo.captured_pieces_count, 3);
            expect_undo_has_captured_piece(&undo, "b1", Piece::DEFENDER);
            expect_undo_has_captured_piece(&undo, "d1", Piece::DEFENDER);
            expect_undo_has_captured_piece(&undo, "d3", Piece::DEFENDER);

            assert_eq!(board.board[get_square_from_algebraic("b1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("d1") as usize], Piece::EMPTY);
            assert_eq!(board.board[get_square_from_algebraic("d3") as usize], Piece::EMPTY);
        }
    }
}