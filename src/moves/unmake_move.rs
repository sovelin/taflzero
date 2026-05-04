use crate::board::Board;
use crate::moves::undo::UndoMove;

impl Board {
    fn remove_position_from_rep_table(&mut self) {
        if let Some(count) = self.rep_table.get_mut(&self.zobrist) {
            if *count > 1 {
                *count -= 1;
            } else {
                self.rep_table.remove(&self.zobrist);
            }
        }
    }

    pub fn unmake_move(&mut self, undo: &mut UndoMove) -> Result<(), &'static str> {
        // Remove current position from repetition table
        self.remove_position_from_rep_table();

        // Move piece back to original square
        let from = undo.mv.from();
        let to = undo.mv.to();

        let piece = self.board[to];
        self.clear_piece(to);
        self.set_piece(from, piece)?;

        // Restore captured pieces
        for captured in undo.captured_pieces().iter() {
            self.set_piece(captured.square, captured.piece)?;
        }

        // Restore side to move
        self.flip_side();

        // Update last move to
        self.last_move_to = undo.last_move_to;
        self.was_capture = undo.was_capture;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::board::Board;
    use crate::board::types::{OptionalSquare, Piece, Side};
    use crate::board::utils::get_square_from_algebraic;
    use crate::moves::mv::Move;
    use crate::moves::undo::{CapturedPiece, UndoMove};
    use crate::tests::{
        expect_attacker_on, expect_attackers_count, expect_defender_on, expect_defenders_count,
        expect_side_to_be,
    };

    #[test]
    fn test_unmake_move() {
        let board = &mut Board::new();
        board.set_side(Side::DEFENDERS);
        board
            .set_piece(get_square_from_algebraic("b5"), Piece::ATTACKER)
            .unwrap();
        board
            .set_piece(get_square_from_algebraic("b7"), Piece::ATTACKER)
            .unwrap();
        println!("{:?}", board);

        let mut undo = UndoMove::new();
        undo.add_captured_piece(CapturedPiece {
            square: get_square_from_algebraic("b6"),
            piece: Piece::DEFENDER,
        });

        undo.mv = Move::new(
            get_square_from_algebraic("b4"),
            get_square_from_algebraic("b5"),
        );

        undo.moved_piece = Piece::ATTACKER;
        undo.last_move_to = get_square_from_algebraic("a9") as OptionalSquare;

        // Unmake the move
        board.unmake_move(&mut undo).unwrap();

        // Verify the board state
        expect_attacker_on(board, "b4");
        expect_attacker_on(board, "b7");
        expect_attackers_count(board, 2);
        expect_defender_on(board, "b6");
        expect_defenders_count(board, 1);
        expect_side_to_be(board, Side::ATTACKERS);
        assert_eq!(
            board.last_move_to,
            get_square_from_algebraic("a9") as OptionalSquare
        );
    }
}
