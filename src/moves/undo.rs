use std::fmt::Debug;
use crate::board::constants::HOLE;
use crate::board::types::{OptionalSquare, Piece, Square};
use crate::moves::mv::Move;

static MAX_CAPTURED_PIECES: u8 = 10;

#[derive(Copy, Clone, PartialEq, Eq)]
pub struct CapturedPiece {
    pub square: Square,
    pub piece: Piece,
}

impl Debug for CapturedPiece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CapturedPiece {{ square: {}, piece: {:?} }}", self.square, self.piece)
    }
}

pub struct UndoMove {
    pub captured_pieces_count: u8,
    pub moved_piece: Piece,
    pub _captured_pieces: [CapturedPiece; MAX_CAPTURED_PIECES as usize],
    pub mv: Move,
    pub last_move_to: OptionalSquare,
    pub was_capture: bool,
}

impl UndoMove {
    pub fn new() -> Self {
        Self {
            captured_pieces_count: 0,
            _captured_pieces: [CapturedPiece {
                square: 0,
                piece: Piece::EMPTY,
            }; MAX_CAPTURED_PIECES as usize],
            mv: Move::default(),
            last_move_to: HOLE,
            moved_piece: Piece::EMPTY,
            was_capture: false,
        }
    }

    pub fn clear_captured_pieces(&mut self) {
        self.captured_pieces_count = 0;

    }

    pub fn captured_pieces(&self) -> &[CapturedPiece] {
        &self._captured_pieces[..self.captured_pieces_count as usize]
    }

    pub fn add_captured_piece(&mut self, sq: CapturedPiece) {
        if self.captured_pieces_count < MAX_CAPTURED_PIECES {
            self._captured_pieces[self.captured_pieces_count as usize] = sq;
            self.captured_pieces_count += 1;
        } else {
            panic!("Exceeded maximum number of captured pieces in UndoMove");
        }
    }
}