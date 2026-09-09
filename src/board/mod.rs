pub mod constants;
mod fen;
mod masks;
pub mod position_export;
mod precompute;
pub mod rules;
mod tests;
pub mod types;
pub mod utils;
mod zobrist;

pub use precompute::*;
pub use utils::get_side_by_piece;

use crate::board::constants::{ATTACKERS_MAX, DEFENDERS_MAX, HOLE, SQS};
use crate::board::fen::FenError;
use crate::board::rules::{Rules, RulesEnum};
use crate::board::types::{OptionalSquare, Piece, Side, Square, ZobristHash};
use crate::board::utils::get_square;
use crate::board::zobrist::ZobristData;
use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;

pub struct Board {
    pub board: Vec<Piece>,
    pub attackers: Vec<Square>,
    pub defenders: Vec<Square>,
    pub piece_index_by_square: Vec<u8>,
    pub king_sq: OptionalSquare,
    pub attackers_count: u8,
    pub defenders_count: u8,
    pub row_occ: Vec<u16>,
    pub col_occ: Vec<u16>,
    pub side_to_move: Side,
    pub zobrist: ZobristHash,
    pub rep_table: HashMap<ZobristHash, u8>,
    pub last_move_to: OptionalSquare,
    pub was_capture: bool,
    pub rules: RulesEnum,
    precomputed: Arc<Precomputed>,
    zobrist_data: ZobristData,
}

impl Default for Board {
    fn default() -> Self {
        Self::new()
    }
}

impl Board {
    pub fn new() -> Self {
        Self::from_rules(RulesEnum::Copenhagen11x11)
    }

    pub fn from_rules(rules: RulesEnum) -> Self {
        let board_size = rules.rules().board_size;
        let sqs = board_size * board_size;

        Self {
            board: vec![Piece::EMPTY; sqs],
            attackers: vec![0; sqs],
            defenders: vec![0; sqs],
            king_sq: HOLE,
            attackers_count: 0,
            defenders_count: 0,
            row_occ: vec![0; board_size],
            col_occ: vec![0; board_size],
            piece_index_by_square: vec![0; sqs],
            side_to_move: Side::ATTACKERS,
            zobrist: 0,
            rep_table: HashMap::new(),
            last_move_to: HOLE,
            was_capture: false,
            rules,
            precomputed: Arc::new(Precomputed::new(board_size)),
            zobrist_data: ZobristData::new(board_size),
        }
    }

    #[inline]
    pub fn precomputed(&self) -> &Arc<Precomputed> {
        &self.precomputed
    }

    pub fn get_rules(&self) -> Rules {
        self.rules.rules()
    }

    pub fn set_rules(&mut self, rules: RulesEnum) {
        let brd = Board::from_rules(rules);
        *self = brd;
    }

    pub fn clear(&mut self) {
        let brd = Board::from_rules(self.rules);
        *self = brd;
    }

    fn set_side_to_move(&mut self, side: Side) {
        self.side_to_move = side;
    }

    fn set_attacker(&mut self, sq: Square) -> Result<(), &'static str> {
        if self.attackers_count >= ATTACKERS_MAX as u8 {
            return Err("Exceeded maximum attackers capacity");
        }

        self.attackers[self.attackers_count as usize] = sq;
        self.attackers_count += 1;
        self.piece_index_by_square[sq] = self.attackers_count - 1;

        Ok(())
    }

    fn set_defender(&mut self, sq: Square) -> Result<(), &'static str> {
        if self.defenders_count >= DEFENDERS_MAX as u8 {
            return Err("Exceeded maximum defenders capacity");
        }

        self.defenders[self.defenders_count as usize] = sq;
        self.defenders_count += 1;
        self.piece_index_by_square[sq] = self.defenders_count - 1;

        Ok(())
    }

    fn clear_defender(&mut self, sq: Square) {
        let index = self.piece_index_by_square[sq];

        if self.defenders_count == 1 {
            self.piece_index_by_square[sq] = 0
        } else {
            let last_sq = self.defenders[self.defenders_count as usize - 1];
            self.defenders[index as usize] = last_sq;
            self.piece_index_by_square[last_sq] = index;
        }

        self.defenders_count -= 1;
    }

    fn clear_attacker(&mut self, sq: Square) {
        let index = self.piece_index_by_square[sq];

        if self.attackers_count == 1 {
            self.piece_index_by_square[sq] = 0
        } else {
            let last_sq = self.attackers[self.attackers_count as usize - 1];
            self.attackers[index as usize] = last_sq;
            self.piece_index_by_square[last_sq] = index;
        }

        self.attackers_count -= 1;
    }

    pub fn set_piece(&mut self, sq: Square, piece: Piece) -> Result<(), &'static str> {
        self.board[sq] = piece;
        self.zobrist ^= self.zobrist_data.table[piece as usize][sq];

        let row = self.precomputed.row[sq];
        let col = self.precomputed.col[sq];

        self.row_occ[row] |= 1 << col;
        self.col_occ[col] |= 1 << row;

        if piece == Piece::ATTACKER {
            self.set_attacker(sq)
        } else if piece == Piece::DEFENDER {
            self.set_defender(sq)
        } else if piece == Piece::KING {
            self.king_sq = sq as OptionalSquare;
            Ok(())
        } else {
            Err("Invalid piece type")
        }
    }
    pub fn clear_piece(&mut self, sq: Square) {
        let piece = self.board[sq];

        self.zobrist ^= self.zobrist_data.table[piece as usize][sq];
        self.board[sq] = Piece::EMPTY;

        let row = self.precomputed.row[sq];
        let col = self.precomputed.col[sq];

        self.row_occ[row] &= !(1 << col);
        self.col_occ[col] &= !(1 << row);

        if piece == Piece::ATTACKER {
            self.clear_attacker(sq);
        } else if piece == Piece::DEFENDER {
            self.clear_defender(sq);
        } else {
            self.king_sq = HOLE;
        }
    }

    pub fn flip_side(&mut self) {
        self.set_side_to_move(Side::opposite(self.side_to_move));
        self.zobrist ^= self.zobrist_data.side;
    }

    pub fn setup_initial_position(&mut self) -> Result<(), FenError> {
        self.clear();
        self.set_fen(self.rules.rules().initial_fen.as_str())
    }

    pub fn set_side(&mut self, side: Side) {
        if self.side_to_move != side {
            self.flip_side();
        }
    }

    pub fn board(&self) -> &Vec<Piece> {
        &self.board
    }

    pub fn board_size(&self) -> usize {
        self.rules.rules().board_size
    }
}

impl Debug for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let size = self.board_size();
        let cell_gap = "  ";

        write!(f, "    ")?;
        for c in 0..size {
            let ch = (b'a' + c as u8) as char;
            write!(f, "{:<2}", ch)?;
            if c + 1 != size {
                write!(f, " ")?;
            }
        }
        writeln!(f)?;

        let border = "─".repeat(size * 3 - 1);
        writeln!(f, "  ┌{}┐", border)?;

        for r in (0..size).rev() {
            write!(f, "{:>2}│ ", r + 1)?;
            for c in 0..size {
                let sq = r * size + c;
                let p = self.board[sq];

                // ANSI colors: 31=red, 34=blue, 33=yellow
                match p {
                    Piece::ATTACKER => write!(f, "\x1b[31mA\x1b[0m")?,
                    Piece::DEFENDER => write!(f, "\x1b[34mD\x1b[0m")?,
                    Piece::KING => write!(f, "\x1b[33mK\x1b[0m")?,
                    Piece::EMPTY => write!(f, ".")?,
                }

                if c + 1 != size {
                    write!(f, "{cell_gap}")?;
                }
            }
            writeln!(f, "│{:>2}", r + 1)?;
        }

        writeln!(f, "  └{}┘", border)?;

        write!(f, "    ")?;
        for c in 0..size {
            let ch = (b'a' + c as u8) as char;
            write!(f, "{:<2}", ch)?;
            if c + 1 != size {
                write!(f, " ")?;
            }
        }
        writeln!(f)?;

        writeln!(f, "zobrist: {}", self.zobrist)?;

        Ok(())
    }
}

impl Display for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub fn set_board_from_str(board: &mut Board, position: &str) {
    board.clear();

    let board_size = board.board_size();

    for (r, line) in position.lines().enumerate() {
        let mut index = 0;
        for ch in line.chars() {
            let sq = get_square(board_size - 1 - r, index, board_size);
            let piece = match ch {
                'A' => Some(Piece::ATTACKER),
                'D' => Some(Piece::DEFENDER),
                'K' => Some(Piece::KING),
                '.' => Some(Piece::EMPTY),
                _ => None,
            };

            if let Some(piece) = piece {
                if piece != Piece::EMPTY {
                    board.set_piece(sq, piece).expect("set_piece");
                }
                index += 1;
            };
        }
    }
}
