use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use wasm_bindgen::prelude::wasm_bindgen;
use crate::board::fen::FenError;
use crate::board::PRECOMPUTED;
use crate::nnue::{calculate_nnue_index, Weights1, Weights2, HIDDEN, INPUTS, NNUE, STM_BIT};
use super::zobrist::{ZOBRIST_DATA};
use super::types::{OptionalSquare, Piece, Side, Square, ZobristHash};
use super::constants::{SQS, ATTACKERS_MAX, DEFENDERS_MAX, BOARD_SIZE, HOLE, INITIAL_FEN};

pub struct Board {
    pub board: [Piece; SQS],
    pub attackers: [Square; ATTACKERS_MAX],
    pub defenders: [Square; DEFENDERS_MAX],
    pub king_sq: OptionalSquare,
    pub attackers_count: u8,
    pub defenders_count: u8,
    pub row_occ: [u16; BOARD_SIZE],
    pub col_occ: [u16; BOARD_SIZE],
    pub piece_index_by_square: [u8; SQS],
    pub side_to_move: Side,
    pub zobrist: ZobristHash,
    pub rep_table: HashMap<ZobristHash, u8>,
    pub last_move_to: OptionalSquare,
    pub nnue: NNUE,
}

impl Board {
    pub fn new() -> Self {
        Self {
            board: [Piece::EMPTY; SQS],
            attackers: [0; ATTACKERS_MAX],
            defenders: [0; DEFENDERS_MAX],
            king_sq: HOLE,
            attackers_count: 0,
            defenders_count: 0,
            row_occ: [0; BOARD_SIZE],
            col_occ: [0; BOARD_SIZE],
            piece_index_by_square: [0; SQS],
            side_to_move: Side::ATTACKERS,
            zobrist: 0,
            rep_table: HashMap::new(),
            last_move_to: HOLE,
            nnue: NNUE::new([[0.0; HIDDEN]; INPUTS], [0.0; HIDDEN])
        }
    }

    pub fn new_with_nnue(w1: Weights1, w2: Weights2) -> Self {
        let mut board = Self::new();
        board.set_nnue(w1, w2);
        board
    }

    pub fn set_nnue(&mut self, w1: Weights1, w2: Weights2) {
        self.nnue = NNUE::new(w1, w2);
    }

    pub fn clear(&mut self) {
        self.board.fill(Piece::EMPTY);
        self.attackers.fill(0);
        self.defenders.fill(0);
        self.king_sq = HOLE;
        self.attackers_count = 0;
        self.defenders_count = 0;
        self.row_occ.fill(0);
        self.col_occ.fill(0);
        self.piece_index_by_square.fill(0);
        self.side_to_move = Side::ATTACKERS;
        self.zobrist = 0;
        self.rep_table.clear();
        self.last_move_to = HOLE;
        self.nnue.clear();
        self.set_side_to_move(Side::ATTACKERS);
    }

    fn set_side_to_move(&mut self, side: Side) {
        self.side_to_move = side;
        let bit = if side == Side::DEFENDERS { 1 } else { 0 };

        if bit == 1 {
            self.nnue.set_input(STM_BIT);
        } else {
            self.nnue.reset_input(STM_BIT);
        }
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
        self.zobrist ^= ZOBRIST_DATA.table[piece as usize][sq];

        let row = PRECOMPUTED.row[sq];
        let col = PRECOMPUTED.col[sq];

        self.row_occ[row] |= 1 << col;
        self.col_occ[col] |= 1 << row;

        self.nnue.set_input(calculate_nnue_index(piece, sq));

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
        self.nnue.reset_input(calculate_nnue_index(piece, sq));


        self.zobrist ^= ZOBRIST_DATA.table[piece as usize][sq];
        self.board[sq] = Piece::EMPTY;

        let row = PRECOMPUTED.row[sq];
        let col = PRECOMPUTED.col[sq];

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
        self.zobrist ^= ZOBRIST_DATA.side;
    }

    pub fn setup_initial_position(&mut self) -> Result<(), FenError> {
        self.set_fen(INITIAL_FEN)
    }

    pub fn set_side(&mut self, side: Side) {
        if self.side_to_move != side {
            self.flip_side();
        }
    }

    pub fn get_eval(&self) -> i32 {
        self.nnue.evaluate()
    }

    pub fn print_eval_side(&self) {
        let nnue_input = self.nnue.inputs[STM_BIT];
        println!("{}", nnue_input);
    }
}

impl Debug for Board {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let size = BOARD_SIZE;
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

                // Цвета ANSI: 31=красный, 34=синий, 33=жёлтый
                match p {
                    Piece::ATTACKER => write!(f, "\x1b[31mA\x1b[0m")?,
                    Piece::DEFENDER => write!(f, "\x1b[34mD\x1b[0m")?,
                    Piece::KING     => write!(f, "\x1b[33mK\x1b[0m")?,
                    Piece::EMPTY    => write!(f, ".")?,
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
        // if let Some(fen) = self.fen() {
        //     writeln!(f, "FEN: {}", fen)?;
        // }

        Ok(())
    }
}