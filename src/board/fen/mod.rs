mod tests;

use std::error::Error;
use std::fmt::Display;

use crate::board::Board;
use crate::board::constants::HOLE;
use crate::board::types::{Col, OptionalSquare, Piece, Row, Side, Square};
use crate::board::utils::get_square;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    MissingSide,
    InvalidRowsCount(usize, usize),
    InvalidChar(char),
    RowTooWide {
        row: usize,
        got: usize,
        should: usize,
    },
    UnknownSide(char),
    InvalidSetPiece,
}

impl Display for FenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FenError::MissingSide => write!(f, "FEN must contain side to move (space + 'a'|'d')"),
            FenError::InvalidRowsCount(got, should) => {
                write!(f, "Invalid number of rows: {got}, expected {should}")
            }
            FenError::InvalidChar(ch) => write!(f, "Invalid FEN character: {ch}"),
            FenError::RowTooWide { row, got, should } => {
                write!(f, "Row {row} is too wide: {got} > {should}")
            }
            FenError::UnknownSide(ch) => write!(f, "Unknown side char: {ch} (expected 'a' or 'd')"),
            FenError::InvalidSetPiece => write!(f, "Failed to set piece on board"),
        }
    }
}
impl Error for FenError {}

#[inline]
fn char_to_piece(ch: char) -> Option<Piece> {
    match ch {
        'k' => Some(Piece::KING),
        'a' => Some(Piece::ATTACKER),
        'd' => Some(Piece::DEFENDER),
        _ => None,
    }
}

#[inline]
fn side_from_char(ch: char) -> Option<Side> {
    match ch {
        'a' => Some(Side::ATTACKERS),
        'd' => Some(Side::DEFENDERS),
        _ => None,
    }
}

#[inline]
fn piece_to_char(p: Piece) -> Option<char> {
    match p {
        Piece::KING => Some('k'),
        Piece::ATTACKER => Some('a'),
        Piece::DEFENDER => Some('d'),
        Piece::EMPTY => None,
    }
}

impl Board {
    fn set_row_from_fen(&mut self, row_fen: &str, row_idx: Row) -> Result<(), FenError> {
        let mut col: Col = 0;
        let mut num: usize = 0;

        for ch in row_fen.chars() {
            if ch.is_ascii_digit() {
                num = num * 10 + (ch as u8 - b'0') as usize;
                continue;
            }

            if num > 0 {
                col = col.saturating_add(num as Col);
                num = 0;
            }

            if let Some(piece) = char_to_piece(ch) {
                if col as usize >= self.board_size() {
                    return Err(FenError::RowTooWide {
                        row: row_idx,
                        got: col as usize + 1,
                        should: self.board_size(),
                    });
                }
                let sq: Square = get_square(row_idx, col, self.board_size());

                match self.set_piece(sq, piece) {
                    Ok(_) => {}
                    Err(_) => return Err(FenError::InvalidSetPiece),
                }

                col += 1;
            } else {
                return Err(FenError::InvalidChar(ch));
            }
        }

        if num > 0 {
            col = col.saturating_add(num as Col);
        }

        if (col as usize) > self.board_size() {
            return Err(FenError::RowTooWide {
                row: row_idx,
                got: col as usize,
                should: self.board_size(),
            });
        }

        Ok(())
    }

    /// Reads `<rows> [last-to] <side> [clock] [count]`.
    pub fn set_fen(&mut self, fen: &str) -> Result<(), FenError> {
        let mut parts = fen.split_whitespace();
        let rows_part = parts.next().ok_or(FenError::MissingSide)?;
        let rest: Vec<&str> = parts.collect();

        let is_side = |s: &str| s == "a" || s == "d";
        let (last_to_part, side_part, counters) = match rest.as_slice() {
            [side, tail @ ..] if is_side(side) => (None, *side, tail),
            [last_to, side, tail @ ..] => (Some(*last_to), *side, tail),
            _ => return Err(FenError::MissingSide),
        };

        let rows: Vec<&str> = rows_part.split('/').collect();

        if rows.len() != self.board_size() {
            return Err(FenError::InvalidRowsCount(rows.len(), self.board_size()));
        }

        self.clear();

        for (i, row_fen) in rows.iter().enumerate() {
            let row_idx: Row = (self.board_size() - 1 - i) as Row;
            self.set_row_from_fen(row_fen, row_idx)?;
        }

        let s = side_part
            .trim()
            .chars()
            .next()
            .ok_or(FenError::MissingSide)?;
        let desired = side_from_char(s).ok_or(FenError::UnknownSide(s))?;
        if self.side_to_move != desired {
            self.flip_side();
        }

        self.last_move_to = match last_to_part {
            None | Some("-") => HOLE,
            Some(sq) => self.get_square_from_algebraic(sq) as OptionalSquare,
        };

        self.halfmove_clock = counters.first().and_then(|c| c.parse().ok()).unwrap_or(0);
        self.halfmove_count = counters.get(1).and_then(|c| c.parse().ok()).unwrap_or(0);

        Ok(())
    }

    pub fn get_fen(&self) -> String {
        let mut rows_out: Vec<String> = Vec::with_capacity(self.board_size());

        for row in (0..self.board_size()).rev() {
            let mut empty = 0usize;
            let mut out = String::new();

            for col in 0..self.board_size() {
                let sq = get_square(row as Row, col as Col, self.board_size());
                let p = self.board[sq as usize];

                if p == Piece::EMPTY {
                    empty += 1;
                } else {
                    if empty > 0 {
                        out.push_str(&empty.to_string());
                        empty = 0;
                    }
                    if let Some(ch) = piece_to_char(p) {
                        out.push(ch);
                    }
                }
            }

            if empty > 0 {
                out.push_str(&empty.to_string());
            }

            rows_out.push(out);
        }

        let side_ch = match self.side_to_move {
            Side::ATTACKERS => 'a',
            Side::DEFENDERS => 'd',
        };

        let last_to = if self.last_move_to == HOLE {
            "-".to_string()
        } else {
            self.get_sq_algebraic(self.last_move_to as Square)
        };

        format!(
            "{rows} {last_to} {side} {clock} {count}",
            rows = rows_out.join("/"),
            side = side_ch,
            clock = self.halfmove_clock,
            count = self.halfmove_count,
        )
    }
}
