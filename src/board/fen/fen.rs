use std::error::Error;
use std::fmt::Display;

use crate::board::constants::BOARD_SIZE;
use crate::board::types::{Piece, Side, Square, Row, Col};
use crate::board::utils::get_square;
use crate::board::{Board};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FenError {
    MissingSide,
    InvalidRowsCount(usize),
    InvalidChar(char),
    RowTooWide{ row: usize, got: usize },
    UnknownSide(char),
    InvalidSetPiece,
}

impl Display for FenError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FenError::MissingSide => write!(f, "FEN must contain side to move (space + 'a'|'d')"),
            FenError::InvalidRowsCount(got) => write!(f, "Invalid number of rows: {got}, expected {BOARD_SIZE}"),
            FenError::InvalidChar(ch) => write!(f, "Invalid FEN character: {ch}"),
            FenError::RowTooWide{row, got} => write!(f, "Row {row} is too wide: {got} > {BOARD_SIZE}"),
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
                // копим число пустот
                num = num * 10 + (ch as u8 - b'0') as usize;
                continue;
            }

            // если до этого шли цифры — сначала пролистываем пустые
            if num > 0 {
                col = col.saturating_add(num as Col);
                num = 0;
            }

            if let Some(piece) = char_to_piece(ch) {
                if col as usize >= BOARD_SIZE {
                    return Err(FenError::RowTooWide { row: row_idx as usize, got: col as usize + 1 });
                }
                let sq: Square = get_square(row_idx, col);

                match self.set_piece(sq, piece) {
                    Ok(_) => {}
                    Err(_) => return Err(FenError::InvalidSetPiece),
                }

                col += 1;
            } else {
                return Err(FenError::InvalidChar(ch));
            }
        }

        // trailing цифры в конце строки (допускаем)
        if num > 0 {
            col = col.saturating_add(num as Col);
        }

        if (col as usize) > BOARD_SIZE {
            return Err(FenError::RowTooWide { row: row_idx as usize, got: col as usize });
        }

        Ok(())
    }

    pub fn set_fen(&mut self, fen: &str) -> Result<(), FenError> {
        let (rows_part, side_part) = fen.split_once(' ').ok_or(FenError::MissingSide)?;
        let rows: Vec<&str> = rows_part.split('/').collect();

        if rows.len() != BOARD_SIZE {
            return Err(FenError::InvalidRowsCount(rows.len()));
        }

        // очистка (важно для корректных счётчиков/зобриста)
        self.clear();

        // В FEN первая строка — верхняя (максимальный ряд).
        // На твоей утилите ряды идут от 0 снизу, поэтому маппим:
        for (i, row_fen) in rows.iter().enumerate() {
            let row_idx: Row = (BOARD_SIZE - 1 - i) as Row;
            self.set_row_from_fen(row_fen, row_idx)?;
        }

        // сторона хода
        let s = side_part.trim().chars().next().ok_or(FenError::MissingSide)?;
        let desired = side_from_char(s).ok_or(FenError::UnknownSide(s))?;
        if self.side_to_move != desired {
            self.flip_side();
        }

        Ok(())
    }

    pub fn get_fen(&self) -> String {
        let mut rows_out: Vec<String> = Vec::with_capacity(BOARD_SIZE);

        // идём сверху вниз (BOARD_SIZE-1 .. 0)
        for row in (0..BOARD_SIZE).rev() {
            let mut empty = 0usize;
            let mut out = String::new();

            for col in 0..BOARD_SIZE {
                let sq = get_square(row as Row, col as Col);
                let p = self.board[sq as usize]; // поле публичное — читаем напрямую

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

        format!("{rows} {side}", rows = rows_out.join("/"), side = side_ch)
    }
}