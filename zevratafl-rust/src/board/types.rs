use std::fmt::Display;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Side {
    ATTACKERS = 0,
    DEFENDERS = 1,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    EMPTY = 0,
    ATTACKER = 1,
    DEFENDER = 2,
    KING = 3,
}

pub type Square = usize;
pub type Row = usize;
pub type Col = usize;
pub type OptionalSquare = isize;
pub type ZobristHash = u64;

impl Side {
    pub fn opposite(side: Side) -> Side {
        match side {
            Side::ATTACKERS => Side::DEFENDERS,
            Side::DEFENDERS => Side::ATTACKERS,
        }
    }
}

impl Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::ATTACKERS => write!(f, "Attackers"),
            Side::DEFENDERS => write!(f, "Defenders"),
        }
    }
}

impl Display for Piece {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Piece::EMPTY => write!(f, "."),
            Piece::ATTACKER => write!(f, "A"),
            Piece::DEFENDER => write!(f, "D"),
            Piece::KING => write!(f, "K"),
        }
    }
}