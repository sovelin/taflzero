use crate::evaluation::is_mate_score;
use crate::moves::mv::Move;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum TTFlag {
    Exact,
    LowerBound,
    UpperBound,
}

#[derive(Copy, Clone)]
pub struct TTEntry {
    key: u64,
    depth: u8,
    score: i32,
    flag: TTFlag,
    best_move: Move,
}

impl TTEntry {
    pub fn is_valid(&self, key: u64) -> bool {
        self.key == key
    }

    pub fn best_move(&self) -> Move {
        self.best_move
    }

    pub fn flag(&self) -> TTFlag {
        self.flag
    }

    pub fn score(&self, height: u32) -> i32 {
        if is_mate_score(self.score) {
            if self.score > 0 {
                self.score - (height as i32)
            } else {
                self.score + (height as i32)
            }
        } else {
            self.score
        }
    }

    pub fn depth(&self) -> u8 {
        self.depth
    }
}

impl Default for TTEntry {
    fn default() -> Self {
        Self {
            key: 0,
            depth: 0,
            score: 0,
            flag: TTFlag::Exact,
            best_move: Move::default(),
        }
    }
}

pub struct TranspositionTable {
    table: Vec<TTEntry>,
    size: usize,
    mask: usize,
}

impl TranspositionTable {
    pub fn new(size_mb: usize) -> Self {
        let size = (size_mb * 1024 * 1024) / size_of::<TTEntry>();

        let size = size.next_power_of_two();

        Self {
            table: vec![TTEntry::default(); size],
            size,
            mask: size - 1,
        }
    }

    pub fn probe(&self, key: u64) -> &TTEntry {
        let index = (key as usize) & self.mask;
        &self.table[index]
    }

    pub fn store(&mut self, key: u64, depth: u8, score: i32, flag: TTFlag, best_move: Move, height: u32) {
        let score = if is_mate_score(score) {
            if score > 0 {
                score + (height as i32)
            } else {
                score - (height as i32)
            }
        } else {
            score
        };

        let index = (key as usize) & self.mask;

        self.table[index] = TTEntry {
            key,
            depth,
            score,
            flag,
            best_move,
        };
    }
}