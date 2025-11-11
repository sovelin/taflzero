use crate::moves::mv::Move;
use crate::search::constants::MAX_PLY;

pub struct Killer {
    killer_moves: [[Move; 2]; MAX_PLY],
}

impl Killer {
    pub fn new() -> Self {
        Self {
            killer_moves: [[Move::default(); 2]; MAX_PLY],
        }
    }

    pub fn save(&mut self, ply: usize, mv: Move) {
        if self.killer_moves[ply][0].is_null() {
            self.killer_moves[ply][0] = mv;
        } else if self.killer_moves[ply][0] != mv {
            self.killer_moves[ply][1] = self.killer_moves[ply][0];
            self.killer_moves[ply][0] = mv;
        }
    }

    pub fn clear(&mut self) {
        for ply in 0..MAX_PLY {
            self.killer_moves[ply][0] = Move::default();
            self.killer_moves[ply][1] = Move::default();
        }
    }

    pub fn get(&self, ply: usize) -> [Move; 2] {
        self.killer_moves[ply]
    }
}