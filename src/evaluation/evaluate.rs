use std::cell::RefCell;
use rand::prelude::StdRng;
use rand::{Rng, SeedableRng};
use crate::board::{Board, PRECOMPUTED};
use crate::board::constants::SQS;
use crate::board::types::{Piece, Square};
use crate::evaluation::sided_evaluation;
use crate::moves::POSSIBLE_MOVES_COUNT;

const DEFENDER_WEIGHT: i32 = 150;
const ATTACKER_WEIGHT: i32 = 100;

const DEFENDERS_MOBILITY_SCORES: [i32; 21] = [
	-20, -15, -10, 0, 7, 11, 16, 20, 24, 26, 27, 28, 30, 33, 37, 41, 44, 46, 48, 50, 52,
];

const ATTACKERS_MOBILITY_SCORES: [i32; 21] = [
	-10, -7, -5, 0, 5, 8, 11, 14, 17, 19, 21, 23, 25, 27, 30, 32, 34, 36, 38, 40, 42,
];

const ATTACKERS_DISTANCE_TO_KING_SCORES: [i32; 25] = [
	0, 80, 75, 65, 55, 45, 35, 25, 15, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const DEFENDERS_DISTANCE_TO_KING_SCORES: [i32; 25] = [
	0, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const KING_SURROUNDING_BONUSES: [i32; 5] = [0, 20, 50, 90, 100];

const PSQT_KING: [i16; SQS] = [
	108, 96, 108, 76, 68, 60, 68, 76, 108, 96, 108,
	96, 84, 100, 68, 60, 52, 60, 68, 100, 84, 96,
	108, 100, 116, 84, 76, 68, 76, 84, 116, 100, 108,
	76, 68, 84, 52, 44, 36, 44, 52, 84, 68, 76,
	68, 60, 76, 44, 36, 28, 36, 44, 76, 60, 68,
	60, 52, 68, 36, 28, 0, 28, 36, 68, 52, 60,
	68, 60, 76, 44, 36, 28, 36, 44, 76, 60, 68,
	76, 68, 84, 52, 44, 36, 44, 52, 84, 68, 76,
	108, 100, 116, 84, 76, 68, 76, 84, 116, 100, 108,
	96, 84, 100, 68, 60, 52, 60, 68, 100, 84, 96,
	108, 96, 108, 76, 68, 60, 68, 76, 108, 96, 108,
];

const PSQT_DEF: [i16; SQS] = [
	0, 2, 6, 8, 10, 12, 10, 8, 6, 2, 0,
	2, 6, 10, 14, 16, 18, 16, 14, 10, 6, 2,
	6, 10, 16, 20, 24, 26, 24, 20, 16, 10, 6,
	8, 14, 20, 24, 28, 30, 28, 24, 20, 14, 8,
	10, 16, 24, 28, 32, 34, 32, 28, 24, 16, 10,
	12, 18, 26, 30, 34, 36, 34, 30, 26, 18, 12,
	10, 16, 24, 28, 32, 34, 32, 28, 24, 16, 10,
	8, 14, 20, 24, 28, 30, 28, 24, 20, 14, 8,
	6, 10, 16, 20, 24, 26, 24, 20, 16, 10, 6,
	2, 6, 10, 14, 16, 18, 16, 14, 10, 6, 2,
	0, 2, 6, 8, 10, 12, 10, 8, 6, 2, 0,
];

const PSQT_ATK: [i16; SQS] = [
	6, 2, -4, -8, -12, -14, -12, -8, -4, 2, 6,
	2, -2, -10, -14, -18, -20, -18, -14, -10, -2, 2,
	-4, -10, -18, -22, -26, -28, -26, -22, -18, -10, -4,
	-8, -14, -22, -28, -32, -34, -32, -28, -22, -14, -8,
	-12, -18, -26, -32, -36, -38, -36, -32, -26, -18, -12,
	-14, -20, -28, -34, -38, -40, -38, -34, -28, -20, -14,
	-12, -18, -26, -32, -36, -38, -36, -32, -26, -18, -12,
	-8, -14, -22, -28, -32, -34, -32, -28, -22, -14, -8,
	-4, -10, -18, -22, -26, -28, -26, -22, -18, -10, -4,
	2, -2, -10, -14, -18, -20, -18, -14, -10, -2, 2,
	6, 2, -4, -8, -12, -14, -12, -8, -4, 2, 6,
];

#[inline]
fn possible_moves_count(board: &Board, sq: Square) -> usize {
	let row = PRECOMPUTED.row[sq];
	let col = PRECOMPUTED.col[sq];

	let horizontal = POSSIBLE_MOVES_COUNT[col][board.row_occ[row] as usize] as usize;
	let vertical = POSSIBLE_MOVES_COUNT[row][board.col_occ[col] as usize] as usize;

	horizontal + vertical
}

#[inline]
fn evaluate_defenders_mobility(board: &Board, king_sq: Square) -> i32 {
	let mut result = 0;
	let limit = DEFENDERS_MOBILITY_SCORES.len() - 1;

	for idx in 0..board.defenders_count as usize {
		let moves = possible_moves_count(board, board.defenders[idx]).min(limit);
		result += DEFENDERS_MOBILITY_SCORES[moves];
	}

	let king_moves = possible_moves_count(board, king_sq).min(limit);
	result += DEFENDERS_MOBILITY_SCORES[king_moves] * 2;

	result
}

#[inline]
pub fn evaluate_king_mobility(board: &Board) -> i32 {
	if board.king_sq < 0 {
		return 0;
	}

	let king_sq = board.king_sq as Square;
	possible_moves_count(board, king_sq) as i32
}

thread_local! {
    static RNG: RefCell<StdRng> = RefCell::new(StdRng::seed_from_u64(12345));
}