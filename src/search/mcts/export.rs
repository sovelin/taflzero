use crate::board::Board;
use crate::board::position_export::BitPosition;
use crate::board::types::Side;
use crate::mcts::mcts::MCTSTree;
use crate::mcts::utils::move_to_policy_index;
use std::io::{Result, Write};

pub const ACTIONS: usize = 121 * 4 * 10; // 4840
pub const LEGAL_MASK_BYTES: usize = ACTIONS.div_ceil(8); // 605

#[repr(C)]
#[derive(Clone)]
pub struct LegalMask {
    data: [u8; LEGAL_MASK_BYTES],
}

impl Default for LegalMask {
    fn default() -> Self {
        Self::new()
    }
}

impl LegalMask {
    pub fn new() -> Self {
        Self {
            data: [0u8; LEGAL_MASK_BYTES],
        }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    #[inline]
    pub fn set(&mut self, action_index: usize) {
        debug_assert!(action_index < ACTIONS);

        let byte = action_index / 8;
        let bit = action_index % 8;

        self.data[byte] |= 1 << bit;
    }

    #[inline]
    pub fn is_set(&self, action_index: usize) -> bool {
        debug_assert!(action_index < ACTIONS);

        let byte = action_index / 8;
        let bit = action_index % 8;

        (self.data[byte] >> bit) & 1 == 1
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
}

fn compute_value(side_to_move: Side, result: Option<Side>) -> i8 {
    match result {
        None => 0, // draw / cutoff
        Some(winner) => {
            if winner == side_to_move {
                1
            } else {
                -1
            }
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct PolicyTarget {
    move_index: u16,
    visits: u16,
}

/// flags bit 0: policy target is valid (full search, not a cheap PCR search)
pub const FLAG_POLICY_VALID: u8 = 1 << 0;
/// flags bit 1: this sample is the last move of its game
pub const FLAG_LAST_OF_GAME: u8 = 1 << 1;

/// king_corner value when the game did not end with a corner escape
pub const KING_CORNER_NONE: u8 = 255;

/// Corner squares in the canonical order used for the king_corner target index.
pub const CORNER_SQUARES: [isize; 4] = [0, 10, 110, 120];

pub fn king_corner_index(king_sq: isize) -> u8 {
    CORNER_SQUARES
        .iter()
        .position(|&sq| sq == king_sq)
        .map(|i| i as u8)
        .unwrap_or(KING_CORNER_NONE)
}

pub struct PendingSample {
    bit_position: BitPosition,
    legal_mask: LegalMask,
    policy: Vec<PolicyTarget>,
    value: i8,
    /// MCTS root value estimate from the side-to-move perspective,
    /// quantized to [-127, 127]. Used for value target bootstrapping.
    root_q: i8,
    /// FLAG_* bits.
    flags: u8,
    /// Corner index (0-3) the king escaped through, KING_CORNER_NONE otherwise.
    king_corner: u8,
}

impl PendingSample {
    pub fn from_manual(
        bit_position: BitPosition,
        legal_mask: LegalMask,
        policy: Vec<(u16, u16)>,
        value: i8,
        root_q: i8,
    ) -> Self {
        let policy = policy
            .into_iter()
            .map(|(move_index, visits)| PolicyTarget { move_index, visits })
            .collect();

        PendingSample {
            bit_position,
            legal_mask,
            policy,
            value,
            root_q,
            flags: FLAG_POLICY_VALID,
            king_corner: KING_CORNER_NONE,
        }
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        // Keep the exact on-disk format, but serialize in-memory first so each sample
        // is written as a single contiguous block.
        let policy_len = self.policy.len() as u16;
        let total_len = self.bit_position.as_bytes().len()
            + self.legal_mask.as_bytes().len()
            + 2
            + (self.policy.len() * 4)
            + 4;

        let mut buf = Vec::with_capacity(total_len);
        buf.extend_from_slice(self.bit_position.as_bytes());
        buf.extend_from_slice(self.legal_mask.as_bytes());
        buf.extend_from_slice(&policy_len.to_le_bytes());

        for t in &self.policy {
            buf.extend_from_slice(&t.move_index.to_le_bytes());
            buf.extend_from_slice(&t.visits.to_le_bytes());
        }

        buf.push(self.value as u8);
        buf.push(self.root_q as u8);
        buf.push(self.flags);
        buf.push(self.king_corner);
        w.write_all(&buf)
    }

    pub fn set_value_from_result(&mut self, result: Option<Side>) {
        let stm_side = if self.bit_position.stm == 0 {
            Side::ATTACKERS
        } else {
            Side::DEFENDERS
        };
        self.value = compute_value(stm_side, result);
    }

    pub fn set_policy_valid(&mut self, valid: bool) {
        if valid {
            self.flags |= FLAG_POLICY_VALID;
        } else {
            self.flags &= !FLAG_POLICY_VALID;
        }
    }

    pub fn set_last_of_game(&mut self) {
        self.flags |= FLAG_LAST_OF_GAME;
    }

    pub fn set_king_corner(&mut self, corner: u8) {
        self.king_corner = corner;
    }

    pub fn bit_position(&self) -> &BitPosition {
        &self.bit_position
    }
}

impl MCTSTree {
    fn build_legal_mask_from_board(&mut self, board: &Board) -> LegalMask {
        let mut legal_mask = LegalMask::new();
        self.move_gen.generate_moves(board);

        for i in 0..self.move_gen.count {
            let mv = self.move_gen.moves[i];
            let move_index = move_to_policy_index(mv);
            legal_mask.set(move_index as usize);
        }

        legal_mask
    }

    /*
    Format (v2):

    [BitPosition]          50 bytes (48 planes + stm + rep)
    [LegalMask]            605 bytes
    [policy_len]           u16
    [PolicyTarget × N]     N * 4 bytes
    [value]                i8
    [root_q]               i8  (MCTS root Q × 127, stm perspective)
    [flags]                u8  (bit0 policy_valid, bit1 last_of_game)
    [king_corner]          u8  (0-3 escape corner index, 255 none)
     */
    /// Build a training sample from the current root.
    ///
    /// `prune_forced_k` > 0 enables KataGo-style policy target pruning: for every
    /// non-best child, up to n_forced = sqrt(k * P * N) visits are subtracted —
    /// but never below the visit count at which the child's PUCT score would
    /// exceed the best child's (i.e. visits the move earned on merit are kept).
    /// Children pruned below 1 visit are dropped from the target entirely.
    pub fn make_pending_sample(
        &mut self,
        board: &Board,
        prune_forced_k: f32,
        c_puct: f32,
    ) -> PendingSample {
        let root = self.get_root();

        // Child wins are stored from the root side-to-move perspective, so the
        // visit-weighted average over children is the search value estimate.
        let mut wins_sum = 0.0f32;
        let mut visits_sum = 0.0f32;
        // (move_index, visits, q, prior) per visited child
        let mut entries: Vec<(u16, f32, f32, f32)> = vec![];

        for &child_id in root.children() {
            let node = self.get_node(child_id);
            let visits_f = node.visits();

            wins_sum += node.wins();
            visits_sum += visits_f;

            if let Some(mv) = node.mv() {
                if visits_f < 0.5 {
                    continue;
                }
                let q = node.wins() / visits_f;
                entries.push((move_to_policy_index(mv), visits_f, q, node.prior()));
            }
        }

        let root_q = if visits_sum > 0.0 {
            ((wins_sum / visits_sum) * 127.0)
                .round()
                .clamp(-127.0, 127.0) as i8
        } else {
            0
        };

        if prune_forced_k > 0.0 && entries.len() > 1 {
            let sqrt_n = visits_sum.sqrt();
            let best_idx = entries
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.1.partial_cmp(&b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();
            let (_, best_v, best_q, best_p) = entries[best_idx];
            let uct_best = best_q + c_puct * best_p * sqrt_n / (1.0 + best_v);

            for (i, entry) in entries.iter_mut().enumerate() {
                if i == best_idx {
                    continue;
                }
                let (_, v, q, p) = *entry;
                let n_forced = (prune_forced_k * p * visits_sum).sqrt();
                // Minimum visits this child keeps: below this its PUCT score
                // would exceed the best move's, meaning the visits were earned.
                let v_min = if uct_best > q {
                    (c_puct * p * sqrt_n / (uct_best - q) - 1.0).max(0.0)
                } else {
                    // Child's Q alone beats best's PUCT — nothing was forced.
                    v
                };
                let subtract = n_forced.min((v - v_min).max(0.0));
                entry.1 = v - subtract;
            }
        }

        let policy: Vec<PolicyTarget> = entries
            .iter()
            .filter(|(_, v, _, _)| *v >= 1.0)
            .map(|&(move_index, v, _, _)| PolicyTarget {
                move_index,
                visits: v.round().min(u16::MAX as f32) as u16,
            })
            .filter(|t| t.visits > 0)
            .collect();

        let rep = board.rep_table.get(&board.zobrist).copied().unwrap_or(1);

        PendingSample {
            bit_position: BitPosition::from_board(board, rep),
            legal_mask: self.build_legal_mask_from_board(board),
            policy,
            value: 0,
            root_q,
            flags: FLAG_POLICY_VALID,
            king_corner: KING_CORNER_NONE,
        }
    }
}
