use crate::Board;
use crate::mcts::mcts::MCTSTree;
use crate::mcts::utils::move_to_policy_index;
use crate::position_export::BitPosition;
use crate::types::Side;
use std::io::{Write, Result};

pub const ACTIONS: usize = 121 * 4 * 10; // 4840
pub const LEGAL_MASK_BYTES: usize = (ACTIONS + 7) / 8; // 605

#[repr(C)]
#[derive(Clone)]
pub struct LegalMask {
    data: [u8; LEGAL_MASK_BYTES],
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

pub struct PendingSample {
    bit_position: BitPosition,
    legal_mask: LegalMask,
    policy: Vec<PolicyTarget>,
    value: i8,
}

impl PendingSample {
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<()> {
        // Keep the exact on-disk format, but serialize in-memory first so each sample
        // is written as a single contiguous block.
        let policy_len = self.policy.len() as u16;
        let total_len = self.bit_position.as_bytes().len()
            + self.legal_mask.as_bytes().len()
            + 2
            + (self.policy.len() * 4)
            + 1;

        let mut buf = Vec::with_capacity(total_len);
        buf.extend_from_slice(self.bit_position.as_bytes());
        buf.extend_from_slice(self.legal_mask.as_bytes());
        buf.extend_from_slice(&policy_len.to_le_bytes());

        for t in &self.policy {
            buf.extend_from_slice(&t.move_index.to_le_bytes());
            buf.extend_from_slice(&t.visits.to_le_bytes());
        }

        buf.push(self.value as u8);
        w.write_all(&buf)
    }

    pub fn set_value_from_result(&mut self, result: Option<Side>) {
        let stm_side = if self.bit_position.stm == 0 { Side::DEFENDERS } else { Side::ATTACKERS };
        self.value = compute_value(stm_side, result);
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
    Format:

    [BitPosition]          49 bytes
    [LegalMask]            605 bytes
    [policy_len]           u16
    [PolicyTarget × N]     N * 4 bytes
    [value]                i8
     */
    pub fn make_pending_sample(&mut self, board: &Board) -> PendingSample {
        let root = self.get_root();
        let mut policy: Vec<PolicyTarget> = vec![];


        for &child_id in &root.children() {
            let node = self.get_node(child_id);
            let visits_f = node.visits();
            let visits_u16 = visits_f.round().min(u16::MAX as f32) as u16;

            if let Some(mv) = node.mv() {
                if visits_u16 == 0 {
                    continue; // skip moves that were not visited
                }

                let move_index = move_to_policy_index(mv);

                policy.push(PolicyTarget {
                    move_index,
                    visits: visits_u16,
                });
            }
        }

        PendingSample {
            bit_position:  BitPosition::from_board(board),
            legal_mask: self.build_legal_mask_from_board(board),
            policy,
            value: 0, // to be set later
        }
    }
}
