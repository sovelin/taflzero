use crate::board::PRECOMPUTED;
use crate::board::constants::SQS;
use crate::board::position_export::BitPosition;
use crate::utils::bfs::bfs;

pub const NUM_PLANES: usize = 11;
pub const POLICY_SIZE: usize = 4840;
pub const SAMPLE_SIZE: usize = NUM_PLANES * SQS;

pub struct NnOutput {
    pub policy: [f32; POLICY_SIZE],
    pub value: f32,
}

pub fn fill_input(input: &mut [f32], pos: &BitPosition) {
    debug_assert!(input.len() == SAMPLE_SIZE);

    // Planes 0-2: unpack bit planes (attackers, defenders, king)
    for plane in 0..3 {
        let base = plane * 16;
        let out_offset = plane * SQS;
        for idx in 0..SQS {
            let byte = idx / 8;
            let bit = idx % 8;
            if (pos.planes[base + byte] >> bit) & 1 == 1 {
                input[out_offset + idx] = 1.0;
            }
        }
    }

    // Plane 3: side to move (all 1s if defenders, stm == 1)
    let stm_val = if pos.stm == 1 { 1.0f32 } else { 0.0f32 };
    let stm_offset = 3 * SQS;
    for i in 0..SQS {
        input[stm_offset + i] = stm_val;
    }

    // Plane 4: throne
    let throne_offset = 4 * SQS;
    input[throne_offset + PRECOMPUTED.throne_sq] = 1.0;

    // Plane 5: corners
    let corners_offset = 5 * SQS;
    for &sq in &PRECOMPUTED.corners_sq {
        input[corners_offset + sq] = 1.0;
    }

    // Plane 6: edge squares
    let edges_offset = 6 * SQS;
    for &sq in &PRECOMPUTED.edges_sq {
        input[edges_offset + sq] = 1.0;
    }

    // Reconstruct piece masks from bit planes for BFS
    let mut king_sq: Option<usize> = None;
    let mut defender_seeds: Vec<usize> = Vec::new();
    let mut group_seeds: Vec<usize> = Vec::new();

    for idx in 0..SQS {
        let byte = idx / 8;
        let bit = idx % 8;
        let is_atk = (pos.planes[byte] >> bit) & 1 == 1;
        let is_def = (pos.planes[16 + byte] >> bit) & 1 == 1;
        let is_king = (pos.planes[32 + byte] >> bit) & 1 == 1;

        if is_king {
            king_sq = Some(idx);
            group_seeds.push(idx);
        } else if is_def {
            defender_seeds.push(idx);
            group_seeds.push(idx);
        }
        let _ = (is_atk, is_def);
    }

    // Plane 7: group BFS — king + all defenders, passable = non-attacker squares
    let group_reach = if !group_seeds.is_empty() {
        bfs(
            |sq| {
                let byte = sq / 8;
                let bit = sq % 8;
                (pos.planes[byte] >> bit) & 1 == 0 // not attacker
            },
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &group_seeds,
        )
    } else {
        [false; SQS]
    };
    let group_offset = 7 * SQS;
    for idx in 0..SQS {
        if group_reach[idx] {
            input[group_offset + idx] = 1.0;
        }
    }

    // Plane 8: king BFS — king only, passable = king's own square or empty (not attacker, not defender)
    let king_reach = if let Some(ksq) = king_sq {
        bfs(
            |sq| {
                let byte = sq / 8;
                let bit = sq % 8;
                let is_atk = (pos.planes[byte] >> bit) & 1 == 1;
                let is_def = (pos.planes[16 + byte] >> bit) & 1 == 1;
                !is_atk && !is_def
            },
            &PRECOMPUTED.vertical_horizontal_neighbors,
            &[ksq],
        )
    } else {
        [false; SQS]
    };
    let king_offset = 8 * SQS;
    // Include king square itself in king BFS plane
    if let Some(ksq) = king_sq {
        input[king_offset + ksq] = 1.0;
    }
    for idx in 0..SQS {
        if king_reach[idx] {
            input[king_offset + idx] = 1.0;
        }
    }

    // Plane 9: repetition x1 — position seen at least once before (rep >= 2)
    let rep1_val = if pos.rep >= 2 { 1.0f32 } else { 0.0 };
    let rep1_offset = 9 * SQS;
    for i in 0..SQS {
        input[rep1_offset + i] = rep1_val;
    }

    // Plane 10: repetition x2 — position seen at least twice before (rep >= 3)
    let rep2_val = if pos.rep >= 3 { 1.0f32 } else { 0.0 };
    let rep2_offset = 10 * SQS;
    for i in 0..SQS {
        input[rep2_offset + i] = rep2_val;
    }
}

pub fn build_input_data(positions: &[&BitPosition]) -> Vec<f32> {
    let batch_size = positions.len();
    let mut input_data = vec![0.0f32; batch_size * SAMPLE_SIZE];

    for (i, pos) in positions.iter().enumerate() {
        let start = i * SAMPLE_SIZE;
        let end = start + SAMPLE_SIZE;
        fill_input(&mut input_data[start..end], pos);
    }

    input_data
}
