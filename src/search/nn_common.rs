use crate::board::constants::SQS;
use crate::PRECOMPUTED;
use crate::position_export::BitPosition;

pub const NUM_PLANES: usize = 6;
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

    // Plane 3: side to move (all 1s if attackers, stm == 1)
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
