use std::num::FpCategory;
use crate::constants::SQS;
use crate::types::{Piece, Square};

pub const INPUTS: usize = 363;
pub const HIDDEN: usize = 32;

pub const QA: i32 = 255;     // quant for layer 1
pub const QB: i32 = 64;      // quant for layer 2
pub const SCALE: i32 = 400;

pub static FC1_RAW: &str = include_str!("../../nnue-gen2/fc1.62.weights.csv");
pub static FC2_RAW: &str = include_str!("../../nnue-gen2/fc2.62.weights.csv");

#[derive(Clone)]
pub struct NNUE {
    pub inputs: [u8; INPUTS],
    pub acc: [f32; HIDDEN],
    pub w1: [[f32; HIDDEN]; INPUTS],
    pub w2: [f32; HIDDEN],
    pub eval: f32,
}

pub type Weights1 = [[f32; HIDDEN]; INPUTS];
pub type Weights2 = [f32; HIDDEN];

impl NNUE {
    pub fn new(w1: Weights1, w2: Weights2) -> Self {
        NNUE {
            inputs: [0; INPUTS],
            acc: [0.0; HIDDEN],
            w1,
            w2,
            eval: 0.0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.inputs = [0; INPUTS];
        self.acc = [0.0; HIDDEN];
        self.eval = 0.0;
    }

    // ===============================
    // Incremental SET/RESET
    // ===============================

    #[inline]
    pub fn set_input(&mut self, idx: usize) {
        if self.inputs[idx] == 1 {
            return;
        }
        self.inputs[idx] = 1;

        let w = &self.w1[idx];
        for h in 0..HIDDEN {
            self.acc[h] += w[h];
        }
    }

    #[inline]
    pub fn reset_input(&mut self, idx: usize) {
        if self.inputs[idx] == 0 {
            return;
        }
        self.inputs[idx] = 0;

        let w = &self.w1[idx];
        for h in 0..HIDDEN {
            self.acc[h] -= w[h];
        }
    }

    pub fn debug_float_full(&self) {
        println!("\n=== ACC (float, до ReLU) ===");

        // ---------- ACC первого слоя ----------
        let mut acc = [0.0f32; HIDDEN];

        for i in 0..INPUTS {
            if self.inputs[i] != 0 {
                let w = &self.w1[i]; // [HIDDEN]
                for h in 0..HIDDEN {
                    acc[h] += w[h];
                }
            }
        }

        for h in 0..HIDDEN {
            println!("Acc[{h}] = {}", acc[h]);
        }

        // ---------- ReLU ----------
        println!("\n=== ReLU(ACC) ===");
        let mut relu_acc = [0.0f32; HIDDEN];

        for h in 0..HIDDEN {
            relu_acc[h] = acc[h].max(0.0);
            println!("ReLU[{h}] = {}", relu_acc[h]);
        }

        // ---------- Второй слой ----------
        println!("\n=== Вклад второго слоя (relu * w2) ===");
        let mut sum_out: f32 = 0.0;

        for h in 0..HIDDEN {
            let contrib = relu_acc[h] * self.w2[h];
            sum_out += contrib;
            println!(
                "h={}: relu={}, w2={}, contrib={}, running_sum={}",
                h,
                relu_acc[h],
                self.w2[h],
                contrib,
                sum_out
            );
        }
    }

    // ---------- Итог ----------


        // ===============================
    // Evaluate (scalar version)
    // ===============================


    pub fn debug_full_recompute(&self) {
        let mut acc = [0.0; HIDDEN];

        for i in 0..INPUTS {
            if self.inputs[i] == 1 {
                let w = &self.w1[i];
                for h in 0..HIDDEN {
                    acc[h] += w[h];
                }
            }
        }

        println!("-- ACC FULL RECOMPUTE --");
        for h in 0..HIDDEN {
            println!("AccFull[{h}] = {}", acc[h]);
        }
    }

    pub fn evaluate(&self) -> i32 {
        let mut sum: f32 = 0.0;

        for h in 0..HIDDEN {
            let x = self.acc[h].max(0.0);
            let w = self.w2[h];
            sum += x * w;
        }
        
        (sum * SCALE as f32) as i32

        // let den = (QA as i64) * (QB as i64);
        // (num / den) as i32
    }

    pub fn clear(&mut self) {
        self.inputs = [0; INPUTS];
        self.acc = [0.0; HIDDEN];
        self.eval = 0.0;
    }
}

pub fn calculate_nnue_index(piece: Piece, square: Square) -> usize {
    let match_piece = match piece {
        Piece::ATTACKER => 0,
        Piece::DEFENDER => 1,
        Piece::KING => 2,
        Piece::EMPTY => panic!("Empty piece in NNUE index calculation"),
    };

    match_piece * SQS + square
}

pub fn load_fc1(text: &str) -> Weights1 {
    let floats: Vec<f32> = text
        .split(|c| c == ',' || c == '\n' || c == '\r')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>().expect("Bad float"))
        .collect();

    assert!(
        floats.len() == INPUTS * HIDDEN,
        "Wrong FC1 size: expected {}, got {}",
        INPUTS * HIDDEN,
        floats.len()
    );

    let mut w1 = [[0.0; HIDDEN]; INPUTS];

    for flat in 0..(INPUTS * HIDDEN) {
        let h = flat / INPUTS;  // hidden index (0..31)
        let i = flat % INPUTS;  // input index  (0..362)

        w1[i][h] = floats[flat];
        println!("i:{}; h:{}, w: {}", i, h, floats[flat]);
    }

    w1
}

pub fn load_fc2(text: &str) -> Weights2 {
    let floats: Vec<f32> = text
        .split(|c| c == ',' || c == '\n' || c == '\r')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>().expect("Bad float"))
        .collect();


    assert_eq!(floats.len(), HIDDEN);

    let mut w2 = [0.0; HIDDEN];
    for h in 0..HIDDEN {
        w2[h] = floats[h];
    }

    w2
}

pub fn load_fc1_single_line(path: &str) -> Weights1 {
    use std::fs;

    let text = fs::read_to_string(path).unwrap();

    load_fc1(&text)
}


pub fn load_fc2_single_line(path: &str) -> Weights2 {
    use std::fs;

    let text = fs::read_to_string(path).unwrap();

    load_fc2(&text)
}

pub fn load_fc1_from_raw() -> Weights1 {
    load_fc1(FC1_RAW)
}

pub fn load_fc2_from_raw() -> Weights2 {
    println!("{}", FC2_RAW);
    load_fc2(FC2_RAW)
}
