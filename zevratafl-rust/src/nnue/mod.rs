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
    pub acc: [i32; HIDDEN],
    pub w1: [[i32; HIDDEN]; INPUTS],
    pub w2: [i32; HIDDEN],
    pub eval: f32,
}

pub type Weights1 = [[i32; HIDDEN]; INPUTS];
pub type Weights2 = [i32; HIDDEN];

impl NNUE {
    pub fn new(w1: Weights1, w2: Weights2) -> Self {
        NNUE {
            inputs: [0; INPUTS],
            acc: [0; HIDDEN],
            w1,
            w2,
            eval: 0.0,
        }
    }

    #[inline]
    pub fn reset(&mut self) {
        self.inputs = [0; INPUTS];
        self.acc = [0; HIDDEN];
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

    // ===============================
    // Evaluate (scalar version)
    // ===============================

    pub fn evaluate(&self) -> i32 {
        let mut sum: i64 = 0;

        for h in 0..HIDDEN {
            let x = self.acc[h].max(0) as i64;
            let w = self.w2[h] as i64;
            sum += x * w;
        }

        let num = sum * SCALE as i64;
        let den = (QA as i64) * (QB as i64);

        (num / den) as i32
    }

    pub fn clear(&mut self) {
        self.inputs = [0; INPUTS];
        self.acc = [0; HIDDEN];
        self.eval = 0.0;
    }

    pub fn print_weights(&self) {
        for i in 0..INPUTS {
            if self.inputs[i] == 1 {
                println!("Input {}:", i);
            }
        }

        for i in 0..INPUTS {
            for h in 0..HIDDEN {
                if self.w1[i][h] != 0 {
                    println!("W1[{}][{}] = {}", i, h, self.w1[i][h]);
                }
            }
        }

        for h in 0..HIDDEN {
            if self.w2[h] != 0 {
                println!("W2[{}] = {}", h, self.w2[h]);
            }
        }

        // acc print
        for h in 0..HIDDEN {
            if self.acc[h] != 0 {
                println!("Acc[{}] = {}", h, self.acc[h]);
            }
        }
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

    let mut w1 = [[0i32; HIDDEN]; INPUTS];

    for flat in 0..(INPUTS * HIDDEN) {
        let h = flat / INPUTS;  // hidden index (0..31)
        let i = flat % INPUTS;  // input index  (0..362)

        w1[i][h] = (floats[flat] * QA as f32).round() as i32;
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

    let mut w2 = [0i32; HIDDEN];
    for h in 0..HIDDEN {
        w2[h] = (floats[h] * QB as f32).round() as i32;
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
