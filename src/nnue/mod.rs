use once_cell::sync::Lazy;
use std::sync::Arc;
use crate::board::constants::SQS;
use crate::types::{Piece, Square};

pub const INPUTS: usize = 364;
pub const HIDDEN: usize = 256;

pub const QA: i32 = 1000; // quant for layer 1
pub const QB: i32 = 600; // quant for layer 2
pub const SCALE: i32 = 400;
pub const STM_BIT: usize = INPUTS - 1;

pub static FC1_RAW: &str = include_str!("../../nnue-324x256-gen4/fc1.25.weights.csv");
pub static FC2_RAW: &str = include_str!("../../nnue-324x256-gen4/fc2.25.weights.csv");

// pub static FC1_RAW: &str = include_str!("../../nnue-fair-gen6-324x32/fc1.23.weights.csv");
// pub static FC2_RAW: &str = include_str!("../../nnue-fair-gen6-324x32/fc2.23.weights.csv");

// pub static FC1_RAW: &str = include_str!("../../nnue-fair-gen7-324x32/fc1.19.weights.csv");
// pub static FC2_RAW: &str = include_str!("../../nnue-fair-gen7-324x32/fc2.19.weights.csv");

// pub static FC1_RAW: &str = include_str!("../../nnue-fair-gen9-324x64/fc1.7.weights.csv");
// pub static FC2_RAW: &str = include_str!("../../nnue-fair-gen9-324x64/fc2.7.weights.csv");

// pub static FC1_RAW: &str = include_str!("../../nnue-fair-gen10-324x64/fc1.18.weights.csv");
// pub static FC2_RAW: &str = include_str!("../../nnue-fair-gen10-324x64/fc2.18.weights.csv");

// pub static FC1_RAW: &str = include_str!("../../nnue-fair-gen11-324x64/fc1.57.weights.csv");
// pub static FC2_RAW: &str = include_str!("../../nnue-fair-gen11-324x64/fc2.57.weights.csv");

pub type Weights1 = Arc<[i16]>;
pub type Weights2 = Arc<[i16]>;

static DEFAULT_W1: Lazy<Weights1> = Lazy::new(|| load_fc1(FC1_RAW));
static DEFAULT_W2: Lazy<Weights2> = Lazy::new(|| load_fc2(FC2_RAW));

pub fn load_default_weights() -> (Weights1, Weights2) {
    (DEFAULT_W1.clone(), DEFAULT_W2.clone())
}

#[derive(Clone)]
pub struct NNUE {
    pub inputs: [u8; INPUTS],
    pub acc: [i32; HIDDEN],
    pub w1: Weights1,
    pub w2: Weights2,
    pub eval: i32,
}

impl NNUE {
    pub fn new(w1: Weights1, w2: Weights2) -> Self {
        NNUE {
            inputs: [0; INPUTS],
            acc: [0; HIDDEN],
            w1,
            w2,
            eval: 0,
        }
    }

    #[inline]
    fn w1_row(weights: &Weights1, idx: usize) -> &[i16] {
        let start = idx * HIDDEN;
        &weights[start..start + HIDDEN]
    }

    #[inline]
    pub fn reset(&mut self) {
        self.inputs = [0; INPUTS];
        self.acc = [0; HIDDEN];
        self.eval = 0;
    }

    #[inline]
    pub fn set_input(&mut self, idx: usize) {
        if self.inputs[idx] == 1 {
            return;
        }
        self.inputs[idx] = 1;

        let values = Self::w1_row(&self.w1, idx);
        for h in 0..HIDDEN {
            self.acc[h] += values[h] as i32;
        }
    }

    #[inline]
    pub fn reset_input(&mut self, idx: usize) {
        if self.inputs[idx] == 0 {
            return;
        }
        self.inputs[idx] = 0;

        let values = Self::w1_row(&self.w1, idx);
        for h in 0..HIDDEN {
            self.acc[h] -= values[h] as i32;
        }
    }

    pub fn evaluate(&self) -> i32 {
        let mut sum: i64 = 0;

        for h in 0..HIDDEN {
            let x = self.acc[h].max(0) as i64;
            sum += x * self.w2[h] as i64;
        }

        ((sum * SCALE as i64) / ((QA as i64) * (QB as i64))) as i32
    }

    pub fn clear(&mut self) {
        self.inputs = [0; INPUTS];
        self.acc = [0; HIDDEN];
        self.eval = 0;
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

fn parse_csv_floats(text: &str, expected: usize) -> Vec<f32> {
    let floats: Vec<f32> = text
        .split(|c| c == ',' || c == '\n' || c == '\r')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<f32>().expect("Bad float"))
        .collect();

    assert_eq!(
        floats.len(),
        expected,
        "Wrong weights size: expected {expected}, got {}",
        floats.len()
    );

    floats
}

fn quantize_to_i16(value: f32, scale: i32) -> i16 {
    let scaled = (value * scale as f32).round();
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

pub fn load_fc1(text: &str) -> Weights1 {
    let floats = parse_csv_floats(text, INPUTS * HIDDEN);

    let mut out = vec![0i16; INPUTS * HIDDEN];

    for flat in 0..(INPUTS * HIDDEN) {
        let h = flat / INPUTS;
        let i = flat % INPUTS;
        out[i * HIDDEN + h] = quantize_to_i16(floats[flat], QA);
    }

    Arc::from(out.into_boxed_slice())
}

pub fn load_fc2(text: &str) -> Weights2 {
    let floats = parse_csv_floats(text, HIDDEN);
    let mut out = vec![0i16; HIDDEN];

    for h in 0..HIDDEN {
        out[h] = quantize_to_i16(floats[h], QB);
    }

    Arc::from(out.into_boxed_slice())
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_fc1_from_file(path: &str) -> Weights1 {
    use std::fs;

    let text = fs::read_to_string(path).unwrap();
    load_fc1(&text)
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_fc2_from_file(path: &str) -> Weights2 {
    use std::fs;

    let text = fs::read_to_string(path).unwrap();
    load_fc2(&text)
}

pub fn load_fc1_from_raw() -> Weights1 {
    DEFAULT_W1.clone()
}

pub fn load_fc2_from_raw() -> Weights2 {
    DEFAULT_W2.clone()
}
