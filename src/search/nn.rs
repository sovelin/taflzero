use ndarray::{Array, IxDyn};
use ort::session::Session;
use ort::value::Value;

use crate::Board;
use crate::board::utils::get_square;
use crate::types::{Piece, Side};

const BOARD_DIM: usize = 11;
const NUM_PLANES: usize = 6;
const POLICY_SIZE: usize = 4840;

// Throne: center (5,5). Corners: (0,0), (0,10), (10,0), (10,10).
const THRONE_SQ: usize = 5 * BOARD_DIM + 5;
const CORNER_SQUARES: [usize; 4] = [0, 10, 110, 120];

pub struct NnOutput {
    pub policy: [f32; POLICY_SIZE],
    pub value: f32,
}

pub struct NeuralNet {
    session: Session,
}

impl NeuralNet {
    pub fn new(path: &str) -> Self {
        let session = Session::builder()
            .unwrap()
            .commit_from_file(path)
            .expect("Unable to commit neural net");

        Self { session }
    }

    pub fn evaluate_position(&mut self, board: &Board) -> NnOutput {
        let mut input = [0.0f32; NUM_PLANES * BOARD_DIM * BOARD_DIM];

        for row in 0..BOARD_DIM {
            for col in 0..BOARD_DIM {
                let sq = get_square(row, col);
                let idx = row * BOARD_DIM + col;

                match board.board[sq] {
                    Piece::ATTACKER => input[idx] = 1.0,
                    Piece::DEFENDER => input[BOARD_DIM * BOARD_DIM + idx] = 1.0,
                    Piece::KING => input[2 * BOARD_DIM * BOARD_DIM + idx] = 1.0,
                    Piece::EMPTY => {}
                }
            }
        }

        // Plane 3: side to move (all 1s if attackers)
        let stm_val = if board.side_to_move == Side::ATTACKERS { 1.0f32 } else { 0.0f32 };
        let stm_offset = 3 * BOARD_DIM * BOARD_DIM;
        for i in 0..BOARD_DIM * BOARD_DIM {
            input[stm_offset + i] = stm_val;
        }

        // Plane 4: throne
        let throne_offset = 4 * BOARD_DIM * BOARD_DIM;
        input[throne_offset + THRONE_SQ] = 1.0;

        // Plane 5: corners
        let corners_offset = 5 * BOARD_DIM * BOARD_DIM;
        for &sq in &CORNER_SQUARES {
            input[corners_offset + sq] = 1.0;
        }

        let input_tensor = Array::from_shape_vec(
            IxDyn(&[1, NUM_PLANES, BOARD_DIM, BOARD_DIM]),
            input.to_vec(),
        )
        .unwrap();

        let input_value = Value::from_array(input_tensor).unwrap();
        let outputs = self.session.run(ort::inputs![input_value]).unwrap();

        let (_, policy_data) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let (_, value_data) = outputs[1].try_extract_tensor::<f32>().unwrap();

        let mut policy = [0.0f32; POLICY_SIZE];
        policy.copy_from_slice(&policy_data[..POLICY_SIZE]);

        let value = value_data[0];

        NnOutput { policy, value }
    }
}