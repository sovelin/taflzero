use crate::board::{Board, PRECOMPUTED};
use crate::board::types::OptionalSquare;
use crate::board::utils::{get_col, get_row};
use crate::masks::LINE_MOVES;
use crate::moves::movegen::MoveGen;
use crate::moves::mv::Move;
use crate::POSSIBLE_MOVES_COUNT;
use crate::search::history::History;
use crate::search::killer::Killer;
use crate::search::king_mobility_change::king_mobility_change;
use crate::types::Side;

static HASH_MOVE_BONUS: i32 = 1000000;
static KILLER_MOVE_BONUS: i32 = 900000;
static SECOND_KILLER_MOVE_BONUS: i32 = 800000;

impl MoveGen {
    pub fn order_moves(&mut self, board: &Board, tt_move: Move, killers: &Killer, history: &History, ply: usize) {
        for i in 0..self.count() {
            self.move_scores[i] = 0;


            let from = self.moves[i].from();
            let to = self.moves[i].to();

            // mate moves get the highest priority
            if board.king_sq == from as OptionalSquare && PRECOMPUTED.corners_sq.contains(&to) {
                self.move_scores[i] += 2000000;
            }

            if self.moves[i] == tt_move {
                self.move_scores[i] += 1000000;
            }

            let killer_moves = killers.get(ply);

            if killer_moves[0] == self.moves[i] {
                self.move_scores[i] += KILLER_MOVE_BONUS;
            } else if killer_moves[1] == self.moves[i] {
                self.move_scores[i] += SECOND_KILLER_MOVE_BONUS;
            }

            self.move_scores[i] += history.get(board.side_to_move, from, to) / 1000;
        }
    }

    pub fn pick_move(&mut self) -> Option<Move> {
        if self.count == 0 {
            return None;
        }

        let mut best_index = 0;
        let mut best_score = self.move_scores[0];

        for i in 1..self.count {
            if self.move_scores[i] > best_score {
                best_score = self.move_scores[i];
                best_index = i;
            }
        }

        let best_move = self.moves[best_index];


        self.moves[best_index] = self.moves[self.count - 1];
        self.move_scores[best_index] = self.move_scores[self.count - 1];

        self.count -= 1;
        Some(best_move)
    }
}

#[cfg(test)]
mod tests {
    use crate::moves::movegen::MoveGen;
    use crate::moves::mv::{create_move_from_algebraic};

    #[test]
    fn test_ordering() {
        let mut move_gen = MoveGen::new();

        move_gen.moves[0] = create_move_from_algebraic("a1a2").unwrap();
        move_gen.moves[1] = create_move_from_algebraic("b1b2").unwrap();
        move_gen.moves[2] = create_move_from_algebraic("c1c2").unwrap();
        move_gen.count = 3;

        move_gen.move_scores[0] = 10;
        move_gen.move_scores[1] = 30;
        move_gen.move_scores[2] = 20;

        let mv = move_gen.pick_move();
        assert_eq!(mv, Some(create_move_from_algebraic("b1b2").unwrap()));

        let mv = move_gen.pick_move();
        assert_eq!(mv, Some(create_move_from_algebraic("c1c2").unwrap()));

        let mv = move_gen.pick_move();
        assert_eq!(mv, Some(create_move_from_algebraic("a1a2").unwrap()));

        let mv = move_gen.pick_move();
        assert_eq!(mv, None);
    }}