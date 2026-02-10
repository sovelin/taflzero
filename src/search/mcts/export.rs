use crate::mcts::mcts::MCTSTree;

impl MCTSTree {
    pub save_root_position_for_training(&self) -> String {
        self.root.board.get_fen()
    }
}