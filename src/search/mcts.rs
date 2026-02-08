use crate::Board;
use crate::mv::Move;
use crate::search_data::SearchData;
use crate::search_root::SearchIterationResponse;

type NodeId = usize;

struct MCTSNode {
    mv: Option<Move>,
    children: Vec<NodeId>,
    parent: Option<NodeId>,
    left_moves: Vec<Move>,
    visits: usize,
    wins: f32,
}

impl MCTSNode {
    fn new_root(left_moves: Vec<Move>) -> MCTSNode {
        MCTSNode {
            mv: None,
            parent: None,
            children: vec![],
            left_moves,
            visits: 0,
            wins: 0.0,
        }
    }

    fn new_child(mv: Move, parent: NodeId, left_moves: Vec<Move>) -> MCTSNode {
        MCTSNode {
            mv: Some(mv),
            parent: Some(parent),
            children: vec![],
            left_moves,
            visits: 0,
            wins: 0.0,
        }
    }

    fn not_fully_expanded(&self) -> bool {
        !self.left_moves.is_empty()
    }
}

struct MCTSTree {
    nodes: Vec<MCTSNode>,
}

impl MCTSTree {
    const ROOT_ID: NodeId = 0;

    fn new(left_moves: Vec<Move>) -> Self {
        MCTSTree { nodes: vec![
            MCTSNode::new_root(left_moves),
        ] }
    }

    fn get_node(&mut self, id: NodeId) -> &mut MCTSNode {
        &mut self.nodes[id]
    }

    fn get_root(&mut self) -> &mut MCTSNode {
        &mut self.nodes[Self::ROOT_ID]
    }
}

pub fn mcts_search(
    board: &mut Board,
    search_data: &mut SearchData,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
) {
    // go to the three

    while true {
        // 1) Selection
    }
}