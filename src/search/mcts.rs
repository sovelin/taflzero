use crate::Board;
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::search_data::SearchData;
use crate::search_root::SearchIterationResponse;
use crate::undo::UndoMove;

type NodeId = usize;

struct MCTSNode {
    mv: Option<Move>,
    children: Vec<NodeId>,
    parent: Option<NodeId>,
    left_moves: Vec<Move>,
    visits: f32,
    wins: f32,
}

impl MCTSNode {
    fn new_root(left_moves: Vec<Move>) -> MCTSNode {
        MCTSNode {
            mv: None,
            parent: None,
            children: vec![],
            left_moves,
            visits: 0.0,
            wins: 0.0,
        }
    }

    fn new_child(mv: Move, parent: NodeId, left_moves: Vec<Move>) -> MCTSNode {
        MCTSNode {
            mv: Some(mv),
            parent: Some(parent),
            children: vec![],
            left_moves,
            visits: 0.0,
            wins: 0.0,
        }
    }

    fn is_fully_expanded(&self) -> bool {
        self.left_moves.is_empty()
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

    fn get_node(&self, id: NodeId) -> &MCTSNode {
        &self.nodes[id]
    }

    fn get_node_mut(&mut self, id: NodeId) -> &mut MCTSNode {
        &mut self.nodes[id]
    }

    fn get_root(&self) -> &MCTSNode {
        &self.nodes[Self::ROOT_ID]
    }

    fn get_root_mut(&mut self) -> &mut MCTSNode {
        &mut self.nodes[Self::ROOT_ID]
    }

    fn get_root_id(&self) -> NodeId {
        Self::ROOT_ID
    }
}

fn get_left_moves(board: &Board, move_gen: &mut MoveGen) -> Vec<Move> {
    move_gen.generate_moves(board);
    move_gen.moves[0..move_gen.count].to_vec()
}

fn uct_select(tree: &MCTSTree, from_id: NodeId) -> NodeId {
    let from = tree.get_node(from_id);
    let mut best_score = f32::NEG_INFINITY;
    let mut best_child: Option<NodeId> = None;

    for id in from.children.iter() {
        let child = tree.get_node(*id);

        if child.visits == 0.0 {
            return *id;
        }

        let q = child.wins / child.visits;
        let c = 1.4f32;

        let ln_parent = from.visits.max(1.0).ln();
        let uct_value = q + c * (ln_parent / child.visits).sqrt();

        if uct_value > best_score {
            best_score = uct_value;
            best_child = Some(*id);
        }
    }

    best_child.expect("No child found!")
}


struct MovesStack {
    undo: Vec<UndoMove>,
}

impl MovesStack {
    fn new() -> Self {
        MovesStack { undo: Vec::new() }
    }

    fn make_move(&mut self, board: &mut Board, mv: Move) {
        let mut undo = UndoMove::new();
        board.make_move(mv, &mut undo).expect("Failed to make move");
        self.undo.push(undo);
    }

    fn unmake_last(&mut self, board: &mut Board) {
        let mut last_mv = self.undo.pop().expect("UndoMove empty");
        board.unmake_move(&mut last_mv).expect("Failed to make undo");
    }

    fn unmake_all(&mut self, board: &mut Board) {
        while self.undo.len() > 0 {
            self.unmake_last(board);
        }
    }
}

pub fn mcts_search(
    board: &mut Board,
    search_data: &mut SearchData,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
) {
    let mut mv_generator = MoveGen::new();
    let left_moves = get_left_moves(board, &mut mv_generator);
    let mut tree = MCTSTree::new(left_moves);

    let mut move_stack = MovesStack::new();


    loop {
        let mut cur = tree.get_root_id();
        // 1) Selection
        while tree.get_node(cur).is_fully_expanded() {
            cur = uct_select(&tree, cur);
            let node = tree.get_node(cur);
            move_stack.make_move(board, node.mv.expect("Move not found"));
        }

        // 2) Expansion
    }
}