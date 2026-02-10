use rand::prelude::StdRng;
use rand::Rng;
use crate::Board;
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::search_data::SearchData;
use crate::search_root::SearchIterationResponse;
use crate::terminal::check_terminal;
use crate::types::Side;
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

    fn append_child(&mut self, node: NodeId) {
        self.children.push(node);
    }

    fn remove_left_move(&mut self, mv: Move) {
        if let Some(pos) = self.left_moves.iter().position(|&m| m == mv) {
            self.left_moves.remove(pos);
        }
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

    fn new_child(&mut self, mv: Move, parent_id: NodeId, left_moves: Vec<Move>) -> NodeId {
        let index: NodeId = self.nodes.len();
        let new_child = MCTSNode::new_child(mv, parent_id, left_moves);
        self.nodes.push(new_child);
        let parent = self.get_node_mut(parent_id);
        parent.append_child(index);
        index
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
        let c = 3f32;

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

fn select_random_move(move_gen: &mut MoveGen, rnd_gen: &mut StdRng) -> Move {
    let idx = rnd_gen.gen_range(0..move_gen.count);
    move_gen.moves[idx]
}

fn rollout(board: &mut Board, move_gen: &mut MoveGen, rnd_gen: &mut StdRng, limit: usize) -> Option<Side> {
    let mut stack = MovesStack::new();
    let mut res: Option<Side> = None;

    let mut iteration = 0;
    loop {
        iteration += 1;
        let is_terminal = check_terminal(board);

        if let Some(x) = is_terminal {
            res = is_terminal;
            break;
        }

        move_gen.generate_moves(board);

        if move_gen.count == 0 {
            res = Some(if board.side_to_move == Side::ATTACKERS {Side::DEFENDERS} else {Side::ATTACKERS});
            break;
        }

        let mv = select_random_move(move_gen, rnd_gen);
        stack.make_move(board, mv);

        if iteration >= limit {
            stack.unmake_all(board);
            return None;
        }
    }

    stack.unmake_all(board);
    Some(res.expect("No side found"))
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

    let mut iteration = 0;

    loop {
        iteration += 1;
        let mut cur = tree.get_root_id();
        // 1) Selection
        while tree.get_node(cur).is_fully_expanded() && !tree.get_node(cur).children.is_empty() {
            cur = uct_select(&tree, cur);
            let node = tree.get_node(cur);
            move_stack.make_move(board, node.mv.expect("Move not found"));
        }

        // 2) Expansion
        let is_terminal = check_terminal(board);

        let result = if let Some(x) = is_terminal {
            Some(x)
        } else {
            let node = tree.get_node_mut(cur);

            if node.left_moves.is_empty() {
                panic!("No moves left to expand!");
            }

            let next_mv = node.left_moves[0];
            move_stack.make_move(board, next_mv);
            let left_moves = get_left_moves(board, &mut mv_generator);
            node.remove_left_move(next_mv);
            cur = tree.new_child(next_mv, cur, left_moves);

            // 3) Rollouts
            rollout(board, &mut mv_generator, &mut search_data.random_generator, 500000)
        };

        let leaf_player = if board.side_to_move == Side::ATTACKERS {
            Side::DEFENDERS
        } else {
            Side::ATTACKERS
        };

        let mut value = match result {
            Some(w) => if w == leaf_player { 1.0 } else { 0.0 },
            None => 0.5,
        };




        // 4) Backpropagation
        let mut is_reversed = false;

        while cur != tree.get_root_id() {
            move_stack.unmake_last(board);
            let node = tree.get_node_mut(cur);
            node.visits += 1.0;
            node.wins += if is_reversed { 1.0 - value } else { value };
            cur = node.parent.expect("Parent not found");
            is_reversed = !is_reversed;
        }

        // 5) print all
        if iteration % 1000 == 0 {
            let root = tree.get_root();

            let top_n = 10;

            let mut children: Vec<NodeId> = root.children.clone();
            children.sort_by(|&a, &b| {
                let va = tree.get_node(a).visits;
                let vb = tree.get_node(b).visits;
                vb.partial_cmp(&va).unwrap()
            });

            for (i, &child_id) in children.iter().take(top_n).enumerate() {
                let node = tree.get_node(child_id);
                let visits = node.visits;
                let score = if visits > 0.0 {
                    node.wins / visits
                } else {
                    0.0
                };

                println!(
                    "#{:<2} visits={:<8.0} score={:.3} move={:?}",
                    i + 1,
                    visits,
                    score,
                    node.mv
                );
            }

        }
    }
}