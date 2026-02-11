use crate::Board;
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::position_export::BitPosition;
use crate::search::mcts::utils::move_to_policy_index;
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::search_root::SearchIterationResponse;
use crate::terminal::check_terminal;
use crate::undo::UndoMove;

type NodeId = usize;

pub struct MCTSNode {
    mv: Option<Move>,
    children: Vec<NodeId>,
    parent: Option<NodeId>,
    expanded: bool,
    visits: f32,
    wins: f32,
    prior: f32,
}

impl MCTSNode {
    fn new_root() -> MCTSNode {
        MCTSNode {
            mv: None,
            parent: None,
            children: vec![],
            expanded: false,
            visits: 0.0,
            wins: 0.0,
            prior: 0.0,
        }
    }

    pub fn children(&self) -> &Vec<NodeId> {
        &self.children
    }

    pub fn visits(&self) -> f32 {
        self.visits
    }

    pub fn mv(&self) -> Option<Move> {
        self.mv
    }

    fn new_child(mv: Move, parent: NodeId, prior: f32) -> MCTSNode {
        MCTSNode {
            mv: Some(mv),
            parent: Some(parent),
            children: vec![],
            expanded: false,
            visits: 0.0,
            wins: 0.0,
            prior,
        }
    }

    fn is_leaf(&self) -> bool {
        !self.expanded
    }

    fn append_child(&mut self, node: NodeId) {
        self.children.push(node);
    }
}

pub struct MCTSTree {
    nodes: Vec<MCTSNode>,
    pub move_gen: MoveGen,
}

impl MCTSTree {
    const ROOT_ID: NodeId = 0;

    pub fn new() -> Self {
        MCTSTree { nodes: vec![
            MCTSNode::new_root(),
        ], move_gen: MoveGen::new() }
    }

    pub fn get_node(&self, id: NodeId) -> &MCTSNode {
        &self.nodes[id]
    }

    fn get_node_mut(&mut self, id: NodeId) -> &mut MCTSNode {
        &mut self.nodes[id]
    }

    pub fn get_root(&self) -> &MCTSNode {
        &self.nodes[Self::ROOT_ID]
    }

    fn get_root_mut(&mut self) -> &mut MCTSNode {
        &mut self.nodes[Self::ROOT_ID]
    }

    fn get_root_id(&self) -> NodeId {
        Self::ROOT_ID
    }

    fn new_child(&mut self, mv: Move, parent_id: NodeId, prior: f32) -> NodeId {
        let index: NodeId = self.nodes.len();
        let new_child = MCTSNode::new_child(mv, parent_id, prior);
        self.nodes.push(new_child);
        let parent = self.get_node_mut(parent_id);
        parent.append_child(index);
        index
    }
}

pub fn get_left_moves(board: &Board, move_gen: &mut MoveGen) -> Vec<Move> {
    move_gen.generate_moves(board);
    move_gen.moves[0..move_gen.count].to_vec()
}

fn puct_select(tree: &MCTSTree, from_id: NodeId) -> NodeId {
    let from = tree.get_node(from_id);
    let mut best_score = f32::NEG_INFINITY;
    let mut best_child: Option<NodeId> = None;

    let sqrt_parent = from.visits.sqrt();
    let c = 1.4f32;

    for id in from.children.iter() {
        let child = tree.get_node(*id);

        let q = if child.visits > 0.0 { child.wins / child.visits } else { 0.0 };
        let puct_value = q + c * child.prior * sqrt_parent / (1.0 + child.visits);

        if puct_value > best_score {
            best_score = puct_value;
            best_child = Some(*id);
        }
    }

    best_child.expect("No child found!")
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|&x| x / sum).collect()
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

fn expand_node(
    board: &mut Board,
    tree: &mut MCTSTree,
    node_id: NodeId,
    nn: &mut NeuralNet,
    move_gen: &mut MoveGen,
) -> f32 {
    let position = BitPosition::from_board(board);
    let nn_out = nn.evaluate_position(&position);

    let moves = get_left_moves(board, move_gen);

    if !moves.is_empty() {
        let logits: Vec<f32> = moves.iter()
            .map(|mv| nn_out.policy[move_to_policy_index(*mv) as usize])
            .collect();
        let priors = softmax(&logits);

        for (i, &mv) in moves.iter().enumerate() {
            tree.new_child(mv, node_id, priors[i]);
        }
    }

    tree.get_node_mut(node_id).expanded = true;

    nn_out.value
}

#[allow(dead_code)]
fn debug_print_top_moves(tree: &MCTSTree, top_n: usize) {
    let root = tree.get_root();
    let mut children: Vec<NodeId> = root.children.clone();
    children.sort_by(|&a, &b| {
        let va = tree.get_node(a).visits;
        let vb = tree.get_node(b).visits;
        vb.partial_cmp(&va).unwrap()
    });

    for (i, &child_id) in children.iter().take(top_n).enumerate() {
        let node = tree.get_node(child_id);
        let visits = node.visits;
        let score = if visits > 0.0 { node.wins / visits } else { 0.0 };

        println!(
            "#{:<2} visits={:<8.0} score={:.3} prior={:.3} move={:?}",
            i + 1, visits, score, node.prior, node.mv
        );
    }
}

fn get_best_child(tree: &MCTSTree) -> Option<NodeId> {
    let root = tree.get_root();
    root.children.iter()
        .max_by(|&&a, &&b| {
            let va = tree.get_node(a).visits;
            let vb = tree.get_node(b).visits;
            va.partial_cmp(&vb).unwrap()
        })
        .copied()
}

pub fn mcts_search(
    board: &mut Board,
    tree: &mut MCTSTree,
    nn: &mut NeuralNet,
    search_data: &mut SearchData,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
) -> Option<Move> {
    let mut mv_generator = MoveGen::new();
    let mut move_stack = MovesStack::new();
    let mut iteration: u64 = 0;
    let mut last_report_ms: u64 = 0;

    // Expand root
    if tree.get_root().is_leaf() {
        expand_node(board, tree, MCTSTree::ROOT_ID, nn, &mut mv_generator);
    }

    loop {
        // Check time limit
        if search_data.time_exceeded() {
            break;
        }

        iteration += 1;
        let mut cur = tree.get_root_id();

        // 1) Selection — descend using PUCT until we hit a leaf
        while !tree.get_node(cur).is_leaf() && !tree.get_node(cur).children.is_empty() {
            cur = puct_select(&tree, cur);
            let node = tree.get_node(cur);
            move_stack.make_move(board, node.mv.expect("Move not found"));
        }

        // 2) Expansion + Evaluation
        let is_terminal = check_terminal(board);

        let mut result: f32 = if let Some(x) = is_terminal {
            tree.get_node_mut(cur).expanded = true;
            if board.side_to_move == x {
                1.0
            } else {
                -1.0
            }
        } else if tree.get_node(cur).children.is_empty() && tree.get_node(cur).expanded {
            // No legal moves — loss
            -1.0
        } else {
            expand_node(board, tree, cur, nn, &mut mv_generator)
        };

        // 3) Backpropagation

        loop {
            result = -result;
            let parent = {
                let node = tree.get_node_mut(cur);
                node.visits += 1.0;
                node.wins += result;
                node.parent
            };

            if cur == tree.get_root_id() {
                break;
            }

            move_stack.unmake_last(board);
            cur = parent.expect("Parent not found");
        }

        // 4) Report every second
        let elapsed = search_data.timer.elapsed_ms();
        if elapsed >= last_report_ms + 1000 {
            last_report_ms = elapsed;

            if let Some(callback) = on_iteration {
                if let Some(best_id) = get_best_child(tree) {
                    let best = tree.get_node(best_id);
                    let score = if best.visits > 0.0 {
                        (best.wins / best.visits * 1000.0) as i32
                    } else {
                        0
                    };
                    let speed = if elapsed > 0 { iteration * 1000 / elapsed } else { 0 };

                    callback(SearchIterationResponse {
                        depth: 0,
                        mv: best.mv.unwrap_or_default(),
                        score,
                        nodes: iteration,
                        time: elapsed,
                        speed,
                    });
                }
            }
        }
    }

    get_best_child(tree).map(|id| tree.get_node(id).mv.unwrap())
}