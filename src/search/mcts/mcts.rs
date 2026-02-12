use std::hash::{BuildHasher, Hasher};
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::Gamma;
use crate::Board;
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::position_export::BitPosition;
use crate::search::mcts::utils::move_to_policy_index;
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::search_root::SearchIterationResponse;
use crate::terminal::check_terminal;
use crate::types::ZobristHash;
use crate::undo::UndoMove;

type NodeId = usize;

pub struct MCTSConfig {
    /// Dirichlet noise alpha (0.0 = no noise). Typical: 0.03 for large boards, 0.3 for small.
    pub dirichlet_alpha: f32,
    /// Fraction of noise mixed into root priors. Typical: 0.25.
    pub dirichlet_epsilon: f32,
    /// Temperature for final move selection. 0.0 = pick best, 1.0 = proportional to visits.
    pub temperature: f32,
}

impl MCTSConfig {
    pub fn default_play() -> Self {
        MCTSConfig {
            dirichlet_alpha: 0.0,
            dirichlet_epsilon: 0.0,
            temperature: 0.0,
        }
    }

    pub fn default_train() -> Self {
        MCTSConfig {
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            temperature: 1.0,
        }
    }
}

pub struct MCTSNode {
    mv: Option<Move>,
    children: Vec<NodeId>,
    parent: Option<NodeId>,
    expanded: bool,
    visits: f32,
    wins: f32,
    prior: f32,
    zobrist_hash: ZobristHash
    ,
}

impl MCTSNode {
    fn new_root(zobrist_hash: ZobristHash) -> MCTSNode {
        MCTSNode {
            mv: None,
            parent: None,
            children: vec![],
            expanded: false,
            visits: 0.0,
            wins: 0.0,
            prior: 0.0,
            zobrist_hash,
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

    fn new_child(mv: Move, parent: NodeId, prior: f32, zobrist_hash: ZobristHash) -> MCTSNode {
        MCTSNode {
            mv: Some(mv),
            parent: Some(parent),
            children: vec![],
            expanded: false,
            visits: 0.0,
            wins: 0.0,
            prior,
            zobrist_hash
        }
    }

    fn is_leaf(&self) -> bool {
        !self.expanded
    }

    fn append_child(&mut self, node: NodeId) {
        self.children.push(node);
    }
}

const ROOT_ID: NodeId = 0;

pub struct MCTSTree {
    nodes: Vec<MCTSNode>,
    pub move_gen: MoveGen,
}

impl MCTSTree {
    pub fn new() -> Self {
        MCTSTree { nodes: vec![], move_gen: MoveGen::new() }
    }

    pub fn get_node(&self, id: NodeId) -> &MCTSNode {
        &self.nodes[id]
    }

    fn get_node_mut(&mut self, id: NodeId) -> &mut MCTSNode {
        &mut self.nodes[id]
    }

    pub fn get_root(&self) -> &MCTSNode {
        &self.nodes[ROOT_ID]
    }

    fn get_root_mut(&mut self) -> &mut MCTSNode {
        &mut self.nodes[ROOT_ID]
    }

    fn get_root_id(&self) -> NodeId {
        ROOT_ID
    }

    fn new_child(&mut self, mv: Move, parent_id: NodeId, prior: f32, zobrist_hash: ZobristHash) -> NodeId {
        let index: NodeId = self.nodes.len();
        let new_child = MCTSNode::new_child(mv, parent_id, prior, zobrist_hash);
        self.nodes.push(new_child);
        let parent = self.get_node_mut(parent_id);
        parent.append_child(index);
        index
    }

    /// Reroot to the child of current root that matches `mv`.
    /// Returns true if found, false if tree was reset.
    pub fn reroot(&mut self, zobrist: ZobristHash) {
        let Some(old_root_id) = self.nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.zobrist_hash == zobrist)
            .max_by(|(_, a), (_, b)| a.visits.partial_cmp(&b.visits).unwrap())
            .map(|(i, _)| i)
        else {
            self.nodes.clear();
            self.nodes.push(MCTSNode::new_root(zobrist));
            return;
        };

        let mut stack = vec![old_root_id];
        let mut mapping = std::collections::HashMap::new();
        let mut new_nodes = Vec::new();

        while let Some(old_id) = stack.pop() {
            if mapping.contains_key(&old_id) {
                continue;
            }

            let new_id = new_nodes.len();
            mapping.insert(old_id, new_id);

            let old_node = &self.nodes[old_id];

            new_nodes.push(MCTSNode {
                mv: old_node.mv,
                parent: None,
                children: vec![],
                expanded: old_node.expanded,
                visits: old_node.visits,
                wins: old_node.wins,
                prior: old_node.prior,
                zobrist_hash: old_node.zobrist_hash,
            });

            for &child in &old_node.children {
                stack.push(child);
            }
        }

        for (old_id, &new_id) in &mapping {
            let old_node = &self.nodes[*old_id];
            let new_node = &mut new_nodes[new_id];

            if let Some(old_parent) = old_node.parent {
                if let Some(&mapped_parent) = mapping.get(&old_parent) {
                    new_node.parent = Some(mapped_parent);
                }
            }

            for &old_child in &old_node.children {
                if let Some(&mapped_child) = mapping.get(&old_child) {
                    new_node.children.push(mapped_child);
                }
            }
        }

        new_nodes[0].mv = None;
        new_nodes[0].parent = None;
        self.nodes = new_nodes;
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

        let mut undo = UndoMove::new();
        for (i, &mv) in moves.iter().enumerate() {
            board.make_move(mv, &mut undo).expect("Failed to make move");
            let zobrist = board.zobrist;
            board.unmake_move(&mut undo).expect("Failed to unmake move");
            tree.new_child(mv, node_id, priors[i], zobrist);
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

fn sample_dirichlet(alpha: f32, n: usize) -> Vec<f32> {
    let gamma = Gamma::new(alpha as f64, 1.0).unwrap();
    let mut rng = StdRng::seed_from_u64(std::hash::RandomState::new().build_hasher().finish());
    let samples: Vec<f64> = (0..n).map(|_| gamma.sample(&mut rng)).collect();
    let sum: f64 = samples.iter().sum();
    samples.iter().map(|&x| (x / sum) as f32).collect()
}

fn add_dirichlet_noise(tree: &mut MCTSTree, node_id: NodeId, alpha: f32, epsilon: f32) {
    let children: Vec<NodeId> = tree.get_node(node_id).children.clone();
    if children.is_empty() {
        return;
    }
    let noise = sample_dirichlet(alpha, children.len());
    for (i, &child_id) in children.iter().enumerate() {
        let child = tree.get_node_mut(child_id);
        child.prior = (1.0 - epsilon) * child.prior + epsilon * noise[i];
    }
}

fn get_best_child(tree: &MCTSTree, temperature: f32) -> Option<NodeId> {
    let root = tree.get_root();
    if root.children.is_empty() {
        return None;
    }

    if temperature <= 0.0 {
        // Greedy: pick most visited
        return root.children.iter()
            .max_by(|&&a, &&b| {
                let va = tree.get_node(a).visits;
                let vb = tree.get_node(b).visits;
                va.partial_cmp(&vb).unwrap()
            })
            .copied();
    }

    // Temperature-based sampling proportional to visits^(1/T)
    let inv_t = 1.0 / temperature;
    let weights: Vec<f64> = root.children.iter()
        .map(|&id| (tree.get_node(id).visits as f64).powf(inv_t as f64))
        .collect();
    let sum: f64 = weights.iter().sum();
    let probs: Vec<f64> = weights.iter().map(|&w| w / sum).collect();

    let mut rng = StdRng::seed_from_u64(std::hash::RandomState::new().build_hasher().finish());
    let r: f64 = rand::distr::Uniform::new(0.0f64, 1.0).unwrap().sample(&mut rng);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return Some(root.children[i]);
        }
    }
    Some(*root.children.last().unwrap())
}

pub fn mcts_search(
    board: &mut Board,
    tree: &mut MCTSTree,
    nn: &mut NeuralNet,
    search_data: &mut SearchData,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
    iter_max: Option<u64>,
    config: &MCTSConfig,
) -> Option<Move> {
    tree.reroot(board.zobrist);
    let mut mv_generator = MoveGen::new();
    let mut move_stack = MovesStack::new();
    let mut iteration: u64 = 0;
    let mut last_report_ms: u64 = 0;

    let root_id = tree.get_root_id();

    // Expand root
    if tree.get_root().is_leaf() {
        expand_node(board, tree, root_id, nn, &mut mv_generator);
    }

    // Add Dirichlet noise to root priors
    if config.dirichlet_alpha > 0.0 {
        add_dirichlet_noise(tree, root_id, config.dirichlet_alpha, config.dirichlet_epsilon);
    }

    loop {
        // Check time limit (skip if iter_max is set)
        if iter_max.is_none() && search_data.time_exceeded() {
            break;
        }

        iteration += 1;

        if let Some(max) = iter_max {
            if iteration > max {
                break;
            }
        }

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
                if let Some(best_id) = get_best_child(tree, 0.0) {
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

    // print
        //debug_print_top_moves(tree, 10);

    get_best_child(tree, config.temperature).map(|id| tree.get_node(id).mv.unwrap())
}