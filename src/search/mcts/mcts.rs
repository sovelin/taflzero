use crate::board::Board;
use crate::board::position_export::BitPosition;
use crate::board::types::ZobristHash;
use crate::movegen::MoveGen;
use crate::mv::Move;
use crate::search::mcts::utils::move_to_policy_index;
use crate::search::nn::NeuralNet;
use crate::search_data::SearchData;
use crate::search_root::SearchIterationResponse;
use crate::terminal::check_terminal;
use crate::undo::UndoMove;
use rand::distr::Distribution;
use rand::prelude::*;
use rand_distr::Gamma;
use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use sysinfo::System;

type NodeId = usize;

/// PUCT exploration constant, shared by selection and policy target pruning.
pub const C_PUCT: f32 = 1.4;

pub struct MCTSConfig {
    /// Dirichlet noise alpha (0.0 = no noise). Typical: 0.03 for large boards, 0.3 for small.
    pub dirichlet_alpha: f32,
    /// Fraction of noise mixed into root priors. Typical: 0.25.
    pub dirichlet_epsilon: f32,
    /// Temperature for final move selection. 0.0 = pick best, 1.0 = proportional to visits.
    pub temperature: f32,
    /// Number of leaves to collect before batched NN evaluation.
    pub batch_size: usize,
    /// Forced playouts at root (KataGo): every root child is guaranteed at least
    /// n_forced = sqrt(k * prior * total_visits) visits. 0.0 = disabled.
    pub forced_playouts_k: f32,
}

impl MCTSConfig {
    pub fn default_play() -> Self {
        MCTSConfig {
            dirichlet_alpha: 0.0,
            dirichlet_epsilon: 0.0,
            temperature: 0.0,
            batch_size: 8,
            forced_playouts_k: 0.0,
        }
    }

    pub fn default_train() -> Self {
        MCTSConfig {
            dirichlet_alpha: 0.1,
            dirichlet_epsilon: 0.25,
            temperature: 1.0,
            batch_size: 8,
            forced_playouts_k: 2.0,
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
    zobrist_hash: ZobristHash,
    virtual_loss: f32,
    /// Proven game-theoretic result from this node's side-to-move perspective:
    /// 0 = unknown, +1 = proven forced WIN, -1 = proven forced LOSS.
    proven: i8,
    /// Plies to the proven mate (shortest for a win, longest for a loss).
    proven_dist: u16,
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
            virtual_loss: 0.0,
            proven: 0,
            proven_dist: 0,
        }
    }

    pub fn children(&self) -> &Vec<NodeId> {
        &self.children
    }

    pub fn visits(&self) -> f32 {
        self.visits
    }

    pub fn wins(&self) -> f32 {
        self.wins
    }

    pub fn mv(&self) -> Option<Move> {
        self.mv
    }

    pub fn prior(&self) -> f32 {
        self.prior
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
            zobrist_hash,
            virtual_loss: 0.0,
            proven: 0,
            proven_dist: 0,
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
    pub memory_limit: Option<f64>,
    pub move_gen: MoveGen,
}

impl Default for MCTSTree {
    fn default() -> Self {
        Self::new()
    }
}

impl MCTSTree {
    pub fn new() -> Self {
        MCTSTree {
            nodes: vec![],
            memory_limit: None,
            move_gen: MoveGen::new(),
        }
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

    pub fn get_pv_from(&self, node_id: NodeId) -> Vec<Move> {
        let mut cur = node_id;

        let first_node = self.get_node(cur);

        let mut pv = vec![first_node.mv.unwrap()];

        loop {
            let node = self.get_node(cur);
            if node.children.is_empty() {
                break;
            }
            let best = node.children.iter().max_by(|&&a, &&b| {
                self.get_node(a)
                    .visits
                    .partial_cmp(&self.get_node(b).visits)
                    .unwrap()
            });
            match best {
                Some(&child_id) => {
                    let child = self.get_node(child_id);
                    if child.visits == 0.0 {
                        break;
                    }
                    if let Some(mv) = child.mv {
                        pv.push(mv);
                    }
                    cur = child_id;
                }
                None => break,
            }
        }

        pv
    }

    fn get_root_mut(&mut self) -> &mut MCTSNode {
        &mut self.nodes[ROOT_ID]
    }

    fn get_root_id(&self) -> NodeId {
        ROOT_ID
    }

    fn new_child(
        &mut self,
        mv: Move,
        parent_id: NodeId,
        prior: f32,
        zobrist_hash: ZobristHash,
    ) -> NodeId {
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
        let Some(old_root_id) = self
            .nodes
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
                virtual_loss: 0.0,
                proven: old_node.proven,
                proven_dist: old_node.proven_dist,
            });

            for &child in &old_node.children {
                stack.push(child);
            }
        }

        for (old_id, &new_id) in &mapping {
            let old_node = &self.nodes[*old_id];
            let new_node = &mut new_nodes[new_id];

            if let Some(old_parent) = old_node.parent
                && let Some(&mapped_parent) = mapping.get(&old_parent)
            {
                new_node.parent = Some(mapped_parent);
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

/// PUCT child selection. `forced_k` > 0 (root only, self-play) enables forced
/// playouts: a child whose visits are below sqrt(k * prior * parent_visits) is
/// selected unconditionally (largest deficit first), so root Dirichlet noise
/// reliably translates into exploration.
fn puct_select(tree: &MCTSTree, from_id: NodeId, forced_k: f32) -> NodeId {
    let from = tree.get_node(from_id);
    let mut best_score = f32::NEG_INFINITY;
    let mut best_child: Option<NodeId> = None;

    let parent_effective = from.visits + from.virtual_loss;
    let sqrt_parent = parent_effective.sqrt();
    let c = C_PUCT;

    if forced_k > 0.0 && from.visits > 0.0 {
        let mut best_deficit = 0.0f32;
        let mut forced_child: Option<NodeId> = None;
        for id in from.children.iter() {
            let child = tree.get_node(*id);
            let effective_visits = child.visits + child.virtual_loss;
            if effective_visits <= 0.0 {
                continue; // unvisited children are reached via normal PUCT
            }
            let n_forced = (forced_k * child.prior * from.visits).sqrt();
            let deficit = n_forced - effective_visits;
            if deficit > best_deficit {
                best_deficit = deficit;
                forced_child = Some(*id);
            }
        }
        if let Some(id) = forced_child {
            return id;
        }
    }

    // FPU reduction: unvisited children are assumed worse than parent average
    const FPU_REDUCTION: f32 = 0.0;
    let parent_q = if from.visits > 0.0 {
        -(from.wins / from.visits)
    } else {
        0.0
    };
    let fpu_value = parent_q - FPU_REDUCTION;

    for id in from.children.iter() {
        let child = tree.get_node(*id);

        let effective_visits = child.visits + child.virtual_loss;
        let q = if effective_visits > 0.0 {
            (child.wins - child.virtual_loss) / effective_visits
        } else {
            fpu_value
        };
        let puct_value = q + c * child.prior * sqrt_parent / (1.0 + effective_visits);

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
        board
            .unmake_move(&mut last_mv)
            .expect("Failed to make undo");
    }

    fn unmake_all(&mut self, board: &mut Board) {
        while !self.undo.is_empty() {
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
    let rep = board.rep_table.get(&board.zobrist).copied().unwrap_or(1);
    let position = BitPosition::from_board(board, rep);
    let nn_out = nn.evaluate_position(&position);

    let moves = get_left_moves(board, move_gen);

    if !moves.is_empty() {
        let logits: Vec<f32> = moves
            .iter()
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
        let score = if visits > 0.0 {
            node.wins / visits
        } else {
            0.0
        };

        println!(
            "#{:<2} visits={:<8.0} score={:.3} prior={:.3} move={:?}",
            i + 1,
            visits,
            score,
            node.prior,
            node.mv
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

    // --- proven-result handling (overrides the visit-based choice) ---
    // A child that is a proven LOSS for its side-to-move (the opponent) is a
    // forced WIN for us. Play the *nearest* such mate. If every move is a proven
    // loss for us, delay it as long as possible.
    let mut best_win: Option<(NodeId, u16, f32)> = None; // (id, dist, visits)
    let mut best_loss: Option<(NodeId, u16)> = None; // (id, dist)
    let mut has_non_losing = false;
    for &c in &root.children {
        let child = tree.get_node(c);
        match child.proven {
            -1 => {
                let better = best_win.is_none_or(|(_, d, v)| {
                    child.proven_dist < d || (child.proven_dist == d && child.visits > v)
                });
                if better {
                    best_win = Some((c, child.proven_dist, child.visits));
                }
                has_non_losing = true;
            }
            1 => {
                if best_loss.is_none_or(|(_, d)| child.proven_dist > d) {
                    best_loss = Some((c, child.proven_dist));
                }
            }
            _ => has_non_losing = true,
        }
    }
    if let Some((id, _, _)) = best_win {
        return Some(id); // nearest forced win
    }
    if !has_non_losing {
        return best_loss.map(|(id, _)| id); // everything loses -> longest resistance
    }

    // --- normal visit-based selection over non-losing moves ---
    let pool: Vec<NodeId> = root
        .children
        .iter()
        .copied()
        .filter(|&c| tree.get_node(c).proven != 1) // avoid proven-losing moves
        .collect();

    if temperature <= 0.0 {
        // Greedy: pick most visited
        return pool
            .iter()
            .max_by(|&&a, &&b| {
                tree.get_node(a)
                    .visits
                    .partial_cmp(&tree.get_node(b).visits)
                    .unwrap()
            })
            .copied();
    }

    // Temperature-based sampling proportional to visits^(1/T)
    let inv_t = 1.0 / temperature;
    let weights: Vec<f64> = pool
        .iter()
        .map(|&id| (tree.get_node(id).visits as f64).powf(inv_t as f64))
        .collect();
    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return pool.first().copied();
    }
    let probs: Vec<f64> = weights.iter().map(|&w| w / sum).collect();

    let mut rng = StdRng::seed_from_u64(std::hash::RandomState::new().build_hasher().finish());
    let r: f64 = rand::distr::Uniform::new(0.0f64, 1.0)
        .unwrap()
        .sample(&mut rng);
    let mut cumulative = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumulative += p;
        if r < cumulative {
            return Some(pool[i]);
        }
    }
    pool.last().copied()
}

/**
 * Get the child corresponding to the multi-PV rank (1 = best, 2 = second best, etc).
 * Returns None if there are fewer children than multi_pv.
 */
fn get_multi_pv_child(tree: &MCTSTree, node_id: NodeId, multi_pv: usize) -> Option<NodeId> {
    let children: Vec<NodeId> = tree.get_node(node_id).children.clone();
    if children.is_empty() {
        return None;
    }

    let mut sorted_children = children.clone();
    sorted_children.sort_by(|&a, &b| {
        let va = tree.get_node(a).visits;
        let vb = tree.get_node(b).visits;
        vb.partial_cmp(&va).unwrap()
    });

    if multi_pv - 1 < sorted_children.len() {
        Some(sorted_children[multi_pv - 1])
    } else {
        None
    }
}

/// Data collected for a single leaf during batched selection.
struct PendingLeaf {
    node_id: NodeId,
    path: Vec<NodeId>,
    terminal_value: Option<f32>,
    position: Option<BitPosition>,
    legal_moves: Vec<Move>,
    child_zobrists: Vec<ZobristHash>,
}

/// Apply virtual loss along the path (makes nodes look worse to encourage diversity).
fn apply_virtual_loss(tree: &mut MCTSTree, path: &[NodeId]) {
    for &node_id in path {
        let node = tree.get_node_mut(node_id);
        node.virtual_loss += 1.0;
    }
}

/// Remove virtual loss along the path.
fn remove_virtual_loss(tree: &mut MCTSTree, path: &[NodeId]) {
    for &node_id in path {
        let node = tree.get_node_mut(node_id);
        node.virtual_loss -= 1.0;
    }
}

/// Select a single leaf, collecting the path and board state info.
/// Returns None if no selection is possible (e.g. root has no children).
fn select_leaf(
    board: &mut Board,
    tree: &MCTSTree,
    move_stack: &mut MovesStack,
    move_gen: &mut MoveGen,
    root_forced_k: f32,
) -> Option<PendingLeaf> {
    let mut cur = tree.get_root_id();
    let mut path = vec![cur];

    // Descend via PUCT until we hit a leaf or a newly-terminal node
    while !tree.get_node(cur).is_leaf() && !tree.get_node(cur).children.is_empty() {
        // Forced playouts apply at the root only
        let forced_k = if cur == tree.get_root_id() {
            root_forced_k
        } else {
            0.0
        };
        cur = puct_select(tree, cur, forced_k);
        path.push(cur);
        let node = tree.get_node(cur);
        move_stack.make_move(board, node.mv.expect("Move not found"));
    }

    // Check if this is a terminal position
    let is_terminal = check_terminal(board);

    let pending = if let Some(winner) = is_terminal {
        let value = if board.side_to_move == winner {
            1.0
        } else {
            -1.0
        };
        PendingLeaf {
            node_id: cur,
            path,
            terminal_value: Some(value),
            position: None,
            legal_moves: vec![],
            child_zobrists: vec![],
        }
    } else if tree.get_node(cur).children.is_empty() && tree.get_node(cur).expanded {
        // Expanded but no legal moves — loss
        PendingLeaf {
            node_id: cur,
            path,
            terminal_value: Some(-1.0),
            position: None,
            legal_moves: vec![],
            child_zobrists: vec![],
        }
    } else {
        // Need NN evaluation — collect position and legal moves
        let rep = board.rep_table.get(&board.zobrist).copied().unwrap_or(1);
        let position = BitPosition::from_board(board, rep);
        let moves = get_left_moves(board, move_gen);

        let mut child_zobrists = Vec::with_capacity(moves.len());
        let mut undo = UndoMove::new();
        for &mv in &moves {
            board.make_move(mv, &mut undo).expect("Failed to make move");
            child_zobrists.push(board.zobrist);
            board.unmake_move(&mut undo).expect("Failed to unmake move");
        }

        PendingLeaf {
            node_id: cur,
            path,
            terminal_value: None,
            position: Some(position),
            legal_moves: moves,
            child_zobrists,
        }
    };

    // Unmake all moves back to root
    move_stack.unmake_all(board);

    Some(pending)
}

/// Expand a node using pre-computed NN output, legal moves, and child zobrists.
fn expand_with_nn_output(
    tree: &mut MCTSTree,
    node_id: NodeId,
    policy: &[f32; 4840],
    legal_moves: &[Move],
    child_zobrists: &[ZobristHash],
) {
    if !legal_moves.is_empty() {
        let logits: Vec<f32> = legal_moves
            .iter()
            .map(|mv| policy[move_to_policy_index(*mv) as usize])
            .collect();
        let priors = softmax(&logits);

        for (i, &mv) in legal_moves.iter().enumerate() {
            tree.new_child(mv, node_id, priors[i], child_zobrists[i]);
        }
    }
    tree.get_node_mut(node_id).expanded = true;
}

/// Backpropagate a result from a leaf node up to the root.
fn backpropagate(tree: &mut MCTSTree, path: &[NodeId], mut result: f32) {
    // Path goes from root to leaf. We iterate from leaf to root.
    for &node_id in path.iter().rev() {
        result = -result;
        let node = tree.get_node_mut(node_id);
        node.visits += 1.0;
        node.wins += result;
    }
}

/// Propagate proven win/loss up the search path (MCTS-Solver style).
/// A node is a proven WIN if any child is a proven LOSS (for that child's
/// side-to-move); a proven LOSS if it is fully expanded and *every* child is a
/// proven WIN. Distance is the shortest path to a forced win (so the root plays
/// the nearest mate) or the longest resistance before a forced loss.
fn propagate_proven(tree: &mut MCTSTree, path: &[NodeId]) {
    for &node_id in path.iter().rev() {
        update_proven(tree, node_id);
    }
}

fn update_proven(tree: &mut MCTSTree, node_id: NodeId) {
    let node = tree.get_node(node_id);
    if node.children.is_empty() {
        return; // leaf/terminal: `proven` is set at expansion time
    }
    let expanded = node.expanded;
    let children = node.children.clone();

    let mut win_dist: Option<u16> = None; // shortest dist among LOSS children (we win)
    let mut loss_dist: u16 = 0; // longest dist among WIN children (we lose)
    let mut all_win = expanded; // is every child a proven WIN-for-opponent?

    for c in children {
        let child = tree.get_node(c);
        match child.proven {
            -1 => {
                // opponent is lost in this line -> a winning move for us
                win_dist = Some(win_dist.map_or(child.proven_dist, |w| w.min(child.proven_dist)));
                all_win = false;
            }
            1 => loss_dist = loss_dist.max(child.proven_dist),
            _ => all_win = false,
        }
    }

    let node = tree.get_node_mut(node_id);
    if let Some(d) = win_dist {
        node.proven = 1;
        node.proven_dist = d.saturating_add(1);
    } else if all_win {
        node.proven = -1;
        node.proven_dist = loss_dist.saturating_add(1);
    } else {
        node.proven = 0;
    }
}

pub fn mcts_search(
    board: &mut Board,
    tree: &mut MCTSTree,
    nn: &mut NeuralNet,
    search_data: &mut SearchData,
    on_iteration: Option<&dyn Fn(SearchIterationResponse)>,
    iter_max: Option<u64>,
    config: &MCTSConfig,
    multi_pv: Option<usize>,
) -> Option<Move> {
    tree.reroot(board.zobrist);
    let mut check_memory = false;
    let mut check_memory_count = 0;
    let mut mv_generator = MoveGen::new();
    let mut move_stack = MovesStack::new();
    let mut iteration: u64 = 0;
    let mut last_report_ms: u64 = 0;
    let mut sys = System::new_all();

    let root_id = tree.get_root_id();
    let batch_size = config.batch_size.max(1);

    // Expand root (single eval, not batched). Backpropagate the root's own
    // value so the first PUCT selection has sqrt(parent) > 0 and a correct
    // FPU baseline — otherwise sim #1 always lands on children[0].
    if tree.get_root().is_leaf() {
        let root_value = expand_node(board, tree, root_id, nn, &mut mv_generator);
        backpropagate(tree, &[root_id], root_value);
    }

    // Add Dirichlet noise to root priors
    if config.dirichlet_alpha > 0.0 {
        add_dirichlet_noise(
            tree,
            root_id,
            config.dirichlet_alpha,
            config.dirichlet_epsilon,
        );
    }

    loop {
        // Check memory limit
        if check_memory && let Some(memory_limit) = tree.memory_limit {
            check_memory = false;
            sys.refresh_memory();
            let used_memory = sys.used_memory() as f64 / sys.total_memory() as f64;

            if used_memory > memory_limit {
                eprintln!("info string high memory usage: {used_memory}");
                break;
            }
        }

        // Check time limit
        if iter_max.is_none() && search_data.time_exceeded() {
            break;
        }

        // Check external stop signal
        if search_data.is_stopped() {
            break;
        }

        // Check iteration limit
        if let Some(max) = iter_max
            && iteration >= max
        {
            break;
        }

        // Stop early once the root is a proven win/loss — more search cannot
        // change the outcome. Play mode only (temperature <= 0); self-play keeps
        // searching so the policy targets (visit distribution) are unaffected.
        if config.temperature <= 0.0 && tree.get_root().proven != 0 {
            break;
        }

        // --- Collect batch of leaves ---
        let mut pending_leaves: Vec<PendingLeaf> = Vec::with_capacity(batch_size);
        let remaining = if let Some(max) = iter_max {
            (max - iteration) as usize
        } else {
            batch_size
        };
        let this_batch = batch_size.min(remaining);
        let mut selected_nodes: HashSet<NodeId> = HashSet::with_capacity(this_batch);

        for _ in 0..this_batch {
            if let Some(leaf) = select_leaf(
                board,
                tree,
                &mut move_stack,
                &mut mv_generator,
                config.forced_playouts_k,
            ) {
                if !selected_nodes.insert(leaf.node_id) {
                    // Strict dedup: do not add the same leaf twice in one micro-batch.
                    break;
                }
                apply_virtual_loss(tree, &leaf.path);
                pending_leaves.push(leaf);
            } else {
                break;
            }
        }
        if pending_leaves.is_empty() {
            break;
        }

        // --- Batch NN evaluation for non-terminal leaves ---
        let nn_indices: Vec<usize> = pending_leaves
            .iter()
            .enumerate()
            .filter(|(_, l)| l.terminal_value.is_none() && l.position.is_some())
            .map(|(i, _)| i)
            .collect();

        let nn_results = if !nn_indices.is_empty() {
            let positions: Vec<&BitPosition> = nn_indices
                .iter()
                .map(|&i| pending_leaves[i].position.as_ref().unwrap())
                .collect();
            nn.evaluate_batch(&positions)
        } else {
            vec![]
        };
        // --- Expand and backpropagate ---
        let mut nn_result_idx = 0;

        for leaf in &pending_leaves {
            // Remove virtual loss
            remove_virtual_loss(tree, &leaf.path);

            let result = if let Some(terminal_val) = leaf.terminal_value {
                // Terminal — mark expanded and record the proven result (mate in 0).
                let node = tree.get_node_mut(leaf.node_id);
                node.expanded = true;
                node.proven = if terminal_val > 0.0 { 1 } else { -1 };
                node.proven_dist = 0;
                terminal_val
            } else {
                // Use NN output to expand
                let nn_out = &nn_results[nn_result_idx];
                nn_result_idx += 1;

                // Guard against duplicate expansion if two leaves hit the same node
                if !tree.get_node(leaf.node_id).expanded {
                    expand_with_nn_output(
                        tree,
                        leaf.node_id,
                        &nn_out.policy,
                        &leaf.legal_moves,
                        &leaf.child_zobrists,
                    );
                }

                nn_out.value
            };

            backpropagate(tree, &leaf.path, result);
            propagate_proven(tree, &leaf.path);
        }
        let pending_leaves_count = pending_leaves.len();

        check_memory_count += pending_leaves_count;
        if check_memory_count >= 1024 {
            check_memory = true;
            check_memory_count = 0;
        }

        iteration += pending_leaves_count as u64;

        // Report every second
        let elapsed = search_data.timer.elapsed_ms();
        if elapsed >= last_report_ms + 100 {
            last_report_ms = elapsed;

            if let Some(callback) = on_iteration {
                report_iteration(tree, callback, elapsed, iteration, multi_pv);
            }
        }
    }

    // Final report so the last PV/score (including a proven `mate N`) is emitted
    // even when the search stopped early on a proven result or a time cutoff.
    if let Some(callback) = on_iteration {
        let elapsed = search_data.timer.elapsed_ms();
        report_iteration(tree, callback, elapsed, iteration, multi_pv);
    }

    get_best_child(tree, config.temperature).map(|id| tree.get_node(id).mv.unwrap())
}

fn response_from_move(
    node_id: NodeId,
    tree: &MCTSTree,
    elapsed: u64,
    callback: &dyn Fn(SearchIterationResponse),
    iteration: u64,
    multi_pv: Option<usize>,
) {
    let node = tree.get_node(node_id);

    let (score, winrate) = if node.visits > 0.0 {
        let v = (node.wins / node.visits).clamp(-0.9999, 0.9999);
        let winrate = (v + 1.0) / 2.0;
        ((111.714_64 * (1.562_068_8 * v).tan()) as i32, winrate)
    } else {
        (0, 0.5)
    };
    let speed = (
        iteration * 1_000).checked_div( elapsed
    ).unwrap_or_default();

    // Plies to mate from the current position (node is 1 ply from root, so its
    // dist + 1 = the root's distance to mate).
    let mate = match node.proven {
        -1 => Some(node.proven_dist as i32 + 1), // we force the mate
        1 => Some(-(node.proven_dist as i32 + 1)), // we get mated
        _ => None,
    };
    // A proven result is a certainty, so report 100%/0% instead of the value
    // net's (saturated but not exact) estimate.
    let winrate = match node.proven {
        -1 => 1.0,
        1 => 0.0,
        _ => winrate,
    };

    callback(SearchIterationResponse {
        score,
        nodes: iteration,
        time: elapsed,
        speed,
        pv: tree.get_pv_from(node_id),
        winrate,
        multi_pv,
        mate,
    });
}

/// Emit one report: all ranks in MultiPV mode, otherwise just the best move.
fn report_iteration(
    tree: &MCTSTree,
    callback: &dyn Fn(SearchIterationResponse),
    elapsed: u64,
    iteration: u64,
    multi_pv: Option<usize>,
) {
    if let Some(mpv) = multi_pv {
        for rank in 1..=mpv {
            if let Some(id) = get_multi_pv_child(tree, tree.get_root_id(), rank) {
                response_from_move(id, tree, elapsed, callback, iteration, Some(rank));
            }
        }
    } else if let Some(best_id) = get_best_child(tree, 0.0) {
        response_from_move(best_id, tree, elapsed, callback, iteration, None);
    }
}
