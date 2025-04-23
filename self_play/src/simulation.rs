// One game of self-play using MCTS and a neural network
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};
use core::f32;
use std::vec;

use pyo3::prelude::*;

use crate::node::Node;
use blokus::{board::RepType, game::Game};


#[derive(FromPyObject)]
pub struct Config {
    dim: usize,
    rep: usize,
    sims_per_move: usize,
    sample_moves: usize,
    c_puct: f32,
    dirichlet_alpha: f64,
    exploration_fraction: f32,
}

pub struct Runtime<'py> {
    pub config:         Config,
    pub id:             i32,
    pub result_queue:   &'py Bound<'py, PyAny>,
    pub queue:          &'py Bound<'py, PyAny>,
    pub pipe:           &'py Bound<'py, PyAny>
}

/// Rotates the policy 90 degrees to the right
// fn rotate_policy(state: Vec<f32>) -> Vec<f32> {
//     let mut rotated = vec![0.0; BOARD_SIZE];
//     for i in 0..D {
//         for j in 0..D {
//             rotated[j * D + (D - 1 - i)] = state[i * D + j];
//         }
//     }

//     rotated.to_vec()
// }

/// Sample from a softmax distribution
/// Used to select actions during the first few moves to encourage exploration
fn softmax_sample(visit_dist: Vec<(usize, u32)>) -> usize {
    let total_visits: u32 = visit_dist.iter().fold(0, |acc, (_, visits)| acc + visits);
    let sample = rand::thread_rng().gen_range(0.0..1.0);
    let mut sum = 0.0;

    for (tile, visits) in &visit_dist {
        sum += (*visits as f32) / (total_visits as f32);
        if sum > sample {
            return *tile;
        }
    }
    visit_dist.last().expect("Dist should not be empty").0
}


fn softmax(values: Vec<f32>) -> Vec<f32> {
    let max = values
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let exp_vals: Vec<f32> = values.iter().map(|v| (*v - max).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    exp_vals.iter().map(|v| v / sum).collect()
}

fn masked_softmax(logits: &[f32], legal_moves: &[usize]) -> Vec<f32> {
    let legal_logits: Vec<f32> = legal_moves.iter().map(|&i| logits[i]).collect();

    // Subtract max for numerical stability
    let max_logit = legal_logits
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = legal_logits.iter().map(|&x| (x - max_logit).exp()).collect();

    let sum_exp: f32 = exp_vals.iter().sum();
    let softmax_vals: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

    // Rebuild full-sized vector, with zeros elsewhere
    let mut result = vec![0.0; logits.len()];
    for (&idx, &val) in legal_moves.iter().zip(softmax_vals.iter()) {
        result[idx] = val;
    }

    result
}

/// Update node when visitied during backpropagation
fn backpropagate(search_path: Vec<usize>, root: &mut Node, values: Vec<f32>) -> () {
    let mut node = root;
    for tile in search_path {
        node = node.children.get_mut(&tile).expect("Child should have this action");
        node.visits += 1;
        node.value_sum += values[node.to_play];
    }
}

fn random_action(game: &Game) -> Result<usize, String> {
    let actions = game.get_legal_tiles();
    let index = rand::thread_rng().gen_range(0..actions.len());
    return Ok(actions[index]);
}

impl<'py> Runtime<'py> {

    fn evaluate(&self, node: &mut Node, game: &Game) -> Result<Vec<f32>, Box<dyn std::error::Error>> {

        // If the game is over, return the payoff
        if game.is_terminal() {
            return Ok(game.get_payoff());
        }

        // Get the policy and value from the neural network
        let format = match self.config.rep {
            1 => RepType::Channel,
            2 => RepType::Token,
            _ => panic!("Invalid representation type")
        };

        let representation = game.get_game_state(format);
        let request = (self.id, representation);
        self.queue.call_method1("put", (request,))?;

        // Wait for the result
        let inference = self.pipe.call_method0("recv")?;
        let policy_logits: Vec<f32> = inference.get_item(0)?.extract()?;
        let value_logits: Vec<f32> = inference.get_item(1)?.extract()?;
        let current_player = game.current_player();

        // Rotate the policy so they are in order
        // for _ in 0..(current_player) {
        //     policy = rotate_policy(policy);
        // }

        // Normalize policy for node priors, filter out illegal moves
        let legal_moves = game.get_legal_tiles();
        let policy = masked_softmax(&policy_logits, &legal_moves);
        let mut value = softmax(value_logits);
        value.rotate_right(current_player);

        // Expand the node with the policy
        node.to_play = current_player;
        for (tile, prob) in policy.iter().enumerate() {
            if *prob > 0.0 {
                node.children.insert(tile, Node::new(*prob));
            }
        }
        Ok(value)
    }

    /// Get UCB score for a child node
    /// Exploration constant is based on the number of visits to the parent node
    /// so that it will encourage exploration of nodes that have not been visited
    fn ucb_score(&self, parent: &Node, child: &Node) -> f32 {

        let parent_visits = parent.visits as f32;
        let child_visits = child.visits as f32;
        let exploration_constant =  self.config.c_puct * parent_visits.sqrt() / (1.0 + child_visits);
        let exploration = exploration_constant * child.prior;
        let exploitation = child.value();

        let score = exploration + exploitation;
        assert!(score.is_finite(), "NaN/Inf in UCB score");
        score
    }

    /// Add noise to the root node to encourage exploration
    fn add_exploration_noise(&self, root: &mut Node) -> () {
        let num_actions = root.children.len();
        if num_actions <= 1 {
            return;
        }

        let alpha_vec = vec![self.config.dirichlet_alpha; num_actions];
        let dir = Dirichlet::<f64>::new(&alpha_vec).unwrap();
        let noise: Vec<f32> = dir.sample(&mut rand::thread_rng())
                                 .into_iter().map(|x| x as f32).collect();
        
        for (i, (_tile, node)) in root.children.iter_mut().enumerate() {
            node.prior = node.prior * (1.0 - self.config.exploration_fraction)
                + noise[i] * self.config.exploration_fraction;
        }
    }

    /// Select child node to explore
    /// Uses UCB formula to balance exploration and exploitation
    /// Returns the action and the child node's key
    fn select_child(&self, node: &Node) -> Result<usize, &'static str> {
        assert!(node.is_expanded());
        node.children
            .iter()
            .map(|(a,c)| (self.ucb_score(node,c), *a))
            .filter(|(s,_)| s.is_finite())           // discard NaNs defensively
            .max_by(|x,y| x.partial_cmp(y).unwrap())
            .map(|(_,a)| a)
            .ok_or("all UCB scores NaN")
    }

    /// Select action from policy
    fn select_action(&self, root: &Node, num_moves: usize) -> usize {
        let visit_dist: Vec<(usize, u32)> = root
            .children
            .iter()
            .map(|(tile, node)| (*tile, node.visits))
            .collect();

        assert!(visit_dist.len() > 0);
        if num_moves < self.config.sample_moves {
            softmax_sample(visit_dist)
        } else {
            visit_dist.iter().max_by(|a, b| a.1.cmp(&b.1)).expect("visit_dist should not be empty").0
        }
    }

    fn best_action(&self, game: &Game) -> Result<usize, String> {
        let mut root = Node::new(0.0);
        let mut highest_prior = 0.0;
        let mut best_action = 0;

        match self.evaluate(&mut root, game) {
            Ok(_) => (),
            Err(e) => {
                return Err(format!("Error evaluating root node: {:?}", e));
            }
        }

        // Get child with highest prior probability
        for (action, child) in &root.children {
            if child.prior > highest_prior {
                highest_prior = child.prior;
                best_action = *action;
            }
        }
        Ok(best_action)
    }

    /// Run MCTS simulations to get policy for root node
    fn mcts(&self,
        root: &mut Node,
        game: &Game,
        policies: &mut Vec<Vec<(i32, f32)>>,
    ) -> Result<usize, String> {

        // Initialize root for these sims, evaluate it, and add children
        if !root.is_expanded() {
            match self.evaluate(root, game) {
                Ok(_) => (),
                Err(e) => {
                    return Err(format!("Error evaluating root node: {:}", e));
                }
            }
        }

        // As far as I can tell, noise is added for each root node
        self.add_exploration_noise(root);

        for _ in 0..self.config.sims_per_move {
            // Select a leaf node
            let mut node = &mut *root;
            let mut scratch_game = game.clone();
            let mut search_path = Vec::new();
            while node.is_expanded() {
                let action = self.select_child(node)?;
                node = node.children.get_mut(&action).expect("Child should exist for this action");
                let _ = scratch_game.apply(action, None);
                search_path.push(action);
            }

            // Expand and evaluate the leaf node
            let values = self
                .evaluate(node, &scratch_game)
                .map_err(|e| e.to_string())?;

            // Backpropagate the value
            backpropagate(search_path, root, values)
        }

        // Save policy for this state
        let total_visits: u32 = root
            .children
            .iter()
            .map(|(_tile, child)| child.visits)
            .sum();
        let probs = root
            .children
            .iter()
            .map(|(tile, child)| {
                let p = (child.visits as f32) / (total_visits as f32);
                (*tile as i32, p)
            })
            .collect();
        policies.push(probs);

        // Pick action to take
        let action = self.select_action(&root, policies.len());
        Ok(action)
    }

    pub fn training_game(&self) -> u8 {
        // Storage for game data
        let mut game = Game::reset(self.config.dim);
        let mut policies: Vec<Vec<(i32, f32)>> = Vec::new();
        let mut root_node = Node::new(0.0);
        let mut root = &mut root_node;

        // Run self-play to generate data
        while !game.is_terminal() {
            // Get MCTS policy for current state
            let action = match self.mcts(root, &game, &mut policies) {
                Ok(a) => a,
                Err(e) => {
                    println!("Error during training game: {}", e);
                    return 1;
                }
            };

            // println!("Player {} --- {}", game.current_player(), action);
            let _ = game.apply(action, None);
            root = root.children.get_mut(&action).expect("Child should have corresponding action");
            // game.board.print_board();
        }

        // Send data to train the model
        // println!("History: {:?}", game.history);
        let values = game.get_payoff();
        let game_data = (game.history, policies, values.clone());
        let _ = self.result_queue.call_method1("put", ((self.id, game_data),));
        0
    }

    pub fn test_against_random(&self) -> u8 {
        let mut game = Game::reset(self.config.dim); 
        let mut action;
        while !game.is_terminal() {
            // Set queue to query for this action

            if game.current_player() == 0 {
                action = self.best_action(&game);
            } else {
                action = random_action(&game);
            }

            // Get action to take
            let tile = match action{
                Ok(a) => a,
                Err(e) => {
                    println!("Error running MCTS: {:?}", e);
                    return 1;
                }
            };

            // println!("Player {} --- {}", game.current_player(), action);
            let _ = game.apply(tile, None);
        }
        // println!("Finished Game");
        // game.board.print_board();
        let _ = self.result_queue.call_method1("put", ((self.id, game.get_payoff()[0]),));
        0
    }

    pub fn test_game(&mut self, model_queue: &'py Bound<PyAny>, baseline_queue: &'py Bound<PyAny>) -> Result<f32, String> {
        let mut game = Game::reset(20);
        // let mut policies: Vec<Vec<(i32, f32)>> = Vec::new();

        // Run self-play to generate data
        while !game.is_terminal() {
            // Set queue to query for this action
            if game.current_player() == 0 {
                self.queue = model_queue;
            } else {
                self.queue = baseline_queue;
            }

            // Get action to take
            let action = match self.best_action(&game) {
                Ok(a) => a,
                Err(e) => {
                    println!("Error running MCTS: {:?}", e);
                    return Err("Error running MCTS".to_string());
                }
            };

            // println!("Player {} --- {}", game.current_player(), action);
            let _ = game.apply(action, None);
        }
        println!("Finished Game");
        game.board.print_board();
        Ok(game.get_payoff()[0])
    }
}
