// One game of self-play using MCTS and a neural network
use rand::Rng;
use rand_distr::{Dirichlet, Distribution};
use std::vec;

use pyo3::prelude::*;

use crate::node::Node;
use blokus::game::Game;


#[derive(FromPyObject)]
pub struct Config {
    dim: usize,
    sims_per_move: usize,
    sample_moves: usize,
    c_base: f32,
    c_init: f32,
    dirichlet_alpha: f32,
    exploration_fraction: f32,
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

/// Evaluate and Expand the Node
fn evaluate(
    node: &mut Node,
    game: &Game,
    inference_queue: &Bound<PyAny>,
    pipe: &Bound<PyAny>,
    id: i32,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // If the game is over, return the payoff
    if game.is_terminal() {
        return Ok(game.get_payoff());
    }

    // Get the policy and value from the neural network
    let representation = game.get_game_state();
    let request = (id, representation);
    inference_queue.call_method1("put", (request,))?;

    // Wait for the result
    let inference = pipe.call_method0("recv")?;
    let policy: Vec<f32> = inference.get_item(0)?.extract()?;
    let mut value: Vec<f32> = inference.get_item(1)?.extract()?;
    let current_player = game.current_player();

    // Rotate the policy so they are in order
    // for _ in 0..(current_player) {
    //     policy = rotate_policy(policy);
    // }
    value.rotate_right(current_player);

    // Normalize policy for node priors, filter out illegal moves
    let legal_moves = game.get_legal_tiles();
    let mut exp_policy = vec![];
    for tile in legal_moves {
        if policy[tile] > 0.0 {
            exp_policy.push((tile, policy[tile].exp()));
        }
    }
    let total: f32 = exp_policy.iter().map(|(_, p)| p).sum();
    // println!("Policy: {:?}\nExp: {:?}", policy, exp_policy);

    // Expand the node with the policy
    node.to_play = current_player;
    for (tile, prob) in exp_policy {
        node.children.insert(tile, Node::new(prob / total));
    }
    Ok(value)
}

/// Get UCB score for a child node
/// Exploration constant is based on the number of visits to the parent node
/// so that it will encourage exploration of nodes that have not been visited
fn ucb_score(parent: &Node, child: &Node, config: &Config) -> f32 {
    let c_base = config.c_base;
    let c_init = config.c_init;
    let parent_visits = parent.visits as f32;
    let exploration_constant = (((parent_visits + c_base + 1.0) / c_base).ln() + c_init)
        * parent_visits.sqrt()
        / (1.0 + child.visits as f32);
    let prior_score = exploration_constant * child.prior;
    let value_score = child.value();
    prior_score + value_score
}

/// Add noise to the root node to encourage exploration
fn add_exploration_noise(root: &mut Node, config: &Config) -> () {
    let num_actions = root.children.len();
    if num_actions <= 1 {
        return;
    }

    let alpha_vec = vec![config.dirichlet_alpha; num_actions];
    let dirichlet = Dirichlet::new(&alpha_vec).unwrap();
    let noise = dirichlet.sample(&mut rand::thread_rng());
    for (i, (_tile, node)) in root.children.iter_mut().enumerate() {
        node.prior = node.prior * (1.0 - config.exploration_fraction)
            + noise[i] * config.exploration_fraction;
    }
}

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
    visit_dist.last().unwrap().0
}

/// Select child node to explore
/// Uses UCB formula to balance exploration and exploitation
/// Returns the action and the child node's key
fn select_child(node: &Node, config: &Config) -> usize {
    let mut best_score = 0.0;
    let mut best_action = 0;
    for (action, child) in &node.children {
        let score = ucb_score(node, child, config);
        // println!("Action: {}, Score: {}", action, score);
        if score >= best_score {
            best_score = score;
            best_action = *action;
        }
    }
    best_action
}

/// Select action from policy
fn select_action(root: &Node, num_moves: usize, config: &Config) -> usize {
    let visit_dist: Vec<(usize, u32)> = root
        .children
        .iter()
        .map(|(tile, node)| (*tile, node.visits))
        .collect();
    if num_moves < config.sample_moves {
        softmax_sample(visit_dist)
    } else {
        visit_dist.iter().max_by(|a, b| a.1.cmp(&b.1)).unwrap().0
    }
}

/// Update node when visitied during backpropagation
fn backpropagate(search_path: Vec<usize>, root: &mut Node, values: Vec<f32>) -> () {
    let mut node = root;
    for tile in search_path {
        node = node.children.get_mut(&tile).unwrap();
        node.visits += 1;
        node.value_sum += values[node.to_play];
    }
}

/// Run MCTS simulations to get policy for root node
fn mcts(
    game: &Game,
    policies: &mut Vec<Vec<(i32, f32)>>,
    config: &Config,
    inference_queue: &Bound<PyAny>,
    pipe: &Bound<PyAny>,
    id: i32,
) -> Result<usize, String> {
    // Initialize root for these sims, evaluate it, and add children
    let mut root = Node::new(0.0);
    match evaluate(&mut root, game, inference_queue, pipe, id) {
        Ok(_) => (),
        Err(e) => {
            return Err(format!("Error evaluating root node: {:}", e));
        }
    }
    add_exploration_noise(&mut root, config);

    for _ in 0..config.sims_per_move {
        // Select a leaf node
        root.visits += 1;
        let mut node = &mut root;
        let mut scratch_game = game.clone();
        let mut search_path = Vec::new();
        while node.is_expanded() {
            let action = select_child(node, config);
            node = node.children.get_mut(&action).unwrap();
            let _ = scratch_game.apply(action, None);
            search_path.push(action);
        }

        // Expand and evaluate the leaf node
        let values = evaluate(node, &scratch_game, inference_queue, pipe, id).unwrap();

        // Backpropagate the value
        backpropagate(search_path, &mut root, values)
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
    let action = select_action(&root, policies.len(), config);
    Ok(action)
}

fn best_action(
    game: &Game,
    id: i32,
    queue: &Bound<PyAny>,
    pipe: &Bound<PyAny>,
) -> Result<usize, String> {
    let mut root = Node::new(0.0);
    match evaluate(&mut root, game, queue, pipe, id) {
        Ok(_) => (),
        Err(e) => {
            return Err(format!("Error evaluating root node: {:?}", e));
        }
    }

    // Random move for baseline
    if game.current_player() != 0 {
        let num_actions = root.children.len();
        let index = rand::thread_rng().gen_range(0..num_actions);
        let action = root.children.keys().nth(index).unwrap();
        return Ok(*action);
    }

    // Get child with highest prior probability
    let mut highest_prior = 0.0;
    let mut best_action = 0;
    for (action, child) in &root.children {
        if child.prior > highest_prior {
            highest_prior = child.prior;
            best_action = *action;
        }
    }
    Ok(best_action)
}

fn random_action(game: &Game) -> Result<usize, String> {
    Ok(0)
}

pub fn training_game(
    config: &Config,
    inference_queue: &Bound<PyAny>,
    pipe: &Bound<PyAny>,
    id: i32,
) -> Result<(Vec<(i32, i32)>, Vec<Vec<(i32, f32)>>, Vec<f32>), String> {
    // Storage for game data
    let mut game = Game::reset(config.dim);
    let mut policies: Vec<Vec<(i32, f32)>> = Vec::new();

    // Run self-play to generate data
    while !game.is_terminal() {
        // Get MCTS policy for current state
        let action = match mcts(&game, &mut policies, &config, inference_queue, pipe, id) {
            Ok(a) => a,
            Err(e) => {
                return Err(format!("Error running MCTS: {}", e));
            }
        };

        // println!("Player {} --- {}", game.current_player(), action);
        let _ = game.apply(action, None);
    }

    // Send data to train the model
    // println!("History: {:?}", game.history);
    let values = game.get_payoff();
    let game_data = (game.history, policies, values.clone());
    Ok(game_data)
}

pub fn test_against_random(
    config: &Config,
    id: i32,
    inference_queue: &Bound<PyAny>,
    pipe: &Bound<PyAny>,
) -> Result<f32, String> {
    let mut game = Game::reset(config.dim); 
    let mut action;
    while !game.is_terminal() {
        // Set queue to query for this action

        if game.current_player() == 0 {
            action = best_action(&game, id, inference_queue, pipe);
        } else {
            action = random_action(&game);
        }

        // Get action to take
        let tile = match action{
            Ok(a) => a,
            Err(e) => {
                println!("Error running MCTS: {:?}", e);
                return Err("Error running MCTS".to_string());
            }
        };

        // println!("Player {} --- {}", game.current_player(), action);
        let _ = game.apply(tile, None);
    }
    println!("Finished Game");
    game.board.print_board();
    Ok(game.get_payoff()[0])
}

pub fn test_game(
    id: i32,
    model_queue: &Bound<PyAny>,
    baseline_queue: &Bound<PyAny>,
    pipe: &Bound<PyAny>,
) -> Result<f32, String> {
    let mut game = Game::reset(20);
    // let mut policies: Vec<Vec<(i32, f32)>> = Vec::new();

    // Run self-play to generate data
    let mut queue;
    while !game.is_terminal() {
        // Set queue to query for this action
        if game.current_player() == 0 {
            queue = model_queue;
        } else {
            queue = baseline_queue;
        }

        // Get action to take
        let action = match best_action(&game, id, queue, pipe) {
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
