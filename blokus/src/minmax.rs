use std::i32::MAX;
use crate::game::Game;

const NUM_PLAYERS: usize = 4;

struct GameTree {
    state: Game,
    children: Vec<GameTree>,
    value: i32,
}

impl GameTree {

    fn add_children(&self) {

        assert!(self.children.is_empty());

        for tile in 


    }

}


// Calculate the heursitic for a non-terminal state
fn heuristic(state: Game) -> [i32; NUM_PLAYERS] {
    return state.get_score();
}


// Evaluates the game tree to select a move
fn evaluate(tree: GameTree, depth: usize) -> [i32; NUM_PLAYERS] {
    let values = [-MAX; 4];
    
    // Check base cases
    if tree.state.is_terminal() {
        for payoff in tree.state.get_payoff().iter() {
            values[i] = payoff * 1000; // Scale up to work with heuristic
        }
        return values
    } else if depth == 0 {
        return heuristic(tree.state);
    }

    // Recursive case
    if tree.children.is_empty() { 
        tree.add_children();
    }
    for node in tree.children {
        
    }
    // Evaluate all possible moves
    // Maximize "score" for the current player
    // Score will be score_player - max(other player scores)

    return values;
}
