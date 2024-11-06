use std::i32::MAX;

const NUM_PLAYERS: usize = 4;

struct GameTree {
    children: Vec<GameTree>,
    value: i32,
}

// Evaluates the game tree to select a move
fn evaluate(tree: GameTree, depth: usize) -> [i32; NUM_PLAYERS] {
    // Check base cases
    // If terminal state -> return [10000, 1000, 0, -1000]
    // If depth 0 -> return heuristic(state)

    // Recursive case
    let values = [-MAX; 4];
    for node in tree.children {
        continue;
    }
    // Evaluate all possible moves
    // Maximize "score" for the current player
    // Score will be score_player - max(other player scores)

    return values;
}
