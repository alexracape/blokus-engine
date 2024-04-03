use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::rc::Rc;

use gloo_console as console;
use yew::prelude::*;

use crate::grpc::StateRepresentation;
use crate::board::Board;
use crate::pieces::Piece;
use crate::player::Player;

const BOARD_SPACES: usize = 400;

pub enum Action {
    PlacePiece(usize, usize, usize),
    Pass,
    Undo,
    ResetGame,
}


/// Get the legal moves for a piece
fn get_piece_moves(piece: &Piece, board: &Board, player: &Player) -> Vec<Vec<usize>> {
    let mut moves = Vec::new();
    for anchor in &player.get_anchors() {
        for variant in &piece.variants {
            for offset in &variant.offsets {

                // Check underflow
                if offset > anchor {
                    continue;
                }

                let total_offset = anchor - offset; // offset to anchor, then offset to line up piece
                if board.is_valid_move(player, variant, total_offset) {
                    let mut tiles = Vec::new();
                    for (i, square) in variant.variant.iter().enumerate() {
                        if *square {
                            tiles.push(total_offset + i);
                        }
                    }
                    moves.push(tiles);
                }
            }
        }
    }

    moves
}


/// Get the legal moves for a player, tile placements grouped by move
fn get_moves(board: &Board, player: &Player) -> Vec<Vec<usize>> {
    let mut moves = Vec::new();
    for piece in &player.pieces {
        let piece_moves = get_piece_moves(piece, board, player);
        moves.extend(piece_moves);
    }

    moves
}


/// Get the tile bases representation for legal moves
fn get_tile_moves(board: &Board, player: &Player) -> HashMap<usize, HashSet<usize>> {
    let mut tile_rep = HashMap::new();
    let mut moves = get_moves(board, player);
    
    for (i, tiles) in moves.iter().enumerate() {
        for tile in tiles {
            if !tile_rep.contains_key(tile) {
                tile_rep.insert(*tile, HashSet::new());
            }
            tile_rep.get_mut(tile).unwrap().insert(i);
        }
    }

    tile_rep
}


#[derive(Clone)]
pub struct Game {
    pub board: Board,
    players: Vec<Player>,
    history: Vec<Vec<usize>>, // each row is a move consisting of its tiles
    current_player: usize,  // index of current player in players
    legal_tiles: HashMap<usize, HashSet<usize>> // Map tile to index of the overall move
}

impl Reducible for Game {
    type Action = Action;

    fn reduce(self: Rc<Self>, action: Self::Action) -> Rc<Self> {
        match action {
            Action::PlacePiece(p, v, o) => {
                let mut new_state = (*self).clone();
                let player = &mut new_state.players[self.current_player];
                console::log!(
                    "Anchors",
                    player
                        .get_anchors()
                        .iter()
                        .map(|a| a.to_string())
                        .collect::<Vec<String>>()
                        .join(", ")
                );
                let piece = player.pieces[p].variants[v].clone();

                // Check if move is valid
                if !new_state.board.is_valid_move(&player, &piece, o) {
                    console::log!("Invalid move");
                    return self.into();
                }

                // Remove piece from player and place piece
                player.pieces.remove(p);
                let used_spaces = new_state.board.place_piece(player, &piece, o);
                new_state.current_player = self.next_player();

                // Update anchors for all players
                for player in &mut new_state.players {
                    player.use_anchors(&used_spaces);
                }

                // Add move to stack
                new_state.history.push(used_spaces.into_iter().collect());

                // Return new state
                new_state.into()
            }
            Action::Pass => {
                let mut new_state = (*self).clone();
                new_state.players.remove(self.current_player);

                if new_state.is_terminal() {
                    return Game::reset().into(); // TODO - need to handle better with message or something
                }

                new_state.current_player = self.current_player % new_state.players.len();
                new_state.into()
            }
            Action::Undo => {
                let mut new_state = (*self).clone();
                let last_move = new_state.history.pop().unwrap();
                let player = &new_state.players[self.current_player];
                // TODO: Need to implement undo
                new_state.into()
            }
            Action::ResetGame => Game::reset().into(),
        }
    }
}

impl Game {
    pub fn reset() -> Self {
        let mut players = Vec::new();
        for i in 1..5 {
            players.push(Player::new(i));
        }
        Game {
            board: Board::new(),
            players,
            history: Vec::new(),
            current_player: 0,
            legal_tiles: HashMap::new(),
        }
    }

    pub fn apply(&mut self, tile: usize) -> () {

        // Place piece on board
        self.board.place_tile(tile, self.current_player as u8);

        // Update legal tiles
        let valid_moves = self.legal_tiles.remove(&tile).unwrap();
        for (tile, move_set) in self.legal_tiles.clone() {
            move_set.iter().filter(|m| !valid_moves.contains(m));
            if move_set.len() < 1 {
                self.legal_tiles.remove(&tile);
            }
        }

        // Advance to next player if necessary
        while self.legal_tiles.len() == 0 && !self.is_terminal(){
            self.current_player = self.next_player();
            self.legal_tiles = get_tile_moves(&self.board, &self.players[self.current_player]);
        }

    }

    pub fn get_board(&self) -> &[u8; BOARD_SPACES] {
        &self.board.board
    }

    pub fn next_player(&self) -> usize {
        (self.current_player + 1) % self.players.len()
    }

    pub fn current_player(&self) -> usize {
        self.current_player
    }

    pub fn get_current_player_pieces(&self) -> Vec<Piece> {
        self.players[self.current_player].pieces.clone()
    }

    pub fn get_current_anchors(&self) -> HashSet<usize> {
        self.players[self.current_player].get_anchors()
    }

    pub fn legal_tiles(&self) -> Vec<usize> {
        self.legal_tiles.keys().map(|k| *k).collect()
    }

    pub fn get_payoff(&self) -> Vec<f32> {
        vec![0.0; 4] // TODO: flesh out
    }

    pub fn is_terminal(&self) -> bool {
        self.players.len() == 0
    }

    /// Get a representation of the state for the neural network
    /// This representation includes the board and the legal tiles
    pub fn get_representation(&self) -> StateRepresentation {

        // Get rep for the pieces on the board
        let board = &self.board.board;
        let mut board_rep = [[false; BOARD_SPACES]; 5];
        for i in 0..BOARD_SPACES {
            let player = board[i] & 0b1111; // check if there is a player piece
            if player != 0 {
                board_rep[player as usize - 1][i] = true;
            }
        }

        // Get rep for the legal spaces
        let legal_moves = self.legal_tiles();
        for tile in legal_moves {
            board_rep[4][tile] = true;
        }

        StateRepresentation {
            boards: board_rep.into_iter().flat_map(|inner| inner).collect(),
            player: self.current_player() as i32,
        }

    }
}
