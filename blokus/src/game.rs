use std::collections::{HashMap, HashSet};
use std::iter::zip;

use crate::board::{Action, Board};
use crate::pieces::{Piece, PieceVariant};

const D: usize = 20;
const BOARD_SPACES: usize = 400;
const NUM_PLAYERS: usize = 4;

/// Get the legal moves for a piece
fn get_piece_moves(piece_i: usize, board: &Board, player: usize) -> Vec<Action> {
    let mut moves = Vec::new();
    let piece = &board.get_pieces(player)[piece_i];
    for anchor in &board.get_anchors(player) {
        for (var_i, variant) in piece.variants.iter().enumerate() {
            for offset in &variant.offsets {
                // Check underflow
                if offset > anchor {
                    continue;
                }

                let total_offset = anchor - offset; // offset to anchor, then offset to line up piece
                let action = Action::new(player, piece_i, var_i, total_offset);
                if board.is_valid_move(&action) {
                    moves.push(action)
                }
            }
        }
    }

    moves
}

/// Get the legal moves for a player, tile placements grouped by move
fn get_moves(board: &Board, player: usize) -> Vec<Action> {
    let mut moves = Vec::new();
    for piece in 0..board.get_pieces(player).len() {
        moves.extend(get_piece_moves(piece, board, player));
    }

    moves
}

#[derive(Clone)]
pub struct Game {
    pub board: Board,
    pub history: Vec<Action>,            // Stack of moves made
    eliminated: [bool; NUM_PLAYERS],     // True @ index of finished players
    current_player: usize,               // Zero indexed!
    legal_moves: Vec<Action>,            // Set of moves
    last_piece_lens: [u32; NUM_PLAYERS], // Size of the last piece placed by each player
}

impl Game {
    pub fn reset() -> Self {
        let board = Board::new();
        let legal_moves = get_moves(&board, 0);

        Game {
            board,
            history: Vec::new(),
            eliminated: [false; NUM_PLAYERS],
            current_player: 0,
            legal_moves,
            last_piece_lens: [0; NUM_PLAYERS],
        }
    }

    pub fn place_piece(&self, action: Action) -> Game {
        assert_eq!(action.player, self.current_player);

        let mut new_state = self.clone();
        new_state.board.place_piece(&action);
        new_state.history.push(action);
        new_state.advance_player();
        new_state
    }

    /// Cycle to the next player
    /// Eliminates any players that have no legal moves
    pub fn advance_player(&mut self) {
        assert!(!self.is_terminal());
        loop {
            self.current_player = (self.current_player + 1) % NUM_PLAYERS;

            // Skip players that have already been eliminated
            if !self.is_player_active(self.current_player) {
                continue;
            }

            self.legal_moves = get_moves(&self.board, self.current_player);

            // Eliminate players with no valid moves
            if self.legal_moves.is_empty() {
                self.eliminated[self.current_player] = true;
                if self.is_terminal() {
                    break;
                }
            } else {
                break;
            }
        }
    }

    pub fn current_player(&self) -> usize {
        self.current_player
    }

    pub fn get_board(&self) -> &[u8; BOARD_SPACES] {
        &self.board.board
    }

    pub fn get_current_player_pieces(&self) -> Vec<Piece> {
        self.board.get_pieces(self.current_player)
    }

    pub fn get_piece(&self, action: &Action) -> PieceVariant {
        self.board.get_piece_variant(&action)
    }

    pub fn get_current_anchors(&self) -> HashSet<usize> {
        self.board.get_anchors(self.current_player)
    }

    /// Get the scores for the end of the game
    pub fn get_score(&self) -> [i32; 4] {
        self.board.get_scores(self.last_piece_lens)
    }

    /// Player fewest tiles remaining wins, payoff is between 0 and 1
    pub fn get_payoff(&self) -> Vec<f32> {
        let scores = self.board.get_scores(self.last_piece_lens);
        let mut payoff = vec![0.0; 4];
        let mut indices = Vec::new();
        let mut highest_score = scores[0];
        for (i, score) in scores.iter().enumerate() {
            if *score == highest_score {
                indices.push(i);
            } else if *score > highest_score {
                indices.clear();
                indices.push(i);
                highest_score = *score;
            }
        }

        for i in &indices {
            payoff[*i] = 1.0 / indices.len() as f32;
        }

        payoff
    }

    /// Check if all players have been eliminated
    pub fn is_terminal(&self) -> bool {
        self.eliminated.iter().all(|x| *x)
    }

    /// Check if a certain player is still in the game
    pub fn is_player_active(&self, player: usize) -> bool {
        !self.eliminated[player]
    }
}
