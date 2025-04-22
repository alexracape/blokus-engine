use std::collections::{HashMap, HashSet};
use std::iter::zip;

use crate::board::{Board, RepType};
use crate::pieces::{Piece, PieceVariant};

const NUM_PLAYERS: usize = 4;

type Move       = (usize, usize, usize);    // indices for piece, variant, offset
type TileGroup  = Vec<usize>;               // Group of tiles that correspond to a piece

/// Get the legal moves for a piece
fn get_piece_moves(
    piece_i: usize,
    board: &Board,
    player: usize,
) -> (Vec<Move>, Vec<TileGroup>) {
    let mut moves = Vec::new();
    let mut tile_groups = Vec::new();
    let piece = &board.get_pieces(player)[piece_i];
    for anchor in &board.get_anchors(player) {
        for (var_i, variant) in piece.variants.iter().enumerate() {
            for offset in &variant.offsets {
                // Check underflow
                if offset > anchor {
                    continue;
                }

                let total_offset = anchor - offset; // offset to anchor, then offset to line up piece
                if board.is_valid_move(player, variant, total_offset) {
                    let mut tiles = Vec::new();
                    for (j, square) in variant.variant.iter().enumerate() {
                        if *square {
                            tiles.push(total_offset + j);
                        }
                    }
                    tile_groups.push(tiles);
                    moves.push((piece_i, var_i, total_offset))
                }
            }
        }
    }

    (moves, tile_groups)
}

/// Get the legal moves for a player, tile placements grouped by move
fn get_moves(board: &Board, player: usize) -> (Vec<Move>, Vec<TileGroup>) {
    let mut moves = Vec::new();
    let mut tile_groups = Vec::new();
    for piece in 0..board.get_pieces(player).len() {
        let (piece_moves, piece_tiles) = get_piece_moves(piece, board, player);
        moves.extend(piece_moves);
        tile_groups.extend(piece_tiles);
    }

    (moves, tile_groups)
}

/// Get the tile based representation for legal moves
fn get_tile_moves(board: &Board, player: usize) -> HashMap<usize, HashSet<Move>> {
    let mut tile_rep = HashMap::new();
    let (moves, tile_groups) = get_moves(board, player);

    for (id, tiles) in zip(moves, tile_groups) {
        for tile in tiles {
            tile_rep
                .entry(tile)                        // takes ownership of `tile`
                .or_insert_with(HashSet::new)       // creates a new HashSet if absent
                .insert(id);
        }
    }

    tile_rep
}

/// Rotates the tensor of boards 90 degrees to the left
// fn rotate_state(state: &Vec<Vec<Vec<bool>>>) -> Vec<Vec<Vec<bool>>> {
//     let dim = state[0].len();
//     let num_layers = state.len();

//     let mut new_state = vec![vec![vec![false; dim]; dim]; num_layers];
//     for i in 0..num_layers {
//         for j in 0..dim {
//             for k in 0..dim {
//                 new_state[i][j][k] = state[i][k][dim - j - 1];
//             }
//         }
//     }

//     new_state
// }

#[derive(Clone)]
pub struct Game {
    pub board: Board,
    pub history: Vec<(i32, i32)>, // Stack of (player, tile)
    eliminated: [bool; NUM_PLAYERS],
    current_player: usize, // Zero indexed!
    legal_tiles: HashMap<usize, HashSet<Move>>, // Map tile to index of the overall move
    last_piece_lens: [u32; NUM_PLAYERS], // Size of the last piece placed by each player
}

impl Game {
    pub fn reset(dim: usize) -> Self {
        let board = Board::new(dim);
        let legal_tiles = get_tile_moves(&board, 0);

        Game {
            board,
            history: Vec::new(),
            eliminated: [false; NUM_PLAYERS],
            current_player: 0,
            legal_tiles,
            last_piece_lens: [0; NUM_PLAYERS],
        }
    }

    pub fn place_piece(&self, p: usize, v: usize, o: usize) -> Result<Game, String> {
        let mut new_state = self.clone();
        let player = self.current_player;
        let piece = self.get_piece(player, p, v);

        // Check if move is valid
        if !new_state.board.is_valid_move(player, &piece, o) {
            return Err("Invalid move".to_string());
        }

        // Break move into tiles and apply individually
        let offsets = piece.offsets.iter().collect::<Vec<_>>();
        let last_index = offsets.len().saturating_sub(1);
        for (i, tile_offset) in offsets.iter().enumerate() {
            let tile = o + *tile_offset;
            let result = if i == last_index {
                new_state.apply(tile, Some(p))
            } else {
                new_state.apply(tile, None)
            };

            match result {
                Ok(_) => (),
                Err(e) => return Err(e),
            }
        }

        Ok(new_state)
    }

    // Plays a tile on the board
    // Not thrilled with the implementation
    // Right now it forces you to place as many tiles as is legal or you can pass a piece you
    // want to finish playing. This is really only used by the GUI rn
    pub fn apply(&mut self, tile: usize, piece_to_finish: Option<usize>) -> Result<(), String> {
        // Place piece on board
        self.board.place_tile(tile, self.current_player);
        self.history.push((self.current_player as i32, tile as i32));

        // Update legal tiles
        let valid_moves = match self.legal_tiles.remove(&tile) {
            Some(moves) => moves,
            None => {
                return Err(format!(
                    "Invalid move - Player {}, Tile {}",
                    self.current_player, tile
                ))
            }
        };
        for (tile, move_set) in self.legal_tiles.clone() {
            self.legal_tiles.insert(
                tile,
                move_set.intersection(&valid_moves).copied().collect(),
            );
            if let Some(moves) = self.legal_tiles.get(&tile) {
                if moves.is_empty() {
                    self.legal_tiles.remove(&tile);
                }
            }
        }

        // Advance to next player if necessary
        if self.legal_tiles.is_empty() || piece_to_finish.is_some() {
            // Removing the player's piece
            let piece = match piece_to_finish {
                Some(p) => p,
                None => valid_moves
                    .iter()
                    .next()
                    .ok_or("valid_moves must be non-empty")?
                    .0,
            };
            self.last_piece_lens[self.current_player] = self
                .board
                .get_pieces(self.current_player)
                .remove(piece)
                .points;
            self.board.use_piece(self.current_player, piece);

            // Advance to next player
            self.advance_player();
        }

        Ok(())
    }

    /// Cycle to the next player
    /// Eliminates any players that have no legal moves
    /// Returns index of the current player
    pub fn advance_player(&mut self) -> usize {
        // Return if the game is over
        if self.is_terminal() {
            return self.current_player;
        }

        // Cycle to the next player
        self.current_player = (self.current_player + 1) % NUM_PLAYERS;
        self.legal_tiles = get_tile_moves(&self.board, self.current_player);

        // If the player is already out of the game, cycle to the next player
        // If they have no legal moves, eliminate them and advance
        if self.eliminated[self.current_player] {
            self.advance_player();
        } else if self.legal_tiles.is_empty() {
            self.eliminated[self.current_player] = true;
            self.advance_player();
        }

        self.current_player
    }

    pub fn current_player(&self) -> usize {
        // self.players_remaining.get(self.player_index).copied()
        self.current_player
    }

    pub fn get_current_player_pieces(&self) -> Vec<Piece> {
        self.board.get_pieces(self.current_player)
    }

    pub fn get_piece(&self, player: usize, piece: usize, variant: usize) -> PieceVariant {
        self.board.get_pieces(player)[piece].variants[variant].clone()
    }

    pub fn get_current_anchors(&self) -> HashSet<usize> {
        self.board.get_anchors(self.current_player)
    }

    pub fn get_legal_tiles(&self) -> Vec<usize> {
        self.legal_tiles.keys().copied().collect()
    }

    /// Get the scores for the end of the game
    pub fn get_score(&self) -> Vec<i32> {
        self.board.get_scores(self.last_piece_lens)
    }

    /// Player fewest tiles remaining wins, payoff is between 0 and 1
    pub fn get_payoff(&self) -> Vec<f32> {
        let scores = self.board.get_scores(self.last_piece_lens);
        let mut payoff = vec![0.0; 4];
        let mut indices = Vec::new();
        let mut highest_score = scores[0];
        for (i, score) in scores.iter().enumerate() {
            match score.cmp(&highest_score) {
                std::cmp::Ordering::Equal => {
                    indices.push(i);
                }
                std::cmp::Ordering::Greater => {
                    indices.clear();
                    indices.push(i);
                    highest_score = *score;
                }
                std::cmp::Ordering::Less => {
                    // do nothing
                }
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

    pub fn is_player_active(&self, player: usize) -> bool {
        !self.eliminated[player]
    }

    pub fn get_board_state(&self) -> Vec<Vec<Vec<bool>>> {
        self.board.get_rep(RepType::Channel, 0)
    }

    pub fn get_game_state(&self, format: RepType) -> Vec<Vec<Vec<bool>>> {
        let mut board_rep = self.board.get_rep(format.clone(), self.current_player);
        let dim = self.board.get_dim();
        let legal_moves = self.get_legal_tiles();

        // Get rep for the legal spaces
        match format {
            RepType::Channel => {
                let mut legal_move_rep = vec![vec![false; dim]; dim];
                for tile in legal_moves {
                    let row = tile / dim;
                    let col = tile % dim;
                    legal_move_rep[row][col] = true;
                }
                board_rep.push(legal_move_rep);
            },
            RepType::Token => {
                for (row_idx, row) in board_rep.iter_mut().enumerate() {
                    for (col_idx, cell) in row.iter_mut().enumerate() {
                        let tile = row_idx * dim + col_idx;
                        cell.push(legal_moves.contains(&tile));
                    }
                }
            }
        }

        // Rotate the board to the current player perspective
        // Comment out for now for simplicity
        // for _ in 0..self.current_player {
        //     board_rep = rotate_state(board_rep);
        // }

        board_rep
    }
}
