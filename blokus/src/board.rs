/*
Blokus Board
*/

use std::{collections::HashSet, vec};

use crate::pieces::{Piece, PieceVariant, PIECE_TYPES};

const TOTAL_TILES: i32 = 89;

#[derive(Clone)]
pub enum RepType {
    Channel,
    Token,
}


#[derive(Clone)]
pub struct Board {
    dim: usize,
    board: Box<[u8]>, // 20x20 board
    pieces: [Vec<Piece>; 4],
    anchors: [HashSet<usize>; 4],
    corner_offsets: [i32; 4]
}

impl Board {
    pub fn new(dim: usize) -> Board {
        let mut pieces = Vec::new();
        for piece_type in PIECE_TYPES {
            pieces.push(Piece::new(piece_type, dim));
        }
        let player_pieces = [
            pieces.clone(),
            pieces.clone(),
            pieces.clone(),
            pieces.clone(),
        ];

        let mut anchors = [
            HashSet::new(),
            HashSet::new(),
            HashSet::new(),
            HashSet::new(),
        ];
        for (i, player_set) in anchors.iter_mut().enumerate() {
            let start = match i {
                0 => 0,
                1 => dim - 1,
                2 => dim * dim - 1,
                3 => dim * (dim - 1),
                _ => panic!("Invalid player number"),
            };
            player_set.insert(start);
        }

        let corner_offsets = [
            1 + dim as i32,
            -1 - dim as i32,
            1 - dim as i32,
            -1 + dim as i32,
        ];

        Board {
            dim,
            board: vec![0; dim * dim].into_boxed_slice(),
            pieces: player_pieces,
            anchors,
            corner_offsets
        }
    }

    pub fn is_valid_move(
        &self,
        player: usize,
        piece_variant: &PieceVariant,
        offset: usize,
    ) -> bool {
        // Check piece is within bounds and does not go over edge of board
        let variant = &piece_variant.variant;
        let piece_squares = &piece_variant.offsets;
        if offset + variant.len() > self.board.len() || offset % self.dim + piece_variant.width > self.dim {
            return false;
        }

        let board_slice = &self.board[offset..offset + variant.len()];
        let player_restricted: u8 = 1 << (player + 4);
        let on_blanks = board_slice.iter().zip(variant.iter()).all(|(a, b)| {
            if *b && *a & player_restricted != 0 {
                return false;
            }
            true
        });

        let on_anchor = piece_squares
            .iter()
            .any(|i| self.anchors[player].contains(&(offset + i)));
        on_blanks && on_anchor
    }

    /// Place a tile on the board
    pub fn place_tile(&mut self, tile: usize, player: usize) {
        self.board[tile] = 0b1111_0000 | (player as u8 + 1);

        // Restrict adjacent square
        let player_restricted: u8 = 1 << (player + 4);
        let neighbors = [
            (tile % self.dim > 0, -1),                             // Left
            (tile % self.dim < self.dim - 1, 1),                   // Right
            (tile >= self.dim, -(self.dim as isize)),              // Above
            (tile < self.dim * (self.dim - 1), self.dim as isize), // Bellow
        ];

        // Remove tile from all anchors if it is there
        for i in 0..4 {
            self.anchors[i].remove(&tile);
        }

        // Iterate over neighbors, restrict, and remove from anchors if necessary
        for &(in_bounds, offset) in &neighbors {
            if in_bounds {
                let neighbor = (tile as isize + offset) as usize;
                self.board[neighbor] |= player_restricted;
                self.anchors[player].remove(&neighbor);
            }
        }

        // Add new anchors
        for corner_offset in self.corner_offsets.iter() {
            // Skip if corner is above or below board or it is a restricted square
            let corner = tile as i32 + corner_offset;
            if corner < 0
                || corner >= (self.dim * self.dim) as i32
                || self.board[corner as usize] & player_restricted != 0
            {
                continue;
            }

            // Skip if corner wraps around to other side of board
            if tile % self.dim == 0 && (corner as usize) % self.dim == self.dim - 1 {
                continue;
            }
            if tile % self.dim == self.dim - 1 && (corner as usize) % self.dim == 0 {
                continue;
            }
            self.anchors[player].insert(corner as usize);
        }
    }

    pub fn get_dim(&self) -> usize {
        self.dim
    }

    pub fn get_anchors(&self, player: usize) -> HashSet<usize> {
        self.anchors[player].clone()
    }

    pub fn get_pieces(&self, player: usize) -> Vec<Piece> {
        self.pieces[player].clone()
    }

    pub fn use_piece(&mut self, player: usize, piece: usize) {
        self.pieces[player].remove(piece);
    }

    pub fn get_scores(&self, last_piece_lens: [u32; 4]) -> Vec<i32> {
        // Count the number of pieces on the board for each player
        let mut scores = vec![0; 4];
        for cell in self.board.iter() {
            let player = *cell & 0b1111;
            if player != 0 {
                scores[player as usize - 1] += 1;
            }
        }

        // 15 bonus points for playing all pieces
        for (i, pieces) in self.pieces.iter().enumerate() {
            // Subtract to get the number of pieces remaining
            scores[i] -= TOTAL_TILES;

            if pieces.is_empty() {
                scores[i] += 15;

                // 5 bonus points for playing your smallest piece last
                if last_piece_lens[i] == 1 {
                    scores[i] += 5;
                }
            }
        }

        scores
    }

    pub fn get_rep(&self, rep: RepType, current_player: usize) -> Vec<Vec<Vec<bool>>>{
        match rep {
            RepType::Channel => self.get_channel_rep(current_player),
            RepType::Token =>self.get_token_rep(current_player)
        }
    }

    fn get_token_rep(&self, current_player: usize) -> Vec<Vec<Vec<bool>>> {
        let spaces = self.dim * self.dim;
        let mut board_state = vec![vec![vec![false; 4]; self.dim]; self.dim];
        for i in 0..spaces {
            let player = (self.board[i] & 0b1111) as usize; // check if there is a player piece
            if player != 0 {
                // Player here is 1 indexed because 0 is empty
                let player_slot = (4 + (player - 1) - current_player) % 4; // orient to current player (0 indexed)
                let row = i / self.dim;
                let col = i % self.dim;
                board_state[row][col][player_slot] = true;
            }
        }

        board_state
    }

    fn get_channel_rep(&self, current_player: usize) -> Vec<Vec<Vec<bool>>> {
        let spaces = self.dim * self.dim;
        let mut board_state = vec![vec![vec![false; self.dim]; self.dim]; 4];
        for i in 0..spaces {
            let player = (self.board[i] & 0b1111) as usize; // check if there is a player piece
            if player != 0 {
                // Player here is 1 indexed because 0 is empty
                let player_board = (4 + (player - 1) - current_player) % 4; // orient to current player (0 indexed)
                let row = i / self.dim;
                let col = i % self.dim;
                board_state[player_board][row][col] = true;
            }
        }

        board_state
    }

    pub fn print_board(&self) {
        let player1_emoji = "🟥";
        let player2_emoji = "🟦";
        let player3_emoji = "🟨";
        let player4_emoji = "🟩";
        let empty_emoji = "⬜";
        let mut output = String::new();
        for i in 0..self.dim {
            for j in 0..self.dim {
                let cell_value = self.board[i * self.dim + j] & 0b0000_1111;
                let emoji_to_print = match cell_value {
                    1 => player1_emoji,
                    2 => player2_emoji,
                    3 => player3_emoji,
                    4 => player4_emoji,
                    _ => empty_emoji,
                };
                output.push_str(emoji_to_print);
            }
            output.push_str("\n");
        }
        println!("{}", output);
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_creation() {
        let board = Board::new(20);
        assert_eq!(board.board.len(), 400);
    }

    #[test]
    fn test_variable_size() {
        let board = Board::new(10);
        assert_eq!(board.board.len(), 100);
    }

    #[test]
    fn test_is_valid_move() {
        let board = Board::new(20);
        let piece = PieceVariant::new(vec![vec![true, true]], 20);
        assert_eq!(board.is_valid_move(0, &piece, 0), true);
        assert!(board.is_valid_move(0, &piece, 19) == false);
    }

    #[test]
    fn test_rep() {
        let mut board = Board::new(5); // 3x3 board
        let mut expected = vec![vec![vec![false; 5]; 5]; 4];
        assert!(board.get_rep(RepType::Channel, 0) == expected);

        board.place_tile(0, 0);
        expected[0][0][0] = true;
        assert!(board.get_rep(RepType::Channel, 0) == expected);
        expected[0][0][0] = false;

        expected[3][0][0] = true;
        assert!(board.get_rep(RepType::Channel, 1) == expected);
        expected[3][0][0] = false;

        expected = vec![vec![vec![false; 4]; 5]; 5];
        assert!(board.get_rep(RepType::Token, 0) == expected);
    }


}
