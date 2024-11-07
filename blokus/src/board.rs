/*
Blokus Board
*/

use std::collections::HashSet;

use crate::pieces::{Piece, PieceVariant, PIECE_TYPES};

pub const BOARD_SIZE: usize = 20;
const TOTAL_TILES: i32 = 89;
const CORNERS_OFFSETS: [i32; 4] = [
    1 + BOARD_SIZE as i32,
    -1 - BOARD_SIZE as i32,
    1 - BOARD_SIZE as i32,
    -1 + BOARD_SIZE as i32,
];

#[derive(Clone)]
pub struct Action {
    pub player: usize,
    pub piece: usize,
    pub variant: usize,
    pub offset: usize,
}

impl Action {
    pub fn new(player: usize, piece: usize, variant: usize, offset: usize) -> Action {
        Action {
            player,
            piece,
            variant,
            offset,
        }
    }
}

#[derive(Clone)]
pub struct Board {
    pub board: [u8; BOARD_SIZE * BOARD_SIZE], // 20x20 board
    pieces: [Vec<Piece>; 4],
    anchors: [HashSet<usize>; 4],
}

impl Board {
    pub fn new() -> Board {
        let mut pieces = Vec::new();
        for piece_type in PIECE_TYPES {
            pieces.push(Piece::new(piece_type));
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
        for i in 0..4 {
            let start = match i {
                0 => 0,
                1 => BOARD_SIZE - 1,
                2 => BOARD_SIZE * BOARD_SIZE - 1,
                3 => BOARD_SIZE * (BOARD_SIZE - 1),
                _ => panic!("Invalid player number"),
            };
            anchors[i].insert(start);
        }

        Board {
            board: [0; BOARD_SIZE * BOARD_SIZE],
            pieces: player_pieces,
            anchors,
        }
    }

    pub fn is_valid_move(&self, action: &Action) -> bool {
        // Get key data
        let player = action.player;
        let variant = self.get_piece_variant(action);
        let piece_squares = variant.offsets;
        let bits = variant.variant;
        let offset = action.offset;

        // Check if piece starts at a valid offset
        let overflow = offset + bits.len() > self.board.len();
        let wrap_around = offset % BOARD_SIZE + variant.width > BOARD_SIZE;
        if overflow || wrap_around {
            return false;
        }

        let board_slice = &self.board[offset..offset + bits.len()];
        let player_restricted: u8 = 1 << player + 4;
        let on_blanks = board_slice.iter().zip(bits.iter()).all(|(a, b)| {
            if *b {
                if *a & player_restricted != 0 {
                    return false;
                }
            }
            true
        });

        let on_anchor = piece_squares
            .iter()
            .any(|i| self.anchors[player].contains(&(offset + i)));
        on_blanks && on_anchor
    }

    /// Get a piece variant for a player
    pub fn get_piece_variant(&self, action: &Action) -> PieceVariant {
        self.pieces[action.player][action.piece].variants[action.variant].clone()
    }

    /// Place a piece on the board
    pub fn place_piece(&mut self, action: &Action) {
        let variant = self.get_piece_variant(action);

        // Check if move is valid
        assert!(self.is_valid_move(&action));

        // Break move into tiles and apply individually
        let offsets = variant.offsets.iter().collect::<Vec<_>>();
        for tile_offset in offsets.iter() {
            let tile = action.offset + *tile_offset;
            self.place_tile(tile, action.player);
        }
    }

    /// Place a tile on the board
    pub fn place_tile(&mut self, tile: usize, player: usize) {
        self.board[tile] = 0b1111_0000 | (player as u8 + 1);

        // Restrict adjacent square
        let player_restricted: u8 = 1 << player + 4;
        let neighbors = [
            (tile % BOARD_SIZE > 0, -1),                                 // Left
            (tile % BOARD_SIZE < BOARD_SIZE - 1, 1),                     // Right
            (tile >= BOARD_SIZE, -(BOARD_SIZE as isize)),                // Above
            (tile < BOARD_SIZE * (BOARD_SIZE - 1), BOARD_SIZE as isize), // Bellow
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
        for corner_offset in CORNERS_OFFSETS.iter() {
            // Skip if corner is above or below board or it is a restricted square
            let corner = tile as i32 + corner_offset;
            if corner < 0
                || corner >= (BOARD_SIZE * BOARD_SIZE) as i32
                || self.board[corner as usize] & player_restricted != 0
            {
                continue;
            }

            // Skip if corner wraps around to other side of board
            if tile % BOARD_SIZE == 0 && (corner as usize) % BOARD_SIZE == BOARD_SIZE - 1 {
                continue;
            }
            if tile % BOARD_SIZE == BOARD_SIZE - 1 && (corner as usize) % BOARD_SIZE == 0 {
                continue;
            }
            self.anchors[player].insert(corner as usize);
        }
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

    pub fn get_scores(&self, last_piece_lens: [u32; 4]) -> [i32; 4] {
        // Count the number of pieces on the board for each player
        let mut scores = [0; 4];
        for cell in self.board.iter() {
            let player = *cell & 0b1111;
            if player != 0 {
                scores[player as usize - 1] += 1;
            }
        }

        // 15 bonus points for playing all pieces
        for (i, pieces) in self.pieces.iter().enumerate() {
            // Subtract to get the number of pieces remaining
            scores[i] = scores[i] - TOTAL_TILES;

            if pieces.len() == 0 {
                scores[i] += 15;

                // 5 bonus points for playing your smallest piece last
                if last_piece_lens[i] == 1 {
                    scores[i] += 5;
                }
            }
        }

        scores
    }

    pub fn print_board(&self) {
        let player1_emoji = "🟥";
        let player2_emoji = "🟦";
        let player3_emoji = "🟨";
        let player4_emoji = "🟩";
        let empty_emoji = "⬜";
        let mut output = String::new();
        for i in 0..BOARD_SIZE {
            for j in 0..BOARD_SIZE {
                let cell_value = self.board[i * BOARD_SIZE + j] & 0b0000_1111;
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
        let board = Board::new();
        assert_eq!(board.board.len(), 400);
    }

    #[test]
    fn test_is_valid_move() {
        let board = Board::new();
        let action = Action {
            player: 0,
            piece: 0,
            variant: 0,
            offset: 0,
        };
        assert_eq!(board.is_valid_move(&action), true);

        let action = Action {
            player: 0,
            piece: 3,
            variant: 0,
            offset: 19,
        };
        assert!(board.is_valid_move(&action) == false);
    }
}
