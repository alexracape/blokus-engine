use std::rc::Rc;

use yew::prelude::*;
use gloo_console as console;

use crate::board::Board;
use crate::player::{self, Player};
use crate::pieces::Piece;


pub enum Action {
    PlacePiece(usize, usize, usize),
    Undo,
    ResetGame,
}

#[derive(Clone)]
pub struct State {
    board: Board,
    players: Vec<Player>,
    move_stack: Vec<(usize, usize, usize)>,
    current_player: usize,
}

impl Reducible for State {
    type Action = Action;

    fn reduce(self: Rc<Self>, action: Self::Action) -> Rc<Self> {
        match action {
            Action::PlacePiece(p, v, o) => {
                let mut new_state = (*self).clone();
                let mut player = new_state.players[self.current_player].clone();
                let piece = player.pieces[p].variants[v].clone();
                
                // Check if move is valid
                if !new_state.board.is_valid_move(&player, &piece, o) {
                    console::log!("Invalid move");
                    return self.into();
                }

                // Remove piece from player and place piece
                new_state.players[self.current_player].pieces.remove(p);
                new_state.board.place_piece(&mut player, &piece, o);
                new_state.current_player = (self.current_player + 1) % self.players.len();

                // Add move to stack
                new_state.move_stack.push((p, v, o));

                // Return new state
                new_state.into()
            },
            Action::Undo => {
                let mut new_state = (*self).clone();
                let (p, v, o) = new_state.move_stack.pop().unwrap();
                let player = &new_state.players[self.current_player];
                let piece = player.pieces[p].variants[v].clone();
                new_state.board.remove_piece(player, &piece, o);
                new_state.current_player = (self.current_player + self.players.len() - 1) % self.players.len();
                new_state.into()
            },
            Action::ResetGame => State::reset().into()
        }
    }
}

impl State {
    pub fn reset() -> Self {
        let mut players = Vec::new();
        for i in 1..5 {
            players.push(Player::new(i));
        }
        State {
            board: Board::new(),
            players,
            move_stack: Vec::new(),
            current_player: 0,
        }
    }

    pub fn get_board(&self) -> &[u8; 400] {
        &self.board.board
    }

    pub fn get_current_player_pieces(&self) -> Vec<Piece> {
        self.players[self.current_player].pieces.clone()
    }
}
