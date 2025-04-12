use gloo_console as console;
use gloo_dialogs::alert;
use reqwasm::http::Request;
use serde::{Deserialize, Serialize};
use serde_json;
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

use crate::board::BlokusBoard;
use crate::pieces::PieceTray;
use blokus::game::Game;
use blokus::board::RepType;

const SERVER_ADDRESS: &str = "http://127.0.0.1:8000/process_request";
const D: usize = 20;
const BOARD_SIZE: usize = 400;

#[derive(Serialize, Deserialize, Debug)]
struct GameStateRequest {
    player: usize,
    data: Vec<Vec<Vec<bool>>>,
}

#[derive(Serialize, Deserialize, Debug)]
struct GameStateResponse {
    policy: Vec<f32>,
    values: Vec<f32>,
    status: i32,
}

fn print_rep(rep: &Vec<Vec<Vec<bool>>>) {
    let mut str_rep = String::new();
    for i in 0..5 {
        for j in 0..D {
            for k in 0..D {
                if rep[i][j][k] {
                    str_rep.push_str("[X]");
                } else {
                    str_rep.push_str("[ ]");
                }
            }
            str_rep.push_str("\n");
        }
        str_rep.push_str("\n");
    }
    console::log!(str_rep);
}

fn get_state_rep(game: &Game) -> GameStateRequest {
    GameStateRequest {
        player: game.current_player(),
        data: game.get_game_state(RepType::Token),
    }
}

/// Rotates the policy 90 degrees to the right
fn rotate_policy(state: Vec<f32>) -> Vec<f32> {
    let mut rotated = vec![0.0; BOARD_SIZE];
    for i in 0..D {
        for j in 0..D {
            rotated[j * D + (D - 1 - i)] = state[i * D + j];
        }
    }

    rotated.to_vec()
}

/// Query the model server
async fn query_model(state: &Game) -> Result<GameStateResponse, String> {
    let request = get_state_rep(state);
    print_rep(&request.data);
    let serialized_request = serde_json::to_string(&request).unwrap();
    let current_player = state.current_player();

    // Send POST request to FastAPI server
    match Request::post(SERVER_ADDRESS)
        .header("Content-Type", "application/json")
        .body(serialized_request)
        .send()
        .await
    {
        Ok(response) => {
            let json_value = response.json().await.unwrap();
            let mut response: GameStateResponse = serde_json::from_value(json_value).unwrap();
            if response.status != 200 {
                console::error!("AI failed to find a move");
                return Err("Failed to query the model".to_string());
            }

            // Reorient
            for _ in 0..(current_player) {
                response.policy = rotate_policy(response.policy);
            }
            response.values.rotate_right(current_player);
            Ok(response)
        }
        Err(e) => Err(format!("Failed to get AI move: {:?}", e)),
    }
}

/// Takes state and returns tile to place
async fn get_ai_move(state: &Game) -> Result<usize, String> {
    let response = query_model(state).await.unwrap();
    let tile = response
        .policy
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .expect("No policy found");

    Ok(tile)
}

/// Applies AI moves to state after player has gone
async fn handle_ai_moves(state: Game) -> Game {
    let mut next_state = state.clone();
    let mut current_ai = next_state.current_player();
    while current_ai != 0 && !next_state.is_terminal() {
        // THIS IS THE CONDITION, DOESN'T WORK WHEN HUMAN IS ELIMINATED
        let tile = get_ai_move(&next_state).await.unwrap();
        if let Err(e) = next_state.apply(tile, None) {
            console::error!("Failed to apply AI move:m", e);
            break;
        }

        current_ai = next_state.current_player();
        console::log!("AI placed piece at: {:?}", tile);
        console::log!("Current player: ", current_ai);
    }

    next_state
}

fn alert_game_over(game: &Game) {
    let scores = game.get_score();
    let best_score = scores.iter().max().unwrap();
    let winners = scores
        .iter()
        .enumerate()
        .filter_map(|(i, s)| if s == best_score { Some(i) } else { None })
        .collect::<Vec<_>>();

    let mut message = if winners.len() == 1 {
        format!("Player {} wins!", winners[0] + 1)
    } else {
        format!(
            "Players {:?} tie!",
            winners.iter().map(|w| w + 1).collect::<Vec<_>>()
        )
    };

    message.push_str("\n\nScores:\n");
    for (i, score) in scores.iter().enumerate() {
        message.push_str(&format!("Player {}: {}\n", i + 1, score));
    }
    alert(&message);
}

#[function_component]
pub fn App() -> Html {
    let state = use_state(|| Game::reset(D));
    let show_eval = use_state(|| false);
    let show_policy = use_state(|| false);
    let policy = use_state(|| vec![0.0; 400]);
    let scores = use_state(|| vec![0.25; 4]);

    // let policy_state = policy.clone();
    // let scores_state = scores.clone();
    // let copy = state.clone();
    // spawn_local({
    //     async move {
    //         let response = query_model(&copy).await.unwrap();
    //         policy_state.set(response.policy);
    //         scores_state.set(response.values);
    //     }
    // });

    let on_board_drop = {
        let state = state.clone();
        let policy = policy.clone();
        let scores = scores.clone();
        Callback::from(move |(p, v, offset)| {
            // Don't do anything if game is over
            if state.is_terminal() {
                return;
            }

            // Place piece on board
            let new_state = match state.place_piece(p, v, offset) {
                Ok(s) => s,
                Err(e) => {
                    console::error!("Failed to place piece: {:?}", e);
                    return;
                }
            };
            let game = new_state.clone();
            state.set(new_state);

            // Check if game is over
            if game.is_terminal() {
                alert_game_over(&game);
                return;
            }

            // Handle AI moves
            let state = state.clone();
            let policy_state = policy.clone();
            let scores_state = scores.clone();
            spawn_local({
                async move {
                    let new_state = handle_ai_moves(game.clone()).await;
                    state.set(new_state.clone());
                    if new_state.is_terminal() {
                        alert_game_over(&game);
                    }

                    // Get Policy and Eval
                    let response = query_model(&new_state).await.unwrap();
                    policy_state.set(response.policy);
                    scores_state.set(response.values);
                }
            });
        })
    };

    let on_reset = {
        let state = state.clone();
        Callback::from(move |_| state.set(Game::reset(D)))
    };

    let toggle_eval = {
        let show_eval = show_eval.clone();
        Callback::from(move |_| show_eval.set(!*show_eval))
    };

    let toggle_policy = {
        let show_policy = show_policy.clone();
        Callback::from(move |_| show_policy.set(!*show_policy))
    };

    html! {
        <div>
            <div class="title">
                <h1>{ "Blokus Engine" }</h1>
            </div>

            <div class="container">
            <div class="layout ">

                <div class="side-panel">
                    <h2>{ "Players Remaining" }</h2>
                    <div class="player-icons">

                        <div class={format!("square red {}", if !state.is_player_active(0) { "eliminated" } else { "" })}></div>
                        <div class={format!("square blue {}", if !state.is_player_active(1) { "eliminated" } else { "" })}></div>
                        <div class={format!("square green {}", if !state.is_player_active(2) { "eliminated" } else { "" })}></div>
                        <div class={format!("square yellow {}", if !state.is_player_active(3) { "eliminated" } else { "" })}></div>
                    </div>
                    { if *show_eval { html! {
                        <div>
                            <h2>{ "Eval Bar" }</h2>
                            <div class="eval">
                                <div class="eval-section red" style={format!("width: {}%", 100.0 * scores[0])}></div>
                                <div class="eval-section blue" style={format!("width: {}%", 100.0 * scores[1])}></div>
                                <div class="eval-section green" style={format!("width: {}%", 100.0 * scores[2])}></div>
                                <div class="eval-section yellow" style={format!("width: {}%", 100.0 * scores[3])}></div>
                            </div>
                        </div>
                    }} else {
                        html!{}
                    }}

                </div>

                <div class="main-board">
                    <BlokusBoard board={state.get_board_state()} policy={(*policy).clone()} show_policy={*show_policy} on_board_drop={on_board_drop} anchors={state.get_current_anchors()} />
                </div>

                <div class="side-panel">
                    <h2>{ "Controls" }</h2>
                    <p style={"white-space: pre-line"}>{"
                        Select Piece: Click\n
                        Place Piece: Drag\n
                        Rotate Piece: r\n
                        Flip Piece: f\n
                    "}</p>
                    <label>
                        <input type="checkbox" checked={*show_eval} onclick={toggle_eval}/>
                        { "Show Eval Bar" }
                    </label>
                    <label>
                        <input type="checkbox" checked={*show_policy} onclick={toggle_policy}/>
                        { "Show AI Heat Map" }
                    </label>
                    <button onclick={on_reset}>{ "Reset Game" }</button>
                </div>

            </div>
            </div>

            <PieceTray pieces={state.get_current_player_pieces()} player_num={state.current_player() as u8 + 1} />

        </div>
    }
}
