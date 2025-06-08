use gloo_console as console;
use gloo_dialogs::alert;
use reqwasm::http::Request;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use wasm_bindgen_futures::spawn_local;
use yew::prelude::*;

use crate::board::BlokusBoard;
use crate::pieces::PieceTray;
use blokus::game::Game;
use blokus::board::RepType;

const SERVER_ADDRESS: &str = "https://aracape-blokus.hf.space/gradio_api/call/predict";
const D: usize = 20;

#[derive(Deserialize, Debug)]
struct EventResponse {
    event_id: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct ModelOutput {
    policy: Vec<f32>,
    values: Vec<f32>,
}

fn softmax(arr: Vec<f32>) -> Vec<f32> {
    let max_val = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Subtract max for numerical stability, then exp
    let exp_vals: Vec<f32> = arr.iter().map(|&val| (val - max_val).exp()).collect();
    
    // Normalize by sum
    let sum: f32 = exp_vals.iter().sum();
    exp_vals.iter().map(|&val| val / sum).collect()
}

pub fn parse_sse_response(sse_text: &str) -> Result<Vec<Value>, String> 
{    
    for line in sse_text.lines() {
        let trimmed = line.trim();
        
        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with(':') {
            continue;
        }
        
        // Parse SSE field: value format
        if let Some(colon_pos) = trimmed.find(':') {
            let field = &trimmed[..colon_pos].trim();
            let value = trimmed[colon_pos + 1..].trim();
            
            // We're interested in data fields
            if *field == "data" {
                match serde_json::from_str::<Vec<Value>>(value) {
                    Ok(data_array) => return Ok(data_array),
                    Err(e) => {
                        return Err(format!("Failed to parse Gradio data array: {}", e));
                    }
                }
            }
        }
    }
    
    return Err("No valid JSON data found in SSE response".to_string());
}

pub fn parse_python_tuple_string(tuple_str: &str) -> Result<(Vec<f32>, Vec<f32>), String> {
    // Remove outer parentheses
    let content = tuple_str.trim();
    if !content.starts_with('(') || !content.ends_with(')') {
        return Err("String doesn't start and end with parentheses".to_string());
    }
    
    let inner = &content[1..content.len()-1];
    
    // Find the split point between the two arrays
    // Look for "], [" pattern
    let split_pattern = "], [";
    let split_pos = inner.find(split_pattern)
        .ok_or("Could not find split between two arrays")?;
    
    // Extract the two array strings
    let first_array_str = format!("[{}]", &inner[1..split_pos]); // Remove leading '[' and add it back
    let second_array_str = format!("[{}]", &inner[split_pos + split_pattern.len()..inner.len()-1]); // Remove trailing ']' and add it back
    
    // Parse both arrays
    let policy: Vec<f32> = serde_json::from_str(&first_array_str)
        .map_err(|e| format!("Failed to parse policy array: {}", e))?;
    
    let values: Vec<f32> = serde_json::from_str(&second_array_str)
        .map_err(|e| format!("Failed to parse values array: {}", e))?;
    
    Ok((policy, values))
}

/// Query the model server
async fn query_model(state: &Game) -> Result<ModelOutput, String> {
    let rep = state.get_game_state(RepType::Token);
    let request = serde_json::json!({
        "data": [serde_json::to_string(&rep).unwrap()]
    });
    state.board.print_board();

    let post_response = Request::post(SERVER_ADDRESS)
        .header("Content-Type", "application/json")
        .body(serde_json::to_string(&request).unwrap())
        .send()
        .await
        .map_err(|e| format!("Failed to submit request: {:?}", e))?;
    
    let event_response: EventResponse = post_response.json().await
        .map_err(|e| format!("Failed to parse event response: {:?}", e))?;
        
    // Step 2: GET request to fetch results using event_id
    let get_url = format!("{}/{}", SERVER_ADDRESS, event_response.event_id);
    
    // Poll until we get results
    for attempt in 0..60 { // 60 attempts = 60 seconds max
        
        let get_response = Request::get(&get_url)
            .send()
            .await
            .map_err(|e| format!("Failed to poll results: {:?}", e))?;

        let response_text = get_response.text().await
            .map_err(|e| format!("Failed to get response text: {:?}", e))?;

        // Parse the SSE response
        let data = parse_sse_response(&response_text)
            .map_err(|e| format!("Failed to parse SSE response: {}", e))?;

        // Check if we have results
        if data.is_empty() {
            console::log!(format!("Polling attempt {}, no data yet...", attempt + 1));
            continue;
        }

        let tuple_string = data[0].as_str()
        .ok_or("Expected string in data array")?;
    
        // Parse the Python tuple format
        let (policy, values) = parse_python_tuple_string(tuple_string)
            .map_err(|e| format!("Failed to parse tuple string: {}", e))?;
        
        let output = ModelOutput {
            policy,
            values
        };
        return Ok(output);
    }
    
    Err("Timeout waiting for model response after 60 seconds".to_string())
}

/// Takes state and returns tile to place along with policy and values
async fn get_ai_move(state: &Game) -> Result<(usize, Vec<f32>, Vec<f32>), String> {
    let response = query_model(state).await.unwrap();
    let tile = response
        .policy
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .expect("No policy found");

    Ok((tile, response.policy, response.values))
}

/// Handles a single AI move and triggers the next one
async fn handle_single_ai_move(
    current_state: Game,
    state_setter: UseStateHandle<Game>,
    policy_setter: UseStateHandle<Vec<f32>>,
    scores_setter: UseStateHandle<Vec<f32>>,
) {
    let mut next_state = current_state.clone();
    
    // Check if current player is AI and game isn't over
    if next_state.current_player() == 0 || next_state.is_terminal() {
        // Human's turn or game over - get policy/eval and stop
        if !next_state.is_terminal() {
            if let Ok(response) = query_model(&next_state).await {
                policy_setter.set(response.policy);
                scores_setter.set(softmax(response.values));
            }
        } else {
            alert_game_over(&next_state);
        }
        return;
    }
    
    // Make AI move and get updated policy/values for the current position
    match get_ai_move(&next_state).await {
        Ok((tile, policy, values)) => {
            // Update policy and scores first (for the position before the move)
            policy_setter.set(policy);
            scores_setter.set(softmax(values));
            
            if let Err(e) = next_state.apply(tile, None) {
                console::error!("Failed to apply AI move:", e);
                return;
            }
            
            console::log!("AI placed piece at: {:?}", tile);
            console::log!("Current player: ", next_state.current_player());
            
            // Update the UI with the new state
            state_setter.set(next_state.clone());
            
            // Check if game is over after this move
            if next_state.is_terminal() {
                alert_game_over(&next_state);
                return;
            }
            
            // Add a small delay to make the move and updates visible
            //TimeoutFuture::new(5000).await;
            
            // Recursively handle the next AI move
            let state_setter_clone = state_setter.clone();
            let policy_setter_clone = policy_setter.clone();
            let scores_setter_clone = scores_setter.clone();
            
            spawn_local(async move {
                handle_single_ai_move(
                    next_state,
                    state_setter_clone,
                    policy_setter_clone,
                    scores_setter_clone,
                ).await;
            });
        }
        Err(e) => {
            console::error!("Failed to get AI move:", e);
        }
    }
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
    let show_eval = use_state(|| true);
    let show_policy = use_state(|| true);
    let policy = use_state(|| vec![0.0; 400]);
    let scores = use_state(|| vec![0.25; 4]);

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
            
            state.set(new_state.clone());

            // Check if game is over
            if new_state.is_terminal() {
                alert_game_over(&new_state);
                return;
            }

            // Start the chain of AI moves
            let state_clone = state.clone();
            let policy_clone = policy.clone();
            let scores_clone = scores.clone();
            
            spawn_local(async move {
                handle_single_ai_move(
                    new_state,
                    state_clone,
                    policy_clone,
                    scores_clone,
                ).await;
            });
        })
    };

    let on_reset = {
        let state = state.clone();
        let policy_clone = policy.clone();
        let scores_clone = scores.clone();
        Callback::from(move |_| {
            state.set(Game::reset(D));
            policy_clone.set(vec![0.0; 400]);
            scores_clone.set(vec![0.25; 4]);
        })
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
                <h1>{ "AlphaBlokus" }</h1>
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
                    <p class="controls-text">{"
                        Start in the upper left\n
                        Place Piece: Drag\n
                        Rotate Piece: click then 'r'\n
                        Flip Piece: click then 'f'\n
                    "}</p>
                    
                    <div class="checkbox-group">
                        <label>
                            <input type="checkbox" checked={*show_eval} onclick={toggle_eval}/>
                            { "Show Eval Bar" }
                        </label>
                        <label>
                            <input type="checkbox" checked={*show_policy} onclick={toggle_policy}/>
                            { "Show AI Heat Map" }
                        </label>
                    </div>
                    
                    <button onclick={on_reset}>{ "Reset Game" }</button>
                    <a href="https://github.com/alexracape/blokus-engine">{"Learn more about the project"}</a>
                </div>

            </div>
            </div>

            <PieceTray pieces={state.get_current_player_pieces()} player_num={state.current_player() as u8 + 1} />

        </div>
    }
}