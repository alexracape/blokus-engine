use std::collections::HashSet;

use wasm_bindgen::JsCast;
use web_sys::HtmlElement;

use yew::events::DragEvent;
use yew::prelude::*;
use yew::{function_component, html, Properties};


fn softmax(logits: &[f32]) -> Vec<f32> {
    // Find max for numerical stability
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    // Compute exp(x - max) for each logit
    let exp_logits: Vec<f32> = logits.iter()
        .map(|&x| (x - max_logit).exp())
        .collect();
    
    // Sum all exponentials
    let sum_exp: f32 = exp_logits.iter().sum();
    
    // Normalize to get probabilities
    exp_logits.iter().map(|&x| x / sum_exp).collect()
}

fn power_scale_color(prob: f32, max_prob: f32, power: f32) -> String {
    let normalized = prob / max_prob;
    let scaled = normalized.powf(1.0 / power); // power > 1 exaggerates differences
    
    // Use a high-contrast color scheme
    if scaled < 0.2 {
        "rgb(255, 255, 255)".to_string() // White for very low
    } else if scaled < 0.4 {
        let t = (scaled - 0.2) / 0.2;
        let intensity = (t * 128.0) as u8;
        format!("rgb({}, {}, 255)", 255 - intensity, 255 - intensity)
    } else if scaled < 0.6 {
        let t = (scaled - 0.4) / 0.2;
        let red = (t * 255.0) as u8;
        format!("rgb({}, 0, 255)", red)
    } else if scaled < 0.8 {
        let t = (scaled - 0.6) / 0.2;
        let blue = (255.0 - t * 255.0) as u8;
        format!("rgb(255, 0, {})", blue)
    } else {
        let t = (scaled - 0.8) / 0.2;
        let green = (t * 255.0) as u8;
        format!("rgb(255, {}, 0)", green)
    }
}

#[derive(Properties, Clone, PartialEq)]
pub struct Props {
    pub board: Vec<Vec<Vec<bool>>>,
    pub on_board_drop: Callback<(usize, usize, usize)>,
    pub anchors: HashSet<usize>,
    pub policy: Vec<f32>,
    pub show_policy: bool,
}

#[function_component]
pub fn BlokusBoard(props: &Props) -> Html {
    let Props {
        board,
        on_board_drop,
        anchors,
        policy,
        show_policy,
    } = props.clone();

    let dim = board[0].len();
    let policy_probs = softmax(&policy);

    let ondragover = {
        move |event: DragEvent| {
            event.prevent_default();
        }
    };

    html! {
        <div class="board">
        {for (0..dim).map(|i| {

            html! {
                <div class="board-row">
                {
                    for (0..dim).map(|j| {
                
                        // Check each player channel to see who occupies the square
                        let mut player_option: usize = 0;
                        for p in 0..4 {
                            if board[p][i][j] {
                                player_option = p + 1; // Player 1-indexed
                                break;
                            }
                        }

                        let mut square_style = match player_option {
                            1 => "square red".to_string(),
                            2 => "square blue".to_string(),
                            3 => "square green".to_string(),
                            4 => "square yellow".to_string(),
                            _ => "square empty".to_string(),
                        };
                    
                        let index = i * dim + j;
                        if anchors.contains(&index) {
                            square_style = format!("{} anchor", square_style);
                        }

                        let policy_val = policy_probs[index];
                        let color = power_scale_color(policy_val, 1.0, 3.0);

                        let ondrop = {
                            on_board_drop.reform(move |e: DragEvent| {
                                e.prevent_default();

                                let target: HtmlElement = e.target().unwrap().dyn_into().unwrap();
                                let data = e.data_transfer().expect("Data transfer should exist");
                                let id = data.get_data("piece_num").expect("Dragged piece should have an id");
                                let variant = data.get_data("variant").unwrap().parse().unwrap();
                                let clicked_square: usize = data.get_data("piece_offset").unwrap().parse().unwrap();

                                let piece: usize = id.parse().unwrap();
                                let offset: usize = target.id().parse().unwrap();
                                let offset = offset - clicked_square;
                                (piece, variant, offset)
                            })
                        };

                        html! {
                            <div>
                            if show_policy && policy_val > 0.0001 {
                                <div id={index.to_string()}  class={square_style} {ondrop} {ondragover}
                                    style={format!("background-color: {};", color)}>
                                </div>
                            } else {
                                <div id={index.to_string()}  class={square_style} {ondrop} {ondragover}></div>
                            }
                            </div>
                        }
                    })
                }
                </div>
            }
        })}
        </div>
    }
}
