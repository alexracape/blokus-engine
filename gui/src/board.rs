use std::collections::HashSet;

use gloo_console as console;
use wasm_bindgen::JsCast;
use web_sys::HtmlElement;

use yew::events::DragEvent;
use yew::prelude::*;
use yew::{function_component, html, Properties};


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
                            if board[i][j][p] {
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

                        let policy_val = policy[index];
                        console::log!(policy_val);
                        let intensity = 10.0 * policy_val * 255.0;
                        let red = 255.0 - intensity;
                        let blue = 255.0 - intensity;
                        let green = 255.0;
                        let color = format!("rgb({}, {}, {})", red, green, blue);

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
                            if show_policy && policy_val > 0.0 {
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
