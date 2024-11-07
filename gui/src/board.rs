use std::collections::HashSet;

use gloo_console as console;
use wasm_bindgen::JsCast;
use web_sys::HtmlElement;

use yew::events::DragEvent;
use yew::prelude::*;
use yew::{function_component, html, Properties};

use blokus::board::BOARD_SIZE;

#[derive(Properties, Clone, PartialEq)]
pub struct Props {
    pub board: [u8; BOARD_SIZE * BOARD_SIZE],
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

    let ondragover = {
        move |event: DragEvent| {
            event.prevent_default();
        }
    };

    html! {
        <div class="board">
        {for (0..BOARD_SIZE).map(|i| {

            html! {
                <div class="board-row">
                {
                    for (0..BOARD_SIZE).map(|j| {
                        let index = i * BOARD_SIZE + j;
                        let mut square_style = match board[index] & 0b1111 {
                            1 => "square red".to_string(),
                            2 => "square blue".to_string(),
                            3 => "square green".to_string(),
                            4 => "square yellow".to_string(),
                            _ => "square empty".to_string(),
                        };

                        if anchors.contains(&index) {
                            square_style = format!("{} anchor", square_style);
                        }

                        let policy_val = policy[index];
                        // let intensity = (policy_val * 255.0) as u8;
                        // let intensity = (policy_val.powf(0.3)) * 255.0; // Apply a non-linear scaling to make small values more visible
                        let epsilon = 1.0; // Small value to avoid log(0)
                        let intensity = 10.0 * ((policy_val + epsilon).ln()).max(0.0) * 255.0;

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
                                    style={format!("background-color: {}; font-size: 6px;", color)}>
                                    <p>{format!("{} %", (policy_val * 1000.0).round()/10.0)}</p>
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
