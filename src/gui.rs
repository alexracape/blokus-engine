pub mod app;
pub mod board;
pub mod pieces;

use crate::gui::app::App;


fn main() {

    // Run the GUI, will be called by trunk serve
    yew::Renderer::<App>::new().render();

}