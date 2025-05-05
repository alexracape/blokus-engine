# Blokus Engine

![blokus_ai](https://github.com/user-attachments/assets/b0e97f83-328a-4218-b4cf-80c7819ab331)

Checkout the hosted GUI [here](https://alexracape.github.io/blokus-engine/)


## Project Outline

This repository contains several main sections. The game logic is contained to the blokus module. MCTS and 
game simulation logic is in self-play. The model directory contains the model architectures along with scripts
for testing and training the models. There is also a GUI, though it is a work in progress.

## Training Configuration and Experiments

All of the configuration for training is done in the model/training.py config class. For more details on hyperparamter selection and 
various experiments, check out `report.pdf`

## Usage:

### Compilation:

To build a local version of the Rust libraries, run 
`maturin develop`

Alternatively, you can install a published version from PyPI with 
`pip install blokus-engine`

### GUI

To open the GUI in the browser run the model server and the proxy server then run:
`cd gui`
`trunk serve --open`

### Training

To train locally, run:

`python model/training.py --test --workers 1 --dim 20 --load <model_path> --save <new_path>`

Without the test flag, metrics are logged and plotted useing weights and biases.


### Testing

To test the model, you can run the following command:

`python model/testing.py [num_games] [model_path] [benchmark_model_path]`


## References

- https://arc.net/folder/F335B1E1-B433-44A0-B8A5-9E682A51A238
