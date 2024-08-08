# Blokus Engine

![blokus_ai](https://github.com/user-attachments/assets/b0e97f83-328a-4218-b4cf-80c7819ab331)

## Project Outline

This repository contains several main sections. Most of the code is train a neural network using self-play and
Monte Carlo Tree Search. During the process you have a neural network that acts as the brains of your players,
and you have a game which is progressively getting played out. To simulate a game, all four players use the same
network to determine which move they should take. They use MCTS to explore a tree of possible moves, and the game
ultimately takes one path down this tree. Once the game is finished, the model can use all of this data it generated
to train and improve itself. Then another game can be simulated using the better model to create more data. This
is the basic cycle that allows the neural network to improve. To speed things up, you can have one neural network,
and you can have multiple games going in parralel at the same time. This means you can batch requests from multiple
games and you can generate more data at the same time. This project uses a client-server architecture where the neural
network is hosted on a server, and the clients each query the server as they simulate a game. The server code uses Python
and Pytorch, and that is all located in the model_server directory. Meanwhile each client uses Rust code in the self_play
directory. This code is built on top of the blokus directory which contains all of the core game logic. Lastly, there
is a GUI (that currently does not work yet) to play a game against the trained model.

## Training Configuration

All of the configuration for training is done in the model/training.py config class. All of the values in this config are listed below,
along with an example config file. The total number of games played during training is the number of clients times the number of games
per client times the number of training rounds. For example if you have 10 clients, each generating 2 games worth of data per training
round, and you train for 1 round, you will have 20 games worth of data. AlphaZero trained on 21 million games of Go.

| *Variable* | *Description* | *AlphaZero Value* |
| --- | --- | --- |
| PORT | The port the server listens on |  |
| SERVER_URL | The URL of the server | |
| CHECK_INTERVAL | The time clients should wait before checking whether the server is still training |  |
| BATCHING_FREQUENCY | The frequency batched requests are processed during self-play |  |
| TRAINING_ROUNDS | The number of training rounds to run | 4,200 |
| BUFFER_CAPACITY | The number of data points to store in the replay buffer | 1,000,000 games |
| LEARNING_RATE | The learning rate of the neural network | .01 -> .0001 with scheduler |
| BATCH_SIZE | The number of data points per batch | 2,048 |
| TRAINING_STEPS | The number of training steps to run each round | 700,000 |
| NN_WIDTH | The number of filters in each convolutional layer | 256 |
| NN_BLOCKS | The number of residual blocks in the neural network | 20 |
| NUM_CLIENTS | The number of clients to run | 5,000 |
| GAMES_PER_CLIENT | The number of games each client generates per round | 1 |
| SIMS_PER_MOVE | The number of simulations to run during MCTS to derive a policy | 800 |
| SAMPLE_MOVES | The number of moves in a game that sample from the MCTS policy instead of picking the max to encourage exploration | 30 |
| C_BASE | Constant for UCB formula to balance exploration and exploitation | 19,652 |
| C_INIT | Constant for UCB formula to balance exploration and exploitation | 1.25 |
| DIRICHLET_ALPHA | The alpha parameter of the Dirichlet distribution which adds noise to the root node during MCTS to promote exploration | 0.03 |
| EXPLORATION_FRAC | Fraction used to mix noise and prior probability | 0.25 |


## Usage:

### GUI

To open the GUI in the browser run the model server and the proxy server then run:
`cd gui`
`trunk serve --open`

### Training

To train on the HPC, run:

`sbatch -N 1 -n 8 train.sh`

-N is the number of computer nodes and -n is the number of CPUs


## References

- https://sebastianbodenstein.com/post/alphazero/
- https://arxiv.org/pdf/1712.01815.pdf
- https://arc.net/folder/7FE3479D-1752-401F-9DC3-49AAD02B5DF3
