# Blokus Engine

Usage:

To open the GUI in the browser run:
`trunk serve --open`

To run server:
`python src/model/model_server.py`

To run simulation client:
`cargo run --bin client`


On Tap:
- Undo
- Benchmark board performance
- Simulation for self-play
    - MCTS
    - Communicating with python server
    - Python model that at least has input output dimension right
    - Logic for game terminal states
- Something is up with trunk server now, lots of compilation errors - maybe name space

References
- https://sebastianbodenstein.com/post/alphazero/
- https://arxiv.org/pdf/1712.01815.pdf