import torch.multiprocessing as mp

import argparse
import logging
import os
import time
from queue import Empty

import wandb
from tqdm import trange, tqdm
import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import TensorDict

from blokus_self_play import play_training_game, play_test_against_random
from resnet import ResNet
from transformer import BlokusTransformer

MODEL_PATH = "./weights"
CHANNEL = 1
TOKEN = 2

BLUE = '\033[94m'
RESET = '\033[0m'

logging.basicConfig(format=f'{BLUE}blokus:{RESET} %(message)s', level=logging.INFO)


class TrainingContext:
    """Package all of our training objects into one struct"""

    def __init__(self, model, buffer, optimizer, policy_loss, value_loss, device, testing):
        self.model = model
        self.buffer = buffer
        self.optimizer = optimizer
        self.policy_loss = policy_loss
        self.value_loss = value_loss
        self.device = device
        self.testing = testing


class IPC:
    """Package communciation objects"""

    def __init__(self, num_workers):
        self.manager = mp.Manager()
        self.request_queue = self.manager.Queue()
        self.result_queue = self.manager.Queue()
        self.pipes_from_model = []
        self.pipes_to_workers = []
        for i in range(num_workers):
            recv, send = mp.Pipe(duplex=False)
            self.pipes_from_model.append(recv)
            self.pipes_to_workers.append(send)


class Config:
    """Configuration for training the model

    AlphaZero used the following values for training:
        buffer_capacity = 1000000 games
        learning_rate = 0.01 -> 0.0001 with a scheduler
        batch_size = 2048
        training_steps = 700000
        num_workers = 5000
        sims_per_move = 800
        sample_moves = 30
        c_base = 19652
        c_init = 1.25
        dirichlet_alpha = 0.3
        exploration_fraction = 0.25
    """

    def __init__(self, dim=20, num_workers=50):
        self.dim = dim
        self.workers = num_workers
        self.games_per_worker = 2
        self.eval_games_per_worker = 2
        self.rep = TOKEN
        self.training_rounds = 200
        self.transformer = {
            "d_max": 20,
            "embed_dim": 128,
            "num_heads": 4,
            "mlp_dim": 256,
            "num_layers": 4,
            "dropout": 0.1
        }
        self.resnet = {
            "dim": 20,
            "width": 256,
            "depth": 10
        }

        self.learning_rate = 0.01
        self.weight_decay = 1e-4
        self.batch_size = 512
        self.training_steps = 100
        self.buffer_capacity = 50000

        self.sims_per_move = 100
        self.sample_moves = 30
        self.c_base = 305
        self.c_init = 1.25
        self.dirichlet_alpha = 0.03
        self.exploration_fraction = 0.25

    def to_dict(self):
        return self.__dict__

    def games_per_round(self):
        return self.workers * self.games_per_worker
    
    def est_moves_per_game(self):
        return round((self.dim * self.dim) * .7)

    def est_self_play_requests(self):
        return self.workers * self.games_per_worker * self.sims_per_move * self.est_moves_per_game()

    def est_eval_requests(self):
        return self.workers * self.games_per_worker * self.est_moves_per_game()


class TestConfig(Config):
    """Configuration with testing values to speed things up"""

    def __init__(self, dim=20, num_workers=4):
        self.dim = dim
        self.workers = num_workers
        self.games_per_worker = 4
        self.eval_games_per_worker = 4
        self.rep = TOKEN
        self.training_rounds = 2
        self.transformer = {
            "d_max": 20,
            "embed_dim": 16,
            "num_heads": 2,
            "mlp_dim": 32,
            "num_layers": 2,
            "dropout": 0.1
        }
        self.resnet = {
            "dim": 20,
            "width": 256,
            "depth": 10
        }

        self.buffer_capacity = 500000
        self.learning_rate = 0.01
        self.weight_decay = 1e-4
        self.batch_size = 64
        self.training_steps = 10

        self.sims_per_move = 50
        self.sample_moves = 30
        self.c_base = 19652
        self.c_init = 1.25
        self.dirichlet_alpha = 0.3
        self.exploration_fraction = 0.5


def empty_queue(queue):
    ids, items = [], []
    while True:
        try:
            id, data = queue.get(block=False)
            ids.append(id)
            items.append(data)
        except Empty as e:
            break

    return ids, items

def handle_inference_batch(config, context, ipc):
    """Process batches of inputs from the self-play games

    All batches are sent to the
    GPU for processing and the outputs are sent back to the appropriate worker.
    """

    time.sleep(.0001)
    ids, requests = empty_queue(ipc.request_queue)
    if config.rep == CHANNEL:
        batch = torch.tensor(requests, dtype=torch.float32).view(-1, 5, config.dim, config.gdim).to(context.device)
    else:
        batch = torch.tensor(requests, dtype=torch.float32).view(-1, config.dim * config.dim, 5).to(context.device)


    num_requests = len(ids)
    if num_requests == 0:
        return 0

    # Query the model for the batch of inputs
    with torch.no_grad():
        policies, values = context.model(batch)

    # Send the outputs to the appropriate worker
    for i, id in enumerate(ids):
        response = (policies[i].cpu().tolist(), values[i].cpu().tolist())
        ipc.pipes_to_workers[id].send(response)

    return num_requests


def handle_result_batch(ipc, data):
    ids, requests = empty_queue(ipc.result_queue)
    data.extend(requests)


def handle_inference_requests(config, context, ipc, pbar):
    results = []
    while len(results) < config.workers:
        # Handle requests
        num_requests = handle_inference_batch(config, context, ipc)
        pbar.update(num_requests)

        # Check for results
        handle_result_batch(ipc, results)

    return results


def save(game, buffer: ReplayBuffer, config: Config):
    """Save the game data to the replay buffer"""

    # Allocate space for the data
    dim = config.dim
    history, policies, values = game
    num_moves = len(history)
    logging.debug(f"Saving game with {num_moves} moves to the replay buffer")

    if config.rep == CHANNEL:
        shape = [5, dim, dim]
    else:
        shape = [dim * dim, 5]
    
    state_data = torch.zeros(num_moves, *shape, dtype=torch.float32)
    policy_data = torch.zeros(num_moves, dim * dim, dtype=torch.float32)
    value_data = torch.tensor(values, dtype=torch.float32).repeat(num_moves, 1)

    # For each move from this game, update the state and policy
    prev_player = 0
    for i, (move, policy) in enumerate(zip(history, policies)):

        # Shift the state to the current player's perspective
        player, tile = move
        player_dif = (player - prev_player) % 4
        is_token = config.rep == TOKEN
        player_dim = 1 if is_token else 0
        state_slice = state_data[i][:, :4] if is_token else state_data[i][:4]
        board = torch.roll(state_slice, shifts=-player_dif, dims=player_dim) if player_dif else state_slice
        
        # Reset the legal tiles all to 0
        blank_legals = torch.zeros(dim * dim) if is_token else torch.zeros(dim, dim)
        blank_legals = blank_legals.unsqueeze(player_dim)
        
        # Put state rep back together with board reoriented and legals reset
        state_data[i] = torch.cat((board, blank_legals), dim=player_dim)

        # Update the policy for this state
        for element in policy:
            action, prob = element
            policy_data[i, action] = prob

            # Update which squares are legal on this move
            if is_token:
                state_data[i, action, 4] = 1
            else:
                row, col = action // dim, action % dim
                state_data[i, 4, row, col] = 1


        # Rotate state and policy so perspective is the same
        # state_data[i] = torch.rot90(state_data[i], k=player, dims=(1, 2))
        # policy_data[i] = torch.rot90(policy_data[i].reshape(dim, dim), k=player).reshape(-1)

        # Update values
        if player_dif:
            value_data[i] = torch.roll(value_data[i-1], -player_dif)
        else:
            value_data[i] = value_data[i-1] # for i=0, last element equals first

        # No need to update next state on last move
        if i == num_moves - 1:
            break

        # Make the move that was made and update the next state
        state_data[i+1] = state_data[i]
        if is_token:
            state_data[i+1][tile, 0] = 1
        else:
            row, col = tile // dim, tile % dim
            state_data[i+1][0, row, col] = 1

        prev_player = player

        # if  i < 100:
        #     print(f"Player {player}")
        #     print(f"State: {state_data[i]}")
        #     print(f"Policy: {policy_data[i]}")
    # print(f"Value: {value_data}")

    data = TensorDict(
        {
            "states": state_data,
            "policies": policy_data,
            "scores": value_data
        },
        batch_size=[num_moves]
    )
    buffer.extend(data)


def train(step, context):
    """Train the model on a batch of data from the replay buffer"""

    device = context.device
    batch = context.buffer.sample()
    optimizer = context.optimizer

    # Get a batch of data from the replay buffer
    inputs = batch.get("states").to(device)
    policies = batch.get("policies").to(device)
    values = batch.get("scores").to(device)

    # Train the model
    optimizer.zero_grad()
    policy_logits, value_logits = context.model(inputs)
    # mask = inputs[:, :, :, 4].view(inputs.size(0), -1)
    # masked_policy_logits = policy_logits.masked_fill(mask, -1e9)

    policy_loss = context.policy_loss(policy_logits, policies)
    value_loss = context.value_loss(value_logits, values)
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    # Store training statistics
    if not context.testing:
        # var = torch.var(policies, dim=1).mean()
        # print(f"Policy variance in batch: {var}")
        wandb.log({"policy_loss": policy_loss, "value_loss": value_loss}, step=step)


def start_workers(config, ipc, task):

    processes = []
    for i in range(config.workers):
        p = mp.Process(target=task, args=(i, config, ipc.result_queue, ipc.request_queue, ipc.pipes_from_model[i]))
        p.start()
        processes.append(p)

    return processes


def generate_self_play_data(config, context):

    # Spawn asynchronous self-play processes
    pbar = tqdm(total=config.est_self_play_requests(), desc=f"Self-Play Requests")
    game_data = []
    for i in range(config.games_per_worker):
        ipc = IPC(config.workers)
        processes = start_workers(config, ipc, play_training_game)

        # Handling inference requests
        results = handle_inference_requests(config, context, ipc, pbar)
        game_data.extend(results)
        for p in processes:
            p.join()

    # Save the game data to the replay buffer
    pbar.close()
    for game in game_data:
        save(game, context.buffer, config)


def evaluate_against_random(config, context, step):

    # Spawn async self-play processs to test
    pbar = tqdm(total=config.est_eval_requests(), desc=f"Eval Game Requests")
    ipc = IPC(config.workers)
    game_data = []
    for i in range(config.eval_games_per_worker):
        processes = start_workers(config, ipc, play_test_against_random)

        # Handling inference requests
        results = handle_inference_requests(config, context, ipc, pbar)
        game_data.extend(results)
        for p in processes:
            p.join()

    # Save eval stats
    pbar.close()
    wins = sum(game_data)
    average = wins / (config.eval_games_per_worker * config.workers)
    logging.info(f"Score = {average}")
    if not context.testing:
        wandb.log({"win_ratio": average}, step=step)


def main():
    """Train the model

    Creates the model then spawns multiple processes to generate
    training data through self-play. The training data is then used
    to train the model. This process is repeated until the model
    reaches a certain number of training steps.
    """

    # Parse args for number of workerss and testing mode
    parser = argparse.ArgumentParser(description="Training the Blokus Deep Neural Network with Self-Play")
    parser.add_argument('--test', action='store_true', help="Run the program in testing mode")
    parser.add_argument('--dim', type=int, default=20, help="Dimension of game board (default: 20)")
    parser.add_argument('--workers', type=int, default=1, help="Number of workers to use (default: 1)")
    parser.add_argument('--load', type=str, help="Path to load starting model")
    parser.add_argument('--save', type=str, help="Path to save model to")
    args = parser.parse_args()
    logging.info(f"Using {args.workers} workerss")
    logging.info(f"Running in {'test' if args.test else 'full power'} mode")

    save_path = f"{MODEL_PATH}/{args.save}" if args.save else f"{MODEL_PATH}/latest_model.pt"
    if args.save:
        logging.info(f"Keeping track of model checkpoints @ {save_path}")

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    logging.info(f"Using device: {device}")

    if args.test:
        config = TestConfig(args.dim, args.workers)
    else:
        config = Config(args.dim, args.workers)

    # Create the model, optimizer, and loss
    if config.rep == CHANNEL:
        model = ResNet(**config.resnet)
    else:
        model = BlokusTransformer(**config.transformer)
                 
    if args.load:
        logging.info(f"Loading model from {args.load}")
        model.load_state_dict(torch.load(args.load, weights_only=True, map_location=device))
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    policy_loss = torch.nn.CrossEntropyLoss().to(device)
    value_loss = torch.nn.CrossEntropyLoss().to(device)

    # Configure Weights and Biases
    if not args.test:
        wandb.login()
        wandb.init(project="blokus", config=config.to_dict())
        wandb.watch(model, log_freq=100)

    # Set up replay buffer
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(config.buffer_capacity),
        batch_size=config.batch_size
    )

    context = TrainingContext(model, buffer, optimizer, policy_loss, value_loss, device, args.test)

    # Train the model
    global_step = 0
    for round in trange(config.training_rounds):

        generate_self_play_data(config, context)

        # Train the model
        for step in trange(config.training_steps, desc=f"Training round {round}", leave=False):
            train(global_step, context)
            global_step += 1
        torch.save(model.state_dict(), save_path)

        evaluate_against_random(config, context, global_step)

    # Clean up
    logging.info("Training complete")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
