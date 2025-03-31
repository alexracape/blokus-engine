import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import argparse
import logging
import os
import time
from queue import Empty

import wandb
from tqdm import trange, tqdm
import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import tensorclass

from blokus_self_play import play_training_game
from resnet import ResNet

MODEL_PATH = "./weights"

@tensorclass
class Data:
    states: torch.Tensor
    policies: torch.Tensor
    scores: torch.Tensor


def empty_queue(queue, device, dim):
    ids = []
    items = []
    while True:
        try:
            id, input = queue.get(block=False)
            ids.append(id)
            items.append(input)
        except Empty as e:
            break

    return ids, torch.tensor(items, dtype=torch.float32).view(-1, 5, dim, dim).to(device)


def handle_inference_batch(model, dim, device, inference_queue, pipes_to_workers):
    """Process batches of inputs from the self-play games

    Tries to create a batch of size num_workers // 2 from the inference queue.
    If this runs for too long, there are likely stragglers in the queue and we
    should just empty the queue with what is left. All batches are sent to the
    GPU for processing and the outputs are sent back to the appropriate worker.
    """

    time.sleep(.001)
    ids, batch = empty_queue(inference_queue, device, dim)
    num_requests = len(ids)
    if num_requests == 0:
        return 0

    # Query the model for the batch of inputs
    with torch.no_grad():
        policies, values = model(batch)

    # Send the outputs to the appropriate worker
    for i, id in enumerate(ids):
        response = (policies[i].cpu().tolist(), values[i].cpu().tolist())
        pipes_to_workers[id].send(response)

    return num_requests


def save(game, buffer: ReplayBuffer, dim=20):
    """Save the game data to the replay buffer"""

    # Allocate space for the data
    history, policies, values = game
    num_moves = len(history)
    logging.debug(f"Saving game with {num_moves} moves to the replay buffer")

    state_data = torch.zeros(num_moves, 5, dim, dim, dtype=torch.float32)
    policy_data = torch.zeros(num_moves, dim * dim, dtype=torch.float32)
    value_data = torch.tensor(values, dtype=torch.float32).repeat(num_moves, 1)

    # For each move from this game, update the state and policy
    # new_state holds running game state
    new_state = torch.zeros(dim, dim, 5, dtype=torch.float32)
    for i, (move, policy) in enumerate(zip(history, policies)):

        # Shift the state to the correct player's perspective
        player, tile = move
        state_data[i] = torch.cat((new_state[player:4], new_state[:player], new_state[4].unsqueeze(0)), dim=0)

        # Update the policy for this move
        for element in policy:
            action, prob = element
            policy_data[i, action] = prob

            # Update which squares are legal on this move
            row, col = action // dim, action % dim
            state_data[i, 4, row, col] = 1

        # Rotate state and policy so perspective is the same
        # state_data[i] = torch.rot90(state_data[i], k=player, dims=(1, 2))
        # policy_data[i] = torch.rot90(policy_data[i].reshape(dim, dim), k=player).reshape(-1)

        # Make the move that was made
        row, col = tile // dim, tile % dim
        new_state[player, row, col] = 1

        # if  i < 20:
        #     print(f"Player {player}")
        #     print(f"State: {state_data[i]}")
        #     print(f"Policy: {policy_data[i]}")
    print(f"Value: {value_data}")

    data = Data(
        states = state_data,
        policies = policy_data,
        scores = value_data,
        batch_size = [num_moves]
    )
    buffer.extend(data)


def train(step, model, buffer, optimizer, policy_loss, value_loss, device, testing):
    """Train the model on a batch of data from the replay buffer"""

    # Get a batch of data from the replay buffer
    batch = buffer.sample()
    inputs = batch.get("states").to(device)
    policies = batch.get("policies").to(device)
    values = batch.get("scores").to(device)

    # Train the model
    optimizer.zero_grad()
    policy_logits, value_logits = model(inputs, training=True) # [b, d, d]. [b, 4]
    mask = inputs[:, :, :, 4].view(inputs.size(0), -1)
    masked_policy_logits = policy_logits.masked_fill(mask, -1e9)

    policy_loss = policy_loss(masked_policy_logits, policies)
    value_loss = value_loss(value_logits, values)
    loss = policy_loss + value_loss
    loss.backward()
    optimizer.step()

    # Store training statistics
    if not testing:
        # var = torch.var(policies, dim=1).mean()
        # print(f"Policy variance in batch: {var}")
        wandb.log({"policy_loss": policy_loss, "value_loss": value_loss}, step=step)


def main():
    """Train the model

    Creates the model then spawns multiple processes to generate
    training data through self-play. The training data is then used
    to train the model. This process is repeated until the model
    reaches a certain number of training steps.
    """

    # Parse args for number of CPUs and testing mode
    parser = argparse.ArgumentParser(description="Training the Blokus Deep Neural Network with Self-Play")
    parser.add_argument('--test', action='store_true', help="Run the program in testing mode")
    parser.add_argument('--dim', type=int, default=20, help="Dimension of game board (default: 20)")
    parser.add_argument('--cpus', type=int, default=1, help="Number of CPUs to use (default: 1)")
    parser.add_argument('--load', type=str, help="Path to load starting model")
    parser.add_argument('--save', type=str, help="Path to save model to")
    args = parser.parse_args()
    logging.info(f"Using {args.cpus} CPUs")
    logging.info(f"Running in {'test' if args.test else 'full power'} mode")

    save_path = f"{MODEL_PATH}/{args.save}" if args.save else f"{MODEL_PATH}/latest_model.pt"
    if args.load: logging.info(f"Loading model from {args.load}")
    if args.save:
        logging.info(f"Keeping track of model checkpoints @ {save_path}")

    # Load environment variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    if args.test:
        config = TestConfig(args.dim, args.cpus)
    else:
        config = Config(args.dim, args.cpus)


    # Create the model, optimizer, and loss
    model = ResNet(config.nn_depth, config.nn_width)
    if args.load:
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

    # Train the model
    global_step = 0
    for round in trange(config.training_rounds):

        # Create the queues and pipes
        manager = mp.Manager()
        request_queue = manager.Queue(maxsize=config.cpus * config.games_per_cpu)
        pipes_to_model = []
        pipes_to_workers = []
        for i in range(config.games_per_round()):
            a, b = mp.Pipe()
            pipes_to_model.append(a)
            pipes_to_workers.append(b)

        # Generate spawn asynchronous self-play processes
        with mp.get_context("spawn").Pool(config.cpus) as pool:
            game_data = pool.starmap_async(
                play_training_game,
                [(id, config, request_queue, pipes_to_model[id]) for id in range(config.games_per_round())]
            )

            # Start handling inference requests
            total_requests_ish = config.requests_per_round()
            pbar = tqdm(total=total_requests_ish, desc=f"Self-Play Requests Round {round}")
            while not game_data.ready():
                num_requests = handle_inference_batch(model, config.dim, device, request_queue, pipes_to_workers)
                pbar.update(num_requests)
            pbar.close()

            # Save the game data to the replay buffer
            for game in game_data.get():
                save(game, buffer)

        # Train the model
        for step in trange(config.training_steps, desc=f"Training round {round}", leave=False):
            train(global_step, model, buffer, optimizer, policy_loss, value_loss, device, args.test)
            global_step += 1
        torch.save(model.state_dict(), save_path)

    # Clean up
    logging.info("Training complete")



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

    def __init__(self, dim=20, num_cpus=1):
        self.dim = dim
        self.training_rounds = 10

        self.buffer_capacity = 500000
        self.learning_rate = 0.01
        self.weight_decay = 1e-4
        self.batch_size = 256
        self.training_steps = 500
        self.cpus = num_cpus
        self.games_per_cpu = 4

        self.nn_width = 256
        self.nn_depth = 10

        self.sims_per_move = 50
        self.sample_moves = 30
        self.c_base = 19652
        self.c_init = 1.25
        self.dirichlet_alpha = 0.3
        self.exploration_fraction = 0.25

    def to_dict(self):
        return self.__dict__

    def games_per_round(self):
        return self.cpus * self.games_per_cpu

    def requests_per_round(self):
        return self.games_per_round() *  self.dim**2 * (self.sims_per_move + 2)


class TestConfig(Config):
    """Configuration with testing values to speed things up"""

    def __init__(self, dim=20, num_cpus=1):
        self.dim = dim
        self.training_rounds = 2

        self.buffer_capacity = 500000
        self.learning_rate = 0.01
        self.weight_decay = 1e-4
        self.batch_size = 64
        self.training_steps = 10
        self.cpus = num_cpus
        self.games_per_cpu = 4

        self.nn_width = 256
        self.nn_depth = 10

        self.sims_per_move = 10
        self.sample_moves = 30
        self.c_base = 19652
        self.c_init = 1.25
        self.dirichlet_alpha = 0.3
        self.exploration_fraction = 0.5


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
