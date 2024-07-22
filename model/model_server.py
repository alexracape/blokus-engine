import os
import logging
from concurrent.futures import ThreadPoolExecutor
from concurrent import futures
from typing import List, Tuple, Dict, Any
import threading
import json
import time

import grpc
import numpy as np
import pandas as pd
import torch
from torch.nn import Linear, ReLU, Conv2d
from torchrl.data import ReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers import Sampler
from tensordict import tensorclass
from torchsummary import summary
from dotenv import load_dotenv

import model_pb2
import model_pb2_grpc
from resnet import ResNet


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the .env file once if the script is not running in a Docker environment
if not os.environ.get("DOCKER_ENV"):
    load_dotenv()

# Function to load environment variables
def load_env_var(key, cast_type: type = str, default=None):

    value = os.getenv(key)
    if not value:
        logging.warn(f"Environment variable {key} not found, using default: {default}")
        return default

    try:
        return cast_type(value)
    except ValueError:
        logging.error(f"Error casting environment variable {key}. Using default: {default}")
        return default


# Load environment variables
PORT = load_env_var("PORT")
BUFFER_CAPACITY = load_env_var("BUFFER_CAPACITY", int, 1000)
LEARNING_RATE = load_env_var("LEARNING_RATE", float, 0.001)
BATCH_SIZE = load_env_var("BATCH_SIZE", int)
TRAINING_STEPS = load_env_var("TRAINING_STEPS", int, 10)
NUM_CLIENTS = load_env_var("NUM_CLIENTS", int, 1)
GAMES_PER_CLIENT = load_env_var("GAMES_PER_CLIENT", int, 1)
GAMES_PER_ROUND = NUM_CLIENTS * GAMES_PER_CLIENT
TRAINING_ROUNDS = load_env_var("TRAINING_ROUNDS", int)
NN_WIDTH = load_env_var("NN_WIDTH", int, 64)
NN_BLOCKS = load_env_var("NN_BLOCKS", int, 2)
DIM = 20

if None in [PORT, BUFFER_CAPACITY, LEARNING_RATE, BATCH_SIZE, TRAINING_STEPS, NUM_CLIENTS, GAMES_PER_CLIENT, TRAINING_ROUNDS]:
    logging.error("One or more critical environment variables are missing.")


class BlokusModelServicer(model_pb2_grpc.BlokusModelServicer):
    """Servicer for the Blokus model using gRPC

    The model is a CNN that takes input of size 20x20x4 + 21x4 + 4.
    This is from 4 planes for each player's pieces on the board then each
    player's remaining pieces and the player who's turn it is.
    The model outputs a policy and a value. The policy is a probability
    distribution over the possible moves and the value is the expected
    outcome of the game for each player.
    """

    def __init__(self, model_path=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(BUFFER_CAPACITY),
            batch_size=BATCH_SIZE
        )
        self.stats = pd.DataFrame(columns=["round", "loss", "value_loss", "policy_loss", "buffer_size"])
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.training_round = 0
        self.num_saves = 0

        if model_path:
            self.model = torch.load(model_path, map_location=self.device)
        else:
            self.model = ResNet(NN_BLOCKS, NN_WIDTH).to(self.device)

        # summary(self.model, [(5, 20, 20), (1, 1, 1)]) # for some reason dimension when summarizing is (2, 5, 20, 20)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.policy_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self.value_loss = torch.nn.MSELoss().to(self.device)

    def Predict(self, request, context):
        # boards = np.array(request.boards).reshape(5, DIM, DIM)
        boards = torch.tensor(request.boards, dtype=torch.float32).reshape(5, DIM, DIM).unsqueeze(0).to(self.device)
        player = request.player

        with torch.no_grad():
            policy, values = self.model(boards)
        # print(values)
        return model_pb2.Target(policy=policy[0], value=values[0])


    def Check(self, request, context):
        """Check in with the server to see if it is on the next round of training

        This is used intermitently by the client to check if it is in sync
        with the server. If the server is on the next round of training, the
        client will start the next round of self-play / data generation.
        Returns the current training round.
        """

        return model_pb2.Status(code=self.training_round)

    def Save(self, request, context):
        """Store data in the replay buffer"""

        # Allocate space for the data
        num_moves = len(request.history)
        states = torch.zeros(num_moves, 5, DIM, DIM, dtype=torch.float32)
        policies = torch.zeros(num_moves, DIM * DIM, dtype=torch.float32)
        scores = torch.tensor(request.values, dtype=torch.float32).repeat(num_moves, 1)

        # For each move from this game, update the state and policy
        for i, (move, policy) in enumerate(zip(request.history, request.policies)):
            player, tile = move.player, move.tile
            row, col = tile // DIM, tile % DIM
            states[i, player, row, col] = 1

            # Update the policy for this move
            for element in policy.probs:
                action, prob = element.action, element.prob
                policies[i, action] = prob

        data = Data(
            states = states,
            policies = policies,
            scores = scores,
            batch_size = [num_moves]
        )
        self.buffer.extend(data)
        print("Buffer size: ", len(self.buffer))

        # Save the model after every round of training
        self.num_saves += 1
        if self.num_saves == GAMES_PER_ROUND:
            # self.executor.submit(self.Train)
            self.Train()
            self.num_saves = 0

        return model_pb2.Status(code=0)


    def Train(self, training_steps=TRAINING_STEPS):
        """Train the model using the data in the replay buffer"""

        for step in range(training_steps):
            logging.info(f"Training step: {step}")

            # Get a batch of data from the replay buffer
            batch = self.buffer.sample()
            inputs = batch.get("states").to(self.device)
            policies = batch.get("policies").to(self.device)
            values = batch.get("scores").to(self.device)

            # Train the model
            self.optimizer.zero_grad()
            policy, value = self.model(inputs)
            policy_loss = self.policy_loss(policy, policies)
            value_loss = self.value_loss(value, values)
            loss = policy_loss + value_loss
            loss.backward()
            self.optimizer.step()

            # Store training statistics
            row = pd.DataFrame([{
                "round": self.training_round,
                "loss": loss.item(),
                "value_loss": value_loss.item(),
                "policy_loss": policy_loss.item(),
                "buffer_size": len(self.buffer)
            }])
            if self.stats.empty:
                self.stats = row
            else:
                self.stats = pd.concat([self.stats, row])
            self.stats.to_csv("./data/training_stats.csv")

        torch.save(self.model, f"./models/model_{self.training_round}.pt")
        self.training_round += 1

        return model_pb2.Status(code=0)


@tensorclass
class Data:
    states: torch.Tensor
    policies: torch.Tensor
    scores: torch.Tensor


class MoveSampler(Sampler):
    """Sample moves from the replay buffer

    Builds targets dynamically using the history stored
    in the replay buffer.
    """

    def sample(self, storage, batch_size) -> Tuple[torch.Tensor, Dict]:
        """Get game indices for the batch to sample from the storage"""

        total_moves = sum([len(game[0]) for game in storage])
        weights = [len(game[0]) / total_moves for game in storage]
        indices = torch.multinomial(torch.tensor(weights), batch_size, replacement=True)
        return indices, {}

    def _empty(self):
        pass

    def state_dict(self) -> Dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        return

    def dumps(self, path):
        pass

    def loads(self, path):
        pass


class BlokusBuffer(ReplayBuffer):
    """Buffer for storing game states for training the model"""

    def get_batch(self):
        """Sample buffer for training"""

        games = self.sample()
        batch = [self.build_targets(*game) for game in games]
        return batch

    def build_targets(self, history, policies, values):
        """Create training data from a game"""

        # Get random move from the game
        i = np.random.randint(len(history))

        # Get key data from the game
        state = torch.zeros(5, DIM, DIM, dtype=torch.bool)
        for move in history[:i]:
            player, tile = move.player, move.tile
            row, col = tile // DIM, tile % DIM
            state[player, row, col] = True # TODO is this oriented to the current player correctly?
            # Also doesn't update the 5th channel

        policy = torch.zeros(DIM * DIM, dtype=torch.float32)
        for action in policies[i].probs:
            tile, prob = action.action, action.prob
            policy[tile] = prob
            # Can update legal moves here

        # Data augmentation - flip board either horizontally or vertically
        flip_axes = [0, 1]
        for axis in flip_axes:
            if np.random.choice([True, False]):
                state = state.flip(axis)
                policy = policy.view(DIM, DIM).flip(axis).flatten()

        return state, policy, values


def serve():
    logging.info("Starting up server...")
    logging.debug(f"ENV VARS:\n"
                  f"PORT: {PORT}\n"
                  f"BUFFER_CAPACITY: {BUFFER_CAPACITY}\n"
                  f"LEARNING_RATE: {LEARNING_RATE}\n"
                  f"BATCH_SIZE: {BATCH_SIZE}\n"
                  f"TRAINING_STEPS: {TRAINING_STEPS}\n"
                  f"TRAINING_ROUNDS: {TRAINING_ROUNDS}\n"
                  f"NUM_CLIENTS: {NUM_CLIENTS}\n"
                  f"GAMES_PER_ROUND: {GAMES_PER_ROUND}\n"
                  f"WIDTH: {NN_WIDTH}\n"
                  f"BLOCKS: {NN_BLOCKS}\n")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=7))
    model_pb2_grpc.add_BlokusModelServicer_to_server(BlokusModelServicer(), server)
    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
