"""Testing a trained model against another"""
from multiprocessing import set_start_method
set_start_method('spawn', force=True)


import sys
import multiprocessing as mp

import logging
import os
import time
from queue import Empty

from tqdm import trange, tqdm
import torch
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict import tensorclass

from resnet import ResNet
from training import TestConfig, handle_inference_batch
from blokus_self_play import play_test_game

DIM = 20


def main():
    """Run a model against another in multiple rounds of self-play testing

    Usage: python testing.py <num_games> <model1_path> <model2_path>
    """

    # Parse args for number of games
    test_games = int(sys.argv[1])
    logging.info(f"Testing with {test_games} games")
    config = TestConfig(num_cpus=8)

    # Load environment variables
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Load models and set up loss and optimizer
    first_model_path = sys.argv[2]
    model = ResNet(10, 256)
    model.load_state_dict(torch.load(first_model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    second_model_path = sys.argv[3]
    baseline = ResNet(2, 16)
    baseline.load_state_dict(torch.load(second_model_path, weights_only=True, map_location=device))
    baseline.to(device)
    baseline.eval()

    # Create the queues and pipes
    manager = mp.Manager()
    model_queue = manager.Queue(maxsize=test_games)
    baseline_queue = manager.Queue(maxsize=test_games)
    pipes_to_model = []
    pipes_to_workers = []
    for i in range(test_games):
        a, b = mp.Pipe()
        pipes_to_model.append(a)
        pipes_to_workers.append(b)

    # Generate spawn asynchronous self-play processes
    with mp.Pool(config.cpus) as pool:
        game_data = pool.starmap_async(
            play_test_game,
            [(id, model_queue, baseline_queue, pipes_to_model[id]) for id in range(test_games)]
        )

        # Start handling inference requests
        total_requests_ish = test_games * DIM * DIM
        pbar = tqdm(total=total_requests_ish, desc=f"Testing Requests Round {round}")
        while not game_data.ready():
            model_requests = handle_inference_batch(model, device, model_queue, pipes_to_workers)
            baseline_requests = handle_inference_batch(baseline, device, baseline_queue, pipes_to_workers)
            pbar.update(model_requests + baseline_requests)
        pbar.close()

    # Clean up
    logging.info("Test Run Complete")
    logging.info(f"Score = {sum(game_data.get())}/{test_games}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
