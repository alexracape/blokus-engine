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

from resnet import ResNet
from training import TrainingContext, IPC, TestConfig, handle_inference_batch
from blokus_self_play import play_test_game


def main():
    """Run a model against another in multiple rounds of self-play testing

    Usage: python testing.py <num_games> <model1_path> <model2_path>
    """

    # Parse args for number of games
    test_games = int(sys.argv[1])
    first_model_path = sys.argv[2]
    second_model_path = sys.argv[3]
    dim = int(sys.argv[4])
    num_workers = 10
    config = TestConfig(dim, num_workers)

    # Load environment variables
    context = TrainingContext(config, True, first_model_path)
    context.model.eval()
    ipc = IPC(num_workers)

    baseline_context = TrainingContext(config, True, second_model_path)
    baseline_context.model.eval()
    baseline_ipc = IPC(num_workers)
    baseline_ipc.pipes_from_model = ipc.pipes_from_model
    baseline_ipc.pipes_to_workers = ipc.pipes_to_workers

    logging.info(f"Testing with {test_games} games")
    logging.info(f"Using device: {context.device}")
    
    # Generate spawn asynchronous self-play processes
    with mp.Pool(num_workers) as pool:
        game_data = pool.starmap_async(
            play_test_game,
            [(id, ipc.request_queue, baseline_ipc.request_queue, ipc.pipes_from_model[id]) for id in range(test_games)]
        )

        # Start handling inference requests
        total_requests_ish = test_games * dim * dim
        pbar = tqdm(total=total_requests_ish, desc=f"Testing Requests Round {round}")
        while not game_data.ready():
            model_requests = handle_inference_batch(config, context, ipc)
            baseline_requests = handle_inference_batch(config, baseline_context, baseline_ipc)
            pbar.update(model_requests + baseline_requests)
        pbar.close()

    # Clean up
    logging.info("Test Run Complete")
    logging.info(f"Score = {sum(game_data.get())}/{test_games}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
