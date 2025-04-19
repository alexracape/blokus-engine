import logging
import sys
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import numpy as np
import torch

from resnet import ResNet

PORT = 8000
DIM = 20

app = FastAPI()
origins = [
    "http://127.0.0.1:8080"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet(10, 256)


class TensorData(BaseModel):
    player: int
    data: List[List[List[bool]]]

@app.post("/process_request")
async def process_tensor(request: TensorData):

    # Convert the list of numbers to a numpy array
    logging.debug(f"Received request: {request}")
    boards = torch.tensor(request.data, dtype=torch.float32).to(device)

    # Query the model
    with torch.no_grad():
        batch = boards.unsqueeze(0)
        policy, values = model(batch)

    # Format response
    result = {
        "policy": policy[0].tolist(),
        "values": values[0].tolist(),
        "status": 200,
    }
    logging.debug(f"Returning response: {result}")
    return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    path = sys.argv[1]
    model.load_state_dict(torch.load(path, weights_only=True, map_location=device))
    logging.info(f"Loaded model from {path}")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
