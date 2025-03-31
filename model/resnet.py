import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = torch.relu(y)
        y = self.conv2(y)
        y = self.bn2(y)
        y += x

        return torch.relu(y)


class ResNet(nn.Module):
    """ML model that will predict policy and value for game states

    AlphaZero uses a ResNet architecture for the model with 20 residual
    blocks and 256 filters per block.

    This model uses a resnet backbone with a policy head and a value head.
    The input to the model is a 20x20x5 tensor where the first 4 channels
    are binary boards for each player's pieces on the board. The 5th channel
    is a binary board with the valid moves for the current player.
    The policy head outputs a probability distribution over the valid moves
    or the 20x20 spaces. The value head outputs a single value for the expected
    outcome of the game for each player. The value is between 0 and 1.
    """

    def __init__(self, blocks, width, game_dim=20):
        super(ResNet, self).__init__()
        self.dim = game_dim
        self.max_dim = 20
        self.blocks = blocks
        self.width = width
        self.piece_filters = []

        self.input = nn.Conv2d(5, width, kernel_size=3, padding=1)
        self.res_blocks = nn.ModuleList([ResidualBlock(width, width) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(width, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(width, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.max_dim * self.max_dim, 4),
            nn.Tanh(),
        )


    def forward(self, boards, training=False):
        """Get the policy and value for the given board state

        For now, the board is represented by a 20x20x5 tensor where the first 4 channels are
        binary boards for each player's pieces on the board. The 5th channel is a binary board
        with the valid moves for the current player. For now, I'm just going to use the boards.
        It is unclear why the player color is needed in the state.
        """

        # ResNet backbone
        x = self.input(boards)
        for block in self.res_blocks:
            x = block(x)

        # Policy head - mask out illegal moves so they are 0 in policy
        policy = self.policy_head(x)
        mask = boards[:, 4, :, :].view(boards.size(0), -1)
        policy_masked = policy + (1 - mask) * -1e9
        if training:
            policy = policy_masked  # Cross Entropy Loss handles softmax
        else:
            policy = torch.softmax(policy_masked, dim=1)

        # Value head
        value = self.value_head(x)
        value = torch.softmax(value, dim=1)

        return policy, value
