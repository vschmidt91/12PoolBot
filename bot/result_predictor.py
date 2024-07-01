import torch.nn
import torch.nn.functional as f
import torch.optim

from .data import GameReplay, GameState


class QFunction(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        self.fc3 = torch.nn.Linear(16, 4)
        self.fc4 = torch.nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ResultPredictor:
    def __init__(self, input_size: int):
        self.q_function = QFunction(input_size=input_size)
        self.optimizer = torch.optim.NAdam(self.q_function.parameters(), weight_decay=1e-3)
        self.loss = torch.nn.BCEWithLogitsLoss()

    def predict(self, state: GameState) -> float:
        state_tensor = state.to_tensor()
        q_logit = self.q_function(state_tensor)
        q = f.sigmoid(q_logit)
        return q.item()

    def train(self, game: GameReplay) -> None:
        states_tensor = torch.stack([s.to_tensor() for s in game.states])
        result_tensor = game.result.to_tensor()
        result_tensor_repeated = result_tensor.unsqueeze(0).repeat(states_tensor.shape[0], 1)

        self.optimizer.zero_grad()
        prediction = self.q_function(states_tensor)
        loss = self.loss(prediction, result_tensor_repeated)
        loss.backward()
        self.optimizer.step()
