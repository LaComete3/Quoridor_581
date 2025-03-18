#sys.path.append(os.path.abspath(os.path.join(os.path.dirname('../qoridor'))))
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import random # epsilon greedy
import collections # relay buffer
import itertools
import sys
import os

from tqdm.notebook import tqdm # for progression bar
from typing import Callable, cast, List, Tuple, Union
from qoridor.game import QoridorGame
from qoridor.move import Move, MoveType
from qoridor.board import WallOrientation
from qoridor.visualization import QoridorVisualizer
from agents.random_agent import RandomAgent
from qoridor.environment import QoridorEnv

game = QoridorGame(board_size=5, num_walls=3)
visualizer = QoridorVisualizer()
game.state.get_observation()


def state_to_vector(state):
    """Converts the state to a vector representation.

    Args:
        state (_type_): _description_
    """
    player_1_pos = state.get_observation()['player_positions'][1] # tuple (x,y)
    player_2_pos = state.get_observation()['player_positions'][2] # tuple (x,y)
    walls_1 = state.get_observation()['walls_left'][1]  #int
    walls_2 = state.get_observation()['walls_left'][2] #int
    current_player = state.get_observation()['current_player'] # 1 ou 2 
    horizontal_walls = state.get_observation()['horizontal_walls'].astype(int) # matrice (N-1)*(N-1) avec bit de présence
    vertical_walls = state.get_observation()['vertical_walls'].astype(int) # matrice (N-1)*(N-1) avec bit de présence

    state_vector = np.array([player_1_pos[0],player_1_pos[1],player_2_pos[0],player_2_pos[1],walls_1,walls_2,current_player])

    horizontal_walls_vector = horizontal_walls.flatten()
    vertical_walls_vector = vertical_walls.flatten()

    full_state_vector = np.concatenate((state_vector,horizontal_walls_vector,vertical_walls_vector))

    return full_state_vector


class QNetwork(torch.nn.Module):
    def __init__(self, n_observations: int, n_actions: int, nn_l1: int, nn_l2: int):
        super(QNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(n_observations, nn_l1)
        self.bn1 = torch.nn.BatchNorm1d(nn_l1)
        self.layer2 = torch.nn.Linear(nn_l1, nn_l2)
        self.bn2 = torch.nn.BatchNorm1d(nn_l2)
        self.layer3 = torch.nn.Linear(nn_l2, n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si le batch size est > 1, on applique BatchNorm
        if x.size(0) > 1:  
            x = torch.relu(self.bn1(self.layer1(x)))  # Appliquer BatchNorm après la couche linéaire
            x = torch.relu(self.bn2(self.layer2(x)))
        else:
            x = torch.relu(self.layer1(x))  # Pas de BatchNorm si batch size = 1
            x = torch.relu(self.layer2(x))

        output_tensor = self.layer3(x)  # Pas de BatchNorm ici, car c'est la dernière couche
        return output_tensor


    

class EpsilonGreedy:
        """
        An Epsilon-Greedy policy.

        Attributes
        ----------
        epsilon : float
            The initial probability of choosing a random action.
        epsilon_min : float
            The minimum probability of choosing a random action.
        epsilon_decay : float
            The decay rate for the epsilon value after each action.
        env : ????
        q_network : torch.nn.Module
            The Q-Network used to estimate action values.

        Methods
        -------
        __call__(state: np.ndarray) -> np.int64
            Select an action for the given state using the epsilon-greedy policy.
        decay_epsilon()
            Decay the epsilon value after each action.
        """

        def __init__(
            self,
            epsilon_start: float,
            epsilon_min: float,
            epsilon_decay: float,
            env: QoridorEnv,
            q_network: torch.nn.Module,
        ):
        
            self.epsilon = epsilon_start
            self.epsilon_min = epsilon_min
            self.epsilon_decay = epsilon_decay
            self.env = env
            self.q_network = q_network

        def __call__(self, state: np.ndarray) -> np.int64:
            if random.random() < self.epsilon:

                action = self.env.get_legal_actions()
                action = random.choice(action)
            else:
                with torch.no_grad():
                    action = self.env.get_legal_actions()
                    state_tensor = torch.tensor(state_to_vector(state), dtype=torch.float32).unsqueeze(0)
                    q_values = self.q_network(state_tensor)
                    q_values = [q_values[0][i].item() if i in action else -np.inf for i in range(len(q_values[0]))] 
                    action = torch.argmax(torch.tensor(q_values)).item()

            return action

        def decay_epsilon(self):
            """
            Decay the epsilon value after each episode.

            The new epsilon value is the maximum of `epsilon_min` and the product of the current
            epsilon value and `epsilon_decay`.
            """
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

class ReplayBuffer:
        """
        A Replay Buffer.

        Attributes
        ----------
        buffer : collections.deque
            A double-ended queue where the transitions are stored.

        Methods
        -------
        add(state: np.ndarray, action: np.int64, reward: float, next_state: np.ndarray, done: bool)
            Add a new transition to the buffer.
        sample(batch_size: int) -> Tuple[np.ndarray, float, float, np.ndarray, bool]
            Sample a batch of transitions from the buffer.
        __len__()
            Return the current size of the buffer.
        """

        def __init__(self, capacity: int):
            """
            Initializes a ReplayBuffer instance.

            Parameters
            ----------
            capacity : int
                The maximum number of transitions that can be stored in the buffer.
            """
            self.buffer: collections.deque = collections.deque(maxlen=capacity)

        def add(
            self,
            state: np.ndarray,
            action: np.int64,
            reward: float,
            next_state: np.ndarray,
            done: bool,
        ):
            """
            Add a new transition to the buffer.

            Parameters
            ----------
            state : np.ndarray
                The state vector of the added transition.
            action : np.int64
                The action of the added transition.
            reward : float
                The reward of the added transition.
                next_state : np.ndarray
                The next state vector of the added transition.
            done : bool
                The final state of the added transition.
            """
            self.buffer.append((state, action, reward, next_state, done))

        def sample(
            self, batch_size: int
        ) -> Tuple[np.ndarray, Tuple[int], Tuple[float], np.ndarray, Tuple[bool]]:
            """
            Sample a batch of transitions from the buffer.

            Parameters
            ----------
            batch_size : int
                The number of transitions to sample.

            Returns
            -------
            Tuple[np.ndarray, float, float, np.ndarray, bool]
                A batch of `batch_size` transitions.
            """
            # Here, `random.sample(self.buffer, batch_size)`
            # returns a list of tuples `(state, action, reward, next_state, done)`
            # where:
            # - `state`  and `next_state` are numpy arrays
            # - `action` and `reward` are floats
            # - `done` is a boolean
            #
            # `states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))`
            # generates 5 tuples `state`, `action`, `reward`, `next_state` and `done`, each having `batch_size` elements.
            states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
            return np.array(states), actions, rewards, np.array(next_states), dones

        def __len__(self):
            """
            Return the current size of the buffer.

            Returns
            -------
            int
                The current size of the buffer.
            """
            return len(self.buffer)
    
class MinimumExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
        def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            lr_decay: float,
            last_epoch: int = -1,
            min_lr: float = 1e-6,
        ):
            """
            Initialize a new instance of MinimumExponentialLR.

            Parameters
            ----------
            optimizer : torch.optim.Optimizer
                The optimizer whose learning rate should be scheduled.
            lr_decay : float
                The multiplicative factor of learning rate decay.
            last_epoch : int, optional
                The index of the last epoch. Default is -1.
            min_lr : float, optional
                The minimum learning rate. Default is 1e-6.
            """
            self.min_lr = min_lr
            super().__init__(optimizer, lr_decay, last_epoch=-1)

        def get_lr(self) -> List[float]:
            """
            Compute learning rate using chainable form of the scheduler.

            Returns
            -------
            List[float]
                The learning rates of each parameter group.
            """
            return [
                max(base_lr * self.gamma**self.last_epoch, self.min_lr)
                for base_lr in self.base_lrs
            ]
