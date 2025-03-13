import itertools
import os
import random
import numpy as np
from collections import deque
from typing import Callable, Dict, Any, List, Tuple, Optional

from tqdm import tqdm
import torch
from qoridor.environment import QoridorEnv
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int = 10000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum capacity of the buffer
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        """Add an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of experiences."""
        return random.sample(self.buffer, min(len(self.buffer), batch_size))
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)
    
class DNetwork:
    def __init__(self, board_size: int, action_space_size: int, learning_rate: float = 0.001):
        """
        Initialise le réseau de neurones.

        Args:
            board_size (int): Taille du plateau de jeu (ex: 9 pour un plateau 9x9).
            action_space_size (int): Nombre total d'actions possibles.
            learning_rate (float): Taux d'apprentissage de l'algorithme.
        """
        super(DNetwork, self).__init__()
        self.board_size = board_size
        self.action_space_size = action_space_size
        self.lr = learning_rate
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """
        Construit le modèle du réseau de neurones.

        Returns:
            Model: Modèle Keras compilé.
        """
        # Entrée pour l'état du plateau
        board_input = Input(shape=(self.board_size, self.board_size, 3), name="board_input")

        # Couches convolutionnelles pour extraire les caractéristiques du plateau
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Flatten()(x)

        # Entrée pour le nombre de murs restants (2 valeurs : un par joueur)
        walls_input = Input(shape=(2,), name="walls_input")

        # Entrée pour identifier le joueur actuel (1 valeur)
        player_input = Input(shape=(1,), name="player_input")

        # Concaténation des entrées
        concat = Concatenate(name="concatenate_layer")([x, walls_input, player_input])

        # Couches entièrement connectées
        x = Dense(256, activation='relu')(concat)
        x = Dense(128, activation='relu')(x)

        # Couche de sortie : valeurs Q pour chaque action possible
        output = Dense(self.action_space_size, activation='linear', name="output")(x)

        # Création et compilation du modèle
        model = Model(inputs=[board_input, walls_input, player_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')

        return model

    def summary(self):
        """Affiche l'architecture du modèle."""
        self.model.summary()


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
    env : gym.Env
        The environment in which the agent is acting.
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
                state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                action = torch.argmax(q_values, dim=1).item()
        return action

    def decay_epsilon(self):
        """
        Decay the epsilon value after each episode.

        The new epsilon value is the maximum of `epsilon_min` and the product of the current
        epsilon value and `epsilon_decay`.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_naive_agent(
    env: QoridorEnv,
    q_network: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    epsilon_greedy: EpsilonGreedy,
    device: torch.device,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    num_episodes: int,
    gamma: float,
) -> List[float]:
   
    episode_reward_list = []

    for episode_index in tqdm(range(1, num_episodes)):
        state = QoridorEnv.reset()
        episode_reward = 0.0

        for t in itertools.count():

            action = epsilon_greedy(state)

            next_state, reward, done, steps = env.step(action)

            episode_reward += float(reward)

            with torch.no_grad():
                next_state_tensor =torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
                target = reward + gamma* torch.max(q_network.forward(next_state_tensor)) * (1-done)
            state_tensor =torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values=q_network.forward(state_tensor)
            q_value_of_current_action=  q_values[0,action]
            loss = loss_fn(q_value_of_current_action, target.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            if done:
                break
            state = next_state
        episode_reward_list.append(episode_reward)
        epsilon_greedy.decay_epsilon()

    return episode_reward_list

import torch
from torch.optim.lr_scheduler import ExponentialLR
from typing import List

class MinimumExponentialLR(ExponentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        lr_decay: float,
        last_epoch: int = -1,
        min_lr: float = 1e-6,
    ):
        """
        Initialisation du scheduler de taux d'apprentissage exponentiel avec un minimum.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            L'optimiseur dont le taux d'apprentissage doit être programmé.
        lr_decay : float
            Le facteur multiplicatif de la décadence du taux d'apprentissage.
        last_epoch : int, optionnel
            L'index du dernier epoch. Par défaut, -1.
        min_lr : float, optionnel
            Le taux d'apprentissage minimal. Par défaut, 1e-6.
        """
        self.min_lr = min_lr
        super().__init__(optimizer, lr_decay, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Retourne les taux d'apprentissage après l'application de la décadence exponentielle.
        Garantit que le taux d'apprentissage ne tombe pas en dessous du minimum.
        
        Returns
        -------
        List[float]
            Les taux d'apprentissage après application de la décadence.
        """
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min_lr)
            for base_lr in self.base_lrs
        ]