from tqdm import tqdm
import torch
import random
import numpy as np

from dqn import QNetwork
from dqn import MinimumExponentialLR, ReplayBuffer, EpsilonGreedy
from qoridor.environment import QoridorEnv
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.model = QNetwork(state_size, action_size,nn_l1=512,nn_l2=512)
        self.target_model = QNetwork(state_size, action_size,nn_l1=512,nn_l2=512)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, amsgrad=True)
        self.replay_buffer=ReplayBuffer(2000)
        self.lr_scheduler = MinimumExponentialLR(self.optimizer, lr_decay=0.97, min_lr=0.0001)
        self.loss_fn = torch.nn.MSELoss()
        self.env = QoridorEnv(5,3)
        self.epsilon_greedy = EpsilonGreedy(epsilon_start=1,
                                        epsilon_min=0.001,
                                        epsilon_decay=0.999,
                                        env=self.env,
                                        q_network=self.model,
                                    )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.memory = []
        self.gamma = 0.99
        self.batch_size = 64

    def select_action(state: np.ndarray, q_network: torch.nn.Module) -> int:
        """
        Select the action with the highest Q-value for the given state.
        
        Args:
            state: The current state as a numpy array.
            q_network: The trained Q-network.
            
        Returns:
            The action index with the highest Q-value.
        """
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = q_network(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 2000:
            self.memory.pop(0)

    def train(self):
        if len(self.replay_buffer) > self.batch_size:
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = self.replay_buffer.sample(self.batch_size)

            # Convert to PyTorch tensors
            batch_states_tensor = torch.tensor(batch_states, dtype=torch.float32, device=self.device)
            batch_actions_tensor = torch.tensor(batch_actions, dtype=torch.long, device=self.device)
            batch_rewards_tensor = torch.tensor(batch_rewards, dtype=torch.float32, device=self.device)
            batch_next_states_tensor = torch.tensor(batch_next_states, dtype=torch.float32, device=self.device)
            batch_dones_tensor = torch.tensor(batch_dones, dtype=torch.bool, device=self.device)

            # Compute the target Q values for the batch
            # Compute the target Q values for the batch
            with torch.no_grad():
                next_q_values = self.target_model(batch_next_states_tensor)
                best_action_index = torch.argmax(self.model(batch_next_states_tensor), dim=1, keepdim=True)
                targets = batch_rewards_tensor + (1 - batch_dones_tensor.float()) * self.gamma * next_q_values.gather(1, best_action_index).squeeze(1)

            # Get current Q values
            current_q_values = self.model(batch_states_tensor)

            # Calculate the loss
            loss = self.loss_fn(current_q_values.gather(1, batch_actions_tensor.unsqueeze(1)).squeeze(1), targets)

            # Optimize the model
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            # Step with the learning rate scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()


