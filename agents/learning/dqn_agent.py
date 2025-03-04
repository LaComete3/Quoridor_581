"""
Deep Q-Network agent for Qoridor.
"""

import os
import random
import numpy as np
from collections import deque
from typing import Dict, Any, List, Tuple, Optional
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam

from ...qoridor.game import QoridorGame
from ...qoridor.move import Move, MoveType
from ...qoridor.board import WallOrientation
from ...qoridor.rules import QoridorRules


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


class DQNAgent:
    """
    Deep Q-Network agent for Qoridor.
    
    This agent uses a deep neural network to approximate the Q-function.
    """
    
    def __init__(self, player: int = 1, board_size: int = 9, num_walls: int = 10,
                learning_rate: float = 0.001, gamma: float = 0.99,
                epsilon: float = 1.0, epsilon_decay: float = 0.995,
                epsilon_min: float = 0.01, batch_size: int = 32,
                target_update_freq: int = 100):
        """
        Initialize the DQN agent.
        
        Args:
            player: The player number (1 or 2)
            board_size: Size of the board
            num_walls: Number of walls for each player
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor
            epsilon: Exploration rate
            epsilon_decay: Rate at which epsilon decays
            epsilon_min: Minimum value of epsilon
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates
        """
        self.player = player
        self.board_size = board_size
        self.num_walls = num_walls
        
        # Hyperparameters
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Action space
        self.num_move_actions = board_size * board_size
        self.num_wall_actions = 2 * (board_size - 1) * (board_size - 1)
        self.action_space_size = self.num_move_actions + self.num_wall_actions
        
        # Build models
        self.model = self._build_model()
        self.target_model = self._build_model()
        self._update_target_network()
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        
        # Training counter
        self.train_step_counter = 0
    
    def _build_model(self) -> Model:
        """
        Build the neural network model.
        
        Returns:
            Keras Model
        """
        # Input for board state
        board_input = Input(shape=(self.board_size, self.board_size, 3))
        
        # Convolutional layers for board state
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(board_input)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = Flatten()(x)
        
        # Input for walls left
        walls_input = Input(shape=(2,))
        
        # Input for current player
        player_input = Input(shape=(1,))
        
        # Concatenate features
        concat = Concatenate()([x, walls_input, player_input])
        
        # Dense layers
        x = Dense(256, activation='relu')(concat)
        x = Dense(128, activation='relu')(x)
        
        # Output layer
        output = Dense(self.action_space_size, activation='linear')(x)
        
        # Create model
        model = Model(inputs=[board_input, walls_input, player_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.lr), loss='mse')
        
        return model
    
    def _update_target_network(self):
        """Update the target network with weights from the main network."""
        self.target_model.set_weights(self.model.get_weights())
    
    def _preprocess_state(self, observation: Dict[str, Any]) -> List[np.ndarray]:
        """
        Preprocess the observation for the neural network.
        
        Args:
            observation: The observation from the environment
            
        Returns:
            List of input tensors for the model
        """
        # Extract data
        board = observation['board']
        walls_h = observation['walls_h']
        walls_v = observation['walls_v']
        walls_left = observation['walls_left']
        current_player = observation['current_player']
        
        # Create 3-channel board representation
        # Channel 0: Player positions
        # Channel 1: Horizontal walls
        # Channel 2: Vertical walls
        board_tensor = np.zeros((self.board_size, self.board_size, 3))
        
        # Set player positions
        board_tensor[:, :, 0] = board
        
        # Set horizontal walls
        for i in range(walls_h.shape[0]):
            for j in range(walls_h.shape[1]):
                if walls_h[i, j]:
                    board_tensor[i:i+2, j:j+2, 1] = 1
        
        # Set vertical walls
        for i in range(walls_v.shape[0]):
            for j in range(walls_v.shape[1]):
                if walls_v[i, j]:
                    board_tensor[i:i+2, j:j+2, 2] = 1
        
        # Normalize walls left
        walls_left_tensor = walls_left / self.num_walls
        
        # Encode current player
        player_tensor = np.array([current_player - 1])  # 0 for player 1, 1 for player 2
        
        return [
            np.expand_dims(board_tensor, axis=0),
            np.expand_dims(walls_left_tensor, axis=0),
            np.expand_dims(player_tensor, axis=0)
        ]
    
    def get_action(self, observation: Dict[str, Any]) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            observation: The observation from the environment
            
        Returns:
            The selected action index
        """
        # Check if it's our turn
        if observation['current_player'] != self.player:
            # If it's not our turn, return a dummy action
            # This will be ignored by the environment
            return 0
        
        # Convert observation to game state
        game = self._observation_to_game(observation)
        
        # Get legal actions
        legal_moves = game.get_legal_moves()
        legal_actions = [self._move_to_action(move, self.board_size) for move in legal_moves]
        
        # Exploration
        if np.random.rand() < self.epsilon:
            return np.random.choice(legal_actions)
        
        # Exploitation
        state_input = self._preprocess_state(observation)
        q_values = self.model.predict(state_input, verbose=0)[0]
        
        # Mask illegal actions with very negative values
        masked_q_values = np.ones(self.action_space_size) * float('-inf')
        masked_q_values[legal_actions] = q_values[legal_actions]
        
        return np.argmax(masked_q_values)
    
    def update(self, state, action, next_state, reward, done):
        """
        Update the agent with new experience.
        
        Args:
            state: The previous state
            action: The action taken
            next_state: The resulting state
            reward: The reward received
            done: Whether the episode is done
        """
        # Store experience in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)
        
        # Only train if we have enough samples
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Train on a batch of experiences
        self._train_step()
        
        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self._update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def _train_step(self):
        """Perform a single training step on a batch of experiences."""
        # Sample from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Extract batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Process states
        state_inputs = [np.vstack([s[i] for s in states]) for i in range(3)]
        next_state_inputs = [np.vstack([s[i] for s in next_states]) for i in range(3)]
        
        # Get target Q values
        next_q_values = self.target_model.predict(next_state_inputs, verbose=0)
        
        # Process legal actions for next states
        targets = self.model.predict(state_inputs, verbose=0)
        
        for i in range(len(batch)):
            if dones[i]:
                targets[i, actions[i]] = rewards[i]
            else:
                # Get legal actions for next state
                game = self._observation_to_game(self._unprocess_state(next_states[i]))
                legal_moves = game.get_legal_moves()
                legal_actions = [self._move_to_action(move, self.board_size) for move in legal_moves]
                
                # Get max Q-value over legal actions
                masked_next_q = np.ones(self.action_space_size) * float('-inf')
                masked_next_q[legal_actions] = next_q_values[i, legal_actions]
                max_next_q = np.max(masked_next_q)
                
                targets[i, actions[i]] = rewards[i] + self.gamma * max_next_q
        
        # Train model
        self.model.fit(state_inputs, targets, epochs=1, verbose=0)
    
    def _observation_to_game(self, observation: Dict[str, Any]) -> QoridorGame:
        """
        Convert an observation to a QoridorGame instance.
        
        Args:
            observation: The environment observation
            
        Returns:
            A QoridorGame instance
        """
        # Create game
        game = QoridorGame(board_size=self.board_size, num_walls=self.num_walls)
        
        # Set the board state
        game.state.board.grid = observation['board'].copy()
        game.state.board.horizontal_walls = observation['walls_h'].copy()
        game.state.board.vertical_walls = observation['walls_v'].copy()
        
        # Set player positions
        player1_pos = tuple(np.argwhere(observation['board'] == 1)[0])
        player2_pos = tuple(np.argwhere(observation['board'] == 2)[0])
        game.state.board.player1_pos = player1_pos
        game.state.board.player2_pos = player2_pos
        
        # Set walls left
        game.state.player1_walls_left = observation['walls_left'][0]
        game.state.player2_walls_left = observation['walls_left'][1]
        
        # Set current player
        game.state.current_player = observation['current_player']
        
        return game
    
    def _unprocess_state(self, state_inputs: List[np.ndarray]) -> Dict[str, Any]:
        """
        Convert preprocessed state back to observation format.
        
        Args:
            state_inputs: List of input tensors for the model
            
        Returns:
            Observation dictionary
        """
        board_tensor = state_inputs[0][0]
        walls_left_tensor = state_inputs[1][0]
        player_tensor = state_inputs[2][0]
        
        # Extract board
        board = np.round(board_tensor[:, :, 0]).astype(np.int8)
        
        # Extract walls
        walls_h = np.zeros((self.board_size - 1, self.board_size - 1), dtype=bool)
        walls_v = np.zeros((self.board_size - 1, self.board_size - 1), dtype=bool)
        
        # This is a simplification - reconstructing exact wall positions is complex
        # For training purposes, this approximation should be sufficient
        
        # Convert walls left
        walls_left = np.round(walls_left_tensor * self.num_walls).astype(np.int8)
        
        # Convert player
        current_player = int(player_tensor[0]) + 1
        
        return {
            'board': board,
            'walls_h': walls_h,
            'walls_v': walls_v,
            'walls_left': walls_left,
            'current_player': current_player
        }
    
    def _move_to_action(self, move: Move, board_size: int) -> int:
        """
        Convert a Move object to a flat action index.
        
        Args:
            move: The Move object
            board_size: The size of the board
            
        Returns:
            Integer action index
        """
        row, col = move.position
        
        if move.move_type == MoveType.PAWN_MOVE:
            return row * board_size + col
        else:  # WALL_PLACEMENT
            num_move_actions = board_size * board_size
            is_horizontal = move.wall_orientation == WallOrientation.HORIZONTAL
            
            if is_horizontal:
                base = num_move_actions
                index = row * (board_size - 1) + col
            else:
                base = num_move_actions + (board_size - 1) * (board_size - 1)
                index = row * (board_size - 1) + col
            
            return base + index
    
    def save(self, filepath: str):
        """
        Save the model to a file.
        
        Args:
            filepath: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        self.model.save(filepath)
        
        # Save hyperparameters
        np.savez(
            filepath + "_params",
            player=self.player,
            board_size=self.board_size,
            num_walls=self.num_walls,
            lr=self.lr,
            gamma=self.gamma,
            epsilon=self.epsilon,
            epsilon_decay=self.epsilon_decay,
            epsilon_min=self.epsilon_min,
            batch_size=self.batch_size,
            target_update_freq=self.target_update_freq
        )
    
    @classmethod
    def load(cls, filepath: str) -> 'DQNAgent':
        """
        Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            DQNAgent instance
        """
        # Load parameters
        params = np.load(filepath + "_params.npz")
        
        # Create agent
        agent = cls(
            player=int(params['player']),
            board_size=int(params['board_size']),
            num_walls=int(params['num_walls']),
            learning_rate=float(params['lr']),
            gamma=float(params['gamma']),
            epsilon=float(params['epsilon']),
            epsilon_decay=float(params['epsilon_decay']),
            epsilon_min=float(params['epsilon_min']),
            batch_size=int(params['batch_size']),
            target_update_freq=int(params['target_update_freq'])
        )
        
        # Load model
        agent.model = load_model(filepath)
        agent.target_model = load_model(filepath)
        
        return agent