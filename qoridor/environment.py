"""
Gym-like environment wrapper for Qoridor.
"""

from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
from qoridor.game import QoridorGame
from qoridor.move import Move, MoveType
from qoridor.board import WallOrientation


class QoridorEnv:
    """
    Gym-like environment for Qoridor.
    
    This class provides a reinforcement learning friendly interface to Qoridor,
    including observation/action spaces and reward function.
    """
    
    def __init__(self, board_size: int = 9, num_walls: int = 10, opponent=None):
        """
        Initialize the Qoridor environment.
        
        Args:
            board_size: Size of the board (default: 9)
            num_walls: Number of walls each player starts with (default: 10)
            opponent: Optional opponent policy/agent for player 2
        """
        self.board_size = board_size
        self.num_walls = num_walls
        self.game = QoridorGame(board_size=board_size, num_walls=num_walls)
        self.opponent = opponent
        self.player = 1  # Agent is always player 1 for simplicity
        
        # Define action space dimensions
        self.num_move_actions = board_size * board_size
        self.num_wall_actions = 2 * (board_size - 1) * (board_size - 1)  # Horizontal and vertical
        self.action_space_size = self.num_move_actions + self.num_wall_actions
        
        # Track episode info
        self.steps = 0
        self.max_steps = 1000  # Prevent infinite games
    
    def reset(self) -> Dict[str, Any]:
        """
        Reset the environment to a new game.
        
        Returns:
            Initial observation
        """
        self.game.reset()
        self.steps = 0
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Take a step in the environment by applying an action.
        
        Args:
            action: Integer action from the flattened action space
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Convert flat action to actual move
        move = self._action_to_move(action)
        
        # Apply the move
        success = self.game.make_move(move)
        
        # If the move was invalid, return negative reward
        if not success:
            return self._get_observation(), -10.0, True, {"invalid_move": True}
        
        # Check if the game is over
        done = self.game.is_game_over()
        reward = self._calculate_reward(done)
        
        # Let opponent make a move if the game isn't over
        if not done and self.opponent:
            self._make_opponent_move()
            # Check again if the game is over after opponent's move
            done = self.game.is_game_over()
            # Update reward based on new game state
            reward = self._calculate_reward(done)
        
        # Increment step counter
        self.steps += 1
        
        # Check if we've reached the maximum number of steps
        if self.steps >= self.max_steps:
            done = True
            reward = 0.0  # Draw
        
        return self._get_observation(), reward, done, {"steps": self.steps}
    
    def _action_to_move(self, action: int) -> Move:
        """
        Convert a flat action index to a Move object.
        
        Args:
            action: Integer action index
            
        Returns:
            A Move object representing the action
        """
        player = self.game.get_current_player()
        
        # Movement action
        if action < self.num_move_actions:
            row = action // self.board_size
            col = action % self.board_size
            return Move.pawn_move(player, row, col)
        
        # Wall placement action
        else:
            wall_action = action - self.num_move_actions
            is_horizontal = wall_action < (self.board_size - 1) * (self.board_size - 1)
            
            if is_horizontal:
                index = wall_action
            else:
                index = wall_action - (self.board_size - 1) * (self.board_size - 1)
            
            row = index // (self.board_size - 1)
            col = index % (self.board_size - 1)
            
            orientation = WallOrientation.HORIZONTAL if is_horizontal else WallOrientation.VERTICAL
            return Move.wall_placement(player, row, col, orientation)
    
    def _move_to_action(self, move: Move) -> int:
        """
        Convert a Move object to a flat action index.
        
        Args:
            move: The Move object
            
        Returns:
            Integer action index
        """
        row, col = move.position
        
        if move.move_type == MoveType.PAWN_MOVE:
            return row * self.board_size + col
        else:  # WALL_PLACEMENT
            is_horizontal = move.wall_orientation == WallOrientation.HORIZONTAL
            if is_horizontal:
                base = self.num_move_actions
                index = row * (self.board_size - 1) + col
            else:
                base = self.num_move_actions + (self.board_size - 1) * (self.board_size - 1)
                index = row * (self.board_size - 1) + col
            
            return base + index
    
    def _make_opponent_move(self) -> None:
        """Make a move for the opponent."""
        # If we're using a policy-based opponent
        if self.opponent:
            # Get observation from opponent's perspective
            obs = self._get_observation(player=2)
            
            # Get action from opponent policy
            opponent_action = self.opponent(obs)
            
            # Convert to move and apply
            opponent_move = self._action_to_move(opponent_action)
            self.game.make_move(opponent_move)
    
    def _calculate_reward(self, done: bool) -> float:
        """
        Calculate the reward for the current state.
        
        Args:
            done: Whether the game is over
            
        Returns:
            Reward value
        """
        if done:
            winner = self.game.get_winner()
            if winner == self.player:  # Agent won
                return 1.0
            else:  # Agent lost
                return -1.0
        
        # Optionally add intermediate rewards here
        # For example, reward for decreasing distance to goal
        return 0.0
    
    def _get_observation(self, player: int = None) -> Dict[str, Any]:
        """
        Get the current observation.
        
        Args:
            player: The player perspective (default: agent's player)
            
        Returns:
            Dictionary containing the observation
        """
        if player is None:
            player = self.player
        
        state = self.game.state
        
        # Basic observation with raw state
        obs = {
            'board': np.zeros((self.board_size, self.board_size), dtype=np.int8),
            'walls_h': state.board.horizontal_walls.copy(),
            'walls_v': state.board.vertical_walls.copy(),
            'walls_left': np.array([
                state.get_walls_left(1),
                state.get_walls_left(2)
            ]),
            'current_player': state.current_player,
        }
        
        # Set player positions
        p1_pos = state.board.get_player_position(1)
        p2_pos = state.board.get_player_position(2)
        obs['board'][p1_pos[0], p1_pos[1]] = 1
        obs['board'][p2_pos[0], p2_pos[1]] = 2
        
        return obs
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the current state of the environment.
        
        Args:
            mode: Rendering mode ('human' or 'ansi')
            
        Returns:
            String representation if mode is 'ansi', None otherwise
        """
        board_str = str(self.game)
        
        if mode == 'human':
            print(board_str)
            return None
        elif mode == 'ansi':
            return board_str
        else:
            return None
    
    def get_legal_actions(self) -> List[int]:
        """
        Get all legal actions from the current state.
        
        Returns:
            List of legal action indices
        """
        legal_moves = self.game.get_legal_moves()
        return [self._move_to_action(move) for move in legal_moves]
    
    def close(self) -> None:
        """Clean up resources."""
        pass
