"""
State representation for Qoridor game.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from qoridor.board import Board, WallOrientation


class GameState:
    """
    Represents the state of a Qoridor game.
    Includes the board, player information, and game status.
    """
    
    def __init__(self, board_size: int = 9, num_walls: int = 10):
        """
        Initialize a new game state.
        
        Args:
            board_size: Size of the board (default: 9)
            num_walls: Number of walls each player starts with (default: 10)
        """
        self.board = Board(size=board_size)
        self.board_size = board_size
        self.initial_num_walls = num_walls
        
        # Track walls for each player
        self.player1_walls_left = num_walls
        self.player2_walls_left = num_walls
        
        # Current player (1 or 2)
        self.current_player = 1
        
        # Game status
        self.done = False
        self.winner = None
        
        # Action history
        self.action_history = []
    
    def reset(self) -> None:
        """Reset the game state to its initial configuration."""
        self.board.reset()
        self.player1_walls_left = self.initial_num_walls
        self.player2_walls_left = self.initial_num_walls
        self.current_player = 1
        self.done = False
        self.winner = None
        self.action_history = []
    
    def get_legal_actions(self) -> List[Dict[str, Any]]:
        """
        Get all legal actions for the current player.
        
        Returns:
            A list of legal action dictionaries
        """
        actions = []
        
        # Movement actions
        for row, col in self.board.get_legal_moves(self.current_player):
            actions.append({
                'type': 'move',
                'position': (row, col)
            })
        
        # Wall placement actions
        if self.get_walls_left(self.current_player) > 0:
            # Horizontal walls
            for row, col in self.board.get_valid_wall_placements(WallOrientation.HORIZONTAL):
                # Temporarily place the wall to check if it blocks paths
                self.board.horizontal_walls[row, col] = True
                
                # Check if both players still have a path to their goal
                if self.board.has_path_to_goal(1) and self.board.has_path_to_goal(2):
                    actions.append({
                        'type': 'wall',
                        'position': (row, col),
                        'orientation': WallOrientation.HORIZONTAL
                    })
                
                # Remove the temporary wall
                self.board.horizontal_walls[row, col] = False
            
            # Vertical walls
            for row, col in self.board.get_valid_wall_placements(WallOrientation.VERTICAL):
                # Temporarily place the wall to check if it blocks paths
                self.board.vertical_walls[row, col] = True
                
                # Check if both players still have a path to their goal
                if self.board.has_path_to_goal(1) and self.board.has_path_to_goal(2):
                    actions.append({
                        'type': 'wall',
                        'position': (row, col),
                        'orientation': WallOrientation.VERTICAL
                    })
                
                # Remove the temporary wall
                self.board.vertical_walls[row, col] = False
        
        return actions
    
    def apply_action(self, action: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
        """
        Apply an action to the game state.
        
        Args:
            action: Action dictionary with 'type' and other necessary info
            
        Returns:
            Tuple of (success, winner)
            - success: Whether the action was successfully applied
            - winner: The winner if the game is over, None otherwise
        """
        if self.done:
            return False, self.winner
        
        if action['type'] == 'move':
            success = self._handle_move_action(action)
        elif action['type'] == 'wall':
            success = self._handle_wall_action(action)
        else:
            raise ValueError(f"Unknown action type: {action['type']}")
        
        if success:
            self.action_history.append(action)
            
            # Check for win condition
            self._check_win_condition()
            
            # Switch players if the game isn't over
            if not self.done:
                self.current_player = 3 - self.current_player  # Alternates between 1 and 2
        
        return success, self.winner
    
    def _handle_move_action(self, action: Dict[str, Any]) -> bool:
        """
        Handle a move action.
        
        Args:
            action: Action dictionary with 'position' key
            
        Returns:
            Whether the action was successfully applied
        """
        row, col = action['position']
        return self.board.move_player(self.current_player, (row, col))
    
    def _handle_wall_action(self, action: Dict[str, Any]) -> bool:
        """
        Handle a wall placement action.
        
        Args:
            action: Action dictionary with 'position' and 'orientation' keys
            
        Returns:
            Whether the action was successfully applied
        """
        row, col = action['position']
        orientation = action['orientation']
        
        # Check if the player has walls left
        if self.get_walls_left(self.current_player) <= 0:
            return False
        
        # Place the wall
        if self.board.place_wall(row, col, orientation):
            # Decrement wall count
            if self.current_player == 1:
                self.player1_walls_left -= 1
            else:
                self.player2_walls_left -= 1
            return True
        
        return False
    
    def _check_win_condition(self) -> None:
        """Check if the game has ended and update the state accordingly."""
        # Player 1 wins by reaching the bottom row
        player1_row, _ = self.board.get_player_position(1)
        if player1_row == self.board_size - 1:
            self.done = True
            self.winner = 1
            return
        
        # Player 2 wins by reaching the top row
        player2_row, _ = self.board.get_player_position(2)
        if player2_row == 0:
            self.done = True
            self.winner = 2
            return
    
    def get_walls_left(self, player: int) -> int:
        """
        Get the number of walls left for a player.
        
        Args:
            player: The player number (1 or 2)
            
        Returns:
            Number of walls left for the player
        """
        if player == 1:
            return self.player1_walls_left
        else:
            return self.player2_walls_left
    
    def get_observation(self) -> Dict[str, Any]:
        """
        Get an observation of the game state for RL agents.
        
        Returns:
            A dictionary containing the observation
        """
        obs = {
            'board_size': self.board_size,
            'player_positions': {
                1: self.board.get_player_position(1),
                2: self.board.get_player_position(2)
            },
            'walls_left': {
                1: self.player1_walls_left,
                2: self.player2_walls_left
            },
            'horizontal_walls': self.board.horizontal_walls.copy(),
            'vertical_walls': self.board.vertical_walls.copy(),
            'current_player': self.current_player,
            'done': self.done,
            'winner': self.winner
        }
        return obs
    
    def get_serialized_state(self) -> Dict[str, Any]:
        """
        Get a serialized representation of the state (for saving/loading).
        
        Returns:
            A dictionary containing the serialized state
        """
        return {
            'board_size': self.board_size,
            'player1_pos': self.board.get_player_position(1),
            'player2_pos': self.board.get_player_position(2),
            'player1_walls_left': self.player1_walls_left,
            'player2_walls_left': self.player2_walls_left,
            'horizontal_walls': self.board.horizontal_walls.tolist(),
            'vertical_walls': self.board.vertical_walls.tolist(),
            'current_player': self.current_player,
            'done': self.done,
            'winner': self.winner,
            'action_history': self.action_history
        }
    
    @classmethod
    def from_serialized(cls, data: Dict[str, Any]) -> 'GameState':
        """
        Create a GameState from serialized data.
        
        Args:
            data: Serialized state data
            
        Returns:
            A new GameState instance
        """
        state = cls(
            board_size=data['board_size'],
            num_walls=max(data['player1_walls_left'], data['player2_walls_left'])
        )
        
        # Reset the board (clear default positions)
        state.board.grid.fill(0)
        
        # Set player positions
        p1_row, p1_col = data['player1_pos']
        p2_row, p2_col = data['player2_pos']
        state.board.player1_pos = (p1_row, p1_col)
        state.board.player2_pos = (p2_row, p2_col)
        state.board.grid[p1_row, p1_col] = 1
        state.board.grid[p2_row, p2_col] = 2
        
        # Set walls
        state.board.horizontal_walls = np.array(data['horizontal_walls'], dtype=bool)
        state.board.vertical_walls = np.array(data['vertical_walls'], dtype=bool)
        
        # Set player data
        state.player1_walls_left = data['player1_walls_left']
        state.player2_walls_left = data['player2_walls_left']
        
        # Set game state
        state.current_player = data['current_player']
        state.done = data['done']
        state.winner = data['winner']
        
        # Set action history
        state.action_history = data['action_history']
        
        return state
    
    def __str__(self) -> str:
        """String representation of the game state."""
        s = "Qoridor Game State:\n"
        s += f"Board Size: {self.board_size}x{self.board_size}\n"
        s += f"Current Player: {self.current_player}\n"
        s += f"Player 1 Walls Left: {self.player1_walls_left}\n"
        s += f"Player 2 Walls Left: {self.player2_walls_left}\n"
        if self.done:
            s += f"Game Over. Winner: Player {self.winner}\n"
        s += "\n"
        s += str(self.board)
        return s