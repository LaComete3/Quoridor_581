"""
Template for implementing your own Qoridor agent.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ...qoridor.game import QoridorGame
from ...qoridor.move import Move, MoveType
from ...qoridor.board import WallOrientation


class CustomAgent:
    """
    Template for a custom Qoridor agent.
    
    Implement your own learning agent by extending this class and implementing
    the required methods.
    """
    
    def __init__(self, player: int = 1):
        """
        Initialize your agent.
        
        Args:
            player: The player number (1 or 2) this agent controls
        """
        self.player = player
        # Add your initialization code here
        
    def get_action(self, observation: Dict[str, Any]) -> int:
        """
        Select an action based on the current observation.
        
        Args:
            observation: The current observation from the environment
                - 'board': 2D numpy array with player positions
                - 'walls_h': 2D boolean array of horizontal walls
                - 'walls_v': 2D boolean array of vertical walls
                - 'walls_left': Array with number of walls left for each player
                - 'current_player': The current player's turn (1 or 2)
            
        Returns:
            An integer action index
        """
        # Extract information from observation
        board = observation['board']
        walls_h = observation['walls_h']
        walls_v = observation['walls_v']
        walls_left = observation['walls_left']
        current_player = observation['current_player']
        
        # Get board size
        board_size = board.shape[0]
        
        # Convert observation to game state
        game = self._observation_to_game(observation)
        
        # Get legal moves
        legal_moves = game.get_legal_moves()
        
        # Select a move (implement your logic here)
        selected_move = self._select_move(game, legal_moves)
        
        # Convert move to action index
        action = self._move_to_action(selected_move, board_size)
        
        return action
    
    def _observation_to_game(self, observation: Dict[str, Any]) -> QoridorGame:
        """
        Convert an observation to a QoridorGame instance.
        
        Args:
            observation: The environment observation
            
        Returns:
            A QoridorGame instance
        """
        # Get board size
        board_size = observation['board'].shape[0]
        
        # Create game
        game = QoridorGame(board_size=board_size, num_walls=observation['walls_left'].max())
        
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
    
    def _select_move(self, game: QoridorGame, legal_moves: List[Move]) -> Move:
        """
        Select a move from the list of legal moves.
        
        Args:
            game: The current game state
            legal_moves: List of legal moves
            
        Returns:
            The selected move
        """
        # IMPLEMENT YOUR MOVE SELECTION LOGIC HERE
        
        # Example: Just pick a random move
        return np.random.choice(legal_moves)
    
    def _move_to_action(self, move: Move, board_size: int) -> int:
        """
        Convert a Move object to a flat action index.
        
        Args:
            move: The Move object
            board_size: Size of the board
            
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
    
    def update(self, state, action, next_state, reward, done):
        """
        Update the agent based on experience.
        
        Implement this method if your agent needs to learn from experience.
        
        Args:
            state: The previous state
            action: The action taken
            next_state: The resulting state
            reward: The reward received
            done: Whether the episode is done
        """
        # IMPLEMENT YOUR LEARNING ALGORITHM HERE
        pass
    
    def save(self, filepath):
        """
        Save the agent to a file.
        
        Args:
            filepath: The path to save the agent
        """
        # IMPLEMENT MODEL SAVING LOGIC HERE
        pass
    
    @classmethod
    def load(cls, filepath):
        """
        Load an agent from a file.
        
        Args:
            filepath: The path to the saved agent
            
        Returns:
            A new agent instance
        """
        # IMPLEMENT MODEL LOADING LOGIC HERE
        agent = cls()
        # Load agent parameters
        return agent