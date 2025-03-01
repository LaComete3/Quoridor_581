"""
Main game class for Qoridor.
"""

from typing import Dict, Any, List, Tuple, Optional, Callable
import time
from qoridor.state import GameState
from qoridor.move import Move, MoveType
from qoridor.board import WallOrientation


class QoridorGame:
    """
    Main game class for Qoridor.
    
    This class manages the game flow, player interactions, and game rules.
    """
    
    def __init__(self, board_size: int = 9, num_walls: int = 10):
        """
        Initialize a new Qoridor game.
        
        Args:
            board_size: Size of the board (default: 9)
            num_walls: Number of walls each player starts with (default: 10)
        """
        self.state = GameState(board_size=board_size, num_walls=num_walls)
        self.move_log = []
        self.callbacks = {}
    
    def reset(self) -> None:
        """Reset the game to its initial state."""
        self.state.reset()
        self.move_log = []
        self._trigger_callback('on_reset')
    
    def make_move(self, move: Move) -> bool:
        """
        Make a move in the game.
        
        Args:
            move: The Move object representing the move
            
        Returns:
            True if the move was successful, False otherwise
        """
        # Check if it's the correct player's turn
        if move.player != self.state.current_player:
            return False
        
        # Convert Move to action dictionary
        action = move.to_dict()
        
        # Apply the action
        success, winner = self.state.apply_action(action)
        
        if success:
            self.move_log.append(move)
            self._trigger_callback('on_move', move)
            
            if winner is not None:
                self._trigger_callback('on_game_end', winner)
        
        return success
    
    def get_legal_moves(self, player: Optional[int] = None) -> List[Move]:
        """
        Get all legal moves for a player.
        
        Args:
            player: The player to get moves for (default: current player)
            
        Returns:
            A list of legal Move objects
        """
        if player is None:
            player = self.state.current_player
        
        # Save the current player
        original_player = self.state.current_player
        
        # Temporarily set the current player to get the correct legal actions
        self.state.current_player = player
        actions = self.state.get_legal_actions()
        
        # Restore the original player
        self.state.current_player = original_player
        
        # Convert action dictionaries to Move objects
        moves = []
        for action in actions:
            if action['type'] == 'move':
                moves.append(Move.pawn_move(player, action['position'][0], action['position'][1]))
            else:  # 'wall'
                moves.append(Move.wall_placement(
                    player, 
                    action['position'][0], 
                    action['position'][1], 
                    action['orientation']
                ))
        
        return moves
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        
        Returns:
            True if the game is over, False otherwise
        """
        return self.state.done
    
    def get_winner(self) -> Optional[int]:
        """
        Get the winner of the game.
        
        Returns:
            The player number of the winner (1 or 2), or None if the game is not over
        """
        return self.state.winner
    
    def get_current_player(self) -> int:
        """
        Get the current player.
        
        Returns:
            The current player (1 or 2)
        """
        return self.state.current_player
    
    def get_walls_left(self, player: int) -> int:
        """
        Get the number of walls a player has left.
        
        Args:
            player: The player number (1 or 2)
            
        Returns:
            The number of walls left
        """
        return self.state.get_walls_left(player)
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """
        Register a callback for a game event.
        
        Args:
            event: The event name ('on_move', 'on_game_end', 'on_reset')
            callback: The callback function
        """
        if event not in self.callbacks:
            self.callbacks[event] = []
        
        self.callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, *args, **kwargs) -> None:
        """
        Trigger all callbacks for an event.
        
        Args:
            event: The event name
            *args, **kwargs: Arguments to pass to the callback
        """
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                callback(*args, **kwargs)
    
    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the game state.
        
        Returns:
            A dictionary containing the serialized game state
        """
        return {
            'state': self.state.get_serialized_state(),
            'move_log': [move.to_dict() for move in self.move_log]
        }
    
    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> 'QoridorGame':
        """
        Create a game from serialized data.
        
        Args:
            data: Serialized game data
            
        Returns:
            A new QoridorGame instance
        """
        state_data = data['state']
        game = cls(
            board_size=state_data['board_size'],
            num_walls=max(state_data['player1_walls_left'], state_data['player2_walls_left'])
        )
        
        # Set the state
        game.state = GameState.from_serialized(state_data)
        
        # Set the move log
        game.move_log = [Move.from_dict(move_data) for move_data in data['move_log']]
        
        return game
    
    def __str__(self) -> str:
        """String representation of the game."""
        return str(self.state)
