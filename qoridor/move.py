"""
Move representation for Qoridor.
"""

from typing import Tuple, Dict, Any, Optional
from enum import Enum
from qoridor.board import WallOrientation


class MoveType(Enum):
    """Types of moves in Qoridor."""
    PAWN_MOVE = 0
    WALL_PLACEMENT = 1


class Move:
    """
    Represents a move in Qoridor.
    
    A move can be either a pawn movement or a wall placement.
    """
    
    def __init__(self, player: int, move_type: MoveType, position: Tuple[int, int], 
                 wall_orientation: Optional[WallOrientation] = None):
        """
        Initialize a move.
        
        Args:
            player: The player making the move (1 or 2)
            move_type: The type of move (PAWN_MOVE or WALL_PLACEMENT)
            position: The position (row, col) for the move
            wall_orientation: For wall placements, the orientation of the wall
        """
        self.player = player
        self.move_type = move_type
        self.position = position
        self.wall_orientation = wall_orientation
        
        # Validate the move
        if move_type == MoveType.WALL_PLACEMENT and wall_orientation is None:
            raise ValueError("Wall orientation must be specified for wall placements")
    
    @classmethod
    def pawn_move(cls, player: int, row: int, col: int) -> 'Move':
        """
        Create a pawn move.
        
        Args:
            player: The player making the move
            row: The destination row
            col: The destination column
            
        Returns:
            A Move object representing a pawn move
        """
        return cls(player, MoveType.PAWN_MOVE, (row, col))
    
    @classmethod
    def wall_placement(cls, player: int, row: int, col: int, orientation: WallOrientation) -> 'Move':
        """
        Create a wall placement move.
        
        Args:
            player: The player making the move
            row: The row for the wall
            col: The column for the wall
            orientation: The orientation of the wall
            
        Returns:
            A Move object representing a wall placement
        """
        return cls(player, MoveType.WALL_PLACEMENT, (row, col), orientation)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the move to a dictionary.
        
        Returns:
            A dictionary representation of the move
        """
        if self.move_type == MoveType.PAWN_MOVE:
            return {
                'type': 'move',
                'player': self.player,
                'position': self.position
            }
        else:  # WALL_PLACEMENT
            return {
                'type': 'wall',
                'player': self.player,
                'position': self.position,
                'orientation': self.wall_orientation
            }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Move':
        """
        Create a Move from a dictionary.
        
        Args:
            data: Dictionary representation of a move
            
        Returns:
            A Move object
        """
        player = data.get('player', 1)
        position = data['position']
        
        if data['type'] == 'move':
            return cls.pawn_move(player, position[0], position[1])
        elif data['type'] == 'wall':
            return cls.wall_placement(player, position[0], position[1], data['orientation'])
        else:
            raise ValueError(f"Unknown move type: {data['type']}")
    
    def __str__(self) -> str:
        """String representation of the move."""
        row, col = self.position
        
        if self.move_type == MoveType.PAWN_MOVE:
            return f"Player {self.player} moves to ({row}, {col})"
        else:  # WALL_PLACEMENT
            orientation = "horizontal" if self.wall_orientation == WallOrientation.HORIZONTAL else "vertical"
            return f"Player {self.player} places {orientation} wall at ({row}, {col})"
    
    def __eq__(self, other: object) -> bool:
        """Check if two moves are equal."""
        if not isinstance(other, Move):
            return False
        
        return (self.player == other.player and
                self.move_type == other.move_type and
                self.position == other.position and
                self.wall_orientation == other.wall_orientation)
