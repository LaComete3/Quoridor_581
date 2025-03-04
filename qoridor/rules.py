"""
Game rules for Qoridor.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from queue import Queue
from .board import Board, WallOrientation


class QoridorRules:
    """Static methods for checking and enforcing Qoridor rules."""
    
    @staticmethod
    def is_valid_move(board: Board, player: int, row: int, col: int) -> bool:
        """
        Check if a move is valid according to Qoridor rules.
        
        Args:
            board: The game board
            player: The player making the move (1 or 2)
            row: The destination row
            col: The destination column
            
        Returns:
            True if the move is valid, False otherwise
        """
        # Get the player's current position
        current_pos = board.get_player_position(player)
        
        # Get all legal moves for the player
        legal_moves = board.get_legal_moves(player)
        
        # Check if the target position is in the legal moves
        return (row, col) in legal_moves
    
    @staticmethod
    def is_valid_wall_placement(board: Board, row: int, col: int, 
                               orientation: WallOrientation) -> bool:
        """
        Check if a wall placement is valid according to Qoridor rules.
        
        Args:
            board: The game board
            row: The wall row
            col: The wall column
            orientation: The wall orientation
            
        Returns:
            True if the wall placement is valid, False otherwise
        """
        # Check if the placement is valid in terms of position and crossing
        if not board._is_valid_wall_position(row, col, orientation):
            return False
        
        # Temporarily place the wall
        if orientation == WallOrientation.HORIZONTAL:
            board.horizontal_walls[row, col] = True
            # Place the second unit of the wall if in bounds
            if col + 1 < board.size - 1:
                board.horizontal_walls[row, col + 1] = True
        else:  # VERTICAL
            board.vertical_walls[row, col] = True
            # Place the second unit of the wall if in bounds
            if row + 1 < board.size - 1:
                board.vertical_walls[row + 1, col] = True
        
        # Check if both players still have a path to their goal
        path_exists = board.has_path_to_goal(1) and board.has_path_to_goal(2)
        
        # Remove the temporary wall
        if orientation == WallOrientation.HORIZONTAL:
            board.horizontal_walls[row, col] = False
            if col + 1 < board.size - 1:
                board.horizontal_walls[row, col + 1] = False
        else:  # VERTICAL
            board.vertical_walls[row, col] = False
            if row + 1 < board.size - 1:
                board.vertical_walls[row + 1, col] = False
        
        return path_exists
    
    @staticmethod
    def find_shortest_path(board: Board, player: int) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path to the goal for a player using BFS.
        
        Args:
            board: The game board
            player: The player number (1 or 2)
            
        Returns:
            A list of positions forming the shortest path, or None if no path exists
        """
        # Get the player's position
        start = board.get_player_position(player)
        
        # Determine the goal row
        goal_row = board.size - 1 if player == 1 else 0
        
        # BFS to find the shortest path
        visited = set([start])
        queue = Queue()
        queue.put((start, [start]))  # (position, path)
        
        while not queue.empty():
            (row, col), path = queue.get()
            
            # Check if we've reached the goal
            if (player == 1 and row == goal_row) or (player == 2 and row == goal_row):
                return path
            
            # Explore neighbors
            for next_pos in board.get_adjacent_cells((row, col)):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.put((next_pos, path + [next_pos]))
        
        # No path found
        return None
    
    @staticmethod
    def calculate_distances_to_goal(board: Board) -> Dict[int, np.ndarray]:
        """
        Calculate distances to goal for each player from every position.
        
        Args:
            board: The game board
            
        Returns:
            Dictionary mapping player number to distance array
        """
        distances = {}
        
        for player in [1, 2]:
            # Initialize distance array with infinity
            dist = np.full((board.size, board.size), np.inf)
            
            # Determine the goal row
            goal_row = board.size - 1 if player == 1 else 0
            
            # Set goal cells to distance 0
            for col in range(board.size):
                dist[goal_row, col] = 0
            
            # BFS from the goal row
            visited = set()
            queue = Queue()
            
            for col in range(board.size):
                queue.put((goal_row, col))
                visited.add((goal_row, col))
            
            while not queue.empty():
                row, col = queue.get()
                
                # Explore neighbors
                for next_row, next_col in board.get_adjacent_cells((row, col)):
                    if (next_row, next_col) not in visited:
                        visited.add((next_row, next_col))
                        dist[next_row, next_col] = dist[row, col] + 1
                        queue.put((next_row, next_col))
            
            distances[player] = dist
        
        return distances
    
    @staticmethod
    def is_jump_move(board: Board, player: int, start: Tuple[int, int], 
                     end: Tuple[int, int]) -> bool:
        """
        Check if a move is a jump over an opponent.
        
        Args:
            board: The game board
            player: The player making the move
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            True if the move is a jump, False otherwise
        """
        # Get opponent position
        opponent = 3 - player  # Alternates between 1 and 2
        opponent_pos = board.get_player_position(opponent)
        
        # Calculate differences
        start_row, start_col = start
        end_row, end_col = end
        opp_row, opp_col = opponent_pos
        
        # Check if the opponent is between the start and end positions
        
        # Straight jumps (2 steps away)
        if (abs(end_row - start_row) == 2 and end_col == start_col) or \
           (abs(end_col - start_col) == 2 and end_row == start_row):
            # Calculate the position between
            middle_row = (start_row + end_row) // 2
            middle_col = (start_col + end_col) // 2
            
            # Check if the opponent is in the middle
            if (middle_row, middle_col) != opponent_pos:
                return False
                
            # Check if there's no wall between start and middle
            if board.is_wall_between(start, (middle_row, middle_col)):
                return False
                
            # Check if there's no wall between middle and end
            if board.is_wall_between((middle_row, middle_col), end):
                return False
                
            return True
        
        # Diagonal jumps (opponent adjacent but wall blocks straight jump)
        if abs(end_row - start_row) == 1 and abs(end_col - start_col) == 1:
            # Opponent must be adjacent to start
            if abs(opp_row - start_row) + abs(opp_col - start_col) == 1:
                # If trying to move vertically but blocked
                if opp_row != start_row:
                    # Check if there's a wall blocking vertical movement
                    if board.is_wall_between((start_row, start_col), (opp_row, start_col)):
                        # Ensure there's no wall blocking the diagonal move
                        return not board.is_wall_between((opp_row, opp_col), (end_row, end_col))
                
                # If trying to move horizontally but blocked
                if opp_col != start_col:
                    # Check if there's a wall blocking horizontal movement
                    if board.is_wall_between((start_row, start_col), (start_row, opp_col)):
                        # Ensure there's no wall blocking the diagonal move
                        return not board.is_wall_between((opp_row, opp_col), (end_row, end_col))
        
        return False
    
    @staticmethod
    def calculate_move_type(board: Board, player: int, start: Tuple[int, int], 
                           end: Tuple[int, int]) -> str:
        """
        Determine the type of move being made.
        
        Args:
            board: The game board
            player: The player making the move
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            A string describing the move type
        """
        # Calculate differences
        row_diff = abs(end[0] - start[0])
        col_diff = abs(end[1] - start[1])
        
        # Regular move (1 square)
        if row_diff + col_diff == 1:
            return "regular"
        
        # Jump over opponent
        if QoridorRules.is_jump_move(board, player, start, end):
            if row_diff == 2 or col_diff == 2:
                return "straight_jump"
            else:
                return "diagonal_jump"
        
        return "invalid"
