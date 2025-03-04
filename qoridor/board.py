"""
Board representation for Qoridor.
"""

import numpy as np
from enum import Enum
from typing import Tuple, List, Optional, Set


class CellType(Enum):
    """Enum for the type of cell on the board."""
    EMPTY = 0
    PLAYER_1 = 1
    PLAYER_2 = 2


class WallOrientation(Enum):
    """Enum for the orientation of walls."""
    HORIZONTAL = 0
    VERTICAL = 1


class Board:
    """
    Board representation for Qoridor game.
    
    The board consists of a grid where players can move and place walls.
    Walls block movement between cells.
    """
    
    def __init__(self, size: int = 9):
        """
        Initialize the board.
        
        Args:
            size: The size of the board (default is 9, the standard Qoridor board size)
        """
        if size % 2 == 0 or size < 3:
            raise ValueError("Board size must be odd and at least 3")
        
        self.size = size
        
        # Grid representation
        # 0 = empty, 1 = player 1, 2 = player 2
        self.grid = np.zeros((size, size), dtype=np.int8)
        
        # Wall representation:
        # horizontal_walls[i, j] = True means there is a wall blocking movement 
        # between row i and i+1, starting at column j
        self.horizontal_walls = np.zeros((size - 1, size - 1), dtype=bool)
        
        # vertical_walls[i, j] = True means there is a wall blocking movement
        # between column j and j+1, starting at row i
        self.vertical_walls = np.zeros((size - 1, size - 1), dtype=bool)
        
        # Set initial player positions
        self._set_initial_positions()
        
    def _set_initial_positions(self):
        """Set the initial positions of the players."""
        # Player 1 starts at the middle of the top row
        self.grid[0, self.size // 2] = CellType.PLAYER_1.value
        self.player1_pos = (0, self.size // 2)
        
        # Player 2 starts at the middle of the bottom row
        self.grid[self.size - 1, self.size // 2] = CellType.PLAYER_2.value
        self.player2_pos = (self.size - 1, self.size // 2)
    
    def reset(self):
        """Reset the board to its initial state."""
        self.grid.fill(0)
        self.horizontal_walls.fill(False)
        self.vertical_walls.fill(False)
        self._set_initial_positions()
    
    def get_player_position(self, player: int) -> Tuple[int, int]:
        """
        Get the position of a player.
        
        Args:
            player: The player number (1 or 2)
            
        Returns:
            The row and column of the player
        """
        if player == 1:
            return self.player1_pos
        elif player == 2:
            return self.player2_pos
        else:
            raise ValueError("Player must be 1 or 2")
    
    def move_player(self, player: int, new_position: Tuple[int, int]) -> bool:
        """
        Move a player to a new position.
        
        Args:
            player: The player number (1 or 2)
            new_position: The new position (row, col)
            
        Returns:
            True if the move was successful, False otherwise
        """
        if not self._is_valid_cell(new_position):
            return False
        
        row, col = new_position
        
        if player == 1:
            old_row, old_col = self.player1_pos
            self.grid[old_row, old_col] = CellType.EMPTY.value
            self.grid[row, col] = CellType.PLAYER_1.value
            self.player1_pos = (row, col)
        elif player == 2:
            old_row, old_col = self.player2_pos
            self.grid[old_row, old_col] = CellType.EMPTY.value
            self.grid[row, col] = CellType.PLAYER_2.value
            self.player2_pos = (row, col)
        else:
            return False
        
        return True
    
    def place_wall(self, row: int, col: int, orientation: WallOrientation) -> bool:
        """
        Place a wall on the board. Walls are 2 units long.
        
        Args:
            row: The starting row of the wall
            col: The starting column of the wall
            orientation: The orientation of the wall (horizontal or vertical)
            
        Returns:
            True if the wall was successfully placed, False otherwise
        """
        if not self._is_valid_wall_position(row, col, orientation):
            return False
        
        if orientation == WallOrientation.HORIZONTAL:
            # A horizontal wall spans 2 grid cells horizontally
            self.horizontal_walls[row, col] = True
            
            # Check if the second part of the wall would be out of bounds
            if col + 1 < self.size - 1:
                self.horizontal_walls[row, col + 1] = True
        else:  # VERTICAL
            # A vertical wall spans 2 grid cells vertically
            self.vertical_walls[row, col] = True
            
            # Check if the second part of the wall would be out of bounds
            if row + 1 < self.size - 1:
                self.vertical_walls[row + 1, col] = True
            
        return True
    
    def is_wall_at(self, row: int, col: int, orientation: WallOrientation) -> bool:
        """
        Check if there is a wall at the specified position.
        Remember that walls are 2 units long, so this checks if the position
        is either the start or continuation of a wall.
        
        Args:
            row: The row to check
            col: The column to check
            orientation: The orientation to check
            
        Returns:
            True if there is a wall, False otherwise
        """
        if not 0 <= row < self.size - 1 or not 0 <= col < self.size - 1:
            return False
        
        if orientation == WallOrientation.HORIZONTAL:
            # Check if there's a horizontal wall starting at this position
            if self.horizontal_walls[row, col]:
                return True
                
            # Check if this position is the second unit of a horizontal wall (wall starts one cell to the left)
            if col > 0 and self.horizontal_walls[row, col - 1]:
                return True
        else:  # VERTICAL
            # Check if there's a vertical wall starting at this position
            if self.vertical_walls[row, col]:
                return True
                
            # Check if this position is the second unit of a vertical wall (wall starts one cell above)
            if row > 0 and self.vertical_walls[row - 1, col]:
                return True
                
        return False
    
    def get_legal_moves(self, player: int) -> List[Tuple[int, int]]:
        """
        Get all legal moves for a player.
        
        Args:
            player: The player number (1 or 2)
            
        Returns:
            A list of valid positions (row, col) the player can move to
        """
        if player == 1:
            row, col = self.player1_pos
            opponent_pos = self.player2_pos
        else:
            row, col = self.player2_pos
            opponent_pos = self.player1_pos
        
        potential_moves = []
        
        # Check the four cardinal directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Skip if out of bounds
            if not self._is_valid_cell((new_row, new_col)):
                continue
            
            # Check if there's a wall blocking the move using is_wall_between
            if self.is_wall_between((row, col), (new_row, new_col)):
                continue
            
            # Check if the cell is occupied by the opponent
            if (new_row, new_col) == opponent_pos:
                # If opponent is in the way, we can jump over if no wall behind
                jump_row, jump_col = new_row + dr, new_col + dc
                
                # Check if the jump is valid (in bounds and no wall)
                if self._is_valid_cell((jump_row, jump_col)) and not self.is_wall_between((new_row, new_col), (jump_row, jump_col)):
                    potential_moves.append((jump_row, jump_col))
                else:
                    # Check diagonal jumps (when wall blocks straight jump)
                    diagonal_moves = self._get_diagonal_jumps(player, (new_row, new_col), (dr, dc))
                    potential_moves.extend(diagonal_moves)
            else:
                potential_moves.append((new_row, new_col))
        
        return potential_moves
    
    def _get_diagonal_jumps(self, player: int, opponent_pos: Tuple[int, int], 
                           direction: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get possible diagonal jumps over an opponent when straight jump is blocked.
        In Qoridor, if you can't jump straight over an opponent because there's a wall
        or edge of the board, you can move diagonally around them.
        
        Args:
            player: The player number
            opponent_pos: The position of the opponent
            direction: The direction we were trying to move
            
        Returns:
            List of valid diagonal moves
        """
        opp_row, opp_col = opponent_pos
        dr, dc = direction
        jumps = []
        
        # Calculate the position we were trying to reach by jumping straight
        straight_jump_row = opp_row + dr
        straight_jump_col = opp_col + dc
        straight_jump_blocked = (not self._is_valid_cell((straight_jump_row, straight_jump_col)) or 
                               self.is_wall_between(opponent_pos, (straight_jump_row, straight_jump_col)))
        
        # Only consider diagonal jumps if straight jump is blocked
        if straight_jump_blocked:
            # If moving vertically (up or down) and blocked
            if dc == 0:
                # Try jumping diagonally left
                left_pos = (opp_row, opp_col - 1)
                if (self._is_valid_cell(left_pos) and 
                    not self.is_wall_between(opponent_pos, left_pos)):
                    jumps.append(left_pos)
                
                # Try jumping diagonally right
                right_pos = (opp_row, opp_col + 1)
                if (self._is_valid_cell(right_pos) and 
                    not self.is_wall_between(opponent_pos, right_pos)):
                    jumps.append(right_pos)
                    
            # If moving horizontally (left or right) and blocked
            if dr == 0:
                # Try jumping diagonally up
                up_pos = (opp_row - 1, opp_col)
                if (self._is_valid_cell(up_pos) and 
                    not self.is_wall_between(opponent_pos, up_pos)):
                    jumps.append(up_pos)
                
                # Try jumping diagonally down
                down_pos = (opp_row + 1, opp_col)
                if (self._is_valid_cell(down_pos) and 
                    not self.is_wall_between(opponent_pos, down_pos)):
                    jumps.append(down_pos)
        
        return jumps
    
    def is_wall_between(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """
        Check if there is a wall between two adjacent positions.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            True if there is a wall between the positions, False otherwise
        """
        row1, col1 = pos1
        row2, col2 = pos2
        
        # Positions must be adjacent
        if abs(row1 - row2) + abs(col1 - col2) != 1:
            return False
        
        # Vertical movement (moving up or down)
        if col1 == col2:
            min_row = min(row1, row2)
            # Check for a horizontal wall at this position
            if min_row < self.size - 1:
                # Check the wall at the position and the wall to the left (if any)
                if col1 < self.size - 1 and self.horizontal_walls[min_row, col1]:
                    return True
                if col1 > 0 and self.horizontal_walls[min_row, col1 - 1]:
                    return True
        
        # Horizontal movement (moving left or right)
        if row1 == row2:
            min_col = min(col1, col2)
            # Check for a vertical wall at this position
            if min_col < self.size - 1:
                # Check the wall at the position and the wall above (if any)
                if row1 < self.size - 1 and self.vertical_walls[row1, min_col]:
                    return True
                if row1 > 0 and self.vertical_walls[row1 - 1, min_col]:
                    return True
        
        return False
    
    def get_valid_wall_placements(self, orientation: WallOrientation) -> List[Tuple[int, int]]:
        """
        Get all valid positions to place a wall.
        
        Args:
            orientation: The orientation of the wall
            
        Returns:
            List of valid (row, col) positions for wall placement
        """
        valid_positions = []
        
        if orientation == WallOrientation.HORIZONTAL:
            # Horizontal walls can be placed at positions (row, col) where
            # 0 <= row < size-1 and 0 <= col < size-2
            for row in range(self.size - 1):
                for col in range(self.size - 2):  # -2 because walls are 2 units long
                    if self._is_valid_wall_position(row, col, orientation):
                        # Temporarily place the wall to check if it blocks paths
                        self.horizontal_walls[row, col] = True
                        if col + 1 < self.size - 1:
                            self.horizontal_walls[row, col + 1] = True
                            
                        # Check if both players still have a path to their goal
                        if self.has_path_to_goal(1) and self.has_path_to_goal(2):
                            valid_positions.append((row, col))
                            
                        # Remove the temporary wall
                        self.horizontal_walls[row, col] = False
                        if col + 1 < self.size - 1:
                            self.horizontal_walls[row, col + 1] = False
        else:  # VERTICAL
            # Vertical walls can be placed at positions (row, col) where
            # 0 <= row < size-2 and 0 <= col < size-1
            for row in range(self.size - 2):  # -2 because walls are 2 units long
                for col in range(self.size - 1):
                    if self._is_valid_wall_position(row, col, orientation):
                        # Temporarily place the wall to check if it blocks paths
                        self.vertical_walls[row, col] = True
                        if row + 1 < self.size - 1:
                            self.vertical_walls[row + 1, col] = True
                            
                        # Check if both players still have a path to their goal
                        if self.has_path_to_goal(1) and self.has_path_to_goal(2):
                            valid_positions.append((row, col))
                            
                        # Remove the temporary wall
                        self.vertical_walls[row, col] = False
                        if row + 1 < self.size - 1:
                            self.vertical_walls[row + 1, col] = False
        
        return valid_positions
    
    def _is_valid_wall_position(self, row: int, col: int, orientation: WallOrientation) -> bool:
        """
        Check if a wall can be placed at the specified position.
        
        Args:
            row: The row to check
            col: The column to check
            orientation: The orientation to check
            
        Returns:
            True if the wall can be placed, False otherwise
        """
        # Check bounds
        if not 0 <= row < self.size - 1 or not 0 <= col < self.size - 1:
            return False
        
        # Check if there's already a wall at this position (remember walls are 2 units long)
        if orientation == WallOrientation.HORIZONTAL:
            # Need space for a 2-unit horizontal wall
            if col >= self.size - 1:
                return False
                
            # Check if there's already a horizontal wall at this position or the next position
            if self.horizontal_walls[row, col] or (col + 1 < self.size - 1 and self.horizontal_walls[row, col + 1]):
                return False
                
            # Check if this would cross with a vertical wall
            # A horizontal wall at (row,col) would cross with vertical walls at (row,col) and (row,col+1)
            if self.vertical_walls[row, col]:
                return False
            
            # Check if there's a vertical wall to the right that would be crossed
            if col + 1 < self.size - 1 and self.vertical_walls[row, col + 1]:
                return False
                
        else:  # VERTICAL
            # Need space for a 2-unit vertical wall
            if row >= self.size - 1:
                return False
                
            # Check if there's already a vertical wall at this position or the next position
            if self.vertical_walls[row, col] or (row + 1 < self.size - 1 and self.vertical_walls[row + 1, col]):
                return False
                
            # Check if this would cross with a horizontal wall
            # A vertical wall at (row,col) would cross with horizontal walls at (row,col) and (row+1,col)
            if self.horizontal_walls[row, col]:
                return False
                
            # Check if there's a horizontal wall below that would be crossed
            if row + 1 < self.size - 1 and self.horizontal_walls[row + 1, col]:
                return False
        
        return True
    
    def _is_valid_cell(self, position: Tuple[int, int]) -> bool:
        """
        Check if a cell position is valid.
        
        Args:
            position: The position to check (row, col)
            
        Returns:
            True if the position is on the board, False otherwise
        """
        row, col = position
        return 0 <= row < self.size and 0 <= col < self.size
    
    def has_path_to_goal(self, player: int) -> bool:
        """
        Check if a player has a path to their goal.
        
        Args:
            player: The player number (1 or 2)
            
        Returns:
            True if there is a path to the goal, False otherwise
        """
        # BFS to find path to goal
        if player == 1:
            start = self.player1_pos
            goal_row = self.size - 1
        else:
            start = self.player2_pos
            goal_row = 0
        
        visited = set()
        queue = [start]
        visited.add(start)
        
        while queue:
            current = queue.pop(0)
            row, col = current
            
            # Check if we've reached the goal
            if (player == 1 and row == goal_row) or (player == 2 and row == goal_row):
                return True
            
            # Get neighbors
            for next_pos in self.get_adjacent_cells(current):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        
        return False
    
    def get_adjacent_cells(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all cells adjacent to a position that are not blocked by walls.
        
        Args:
            position: The position (row, col)
            
        Returns:
            List of adjacent cell positions
        """
        row, col = position
        adjacent = []
        
        # Check the four cardinal directions
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            
            # Skip if out of bounds
            if not self._is_valid_cell((new_row, new_col)):
                continue
            
            # Check if there's a wall blocking the move
            if not self.is_wall_between(position, (new_row, new_col)):
                adjacent.append((new_row, new_col))
        
        return adjacent
    
    def __str__(self) -> str:
        """
        String representation of the board.
        Renders the board with players and walls, taking into account
        that walls are 2 units long.
        """
        s = ""
        
        # Top border
        s += "+   " + "   +".join(str(i) for i in range(self.size)) + "   +\n"
        
        for i in range(self.size):
            # Row with cells
            s += f"{i} | "
            for j in range(self.size):
                if self.grid[i, j] == CellType.PLAYER_1.value:
                    s += " 1 "
                elif self.grid[i, j] == CellType.PLAYER_2.value:
                    s += " 2 "
                else:
                    s += "   "
                
                # Vertical wall or separator
                if j < self.size - 1:
                    # Check if there's a vertical wall at this position or if this is part of a wall
                    # starting one cell above
                    if (i < self.size - 1 and self.vertical_walls[i, j]) or \
                       (i > 0 and i < self.size and self.vertical_walls[i-1, j]):
                        s += "‖"
                    else:
                        s += "|"
            s += " |\n"
            
            # Horizontal walls or separator
            if i < self.size - 1:
                s += "  | "
                for j in range(self.size):
                    # Check if there's a horizontal wall at this position or if this is part of a wall
                    # starting one cell to the left
                    if j < self.size - 1 and (self.horizontal_walls[i, j] or \
                       (j > 0 and self.horizontal_walls[i, j-1])):
                        s += "====="
                    else:
                        s += "-----"
                s += " |\n"
        
        # Bottom border
        s += "+---" + "-----+".join("" for _ in range(self.size - 1)) + "-----+\n"
        
        return s
