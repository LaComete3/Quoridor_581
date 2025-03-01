"""
Visualization utilities for Qoridor.
"""

from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from qoridor.game import QoridorGame
from qoridor.state import GameState
from qoridor.board import Board, WallOrientation


class QoridorVisualizer:
    """
    Visualizer for Qoridor game states.
    
    This class provides methods to render the game board, player positions,
    and walls using matplotlib.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (8, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size for matplotlib (width, height)
        """
        self.figsize = figsize
        self.colors = {
            'board': '#E0E0E0',
            'grid': '#A0A0A0',
            'player1': '#3498DB',  # Blue
            'player2': '#E74C3C',  # Red
            'wall_h': '#8E44AD',   # Purple
            'wall_v': '#27AE60',   # Green
            'highlight': '#F39C12'  # Orange (for highlighting moves)
        }
    
    def render_game(self, game: QoridorGame, highlight_moves: List[Tuple[int, int]] = None,
                   highlight_walls: List[Tuple[int, int, str]] = None,
                   show: bool = True) -> Optional[plt.Figure]:
        """
        Render a Qoridor game.
        
        Args:
            game: The QoridorGame to render
            highlight_moves: Optional list of (row, col) positions to highlight
            highlight_walls: Optional list of (row, col, orientation) wall positions to highlight
            show: Whether to display the plot
            
        Returns:
            The matplotlib Figure if show is False, None otherwise
        """
        return self.render_state(game.state, highlight_moves, highlight_walls, show)
    
    def render_state(self, state: GameState, highlight_moves: List[Tuple[int, int]] = None,
                    highlight_walls: List[Tuple[int, int, str]] = None,
                    show: bool = True) -> Optional[plt.Figure]:
        """
        Render a Qoridor game state.
        
        Args:
            state: The GameState to render
            highlight_moves: Optional list of (row, col) positions to highlight
            highlight_walls: Optional list of (row, col, orientation) wall positions to highlight
            show: Whether to display the plot
            
        Returns:
            The matplotlib Figure if show is False, None otherwise
        """
        board = state.board
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize)
        ax.set_aspect('equal')
        
        # Draw board
        self._draw_board(ax, board.size)
        
        # Draw players
        self._draw_players(ax, board)
        
        # Draw walls
        self._draw_walls(ax, board)
        
        # Highlight moves if provided
        if highlight_moves:
            self._highlight_moves(ax, highlight_moves, board.size)
        
        # Highlight walls if provided
        if highlight_walls:
            self._highlight_walls(ax, highlight_walls, board.size)
        
        # Add game information
        self._add_game_info(ax, state)
        
        # Set axis limits with padding
        ax.set_xlim(-0.5, board.size - 0.5)
        ax.set_ylim(-0.5, board.size - 0.5)
        
        # Hide axes
        ax.axis('off')
        
        if show:
            plt.tight_layout()
            plt.show()
            return None
        else:
            plt.tight_layout()
            return fig
    
    def _draw_board(self, ax: plt.Axes, size: int) -> None:
        """
        Draw the game board.
        
        Args:
            ax: Matplotlib axes
            size: Board size
        """
        # Draw board background
        board_rect = patches.Rectangle(
            (-0.5, -0.5), size, size, 
            linewidth=2, 
            edgecolor='black', 
            facecolor=self.colors['board']
        )
        ax.add_patch(board_rect)
        
        # Draw grid lines
        for i in range(size):
            # Vertical lines
            ax.plot([i - 0.5, i - 0.5], [-0.5, size - 0.5], 
                   color=self.colors['grid'], linestyle='-', linewidth=1)
            
            # Horizontal lines
            ax.plot([-0.5, size - 0.5], [i - 0.5, i - 0.5],
                   color=self.colors['grid'], linestyle='-', linewidth=1)
        
        # Add coordinates
        for i in range(size):
            # Column labels (letters)
            col_label = chr(65 + i)  # A, B, C, ...
            ax.text(i, -0.7, col_label, ha='center', va='center', fontsize=12)
            
            # Row labels (numbers)
            ax.text(-0.7, i, str(i+1), ha='center', va='center', fontsize=12)
    
    def _draw_players(self, ax: plt.Axes, board: Board) -> None:
        """
        Draw player tokens.
        
        Args:
            ax: Matplotlib axes
            board: Game board
        """
        # Get player positions
        p1_row, p1_col = board.get_player_position(1)
        p2_row, p2_col = board.get_player_position(2)
        
        # Draw player 1
        p1_circle = patches.Circle(
            (p1_col, p1_row), 0.3,
            facecolor=self.colors['player1'],
            edgecolor='black',
            linewidth=2,
            label='Player 1'
        )
        ax.add_patch(p1_circle)
        ax.text(p1_col, p1_row, '1', 
               ha='center', va='center', 
               color='white', fontweight='bold', fontsize=14)
        
        # Draw player 2
        p2_circle = patches.Circle(
            (p2_col, p2_row), 0.3,
            facecolor=self.colors['player2'],
            edgecolor='black',
            linewidth=2,
            label='Player 2'
        )
        ax.add_patch(p2_circle)
        ax.text(p2_col, p2_row, '2', 
               ha='center', va='center', 
               color='white', fontweight='bold', fontsize=14)
    
    def _draw_walls(self, ax: plt.Axes, board: Board) -> None:
        """
        Draw walls on the board.
        
        Args:
            ax: Matplotlib axes
            board: Game board
        """
        # Draw horizontal walls
        for row in range(board.horizontal_walls.shape[0]):
            for col in range(board.horizontal_walls.shape[1]):
                if board.horizontal_walls[row, col]:
                    self._draw_wall(ax, row, col, WallOrientation.HORIZONTAL)
        
        # Draw vertical walls
        for row in range(board.vertical_walls.shape[0]):
            for col in range(board.vertical_walls.shape[1]):
                if board.vertical_walls[row, col]:
                    self._draw_wall(ax, row, col, WallOrientation.VERTICAL)
    
    def _draw_wall(self, ax: plt.Axes, row: int, col: int, orientation: WallOrientation) -> None:
        """
        Draw a single wall.
        
        Args:
            ax: Matplotlib axes
            row: Wall row
            col: Wall column
            orientation: Wall orientation
        """
        if orientation == WallOrientation.HORIZONTAL:
            rect = patches.Rectangle(
                (col - 0.45, row + 0.45), 2.0, 0.1,
                facecolor=self.colors['wall_h'],
                edgecolor='black',
                linewidth=1
            )
        else:  # VERTICAL
            rect = patches.Rectangle(
                (col + 0.45, row - 0.45), 0.1, 2.0,
                facecolor=self.colors['wall_v'],
                edgecolor='black',
                linewidth=1
            )
        ax.add_patch(rect)
    
    def _highlight_moves(self, ax: plt.Axes, moves: List[Tuple[int, int]], board_size: int) -> None:
        """
        Highlight possible move positions.
        
        Args:
            ax: Matplotlib axes
            moves: List of (row, col) positions to highlight
            board_size: Board size
        """
        for row, col in moves:
            if 0 <= row < board_size and 0 <= col < board_size:
                highlight = patches.Circle(
                    (col, row), 0.2,
                    facecolor=self.colors['highlight'],
                    alpha=0.5
                )
                ax.add_patch(highlight)
    
    def _highlight_walls(self, ax: plt.Axes, walls: List[Tuple[int, int, str]], board_size: int) -> None:
        """
        Highlight possible wall placements.
        
        Args:
            ax: Matplotlib axes
            walls: List of (row, col, orientation) wall positions to highlight
            board_size: Board size
        """
        for row, col, orientation in walls:
            if 0 <= row < board_size-1 and 0 <= col < board_size-1:
                if orientation.lower() == 'h' or orientation == WallOrientation.HORIZONTAL:
                    rect = patches.Rectangle(
                        (col - 0.45, row + 0.45), 2.0, 0.1,
                        facecolor=self.colors['highlight'],
                        edgecolor='black',
                        linewidth=1,
                        alpha=0.5
                    )
                else:  # VERTICAL
                    rect = patches.Rectangle(
                        (col + 0.45, row - 0.45), 0.1, 2.0,
                        facecolor=self.colors['highlight'],
                        edgecolor='black',
                        linewidth=1,
                        alpha=0.5
                    )
                ax.add_patch(rect)
    
    def _add_game_info(self, ax: plt.Axes, state: GameState) -> None:
        """
        Add game information to the plot.
        
        Args:
            ax: Matplotlib axes
            state: Game state
        """
        # Create title with game info
        title = f"Qoridor - Player {state.current_player}'s Turn"
        if state.done:
            title = f"Qoridor - Game Over: Player {state.winner} Wins!"
        
        # Add walls left info
        walls_info = (
            f"Walls Left:  "
            f"Player 1: {state.player1_walls_left}  |  "
            f"Player 2: {state.player2_walls_left}"
        )
        
        # Add title and info
        ax.set_title(title, fontsize=16, pad=20)
        ax.text(state.board_size / 2 - 0.5, -1.2, walls_info, 
               ha='center', va='center', fontsize=12)
        
        # Add legend for wall colors
        h_patch = patches.Patch(color=self.colors['wall_h'], label='Horizontal Wall')
        v_patch = patches.Patch(color=self.colors['wall_v'], label='Vertical Wall')
        p1_patch = patches.Patch(color=self.colors['player1'], label='Player 1')
        p2_patch = patches.Patch(color=self.colors['player2'], label='Player 2')
        
        ax.legend(
            handles=[p1_patch, p2_patch, h_patch, v_patch],
            loc='upper left',
            bbox_to_anchor=(1.05, 1),
            fontsize=10
        )

def render_move_sequence(game: QoridorGame, moves: List[Dict[str, Any]], 
                        delay: float = 1.0, save_gif: Optional[str] = None) -> None:
    """
    Render a sequence of moves as an animation.
    
    Args:
        game: The QoridorGame instance
        moves: List of move dictionaries
        delay: Delay between frames (seconds)
        save_gif: Path to save GIF file (optional)
    """
    try:
        import matplotlib.animation as animation
        from IPython.display import display, clear_output
        import time
    except ImportError:
        print("Required libraries not found. Install with: pip install matplotlib ipython")
        return
    
    # Reset the game
    game.reset()
    visualizer = QoridorVisualizer()
    
    # Apply each move and render
    if save_gif:
        # Set up animation
        fig = plt.figure(figsize=(10, 10))
        frames = []
        
        # Initial state
        ax = fig.add_subplot(111)
        frames.append([visualizer.render_state(game.state, show=False)])
        
        # Apply moves
        for move_idx, move_dict in enumerate(moves):
            move = move_dict['move']
            success = game.make_move(move)
            
            if not success:
                print(f"Invalid move at step {move_idx + 1}: {move}")
                continue
            
            # Render the current state
            frame = visualizer.render_state(game.state, show=False)
            frames.append([frame])
        
        # Create animation
        ani = animation.ArtistAnimation(fig, frames, interval=delay*1000, blit=True)
        ani.save(save_gif, writer='pillow')
        plt.close()
        print(f"Animation saved to {save_gif}")
    else:
        # Interactive display
        for move_idx, move_dict in enumerate(moves):
            move = move_dict['move']
            success = game.make_move(move)
            
            if not success:
                print(f"Invalid move at step {move_idx + 1}: {move}")
                continue
            
            # Render the current state
            clear_output(wait=True)
            visualizer.render_state(game.state)
            
            # Pause for the specified delay
            time.sleep(delay)