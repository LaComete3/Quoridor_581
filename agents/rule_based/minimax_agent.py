"""
Minimax agent for Qoridor with alpha-beta pruning.
"""

import time
import random
from typing import Dict, Any, List, Tuple, Optional, Callable
import numpy as np
from qoridor.game import QoridorGame
from qoridor.move import Move, MoveType
from qoridor.state import GameState
from qoridor.board import Board, WallOrientation
from qoridor.rules import QoridorRules


class MinimaxAgent:
    """
    Minimax agent for Qoridor with alpha-beta pruning.
    
    This agent uses the minimax algorithm with alpha-beta pruning to search
    for the best move, with a customizable evaluation function and search depth.
    """
    
    def __init__(self, player: int = 2, max_depth: int = 3, 
                eval_fn: Optional[Callable] = None, time_limit: float = None,
                use_ab_pruning: bool = True):
        """
        Initialize the minimax agent.
        
        Args:
            player: The player number (1 or 2) this agent controls
            max_depth: Maximum search depth
            eval_fn: Optional custom evaluation function
            time_limit: Optional time limit in seconds
            use_ab_pruning: Whether to use alpha-beta pruning
        """
        self.player = player
        self.max_depth = max_depth
        self.eval_fn = eval_fn if eval_fn else self.default_evaluate
        self.time_limit = time_limit
        self.use_ab_pruning = use_ab_pruning
        self.start_time = 0
        self.nodes_evaluated = 0
        
        # Cache for path distances
        self.distance_cache = {}
        
    def select_move(self, game: QoridorGame) -> Move:
        """
        Select the best move using minimax.
        
        Args:
            game: The current game state
            
        Returns:
            The best move according to minimax
        """
        self.start_time = time.time()
        self.nodes_evaluated = 0
        
        # Get legal moves
        legal_moves = game.get_legal_moves(self.player)
        
        if not legal_moves:
            raise ValueError("No legal moves available")
        
        # If there's only one legal move, return it immediately
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Apply minimax to each legal move
        best_score = float('-inf') if self.player == game.get_current_player() else float('inf')
        best_move = None
        
        # Randomize move order for more varied gameplay
        random.shuffle(legal_moves)
        
        # Apply alpha-beta pruning
        alpha = float('-inf')
        beta = float('inf')
        
        for move in legal_moves:
            # Create a copy of the game and apply the move
            game_copy = QoridorGame(board_size=game.state.board_size, 
                                   num_walls=game.state.initial_num_walls)
            game_copy.state = self._copy_game_state(game.state)
            game_copy.make_move(move)
            
            # Get score for this move
            if self.player == game.get_current_player():
                # Maximizing player
                score = self._minimax(game_copy, 1, alpha, beta, True)
                
                if score > best_score:
                    best_score = score
                    best_move = move
                    
                alpha = max(alpha, best_score)
            else:
                # Minimizing player
                score = self._minimax(game_copy, 1, alpha, beta, False)
                
                if score < best_score:
                    best_score = score
                    best_move = move
                    
                beta = min(beta, best_score)
            
            # Check time limit
            if self.time_limit and time.time() - self.start_time > self.time_limit:
                print(f"Minimax agent reached time limit after evaluating {self.nodes_evaluated} nodes.")
                break
        
        if best_move is None:
            # Fallback to first legal move
            best_move = legal_moves[0]
            
        print(f"Minimax selected move with score {best_score} after evaluating {self.nodes_evaluated} nodes.")
        return best_move
        
    def _minimax(self, game: QoridorGame, depth: int, alpha: float, beta: float, 
                is_maximizing: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            game: Current game state
            depth: Current depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            is_maximizing: Whether this is a maximizing node
            
        Returns:
            The evaluated score
        """
        self.nodes_evaluated += 1
        
        # Check for termination conditions
        if game.is_game_over():
            winner = game.get_winner()
            if winner == self.player:
                return 1000  # Win
            elif winner is not None:
                return -1000  # Loss
            return 0  # Draw
        
        if depth >= self.max_depth:
            return self.eval_fn(game, self.player)
        
        if self.time_limit and time.time() - self.start_time > self.time_limit:
            return self.eval_fn(game, self.player)
        
        # Get legal moves for current player
        current_player = game.get_current_player()
        legal_moves = game.get_legal_moves(current_player)
        
        # Randomize move order for more varied gameplay and better pruning
        random.shuffle(legal_moves)
        
        if is_maximizing:
            value = float('-inf')
            for move in legal_moves:
                # Create a copy of the game and apply the move
                game_copy = QoridorGame(board_size=game.state.board_size, 
                                       num_walls=game.state.initial_num_walls)
                game_copy.state = self._copy_game_state(game.state)
                game_copy.make_move(move)
                
                # Recursively evaluate
                move_value = self._minimax(game_copy, depth + 1, alpha, beta, False)
                value = max(value, move_value)
                
                if self.use_ab_pruning:
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break  # Beta cutoff
            
            return value
        else:
            value = float('inf')
            for move in legal_moves:
                # Create a copy of the game and apply the move
                game_copy = QoridorGame(board_size=game.state.board_size, 
                                       num_walls=game.state.initial_num_walls)
                game_copy.state = self._copy_game_state(game.state)
                game_copy.make_move(move)
                
                # Recursively evaluate
                move_value = self._minimax(game_copy, depth + 1, alpha, beta, True)
                value = min(value, move_value)
                
                if self.use_ab_pruning:
                    beta = min(beta, value)
                    if alpha >= beta:
                        break  # Alpha cutoff
            
            return value
    
    def _copy_game_state(self, state: GameState) -> GameState:
        """
        Create a deep copy of a GameState.
        
        Args:
            state: The state to copy
            
        Returns:
            A new GameState instance with the same values
        """
        # Serialize and deserialize the state to create a deep copy
        serialized = state.get_serialized_state()
        return GameState.from_serialized(serialized)
    
    def default_evaluate(self, game: QoridorGame, player: int) -> float:
        """
        Default evaluation function for Qoridor.
        
        This function evaluates the state based on:
        1. Shortest path distance to goal for both players
        2. Number of walls left for both players
        3. Player positions and wall configurations
        
        Args:
            game: The game to evaluate
            player: The player to evaluate for
            
        Returns:
            A numerical evaluation score (higher is better for player)
        """
        state = game.state
        board = state.board
        opponent = 3 - player  # 1->2, 2->1
        
        # Calculate distances to goal for both players
        player_path = QoridorRules.find_shortest_path(board, player)
        opponent_path = QoridorRules.find_shortest_path(board, opponent)
        
        if player_path is None:
            return -1000  # No path to goal for player
        
        if opponent_path is None:
            return 1000  # No path to goal for opponent
        
        player_distance = len(player_path) - 1
        opponent_distance = len(opponent_path) - 1
        
        # Calculate distance score: we want our distance to be small and opponent's to be large
        distance_score = opponent_distance - player_distance
        
        # Wall factor: having more walls is an advantage
        player_walls = state.get_walls_left(player)
        opponent_walls = state.get_walls_left(opponent)
        wall_factor = 0.5 * (player_walls - opponent_walls)
        
        # Position factor: being closer to goal row is good
        player_pos = board.get_player_position(player)
        opponent_pos = board.get_player_position(opponent)
        
        # Determine goal rows
        player_goal = board.size - 1 if player == 1 else 0
        opponent_goal = board.size - 1 if opponent == 1 else 0
        
        # Calculate row distance to goal
        player_row_distance = abs(player_pos[0] - player_goal)
        opponent_row_distance = abs(opponent_pos[0] - opponent_goal)
        
        row_score = opponent_row_distance - player_row_distance
        
        # Combine all factors with weights
        score = (
            3.0 * distance_score +  # Path length difference (most important)
            1.0 * wall_factor +     # Wall advantage
            0.5 * row_score         # Row position advantage
        )
        
        return score
    
    def iterative_deepening(self, game: QoridorGame, max_time: float) -> Move:
        """
        Perform iterative deepening search for the best move.
        
        Args:
            game: The current game state
            max_time: Maximum search time in seconds
            
        Returns:
            The best move found
        """
        start_time = time.time()
        depth = 1
        best_move = None
        
        while time.time() - start_time < max_time:
            self.max_depth = depth
            try:
                move = self.select_move(game)
                best_move = move
                depth += 1
            except Exception as e:
                print(f"Error at depth {depth}: {e}")
                break
                
            if time.time() - start_time > max_time:
                break
        
        print(f"Iterative deepening completed at depth {depth-1}")
        return best_move or game.get_legal_moves(self.player)[0]
    
    def get_action(self, observation: Dict[str, Any]) -> int:
        """
        Get action in the format expected by the environment.
        
        This method converts the high-level move selection to a flat action index
        for use with the QoridorEnv.
        
        Args:
            observation: The current environment observation
            
        Returns:
            An integer action index
        """
        # Recreate game state from observation
        board_size = observation['board'].shape[0]
        game = QoridorGame(board_size=board_size, num_walls=10)  # Default 10 walls
        
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
        
        # Select move using minimax
        move = self.select_move(game)
        
        # Convert to flat action index
        action = self._move_to_action(move, board_size)
        
        return action
    
    def _move_to_action(self, move: Move, board_size: int) -> int:
        """
        Convert a Move object to a flat action index for the environment.
        
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
            # Calculate the offset for wall actions
            num_move_actions = board_size * board_size
            is_horizontal = move.wall_orientation == WallOrientation.HORIZONTAL
            
            if is_horizontal:
                base = num_move_actions
                index = row * (board_size - 1) + col
            else:
                base = num_move_actions + (board_size - 1) * (board_size - 1)
                index = row * (board_size - 1) + col
            
            return base + index
