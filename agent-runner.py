"""
Runner script for testing agents against minimax.
"""

import time
import argparse
import numpy as np
from qoridor.game import QoridorGame
from qoridor.environment import QoridorEnv
from qoridor.visualization import QoridorVisualizer, render_move_sequence
from agents.rule_based.minimax_agent import MinimaxAgent
from qoridor.board import WallOrientation
from qoridor.move import MoveType

def play_game(agent1, agent2, board_size=9, num_walls=10, render=True, delay=0.5, 
             max_steps=1000, save_gif=None):
    """
    Play a game between two agents.
    
    Args:
        agent1: First agent (player 1)
        agent2: Second agent (player 2)
        board_size: Size of the board
        num_walls: Number of walls per player
        render: Whether to render the game
        delay: Delay between moves when rendering
        max_steps: Maximum number of steps
        save_gif: Path to save GIF animation
        
    Returns:
        A tuple of (winner, move_history, steps)
    """
    # Create environment
    env = QoridorEnv(board_size=board_size, num_walls=num_walls)
    obs = env.reset()
    
    move_history = []
    steps = 0
    done = False
    
    # Set up visualization
    visualizer = QoridorVisualizer()
    
    while not done and steps < max_steps:
        # Determine current agent
        current_player = env.game.get_current_player()
        current_agent = agent1 if current_player == 1 else agent2
        
        # Get action from agent
        start_time = time.time()
        action = current_agent.get_action(obs)
        decision_time = time.time() - start_time
        
        # Apply action
        next_obs, reward, done, info = env.step(action)
        
        # Store move information
        move = env.game.move_log[-1] if env.game.move_log else None
        if move:
            move_info = {
                "step": steps,
                "player": current_player,
                "move": move,
                "time": decision_time,
                "reward": reward
            }
            move_history.append(move_info)
        
        # Update observation
        obs = next_obs
        steps += 1
        
        # Render
        if render and not save_gif:
            env.render()
            time.sleep(delay)
            
    # Get winner
    winner = env.game.get_winner()
    
    # Render final state if not saving gif
    if render and not save_gif:
        env.render()
        print(f"Game over in {steps} steps. Winner: Player {winner}")
    
    # Generate animation if requested
    if save_gif:
        render_move_sequence(env.game, move_history, delay=delay, save_gif=save_gif)
    
    return winner, move_history, steps

def evaluate_agents(agent1, agent2, num_games=10, board_size=9, num_walls=10, render_last=True):
    """
    Evaluate agents over multiple games.
    
    Args:
        agent1: First agent (player 1)
        agent2: Second agent (player 2)
        num_games: Number of games to play
        board_size: Size of the board
        num_walls: Number of walls per player
        render_last: Whether to render the last game
        
    Returns:
        Evaluation statistics
    """
    wins = {1: 0, 2: 0, None: 0}
    total_steps = []
    decision_times = {1: [], 2: []}
    
    for game_idx in range(num_games):
        print(f"Playing game {game_idx+1}/{num_games}...")
        
        render = render_last and (game_idx == num_games - 1)
        winner, move_history, steps = play_game(
            agent1, agent2, 
            board_size=board_size, 
            num_walls=num_walls,
            render=render,
            delay=0.1 if render else 0
        )
        
        wins[winner] += 1
        total_steps.append(steps)
        
        # Collect decision times
        for move in move_history:
            decision_times[move["player"]].append(move["time"])
    
    # Calculate statistics
    stats = {
        "wins": wins,
        "win_rate_agent1": wins[1] / num_games,
        "win_rate_agent2": wins[2] / num_games,
        "draw_rate": wins[None] / num_games,
        "avg_steps": np.mean(total_steps),
        "avg_decision_time_agent1": np.mean(decision_times[1]),
        "avg_decision_time_agent2": np.mean(decision_times[2])
    }
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Games played: {num_games}")
    print(f"Player 1 wins: {wins[1]} ({stats['win_rate_agent1']:.2%})")
    print(f"Player 2 wins: {wins[2]} ({stats['win_rate_agent2']:.2%})")
    print(f"Draws: {wins[None]} ({stats['draw_rate']:.2%})")
    print(f"Average game length: {stats['avg_steps']:.1f} steps")
    print(f"Average decision time - Player 1: {stats['avg_decision_time_agent1']*1000:.1f} ms")
    print(f"Average decision time - Player 2: {stats['avg_decision_time_agent2']*1000:.1f} ms")
    
    return stats

def create_minimax_agent(player=2, depth=3, time_limit=None, use_ab=True):
    """Create a minimax agent with the specified parameters."""
    return MinimaxAgent(
        player=player,
        max_depth=depth,
        time_limit=time_limit,
        use_ab_pruning=use_ab
    )

def random_agent(observation):
    """A simple random agent for testing."""
    # Get board size
    board_size = observation['board'].shape[0]
    
    # Create game state from observation
    game = QoridorGame(board_size=board_size, num_walls=10)
    
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
    
    # Get legal moves and pick one randomly
    legal_actions = []
    for move in game.get_legal_moves():
        # Convert Move to action index
        row, col = move.position
        
        if move.move_type == MoveType.PAWN_MOVE:
            action = row * board_size + col
        else:  # WALL_PLACEMENT
            num_move_actions = board_size * board_size
            is_horizontal = move.wall_orientation == WallOrientation.HORIZONTAL
            
            if is_horizontal:
                base = num_move_actions
                index = row * (board_size - 1) + col
            else:
                base = num_move_actions + (board_size - 1) * (board_size - 1)
                index = row * (board_size - 1) + col
                
            action = base + index
            
        legal_actions.append(action)
    
    return np.random.choice(legal_actions)


class RandomAgent:
    """Simple random agent that selects moves uniformly at random."""
    
    def __init__(self, player=1):
        self.player = player
        
    def get_action(self, observation):
        """Select a random legal action."""
        from qoridor.move import MoveType
        from qoridor.board import WallOrientation
        
        # Get board size
        board_size = observation['board'].shape[0]
        
        # Create game state from observation
        game = QoridorGame(board_size=board_size, num_walls=10)
        
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
        
        # Get legal moves and pick one randomly
        legal_moves = game.get_legal_moves()
        selected_move = np.random.choice(legal_moves)
        
        # Convert Move to action index
        row, col = selected_move.position
        
        if selected_move.move_type == MoveType.PAWN_MOVE:
            action = row * board_size + col
        else:  # WALL_PLACEMENT
            num_move_actions = board_size * board_size
            is_horizontal = selected_move.wall_orientation == WallOrientation.HORIZONTAL
            
            if is_horizontal:
                base = num_move_actions
                index = row * (board_size - 1) + col
            else:
                base = num_move_actions + (board_size - 1) * (board_size - 1)
                index = row * (board_size - 1) + col
                
            action = base + index
            
        return action


class PathRushAgent:
    """A simple agent that always moves toward the goal along the shortest path."""
    
    def __init__(self, player=1):
        self.player = player
        
    def get_action(self, observation):
        """Select the move that advances along the shortest path to the goal."""
        from qoridor.rules import QoridorRules
        from qoridor.move import MoveType
        
        # Get board size
        board_size = observation['board'].shape[0]
        
        # Create game state from observation
        game = QoridorGame(board_size=board_size, num_walls=3)
        
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
        
        # Get the shortest path to goal
        current_player = game.get_current_player()
        shortest_path = QoridorRules.find_shortest_path(game, current_player)
        
        # If there's a path, move along it
        if shortest_path and len(shortest_path) > 1:
            next_pos = shortest_path[1]  # Next position in the path
            
            # Find the move that matches this position
            for move in game.get_legal_moves():
                if move.move_type == MoveType.PAWN_MOVE and move.position == next_pos:
                    # Convert Move to action index
                    row, col = move.position
                    return row * board_size + col
        
        print("No path found. Fallback to random move.")
        # Fallback to random move if no path or other issue
        return RandomAgent(player=self.player).get_action(observation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Qoridor Agent Runner")
    parser.add_argument("--board_size", type=int, default=5, help="Board size")
    parser.add_argument("--num_walls", type=int, default=3, help="Number of walls per player")
    parser.add_argument("--num_games", type=int, default=20, help="Number of games to play")
    parser.add_argument("--minimax_depth", type=int, default=3, help="Minimax search depth")
    parser.add_argument("--time_limit", type=float, default=5.0, help="Time limit per move in seconds")
    parser.add_argument("--save_gif", type=str, help="Save the last game as an animated GIF")
    
    args = parser.parse_args()
    
    # Create agents
    random_agent_obj = RandomAgent(player=2)
    path_rush_agent = PathRushAgent(player=2)
    minimax_agent = create_minimax_agent(
        player=1, 
        depth=args.minimax_depth,
        time_limit=args.time_limit
    )
    
    # # Evaluate a random agent against minimax
    # print("\n=== Random Agent vs Minimax ===")
    # evaluate_agents( 
    #     minimax_agent,
    #     random_agent_obj,
    #     num_games=args.num_games,
    #     board_size=args.board_size,
    #     num_walls=args.num_walls
    # )
    
    # Evaluate a path rush agent against minimax
    print("\n=== Path Rush Agent vs Minimax ===")
    evaluate_agents(
        minimax_agent,
        minimax_agent, 
        num_games=args.num_games,
        board_size=args.board_size,
        num_walls=args.num_walls
    )
    
    # Play a sample game and save as GIF if requested
    if args.save_gif:
        print(f"\nPlaying a sample game and saving as {args.save_gif}...")
        play_game(
            minimax_agent, 
            minimax_agent,
            board_size=args.board_size,
            num_walls=args.num_walls,
            render=False,
            save_gif=args.save_gif
        )
        print("Done!")
