"""
Script to train agents against the minimax opponent.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from qoridor.environment import QoridorEnv
from agents.rule_based.minimax_agent import MinimaxAgent
from agents.learning.dqn_agent import DQNAgent

def train_dqn_against_minimax(num_episodes=1000, board_size=9, num_walls=10, 
                              minimax_depth=2, render_every=100, save_path="models/dqn"):
    """
    Train a DQN agent against a minimax opponent.
    
    Args:
        num_episodes: Number of episodes to train
        board_size: Size of the board
        num_walls: Number of walls per player
        minimax_depth: Depth of the minimax search
        render_every: How often to render the game
        save_path: Where to save the trained model
    """
    # Create agents
    dqn_agent = DQNAgent(player=1, board_size=board_size, num_walls=num_walls)
    minimax_agent = MinimaxAgent(player=2, max_depth=minimax_depth)
    
    # Create environment
    env = QoridorEnv(board_size=board_size, num_walls=num_walls, opponent=minimax_agent.get_action)
    
    # Training statistics
    rewards = []
    win_rates = []
    wins = 0
    
    # Training loop
    for episode in tqdm(range(num_episodes)):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select action
            action = dqn_agent.get_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Update agent
            dqn_agent.update(
                dqn_agent._preprocess_state(state),
                action,
                dqn_agent._preprocess_state(next_state),
                reward,
                done
            )
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # Track statistics
        rewards.append(episode_reward)
        
        # Check if the agent won
        if env.game.get_winner() == 1:
            wins += 1
        
        # Calculate win rate
        win_rate = wins / (episode + 1)
        win_rates.append(win_rate)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes} - Reward: {episode_reward:.2f} - Win Rate: {win_rate:.2%}")
        
        # Render the game
        if (episode + 1) % render_every == 0 or episode == 0:
            print(f"\nRendering episode {episode+1}:")
            env.render()
            print(f"Winner: Player {env.game.get_winner()}")
            print(f"Current win rate: {win_rate:.2%}")
            print(f"Current epsilon: {dqn_agent.epsilon:.4f}")
    
    # Save the agent
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dqn_agent.save(save_path)
        print(f"Agent saved to {save_path}")
    
    # Plot training progress
    plot_training_progress(rewards, win_rates, save_path)
    
    return dqn_agent, rewards, win_rates

def plot_training_progress(rewards, win_rates, save_path=None):
    """
    Plot the training progress.
    
    Args:
        rewards: List of episode rewards
        win_rates: List of win rates
        save_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot rewards
    ax1.plot(rewards, label='Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    
    # Plot win rate
    ax2.plot(win_rates, label='Win Rate')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Win Rate')
    ax2.set_title('Win Rate Against Minimax')
    ax2.legend()
    
    # Smooth the win rate for better visualization
    window_size = min(100, len(win_rates))
    smoothed_win_rates = np.convolve(win_rates, np.ones(window_size)/window_size, mode='valid')
    ax2.plot(range(window_size-1, len(win_rates)), smoothed_win_rates, label='Smoothed Win Rate')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}_training.png")
    
    plt.show()

def evaluate_agent(agent, num_games=100, board_size=9, num_walls=10, minimax_depth=3,
                 render_last=True):
    """
    Evaluate an agent against minimax.
    
    Args:
        agent: The agent to evaluate
        num_games: Number of games to play
        board_size: Size of the board
        num_walls: Number of walls per player
        minimax_depth: Depth of the minimax search
        render_last: Whether to render the last game
        
    Returns:
        Evaluation statistics
    """
    # Create minimax opponent
    minimax_agent = MinimaxAgent(player=2, max_depth=minimax_depth)
    
    # Create environment
    env = QoridorEnv(board_size=board_size, num_walls=num_walls, opponent=minimax_agent.get_action)
    
    # Statistics
    wins = 0
    draws = 0
    avg_reward = 0
    
    for game_idx in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            # Get action from agent
            action = agent.get_action(state)
            
            # Take step in environment
            state, reward, done, info = env.step(action)
            avg_reward += reward
        
        # Check outcome
        winner = env.game.get_winner()
        if winner == 1:
            wins += 1
        elif winner is None:
            draws += 1
        
        # Render last game
        if render_last and game_idx == num_games - 1:
            print("\nFinal game:")
            env.render()
            print(f"Winner: Player {winner}")
    
    # Calculate statistics
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = 1 - win_rate - draw_rate
    avg_reward /= num_games
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Games played: {num_games}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Draw rate: {draw_rate:.2%}")
    print(f"Loss rate: {loss_rate:.2%}")
    print(f"Average reward: {avg_reward:.2f}")
    
    return {
        "win_rate": win_rate,
        "draw_rate": draw_rate,
        "loss_rate": loss_rate,
        "avg_reward": avg_reward
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train agents against minimax")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes")
    parser.add_argument("--board_size", type=int, default=9, help="Board size")
    parser.add_argument("--num_walls", type=int, default=10, help="Number of walls")
    parser.add_argument("--minimax_depth", type=int, default=2, help="Minimax search depth")
    parser.add_argument("--render_every", type=int, default=100, help="Render every N episodes")
    parser.add_argument("--save_path", type=str, default="models/dqn", help="Path to save the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate after training")
    parser.add_argument("--eval_games", type=int, default=50, help="Number of games for evaluation")
    
    args = parser.parse_args()
    
    # Train the agent
    print(f"Training DQN agent against minimax (depth={args.minimax_depth})...")
    dqn_agent, rewards, win_rates = train_dqn_against_minimax(
        num_episodes=args.episodes,
        board_size=args.board_size,
        num_walls=args.num_walls,
        minimax_depth=args.minimax_depth,
        render_every=args.render_every,
        save_path=args.save_path
    )
    
    # Evaluate the agent
    if args.evaluate:
        print("\nEvaluating the trained agent...")
        evaluate_agent(
            dqn_agent,
            num_games=args.eval_games,
            board_size=args.board_size,
            num_walls=args.num_walls,
            minimax_depth=args.minimax_depth
        )