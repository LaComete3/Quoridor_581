from typing import List, Union
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from dqn import state_to_vector
from qoridor.environment import QoridorEnv
from dqnAgent import DQNAgent
from agents.random_agent import RandomAgent


def train_dqn_agents(
    agent1: DQNAgent,
    agent2: RandomAgent,
    env: QoridorEnv,
    episodes: int,
    gamma: float,
    batch_size: int,
    target_q_network_sync_period: int
) -> list:
        episode_rewards_agent1 = []  
        episode_rewards_agent2 = []
        for episode in tqdm(range(1, episodes)):
            env.reset()
            done = False
            state = env.game.state
            state_vector = state_to_vector(state)
            episode_reward_agent1 = 0.0  
            episode_reward_agent2 = 0.0

            while not done:
                current_player = env.game.state.get_observation()['current_player']

                if current_player == 1:
                    agent = agent1
                    opponent = agent2
                    episode_reward = episode_reward_agent1
                else:
                    agent = agent2
                    opponent = agent1
                    episode_reward = episode_reward_agent2
                
                if isinstance(agent, DQNAgent):
                    action = agent.epsilon_greedy(state)
                elif isinstance(agent, RandomAgent):
                    move = agent.get_move(env.game)
                    action = env._move_to_action(move)

                _ , reward, done, info = env.step(action)
                #print(f"Action: {action}, Reward: {reward}, Done: {done}")
                if current_player == 1:
                    episode_reward_agent1 += float(reward)
                else:
                    episode_reward_agent2 += float(reward)
                next_state = env.game.state
                next_state_vector = state_to_vector(next_state)
                
                if isinstance(agent, DQNAgent):
                        agent.replay_buffer.add(state_vector, action, float(reward), next_state_vector, done)
                        state_vector = next_state_vector

                        agent.train()

                        if episode % target_q_network_sync_period == 0:
                                agent.target_model.load_state_dict(agent.model.state_dict())

                env.game.state.current_player= 3-current_player
            
                episode_reward += float(reward)

                if done:
                    break

                state = next_state
                if isinstance(agent1, DQNAgent):
                    agent1.epsilon_greedy.decay_epsilon()
                if isinstance(agent2, DQNAgent):
                    agent2.epsilon_greedy.decay_epsilon()
            episode_rewards_agent1.append(episode_reward_agent1)
            episode_rewards_agent2.append(episode_reward_agent2)
            

        return episode_rewards_agent1, episode_rewards_agent2



env1 = QoridorEnv(5, 3)

observation_dim = 7 + (env1.board_size-1)**2 * 2
num_episodes = 500
gamma = 0.9
batch_size = 128
target_q_network_sync_period = 30


train_rewards = []  
train_indices = []  
episode_indices = [] 

NUMBER_OF_TRAININGS = 5
for train_index in range(NUMBER_OF_TRAININGS):
    print(f"Start of the training {train_index + 1}")
    #creating agent
    agent1_dqn = DQNAgent(state_size=observation_dim, action_size=env1.action_space_size)
    agent2_dqn = RandomAgent() 
    episode_rewards_agent1, episode_rewards_agent2 = train_dqn_agents(agent1_dqn, agent2_dqn, env1, num_episodes, gamma, batch_size, target_q_network_sync_period)
    print("Episode rewar :", episode_rewards_agent1)

    train_rewards.extend(episode_rewards_agent1)
    episode_indices.extend(range(len(episode_rewards_agent1)))
    train_indices.extend([train_index for _ in episode_rewards_agent1])

    env1.close()


dqn2_trains_result_df = pd.DataFrame({
    "num_episodes": episode_indices,
    "mean_final_episode_reward": train_rewards,
    "training_index": train_indices,
    "agent": "DQN"
})


unique_trainings = dqn2_trains_result_df['training_index'].nunique()
palette = sns.color_palette("tab10", unique_trainings)  


g = sns.relplot(
    x="num_episodes", 
    y="mean_final_episode_reward", 
    kind="line", 
    hue="training_index",  
    palette=palette,  
    estimator=None, 
    units="training_index",  
    data=dqn2_trains_result_df,
    height=7, 
    aspect=2, 
    alpha=0.7
)

plt.title('Reward of Agent 1 after multiple episodes')
plt.xlabel('episode')
plt.ylabel('Final reward')

plt.show()
