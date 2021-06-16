from collections import deque
from datetime import datetime

import numpy as np
import torch
from unityagents import UnityEnvironment

from dqn_agent import Agent

# Create the environment 
env = UnityEnvironment(file_name="Banana_Windows_x86_64\Banana.exe")

# Get the default Unity brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# Reset the environment and get the state space and action space sizes
env_info = env.reset(train_mode = True)[brain_name]
action_size = brain.vector_action_space_size
state_size = len(env_info.vector_observations[0])

# Init the agent
agent = Agent(state_size = state_size, action_size = action_size, seed=42)


def dqn(num_episodes = 2000, epsilon = 1.0, epsilon_decay = .995, min_epsilon = 0.001, average_every = 100):
    
    filename = datetime.now().strftime('%d-%m-%y-%H_%m_dqnweights.pth')
    
    scores = np.empty(num_episodes)
    score_window = deque(maxlen = average_every)   
    save_score = 0
    
    for episode in range(num_episodes):
        env_info = env.reset(train_mode = True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        
        while True:
            action = agent.act(state,epsilon)                   # Generate action from state and epsilon value
            env_info = env.step(action)[brain_name]             # Perform the action
            next_state = env_info.vector_observations[0]        # Get next state
            reward = env_info.rewards[0]                        # Get reward
            done = env_info.local_done[0]                       # Get done state
            agent.step(state, action, reward, next_state, done) # Update agent
            
            score += reward
            state = next_state
            
            if done:
                scores[episode] = score
                score_window.append(score)
                break
                
        epsilon = max(epsilon*epsilon_decay, min_epsilon)       # Decay epsilon
        
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(score_window)), end="")
        if episode % average_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(score_window)))
        if np.mean(score_window) >= save_score:
            torch.save(agent.qnetwork_local.state_dict(), filename)
            save_score = np.mean(score_window)
                  
    return scores

dqn()