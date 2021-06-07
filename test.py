import torch
from unityagents import UnityEnvironment
from dqn_agent import Agent

env = UnityEnvironment(file_name="Banana_Windows_x86_64\Banana.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=False)[brain_name] # reset the environment
action_size = brain.vector_action_space_size
state = env_info.vector_observations[0]            # get the current state
state_size = len(state)

agent = Agent(state_size = state_size, action_size = action_size, seed = 42, train_mode = False)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

score = 0                                          # initialize the score
while True:
    action = agent.act(state)        # select an action
    env_info = env.step(action)[brain_name]        # send the action to the environment
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    done = env_info.local_done[0]                  # see if episode has finished
    score += reward                                # update the score
    state = next_state                             # roll over the state to next time step
    if done:                                       # exit loop if episode finished
        break
    
print('Score: ', score)
env.close()