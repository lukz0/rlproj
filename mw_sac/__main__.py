from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
import queue
from sac import SoftActorCritic
from replay_buffer import ReplayBuffer
from environment import Environment
import environment
from config import INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, STEPS_PER_EPISODE, EPISODES, UPDATE_AFTER, UPDATE_EVERY, REPLAY_BUFFER_FILENAME, SAC_FILENAME, REPLAY_BUFFER_SIZE, START_FROM_EPISODE, BATCH_SIZE, LOGFILE, EVAL

DEVICE = torch.device('cuda')

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.leaky_relu(self.l1(x), 1/4)
        x = nn.functional.leaky_relu(self.l2(x), 1/4)
        x = nn.functional.leaky_relu(self.l3(x), 1/4)
        return x


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(INPUT_SIZE + OUTPUT_SIZE, HIDDEN_SIZE)
        self.l2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.l4 = nn.Linear(HIDDEN_SIZE, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = nn.functional.leaky_relu(self.l1(torch.cat([state, action], dim=-1)), 1/4)
        x = nn.functional.leaky_relu(self.l2(x), 1/4)
        x = nn.functional.leaky_relu(self.l3(x), 1/4)
        x = torch.squeeze(nn.functional.leaky_relu(self.l4(x), 1/4), -1)
        return x


sac = SoftActorCritic(
    lambda: Actor(),
    lambda: nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    lambda: nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
    lambda: Critic(),
    DEVICE,
    (2,),
    1e-3
)

replay_buffer = ReplayBuffer(INPUT_SIZE, 2, REPLAY_BUFFER_SIZE)
env = Environment()

def get_action(sac: SoftActorCritic, observation: torch.Tensor, deterministic=False):
    action: torch.Tensor = sac.act(observation, deterministic)
    return action.detach().cpu()[0].numpy()

def update(sac: SoftActorCritic, batch):
    sac.update(batch)

# def go_forward(sac: SoftActorCritic, replay_buffer: ReplayBuffer, env: Environment):
#     env.reset_car()
#     print('Car reset')
#     for i in range(5):
#         print('i:', i)
#         q = env.step(False)
#         response: environment.Response = q.get(True)
#         new_observation = np.concatenate((response.state, response.old_state)).flatten()
#         action = get_action(sac, torch.from_numpy(new_observation)[None, :].to(DEVICE), False)
#         response.action_queue.put(np.array([-1.0, 0.5], dtype=float))

#     old_observation, current_observation, new_observation = None, None, None
#     for i2 in range(100):
#         print('i2:', i2)
#         q = env.step(False)
#         response: environment.Response = q.get(True)
#         new_observation = np.concatenate((response.state, response.old_state)).flatten()
#         action = get_action(sac, torch.from_numpy(new_observation)[None, :].to(DEVICE), False)
#         response.action_queue.put(np.array([-1.0, 0.5], dtype=float))

#         old_observation = current_observation
#         current_observation = new_observation

#         if old_observation is not None:
#             replay_buffer.store(
#                 old_observation,
#                 action,
#                 response.reward,
#                 current_observation,
#                 response.done # In contrast to sinningup's implementation we will never hit a time horizon
#             )

if not (START_FROM_EPISODE == 0):
    replay_buffer.load(REPLAY_BUFFER_FILENAME)
    sac.load_state_dict(torch.load(SAC_FILENAME))

if EVAL:
    episode_rewards = 0
    old_observation, current_observation, new_observation = None, None, None
    for step in range(STEPS_PER_EPISODE):
        print("Step: ", step)
        q = env.step(step == 0)

        response: environment.Response = q.get(True)
        new_observation = np.concatenate((response.state, response.old_state)).flatten()

        action = get_action(sac, torch.from_numpy(new_observation)[None, :].to(DEVICE), True)
        response.action_queue.put(action)

        old_observation = current_observation
        current_observation = new_observation

        episode_rewards += response.reward

        if response.done:
            break
    with open(LOGFILE, mode='a') as f: f.write(f"Evaluation Episode: {episode}\n")
    with open(LOGFILE, mode='a') as f:
        f.write(f"Rewards: {episode_rewards}\n")
        f.write(f"Progress: {env.get_progress()}\n\n")
    exit()

for episode in range(START_FROM_EPISODE ,EPISODES):
    # TODO: change this
    episode_rewards = 0
    old_observation, current_observation, new_observation = None, None, None
    steps_without_rewards = 0

    for step in range(STEPS_PER_EPISODE):
        print("Step: ", step)
        t = episode * STEPS_PER_EPISODE + step

        q = env.step(step == 0)
        response: environment.Response = q.get(True)
        new_observation = np.concatenate((response.state, response.old_state)).flatten()

        action = get_action(sac, torch.from_numpy(new_observation)[None, :].to(DEVICE), False)
        response.action_queue.put(action)

        old_observation = current_observation
        current_observation = new_observation

        episode_rewards += response.reward

        # if old_obsLOGFILE = 'log4.txt'ervation is not None:
        #     if response.reward <= 0: steps_without_rewards += 1
        #     if response.reward <= 0 and steps_without_rewards >= 300:
        #         go_forward(sac, replay_buffer, env)
        #         steps_without_rewards = 0
        #         old_observation, current_observation, new_observation = None, None, None
        #     else:
        #         print('Reward:', response.reward)
        #         replay_buffer.store(
        #             old_observation,
        #             action,
        #             response.reward,
        #             current_observation,
        #             response.done # In contrast to sinningup's implementation we will never hit a time horizon
        #         )
        if old_observation is not None:
            #print('Reward:', response.reward)
            replay_buffer.store(
                old_observation,
                action,
                response.reward,
                current_observation,
                response.done # In contrast to sinningup's implementation we will never hit a time horizon
            )

        if response.done:
            break

        if t >= UPDATE_AFTER and t % UPDATE_EVERY == 0:
            # Pause
            env.toggle_pause()
            for j in range(UPDATE_EVERY):
                batch = replay_buffer.sample_batch(batch_size=BATCH_SIZE)
                update(sac, batch)
            # Unpause
            env.toggle_pause()
    # TODO log data
    with open(LOGFILE, mode='a') as f: f.write(f"Episode: {episode}\n")
    replay_buffer.save(REPLAY_BUFFER_FILENAME)
    torch.save(sac.state_dict(), SAC_FILENAME)
    with open(LOGFILE, mode='a') as f:
        f.write(f"Rewards: {episode_rewards}\n")
        f.write(f"Progress: {env.get_progress()}\n\n")
    episode_rewards = 0
