from operator import itemgetter
from typing import Tuple, Optional, Union
import numpy as np
import torch


def combined_shape(
    length: int,
    shape: Optional[Union[int, Tuple[int, ...]]] = None
) -> Tuple[int, ...]:
    if shape is None:
        return (length,)
    elif np.isscalar(shape):
        return (length, shape)
    else:
        return (length, *shape)


class ReplayBuffer():
    """
    A simple FIFO experience replay buffer for SAC agents
    """

    def __init__(self,
                 observation_dimension: Optional[Union[int, Tuple[int, ...]]],
                 action_dimension: Optional[Union[int, Tuple[int, ...]]],
                 size: int
                 ):
        self.observation_buffer = np.zeros(
            combined_shape(size, observation_dimension),
            dtype=np.float32
        )
        self.next_observation_buffer = np.zeros(
            combined_shape(size, observation_dimension),
            dtype=np.float32
        )
        self.action_buffer = np.zeros(
            combined_shape(size, action_dimension),
            dtype=np.float32
        )
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.done_buffer = np.zeros(size, bool)
        self.ptr: int = 0
        self.size: int = 0
        self.max_size: int = size

    def store(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_observation: np.ndarray,
        done: bool
    ):
        self.observation_buffer[self.ptr] = observation
        self.reward_buffer[self.ptr] = reward
        self.next_observation_buffer[self.ptr] = next_observation
        self.action_buffer[self.ptr] = action
        self.done_buffer[self.ptr] = done
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32) -> dict[str, torch.Tensor]:
        idxs: np.array = np.random.randint(
            low=0, high=self.size, size=batch_size)
        batch = dict(
            observations=self.observation_buffer[idxs],
            next_observations=self.next_observation_buffer[idxs],
            actions=self.action_buffer[idxs],
            rewards=self.reward_buffer[idxs],
            dones=self.done_buffer[idxs]
        )
        return {
            k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()
        }

    def save(self, filename: str):
        np.savez_compressed(
            file=filename,
            observation_buffer=self.observation_buffer,
            next_observation_buffer=self.next_observation_buffer,
            action_buffer=self.action_buffer,
            reward_buffer=self.reward_buffer,
            done_buffer=self.done_buffer,
            other=np.array([self.ptr, self.size, self.max_size], dtype=int)
        )

    def load(self, filename: str):
        self.observation_buffer, self.next_observation_buffer, self.action_buffer, self.reward_buffer, self.done_buffer = None, None, None, None, None
        self.observation_buffer, self.next_observation_buffer, self.action_buffer, self.reward_buffer, self.done_buffer, other = itemgetter(
            'observation_buffer',
            'next_observation_buffer',
            'action_buffer',
            'reward_buffer',
            'done_buffer',
            'other'
        )(np.load(filename))
        self.ptr = int(other[0])
        self.size = int(other[1])
        self.max_size = int(other[2])
