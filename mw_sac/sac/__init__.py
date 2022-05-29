import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Iterable
from torch.distributions.normal import Normal
import itertools
from config import POLYAK, ALPHA

class SoftActorCritic():
    def __init__(
        self,
        create_network: Callable[[], nn.Module],
        create_mean_layer: Callable[[], nn.Module],
        create_log_stdev_layer: Callable[[], nn.Module],
        create_q_network: Callable[[], nn.Module],
        device: torch.device,
        action_space_shape: Iterable[int],
        lr: float
    ):
        self.device = device
        self.net = create_network().to(device=device)
        self.net_mean_layer = create_mean_layer().to(device=device)
        self.net_log_stdev_layer = create_log_stdev_layer().to(device=device)
        self.q1, self.q2 = \
            create_q_network().to(device=device), \
            create_q_network().to(device=device)
        self.q1_target, self.q2_target = \
            create_q_network().to(device=device), \
            create_q_network().to(device=device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # init net optimizer and its lr scheduler
        # lr schedulers: TODO
        self.optim = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.q_parameters = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optim = torch.optim.Adam(self.q_parameters, lr)

        #self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr)
        #self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr)
        self.lr = lr

    def forward(self, observation: torch.Tensor, deterministic: bool = True, with_logprob=True):
        """
        Runs an observed state through the actor
        Returns an tuple of both the selected action and the logarithmic probability of actions
        """
        net_output = self.net(observation)
        mean = self.net_mean_layer(net_output)
        if deterministic:
            pi_action = mean
        else:
            LOG_STDEV_MAX = 2
            LOG_STDEV_MIN = -20
            log_stdev = self.net_log_stdev_layer(net_output)
            log_stdev = torch.clamp(log_stdev, LOG_STDEV_MIN, LOG_STDEV_MAX)
            stdev = torch.exp(log_stdev)
            pi_distribution = Normal(mean, stdev)
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None
        
        pi_action = torch.tanh(pi_action)
        return pi_action, logp_pi

    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        with torch.no_grad():
            a, _ = self.forward(observation, deterministic, False)
            return a


    def calc_q(self, state, action):
        """
        Run the state and action through the critic networks
        """
        q1_prediction = self.q1(state, action)
        q2_prediction = self.q2(state, action)
        return torch.squeeze(q1_prediction, -1), torch.squeeze(q2_prediction, -1)


    def compute_loss_q(self, data: dict):
        observations = data['observations'].to(self.device)
        next_observations = data['next_observations'].to(self.device)
        actions = data['actions'].to(self.device)
        rewards = data['rewards'].to(self.device)
        dones = data['dones'].to(self.device)

        q1_result = self.q1(observations, actions)
        q2_result = self.q2(observations, actions)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Actions from current policy
            actions2, logp_a2 = self.forward(observations, False, True)

            # Target Q-values
            q1_pi_targ = self.q1_target(next_observations, actions2)
            q2_pi_targ = self.q2_target(next_observations, actions2)
            q_pi_targ = torch.min(q1_pi_targ, q1_pi_targ)
            backup = rewards + self.lr * (1 - dones) * (q_pi_targ - ALPHA * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1_result - backup) ** 2).mean()
        loss_q2 = ((q2_result - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2
        q_info = dict(
            q1_vals = q1_result.detach().cpu().numpy(),
            q2_vals = q2_result.detach().cpu().numpy()
        )
        return loss_q, q_info

    def compute_loss(self, data: dict):
        observations = data['observations'].to(self.device)
        next_observations = data['next_observations'].to(self.device)
        actions = data['actions'].to(self.device)
        rewards = data['rewards'].to(self.device)
        dones = data['dones'].to(self.device)

        pi, logp_pi = self.forward(observations, False, True)
        q1_pi = self.q1(observations, pi)
        q2_pi = self.q2(observations, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy regularized policy loss
        loss_pi = (ALPHA * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(
            logp_pi = logp_pi.detach().cpu().numpy()
        )

        return loss_pi, pi_info

    def update(self, data):
        # Update the q networks
        self.q_optim.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optim.step()

        for p in self.q_parameters:
            p.requires_grad = False

        # Update actor network
        self.optim.zero_grad()
        loss, info = self.compute_loss(data)
        loss.backward()
        self.optim.step()

        for p in self.q_parameters:
            p.requires_grad = True
        
        # Update q target networks by polyak averaging
        with torch.no_grad():
            for p, p_targ in zip(self.q1.parameters(), self.q1_target.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)
            for p, p_targ in zip(self.q2.parameters(), self.q2_target.parameters()):
                p_targ.data.mul_(POLYAK)
                p_targ.data.add_((1 - POLYAK) * p.data)

        return loss_q.item(), q_info, info


    def state_dict(self):
        return {
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.q1.load_state_dict(state_dict['q1'])
        self.q2.load_state_dict(state_dict['q2'])
        self.q1_target.load_state_dict(state_dict['q1_target'])
        self.q2_target.load_state_dict(state_dict['q2_target'])