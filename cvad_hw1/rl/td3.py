import numpy as np
import torch
import torch.nn.functional as F
from func_timeout import FunctionTimedOut, func_timeout
from utils.rl_utils import generate_noisy_action_tensor

from .off_policy import BaseOffPolicy


class TD3(BaseOffPolicy):
    def _compute_q_loss(self, models, data, exp_r=0.99):
        """Compute q loss for given batch of data."""
        st1, cmd, ac, rew, st2, cmd2, ter = data
        st1, cmd, ac, rew, st2, cmd2, ter = st1.cuda(), cmd.cuda(), ac.cuda(),\
                                     rew.cuda(), st2.cuda(), cmd2.cuda(), ter.cuda()
        # # print(len(st1), st1[:2])
        # print(cmd.shape)
        # print(ac.shape)
        # print(rew.shape)
        # # print(len(st2), st2[:2])
        # print(ter.shape)
        # try:
        #     print(torch.stack(st2).shape)
        # except:
        #     print(st1.shape)
 
        policy, q1, q2 = models 
        q1_pred = q1(st1, ac)
        q2_pred = q2(st1, ac)

        with torch.no_grad():
            ac_pred = policy(st1)
            ac_pred += torch.clamp(torch.randn_like(ac_pred) * 0.1, -0.5, 0.5)
            ac_pred = torch.clamp(ac_pred, -1, 1)

            q1_target = q1(st2, ac_pred)
            q2_target = q2(st2, ac_pred)
            q_final = torch.min(q1_target, q2_target)
            target = rew + exp_r * (1 - ter * 1.0) * q_final

        # MSE loss against Bellman backup
        loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        return loss

    def _compute_p_loss(self, models, data):
        """Compute policy loss for given batch of data."""
        st1, cmd, ac, rew, st2, cmd2, ter = data
        st1, cmd, ac, rew, st2, cmd2, ter = st1.cuda(), cmd.cuda(), ac.cuda(),\
                                     rew.cuda(), st2.cuda(), cmd2.cuda(), ter.cuda()
        policy, q1, q2 = models 
        ac_pred = policy(st1)
        re_pred = q1(st1, ac_pred)
        return -re_pred.mean()

    def _extract_features(self, state):
        """Extract whatever features you wish to give as input to policy and q networks."""
        return torch.Tensor([state['is_junction'] * 1.0,
                          state['tl_state'],
                          state['tl_dist'],
                          state['hazard_dist'],
                          state['lane_dist'],
                          state['lane_angle'],
                          state['route_dist'],
                          state['route_angle'],
                          state['waypoint_dist'],
                          state['waypoint_angle'],
                          state['command']]).float()

    def _take_step(self, state, action):
        try:
            action_dict = {
                "throttle": np.clip(action[0, 0].item(), 0, 1),
                "brake": abs(np.clip(action[0, 0].item(), -1, 0)),
                "steer": np.clip(action[0, 1].item(), -1, 1),
            }
            new_state, reward_dict, is_terminal = func_timeout(
                20, self.env.step, (action_dict,))
        except FunctionTimedOut:
            print("\nEnv.step did not return.")
            raise
        return new_state, reward_dict, is_terminal

    def _collect_data(self, state):
        """Take one step and put data into the replay buffer."""
        features = self._extract_features(state)
        if self.step >= self.config["exploration_steps"]:
            action = self.policy(features, [state["command"]])
            action = generate_noisy_action_tensor(
                action, self.config["action_space"], self.config["policy_noise"], 1.0)
        else:
            action = self._explorer.generate_action(state)
        if self.step <= self.config["augment_steps"]:
            action = self._augmenter.augment_action(action, state)

        # Take step
        new_state, reward_dict, is_terminal = self._take_step(state, action)

        new_features = self._extract_features(state)

        # Prepare everything for storage
        stored_features = torch.Tensor([f.detach().cpu().squeeze(0) for f in features])
        stored_command = state["command"]
        stored_action = action.detach().cpu().squeeze(0)
        stored_reward = torch.tensor([reward_dict["reward"]], dtype=torch.float)
        stored_new_features = torch.Tensor([f.detach().cpu().squeeze(0) for f in new_features])
        stored_new_command = new_state["command"]
        stored_is_terminal = torch.Tensor([bool(is_terminal)])
        
        self._replay_buffer.append(
            (stored_features, stored_command, stored_action, stored_reward,
             stored_new_features, stored_new_command, stored_is_terminal)
        )
        self._visualizer.visualize(new_state, stored_action, reward_dict)
        return reward_dict, new_state, is_terminal
