import numpy as np
class RewardManager():
    """Computes and returns rewards based on states and actions."""
    def __init__(self):
        pass

    def get_reward(self, state, action):
        """Returns the reward as a dictionary. You can include different sub-rewards in the
        dictionary for plotting/logging purposes, but only the 'reward' key is used for the
        actual RL algorithm, which is generated from the sum of all other rewards."""
        reward_dict = {}
        # Your code here
        reward_dict['collision'] = state['collision'] * -10
        reward_dict['speed'] = 1 - np.abs(state['optimal_speed'] - state['speed']) / 100

        if state['command'] == 3: # lane follow
            reward_dict['steer'] = -np.abs(state['lane_angle'] - action['steer'])
        
        # if state['command'] == 2: # staright
        #     reward_dict['steer'] = 1 - np.abs(action['steer'])

        # stopping if optimal speed is less than the speed of the car
        if state['optimal_speed'] < state['speed']:
            # time to stop linearly reagrding of the action
            diff = state['speed'] - state['optimal_speed']
            reward_dict['stopping'] = 1 - np.abs(action['brake'] - diff/100)
        
        # acceleration
        if state['optimal_speed'] > state['speed']:
            # time to stop linearly reagrding of the action
            diff = state['optimal_speed'] - state['speed']
            reward_dict['speeding'] = 1 - np.abs(action['throttle'] - diff/100) 

        # stear (angle should be aligned with waypoint)
        reward['way_steer'] = -np.abs(state['waypoint_angle'] - action['steer'])

        # add reward for suriving one more step
        reward['step'] = 0.5

        # Your code here
        reward = 0.0
        for val in reward_dict.values():
            reward += val
        reward_dict["reward"] = reward
        return reward_dict
