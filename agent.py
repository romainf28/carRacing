from simulate_experiment import SimulateExperiment
import numpy as np
import itertools


class DQN:

    # FIXME : implement the deep Q network

    def __init__(self, env, img_size=(96, 96), num_frame_stack=4,
                 experience_capacity=int(1e5), action_map=None):
        self.experience_history = SimulateExperiment(
            num_frame_stack=num_frame_stack, capacity=experience_capacity, img_size=img_size)
        if action_map is not None:
            self.nb_actions = len(action_map)
        else:
            self.nb_actions = env.action_space.n

    def get_random_action(self):
        return np.random.choice(self.nb_actions)


class CarRacingDQN(DQN):
    def __init__(self, max_negative_reward=100, **kwargs):
        action_map = np.array(
            [k for k in itertools.product([-1, 0-1], [1, 0], [0.2, 0])])
        kwargs['render'] = True
        super().__init__(action_map=action_map, img_size=(96, 96), **kwargs)

        self.gas_actions = np.array(
            [a[1] == 1 and a[2] == 0 for a in action_map])
        self.break_actions = np.array([a[2] == 1 for a in action_map])
        self.nb_gas_actions = self.gas_actions.sum()
        self.neg_reward_counter = 0
        self.max_neg_reward = max_negative_reward
