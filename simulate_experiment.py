import numpy as np


class SimulateExperiment:
    def __init__(self, num_frame_stack=4, capacity=int(1e5), img_size=(96, 96)):

        self.num_frame_stack = num_frame_stack
        self.capacity = capacity
        self.img_size = img_size
        self.count = 0
        self.frame_window = None
        self.init_cache_capicty()
        self.new_episode_expected = True

    def init_cache_capicty(self):
        '''Initialize caching capacity'''
        self.rewards = np.zeros(self.capacity, dtype='float32')
        self.previous_states = - \
            np.ones((self.capacity, self.num_frame_stack), dtype='int32')
        self.next_states = - \
            np.ones((self.capacity, self.num_frame_stack), dtype='int32')
        self.terminated = - np.ones(self.capacity, dtype='int32')
        self.actions = - np.ones(self.capacity, dtype='int32')
        self.cache_size = self.capacity + 2*self.num_frame_stack + 1
        self.cache = - np.ones((self.cache_size,) +
                               self.img_size, dtype='float32')

    def current_state(self):
        ''' Returns current state '''
        assert self.frame_window is not None, "Please run an episode first"
        return self.cache[self.frame_window]

    def store_step_in_cache(self, frame, action, terminated, reward):
        '''Stores a tuple (s,a,r,s') in the frame cache'''
        assert self.frame_window is not None, "Please run an episode first"
        self.count += 1
        frame_index = self.count % self.cache_size
        step = (self.count - 1) % self.capacity

        self.previous_states[step] = self.frame_window
        self.frame_window = np.append(self.frame_window[1:], frame_index)
        self.next_states[step] = self.frame_window

        self.cache[frame_index] = frame
        self.actions[step] = action
        self.terminated[step] = terminated
        self.rewards[step] = reward
        if terminated:
            self.new_episode_expected = True

    def start_episode(self, frame):
        ''' Start a new episode '''
        assert self.new_episode_expected, "last episode is not finished yet, please wait"
        frame_index = self.count % self.cache_size
        self.frame_window = np.array([frame_index]*self.cache_size)
        self.cache[frame_index] = frame
        self.new_episode_expected = False

    def sample(self, batch_size):
        '''Random sampling from the frames cache'''

        upper_bound = min(self.capacity, self.count)
        batch_index = np.random.randint(upper_bound, size=batch_size)

        previous_frames = self.cache[self.previous_states[batch_index]]
        next_frames = self.cache[self.next_states[batch_index]]

        return {
            'previous_states': previous_frames,
            'actions': self.actions[batch_index],
            'rewards': self.rewards[batch_index],
            'next_states': next_frames,
            'done_mask': self.terminated[batch_index]
        }
