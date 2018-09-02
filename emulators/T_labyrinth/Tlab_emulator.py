from ..env_T_2 import make_game, T_lab_observation, T_lab_actions, print_obs

from ..environment import BaseEnvironment

import numpy as np
import operator

def convert_obs(obs):
    obs, info = obs
    keys = sorted(info.keys())
    state = np.stack([info[k] for k in keys])
    return state.astype(np.uint8)

class TLabyrinthEmulator(BaseEnvironment):



    def __init__(self, actor_id, args):
        self.randomness = True
        self.reward_location = np.random.choice([0,1]) #0 if np.random.rand() < 0.5 else 1
        self.visualize = getattr(args,'visualize', False)
        self.legal_actions = [0, 1, 2] #['up', 'left', 'right', 'noop']
        #print(self.legal_actions)
        self.noop = 'pass'
        self.id = actor_id
        self.length_int = [9,10]
        self.game, self.resulting_length = make_game(False, self.reward_location, self.length_int)
        obs_t, r_t, discount_t = self.game.its_showtime()
        obs_t = convert_obs(obs_t)
        self.observation_shape = obs_t.shape

#ДЛИНА ДОЛЖНА ПЕРЕДАВАТЬСЯ ИЗВНЕ. ТУТ НЕ ДОЛЖНО БЫТЬ НАЧАЛЬНОЙ ДЛИНЫ.

    def set_length(self, length_interval):
        self.length_int = length_interval
        return self.length_int



    def reset(self):
        """Starts a new episode and returns its initial state"""
        self.reward_location = np.random.choice([0,1]) #0 if np.random.rand() < 0.5 else 1

        self.game, self.resulting_length = make_game(False, self.reward_location, self.length_int)
        obs_t, r_t, discount_t = self.game.its_showtime()
        obs = convert_obs(obs_t)
        return obs, {'length':self.resulting_length}

    def next(self, action):

        """
        Performs the given action.
        Returns the next state, reward, and terminal signal
        """
        act = [i for i, x in enumerate(action) if x]
        if not self.game.game_over:
            obs, reward, discount = self.game.play(act[0])
            if self.visualize:
                act_names = ['up', 'left', 'right']
                print('action={}, r={}, is_done={}'.format(act_names[act[0]], reward, discount!=1.0))
                print_obs(obs)
        termination = 1-discount
        return convert_obs(obs), reward, termination, {'length':self.resulting_length}


    def get_legal_actions(self):
        #self.legal_actions = T_lab_actions().shape
        return self.legal_actions

    def get_noop(self):
        #self.noop = 'pass'
        return self.noop

    def on_new_frame(self, frame):
        pass

    def close(self):
        pass
