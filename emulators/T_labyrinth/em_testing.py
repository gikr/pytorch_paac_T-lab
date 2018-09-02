import argparse
import logging
import os
import signal
import sys
import torch
from Tlab_emulator import TLabyrinthEmulator as T


def dummy_episode():
	
    game = T.get_initial_state(T)
    #print(game)                                           ##
    action_keys = ['up', 'down', 'left', 'right', 'noop'] ##
    for t in range(1,101): 
        a_t = None                                        ##
        while a_t not in action_keys:                     ##
            a_t = input("Choose one of the following actions: {}:\n".format(action_keys)) ##
        
        action = action_keys.index(a_t)
        zero_to_one = [0.,0.,0.,0.,0.]
        zero_to_one[action] = 1.
        obs_t, r_t, term_t = T.next(T, zero_to_one)
        #obs_t, r_t, discount_t = game.play(action_keys.index(a_t))
        print('r =', r_t, 'gamma = ', term_t)
        #obs_t, r_t = game.play(action_keys.index(a_t))
        
        if term_t == 1:
           game = T.get_initial_state(T)
           
			
        print('===========  Step #{}  ==========:'.format(t))
    
    print('Done!')



if __name__ == '__main__':
  
  dummy_episode()
