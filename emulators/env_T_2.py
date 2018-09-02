from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
import numpy as np
import pandas as pd
import time
import random

from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab.prefab_parts import drapes as prefab_drapes


GAME_ART = [
    ['#########',
     '#L     R#',
     '@#@# #@#@',
     '#@#@ @#@#',  # Testing environment v1.plus we need to show where we star our game
     '##@# #@##',
     '@### ###@',
     '@@## ##@@',
     '+#@@ @@##',
     '#@@# H@@#',
     '@##@A@##@',
     '#########'],

    ['#########',
     '#L     R#',
     '@#@# #@#@',
     '#@#@ @#@#',  # Testing environment v2
     '##@# #@##',
     '@### ###@',
     '@@## ##@@',
     '+#@@ @@##',
     '#@@H #@@#',
     '@##@A@##@',
     '#########']
]

chrs = ['#','@']

AGENT_CHR = 'A'
GOAL_CHR1 = 'L'
GOAL_CHR2 = 'R'
HINT_CHR = 'H'

#MOVEMENT_REWARD = -0.0001

#GOAL_REWARD = 1
#HINT_REWARD = 20

def game_art_function(width, leng1, leng2, reward_location):

    length = random.randint(leng1, leng2)
    matrix = ['#########',
              '#L     R#']
    matrix += [''.join([random.choice(['@', '#']) if i != int(width/2)  else ' ' for i in range(width)]) for j in range(length-5)]
    matrix += ['+@@# #@@#']
    matrix += ['##@# #@##']
    if reward_location == 0:
        matrix += ['@@@# H@@@']
    else:
        matrix += ['#@@H #@@#']
    matrix += ['####A####']
    matrix += ['#########']
    #print(np.asarray(matrix).shape[0])
    return np.asarray(matrix), length



def make_game(randomness, reward_location, length_lab):

  if reward_location is None: #in random case reward location should be None
      if randomness:
          # If the agent is in testing mode, randomly choose a Goal location.

         reward_location = np.random.choice([0, 1])

      else:
         reward_location = 0

  #game = GAME_ART[reward_location]
  game, length = game_art_function(9, *length_lab, reward_location)
  scrolly_info = prefab_drapes.Scrolly.PatternInfo(
      game, STAR_ART, board_northwest_corner_mark='+',
      what_lies_beneath=MAZES_WHAT_LIES_BENEATH[0],
       )
  if reward_location == 0: #0 - reward located on the right side, 1 - left side
     LEFT_REWARD = -1.0
     RIGHT_REWARD = 1.0
  else:
     LEFT_REWARD = 1.0
     RIGHT_REWARD = -1.0

  player_position = scrolly_info.virtual_position('A')
  left_goal_kwarg = scrolly_info.kwargs('L')
  right_goal_kwarg = scrolly_info.kwargs('R')
  hint_position = scrolly_info.kwargs('H')


  wall_1_kwargs = scrolly_info.kwargs('#')
  wall_2_kwargs = scrolly_info.kwargs('@')

  return ascii_art.ascii_art_to_game(
      STAR_ART, what_lies_beneath=' ',
      sprites={'A': ascii_art.Partial(AgentSprite, player_position, left_r=LEFT_REWARD, right_r=RIGHT_REWARD)},
      drapes={'#': ascii_art.Partial(MazeDrape, **wall_1_kwargs),
              '@': ascii_art.Partial(MazeDrape, **wall_2_kwargs),
               'L': ascii_art.Partial(MazeDrape, **left_goal_kwarg),
               'R': ascii_art.Partial(MazeDrape, **right_goal_kwarg),
               'H': ascii_art.Partial(MazeDrape, **hint_position)},
      update_schedule=[['#', 'H','@'], ['A', 'L', 'R']]), length #important for proper changing of partial observable env

MAZES_WHAT_LIES_BENEATH = [  #what lies under +
    '#'
]

STAR_ART = ['         ',
	        '    .    ',
	        '         ',
            '    .    ']   #how large visible part should be




class AgentSprite(prefab_sprites.MazeWalker):

  def __init__(self, corner, position, character, virtual_position, left_r=None, right_r=None):
    """Inform superclass that we can't walk through walls."""
    self.left_r = left_r
    self.right_r = right_r

    super(AgentSprite, self).__init__(
        corner, position, character, egocentric_scroller=True, impassable={'#', 'H','@'})
    self._teleport(virtual_position)


  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop  # Unused.
    rows, cols = self.position

    # Apply motion commands.
    #print(rows, cols, layers['#'][rows-1, cols])
    if actions == 0:    # walk upward?
      self._north(board, the_plot)
      the_plot.add_reward(-0.01)

   # elif actions == 4:  # walk downward?
   #   self._south(board, the_plot)
   #   the_plot.add_reward(-0.1)

    elif actions == 1:  # walk leftward?
      self._west(board, the_plot)
      if (layers['#'][rows, cols - 1] or layers['@'][rows, cols - 1] or layers['H'][rows, cols - 1]):
          the_plot.add_reward(-0.05)
      else:
          the_plot.add_reward(-0.01)


          # if layers['H'][things['A'].position]  == True:   #reward for being inside of hint
     #    the_plot.add_reward(HINT_REWARD)
     # else: the_plot.add_reward(MOVEMENT_REWARD)

    elif actions == 2:  # walk rightward?
      self._east(board, the_plot)
      if (layers['#'][rows, cols + 1] or layers['@'][rows, cols + 1] or layers['H'][rows, cols + 1]):
          the_plot.add_reward(-0.05)
      else:
          the_plot.add_reward(-0.01)

   # elif actions == 3:  # is the player doing nothing?
   #   self._stay(board, the_plot)
   #   the_plot.add_reward(-0.01)

    #global prev_position
    #prev_position.append(self.position)

    if layers['L'][things['A'].position] == True:
      the_plot.add_reward(self.left_r)
      the_plot.terminate_episode()

    if layers['R'][things['A'].position] == True:
      the_plot.add_reward(self.right_r)
      the_plot.terminate_episode()

  #the_plot.terminate_episode()

class MazeDrape(prefab_drapes.Scrolly):
  def update(self, actions, board, layers, backdrop, things, the_plot):
    #del backdrop, things, layers  # Unused
																		#print(actions) - lovely Nones
    if actions == 0:    # is the player going upward?
      self._north(the_plot)
   # elif actions == 4:  # is the player going downward?
   #   self._south(the_plot)
    elif actions == 1:  # is the player going leftward?
      self._west(the_plot)

    elif actions == 2:  # is the player going rightward?
      self._east(the_plot)
   # elif actions == 3:  # is the player doing nothing?
   #   self._stay(the_plot)






def main(argv=()):
  del argv  # Unused.

  # Build a game.

  game = make_game(False, 0)


  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0,
                       curses.KEY_LEFT: 1, curses.KEY_RIGHT: 2,
                       -1: 4}, #curses.KEY_DOWN: not using
      delay=200)

  # Let the game begin!
  ui.play(game)

def print_obs(obs):
    matr = []
    obs, info = obs
    obs = obs.tolist()
    #print("info", info)
    #print("end info")
    #print("obs", obs)
    #print(type(info), type(obs))
    #print("end obs")
    #a = np.array(pd.DataFrame.from_dict(info))
    #print(a)
    #matr.append(1*info['A']) #,1*info['H'], info['@'], info['#'], info['L'], info['R'] )
    #matr.append(1*info['H'])
    #matr, none_var = T_lab_observation(obs)
    #matr = matr.tolist()


    for i in range(len(obs)):
        obs[i] = ''.join([chr(ch) for ch in obs[i]])
        print(obs[i])


def dummy_episode():
    import numpy as np
    game = make_game(True, None, [10,12])

    action_keys = ['up', 'left', 'right'] # 'noop']

    obs_t, r_t, discount_t = game.its_showtime()
    total_r = r_t if r_t else 0.
    for t in range(1,101):
        a_t = None
        print_obs(obs_t)
        while a_t not in action_keys:
            a_t = input("Choose one of the following actions: {}:\n".format(action_keys))
        obs_t, r_t, discount_t = game.play(action_keys.index(a_t))
        total_r += r_t
        print('r =', r_t, 'gamma = ', discount_t)
        #obs_t, r_t = game.play(action_keys.index(a_t))

        if discount_t == 0: break
        print('===========  Step #{}  ==========:'.format(t))

    print('Done!')
    print('total_reward={}, num_steps={}'.format(total_r, t))


 #for i in range(len(keys)):
    #    key = keys[i]
    #    if all([key != '.', key != ' ']):
    #        matr_obs.append(1*info[key])
    #        keys_list.append(key)
    #matr_obs = np.array(matr_obs)
    #print(keys_list)
    #obs = obs.tolist()
    #for i in range(len(obs)):
    #    obs[i] = ''.join([chr(ch) for ch in obs[i]])
    #    print(obs[i])

def T_lab_observation(obs_t):
    import operator

    obs,info = obs_t
    info = sorted(info.items(), key=operator.itemgetter(0))
    #print("sorted_x", info)
    keys = [info[i][0] for i in range(len(info))]
    matrixes_x = [1*info[i][1] for i in range(len(info))]
    matrixes_x = np.asarray(matrixes_x)
    #print("obs_ttttttt22222", keys, matrixes_x)
    #print("obs_ttttttt22222", matrixes_x.shape)
    return matrixes_x, None    # i need any information about env, rather than just 0,1. so i am returning dict too


def T_lab_actions():
	action_keys = [0, 1, 2]
	return(np.ndarray(action_keys))


if __name__ == '__main__':
    #matrix = ['#########',
    #          '#L     R#']
    #matrix += [''.join([random.choice(['#', '@']) if i != 4  else ' ' for i in range(9)   ]) for j in range(2)]
    #print(np.asarray(matrix))
    dummy_episode()
  #main(sys.argv)



	#game = make_game(True, None)
	#obs_t, r_t, discount_t = game.its_showtime()
	#obs,info = 0, 0
