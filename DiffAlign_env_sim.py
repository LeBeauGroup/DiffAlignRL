"""
@author: Michael Xu
LeBeau Group, MIT
STEM beam centering env
v_sim

"""

import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import csv
import os

## with randomized inputs, more state info

import FlucamSimulator_sim as FluSim


class DiffAlign_env_sim(gym.Env):
    def __init__(self, step_size, mode: str, log_path: str, tol):
        self.seed(42)
        self.mode = mode
        self.init_log(log_path)
        self.tol = tol
        self.step_size = step_size
        self.start_vel = 0
        self.goal_pos = np.array([4,8])
        self.goal_vel = 0
        self.max_pos = 127
        self.min_pos = -127
        self.size = self.max_pos-self.min_pos + 1
        self.flu_size = 1040
        self.rotation_axes = 0
        self.reset()
        self.previous_dist = abs(np.linalg.norm(self.goal_pos - self.start_pos))
        self.min_action = -1.0
        self.max_action = 1.0
        self.viewer = None
        self.observation_space = spaces.Box(
            low=self.min_pos, high=self.max_pos, dtype=np.int16, shape=(5,)
        )
        self.action_space = spaces.Box(-1, 1, (2,), dtype=np.float32)

        self.viewer = None
        self.offset = 5
        self.render()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        iterate_x = action[0]/10 # scaling action space down like on microscope
        iterate_y = action[1]/10
        shift = np.array([iterate_x, iterate_y]) #shift, rather than position gets sent to flusim
        self.fluscreen.shift_beam(shifts=shift)

        beam_state = self.fluscreen.read_fluscreen()
        pos_raw = beam_state["pos"]
        pos = self.downsize_coords(pos_raw)
        beam_visible = self.check_beam(pos)
        if beam_visible and (self.current_step < self.max_episode_steps):
            dist = abs(np.linalg.norm(self.goal_pos - pos))
            diff = pos - self.goal_pos
            state = np.array([self.step_size, diff[0], diff[1],
                     1.0 if pos[0] == self.goal_pos[0] else 0.0,
                     1.0 if pos[1] == self.goal_pos[1] else 0.0
                     ])
            self.state = state
            dist_improvement = self.previous_dist - dist
            if dist_improvement <= 0:
                reward = -1- dist / 128
            elif dist_improvement > 0:
                reward = 1 - dist / 128

            done = bool(dist <= self.tol)
            if done:
                reward = 100.0
                print("success!")
                print(pos)
            self.previous_dist = dist
            self.current_step += 1

        else:
            pos = beam_state["pos"]
            reward = -100
            done = True
            state = self.state

        self.render()

        log_dict = {'a1' : action[0],
                    'a2': action[1],
                    'px': pos[0],
                    'py': pos[1],
                    'rew': reward,
                    'done': done,
                    'r1': pos_raw[0],
                    'r2': pos_raw[1]
                    }
        self.log_this(log_dict)

        return state, reward, done, {}

    def reset(self):
        if self.step_size is None:
            print('using step: ', str(self.step_size))
            self.step_size = np.random.randint(25, 100)
        self.fluscreen = FluSim.FlucamSimulator_sim(self.flu_size, source_size = self.size, mode="disk", step=self.step_size)
        start_pos_raw = self.fluscreen.get_pos
        start_pos = self.downsize_coords(start_pos_raw)
        diff = start_pos - self.goal_pos
        self.start_pos = start_pos
        self.previous_dist = abs(np.linalg.norm(self.goal_pos - self.start_pos))

        state = np.array([self.step_size, diff[0], diff[1],
                 1.0 if start_pos[0] == self.goal_pos[0] else 0.0,
                 1.0 if start_pos[1] == self.goal_pos[1] else 0.0
                 ])
        self.state = state
        self.current_step = 0
        log_dict = {'a1' : 0,
                    'a2': 0,
                    'px': start_pos[0],
                    'py': start_pos[1],
                    'rew': 0,
                    'done': False,
                    'r1': start_pos_raw[0],
                    'r2': start_pos_raw[1]
                    }
        self.log_this(log_dict)

        return state

    def check_beam(self, pos):
        if any(pos>= np.array([self.max_pos-self.offset, self.max_pos-self.offset])):
            return False
        elif any(pos<= np.array([self.min_pos+self.offset, self.min_pos+self.offset])):
            return False
        else:
            return True

    def downsize_coords(self, raw_coords):
        # range of image: 0:1040, 0:1040
        # divide by 4: 260
        coords = np.round((raw_coords  - np.array([self.fluscreen.get_size/2, self.fluscreen.get_size/2]))/ (520/127))
        # range of RL environment: -127:127, -127:127
        return coords

    def render(self, mode='human'):
        self.fluscreen.render_fluscreen(self.goal_pos, self.mode)

    def init_log(self, log_path):
        if log_path is not None:
            if not os.path.exists(log_path):
                os.makedirs(log_path[0:-1])
            self.log_path = log_path
            self.log = True
            log_filename = self.log_path + "training_log.csv"
            self.file_handler = open(log_filename, "wt")
            self.logger = csv.DictWriter(self.file_handler, fieldnames=("a1", "a2", "px", "py", "rew", "done", "r1", "r2"))

            self.logger.writeheader()
            self.file_handler.flush()

    def log_this(self, dict_to_log):
        if self.log:
            self.logger.writerow(dict_to_log)
            self.file_handler.flush()

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
        self.file_handler.flush()
        self.file_handler.close()