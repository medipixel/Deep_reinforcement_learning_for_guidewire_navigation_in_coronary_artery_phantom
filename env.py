import os
from glob import glob

import numpy as np
import cv2
import gym
import gym.spaces as spaces
import gym.utils.seeding as seeding


class PhantomDummyEnv(gym.Env):
    def __init__(self):
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(3)
        self.name = self.__class__.__name__

        self.img_root_path = "./data/dummy"
        self.episode = 0
        self.episode_img_path = os.path.join(self.img_root_path, str(self.episode))
        self.idx = 0

    def reset(self):
        self.origin_images = sorted(glob(os.path.join(self.episode_img_path, "origin_crop_image/*.png")))
        self.state_images = sorted(glob(os.path.join(self.episode_img_path, "state_image/*.png")))

        self.idx = 0
        state = cv2.imread(self.state_images[self.idx])

        return state

    def step(self, _):
        self.idx += 1
        next_state = cv2.imread(self.state_images[self.idx])

        reward = 0
        done = True if self.idx == len(self.state_images) - 1 else False

        if done:
            self.episode = self.episode + 1 % 3

        info = dict()
        return next_state, reward, done, info

    def render(self):
        origin = cv2.imread(self.origin_images[self.idx])
        state = cv2.imread(self.state_images[self.idx])

        render_image = cv2.hconcat([origin, state])
        cv2.imshow(render_image)
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    def close(self):
        cv2.destroyAllWindows()

    def seed(self, seed: int = None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]