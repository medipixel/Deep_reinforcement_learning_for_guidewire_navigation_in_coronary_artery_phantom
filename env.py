import os
from glob import glob
import csv

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
        self.idx = 0

        self.episode_img_path = ""
        self.origin_images = []
        self.state_images = []

        self.action_list = ["Forward", "Backward", "Rotate"]
        self.action = 0

    def reset(self):
        self.action = 0
        self.episode_img_path = os.path.join(self.img_root_path, str(self.episode))
        self.origin_images = sorted(glob(os.path.join(self.episode_img_path, "origin_crop_image/*.png")))
        self.state_images = sorted(glob(os.path.join(self.episode_img_path, "state_image/*.png")))
        with open(os.path.join(self.episode_img_path, "reward.csv"), "r") as csvfile:
            self.reward_list = list(csv.reader(csvfile))

        self.idx = 0
        state = np.expand_dims(cv2.imread(self.state_images[self.idx], cv2.IMREAD_GRAYSCALE), axis=-1)

        return state

    def step(self, action):
        self.action = action
        self.idx += 1
        next_state = np.expand_dims(cv2.imread(self.state_images[self.idx], cv2.IMREAD_GRAYSCALE), axis=-1)

        reward = float(self.reward_list[self.idx][0])
        done = True if self.idx == len(self.state_images) - 1 else False

        if done:
            self.episode = self.episode + 1 % 3

        info = dict()
        return next_state, reward, done, info

    def render(self, mode: str = "human"):
        origin = cv2.imread(self.origin_images[self.idx])
        state = cv2.imread(self.state_images[self.idx])

        action = self.action_list[self.action]
        info = np.zeros((84, 168, 3), dtype=np.uint8)
        info = cv2.putText(info, f"step: {self.idx}", (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        info = cv2.putText(info, f"action: {action}", (0, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        render_image = cv2.hconcat([origin, state])
        render_image = cv2.vconcat([render_image, info])
        cv2.imshow("State", render_image)
        cv2.waitKey(80)

    def close(self):
        cv2.destroyAllWindows()

    def seed(self, seed: int = None):
        """Set random seed."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]