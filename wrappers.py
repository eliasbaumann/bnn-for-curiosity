import gym
import numpy as np
# import cv2
from gym import spaces
from collections import deque
from PIL import Image
from copy import copy

# class WarpFrame(gym.ObservationWrapper):
    # def __init__(self, env, width=84, height=84, grayscale=True, normalize=False):
    #     """Warp frames to 84x84 as done in the Nature paper and later work."""
    #     gym.ObservationWrapper.__init__(self, env)
    #     self.width = width
    #     self.height = height
    #     self.grayscale = grayscale
    #     self.normalize = normalize
    #     if self.grayscale:
    #         self.observation_space = spaces.Box(low=0, high=255,
    #                                             shape=(self.height, self.width, 1), dtype=np.uint8)
    #     else:
    #         self.observation_space = spaces.Box(low=0, high=255,
    #                                             shape=(self.height, self.width, 3), dtype=np.uint8)

    # def observation(self, frame):
    #     if self.grayscale:
    #         frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    #     frame = cv2.resize(frame, (self.width, self.height),
    #                        interpolation=cv2.INTER_AREA)
    #     if(self.normalize):
    #         frame = cv2.normalize(
    #             frame, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #     if self.grayscale:
    #         frame = np.expand_dims(frame, -1)
    #     return frame

# taken from OpenAI baselines which requires muJoCo but i dont want to purchase a license just yet
class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env, crop=True):
        self.crop = crop
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84, 1), dtype=np.uint8)

    def observation(self, obs):
        return ProcessFrame84.process(obs, crop=self.crop)

    @staticmethod
    def process(frame, crop=True):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        elif frame.size == 224 * 240 * 3:  # mario resolution
            img = np.reshape(frame, [224, 240, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution." + str(frame.size)
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        size = (84, 110 if crop else 84)
        resized_screen = np.array(Image.fromarray(img).resize(size,
                                                              resample=Image.BILINEAR), dtype=np.uint8)
        x_t = resized_screen[18:102, :] if crop else resized_screen
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)



class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        acc_info = {}
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            acc_info.update(info)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, acc_info


class ExtraTimeLimit(gym.Wrapper):
    def __init__(self, env, max_episode_steps=None):
        gym.Wrapper.__init__(self, env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps > self._max_episode_steps:
            done = True
        return observation, reward, done, info

    def reset(self):
        self._elapsed_steps = 0
        return self.env.reset()

class AddRandomStateToInfo(gym.Wrapper):
    def __init__(self, env):
        """Adds the random state to the info field on the first step after reset
        """
        gym.Wrapper.__init__(self, env)

    def step(self, action):
        ob, r, d, info = self.env.step(action)
        if self.random_state_copy is not None:
            info['random_state'] = self.random_state_copy
            self.random_state_copy = None
        return ob, r, d, info

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.random_state_copy = copy(self.unwrapped.np_random)
        return self.env.reset(**kwargs)