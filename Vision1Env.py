import sys
from typing import Optional
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pygame
from pygame import gfxdraw


class Vision1Env(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, config, render_mode: Optional[str] = None):
        self.grid_size = config["grid_size"]
        self.cardinality = config["cardinality"]
        self.levels = config["levels"]
        self.stage_size = config["stage_size"]
        self.render_wait = config["render_wait"]
        self.episode_length = config["episode_length"]
        self.center_dot = config["center_dot"]
        self.scene_size = self.stage_size * 2 + 1
        self.scene_image_size = self.scene_size * self.grid_size
        self.max_jump = self.scene_size - self.stage_size
        self.pos_xy = np.zeros((self.cardinality, 2), dtype=np.int8)
        self.object_size = np.ones(self.cardinality, dtype=np.int8)
        self.brightness = np.ones(self.cardinality, dtype=np.int8)

        self.action_space = spaces.Box(low=-1 * self.max_jump, high=self.max_jump, shape=(2,), dtype=np.int8)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.scene_image_size, self.scene_image_size, 3),
                                            dtype=np.uint8)
        self.observation = np.zeros((self.scene_image_size, self.scene_image_size, 3), dtype=np.uint8)
        self.gaze = np.array([0, 0])
        self.render_mode = render_mode
        self.isopen = True
        self.episode_count = 0

        pygame.init()
        pygame.display.init()
        self.screen = pygame.display.set_mode((self.scene_image_size, self.scene_image_size))
        self.scene = pygame.Surface((self.scene_image_size, self.scene_image_size))
        self.clock = pygame.time.Clock()

    def step(self, saccade):
        saccade = np.array(saccade)
        print("saccade:", saccade)
        gaze = self.gaze + saccade
        for i in range(self.cardinality):
            center_x = (self.stage_size // 2 + self.pos_xy[i, 0] + 1) * self.grid_size + self.grid_size // 2
            center_y = (self.stage_size // 2 + self.pos_xy[i, 1] + 1) * self.grid_size + self.grid_size // 2
            gfxdraw.filled_circle(self.scene, center_x, center_y, self.object_size[i],
                                  (0, 255 * self.brightness[i] // 5, 0))
        img = pygame.surfarray.array3d(self.scene)
        gaze_offset = gaze * self.grid_size
        if gaze[0] < 0:
            if gaze[1] < 0:
                self.observation[-1 * gaze_offset[0]:,
                                 -1 * gaze_offset[1]:] =\
                    img[:self.scene_image_size + gaze_offset[0],
                        :self.scene_image_size + gaze_offset[1]]
            else:
                self.observation[-1 * gaze_offset[0]:,
                                 0:self.scene_image_size - gaze_offset[1]] =\
                    img[:self.scene_image_size + gaze_offset[0], gaze_offset[1]:]
        else:
            if gaze[1] < 0:
                self.observation[0:self.scene_image_size - gaze_offset[0],
                                 -1 * gaze_offset[1]:] =\
                    img[gaze_offset[0]:, :self.scene_image_size + gaze_offset[1]]
            else:
                self.observation[0:self.scene_image_size - gaze_offset[0],
                                 0:self.scene_image_size - gaze_offset[1]] = \
                    img[gaze_offset[0]:, gaze_offset[1]:]
        self.episode_count += 1
        if self.episode_count >= self.episode_length:
            return self.observation, 0, True, False, {}
        else:
            return self.observation, 0, False, False, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        obs_surf = pygame.surfarray.make_surface(self.observation)
        self.screen.blit(obs_surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
        pygame.display.flip()
        pygame.time.wait(self.render_wait)

    def reset(self, seed=None, options={}):
        self.gaze = np.array([0, 0])
        self.episode_count = 0
        pygame.init()
        self.scene.fill((0, 0, 0))
        if self.center_dot:
            gfxdraw.filled_circle(self.scene, self.scene_image_size // 2, self.scene_image_size // 2, 3, (255, 0, 0))
        self.observation = np.zeros((self.scene_image_size, self.scene_image_size, 3), dtype=np.uint8)
        positions = np.random.choice(self.stage_size * self.stage_size, size=self.cardinality, replace=False)
        for i in range(self.cardinality):
            self.pos_xy[i, 0] = positions[i] % self.stage_size
            self.pos_xy[i, 1] = positions[i] // self.stage_size
            self.object_size[i] = np.random.randint(self.levels) + 1
            self.brightness[i] = np.random.randint(self.levels) + 1
        return self.observation, {}

    def close(self):
        pygame.display.quit()
        pygame.quit()
        self.isopen = False


def run(env, gaze):
    print(gaze)
    env.reset()
    env.step(gaze)
    env.render()


def main():
    config = {
        "stage_size": 5,
        "grid_size": 10,
        "cardinality": 2,
        "levels": 5,
        "render_wait": 3000,
        "episode_length": 3,
        "center_dot": True
    }
    if config["cardinality"] > config["stage_size"] * config["stage_size"]:
        print('Error: cardinality cannot be larger than stage_size^2!', file=sys.stderr)
        sys.exit(1)
    env = Vision1Env(config, render_mode="human")
    # gaze = np.random.randint(-2, 3, size=2)
    print(isinstance(env.action_space, spaces.Box))
    run(env, np.array([0, 0]))
    run(env, np.array([2, 2]))
    run(env, np.array([-2, -2]))
    run(env, np.array([-2, 2]))
    run(env, np.array([2, -2]))
    env.close()


if __name__ == '__main__':
    main()
