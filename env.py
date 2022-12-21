import gym
import pygame
import numpy as np
from random import randint
from src.texts import font
from src.screen import init
from src.bird import Bird
from src.pipe import Pipe
from gym.spaces import Box, Discrete
from src.base import GAP, SCREEN_WIDTH, SCREEN_HEIGHT

caption: str = "Flappy Bird"


class FlappyBirdEnv(gym.Env):
    metadata = {
        "render_modes": ["human"],
        "render_fps": 120,
    }

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        shape = (3,)
        obs_low = np.ones(shape) * -np.inf
        obs_high = np.ones(shape) * np.inf

        # [ birdY, horizontalDistance, verticalDistance ]
        self.observation_space = Box(
            low=obs_low, high=obs_high, shape=(shape), dtype=np.float64
        )

        self.action_space = Discrete(2)

        self.score = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.screen, self.clock = init(caption)

        self.foreground = None
        self.foreground_rect = None
        self.top_pipes = []
        self.bottom_pipes = []

    def _get_obs(self):
        if len(self.bottom_pipes) == 0:
            return np.array([self.bird.y, SCREEN_WIDTH + 10, SCREEN_HEIGHT + 10])

        bottom_pipe = self.bottom_pipes[0]
        horizontal_distance = bottom_pipe.rect.right - self.bird.rect.right
        vertical_distance = bottom_pipe.rect.top - self.bird.rect.bottom

        return np.array([self.bird.y, horizontal_distance, vertical_distance])

    def _get_info(self):
        return {}

    def _generate_pipes(self):
        pipe_x = SCREEN_WIDTH + 10
        pipe_top_y = randint(-200, -50)
        pipe_bottom_y = pipe_top_y + GAP

        self.top_pipes.append(Pipe({"x": pipe_x, "y": pipe_top_y, "type": "top"}))
        self.bottom_pipes.append(
            Pipe({"x": pipe_x, "y": pipe_bottom_y, "type": "bottom"})
        )

    def reset(self):
        self.bird = Bird()
        self.reward = 0
        self.score = 0
        self.done = False
        self.top_pipes = [Pipe({"x": 200, "y": -180, "type": "top"})]
        self.bottom_pipes = [Pipe({"x": 200, "y": 240, "type": "bottom"})]

        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        self.reward = 0
        self.done = False

        # Take an action
        if action == 1 and self.bird.rect.top > 20:
            self.bird.flap()
        elif action == 0:
            pass

        # Update
        self.bird.update()

        if self.bird.rect.top <= 0 or self.bird.rect.bottom >= SCREEN_HEIGHT:
            self.done = True
            self.reward = -1000

        if self.foreground is not None:
            self.foreground_rect.left -= 2
            if self.foreground_rect.right - 10 < SCREEN_WIDTH:
                self.foreground_rect.left = 0

        for top_pipe, bottom_pipe in zip(self.top_pipes, self.bottom_pipes):
            top_pipe.update()
            bottom_pipe.update()

            if (
                top_pipe.rect.left == 80
                and self.bird.rect.top > top_pipe.rect.bottom
                and self.bird.rect.bottom < bottom_pipe.rect.top
            ):
                self.score += 1

            # Check Collision
            if self.bird.check_collision(top_pipe) or self.bird.check_collision(
                bottom_pipe
            ):
                self.reward = -1000
                self.done = True

            # Remove pipes if beyond the screen
            if top_pipe.rect.right < 0:
                self._generate_pipes()
                self.top_pipes.pop(0)
                self.bottom_pipes.pop(0)

        if self.foreground_rect is not None and self.bird.check_collision(
            self.foreground_rect
        ):
            self.reward = -1000
            self.done = True

        if not self.done:
            self.reward += 1

        observation = self._get_obs()
        reward = self.reward
        done = self.done
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        self.foreground = pygame.image.load("./images/base.png").convert_alpha()
        self.foreground_rect = self.foreground.get_rect(bottomleft=(0, 512))

        background = pygame.image.load("./images/background-night.png").convert_alpha()
        self.screen.blit(background, (0, 0))

        for top_pipe, bottom_pipe in zip(self.top_pipes, self.bottom_pipes):
            top_pipe.draw(self.screen)
            bottom_pipe.draw(self.screen)

        self.bird.draw(self.screen)
        score_text = font.render(f"{self.score}", False, "white")
        score_text_rect = score_text.get_rect(center=(SCREEN_WIDTH / 2, 70))
        self.screen.blit(score_text, score_text_rect)

        # Foreground
        self.screen.blit(self.foreground, self.foreground_rect)

        pygame.event.pump()
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])
        # self.clock.tick(60)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
