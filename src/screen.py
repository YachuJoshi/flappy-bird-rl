import pygame
from src.base import SCREEN_HEIGHT, SCREEN_WIDTH

pygame.init()
pygame.display.init()


def init(caption):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(caption)
    clock = pygame.time.Clock()

    return screen, clock
