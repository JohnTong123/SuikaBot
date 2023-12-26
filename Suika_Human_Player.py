from Suika_Simulation import Game, GAME_WIDTH
import Suika_Simulation
import pygame

pygame.init()

from pygame.locals import (
    MOUSEBUTTONUP,
    QUIT,
)

game = Game()
quit = False
# time = -1000
while True:
    position = -1
    for event in pygame.event.get():
        if event.type == MOUSEBUTTONUP and not game.game_joever and Suika_Simulation.canPlace: # Let go of mouse button
            # time = pygame.time.get_ticks()
            position = pygame.mouse.get_pos()[0]
            if position > GAME_WIDTH:
                position = GAME_WIDTH-1
            Suika_Simulation.canPlace = False
        elif event.type == QUIT: 
            game.game_joever = True
            quit = True
    game.update(position)
    if game.game_joever:
        for event in pygame.event.get():
            if event.type == QUIT:
                quit = True
        if quit:
            break
print("Final score: ", game.score)

pygame.quit()