import pygame
import random
import math

from pygame.locals import (
    MOUSEBUTTONUP,
    QUIT,
)

SCREEN_WIDTH = 600
GAME_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.005
FRICTION = 0.001
SKEWED_PROBABILITY = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
FRUITS = pygame.sprite.Group()
game_joever = False
score = 0 #
NAMES = ["Cherry", "Strawberry", "Grape", "Dekopon", "Orange", "Apple", 
         "Pear", "Peach", "Pineapple", "Melon", "Watermelon"] #List with index corresponding to fruits
# Dictionary with names as keys, values index 0 the size, second index a tuple of RGB values, 
# and the last index the index which the fruit corresponds to in the above array
TYPES = {"Cherry":(10, (153,0,0), 0,1), 
          "Strawberry":(15, (255,0,0), 1,2), 
          "Grape":(20, (127,0,255), 2,3), 
          "Dekopon":(25, (255,255,51), 3,4), 
          "Orange":(35, (255,128,0), 4,5), 
          "Apple":(45, (255,51,51), 5,6), 
          "Pear":(60, (178,255,102), 6,7), 
          "Peach":(80, (255,204,153), 7,8), 
          "Pineapple":(90, (255,255,0), 8,9), 
          "Melon":(110, (128,255,0), 9,10), 
          "Watermelon":(130, (0,102,0), 10,11)
          }
SCORES = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55] # Scores for getting certain fruit

class Fruit(pygame.sprite.Sprite): # class of the fruit, including its type, size, vertical and horizontal velocity, x,y pos and angular velocity denoted as w

    def __init__(self, type, x, y = 100):
        
        super(Fruit, self).__init__() 
        self.dx = 0
        self.dy = 0
        self.w = 0
        self.x = x
        self.y= y
        self.type = type
        self.radius = TYPES[type][0]
        self.mass = TYPES[type][3]

        self.color = TYPES[type][1]
        self.surf = pygame.Surface((self.radius*2, self.radius*2),pygame.SRCALPHA, 32)
        self.offset = 100 #Offset pixels from the top of the screen when placing fruits
        self.timeAboveLine = 0
        pygame.draw.circle(self.surf, self.color, (self.radius, self.radius), self.radius) # could create non circular hitboxes, will have to see
        self.rect = self.surf.get_rect()
        self.rect.center = (x, y)
        
        
    def update(self): # Update position, velocity
        # pygame.draw.circle(self.surf, self.color, (self.rect.center[0], self.rect.center[1]), self.radius) # could create non circular hitboxes, will have to see

        self.dy += GRAVITY
        if self.dx > 0:
            self.dx -= FRICTION
        elif self.dx < 0:
            self.dx += FRICTION
        collidedFruits = []
        for fruit in FRUITS:
            if fruit != self and pygame.sprite.collide_circle(self, fruit):
                collidedFruits.append(fruit)
        if len(collidedFruits) > 0:
            for fruit in collidedFruits:
                if self.type == fruit.type:
                    FRUITS.remove(self)
                    FRUITS.remove(fruit)
                    # Makes a new fruit the next level up
                    if self.type != "Watermelon":
                        newFruit = Fruit(NAMES[TYPES[self.type][2]+1], 
                                        (self.rect.center[0] + fruit.rect.center[0])/2, 
                                        (self.rect.center[1] + fruit.rect.center[1])/2) 
                    FRUITS.add(newFruit)
                else:
                    # hypotenuse = 
                    if self.rect.center[0] > fruit.rect.center[0]: # Self is on the right
                        # self.rect.left = fruit.rect.right
                        self.dx += 0.01 # Fix
                        # self.dx += self.dx*math.sin()
                    if self.rect.center[0] < fruit.rect.center[0]: # Self is on the left
                        # self.rect.right = fruit.rect.left
                        self.dx += -0.01 # Fix
                    if (self.rect.center[1] > fruit.rect.center[1]): # Self is below
                        # self.rect.top = fruit.rect.bottom
                        self.dy += 0.005 # Fix
                    if (self.rect.center[1] < fruit.rect.center[1]): # Self is on top
                        # self.rect.bottom = fruit.rect.top
                        self.dy = 0 # Fix
                    
                    # vel1 = math.sqrt(self.dx**2 + self.dy**2)

                    # vel2 = math.sqrt(fruit.dx**2 + fruit.dy**2)
                    # ctheta1 = 0
                    # stheta2=0
                    # ctheta2=0
                    # ctheta1=0
                    # if(vel1 != 0):
                    #     ctheta1 = self.dx /(math.sqrt(self.dx**2 +self.dy**2))
                    #     stheta1 = -self.dy /(math.sqrt(self.dx**2 +self.dy**2))
                    # if(vel2 != 0):
                    #     ctheta2 = fruit.dx /(math.sqrt(fruit.dx**2 +fruit.dy**2))
                    #     stheta2 = -fruit.dy /(math.sqrt(fruit.dx**2 +fruit.dy**2))
                    # ccontact = ((fruit.rect.center[0]-self.rect.center[0]) /math.sqrt((fruit.rect.center[0]-self.rect.center[0])**2 + (fruit.rect.center[1]-self.rect.center[1])**2))
                    # scontact = ((fruit.rect.center[1]-self.rect.center[1] )/math.sqrt((fruit.rect.center[0]-self.rect.center[0])**2 + (fruit.rect.center[1]-self.rect.center[1])**2))

                    # self.dx+=(vel1* (ctheta1 * ccontact + stheta1 * scontact)* (self.mass - fruit.mass) + 2* fruit.mass * vel2 * (ctheta2 * ccontact + stheta2 * scontact))/(self.mass + fruit.mass)*ccontact+vel1 * (stheta1 * ctheta2 - ctheta1 * stheta2)*(-scontact)
                    # self.dy+=(vel1* (ctheta1 * ccontact + stheta1 * scontact)* (self.mass - fruit.mass) + 2* fruit.mass * vel2 * (ctheta2 * ccontact + stheta2 * scontact))/(self.mass + fruit.mass)*scontact+vel1 * (stheta1 * ctheta2 - ctheta1 * stheta2)*(ccontact)



        if self.rect.left < 0: 
            self.rect.left = 0
            self.dx *= -1 # Fix
        if self.rect.right > GAME_WIDTH:
            self.rect.right = GAME_WIDTH
            self.dx *= -1 # Fix
        if self.rect.center[1] < self.offset+10: # Fruit goes above loss line Fix
            self.timeAboveLine+=1
            if self.timeAboveLine > 600:
                global game_joever
                game_joever = True
        else:
            self.timeAboveLine = 0
        if self.rect.bottom >= SCREEN_HEIGHT:
            self.rect.bottom = SCREEN_HEIGHT
            self.dy = 0 # Fix (maybe)
        self.x +=self.dx
        self.y += self.dy
        self.rect.center = (self.x,self.y)

        

pygame.init()

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
clock.tick(30)
pygame.font.init()

running = True

nextFruitName = NAMES[random.choices(range(0,6),weights=SKEWED_PROBABILITY, k=1)[0]] # Weigh probabilities
while running:
    for event in pygame.event.get():
        if event.type == MOUSEBUTTONUP and not game_joever: # Let go of mouse button
            # Add a new random fruit at the x position of the cursor
            FRUITS.add(Fruit(nextFruitName, pygame.mouse.get_pos()[0]))
            nextFruitName = NAMES[random.choices(range(0,6),weights=SKEWED_PROBABILITY, k=1)[0]]
        elif event.type == QUIT:
            running = False
    if (game_joever):
        font = pygame.font.SysFont("Times New Roman", 50)
        text_surface = font.render("Game Joever", False, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(GAME_WIDTH/2, SCREEN_HEIGHT/2))
        screen.fill((0, 0, 0))
        screen.blit(text_surface, text_rect)
    else:
        screen.fill((0, 0, 0))
        screen.fill((255, 255, 255), (GAME_WIDTH, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
        #Draw the next fruit
        radius = TYPES[nextFruitName][0]
        color = TYPES[nextFruitName][1]
        surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA, 32)
        pygame.draw.circle(surf, color, (radius, radius), radius) # could create non circular hitboxes, will have to see
        rect = surf.get_rect()
        rect.center = (GAME_WIDTH/2, 50)
        screen.blit(surf,rect)
        FRUITS.update() # Update every fruit
        # FRUITS.draw(screen)
        for fruit in FRUITS:
            screen.blit(fruit.surf, fruit.rect) 
    pygame.display.flip()