import pygame
import random
import pymunk

pygame.init()
clock = pygame.time.Clock()
FRAME_RATE = 30
clock.tick(FRAME_RATE)

SCREEN_WIDTH = 600
GAME_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.005
FRICTION = 0.05
XLOSS  = 0.00
YLOSS = 0.00
OFFSET = 100
SKEWED_PROBABILITY = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
FRUITS = pygame.sprite.Group()

space = pymunk.Space()
space.gravity = 0.0, -5.0

space.add_collision_handler(2, 2)

balls = []

NAMES = ["Cherry", "Strawberry", "Grape", "Dekopon", "Orange", "Apple", 
         "Pear", "Peach", "Pineapple", "Melon", "Watermelon"] #List with index corresponding to fruits
# Dictionary with names as keys, values index 0 the size, second index a tuple of RGB values, 
# and the last index the index which the fruit corresponds to in the above array
TYPES = {"Cherry":(10, (153,0,0), 0,10), 
          "Strawberry":(15, (255,0,0), 1,12.5), 
          "Grape":(20, (127,0,255), 2,20), 
          "Dekopon":(25, (255,255,51), 3,30), 
          "Orange":(35, (255,128,0), 4,40), 
          "Apple":(45, (255,51,51), 5,50), 
          "Pear":(60, (178,255,102), 6,60), 
          "Peach":(70, (255,204,153), 7,70), 
          "Pineapple":(80, (255,255,0), 8,80), 
          "Melon":(95, (128,255,0), 9,90), 
          "Watermelon":(110, (0,102,0), 10,100)
          }
SCORES = [0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55] # Scores for getting certain fruit

def flipY(y):
    return SCREEN_HEIGHT - y

class Fruit(pygame.sprite.Sprite, pymunk.Circle): # class of the fruit, including its type, size, vertical and horizontal velocity, x,y pos and angular velocity denoted as w

    def __init__(self, body, type, x, y = 100):
        pygame.sprite.Sprite.__init__(self)
        self.fruitBody = body
        pymunk.Circle.__init__(self, self.fruitBody, TYPES[type][0], (0, 0))
        # self.dx = 0
        # self.dy = 0
        # self.w = 0
        # self.x = x
        # self.y= y
        # self.pastdx = 0
        # self.pastdy = 0
        self.type = type
        # self.radius = TYPES[type][0]
        # self.mass = TYPES[type][3]

        self.color = TYPES[type][1]
        self.surf = pygame.Surface((self.radius*2, self.radius*2),pygame.SRCALPHA, 32)
        self.timeAboveLine = 0
        pygame.draw.circle(self.surf, self.color, (self.radius, self.radius), self.radius) # could create non circular hitboxes, will have to see
        self.rect = self.surf.get_rect()
        self.rect.center = (x, y)
        
        
    def update(self, values): # Update position, velocity
        # pygame.draw.circle(self.surf, self.color, (self.rect.center[0], self.rect.center[1]), self.radius) # could create non circular hitboxes, will have to see
        # self.dy += GRAVITY
        # if(self.y + self.radius >= SCREEN_HEIGHT):
        #     self.dx = (1-FRICTION)*self.dx
        self.rect.center = self.fruitBody.position[0], flipY(self.fruitBody.position[1])
        collidedFruits = []
        for fruit in FRUITS:
            if fruit != self and pygame.sprite.collide_circle(self, fruit):
                collidedFruits.append(fruit)
        if len(collidedFruits) > 0:
            # self.pastdx = self.dx
            # self.pastdy = self.dy
            # self.dx=0
            # self.dy=0
            for fruit in collidedFruits:
                if self.type == fruit.type:
                    FRUITS.remove(self)
                    FRUITS.remove(fruit)
                    # Makes a new fruit the next level up
                    if self.type != "Watermelon":
                        newBody = pymunk.Body(TYPES[self.type][3], 100)
                        newBody.position = ((self.fruitBody.position[0] + fruit.fruitBody.position[0])/2, 
                        (self.fruitBody.position[1] + fruit.fruitBody.position[1])/2)
                        newFruit = Fruit(newBody, NAMES[TYPES[self.type][2]+1], newBody.position[0], flipY(newBody.position[1]))
                        newFruit.friction = 0.5
                        newFruit.collision_type = 2
                        space.add(newBody, newFruit)
                        balls.append(newFruit)
                        FRUITS.add(newFruit)
                        values[0] += SCORES[TYPES[self.type][2]+1]
        #         else:
        #             # if(math.sqrt((self.x-fruit.x)**2 + (self.y - fruit.y)**2)+3 <self.radius + fruit.radius):
        #             #     if(self.y<fruit.y):
        #             #         self.y -= ((self.radius + fruit.radius)-math.sqrt((self.x-fruit.x)**2 + (self.y - fruit.y)**2))* self.y/math.sqrt(self.y**2+self.x**2)
        #             # # hypotenuse = 
        #             if self.rect.center[0] > fruit.rect.center[0]: # Self is on the right
        #                 # self.rect.left = fruit.rect.right
        #                 self.dx += 0.01 # Fix
        #                 # self.dx += self.dx*math.sin()
        #             if self.rect.center[0] < fruit.rect.center[0]: # Self is on the left
        #                 # self.rect.right = fruit.rect.left
        #                 self.dx += -0.01 # Fix
        #             # if (self.rect.center[1] > fruit.rect.center[1]): # Self is below
        #             #     # self.rect.top = fruit.rect.bottom
        #             #     self.dy = 0.005 # Fix
        #             if (self.rect.center[1] < fruit.rect.center[1]): # Self is on top
        #                 # self.rect.bottom = fruit.rect.top
        #                 self.dy += -0.005 # Fix
        #             # vel1 = math.sqrt(self.pastdx**2 + self.pastdy**2)

        #             # vel2 = math.sqrt(fruit.pastdx**2 + fruit.pastdy**2)
        #             # ctheta1 = 0
        #             # stheta2=0
        #             # ctheta2=0
        #             # stheta1=0
        #             # if(vel1 != 0):
        #             #     ctheta1 = self.pastdx /vel1
        #             #     stheta1 = self.pastdy /vel1
        #             # if(vel2 != 0):
        #             #     ctheta2 = fruit.pastdx /vel2
        #             #     stheta2 = fruit.pastdy /vel2
        #             # ccontact = ((fruit.x-self.x) /math.sqrt((fruit.x-self.x)**2 + (fruit.y-self.y)**2))
        #             # scontact = ((fruit.y-self.y )/math.sqrt((fruit.x-self.x)**2 + (fruit.y-self.y)**2))
        #             # if(self.x > fruit.x):
        #             #     self.dx+= abs((1-XLOSS) * ((vel1* (ctheta1 * ccontact + stheta1 * scontact)* (self.mass - fruit.mass) + 2* fruit.mass * vel2 * (ctheta2 * ccontact + stheta2 * scontact))/(self.mass + fruit.mass)*ccontact+vel1 * (stheta1 * ctheta2 - ctheta1 * stheta2)*(-scontact)))
        #             # else:
        #             #     self.dx-= abs((1-XLOSS) * ((vel1* (ctheta1 * ccontact + stheta1 * scontact)* (self.mass - fruit.mass) + 2* fruit.mass * vel2 * (ctheta2 * ccontact + stheta2 * scontact))/(self.mass + fruit.mass)*ccontact+vel1 * (stheta1 * ctheta2 - ctheta1 * stheta2)*(-scontact)))
        #             # if(self.y > fruit.y):
        #             #     self.dy+=abs((1-YLOSS)*((vel1* (ctheta1 * ccontact + stheta1 * scontact)* (self.mass - fruit.mass) + 2* fruit.mass * vel2 * (ctheta2 * ccontact + stheta2 * scontact))/(self.mass + fruit.mass)*scontact+vel1 * (stheta1 * ctheta2 - ctheta1 * stheta2)*(ccontact)))
        #             # else:
        #             #     self.dy-=abs((1-YLOSS)*((vel1* (ctheta1 * ccontact + stheta1 * scontact)* (self.mass - fruit.mass) + 2* fruit.mass * vel2 * (ctheta2 * ccontact + stheta2 * scontact))/(self.mass + fruit.mass)*scontact+vel1 * (stheta1 * ctheta2 - ctheta1 * stheta2)*(ccontact)))

        #             # print(self.radius)
        #             # print(self.dx)
        #             # print(self.dy)


        if self.rect.left < 0: 
            self.fruitBody.position = self.radius, self.fruitBody.position[1]
            # self.x = 0 + self.radius
            # self.dx *= -0.8 # Fix
        if self.rect.right > GAME_WIDTH:
            self.fruitBody.position = GAME_WIDTH - self.radius, self.fruitBody.position[1]
            # self.x = GAME_WIDTH - self.radius
            # self.dx *= -0.8 # Fix
        if self.rect.center[1] < OFFSET+10: # Fruit goes above loss line Fix
            self.timeAboveLine+=1
            if self.timeAboveLine > 600:
                values[1] = True
        else:
            self.timeAboveLine = 0
        # # if self.rect.bottom == SCREEN_HEIGHT:
        # #     self.rect.bottom = SCREEN_HEIGHT
        # #     self.dy = 0 # Fix (maybe)
        if self.rect.center[1] + self.radius >= SCREEN_HEIGHT:
            self.fruitBody.position = self.fruitBody.position[0], flipY(SCREEN_HEIGHT - self.radius)
            # self.y = SCREEN_HEIGHT - self.radius
            # self.dy = 0 # Fix (maybe)
        # self.x += self.dx
        # self.y += self.dy
        self.rect.center = self.fruitBody.position[0], flipY(self.fruitBody.position[1])

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.font.init()
        self.queuedFruitName = NAMES[random.choices(range(0,6), weights = SKEWED_PROBABILITY, k = 1)[0]] # Weigh probabilities
        self.nextFruitName = NAMES[random.choices(range(0,6), weights = SKEWED_PROBABILITY, k = 1)[0]]
        self.score = 0
        self.game_joever = False
    
    def reset(self):
        self.queuedFruitName = NAMES[random.choices(range(0,6), weights = SKEWED_PROBABILITY, k = 1)[0]] # Weigh probabilities
        self.nextFruitName = NAMES[random.choices(range(0,6), weights = SKEWED_PROBABILITY, k = 1)[0]]
        self.score = 0
        self.game_joever = False
        for fruit in FRUITS:
            fruit.kill()
        global space
        space = pymunk.Space()
        space.gravity = 0.0, -5.0

        space.add_collision_handler(2, 2)

        global balls
        balls = []

    def update(self, position):
        if (not self.game_joever):
            if position != -1:
                body = pymunk.Body(TYPES[self.queuedFruitName][3], 100)
                body.position = position, flipY(100)
                fruit = Fruit(body, self.queuedFruitName, position)
                fruit.friction = 0.5
                fruit.collision_type = 2
                space.add(body, fruit)
                balls.append(fruit)
                FRUITS.add(fruit)
                self.queuedFruitName = self.nextFruitName
                self.nextFruitName = NAMES[random.choices(range(0,6), weights=SKEWED_PROBABILITY, k=1)[0]]
            dt = 1.0 / 30.0
            for x in range(1):
                space.step(dt)
            self.screen.fill((0, 0, 0))
            self.screen.fill((255, 255, 255), (GAME_WIDTH, 0, SCREEN_WIDTH, SCREEN_HEIGHT))
            # Make a line for GAME_JOEVER
            self.screen.fill((255, 255, 255), (0, OFFSET, GAME_WIDTH, 10))
            # Display the score
            font = pygame.font.SysFont("Times New Roman", 30)
            score_surface = font.render("Score: " + str(self.score), False, (0, 0, 0))
            self.screen.blit(score_surface, (GAME_WIDTH + 10, 10))
            # Draw the queued fruit
            radius = TYPES[self.queuedFruitName][0]
            color = TYPES[self.queuedFruitName][1]
            surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA, 32)
            pygame.draw.circle(surf, color, (radius, radius), radius)
            rect = surf.get_rect()
            rect.center = (GAME_WIDTH/2, 50)
            self.screen.blit(surf,rect)

            # Draw the next fruit
            radius = TYPES[self.nextFruitName][0]
            color = TYPES[self.nextFruitName][1]
            surf = pygame.Surface((radius*2, radius*2), pygame.SRCALPHA, 32)
            pygame.draw.circle(surf, color, (radius, radius), radius)
            rect = surf.get_rect()
            rect.center = ((GAME_WIDTH + SCREEN_WIDTH)/2, 130)
            self.screen.blit(surf,rect)

            # Draw word next
            font = pygame.font.SysFont("Times New Roman", 20)
            next_surface = font.render("Next", False, (0, 0, 0))
            next_rect = next_surface.get_rect(center=((GAME_WIDTH + SCREEN_WIDTH)/2, 70))
            self.screen.blit(next_surface, next_rect)
            values = [self.score, self.game_joever]
            FRUITS.update(values) # Update every fruit
            self.score = values[0]
            self.game_joever = values[1]
            # FRUITS.draw(screen)
            for fruit in FRUITS:
                self.screen.blit(fruit.surf, fruit.rect)
        else: # Game (j)o(e)ver
            font = pygame.font.SysFont("Times New Roman", 50)
            game_joever_surface = font.render("Game Joever", False, (255, 255, 255))
            game_joever_rect = game_joever_surface.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
            score_surface = font.render("Score: " + str(self.score), False, (255, 255, 255))
            score_rect = score_surface.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2 + 200))
            self.screen.fill((0, 0, 0))
            self.screen.blit(game_joever_surface, game_joever_rect)
            self.screen.blit(score_surface, score_rect)  
        pygame.display.flip()


pygame.quit()
        