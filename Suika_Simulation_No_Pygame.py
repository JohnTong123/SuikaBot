import random
import pymunk
from pymunk import Vec2d

SCREEN_WIDTH = 600
GAME_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.05
FRICTION = 50
FRUITFRICTION = 0.9
XLOSS  = 0.00
YLOSS = 0.00
OFFSET = 100
SKEWED_PROBABILITY = [0.2, 0.2, 0.2, 0.2, 0.2]
# SKEWED_PROBABILITY = [1.0, 0.0, 0.0, 0.0, 0.0]
# FRUITS = pygame.sprite.Group()

space = pymunk.Space()
space.gravity = 0.0, -10.0
canPlace = True

FRUITS = []

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
score = 0 # TODO: Potential problem: global makes running multiple classes at once weird
pseudoscore = 0
def flipY(y):
    return SCREEN_HEIGHT - y

def checkCollision(arbiter, space, data):
    f1, f2 = arbiter.shapes
    global canPlace
    if f1.justPlaced:
        f1.justPlaced=False
        if not canPlace:
            canPlace = True
    if f2.justPlaced:
        f2.justPlaced=False
        if not canPlace:
            canPlace = True
    if f1.justMerged:
        f1.justMerged=False
    if f2.justMerged:
        f2.justMerged=False
    if f1.type == f2.type:
        remove = False
        if f1 in FRUITS and f2 in FRUITS:
            remove=True
        if remove:
            if f1 in space.shapes:
                space.remove(f1)
            if f2 in space.shapes:
                space.remove(f2)
            if f1 in FRUITS:
                FRUITS.remove(f1)
            if f2 in FRUITS:
                FRUITS.remove(f2)
            # Makes a new fruit the next level up
            newBody = pymunk.Body(TYPES[NAMES[(TYPES[f1.type][2]+1)%11]][3], TYPES[NAMES[(TYPES[f1.type][2]+1)%11]][3] * 10)
            newBody.position = ((f1.fruitBody.position[0] + f2.fruitBody.position[0])/2, 
            (f1.fruitBody.position[1] + f2.fruitBody.position[1])/2)
            newFruit = Fruit(newBody, NAMES[(TYPES[f1.type][2]+1)%11], newBody.position[0], flipY(newBody.position[1]))
            newFruit.friction = FRUITFRICTION
            newFruit.collision_type = 2
            newFruit.justPlaced = False
            newFruit.justMerged = True

            space.add(newBody, newFruit)
            FRUITS.append(newFruit)
            global score
            score += SCORES[(TYPES[f1.type][2]+1)%11]
            global pseudoscore
            pseudoscore+=SCORES[(TYPES[f1.type][2]+1)%11] - 2*0

    return True

def checkSegmentCollision(arbiter, space, data):
    f1, f2 = arbiter.shapes
    global canPlace
    if f2.justPlaced == True:
        f2.justPlaced=False
        if not canPlace:
            canPlace = True
    if f2.justMerged == True:
        f2.justMerged=False
    return True

class Fruit(pymunk.Circle): # class of the fruit, including its type, size, vertical and horizontal velocity, x,y pos and angular velocity denoted as w

    def __init__(self, body, type, x, y = 100):
        self.fruitBody = body
        pymunk.Circle.__init__(self, self.fruitBody, TYPES[type][0], (0, 0))
        self.type = type
        # self.radius = TYPES[type][0]
        # self.mass = TYPES[type][3]
        self.justPlaced= True
        self.justMerged = False 
       

space.add_collision_handler(2, 2).pre_solve = checkCollision
space.add_collision_handler(1, 2).pre_solve = checkSegmentCollision

class Game:
    def __init__(self):
        self.queuedFruitName = NAMES[random.choices(range(0,5), weights = SKEWED_PROBABILITY, k = 1)[0]] # Weigh probabilities
        self.nextFruitName = NAMES[random.choices(range(0,5), weights = SKEWED_PROBABILITY, k = 1)[0]]
        self.score = 0
        self.pseudoscore = 0
        self.game_joever = False
        shape = pymunk.Segment(
            space.static_body, Vec2d(0, SCREEN_HEIGHT), Vec2d(0, 0), 0.0
        )
        shape.friction = FRICTION
        space.add(shape)

        shape = pymunk.Segment(
            space.static_body, Vec2d(0, 0), Vec2d(GAME_WIDTH, 0), 0.0
        )
        shape.friction = FRICTION
        shape.collision_type = 1
        space.add(shape)
        shape = pymunk.Segment(
            space.static_body, Vec2d(GAME_WIDTH, SCREEN_HEIGHT), Vec2d(GAME_WIDTH, 0), 0.0
        )
        shape.friction = FRICTION
        space.add(shape)
    
    def reset(self):
        self.queuedFruitName = NAMES[random.choices(range(0,5), weights = SKEWED_PROBABILITY, k = 1)[0]] # Weigh probabilities
        self.nextFruitName = NAMES[random.choices(range(0,5), weights = SKEWED_PROBABILITY, k = 1)[0]]
        self.score = 0
        self.pseudoscore = 0
        self.game_joever = False
        global space
        space = pymunk.Space()
        space.gravity = 0.0, -5.0
        shape = pymunk.Segment(
            space.static_body, Vec2d(0, SCREEN_HEIGHT), Vec2d(0, 0), 0.0
        )
        shape.friction = FRICTION
        space.add(shape)

        shape = pymunk.Segment(
            space.static_body, Vec2d(0, 0), Vec2d(GAME_WIDTH, 0), 0.0
        )
        shape.friction = FRICTION
        shape.collision_type = 1
        space.add(shape)
        shape = pymunk.Segment(
            space.static_body, Vec2d(GAME_WIDTH, SCREEN_HEIGHT), Vec2d(GAME_WIDTH, 0), 0.0
        )
        shape.friction = FRICTION
        space.add(shape)
        space.add_collision_handler(1, 2).pre_solve = checkSegmentCollision
        space.add_collision_handler(2, 2).pre_solve = checkCollision
        
        global score
        score = 0
        
        global pseudoscore
        pseudoscore = 0

        global canPlace
        canPlace = True

        while FRUITS:
            FRUITS.pop()

    def update(self, position):
        if (not self.game_joever):
            if position != -1:
                body = pymunk.Body(TYPES[self.queuedFruitName][3], TYPES[self.queuedFruitName][3]*10)
                body.position = position, 20 + flipY(100)
                fruit = Fruit(body, self.queuedFruitName, position)
                fruit.friction = FRUITFRICTION
                fruit.collision_type = 2
                space.add(body, fruit)
                FRUITS.append(fruit)
                self.queuedFruitName = self.nextFruitName
                self.nextFruitName = NAMES[random.choices(range(0,5), weights=SKEWED_PROBABILITY, k=1)[0]]
            for fruit in FRUITS:
                if flipY(fruit.fruitBody.position[1]) - fruit.radius < OFFSET-20 :
                    if not fruit.justPlaced and not fruit.justMerged:
                        self.game_joever = True
            dt = 1.0 / 30.0
            for x in range(1):
                space.step(dt)
            self.score = score
            self.pseudoscore = pseudoscore