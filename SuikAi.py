import torch
import random
import numpy as np
from collections import deque
from queue import PriorityQueue
# from Suika_Simulation import Game, FRUITS, TYPES, NAMES

from Suika_Simulation import Game, FRUITS, TYPES, NAMES
from SuikAimodel import Linear_QNet, QTrainer
import pygame
from helper import plot #prolly need a plot or smth idk

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

from pygame.locals import (
    QUIT,
)




class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(160, 512,256,128, 81) #81 outputs 11 inputs
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def dfs(self,points, value, x, y,depth):
        for i in range(0,len(points[value-1])):
            dist = ((x - points[value-1][i][0])**2  + abs(y - points[value-1][i][1])**2)**0.5 - TYPES[NAMES[value]][0] - TYPES[NAMES[value-1]][0]
            if value==1 and dist < 10:
                return depth
            elif value ==1:
                return 0
            elif dist<10:
                depth =  self.dfs(points,value-1,points[value-1][i][0],points[value-1][i][1], depth + 1)
        return depth
    def get_state(self, game):
        topPoints = PriorityQueue()
        for i in range(0,50):
            topPoints.put((0,0,0,0))
        points =  [[], [], [], [], [], [], [], [], [], [], []]
        fruitHeight = 0
        bigCorner = 0
        fruitStack = 1
        beegfruit = 0
        depth = 0
        for fruit in FRUITS:
            type = TYPES[fruit.type][2]
            fruitHeight = max (fruit.fruitBody.position[1]+fruit.radius ,fruitHeight )
            if(beegfruit < type):
                beegfruit = type
                bigCorner = abs(-fruit.radius + fruit.fruitBody.position[1] )+  min(abs(400-fruit.radius - fruit.fruitBody.position[0] ),abs(0-fruit.radius + fruit.fruitBody.position[0] ))<=10
            elif(beegfruit == type):
                if(not bigCorner):
                    bigCorner = abs(-fruit.radius +fruit.fruitBody.position[1] )+  min(abs(400-fruit.radius - fruit.fruitBody.position[0]),abs(0-fruit.radius - fruit.fruitBody.position[0]))<=10
            points[type].append((fruit.fruitBody.position[0],fruit.fruitBody.position[1]))
            peek =topPoints.get()
            if(fruit.radius + fruit.fruitBody.position[1] > peek[0]):
                topPoints.put((fruit.radius + fruit.fruitBody.position[1], type, fruit.fruitBody.position[0], fruit.fruitBody.position[1]))
            else:
                topPoints.put(peek)
                

        for x, y in points[beegfruit]:
            fruitStack = max(fruitStack, self.dfs(points, beegfruit, x, y,depth))
        
        state = [
            game.score,
            fruitHeight,
            bigCorner,
            fruitStack,
            TYPES[game.nextFruitName][2],
            TYPES[game.queuedFruitName][2],  
            len(points[0]),
            len(points[1]),
            len(points[2]),  
            len(points[3])       
            ]
        while(not topPoints.empty()):
            peek = topPoints.get()
            state.append(peek[1])
            state.append(peek[2])
            state.append(peek[3])
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0]*81
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 80)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    time = -1000
    quit = False
    while True:
        for event in pygame.event.get():
            if event.type == QUIT:
                quit = True
        if quit:
            break
        if not game.game_joever and pygame.time.get_ticks() > time + 1350:
            time = pygame.time.get_ticks()
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)
            position = max(range(len(final_move)), key=lambda i: final_move[i])

            if position == 0:
                position = -1
                time -= 1350
            else:
                position = (position-1) * 5
            # perform move and get new state
            old_score = game.score
            game.update(position)
            score = game.score
            reward = score - old_score
            reward = reward 
            state_new = agent.get_state(game)
            done = game.game_joever

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)
        elif game.game_joever:
            # train long memory, plot result
            reward = -1e9
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
        else:
            game.update(-1)


if __name__ == '__main__':
    pygame.init()
    train()
    pygame.quit()