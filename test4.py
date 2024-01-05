import torch
import random
import numpy as np
from collections import deque
from queue import PriorityQueue
# from Suika_Simulation import Game, FRUITS, TYPES, NAMES

from Suika_Simulation_No_Pygame import Game, FRUITS, TYPES, GAME_WIDTH
import Suika_Simulation_No_Pygame
from SuikAiModelTest import Linear_QNet, QTrainer
# import pygame
from helper import plot, plotWithRewards #prolly need a plot or smth idk

# MAX_MEMORY = 100_000
MAX_MEMORY = 10000
# BATCH_SIZE = 1000
BATCH_SIZE = 128
EPSILON_START = 1.0
EPSILON_END = 0.05
GAMMA = 0.95
TAU = 0.05

LR = 0.005
K = 4

# from pygame.locals import (
#     QUIT,
# )

class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START # randomness
        self.gamma = GAMMA # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(403, 512, 256, 4)
        self.target = Linear_QNet(403, 512, 256, 4)
        self.target.load_state_dict(self.model.state_dict())
        self.trainer = QTrainer(self.model, self.target, lr=LR, gamma=self.gamma)

    # def dfs(self,points, value, x, y,depth):
    #     for i in range(0,len(points[value-1])):
    #         dist = ((x - points[value-1][i][0])**2  + abs(y - points[value-1][i][1])**2)**0.5 - TYPES[NAMES[value]][0] - TYPES[NAMES[value-1]][0]
    #         if value==1 and dist < 10:
    #             return depth
    #         elif value ==1:
    #             return 0
    #         elif dist<10:
    #             depth =  self.dfs(points,value-1,points[value-1][i][0],points[value-1][i][1], depth + 1)
    #     return depth
    def get_state(self, game,position):
        topPoints = PriorityQueue()
        for i in range(0,80):
            topPoints.put((0,0,0,0,0,0))
        # points =  [[], [], [], [], [], [], [], [], [], [], []]
        # fruitHeight = 0
        # bigCorner = 0
        # fruitStack = 1
        # beegfruit = 0
        # depth = 0
        for fruit in FRUITS:
            type = TYPES[fruit.type][2]
            # fruitHeight = max (fruit.fruitBody.position[1]+fruit.radius ,fruitHeight )
            # if(beegfruit < type):
            #     beegfruit = type
            #     bigCorner = abs(-fruit.radius + fruit.fruitBody.position[1] )+  min(abs(400-fruit.radius - fruit.fruitBody.position[0] ),abs(0-fruit.radius + fruit.fruitBody.position[0] ))<=10
            # elif(beegfruit == type):
            #     if(not bigCorner):
            #         bigCorner = abs(-fruit.radius +fruit.fruitBody.position[1] )+  min(abs(400-fruit.radius - fruit.fruitBody.position[0]),abs(0-fruit.radius - fruit.fruitBody.position[0]))<=10
            # points[type].append((fruit.fruitBody.position[0],fruit.fruitBody.position[1]))
            peek =topPoints.get()
            if(fruit.radius + fruit.fruitBody.position[1] > peek[0]):
                topPoints.put((fruit.radius + fruit.fruitBody.position[1], type, fruit.fruitBody.position[0], fruit.fruitBody.position[1],fruit.fruitBody.velocity[0],fruit.fruitBody.velocity[1]))
            else:
                topPoints.put(peek)
                

        # for x, y in points[beegfruit]:
        #     fruitStack = max(fruitStack, self.dfs(points, beegfruit, x, y,depth))
        
        state = [
            # game.score,
            # fruitHeight,
            # bigCorner,
            # fruitStack,
            TYPES[game.nextFruitName][2],
            TYPES[game.queuedFruitName][2],  
            position
            ]
        while(not topPoints.empty()):
            peek = topPoints.get()
            state.append(peek[1])
            state.append(peek[2])
            state.append(peek[3])
            state.append(peek[4])
            state.append(peek[5])
        return np.array(state, dtype=int)

    # def get_reward(self, game):
    #     fruitStack = 1
    #     beegfruit = 0
    #     bigCorner = 0 
    #     points =  [[], [], [], [], [], [], [], [], [], [], []]

    #     for fruit in FRUITS:
    #         type = TYPES[fruit.type][2]
    #         if(beegfruit < type):
    #             beegfruit = type
    #             bigCorner = abs(-fruit.radius + fruit.fruitBody.position[1] )+  min(abs(400-fruit.radius - fruit.fruitBody.position[0] ),abs(0-fruit.radius + fruit.fruitBody.position[0] ))<=10
    #         elif(beegfruit == type):
    #             if(not bigCorner):
    #                 bigCorner = abs(-fruit.radius +fruit.fruitBody.position[1] )+  min(abs(400-fruit.radius - fruit.fruitBody.position[0]),abs(0-fruit.radius - fruit.fruitBody.position[0]))<=10
    #         points[type].append((fruit.fruitBody.position[0],fruit.fruitBody.position[1]))

    #     for x, y in points[beegfruit]:
    #         fruitStack = max(fruitStack, self.dfs(points, beegfruit, x, y,0))
    #     return (fruitStack, bigCorner)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    # def train_long_memory(self):
    #     if len(self.memory) > BATCH_SIZE:
    #         mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
    #     else:
    #         mini_sample = self.memory

    #     states, actions, rewards, next_states, dones = zip(*mini_sample)
    #     self.trainer.train_step(states, actions, rewards, next_states, dones)
    #     #for state, action, reward, nexrt_state, done in mini_sample:
    #     #    self.trainer.train_step(state, action, reward, next_state, done)

    # def train_short_memory(self, state, action, reward, next_state, done):
    #     self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = max(self.epsilon * 0.99995, EPSILON_END)
        # q_value = self.predict(state)[0]
        final_move = [0]*4
        if np.random.random() < self.epsilon:
            x  = random.randint(0, 3)
            final_move[x] = 1
            if x ==3:
                x  = random.randint(0, 2)
                final_move[x] = 0.5
            return final_move
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        # print(prediction)
        # move = torch.argmax(prediction).item()
        
        # final_move[move] = 1
        # print(pred)
        return prediction.tolist()
        # return list(prediction.numpy())

        # return np.argmax(q_value)
    

        # # random moves: tradeoff exploration / exploitation
        # self.epsilon = 80 - self.n_games
        # final_move = [0]*401
        # if random.randint(0, 200) < self.epsilon:
        #     move = random.randint(0, 400)
        #     final_move[move] = 1
        # else:
        #     state0 = torch.tensor(state, dtype=torch.float)
        #     prediction = self.model(state0)
        #     move = torch.argmax(prediction).item()
        #     final_move[move] = 1
        
        # return final_move


def train():
    plot_scores = []
    plot_mean_scores = []
    rewards = []
    mean_rewards = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Game()
    # time = -1000
    quit = False
    count = 0
    position = 0
    move = 0
    reward = 0
    final_move = [1, 0, 0, 0]
    totalreward = 0
    old_score = 0
    # agent.model.load1()
    while True:
        # for event in pygame.event.get():
        #     if event.type == QUIT:
        #         quit = True
        # if quit:
        #     break
        if not game.game_joever:
            if count % K == 0:
            # print(Suika_Simulation.canPlace)
            # if not game.game_joever and Suika_Simulation.canPlace:
                # time = pygame.time.get_ticks()
                # get old state
                state_old = agent.get_state(game, position)
                final_move = agent.get_action(state_old)
                # print(final_move)
                if Suika_Simulation_No_Pygame.canPlace == False:
                    move = max(range(0,len(final_move)-1), key=lambda i: final_move[i]) # Don't include place as an option
                else:
                    # get move
                    move = max(range(0,len(final_move)), key=lambda i: final_move[i])
                    # move = 3
                    # position = max(range(0,len(final_move)), key=lambda i: final_move[i])
                    # if position == 0:
                    #     position = -1
                    #     # time -= 1350
                    
                    # else:
                    #     position = (position-1) * 1
                    #     Suika_Simulation.canPlace = False
                        # print("test")
                
                # perform move and get new state
                if move == 3:
                    Suika_Simulation_No_Pygame.canPlace = False
                    game.update(position)
                    # while not Suika_Simulation.canPlace and not game.game_joever:
                    #     game.update(-1)
                else:
                    if move == 1: # Move left
                        position = max(position - 1, 0)
                    elif move == 2: # Move right
                        position = min(position + 1, GAME_WIDTH - 1)
                    game.update(-1)
                score = game.score
                reward = 0
                # reward=score**0.5
                # reward = score - old_score
                if score > old_score:
                    reward = 1

                totalreward += reward
                # if(score>old_score):
                #     reward = 10
                # reward += agent.get_reward(game)[0] * agent.get_reward(game)[0] -1+ agent.get_reward(game)[1]*5  

                # reward = reward + game.pseudoscore

                state_new = agent.get_state(game, position)
                done = game.game_joever
                final_move = [0] * 4
                final_move[move] = 1
                # train short memory
                # agent.train_short_memory(state_old, final_move, reward, state_new, done)

                # # remember
                agent.remember(state_old, final_move, reward, state_new, done)

                agent.trainer.train_step(agent.memory)

                target_net_state_dict = agent.target.state_dict()
                policy_net_state_dict = agent.model.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
                agent.target.load_state_dict(target_net_state_dict)
                
                old_score = score
            else: # count % K != 0
                move = max(range(0,len(final_move)), key=lambda i: final_move[i]) # Fix this
                if move == 1:
                    position = max(position - 1, 0)
                elif move == 2:
                    position = min(position + 1, GAME_WIDTH - 1)
                game.update(-1)
            count+=1
        elif game.game_joever:
            # train long memory, plot result
            # reward = (score-3000)/100
            reward = -1
            totalreward += reward
            rewards.append(totalreward)
            if len(rewards) == 1:
                mean_rewards.append(totalreward)
            else:
                mean_rewards.append((mean_rewards[len(mean_rewards)-1] * agent.n_games + totalreward) / (agent.n_games + 1))
            totalreward = 0
            old_score = 0
            position = 0
            game.reset()
            agent.n_games += 1
            # agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', agent.n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            plotWithRewards(plot_scores, plot_mean_scores, rewards, mean_rewards)
        



if __name__ == '__main__':
    # pygame.init()
    train()
    # pygame.quit()