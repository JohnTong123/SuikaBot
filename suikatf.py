# import Suika_Simulation_No_Pygame
# from Suika_Simulation_No_Pygame import Game, FRUITS, TYPES, NAMES, GAME_WIDTH

import Suika_Simulation
from Suika_Simulation import Game, FRUITS, TYPES, NAMES, GAME_WIDTH

# import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from queue import PriorityQueue

# import gym
import argparse
import numpy as np
from collections import deque
import random
from helper import plotWithRewards

import pygame
from pygame.locals import (
    QUIT,
)

MAX_MEMORY = 10000
BATCH_SIZE = 32

LR = 0.005
K = 4

tf.keras.backend.set_floatx('float64')
# wandb.init(name='DQN', project="deep-rl-tf2")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim  = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        
        self.model = self.create_model()
    
    def create_model(self):
        model = tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.action_dim)
        ])
        model.compile(loss='mse', optimizer=Adam(args.lr))
        return model
    
    def predict(self, state):
        return self.model.predict(state, verbose = 0)
    
    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        # q_value = self.predict(state)[0]  # Probably return this
        final_move = [0] *4
        if np.random.random() < self.epsilon:
            x  = random.randint(0, 3)
            final_move[x] = 1
            if x ==3:
                x = random.randint(0, 2)
                final_move[x] = 0.5
            return final_move
        # time = pygame.time.get_ticks()
        prediction = self.predict(state)[0]
        # print(pygame.time.get_ticks() - time)
        return prediction.tolist()

    # def get_action(self, state):
    #     self.epsilon = max(self.epsilon * 0.995, 0.01)
    #     # q_value = self.predict(state)[0]
    #     final_move = [0]*4
    #     if np.random.random() < self.epsilon:
    #         x  = random.randint(0, 3)
    #         final_move[x] = 1
    #         if x ==3:
    #             x  = random.randint(0, 2)
    #             final_move[x] = 0.5
    #         return final_move
    #     state0 = torch.tensor(state, dtype=torch.float)
    #     prediction = self.model(state0)
    #     # print(prediction)
    #     # move = torch.argmax(prediction).item()
        
    #     # final_move[move] = 1
    #     # print(pred)
    #     return prediction.tolist()

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)
    
    def save(self):
        self.model.save_weights('./checkpoints/my_checkpoint')
    
    def load(self):
        self.model.load_weights("./checkpoints/my_checkpoint")

class Agent:
    def __init__(self, env):
        self.env = env
        # self.state_dim = self.env.observation_space.shape[0]
        self.state_dim = 403
        self.action_dim = 4
        # self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        # self.model.model.summary()
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.memory = deque(maxlen=MAX_MEMORY)
        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states).max(axis=1)
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)
    
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
    
    def train(self, max_episodes=5000):
        plot_scores = []
        plot_mean_scores = []
        rewards = []
        mean_rewards = []
        total_score = 0
        record = 0
        quit = False
        count = 0
        position = 0
        move = 0
        reward = 0
        final_move = [1, 0, 0, 0]
        totalreward = 0
        old_score = 0
        # agent.model.load1()
        for ep in range(max_episodes):
            if quit:
                break
            while not self.env.game_joever:
                for event in pygame.event.get():
                    if event.type == QUIT:
                        quit = True
                if quit:
                    break
                if count % K == 0:
                # print(Suika_Simulation.canPlace)
                # if not game.game_joever and Suika_Simulation.canPlace:
                    # time = pygame.time.get_ticks()
                    # get old state
                    state_old = self.get_state(self.env, position)
                    final_move = self.model.get_action(state_old)
                    
                    # print(final_move)
                    # if Suika_Simulation_No_Pygame.canPlace == False:
                    if Suika_Simulation.canPlace == False:
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
                        # Suika_Simulation_No_Pygame.canPlace = False
                        Suika_Simulation.canPlace = False
                        self.env.update(position)
                        # while not Suika_Simulation.canPlace and not game.game_joever:
                        #     game.update(-1)
                    else:
                        if move == 1: # Move left
                            position = max(position - 1, 0)
                        elif move == 2: # Move right
                            position = min(position + 1, GAME_WIDTH - 1)
                        self.env.update(-1)
                    score = self.env.score
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

                    state_new = self.get_state(self.env, position)
                    done = self.env.game_joever
                    final_move = [0] * 4
                    final_move[move] = 1

                    self.buffer.put(state_old, move, reward, state_new, done)
                    old_score = score
                else: # count % K != 0
                    move = max(range(0,len(final_move)), key=lambda i: final_move[i]) # Fix this
                    if move == 1:
                        position = max(position - 1, 0)
                    elif move == 2:
                        position = min(position + 1, GAME_WIDTH - 1)
                    self.env.update(-1)
                count+=1
            # train long memory, plot result
            # reward = (score-3000)/100
            reward = -1
            totalreward += reward
            rewards.append(totalreward)
            if len(rewards) == 1:
                mean_rewards.append(totalreward)
            else:
                mean_rewards.append((mean_rewards[len(mean_rewards)-1] * (ep+1) + totalreward) / (ep + 2))
            totalreward = 0
            old_score = 0
            position = 0
            self.env.reset()
            if self.buffer.size() >= args.batch_size:
                self.replay()

            if score > record:
                record = score
                self.model.save()

            self.target_update()

            print('Game', ep+1, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / (ep+1)
            plot_mean_scores.append(mean_score)
            # plot(plot_scores, plot_mean_scores)
            plotWithRewards(plot_scores, plot_mean_scores, rewards, mean_rewards)
            
            

            # done, total_reward = False, 0
            # state = self.env.reset()
            # while not done:
            #     action = self.model.get_action(state)
            #     next_state, reward, done, _ = self.env.step(action)
            #     self.buffer.put(state, action, reward*0.01, next_state, done)
            #     total_reward += reward
            #     state = next_state
            # if self.buffer.size() >= args.batch_size:
            #     self.replay()
            # self.target_update()
            # print('EP{} EpisodeReward={}'.format(ep, total_reward))
            # wandb.log({'Reward': total_reward})


def main():
    pygame.init()
    # env = gym.make('CartPole-v1')
    env =  Game()

    agent = Agent(env)
    agent.train(max_episodes=5000)
    pygame.quit()

if __name__ == "__main__":
    main()
    